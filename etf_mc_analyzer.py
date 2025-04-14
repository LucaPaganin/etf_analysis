# --- Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm # For parametric VaR
# from scipy.optimize import minimize # Optional for exact Efficient Frontier
import traceback # For detailed error reporting

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="ETF Portfolio Analyzer",
    page_icon="📊",
    layout="wide" # Use more horizontal space
)

# --- Utility Functions ---

@st.cache_data # Cache to avoid re-downloading the same data
def download_data(tickers, start_date, end_date):
    """Downloads historical Adjusted Close data for the given tickers (robust version)."""
    try:
        # Download all available data
        # Using auto_adjust=True might simplify but changes structure; try without first.
        # raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True, actions=False)
        raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)


        if raw_data.empty:
            st.error(f"No raw data found for tickers {tickers} in the specified period.")
            return pd.DataFrame()

        # Extract only 'Adj Close'
        adj_close_data = pd.DataFrame()
        if isinstance(raw_data.columns, pd.MultiIndex):
            # Multi Ticker case: Select 'Adj Close' level if it exists
            if 'Adj Close' in raw_data.columns.get_level_values(0):
                 adj_close_data = raw_data['Adj Close']
            # If 'Adj Close' isn't there, try 'Close' as a fallback? (Might be less accurate)
            elif 'Close' in raw_data.columns.get_level_values(0):
                 st.warning("Column 'Adj Close' not found, using 'Close' as fallback.")
                 adj_close_data = raw_data['Close']
            else:
                 st.error(f"Neither 'Adj Close' nor 'Close' found in downloaded data for {tickers}. Columns: {raw_data.columns.get_level_values(0)}")
                 return pd.DataFrame()
        elif len(tickers) == 1 and isinstance(raw_data, pd.DataFrame):
            # Single Ticker case (DataFrame returned)
            ticker = tickers[0]
            if 'Adj Close' in raw_data.columns:
                adj_close_data = raw_data[['Adj Close']] # Keep as DataFrame
            elif 'Close' in raw_data.columns:
                st.warning(f"Column 'Adj Close' not found for {ticker}, using 'Close'.")
                adj_close_data = raw_data[['Close']]
            else:
                st.error(f"Neither 'Adj Close' nor 'Close' found in downloaded data for {ticker}. Columns: {raw_data.columns}")
                return pd.DataFrame()
            # Rename column with the ticker for consistency
            adj_close_data.columns = [ticker]
        else:
             # Unexpected case (e.g., Series for single ticker? Or error)
             st.error(f"Unrecognized or empty data structure received from yfinance for {tickers}.")
             st.dataframe(raw_data.head()) # Show data preview for debugging
             return pd.DataFrame()

        # Remove any completely empty columns resulting from the selection
        adj_close_data = adj_close_data.dropna(axis=1, how='all')

        if adj_close_data.empty:
            st.error(f"No 'Adj Close' (or 'Close') data available for the requested tickers after initial filtering.")
            return pd.DataFrame()

        # Remove rows with NaN values (important!)
        # how='any' removes the row if *at least one* value is NaN (safer for calculations)
        cleaned_data = adj_close_data.dropna(axis=0, how='any')

        if cleaned_data.empty:
            st.error(f"No valid data found for tickers {tickers} in the period after removing NaNs. Try a different period or check tickers.")
            return pd.DataFrame()

        # Ensure it's always a DataFrame, even if only one ticker remains
        if isinstance(cleaned_data, pd.Series):
             cleaned_data = cleaned_data.to_frame(name=cleaned_data.name or tickers[0]) # Use existing name or ticker

        return cleaned_data

    except KeyError as e:
         st.error(f"Key error while accessing downloaded data: {e}. This indicates a problem extracting the desired column.")
         # Print available columns for debugging, if possible
         try:
             st.warning(f"Columns received from yfinance: {raw_data.columns}")
         except NameError:
             pass # raw_data not defined
         return pd.DataFrame()
    except Exception as e:
        st.error(f"Generic error during data download or processing: {e}")
        st.error(traceback.format_exc()) # Show more error details
        return pd.DataFrame()

def calculate_returns(data, method='log'):
    """Calculates daily or logarithmic returns."""
    if method == 'log':
        returns = np.log(data / data.shift(1))
    else:
        returns = data.pct_change()
    return returns.dropna() # Remove the first NaN

def calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate, trading_days=252):
    """Calculates annualized portfolio return, volatility, and Sharpe Ratio."""
    portfolio_return = np.sum(mean_returns * weights) * trading_days
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_days)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev if portfolio_stddev != 0 else 0
    return portfolio_return, portfolio_stddev, sharpe_ratio

def calculate_sortino_ratio(portfolio_daily_returns, risk_free_rate, trading_days=252):
    """Calculates the annualized Sortino Ratio."""
    # Daily returns below the daily risk-free rate target
    target_return = (1 + risk_free_rate)**(1/trading_days) - 1
    downside_returns = portfolio_daily_returns[portfolio_daily_returns < target_return]

    if len(downside_returns) == 0:
        return 0 # Or np.nan or a very large number, depending on convention

    # Calculate standard deviation of downside returns (downside deviation)
    downside_stddev = downside_returns.std() * np.sqrt(trading_days)

    if downside_stddev == 0:
        return 0 # Or np.nan

    # Calculate annualized portfolio return (arithmetic mean often used for simplicity)
    portfolio_return_annualized = portfolio_daily_returns.mean() * trading_days

    sortino_ratio = (portfolio_return_annualized - risk_free_rate) / downside_stddev
    return sortino_ratio

def calculate_max_drawdown(cumulative_returns):
    """Calculates the Maximum Drawdown."""
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    # Find start and end dates
    end_date = drawdown.idxmin()
    start_date = cumulative_returns.loc[:end_date].idxmax()
    return max_drawdown, start_date, end_date

def calculate_cagr(cumulative_returns):
    """Calculates the Compound Annual Growth Rate (CAGR)."""
    total_return = cumulative_returns.iloc[-1] / cumulative_returns.iloc[0] - 1
    years = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    return cagr

def plot_cumulative_returns(cumulative_returns, title="Portfolio Growth"):
    """Plotly chart of cumulative growth."""
    fig = px.line(cumulative_returns, title=title, labels={'value': 'Cumulative Value', 'index': 'Date'})
    fig.update_layout(hovermode="x unified")
    return fig

def plot_drawdown(cumulative_returns, title="Historical Drawdown"):
    """Plotly chart of drawdown."""
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max * 100 # In percentage
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill='tozeroy', name='Drawdown', line=dict(color='red')))
    fig.update_layout(title=title, yaxis_title="Drawdown (%)", hovermode="x unified")
    return fig

def plot_returns_histogram(returns, period='Monthly', title="Returns Histogram"):
    """Plotly histogram of periodic returns."""
    if period == 'Monthly':
        resampled_returns = returns.resample('M').sum() # Sum log returns or use .apply(lambda x: (1+x).prod()-1) for arithmetic
        bins=20
    elif period == 'Annual':
        resampled_returns = returns.resample('Y').sum()
        bins=10
    else: # Daily
         resampled_returns = returns
         bins=50

    fig = px.histogram(resampled_returns * 100, nbins=bins, title=f"{title} ({period})", labels={'value': f'{period} Return (%)'})
    return fig

def plot_rolling_std_dev(returns, window=252, title="Annualized Rolling Volatility"):
     """Plotly chart of rolling standard deviation."""
     rolling_std = returns.rolling(window=window).std() * np.sqrt(window) # Annualize
     fig = px.line(rolling_std, title=title, labels={'value': 'Rolling Volatility', 'index': 'Date'})
     fig.update_layout(hovermode="x unified")
     return fig

def monte_carlo_simulation(returns_df, num_portfolios, risk_free_rate, trading_days=252):
    """Performs the Monte Carlo simulation."""
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    num_assets = len(returns_df.columns)
    # 0:Return, 1:StdDev, 2:Sharpe, 3: Sortino
    results = np.zeros((4, num_portfolios))
    weights_list = []
    portfolio_daily_returns_list = [] # For Sortino calculation

    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights) # Normalize to 1
        weights_list.append(weights)

        # Calculate performance metrics
        portfolio_return, portfolio_stddev, sharpe_ratio = calculate_portfolio_performance(
            weights, mean_returns, cov_matrix, risk_free_rate, trading_days
        )
        results[0,i] = portfolio_return
        results[1,i] = portfolio_stddev
        results[2,i] = sharpe_ratio

        # Calculate daily portfolio returns for Sortino
        portfolio_daily_ret = returns_df.dot(weights)
        portfolio_daily_returns_list.append(portfolio_daily_ret)
        results[3,i] = calculate_sortino_ratio(portfolio_daily_ret, risk_free_rate, trading_days)


    results_df = pd.DataFrame({
        'Return': results[0,:],
        'Volatility': results[1,:],
        'Sharpe Ratio': results[2,:],
        'Sortino Ratio': results[3,:],
        'Weights': weights_list
    })

    return results_df

def calculate_rolling_cagr(daily_returns, windows_years, trading_days=252):
    """Calculates rolling annualized returns (CAGR) for different window periods."""
    if daily_returns.empty:
        return pd.DataFrame()

    rolling_cagrs = {}
    # Calculate 1 + daily returns for geometric linking
    geom_returns_factor = 1 + daily_returns

    for years in windows_years:
        window_days = int(years * trading_days)
        if window_days <= 1 or window_days > len(geom_returns_factor):
            st.warning(f"Skipping rolling window of {years} years ({window_days} days) as it's too short or longer than the data history.")
            continue # Skip if window is too small or larger than data

        # Use rolling product and annualize
        # (Product of (1+r) over window)^(trading_days/window_days) - 1
        rolling_prod = geom_returns_factor.rolling(window=window_days).apply(np.prod, raw=True)
        # Handle potential zero or negative values in rolling_prod if using older pandas/numpy
        rolling_prod = rolling_prod.replace([0, -np.inf, np.inf], np.nan) # Avoid errors in power calc

        annualized_return = rolling_prod ** (trading_days / window_days) - 1
        rolling_cagrs[f'{years}-Year Rolling CAGR'] = annualized_return

    return pd.DataFrame(rolling_cagrs)

def plot_rolling_returns(rolling_returns_df, title="Rolling Annualized Returns (CAGR)"):
    """Plots the rolling annualized returns using Plotly."""
    if rolling_returns_df.empty:
        st.info("Not enough data to calculate rolling returns.")
        return go.Figure()

    fig = px.line(rolling_returns_df, title=title, labels={'value': 'Annualized Return (%)', 'index': 'Date', 'variable': 'Window'})
    fig.update_layout(hovermode="x unified", yaxis_tickformat=".1%")
    fig.update_traces(hovertemplate='%{y:.2%}')
    # Add a horizontal line at 0%
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
    return fig


# --- Streamlit User Interface ---

st.title("📊 ETF Portfolio Analyzer")

# --- Sidebar for Global Inputs ---
st.sidebar.header("Global Settings")

# ETF Selection (using text_area for flexibility)
default_tickers_en = "SWDA.MI\nXEON.MI\nSGLD.MI" # Example tickers
ticker_input = st.sidebar.text_area("Enter ETF Tickers (one per line, yfinance format e.g., 'SWDA.MI')", value=default_tickers_en, height=100)
tickers = [ticker.strip().upper() for ticker in ticker_input.split('\n') if ticker.strip()]

# Date Selection
default_start_en = pd.to_datetime("2015-01-01")
default_end_en = pd.to_datetime("today")
start_date = st.sidebar.date_input("Start Date", value=default_start_en)
end_date = st.sidebar.date_input("End Date", value=default_end_en)

# Date Validation
if start_date >= end_date:
    st.sidebar.error("Error: Start date must precede end date.")
    st.stop() # Stop execution

# Risk-Free Rate
risk_free_rate = st.sidebar.number_input("Annual Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=1.0, step=0.1) / 100

# Page Selection
page = st.sidebar.radio("Select Analysis", ["Monte Carlo Simulation", "Single Portfolio Historical Analysis"])

# --- Data Download (runs once if inputs don't change) ---
if not tickers:
    st.warning("Please enter at least one ticker in the sidebar.")
    st.stop()

data = download_data(tickers, start_date, end_date)

if data.empty:
    st.error("Cannot proceed without valid historical data.")
    st.stop()

# Calculate returns (use log for MC, arithmetic might be needed elsewhere)
returns = calculate_returns(data, method='log') # Log returns often used for MC modeling
returns_arithmetic = calculate_returns(data, method='arithmetic') # Useful for historical backtesting

# --- Page 1: Monte Carlo Simulation ---
if page == "Monte Carlo Simulation":
    st.header("🌍 Monte Carlo Simulation of Random Portfolios")
    st.markdown("""
    This section performs a Monte Carlo simulation by generating a large number of portfolios with random weights
    for the selected ETFs. It uses the mean returns and covariance calculated over the specified historical period.
    This helps visualize the risk/return trade-off and identify potentially interesting portfolios.
    """)

    num_portfolios = st.slider("Number of Portfolios to Simulate", min_value=500, max_value=10000, value=3000, step=500)

    if st.button("Run Monte Carlo Simulation"):
        with st.spinner("Calculation in progress..."):
            # Run simulation
            mc_results = monte_carlo_simulation(returns, num_portfolios, risk_free_rate) # Use log returns for mean/cov

            # Find optimal portfolios (among the simulated ones)
            max_sharpe_port = mc_results.loc[mc_results['Sharpe Ratio'].idxmax()]
            min_vol_port = mc_results.loc[mc_results['Volatility'].idxmin()]
            max_sortino_port = mc_results.loc[mc_results['Sortino Ratio'].idxmax()] # Added Sortino

            st.subheader("Simulation Results")

            # Risk/Return Scatter Plot (Simulated Efficient Frontier)
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(
                x=mc_results['Volatility'],
                y=mc_results['Return'],
                mode='markers',
                marker=dict(
                    color=mc_results['Sharpe Ratio'], # Color by Sharpe Ratio
                    colorscale='viridis', # Color scheme
                    showscale=True,
                    colorbar=dict(title='Sharpe Ratio'),
                    opacity=0.7,
                    size=5
                ),
                text=[f"Sharpe: {sr:.2f}<br>Sortino: {sor:.2f}<br>Return: {r:.2%}<br>Vol: {v:.2%}" # Use English labels
                      for sr, sor, r, v in zip(mc_results['Sharpe Ratio'], mc_results['Sortino Ratio'], mc_results['Return'], mc_results['Volatility'])],
                 hoverinfo='text'
            ))

            # Highlight optimal portfolios
            fig_mc.add_trace(go.Scatter(
                x=[max_sharpe_port['Volatility']],
                y=[max_sharpe_port['Return']],
                mode='markers',
                marker=dict(color='red', size=12, symbol='star'),
                name='Max Sharpe Ratio'
            ))
            fig_mc.add_trace(go.Scatter(
                x=[min_vol_port['Volatility']],
                y=[min_vol_port['Return']],
                mode='markers',
                marker=dict(color='green', size=12, symbol='diamond'),
                name='Min Volatility'
            ))
            fig_mc.add_trace(go.Scatter(
                x=[max_sortino_port['Volatility']],
                y=[max_sortino_port['Return']],
                mode='markers',
                marker=dict(color='orange', size=12, symbol='cross'),
                name='Max Sortino Ratio'
            ))

            fig_mc.update_layout(
                title='Monte Carlo Simulation: Risk vs. Return',
                xaxis_title='Annualized Volatility (Risk)',
                yaxis_title='Expected Annualized Return',
                yaxis_tickformat=".2%",
                xaxis_tickformat=".2%",
                hovermode='closest',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_mc, use_container_width=True)

            # --- Display Optimal Weights and Other Charts ---
            st.subheader("Identified Optimal Portfolios")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Max Sharpe Ratio Portfolio**")
                st.metric("Sharpe Ratio", f"{max_sharpe_port['Sharpe Ratio']:.3f}")
                st.metric("Sortino Ratio", f"{max_sharpe_port['Sortino Ratio']:.3f}")
                st.metric("Annual Return", f"{max_sharpe_port['Return']:.2%}")
                st.metric("Annual Volatility", f"{max_sharpe_port['Volatility']:.2%}")
                st.write("Weights:")
                st.dataframe(pd.DataFrame({'ETF': tickers, 'Weight': [f"{w:.2%}" for w in max_sharpe_port['Weights']]}).set_index('ETF'))

            with col2:
                st.markdown("**Min Volatility Portfolio**")
                st.metric("Sharpe Ratio", f"{min_vol_port['Sharpe Ratio']:.3f}")
                st.metric("Sortino Ratio", f"{min_vol_port['Sortino Ratio']:.3f}")
                st.metric("Annual Return", f"{min_vol_port['Return']:.2%}")
                st.metric("Annual Volatility", f"{min_vol_port['Volatility']:.2%}")
                st.write("Weights:")
                st.dataframe(pd.DataFrame({'ETF': tickers, 'Weight': [f"{w:.2%}" for w in min_vol_port['Weights']]}).set_index('ETF'))

            with col3:
                 st.markdown("**Max Sortino Ratio Portfolio**")
                 st.metric("Sharpe Ratio", f"{max_sortino_port['Sharpe Ratio']:.3f}")
                 st.metric("Sortino Ratio", f"{max_sortino_port['Sortino Ratio']:.3f}")
                 st.metric("Annual Return", f"{max_sortino_port['Return']:.2%}")
                 st.metric("Annual Volatility", f"{max_sortino_port['Volatility']:.2%}")
                 st.write("Weights:")
                 st.dataframe(pd.DataFrame({'ETF': tickers, 'Weight': [f"{w:.2%}" for w in max_sortino_port['Weights']]}).set_index('ETF'))


            # Histograms of simulated metrics
            st.subheader("Distribution of Simulated Metrics")
            col_hist1, col_hist2, col_hist3 = st.columns(3)
            with col_hist1:
                 fig_hist_ret = px.histogram(mc_results, x='Return', title='Distribution of Simulated Returns', nbins=50)
                 fig_hist_ret.update_layout(xaxis_tickformat=".1%")
                 st.plotly_chart(fig_hist_ret, use_container_width=True)
            with col_hist2:
                 fig_hist_vol = px.histogram(mc_results, x='Volatility', title='Distribution of Simulated Volatilities', nbins=50)
                 fig_hist_vol.update_layout(xaxis_tickformat=".1%")
                 st.plotly_chart(fig_hist_vol, use_container_width=True)
            with col_hist3:
                 fig_hist_sharpe = px.histogram(mc_results, x='Sharpe Ratio', title='Distribution of Simulated Sharpe Ratios', nbins=50)
                 st.plotly_chart(fig_hist_sharpe, use_container_width=True)


# --- Page 2: Single Portfolio Historical Analysis ---
elif page == "Single Portfolio Historical Analysis":
    st.header("📜 Single Portfolio Historical Analysis")
    st.markdown("""
    This section analyzes the historical performance of a single portfolio with user-specified weights,
    based on the actual data for the selected period. It includes performance and risk metrics,
    along with charts similar to those found on platforms like Curvo.
    """)

    # Portfolio Weights Input
    st.subheader("Define Portfolio Weights (%)")
    weights = []
    cols = st.columns(len(tickers))
    total_weight = 0
    for i, ticker in enumerate(tickers):
        # Use weight / 100.0 later, keep input as percentage
        weight_pct = cols[i].number_input(f"Weight {ticker}", min_value=0.0, max_value=100.0, value=100.0/len(tickers), step=1.0, key=f"weight_{ticker}")
        weights.append(weight_pct / 100.0) # Convert to decimal for calculations
        total_weight += weight_pct

    # Weight sum validation
    if not np.isclose(total_weight, 100.0):
         st.warning(f"The sum of weights is {total_weight:.2f}%. Please ensure it is 100%.")
         # You could disable the button or stop the analysis here
         run_analysis = False
    else:
         run_analysis = True
         st.success(f"Weights sum: {total_weight:.2f}%")

    if run_analysis and st.button("Run Historical Analysis"):
        with st.spinner("Calculating historical performance..."):
            # Calculate historical daily portfolio returns (using arithmetic returns for backtesting)
            portfolio_returns_hist = returns_arithmetic.dot(weights)

            # 1. Calculate Cumulative Value (Equity Curve)
            initial_investment = 10000 # Hypothetical
            cumulative_returns = (1 + portfolio_returns_hist).cumprod() * initial_investment

            # 2. Calculate Key Metrics
            cagr = calculate_cagr(cumulative_returns)
            volatility = portfolio_returns_hist.std() * np.sqrt(252) # Annualize daily std dev
            sharpe = (cagr - risk_free_rate) / volatility if volatility !=0 else 0
            sortino = calculate_sortino_ratio(portfolio_returns_hist, risk_free_rate, trading_days=252)
            max_dd, dd_start, dd_end = calculate_max_drawdown(cumulative_returns)

            # Best/Worst Year (based on annual arithmetic returns)
            annual_returns = portfolio_returns_hist.resample('Y').apply(lambda x: (1+x).prod()-1)
            best_year = annual_returns.max() if not annual_returns.empty else np.nan
            worst_year = annual_returns.min() if not annual_returns.empty else np.nan

            # Value at Risk (VaR) 95% - Historical and Parametric (Normal assumption)
            var_95_hist = portfolio_returns_hist.quantile(0.05) # Daily
            # var_95_hist_ann = ((1 + var_95_hist)**252 - 1) # Very rough annual estimate
            # Parametric (assuming normality)
            mean_ret_daily = portfolio_returns_hist.mean()
            std_dev_daily = portfolio_returns_hist.std()
            z_score_95 = norm.ppf(0.05)
            var_95_param_daily = mean_ret_daily + std_dev_daily * z_score_95
            # var_95_param_ann = ((1 + var_95_param_daily)**252 - 1) # Rough annual estimate


            st.subheader("Historical Performance Metrics")
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("CAGR (Compound Annual Growth Rate)", f"{cagr:.2%}")
            mcol2.metric("Annualized Volatility", f"{volatility:.2%}")
            mcol3.metric("Sharpe Ratio", f"{sharpe:.3f}")
            mcol1.metric("Sortino Ratio", f"{sortino:.3f}")
            mcol2.metric("Maximum Drawdown", f"{max_dd:.2%}")
            mcol3.metric(f"Max DD Period", f"{dd_start.strftime('%Y-%m-%d')} to {dd_end.strftime('%Y-%m-%d')}")
            mcol1.metric("Best Year", f"{best_year:.2%}" if not pd.isna(best_year) else "N/A")
            mcol2.metric("Worst Year", f"{worst_year:.2%}" if not pd.isna(worst_year) else "N/A")
            mcol3.metric("Historical VaR 95% (Daily)", f"{var_95_hist:.2%}")
            # mcol3.metric("Parametric VaR 95% (Daily)", f"{var_95_param_daily:.2%}")

            # --- Historical Charts ---
            st.subheader("Historical Portfolio Charts")

            # Growth Chart
            fig_cum = plot_cumulative_returns(cumulative_returns, title=f"Growth of ${initial_investment:,.0f}") # Assuming USD/EUR symbol preference
            st.plotly_chart(fig_cum, use_container_width=True)

            # Drawdown Chart
            fig_dd = plot_drawdown(cumulative_returns, title="Historical Drawdown (%)")
            st.plotly_chart(fig_dd, use_container_width=True)

            # Returns Histograms
            hist_cols = st.columns(2)
            with hist_cols[0]:
                fig_hist_m = plot_returns_histogram(portfolio_returns_hist, period='Monthly', title="Histogram of Monthly Returns") # Use arithmetic for actual return distribution
                st.plotly_chart(fig_hist_m, use_container_width=True)
            with hist_cols[1]:
                 fig_hist_y = plot_returns_histogram(portfolio_returns_hist, period='Annual', title="Histogram of Annual Returns") # Use arithmetic
                 st.plotly_chart(fig_hist_y, use_container_width=True)

            # Rolling Volatility
            fig_roll_vol = plot_rolling_std_dev(portfolio_returns_hist, window=252, title="Rolling Volatility (Annualized, 252-day window)") # Use arithmetic returns
            st.plotly_chart(fig_roll_vol, use_container_width=True)

            # --- Correlation Matrix ---
            st.subheader("Correlation Matrix of Returns")
            corr_matrix = returns.corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix of Returns")
            fig_corr.update_layout(xaxis_title="ETFs", yaxis_title="ETFs", coloraxis_colorbar=dict(title="Correlation"))
            st.plotly_chart(fig_corr, use_container_width=True)
            # --- Portfolio Weights Pie Chart ---
            st.subheader("Portfolio Weights Distribution")
            weight_df = pd.DataFrame({'ETF': tickers, 'Weight': weights})
            weight_df.set_index('ETF', inplace=True)
            fig_weights = px.pie(weight_df, values='Weight', names=weight_df.index, title="Portfolio Weights Distribution", hole=0.3)
            fig_weights.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_weights, use_container_width=True)
            # --- Rolling Returns Chart ---
            st.subheader("Rolling Returns Analysis")
            # Define desired rolling window periods in years
            rolling_windows_years = [1, 3, 5] # You can customize this list

            # Ensure enough data for the smallest window
            min_days_required = int(min(rolling_windows_years) * 252)
            if len(portfolio_returns_hist) > min_days_required:
                rolling_cagr_results = calculate_rolling_cagr(portfolio_returns_hist, rolling_windows_years)
                fig_roll_ret = plot_rolling_returns(rolling_cagr_results)
                st.plotly_chart(fig_roll_ret, use_container_width=True)
            else:
                st.warning(f"Insufficient historical data ({len(portfolio_returns_hist)} days) to calculate rolling returns for the selected windows (min required: {min_days_required} days for 1 year).")

