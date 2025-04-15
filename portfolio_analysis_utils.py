import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
import traceback
import yfinance as yf
import streamlit as st

def download_data(tickers, start_date, end_date):
    """Downloads historical Adjusted Close data for the given tickers (robust version)."""
    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)

        if raw_data.empty:
            return pd.DataFrame()

        adj_close_data = pd.DataFrame()
        if isinstance(raw_data.columns, pd.MultiIndex):
            if 'Adj Close' in raw_data.columns.get_level_values(0):
                adj_close_data = raw_data['Adj Close']
            elif 'Close' in raw_data.columns.get_level_values(0):
                adj_close_data = raw_data['Close']
            else:
                return pd.DataFrame()
        elif len(tickers) == 1 and isinstance(raw_data, pd.DataFrame):
            ticker = tickers[0]
            if 'Adj Close' in raw_data.columns:
                adj_close_data = raw_data[['Adj Close']]
            elif 'Close' in raw_data.columns:
                adj_close_data = raw_data[['Close']]
            else:
                return pd.DataFrame()
            adj_close_data.columns = [ticker]
        else:
            return pd.DataFrame()

        adj_close_data = adj_close_data.dropna(axis=1, how='all')

        if adj_close_data.empty:
            return pd.DataFrame()

        cleaned_data = adj_close_data.dropna(axis=0, how='any')

        if cleaned_data.empty:
            return pd.DataFrame()

        if isinstance(cleaned_data, pd.Series):
            cleaned_data = cleaned_data.to_frame(name=cleaned_data.name or tickers[0])

        return cleaned_data

    except KeyError as e:
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def calculate_returns(data, method='log'):
    """Calculates daily or logarithmic returns."""
    if method == 'log':
        returns = np.log(data / data.shift(1))
    else:
        returns = data.pct_change()
    return returns.dropna()

def calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate, trading_days=252):
    """Calculates annualized portfolio return, volatility, and Sharpe Ratio."""
    portfolio_return = np.sum(mean_returns * weights) * trading_days
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_days)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev if portfolio_stddev != 0 else 0
    return portfolio_return, portfolio_stddev, sharpe_ratio

def calculate_sortino_ratio(portfolio_daily_returns, risk_free_rate, trading_days=252):
    """Calculates the annualized Sortino Ratio."""
    target_return = (1 + risk_free_rate)**(1/trading_days) - 1
    downside_returns = portfolio_daily_returns[portfolio_daily_returns < target_return]

    if len(downside_returns) == 0:
        return 0

    downside_stddev = downside_returns.std() * np.sqrt(trading_days)

    if downside_stddev == 0:
        return 0

    portfolio_return_annualized = portfolio_daily_returns.mean() * trading_days

    sortino_ratio = (portfolio_return_annualized - risk_free_rate) / downside_stddev
    return sortino_ratio

def calculate_max_drawdown(cumulative_returns):
    """Calculates the Maximum Drawdown."""
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
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
    drawdown = (cumulative_returns - running_max) / running_max * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill='tozeroy', name='Drawdown', line=dict(color='red')))
    fig.update_layout(title=title, yaxis_title="Drawdown (%)", hovermode="x unified")
    return fig

def plot_returns_histogram(returns, period='Monthly', title="Returns Histogram"):
    """Plotly histogram of periodic returns."""
    if period == 'Monthly':
        resampled_returns = returns.resample('M').sum()
        bins=20
    elif period == 'Annual':
        resampled_returns = returns.resample('Y').sum()
        bins=10
    else:
        resampled_returns = returns
        bins=50

    fig = px.histogram(resampled_returns * 100, nbins=bins, title=f"{title} ({period})", labels={'value': f'{period} Return (%)'})
    return fig

def plot_rolling_std_dev(returns, window=252, title="Annualized Rolling Volatility"):
    """Plotly chart of rolling standard deviation."""
    rolling_std = returns.rolling(window=window).std() * np.sqrt(window)
    fig = px.line(rolling_std, title=title, labels={'value': 'Rolling Volatility', 'index': 'Date'})
    fig.update_layout(hovermode="x unified")
    return fig

def monte_carlo_simulation(returns_df, num_portfolios, risk_free_rate, trading_days=252):
    """Performs the Monte Carlo simulation."""
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    num_assets = len(returns_df.columns)
    results = np.zeros((4, num_portfolios))
    weights_list = []
    portfolio_daily_returns_list = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_list.append(weights)

        portfolio_return, portfolio_stddev, sharpe_ratio = calculate_portfolio_performance(
            weights, mean_returns, cov_matrix, risk_free_rate, trading_days
        )
        results[0,i] = portfolio_return
        results[1,i] = portfolio_stddev
        results[2,i] = sharpe_ratio

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
    geom_returns_factor = 1 + daily_returns

    for years in windows_years:
        window_days = int(years * trading_days)
        if window_days <= 1 or window_days > len(geom_returns_factor):
            continue

        rolling_prod = geom_returns_factor.rolling(window=window_days).apply(np.prod, raw=True)
        rolling_prod = rolling_prod.replace([0, -np.inf, np.inf], np.nan)

        annualized_return = rolling_prod ** (trading_days / window_days) - 1
        rolling_cagrs[f'{years}-Year Rolling CAGR'] = annualized_return

    return pd.DataFrame(rolling_cagrs)


def plot_rolling_returns(rolling_returns_df, title="Rolling Annualized Returns (CAGR)"):
    """Plots the rolling annualized returns using Plotly."""
    if rolling_returns_df.empty:
        return go.Figure()

    fig = px.line(rolling_returns_df, title=title, labels={'value': 'Annualized Return (%)', 'index': 'Date', 'variable': 'Window'})
    fig.update_layout(hovermode="x unified", yaxis_tickformat=".1%")
    fig.update_traces(hovertemplate='%{y:.2%}')
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
    return fig


def search_etf_ticker(search_term):
    """
    Searches for ETFs matching the search term using yfinance.
    
    Args:
        search_term (str): Text to search for in ETF names or tickers
        
    Returns:
        list: List of matching ETF dictionaries with ticker, name and exchange info
    """
    matching_tickers = []
    
    try:
        # Use yf.search.Search to find matching securities
        search_results = yf.search.Search(search_term, max_results=10)
        
        if len(search_results.quotes):            
            for res in search_results.quotes:
                ticker = res['symbol']
                name = res.get('longname', res.get('shortname', 'Unknown Name'))
                exchange = res.get('exchange', '')
                
                # Add to our results in the expected format
                matching_tickers.append({
                    'ticker': ticker,
                    'name': name,
                    'exchange': exchange
                })
    
    except Exception as e:
        st.warning(f"Error searching for ETFs: {str(e)[:100]}...")
    
    return matching_tickers


def display_ticker_search_interface():
    """
    Displays a Streamlit interface for searching ETFs by name or ticker.
    
    Returns:
        str: Selected ticker or None if no selection was made
    """
    st.subheader("Search for ETFs")
    search_term = st.text_input("Enter ETF name or ticker to search", "")
    
    if st.button("Search") and search_term:
        with st.spinner(f"Searching for '{search_term}'..."):
            results = search_etf_ticker(search_term)
            
            if results:
                st.success(f"Found {len(results)} matching ETFs")
                
                # Create a DataFrame for better display
                results_df = pd.DataFrame(results)
                
                # Display the results in a table
                st.dataframe(results_df)
                
                # Allow selection from the results
                ticker_options = [f"{r['ticker']} - {r['name']}" for r in results]
                selected_option = st.selectbox("Select an ETF to use", ["None"] + ticker_options)
                
                if selected_option != "None":
                    # Extract just the ticker part before the dash
                    selected_ticker = selected_option.split(" - ")[0]
                    return selected_ticker
            else:
                st.info("No matching ETFs found. Try a different search term.")
                
    return None


