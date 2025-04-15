import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from portfolio_analysis_utils import (monte_carlo_simulation, calculate_cagr, calculate_sortino_ratio, 
                                       calculate_max_drawdown, plot_cumulative_returns, plot_drawdown, 
                                       plot_returns_histogram, plot_rolling_std_dev, calculate_rolling_cagr, 
                                       plot_rolling_returns)

def monte_carlo_simulation_page(returns, risk_free_rate, tickers):
    st.header("🌍 Monte Carlo Simulation of Random Portfolios")
    st.markdown("""
    This section performs a Monte Carlo simulation by generating a large number of portfolios with random weights
    for the selected ETFs. It uses the mean returns and covariance calculated over the specified historical period.
    This helps visualize the risk/return trade-off and identify potentially interesting portfolios.
    """)

    col1, _ = st.columns([1, 3])
    with col1:
        num_portfolios = st.number_input(
            "Number of Portfolios to Simulate", 
            min_value=1000, max_value=30000, value=3000, step=100
        )

    if st.button("Run Monte Carlo Simulation"):
        with st.spinner("Calculation in progress..."):
            mc_results = monte_carlo_simulation(returns, num_portfolios, risk_free_rate)

            max_sharpe_port = mc_results.loc[mc_results['Sharpe Ratio'].idxmax()]
            min_vol_port = mc_results.loc[mc_results['Volatility'].idxmin()]
            max_sortino_port = mc_results.loc[mc_results['Sortino Ratio'].idxmax()]

            st.subheader("Simulation Results")

            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(
                x=mc_results['Volatility'],
                y=mc_results['Return'],
                mode='markers',
                marker=dict(
                    color=mc_results['Sharpe Ratio'],
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title='Sharpe Ratio'),
                    opacity=0.7,
                    size=5
                ),
                text=[f"Sharpe: {sr:.2f}<br>Sortino: {sor:.2f}<br>Return: {r:.2%}<br>Vol: {v:.2%}"
                      for sr, sor, r, v in zip(mc_results['Sharpe Ratio'], mc_results['Sortino Ratio'], mc_results['Return'], mc_results['Volatility'])],
                 hoverinfo='text'
            ))

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


def single_portfolio_analysis_page(returns_arithmetic, risk_free_rate, tickers):
    st.header("📜 Single Portfolio Historical Analysis")
    st.markdown("""
    This section analyzes the historical performance of a single portfolio with user-specified weights,
    based on the actual data for the selected period. It includes performance and risk metrics,
    along with charts similar to those found on platforms like Curvo.
    """)

    st.subheader("Define Portfolio Weights (%)")
    weights = []
    cols = st.columns(len(tickers))
    total_weight = 0
    for i, ticker in enumerate(tickers):
        weight_pct = cols[i].number_input(f"Weight {ticker}", min_value=0.0, max_value=100.0, value=100.0/len(tickers), step=1.0, key=f"weight_{ticker}")
        weights.append(weight_pct / 100.0)
        total_weight += weight_pct

    if not np.isclose(total_weight, 100.0):
         st.warning(f"The sum of weights is {total_weight:.2f}%. Please ensure it is 100%.")
         run_analysis = False
    else:
         run_analysis = True
         st.success(f"Weights sum: {total_weight:.2f}%")

    if run_analysis and st.button("Run Historical Analysis"):
        with st.spinner("Calculating historical performance..."):
            portfolio_returns_hist = returns_arithmetic.dot(weights)

            initial_investment = 10000
            cumulative_returns = (1 + portfolio_returns_hist).cumprod() * initial_investment

            cagr = calculate_cagr(cumulative_returns)
            volatility = portfolio_returns_hist.std() * np.sqrt(252)
            sharpe = (cagr - risk_free_rate) / volatility if volatility !=0 else 0
            sortino = calculate_sortino_ratio(portfolio_returns_hist, risk_free_rate, trading_days=252)
            max_dd, dd_start, dd_end = calculate_max_drawdown(cumulative_returns)

            st.subheader("Historical Performance Metrics")
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("CAGR (Compound Annual Growth Rate)", f"{cagr:.2%}")
            mcol2.metric("Annualized Volatility", f"{volatility:.2%}")
            mcol3.metric("Sharpe Ratio", f"{sharpe:.3f}")
            mcol1.metric("Sortino Ratio", f"{sortino:.3f}")
            mcol2.metric("Maximum Drawdown", f"{max_dd:.2%}")
            mcol3.metric(f"Max DD Period", f"{dd_start.strftime('%Y-%m-%d')} to {dd_end.strftime('%Y-%m-%d')}")

            st.subheader("Historical Portfolio Charts")
            fig_cum = plot_cumulative_returns(cumulative_returns, title=f"Growth of ${initial_investment:,.0f}")
            st.plotly_chart(fig_cum, use_container_width=True)

            fig_dd = plot_drawdown(cumulative_returns, title="Historical Drawdown (%)")
            st.plotly_chart(fig_dd, use_container_width=True)

            hist_cols = st.columns(2)
            with hist_cols[0]:
                fig_hist_m = plot_returns_histogram(portfolio_returns_hist, period='Monthly', title="Histogram of Monthly Returns")
                st.plotly_chart(fig_hist_m, use_container_width=True)
            with hist_cols[1]:
                 fig_hist_y = plot_returns_histogram(portfolio_returns_hist, period='Annual', title="Histogram of Annual Returns")
                 st.plotly_chart(fig_hist_y, use_container_width=True)

            fig_roll_vol = plot_rolling_std_dev(portfolio_returns_hist, window=252, title="Rolling Volatility (Annualized, 252-day window)")
            st.plotly_chart(fig_roll_vol, use_container_width=True)
