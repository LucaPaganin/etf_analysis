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
from portfolio_analysis_utils import (
    download_data, calculate_returns, fetch_valid_tickers,
    search_etf_ticker, display_ticker_search_interface
)
from pages import monte_carlo_simulation_page, single_portfolio_analysis_page


def configure_main_settings():
    """Configures the main settings and global inputs for the Streamlit app."""
    st.set_page_config(
        page_title="ETF Portfolio Analyzer",
        page_icon="📊",
        layout="wide" # Use more horizontal space
    )

    st.title("📊 ETF Portfolio Analyzer")

    st.sidebar.header("Global Settings")
    
    # Initialize session state for persisting search results if not already done
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = ["SWDA.MI", "XEON.MI"]  # Default tickers
    
    # Search interface in the sidebar
    st.sidebar.subheader("ETF Search")
    search_term = st.sidebar.text_input("Search for ETFs by name or ticker", "")
    
    if st.sidebar.button("Search ETFs"):
        if search_term:
            with st.sidebar.spinner(f"Searching for '{search_term}'..."):
                results = search_etf_ticker(search_term)
                if results:
                    st.sidebar.success(f"Found {len(results)} ETFs")
                    st.session_state.search_results = results
                else:
                    st.sidebar.info("No ETFs found. Try another search term.")
    
    # Display search results if we have any
    if st.session_state.search_results:
        # Create a simple display of search results
        result_options = [f"{r['ticker']} - {r['name']}" for r in st.session_state.search_results]
        
        selected_option = st.sidebar.selectbox(
            "Found ETFs (select to add to portfolio)", 
            ["Select an ETF..."] + result_options
        )
        
        if selected_option != "Select an ETF..." and st.sidebar.button("Add to portfolio"):
            # Extract the ticker part
            new_ticker = selected_option.split(" - ")[0]
            if new_ticker not in st.session_state.selected_tickers:
                st.session_state.selected_tickers.append(new_ticker)
                st.sidebar.success(f"Added {new_ticker} to your portfolio")
    
    # Fetch a base list of valid tickers to supplement the selected ones
    base_tickers = fetch_valid_tickers()
    
    # Combine search results with base tickers to provide options
    all_available_tickers = list(set(base_tickers + [t['ticker'] for t in st.session_state.search_results]))
    
    # Multiselect showing currently selected tickers
    tickers = st.sidebar.multiselect(
        "Selected ETF Tickers",
        options=all_available_tickers,
        default=st.session_state.selected_tickers
    )
    
    # Update the session state based on the multiselect
    st.session_state.selected_tickers = tickers

    default_start_en = pd.to_datetime("2005-01-01")
    default_end_en = pd.to_datetime("today")
    start_date = st.sidebar.date_input("Start Date", value=default_start_en)
    end_date = st.sidebar.date_input("End Date", value=default_end_en)

    if start_date >= end_date:
        st.sidebar.error("Error: Start date must precede end date.")
        st.stop() # Stop execution

    risk_free_rate = st.sidebar.number_input("Annual Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=1.0, step=0.1) / 100

    page = st.sidebar.radio("Select Analysis", ["Monte Carlo Simulation", "Single Portfolio Historical Analysis"])

    if not tickers:
        st.warning("Please select at least one ticker in the sidebar.")
        st.stop()

    return tickers, start_date, end_date, risk_free_rate, page


if __name__ == "__main__":
    
    # --- Configure Main Settings and Global Inputs ---
    tickers, start_date, end_date, risk_free_rate, page = configure_main_settings()

    # --- Data Download (runs once if inputs don't change) ---
    data = download_data(tickers, start_date, end_date)

    if data.empty:
        st.error("Cannot proceed without valid historical data.")
        st.stop()

    # Calculate returns (use log for MC, arithmetic might be needed elsewhere)
    returns = calculate_returns(data, method='log') # Log returns often used for MC modeling
    returns_arithmetic = calculate_returns(data, method='arithmetic') # Useful for historical backtesting

    # --- Page 1: Monte Carlo Simulation ---
    if page == "Monte Carlo Simulation":
        monte_carlo_simulation_page(returns, risk_free_rate, tickers)

    # --- Page 2: Single Portfolio Historical Analysis ---
    elif page == "Single Portfolio Historical Analysis":
        single_portfolio_analysis_page(returns_arithmetic, risk_free_rate, tickers)

