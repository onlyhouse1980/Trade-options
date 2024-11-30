import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import time
import warnings

warnings.filterwarnings("ignore")

# Set the title of the app
st.title('Options Trade Recommender')

# Description
st.write("""
This application analyzes options across multiple stocks to recommend potentially attractive trades based on specified criteria.
""")

# User inputs for filtering criteria
st.sidebar.header('Filter Criteria')

# Select stock universe
stock_universe = st.sidebar.selectbox(
    'Select Stock Universe',
    options=['S&P 500', 'Custom List']
)

# Custom stock list input
if stock_universe == 'Custom List':
    custom_stocks = st.sidebar.text_area(
        'Enter Stock Ticker Symbols (separated by commas)',
        value='AAPL, MSFT, AMZN, GOOG, META'
    )
    stock_list = [symbol.strip().upper() for symbol in custom_stocks.split(',')]
else:
    # Load S&P 500 tickers
    @st.cache_data
    def load_sp500_tickers():
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        return df['Symbol'].tolist()

    stock_list = load_sp500_tickers()

# Filtering criteria
open_interest_threshold = st.sidebar.number_input('Minimum Open Interest', min_value=0, value=500)
volume_threshold = st.sidebar.number_input('Minimum Volume', min_value=0, value=100)
implied_volatility_threshold = st.sidebar.slider('Implied Volatility (IV) % Range', min_value=0, max_value=200, value=(30, 100))
max_strike_distance = st.sidebar.slider('Max Strike Distance from Current Price (%)', min_value=0, max_value=100, value=10)
expiration_days = st.sidebar.slider('Days Until Expiration', min_value=1, max_value=365, value=30)

# Button to start analysis
start_analysis = st.button('Start Analysis')

if start_analysis:
    st.write("Analyzing options data... This may take several minutes.")
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []
    total_stocks = len(stock_list)

    for idx, ticker in enumerate(stock_list):
        try:
            stock = yf.Ticker(ticker)
            # Get current stock price
            hist = stock.history(period='1d')
            if hist.empty:
                continue
            current_price = hist['Close'].iloc[-1]

            # Get options expiration dates
            exp_dates = stock.options

            if not exp_dates:
                continue

            # Filter expiration dates within the desired range
            exp_dates_filtered = []
            for date_str in exp_dates:
                exp_date = pd.to_datetime(date_str)
                days_to_exp = (exp_date - pd.Timestamp.today()).days
                if 0 < days_to_exp <= expiration_days:
                    exp_dates_filtered.append(date_str)

            for exp_date in exp_dates_filtered:
                options_chain = stock.option_chain(exp_date)
                calls = options_chain.calls

                # Calculate strike price bounds
                lower_strike = current_price * (1 - max_strike_distance / 100)
                upper_strike = current_price * (1 + max_strike_distance / 100)

                # Filter calls based on criteria
                filtered_calls = calls[
                    (calls['openInterest'] >= open_interest_threshold) &
                    (calls['volume'] >= volume_threshold) &
                    (calls['impliedVolatility'] * 100 >= implied_volatility_threshold[0]) &
                    (calls['impliedVolatility'] * 100 <= implied_volatility_threshold[1]) &
                    (calls['strike'] >= lower_strike) &
                    (calls['strike'] <= upper_strike)
                ]

                if not filtered_calls.empty:
                    filtered_calls['Ticker'] = ticker
                    filtered_calls['Expiration'] = exp_date
                    filtered_calls['DaysToExp'] = (pd.to_datetime(exp_date) - pd.Timestamp.today()).days
                    results.append(filtered_calls)
        except Exception as e:
            # Handle exceptions (e.g., network errors, missing data)
            pass

        # Update progress bar and status text
        progress = (idx + 1) / total_stocks
        progress_bar.progress(progress)
        status_text.text(f"Processing {ticker} ({idx + 1}/{total_stocks})")

    progress_bar.empty()
    status_text.text("Analysis complete.")

    # Combine results
    if results:
        combined_results = pd.concat(results, ignore_index=True)

        # Sort results by Implied Volatility descending
        combined_results = combined_results.sort_values(by='impliedVolatility', ascending=False)

        # Select relevant columns
        columns_to_display = ['Ticker', 'Expiration', 'DaysToExp', 'strike', 'lastPrice',
                              'bid', 'ask', 'change', 'percentChange', 'volume',
                              'openInterest', 'impliedVolatility']

        st.subheader('Recommended Options Trades')
        st.write(combined_results[columns_to_display].reset_index(drop=True))
    else:
        st.write("No options found matching the criteria.")

    # Disclaimer at the bottom of the app
    st.markdown("""
    ---
    *Disclaimer: This application is for educational purposes only and does not constitute financial advice. Options trading involves significant risk and is not suitable for all investors. Before making any investment decisions, consult with a qualified financial advisor.*
    """)
