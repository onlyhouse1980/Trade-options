import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

# Set the title of the app
st.title('Options Trade Finder')

# Input for stock ticker symbol
ticker_symbol = st.text_input('Enter Stock Ticker Symbol (e.g., AAPL)', 'AAPL')

# Fetch the stock data
stock = yf.Ticker(ticker_symbol)

# Get available expiration dates
exp_dates = stock.options

if exp_dates:
    # Select an expiration date
    exp_date = st.selectbox('Select Expiration Date', exp_dates)

    # Fetch the options chain
    options_chain = stock.option_chain(exp_date)

    # Separate calls and puts
    calls = options_chain.calls
    puts = options_chain.puts

    # Fetch the current stock price
    current_stock_price = stock.history(period='1d')['Close'].iloc[-1]

    # User inputs for filtering criteria
    st.subheader('Filter Criteria')
    open_interest_threshold = st.number_input('Minimum Open Interest', min_value=0, value=100)
    volume_threshold = st.number_input('Minimum Volume', min_value=0, value=50)
    strike_price_range = st.slider('Strike Price Range (% of Current Price)', min_value=50, max_value=150, value=(95, 105))

    # Calculate strike price bounds
    lower_strike = current_stock_price * (strike_price_range[0] / 100)
    upper_strike = current_stock_price * (strike_price_range[1] / 100)

    # Filter calls based on user criteria
    filtered_calls = calls[
        (calls['openInterest'] >= open_interest_threshold) &
        (calls['volume'] >= volume_threshold) &
        (calls['strike'] >= lower_strike) &
        (calls['strike'] <= upper_strike)
    ]

    # Sort by implied volatility
    filtered_calls = filtered_calls.sort_values(by='impliedVolatility', ascending=False)

    # Display the filtered call options
    st.subheader('Filtered Call Options')
    st.write(filtered_calls.reset_index(drop=True))

    # Filter puts based on user criteria (optional)
    st.subheader('Filtered Put Options')
    filtered_puts = puts[
        (puts['openInterest'] >= open_interest_threshold) &
        (puts['volume'] >= volume_threshold) &
        (puts['strike'] >= lower_strike) &
        (puts['strike'] <= upper_strike)
    ]
    filtered_puts = filtered_puts.sort_values(by='impliedVolatility', ascending=False)
    st.write(filtered_puts.reset_index(drop=True))

else:
    st.write('No options data available for this ticker symbol.')

# Disclaimer at the bottom of the app
st.markdown("""
---
*Disclaimer: This application is for educational purposes only and does not constitute financial advice. Options trading involves significant risk and is not suitable for all investors. Before making any investment decisions, consult with a qualified financial advisor.*
""")
