import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import time
import warnings

warnings.filterwarnings("ignore")

# Set the title of the app
st.title('📈 Options Trade Recommender')

# Description
st.write("""
Welcome to the **Options Trade Recommender** app! This tool helps you find potentially attractive options trades based on your selected criteria.

If you're new to options trading, we've included explanations of key terms and concepts throughout the app.
""")

# User inputs for filtering criteria
st.sidebar.header('Filter Criteria')

# Select stock universe
stock_universe = st.sidebar.selectbox(
    'Select Stock Universe',
    options=['S&P 500', 'Custom List'],
    help='Choose "S&P 500" to analyze options for all companies in the S&P 500 index, or "Custom List" to specify your own stocks.'
)

# Custom stock list input
if stock_universe == 'Custom List':
    custom_stocks = st.sidebar.text_area(
        'Enter Stock Ticker Symbols (separated by commas)',
        value='AAPL, MSFT, AMZN, GOOG, META',
        help='Enter the stock ticker symbols you want to analyze, separated by commas (e.g., AAPL, MSFT, AMZN).'
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

# Filtering criteria with explanations
open_interest_threshold = st.sidebar.number_input(
    'Minimum Open Interest',
    min_value=0,
    value=500,
    help=(
        'Open Interest refers to the total number of outstanding option contracts that '
        'have not been settled. A higher open interest indicates greater liquidity, '
        'making it easier to buy or sell the option.'
    )
)
volume_threshold = st.sidebar.number_input(
    'Minimum Volume',
    min_value=0,
    value=100,
    help=(
        'Volume is the number of option contracts traded during the current day. '
        'Higher volume suggests more active trading and better liquidity.'
    )
)
implied_volatility_threshold = st.sidebar.slider(
    'Implied Volatility (IV) % Range',
    min_value=0,
    max_value=200,
    value=(30, 100),
    help=(
        'Implied Volatility represents the market\'s forecast of a likely movement '
        'in a security\'s price. Higher IV indicates higher expected volatility, '
        'which can lead to higher option premiums.'
    )
)
max_strike_distance = st.sidebar.slider(
    'Max Strike Distance from Current Price (%)',
    min_value=0,
    max_value=100,
    value=10,
    help=(
        'The Strike Price is the set price at which the option can be bought or sold '
        'when exercised. This setting limits how far the strike price can be from the '
        'current stock price, expressed as a percentage.'
    )
)
expiration_days = st.sidebar.slider(
    'Days Until Expiration',
    min_value=1,
    max_value=365,
    value=30,
    help=(
        'Options have an expiration date, after which they are no longer valid. '
        'This setting filters options that expire within a certain number of days.'
    )
)

# Button to start analysis
start_analysis = st.button('🔍 Start Analysis')

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
                    filtered_calls['Expiration Date'] = exp_date
                    filtered_calls['Days Until Expiration'] = (pd.to_datetime(exp_date) - pd.Timestamp.today()).days
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

        # Rename columns for clarity
        combined_results.rename(columns={
            'strike': 'Strike Price',
            'lastPrice': 'Last Price',
            'bid': 'Bid Price',
            'ask': 'Ask Price',
            'change': 'Price Change',
            'percentChange': 'Percent Change',
            'volume': 'Volume',
            'openInterest': 'Open Interest',
            'impliedVolatility': 'Implied Volatility'
        }, inplace=True)

        # Convert Implied Volatility to percentage
        combined_results['Implied Volatility'] = combined_results['Implied Volatility'] * 100

        # Select relevant columns
        columns_to_display = [
            'Ticker',
            'Expiration Date',
            'Days Until Expiration',
            'Strike Price',
            'Last Price',
            'Bid Price',
            'Ask Price',
            'Price Change',
            'Percent Change',
            'Volume',
            'Open Interest',
            'Implied Volatility'
        ]

        st.subheader('💡 Recommended Options Trades')
        st.write("""
        Below is a list of call options that meet your criteria. You can sort the table by clicking on the column headers.
        """)

        # Display the DataFrame with additional formatting
        st.dataframe(combined_results[columns_to_display].reset_index(drop=True))

    else:
        st.write("No options found matching the criteria.")

    # Glossary Section
    st.markdown("""
    ---
    ### 📚 Glossary of Terms

    **Stock Ticker Symbol**: A unique series of letters assigned to a security for trading purposes (e.g., AAPL for Apple Inc.).

    **Options Contract**: A financial derivative that gives the buyer the right, but not the obligation, to buy or sell an asset at a set price on or before a certain date.

    **Call Option**: An options contract that gives the holder the right to buy an asset at a specified price within a specific time period.

    **Strike Price**: The set price at which the option holder can buy (call option) or sell (put option) the underlying asset when the option is exercised.

    **Expiration Date**: The date on which the options contract becomes void and the right to exercise no longer exists.

    **Days Until Expiration**: The number of days left before the option's expiration date.

    **Last Price**: The most recent price at which the option was traded.

    **Bid Price**: The highest price that a buyer is willing to pay for an option.

    **Ask Price**: The lowest price that a seller is willing to accept for an option.

    **Price Change**: The difference in the option's price from the previous day's closing price.

    **Percent Change**: The percentage change in the option's price from the previous day's closing price.

    **Volume**: The total number of option contracts traded during a specific period, typically a single trading day.

    **Open Interest**: The total number of outstanding option contracts that have not been settled or closed. Higher open interest indicates greater liquidity.

    **Implied Volatility (IV)**: A metric that reflects the market's view on the likelihood of changes in a security's price. Higher IV suggests that the market expects significant price movements.

    **Liquidity**: A measure of how easily an asset or security can be bought or sold in the market without affecting its price.

    **Premium**: The price that the buyer pays to the seller to acquire the option.

    **In-the-Money (ITM)**: A call option is "in the money" if the current price of the underlying asset is above the strike price.

    **Out-of-the-Money (OTM)**: A call option is "out of the money" if the current price of the underlying asset is below the strike price.

    **At-the-Money (ATM)**: An option is "at the money" if the current price of the underlying asset is equal to the strike price.

    ---

    *Disclaimer: This application is for educational purposes only and does not constitute financial advice. Options trading involves significant risk and may not be suitable for all investors. Before making any investment decisions, consult with a qualified financial advisor.*
    """)

    # Additional educational content
    st.markdown("""
    ### 🤔 Understanding Options Trading

    Options trading can be complex, but here are some basic concepts to help you get started:

    - **What Is an Option?**
      - An option is a contract that allows you to buy or sell an underlying asset at a predetermined price before a certain date.

    - **Why Trade Options?**
      - Options can provide leverage, allowing you to control a larger amount of stock with a smaller investment.
      - They can be used for hedging to protect against potential losses in your portfolio.
      - Options offer strategies for income generation through premium collection.

    - **Key Considerations:**
      - **Risk vs. Reward**: Options can offer high rewards but also come with high risks.
      - **Time Decay**: The value of options can decrease over time, especially as they approach expiration.
      - **Market Volatility**: Changes in market volatility can impact option prices.

    - **Getting Started:**
      - **Educate Yourself**: Before trading options, it's important to understand how they work.
      - **Paper Trading**: Consider practicing with a simulated trading account to gain experience without risking real money.
      - **Consult Professionals**: Seek advice from financial advisors or professionals if you're unsure.

    ---
    """)

