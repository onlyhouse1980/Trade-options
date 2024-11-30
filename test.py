# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------
# User Inputs
# ----------------------------------------

# List of cryptocurrency symbols
crypto_symbols = ['ETH-USD']  # You can add more symbols like 'ETH-USD', 'ADA-USD'

# Define the date range
start_date = '2024-01-01'
end_date = '2024-11-29'

# Number of future days to predict
num_future_days = 30

# ----------------------------------------
# Function Definitions
# ----------------------------------------

def fetch_crypto_data(symbol, start_date, end_date):
    """
    Fetch historical cryptocurrency data.
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    if data.empty:
        print(f"Failed to retrieve data for {symbol}.")
    return data

def create_dataset(dataset, look_back=60):
    """
    Create sequences of data for LSTM input.
    """
    x = []
    y = []
    for i in range(look_back, len(dataset)):
        x.append(dataset[i - look_back:i])
        y.append(dataset[i, 0])  # Predicting the 'Close' price
    return np.array(x), np.array(y)

def build_lstm_model(input_shape):
    """
    Build and compile an LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_predictions(y_test, predictions, symbol, title):
    """
    Plot the actual vs. predicted prices.
    """
    plt.figure(figsize=(14, 7))
    plt.title(f'{symbol} {title}')
    plt.plot(y_test, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_future_predictions(data, future_df, symbol, title):
    """
    Plot future price predictions.
    """
    plt.figure(figsize=(14, 7))
    plt.title(f'{symbol} {title}')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(data['Close'], label='Historical Price')
    plt.plot(future_df['Predicted_Close'], label='Future Predicted Price')
    plt.legend()
    plt.show()

# ----------------------------------------
# Main Script
# ----------------------------------------

for crypto_symbol in crypto_symbols:
    print(f"\nProcessing {crypto_symbol}...\n")
    # ----------------------------------------
    # 1. Data Collection
    # ----------------------------------------

    data = fetch_crypto_data(crypto_symbol, start_date, end_date)

    if data.empty:
        continue  # Skip to the next symbol if data retrieval failed

    # ----------------------------------------
    # 2. Data Preprocessing
    # ----------------------------------------

    # Check for missing values and drop them
    data.dropna(inplace=True)

    # Calculate technical indicators
    data['MA7'] = data['Close'].rolling(window=7).mean()
    data['MA21'] = data['Close'].rolling(window=21).mean()
    data['EMA'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['20SD'] = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['MA21'] + (data['20SD']*2)
    data['Lower_Band'] = data['MA21'] - (data['20SD']*2)

    # Drop rows with NaN values after adding technical indicators
    data.dropna(inplace=True)

    # Use the 'Close' price as the target variable
    features = ['Close', 'MA7', 'MA21', 'EMA', 'Upper_Band', 'Lower_Band']
    values = data[features].values

    # Feature Scaling for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values)

    # ----------------------------------------
    # 3. Create Training and Testing Datasets for LSTM
    # ----------------------------------------

    # Define the training data length (80% of the data)
    training_data_len = int(np.ceil(0.8 * len(scaled_data)))

    # Create the training data set
    train_data = scaled_data[0:training_data_len, :]

    look_back = 60  # Number of previous time steps to use
    x_train, y_train = create_dataset(train_data, look_back)

    # Reshape the data for LSTM input
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(features)))

    # ----------------------------------------
    # 4. Build and Train the LSTM Model
    # ----------------------------------------

    model = build_lstm_model((x_train.shape[1], x_train.shape[2]))

    # Train the model
    epochs = 20
    batch_size = 32

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    # ----------------------------------------
    # 5. Prepare Test Data and Make Predictions with LSTM
    # ----------------------------------------

    # Create the testing data set
    test_data = scaled_data[training_data_len - look_back:, :]

    # Create x_test and y_test
    x_test, y_test = create_dataset(test_data, look_back)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], len(features)))

    # Get the model's predicted price values
    predictions = model.predict(x_test)
    predictions_full = np.concatenate((predictions, np.zeros((predictions.shape[0], len(features)-1))), axis=1)
    predictions_rescaled = scaler.inverse_transform(predictions_full)[:, 0]

    # Inverse transform y_test as well
    y_test_full = np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], len(features)-1))), axis=1)
    y_test_rescaled = scaler.inverse_transform(y_test_full)[:, 0]

    # ----------------------------------------
    # 6. Evaluate the LSTM Model
    # ----------------------------------------

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
    print(f"Root Mean Squared Error (LSTM) for {crypto_symbol}: {rmse:.2f}")

    # Calculate MAPE
    mape = mean_absolute_percentage_error(y_test_rescaled, predictions_rescaled) * 100
    print(f"Mean Absolute Percentage Error (LSTM) for {crypto_symbol}: {mape:.2f}%")

    # RMSE as a percentage of average price
    average_price = np.mean(y_test_rescaled)
    rmse_percentage = (rmse / average_price) * 100
    print(f"RMSE as a percentage of average price: {rmse_percentage:.2f}%")

    # ----------------------------------------
    # 7. Visualization of LSTM Predictions
    # ----------------------------------------

    # Visualize the data
    plot_predictions(y_test_rescaled, predictions_rescaled, crypto_symbol, 'Actual vs. Predicted Prices with LSTM')

    # ----------------------------------------
    # 8. Making Future Predictions with LSTM
    # ----------------------------------------

    # Get the last 'look_back' days of data
    last_sequence = scaled_data[-look_back:]

    # Reshape to match LSTM input
    last_sequence = last_sequence.reshape(1, look_back, len(features))

    # Predict future prices
    future_predictions = [30]
    last_input = last_sequence.copy()

    for _ in range(num_future_days):
        next_pred = model.predict(last_input)
        future_predictions.append(next_pred[0][0])

        # Create the next input sequence
        next_input = np.concatenate((next_pred, last_input[:, -1, 1:].reshape(1, 1, -1)), axis=2)
        last_input = np.concatenate((last_input[:, 1:, :], next_input), axis=1)

    # Inverse transform the predictions
    future_predictions_full = np.concatenate(
        (np.array(future_predictions).reshape(-1, 1), np.zeros((num_future_days, len(features)-1))),
        axis=1
    )
    future_predictions_rescaled = scaler.inverse_transform(future_predictions_full)[:, 0]

    # Create a DataFrame for plotting future predictions
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=num_future_days, freq='D')
    future_df = pd.DataFrame(data=future_predictions_rescaled, index=future_dates, columns=['Predicted_Close'])

    # Plot the future predictions
    plot_future_predictions(data, future_df, crypto_symbol, 'Future Price Prediction with LSTM')

    # ----------------------------------------
    # End of Script for Current Cryptocurrency
    # ----------------------------------------

print("\nScript execution completed.")
