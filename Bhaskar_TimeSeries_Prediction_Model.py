import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load the data from CSV (Replace with your actual file path)
data = pd.read_csv('Bhaskar_Historical_Data.csv', index_col='Date', parse_dates=True)

# Display the first few rows of the data
print(data.head())

# Use the 'Close' price for prediction
data = data[['Close']]

# Normalize the 'Close' price using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to create the dataset with sequences (X, y)
def create_dataset(data, timesteps=60):
    X = []
    y = []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i, 0])  # Use the past 60 days for prediction
        y.append(data[i, 0])  # Predict the next day's price
    return np.array(X), np.array(y)

# Create the dataset with sequences
X, y = create_dataset(scaled_data)

# Reshape X to be 3D for LSTM input
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)  # Use 80% of the data for training
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()

# First LSTM layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))

# Second LSTM layer
model.add(LSTM(units=50, return_sequences=False))

# Dense layer to output the prediction
model.add(Dense(units=1))  # Predict the next day's closing price

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Forecast the next 10 years of stock data
# We need to predict the future 10 years worth of daily stock prices (10 * 252 = 2520 days)
future_days = 2520  # Approximate number of trading days in 10 years
predicted_prices = []

# Start from the last known data point (the most recent data in X_test)
current_input = X_test[-1]

# Predict future values one step at a time
for _ in range(future_days):
    predicted_price = model.predict(current_input.reshape(1, current_input.shape[0], 1))
    predicted_prices.append(predicted_price[0, 0])
    current_input = np.append(current_input[1:], predicted_price)  # Update the input for the next prediction

# Inverse transform the predictions to get the original scale of the data
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Plot the results
plt.figure(figsize=(12, 6))

# Plot the original data (last 100 days for comparison)
plt.plot(data.index[-100:], scaler.inverse_transform(scaled_data[-100:]), label='Original Data')

# Plot the predicted data (for the next 10 years)
# Generate the future dates for 2520 business days (10 years)
future_dates = pd.date_range(start=data.index[-1], periods=future_days, freq='B')  # 'B' for business day frequency

# Plot the results
plt.figure(figsize=(12, 6))

# Plot the original data (last 100 days for comparison)
plt.plot(data.index[-100:], scaler.inverse_transform(scaled_data[-100:]), label='Original Data')

# Plot the predicted data (for the next 10 years)
plt.plot(future_dates, predicted_prices, label='Predicted Future Prices', color='red')

plt.title('GOOGL Stock Price Prediction for the Next 10 Years')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

