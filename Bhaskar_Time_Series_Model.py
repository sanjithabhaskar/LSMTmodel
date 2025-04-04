# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

from sklearn.model_selection import train_test_split

# Load the CSV data (Replace with your actual file path)
data = pd.read_csv('Bhaskar_Historical_Data.csv', index_col='Date', parse_dates=True)

# Display the first few rows
print(data.head())

# We are using only the 'Close' price to predict
data = data[['Close']]

# Normalize the 'Close' price using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create the data structure with 60 timesteps and 1 output (predicting the next day's price)
def create_dataset(data, timesteps=60):
    X = []
    y = []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i, 0])  # Last 60 closing prices
        y.append(data[i, 0])  # Next day's closing price
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)

# Reshape X to be 3D for LSTM input [samples, timesteps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the LSTM model
model = Sequential()

# LSTM layer with 50 units
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))

# Another LSTM layer
model.add(LSTM(units=50, return_sequences=False))

# Add Dense layer
model.add(Dense(units=1))  # Single output (next day's closing price)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Make predictions on the test set
predictions = model.predict(X_test)

# Inverse transform the predictions and actual values
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(y_test, color='blue', label='Actual GOOGL Prices')
plt.plot(predictions, color='red', label='Predicted GOOGL Prices')
plt.title('GOOGL Stock Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
