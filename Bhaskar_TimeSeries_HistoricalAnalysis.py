# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the CSV data
data = pd.read_csv('Bhaskar_Historical_Data.csv', index_col='Date', parse_dates=True)

# Display the first few rows
print(data.head())

# Keep only the 'Close' column
data = data[['Close']].dropna()

# Decompose the time series (multiplicative model)
# Assuming 252 trading days in a year
decomposition = seasonal_decompose(data['Close'], model='multiplicative', period=252)

# Extract components
original = data['Close']
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Approximate the cyclical component:
# Detrended = original / trend → remove trend
# Remove seasonal: (original / trend) / seasonal = cyclical
# We'll smooth it with a rolling mean to visualize better
cyclical = (original / trend / seasonal).dropna()
cyclical_smoothed = cyclical.rolling(window=30, center=True).mean()

# Plot all components
plt.figure(figsize=(14, 12))

plt.subplot(511)
plt.plot(original, label='Original', color='black')
plt.title('Original Stock Prices')
plt.legend()

plt.subplot(512)
plt.plot(trend, label='Trend', color='blue')
plt.title('Trend Component')
plt.legend()

plt.subplot(513)
plt.plot(seasonal, label='Seasonal', color='green')
plt.title('Seasonal Component')
plt.legend()

plt.subplot(514)
plt.plot(cyclical_smoothed, label='Cyclical (Smoothed)', color='orange')
plt.title('Cyclical Component (Approximate)')
plt.legend()

plt.subplot(515)
plt.plot(residual, label='Irregular / Residual', color='red')
plt.title('Irregular Component (Residual)')
plt.legend()

plt.tight_layout()
plt.show()
