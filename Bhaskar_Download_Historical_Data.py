import yfinance as yf
import pandas as pd
import datetime

# Define stock symbol
stock = "GOOGL"

# Set time range for the past 20 years
start_date = datetime.datetime(2004, 1, 1)
end_date = datetime.datetime.today()

# Create ticker object
goog = yf.Ticker(stock)

# Fetch history using Ticker().history()
goog_hist = goog.history(start=start_date, end=end_date, interval="1d", auto_adjust=False)

# Ensure correct columns are present
goog_hist = goog_hist[["Open", "High", "Low", "Close", "Volume"]]

# Save to CSV file
csv_filename = "Bhaskar_Historical_Data.csv"
goog_hist.to_csv(csv_filename)

# Print first few rows
print(goog_hist.head(20))
print(f"Data saved successfully to {csv_filename} ✅")
