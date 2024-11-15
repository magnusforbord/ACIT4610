import os
import pandas as pd
import yfinance as yf

# List of stock tickers
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NFLX', 'NVDA',
    'JPM', 'BAC', 'WFC', 'T', 'VZ', 'DIS', 'INTC', 'CSCO',
    'XOM', 'CVX', 'BA', 'IBM'
]

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to data directories
raw_data_dir = os.path.join(script_dir, 'data', 'raw')
processed_data_dir = os.path.join(script_dir, 'data', 'processed')
os.makedirs(raw_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)

# Download historical daily stock data for each ticker and save as CSV files
for ticker in tickers:
    # Download data for the specified date range
    data = yf.download(
        tickers=ticker,
        start='2018-01-01',
        end='2022-12-31',
        interval='1d',
        progress=False
    )
    
    # Ensure required columns are present
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(column in data.columns for column in required_columns):
        raise ValueError(f"Data for {ticker} is missing required columns.")
    
    # Reset index to make 'Date' a column
    data.reset_index(inplace=True)
    
    # Save data to CSV in the 'raw' data directory
    data.to_csv(os.path.join(raw_data_dir, f'{ticker}.csv'), index=False)

# Initialize a dictionary to hold closing prices for all tickers
price_data = {}

# Read each CSV file and extract the 'Close' price
for ticker in tickers:
    file_path = os.path.join(raw_data_dir, f'{ticker}.csv')
    df = pd.read_csv(file_path)

    # Check if 'Date' column exists
    if 'Date' not in df.columns:
        raise ValueError(f"'Date' column not found in {ticker}.csv")

    # Convert 'Date' column to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index('Date', inplace=True)

    # Remove rows with invalid dates
    df = df[~df.index.isnull()]

    # Ensure index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Index of {ticker} is not a DatetimeIndex")

    # Extract the 'Close' price series
    price_series = df['Close']
    # Rename the series to the ticker symbol
    price_series.name = ticker

    # Store the price series in the dictionary
    price_data[ticker] = price_series

print("Data processing complete.\n")

# Combine all price series into a DataFrame
all_prices = pd.DataFrame(price_data)

# Ensure the DataFrame is sorted by date
all_prices = all_prices.sort_index()

# Forward-fill missing values and drop any remaining NaNs
all_prices = all_prices.ffill().dropna()

# Resample to monthly prices using 'M' for month-end frequency
monthly_prices = all_prices.resample('M').last()

# Calculate monthly returns
monthly_returns = monthly_prices.pct_change().dropna()

# Calculate covariance matrix of monthly returns
covariance_matrix = monthly_returns.cov()

# Save processed data to CSV files
monthly_returns.to_csv(os.path.join(processed_data_dir, 'monthly_returns.csv'))
covariance_matrix.to_csv(os.path.join(processed_data_dir, 'covariance_matrix.csv'))
