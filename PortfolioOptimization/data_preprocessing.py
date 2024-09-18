import yfinance as yf
import pandas as pd
import os

# List of stock tickers (include all 20 stocks)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NFLX', 'NVDA', 'JPM', 'BAC',
           'WFC', 'T', 'VZ', 'DIS', 'INTC', 'CSCO', 'XOM', 'CVX', 'BA', 'IBM']

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to data directories
raw_data_dir = os.path.join(script_dir, 'data', 'raw')
processed_data_dir = os.path.join(script_dir, 'data', 'processed')
os.makedirs(raw_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)

# Download data for each ticker and save as individual CSV files
for ticker in tickers:
    # Download data
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
    
    # Save to CSV in the 'raw' data directory
    data.to_csv(os.path.join(raw_data_dir, f'{ticker}.csv'), index=False)

# Initialize a dictionary to hold all close prices
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

    # Check if index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Index of {ticker} is not a DatetimeIndex")

    # Extract the 'Close' price
    price_series = df['Close']
    price_series.name = ticker  # Rename the series to the ticker symbol

    # Store the price series in the dictionary
    price_data[ticker] = price_series
print("Data processing complete.\n")

# Combine all price series into a DataFrame
all_prices = pd.DataFrame(price_data)

# Ensure the DataFrame is sorted by date
all_prices = all_prices.sort_index()

# Check the type of all_prices.index

all_prices = all_prices.ffill().dropna()


# Resample to monthly prices using 'ME' for month-end frequency
monthly_prices = all_prices.resample('ME').last()

# Calculate monthly returns
monthly_returns = monthly_prices.pct_change().dropna()

# Calculate covariance matrix
covariance_matrix = monthly_returns.cov()

# Save processed data
monthly_returns.to_csv(os.path.join(processed_data_dir, 'monthly_returns.csv'))
covariance_matrix.to_csv(os.path.join(processed_data_dir, 'covariance_matrix.csv'))
