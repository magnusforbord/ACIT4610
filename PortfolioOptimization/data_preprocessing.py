import yfinance as yf
import pandas as pd
import os

# List of stock tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# Download historical data
data = yf.download(tickers, start='2018-01-01', end='2022-12-31', interval='1d')

# Extract adjusted close prices
adj_close = data['Adj Close']

# Resample to monthly prices
monthly_prices = adj_close.resample('M').last()

# Calculate monthly returns
monthly_returns = monthly_prices.pct_change().dropna()

# Calculate covariance matrix
covariance_matrix = monthly_returns.cov()

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Optionally, create a 'data' directory inside PortfolioOptimization
data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

# Save CSV files inside the 'data' directory within PortfolioOptimization
monthly_returns.to_csv(os.path.join(data_dir, 'monthly_returns.csv'))
covariance_matrix.to_csv(os.path.join(data_dir, 'covariance_matrix.csv'))
