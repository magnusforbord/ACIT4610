import yfinance as yf
import pandas as pd


tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', '...']  # Add additional 15 stock tickers

data = yf.download(tickers, start='2018-01-01', end='2022-12-31', interval='1d')

adj_close = data['Adj Close']

monthly_prices = adj_close.resample('M').last()

monthly_returns = monthly_prices.pct_change().dropna()

covariance_matrix = monthly_returns.cov()

monthly_returns.to_csv('monthly_returns.csv')
covariance_matrix.to_csv('covariance_matrix.csv')
