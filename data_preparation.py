import pandas as pd
import numpy as np
import yfinance as yf

# Download the data
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date) #Download from Yahoo Finance
    return data

# Compute the RSI
def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Compute the indicators
def compute_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean() #Simple Moving Averages
    df['SMA_200'] = df['Close'].rolling(window=200).mean() 
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean() #Moving Average Convergence Divergence
    df['RSI'] = compute_rsi(df['Close']) #Relative Strength Index
    return df

# Prepare the data and date
def prepare_data(ticker):
    data = download_data(ticker, start_date="2022-12-10", end_date="2024-05-26") #Dates
    
    df_ticker = data.copy()
    df_ticker = compute_indicators(df_ticker)
    
    for col in df_ticker.columns:
        data[col] = df_ticker[col]

    data.ffill(inplace=True)
    data.dropna(inplace=True)#Drop Null values
    return data

# Choose the stock and save the data to a CSV file
if __name__ == "__main__":
    ticker = 'BFREN.IS'  # Tek bir hisse senedi sembolü
    data = prepare_data(ticker)
    data.to_csv('stock_data.csv')
