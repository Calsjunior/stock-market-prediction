import yfinance as yf
import pandas as pd
import os

def fetch_data(ticker_symbol, period="5y"):
    """
    Fetches historical stock data from Yahoo Finance.
    """
    print(f"Fetching data for {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)
    
    # Get historical market data
    df = ticker.history(period=period)
    
    # yfinance returns a timezone-aware index. Let's remove the timezone for easier handling.
    df.index = df.index.tz_localize(None)
    
    # Drop columns we likely won't need for basic modeling
    df = df.drop(columns=['Dividends', 'Stock Splits'])
    
    return df

def save_data(df, filename="raw_data.csv"):
    """
    Saves the dataframe to the data/ directory.
    """
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)
    df.to_csv(filepath)
    print(f"Data saved to {filepath}")
