import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.global_config import config

def fetch_stock_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance.
    Falls back to synthetic data if download fails.
    """
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        
        if data.empty or data.shape[1] != len(tickers):
            raise ValueError("Incomplete data downloaded")
            
        print("Data fetch successful.")
        return data
        
    except Exception as e:
        print(f"Warning: Data fetch failed ({e}). Using synthetic data.")
        return generate_synthetic_data(tickers, start_date, end_date)

def generate_synthetic_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generates synthetic stock price data for demo purposes.
    Ensures the system works even without internet or API access.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)
    n_assets = len(tickers)
    
    # Generate random walks
    np.random.seed(config.SEED)
    returns = np.random.normal(0.0005, 0.01, (n_days, n_assets))
    price_paths = 100 * np.cumprod(1 + returns, axis=0)
    
    df = pd.DataFrame(price_paths, index=dates, columns=tickers)
    return df

if __name__ == "__main__":
    # Test run
    df = fetch_stock_data(config.ASSETS, config.START_DATE, config.END_DATE)
    print(df.head())
