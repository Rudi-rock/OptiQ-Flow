import pandas as pd
import numpy as np
from typing import Tuple

def calculate_covariance_matrix(prices: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the expected returns (mu) and covariance matrix (sigma) 
    from historical stock prices.
    
    Args:
        prices: DataFrame of adjusted close prices
        
    Returns:
        mu: Expected returns vector
        sigma: Covariance matrix
    """
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    
    # Annualize expected returns (assuming 252 trading days)
    mu = returns.mean().values * 252
    
    # Annualize covariance matrix
    sigma = returns.cov().values * 252
    
    return mu, sigma

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from data.fetch_data import fetch_stock_data
    from config.global_config import config
    
    df = fetch_stock_data(config.ASSETS, config.START_DATE, config.END_DATE)
    mu, sigma = calculate_covariance_matrix(df)
    
    print("Expected Returns (mu):")
    print(mu)
    print("\nCovariance Matrix (sigma):")
    print(sigma)
