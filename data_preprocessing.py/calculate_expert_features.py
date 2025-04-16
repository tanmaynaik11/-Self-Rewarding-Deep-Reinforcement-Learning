import pandas as pd
import os
import numpy as np

def calculate_expert_features(data, k=10, m=20):
    """
    Calculate expert features (Short-term Sharpe Ratio, Return Reward, Min-Max Reward).
    
    Parameters:
    - data (DataFrame): Combined dataset containing 'Close' prices with 'Date' as index.
    - k (int): Window size for Short-term Sharpe Ratio calculation (e.g., 10 days).
    - m (int): Window size for Min-Max Reward calculation (e.g., 20 days).
    
    Returns:
    - data (DataFrame): Updated DataFrame with expert features added.
    """
    # Ensure data is sorted by date
    data = data.sort_index()

    # Calculate Daily Returns
    data['Daily_Return'] = data['Close'].pct_change() * 100

    ## Short-term Sharpe Ratio Calculation
    rolling_returns = pd.concat([data['Daily_Return'].shift(-i) for i in range(1, k+1)], axis=1)
    rolling_mean = rolling_returns.mean(axis=1)
    rolling_std = rolling_returns.std(axis=1)
    data['Sharpe_Reward'] = rolling_mean / (rolling_std + 1e-8)  # Adding small value to avoid division by zero

    ## Return Reward Calculation
    data['Return_Reward'] = data['Daily_Return']

    ## Min-Max Reward Calculation
    rolling_max = pd.concat([data['Daily_Return'].shift(-i) for i in range(1, m+1)], axis=1).max(axis=1)
    rolling_min = pd.concat([data['Daily_Return'].shift(-i) for i in range(1, m+1)], axis=1).min(axis=1)

    conditions = [
        (rolling_max > 0) | ((rolling_max + rolling_min) > 0),
        (rolling_min < 0) | ((rolling_max + rolling_min) < 0)
    ]
    choices = [rolling_max, rolling_min]
    
    data['MinMax_Reward'] = np.select(conditions, choices, default=(rolling_max - rolling_min))

    # Remove NaN values resulting from rolling operations
    data.dropna(subset=['Sharpe_Reward', 'Return_Reward', 'MinMax_Reward'], inplace=True)
    
    return data