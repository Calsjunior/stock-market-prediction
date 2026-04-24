# src/features.py
import pandas as pd

def add_target(df):
    """
    Creates the target variable for classification.
    Target = 1 if tomorrow's close > today's close, else 0.
    """
    # Shift the Close column backwards by 1 to get tomorrow's price
    df["Tomorrow"] = df["Close"].shift(-1)
    
    # Create the binary target
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
    
    return df

def add_rolling_features(df, horizons=[2, 5, 60, 250, 1000]):
    """
    Calculates rolling averages and trends for different time horizons.
    horizons represent days: 2 (couple days), 5 (week), 60 (quarter), 
    250 (year), 1000 (four years).
    """
    new_predictors = []
    
    for horizon in horizons:
        # Calculate the rolling average for the given horizon
        rolling_averages = df["Close"].rolling(window=horizon).mean()
        
        # Feature 1: Ratio between today's close and the rolling average
        ratio_column = f"Close_Ratio_{horizon}"
        df[ratio_column] = df["Close"] / rolling_averages
        
        # Feature 2: Trend (how many days in the past 'horizon' the stock went up)
        trend_column = f"Trend_{horizon}"
        df[trend_column] = df.shift(1).rolling(window=horizon).sum()["Target"]
        
        new_predictors += [ratio_column, trend_column]
        
    # Drop rows with NaN values created by the rolling windows
    # We exclude "Tomorrow" from the dropna check so we can still predict the current day
    df = df.dropna(subset=df.columns[df.columns != "Tomorrow"])
    
    return df, new_predictors