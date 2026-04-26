"""Feature engineering for the stock market prediction project."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


BASE_PREDICTORS = ["Close", "Volume", "Open", "High", "Low"]
DEFAULT_HORIZONS = [2, 5, 60, 250, 1000]


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the classification target.

    Target is 1 when tomorrow's closing price is higher than today's close,
    otherwise it is 0.
    """
    data = df.copy()
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    return data


def add_rolling_features(
    df: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Add rolling price ratio and trend features for each horizon.

    ``Close_Ratio_N`` compares today's close to the N-day rolling average.
    ``Trend_N`` counts how many of the previous N trading days went up.
    """
    data = df.copy()
    new_predictors: list[str] = []
    new_columns = {}

    for horizon in horizons:
        rolling_averages = data["Close"].rolling(window=horizon).mean()

        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data["Close"] / rolling_averages

        trend_column = f"Trend_{horizon}"
        data[trend_column] = data["Target"].shift(1).rolling(window=horizon).sum()

        new_predictors.extend([ratio_column, trend_column])

    new_features_df = pd.DataFrame(new_columns, index=data.index)
    data = pd.concat([data, new_features_df], axis=1)

    cols_to_check = [c for c in data.columns if c != "Tomorrow"]
    data = data.dropna(subset=cols_to_check)
    return data, new_predictors


def build_feature_dataset(
    df: pd.DataFrame,
    start_date: str = "1990-01-01",
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the final modeling table used for training and prediction."""
    data = df.loc[start_date:].copy()
    data = add_target(data)
    data, predictors = add_rolling_features(data, horizons=horizons)
    return data, predictors
