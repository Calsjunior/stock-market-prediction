"""Prediction helpers for using a saved model artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data_loader import load_market_data
from src.features import DEFAULT_HORIZONS, build_feature_dataset
from src.train import DEFAULT_MODEL_PATH, load_model


def predict_latest(
    data: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold: float = 0.6,
) -> dict[str, Any]:
    """Predict whether the next trading close will be higher than the latest close."""
    model, predictors, metadata = load_model(model_path)
    horizons = metadata.get("horizons", DEFAULT_HORIZONS)
    start_date = metadata.get("start_date", "1990-01-01")
    features, _ = build_feature_dataset(data, start_date=start_date, horizons=horizons)

    latest_row = features.iloc[[-1]]
    probability = float(model.predict_proba(latest_row[predictors])[:, 1][0])
    prediction = int(probability >= threshold)

    return {
        "date": latest_row.index[-1],
        "close": float(latest_row["Close"].iloc[0]),
        "probability_up": probability,
        "prediction": prediction,
        "prediction_label": "Up" if prediction == 1 else "Down",
        "threshold": threshold,
    }


def predict_from_saved_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    csv_path: str | Path = "data/sp500.csv",
    ticker_symbol: str = "^GSPC",
    period: str = "max",
    refresh: bool = False,
    threshold: float = 0.6,
) -> dict[str, Any]:
    """Load/fetch data, load a saved model, and return the latest prediction."""
    data = load_market_data(
        ticker_symbol=ticker_symbol,
        period=period,
        csv_path=csv_path,
        refresh=refresh,
    )
    return predict_latest(data=data, model_path=model_path, threshold=threshold)
