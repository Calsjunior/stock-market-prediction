"""Data loading utilities for the stock market prediction project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TICKER = "^GSPC"
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_DATA_PATH = DEFAULT_DATA_DIR / "sp500.csv"
LEGACY_DATA_PATH = _PROJECT_ROOT / "sp500.csv"
DROP_COLUMNS = ("Dividends", "Stock Splits")


def _clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dates and remove columns that are not used by the model."""
    data = df.copy()
    data.index = pd.to_datetime(data.index, utc=True).tz_convert(None).normalize()
    data.index.name = "Date"
    data = data.sort_index()
    data = data.drop(columns=[col for col in DROP_COLUMNS if col in data.columns])
    return data


def read_csv_data(csv_path: str | Path) -> pd.DataFrame:
    """Read historical market data from a CSV file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find market data at {csv_path}")

    data = pd.read_csv(csv_path, index_col=0)
    return _clean_market_data(data)


def fetch_data(ticker_symbol: str = DEFAULT_TICKER, period: str = "max") -> pd.DataFrame:
    """Fetch historical market data from Yahoo Finance."""
    import yfinance as yf

    print(f"Fetching {ticker_symbol} history from Yahoo Finance...")
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=period)

    if data.empty:
        raise ValueError(f"No data returned for ticker {ticker_symbol}")

    return _clean_market_data(data)


def save_data(df: pd.DataFrame, csv_path: str | Path = DEFAULT_DATA_PATH) -> Path:
    """Save market data to disk and return the path."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path)
    print(f"Data saved to {csv_path}")
    return csv_path


def load_market_data(
    ticker_symbol: str = DEFAULT_TICKER,
    period: str = "max",
    csv_path: str | Path = DEFAULT_DATA_PATH,
    prefer_local: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Load market data from disk when available, otherwise fetch it with yfinance.

    The notebook originally cached data in ``sp500.csv`` at the project root.
    This loader supports that file too, then saves a normalized copy under
    ``data/`` to match the full project structure.
    """
    csv_path = Path(csv_path)

    if prefer_local and not refresh:
        if csv_path.exists():
            return read_csv_data(csv_path)

        if LEGACY_DATA_PATH.exists():
            data = read_csv_data(LEGACY_DATA_PATH)
            save_data(data, csv_path)
            return data

    data = fetch_data(ticker_symbol=ticker_symbol, period=period)
    save_data(data, csv_path)
    return data
