# Stock Market Prediction

This project turns the original `market_prediction.ipynb` notebook into a small
machine learning pipeline. It trains a Random Forest classifier to predict
whether the S&P 500 will close higher on the next trading day.

## Project Structure

```text
STOCK-MARKET-PREDICTION/
│
├── data/                   # Raw or processed CSV files
│   └── sp500.csv
├── models/                 # Saved model files
│   └── random_forest.joblib
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Loads local CSV data or fetches yfinance data
│   ├── features.py         # Creates Target, rolling averages, and trends
│   ├── train.py            # Backtesting, training, metrics, and model saving
│   └── predict.py          # Loads the saved model and predicts the next day
│
├── market_prediction.ipynb # Original notebook
├── requirements.txt
└── main.py                 # Command line entry point
```

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

## How To Run

Train the model, run backtesting, and print the latest prediction:

```bash
python main.py run-all
```

Train and save only the model:

```bash
python main.py train
```

Predict with an already trained model:

```bash
python main.py predict
```

The project will use `data/sp500.csv` if it exists. If not, it will copy the
original notebook cache from `sp500.csv` into `data/sp500.csv`. If no local CSV
exists, it will fetch S&P 500 data from Yahoo Finance with `yfinance`.

To force a fresh download:

```bash
python main.py run-all --refresh
```

## Model Approach

The final model follows the improved notebook version:

- Target: `1` if tomorrow's close is higher than today's close, otherwise `0`
- Model: `RandomForestClassifier`
- Features: rolling close ratios and recent upward-trend counts
- Horizons: `2, 5, 60, 250, 1000` trading days
- Prediction threshold: `0.6`
- Backtest: expanding training window, tested in 250-trading-day chunks

The saved model artifact includes both the trained model and the predictor
column names, so `predict.py` can rebuild the same features later.

## Notes For Class

This is a classification project, not a guarantee of market performance. The
main metric printed by the notebook and this project is precision, which answers:
when the model predicts the market will go up, how often is it correct?
