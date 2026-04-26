# Stock Market Prediction

## Abstract

Predicting daily stock market movements is a notoriously difficult task due to the high noise-to-signal ratio in financial data.
This project develops a machine learning pipeline to predict whether the S&P 500 index will close higher tomorrow than today.
Using a Random Forest Classifier, we prioritize precision to minimize false positive "buy" signals rather than maximizing total trades.

This project turns the original `market_prediction.ipynb` notebook into a small
machine learning pipeline. It trains a Random Forest classifier to predict
whether the S&P 500 will close higher on the next trading day.

## Project Overview

This project addresses the problem of binary market direction classification.
The core philosophy is that in trading, the cost of being wrong (a False Positive) is often higher than the cost of missing an opportunity.
Therefore, the pipeline is tuned to only fire "Buy" signals when the model probability exceeds a strict confidence barrier.

The project is modularly designed, separating data ingestion, feature engineering, and training into a production-ready Python package structure.

## Dataset and Problem Formulation

The dataset consists of daily historical prices for the S&P 500 index (^GSPC) fetched via Yahoo Finance.

### Task Definition:

Given the historical technical indicators for today, predict if the price will rise tomorrow.
This is formulated as a classification task where the decision threshold is tuned to optimize precision.

### Key Features:

- **Base Metrics:** Open, High, Low, Close, and Volume.
- **Rolling Ratios:** The ratio between the current close and moving averages over horizons of 2, 5, 60, 250, and 1000 days.
- **Trend Indicators:** The sum of "up days" over the same horizons.
- **Target Variable:** A binary indicator (1 if Tomorrow's Close > Today's Close, 0 otherwise).

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

## Exploratory Data Analysis

Analysis revealed that while the S&P 500 has a natural "upward bias" (rising ~53% of the time), a naive model guessing "Up" every day would suffer significant drawdowns during volatility.

<img width="684" height="484" alt="image" src="https://github.com/user-attachments/assets/6a3389be-7f33-4980-b908-bf7aa619a576" />

*Figure 1: Distribution of Up (1) vs Down (0) days in the S&P 500.*

## Performance Analysis

By prioritizing precision, the model achieves a higher win rate on its selected signals compared to a random baseline.

| Metric                        | Value (at 0.6 Threshold) |
| :---------------------------- | :----------------------- |
| **Precision**                                     | ~58% |
| **Total Buy Signals**             | ~850 (from backtest) |
| **Baselin Market Up-Bias**                        | ~54% |

<img width="1246" height="701" alt="image" src="https://github.com/user-attachments/assets/87b8085c-a2ef-44a0-8048-5d17f1e098bb" />

*Figure 2: Model "Buy" signals (Green Arrows) plotted against the S&P 500 price movement.*

## Project Structure

```text
STOCK-MARKET-PREDICTION/
│
├── data/                   # Raw or processed CSV files
│   └── sp500.csv
├── models/                 # Saved model files
│   └── random_forest.joblib
│
├── notebooks/                 # Notebooks for visualization
│   └── data_exploration.ipynb
│   └── feature_analysis.ipynb
│   └── model_evaluation.ipynb
|
├── src/
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

## Notes For Class

This is a classification project, not a guarantee of market performance. The
main metric printed by the notebook and this project is precision, which answers:
when the model predicts the market will go up, how often is it correct?
