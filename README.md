# stock market prediction

## Project Structure
The current structure should look like:

```text
STOCK-MARKET-PREDICTION/
│
├── data/                   # Save raw or processed CSVs here
├── models/                 # Directory to save your trained model files
│   └── random_forest.joblib
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Fetches yfinance data and does basic cleaning
│   ├── features.py         # Creates the Target, rolling averages, and trends
│   ├── train.py            # Initializes the model, runs backtesting, and saves the model
│   └── predict.py          # Loads the saved model and runs it on today's data to predict tomorrow
│
├── requirements.txt        # yfinance, pandas, scikit-learn, joblib
└── main.py                 # The entry point to run your pipeline
```