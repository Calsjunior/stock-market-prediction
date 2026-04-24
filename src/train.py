import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import joblib
import os

def predict_single_step(train, test, predictors, model):
    """
    Trains the model on the training set and makes predictions on the test set.
    """
    # Train the model
    model.fit(train[predictors], train["Target"])
    
    # Generate predictions (returns 0 or 1)
    preds = model.predict(test[predictors])
    
    # Combine actual targets and predictions into a DataFrame for easy comparison
    combined = pd.DataFrame({
        "Target": test["Target"],
        "Predictions": preds
    }, index=test.index)
    
    return combined

def backtest(data, model, predictors, start=1000, step=250):
    """
    Backtests the model across the historical data.
    start: The number of days to use for the initial training set (~4 years of trading days)
    step: The number of days to test on before retraining (~1 year of trading days)
    """
    all_predictions = []

    # Loop through the data in chunks
    for i in range(start, data.shape[0], step):
        # Everything up to 'i' is training data
        train = data.iloc[0:i].copy()
        # Everything from 'i' to 'i+step' is testing data
        test = data.iloc[i:(i+step)].copy()
        
        predictions = predict_single_step(train, test, predictors, model)
        all_predictions.append(predictions)
    
    # Concatenate all prediction chunks into one DataFrame
    return pd.concat(all_predictions)

def save_model(model, filename="random_forest.joblib"):
    """
    Saves the trained model to disk so it can be loaded by predict.py later.
    """
    os.makedirs("models", exist_ok=True)
    filepath = os.path.join("models", filename)
    joblib.dump(model, filepath)
    print(f"Model successfully saved to {filepath}")
