"""Model training, evaluation, and persistence utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


DEFAULT_MODEL_PATH = Path("models") / "random_forest.joblib"


def create_model(
    n_estimators: int = 200,
    min_samples_split: int = 50,
    random_state: int = 1,
) -> RandomForestClassifier:
    """Create the Random Forest model used in the final notebook version."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=1,
    )


def predict_single_step(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictors: list[str],
    model: RandomForestClassifier,
    threshold: float = 0.6,
) -> pd.DataFrame:
    """Train on historical rows, then predict the next test window."""
    model.fit(train[predictors], train["Target"])
    probabilities = model.predict_proba(test[predictors])[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return pd.DataFrame(
        {
            "Target": test["Target"],
            "Predictions": predictions,
            "Probability": probabilities,
        },
        index=test.index,
    )


def backtest(
    data: pd.DataFrame,
    model: RandomForestClassifier,
    predictors: list[str],
    start: int = 2500,
    step: int = 250,
    threshold: float = 0.6,
) -> pd.DataFrame:
    """Backtest the model by retraining on expanding historical windows."""
    all_predictions: list[pd.DataFrame] = []
    total_rows = data.shape[0]

    print(f"\nStarting backtest on {total_rows} rows. This may take a few minutes...")

    for i in range(start, total_rows, step):
        print(f" Training fold: {i} / {total_rows}...")

        train = data.iloc[:i].copy()
        test = data.iloc[i : i + step].copy()
        predictions = predict_single_step(
            train=train,
            test=test,
            predictors=predictors,
            model=clone(model),
            threshold=threshold,
        )
        all_predictions.append(predictions)

    if not all_predictions:
        raise ValueError(f"Not enough rows for backtesting. Need more than start={start} rows, found {data.shape[0]}.")

    return pd.concat(all_predictions)


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """Return model metrics and a simple baseline comparison."""

    model_precision = precision_score(
        predictions["Target"],
        predictions["Predictions"],
        zero_division=0,
    )

    # Baseline: always predict the market goes up tomorrow.
    baseline_precision = predictions["Target"].mean()

    return {
        "model_precision": model_precision,
        "baseline_always_up_precision": baseline_precision,
        "improvement_over_baseline": model_precision - baseline_precision,
        "prediction_counts": predictions["Predictions"].value_counts().to_dict(),
        "target_distribution": predictions["Target"].value_counts(normalize=True).sort_index().to_dict(),
    }


def train_final_model(
    data: pd.DataFrame,
    predictors: list[str],
    model: RandomForestClassifier | None = None,
) -> RandomForestClassifier:
    """Fit one final model on all feature rows that have a known target."""
    final_model = model if model is not None else create_model()
    trainable = data.dropna(subset=["Tomorrow"]).copy()
    final_model.fit(trainable[predictors], trainable["Target"])
    return final_model


def save_model(
    model: RandomForestClassifier,
    predictors: list[str],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save the trained model and feature metadata to disk."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "predictors": predictors,
            "metadata": metadata or {},
        },
        model_path,
    )
    print(f"Model saved to {model_path}")
    return model_path


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> tuple[RandomForestClassifier, list[str], dict[str, Any]]:
    """Load a model saved by ``save_model``."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find trained model at {model_path}")

    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"], artifact["predictors"], artifact.get("metadata", {})

    raise ValueError("Unsupported model file format. Re-run `python main.py train` to create a model artifact with predictors included.")
