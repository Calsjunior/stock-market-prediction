"""Command line entry point for the stock market prediction project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

from src.data_loader import DEFAULT_DATA_PATH, DEFAULT_TICKER, load_market_data
from src.features import build_feature_dataset
from src.predict import predict_latest
from src.train import (
    DEFAULT_MODEL_PATH,
    backtest,
    create_model,
    evaluate_predictions,
    save_model,
    train_final_model,
)


def parse_horizons(value: str) -> list[int]:
    """Parse comma-separated rolling windows from the command line."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and use a Random Forest stock market direction model.")
    subparsers = parser.add_subparsers(dest="command")

    def add_shared_args(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--ticker", default=DEFAULT_TICKER)
        command_parser.add_argument("--period", default="max")
        command_parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
        command_parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
        command_parser.add_argument("--start-date", default="1990-01-01")
        command_parser.add_argument("--horizons", default="2,5,60,250,1000")
        command_parser.add_argument("--threshold", type=float, default=0.6)
        command_parser.add_argument(
            "--refresh",
            action="store_true",
            help="Fetch fresh data from yfinance instead of using local CSV data.",
        )

    train_parser = subparsers.add_parser("train", help="Train and save the model.")
    add_shared_args(train_parser)
    train_parser.add_argument("--backtest-start", type=int, default=2500)
    train_parser.add_argument("--backtest-step", type=int, default=250)
    train_parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Fit and save the final model without running historical backtesting.",
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Use a saved model to predict the next trading day's direction.",
    )
    add_shared_args(predict_parser)

    run_all_parser = subparsers.add_parser(
        "run-all",
        help="Train, save, and immediately print the latest prediction.",
    )
    add_shared_args(run_all_parser)
    run_all_parser.add_argument("--backtest-start", type=int, default=2500)
    run_all_parser.add_argument("--backtest-step", type=int, default=250)

    return parser


def train_pipeline(args: argparse.Namespace) -> tuple[dict, Path]:
    horizons = parse_horizons(args.horizons)
    data = load_market_data(
        ticker_symbol=args.ticker,
        period=args.period,
        csv_path=args.data_path,
        refresh=args.refresh,
    )

    feature_data, predictors = build_feature_dataset(
        data,
        start_date=args.start_date,
        horizons=horizons,
    )

    metrics: dict = {}
    if not getattr(args, "skip_backtest", False):
        model_for_backtest = create_model()
        backtest_data = feature_data.dropna(subset=["Tomorrow"]).copy()
        predictions = backtest(
            data=backtest_data,
            model=model_for_backtest,
            predictors=predictors,
            start=args.backtest_start,
            step=args.backtest_step,
            threshold=args.threshold,
        )
        metrics = evaluate_predictions(predictions)
        print("\nBacktest metrics:")
        pprint(metrics)

    final_model = train_final_model(feature_data, predictors, model=create_model())
    model_path = save_model(
        final_model,
        predictors,
        model_path=args.model_path,
        metadata={
            "ticker": args.ticker,
            "period": args.period,
            "start_date": args.start_date,
            "horizons": horizons,
            "threshold": args.threshold,
        },
    )
    return metrics, model_path


def predict_pipeline(args: argparse.Namespace) -> dict:
    data = load_market_data(
        ticker_symbol=args.ticker,
        period=args.period,
        csv_path=args.data_path,
        refresh=args.refresh,
    )
    result = predict_latest(
        data=data,
        model_path=args.model_path,
        threshold=args.threshold,
    )
    print("\nLatest prediction:")
    pprint(result)
    return result


def main() -> None:
    parser = build_parser()
    argv = sys.argv[1:] or ["run-all"]
    args = parser.parse_args(argv)

    if args.command == "train":
        train_pipeline(args)
    elif args.command == "predict":
        predict_pipeline(args)
    elif args.command == "run-all":
        train_pipeline(args)
        predict_pipeline(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
