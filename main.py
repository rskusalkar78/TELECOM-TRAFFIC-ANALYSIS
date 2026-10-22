"""
Main Execution Script for Telecom Traffic Analysis and Prediction.

Workflow:
1. Validates presence of dataset (or handles synthetic test mode).
2. Loads and reproducibly samples dataset (conserving memory).
3. Preprocesses activity metrics and extracts temporal features.
4. Performs Exploratory Data Analysis (EDA) and outputs graphs & summaries.
5. Trains a Random Forest Regressor using leakage-free group partitioning.
6. Evaluates model with MAE, RMSE, R² against a baseline predictor.
7. Saves artifacts to outputs/ and models/.
8. Runs a sample inference demonstration.

Run via:
    python main.py
Or with synthetic data mode for quick testing / CI:
    python main.py --use-synthetic
"""

import sys
import argparse
import logging
from pathlib import Path

from src.data_loader import (
    DEFAULT_DATA_PATH,
    check_dataset_exists,
    get_missing_dataset_instructions,
    load_and_sample_data,
    generate_synthetic_data,
)
from src.preprocessing import preprocess_telecom_data
from src.eda import run_eda_pipeline
from src.train import run_training_pipeline
from src.evaluate import evaluate_model_performance
from src.predict import TelecomTrafficPredictor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Telecom Traffic Analysis and Activity Estimation Pipeline"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the telecom dataset (default: data/sms-call-internet-mi-2013-11-01.txt)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=75_000,
        help="Number of records to sample for memory-efficient training (default: 75,000)",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Run pipeline on a synthetic test dataset for validation when real data is not downloaded.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("   TELECOM TRAFFIC ANALYSIS & PREDICTION PIPELINE")
    print("   Department of Electronics Engineering - Mini-Project")
    print("=" * 70)

    # 1. Dataset Validation
    is_real_data = check_dataset_exists(args.data_path)

    if not is_real_data and not args.use_synthetic:
        print(get_missing_dataset_instructions(args.data_path))
        print("Tip: Run 'python main.py --use-synthetic' to verify the entire pipeline code.\n")
        sys.exit(0)

    # 2. Loading / Sampling Data
    if args.use_synthetic:
        print("\n[STEP 1/6] Generating synthetic test dataset for pipeline verification...")
        print(">>> WARNING: SYNTHETIC DATA MODE. Results are for code verification only! <<<")
        raw_df = generate_synthetic_data(n_records=3000, random_state=args.random_state)
    else:
        print(f"\n[STEP 1/6] Loading real dataset from '{args.data_path}'...")
        sample_output_path = Path("outputs") / "sampled_telecom_data.csv"
        raw_df = load_and_sample_data(
            data_path=args.data_path,
            sample_size=args.sample_size,
            random_state=args.random_state,
            save_sample_path=sample_output_path,
        )

    print(f"Loaded {len(raw_df):,} raw telecom records.")

    # 3. Preprocessing
    print("\n[STEP 2/6] Preprocessing data and engineering temporal features...")
    processed_df = preprocess_telecom_data(raw_df)
    print(f"Cleaned dataset contains {len(processed_df):,} valid records.")

    # 4. Exploratory Data Analysis (EDA)
    print("\n[STEP 3/6] Running Exploratory Data Analysis (EDA)...")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_eda_pipeline(processed_df, output_dir=output_dir)
    print(f"EDA charts and summary written to: {output_dir.resolve()}")

    # 5. Training
    print("\n[STEP 4/6] Training Random Forest Regressor (leakage-free group split)...")
    model_dir = Path("models")
    rf_model, X_test, y_test, model_path = run_training_pipeline(
        df=processed_df,
        model_dir=model_dir,
        random_state=args.random_state,
    )
    print(f"Model saved to: {model_path.resolve()}")

    # 6. Evaluation
    print("\n[STEP 5/6] Evaluating model against baseline predictor...")
    # Calculate training set mean for baseline
    y_train_mean = float(processed_df["Internet"].mean())
    metrics_df = evaluate_model_performance(
        model=rf_model,
        X_test=X_test,
        y_test=y_test,
        y_train_mean=y_train_mean,
        output_dir=output_dir,
    )

    print("\n--- Model Performance Comparison ---")
    print(metrics_df.to_string(index=False))
    print(f"\nDetailed evaluation metrics saved to '{output_dir / 'model_evaluation_metrics.csv'}'")
    print(f"Actual vs Predicted plot saved to '{output_dir / 'actual_vs_predicted.png'}'")

    # 7. Sample Inference Demo
    print("\n[STEP 6/6] Verifying Reusable Predictor Module...")
    predictor = TelecomTrafficPredictor(model_path=model_path)
    sample_obs = {
        "SMS-in": 2.5,
        "SMS-out": 1.8,
        "Call-in": 3.4,
        "Call-out": 2.9,
        "hour": 14,
        "day_of_week": 4,
    }
    estimated_traffic = predictor.predict(sample_obs)[0]
    print(f"Sample Input: {sample_obs}")
    print(f"Estimated Mobile Internet Activity: {estimated_traffic:.2f} (Activity units)")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    if args.use_synthetic:
        print("Note: Ran with SYNTHETIC data. Remember to download real data for college report.")
    else:
        print("Results generated using real dataset.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
