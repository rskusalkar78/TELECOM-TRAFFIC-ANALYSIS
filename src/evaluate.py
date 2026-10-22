"""
Model Evaluation Module for Telecom Traffic Analysis.

Responsible for:
- Computing standard regression metrics:
  * Mean Absolute Error (MAE)
  * Root Mean Squared Error (RMSE)
  * Coefficient of Determination (R²)
- Benchmarking the Random Forest model against a Mean Baseline Predictor.
- Saving metrics to a CSV file (outputs/model_evaluation_metrics.csv).
- Generating an Actual vs. Predicted scatter/distribution plot (outputs/actual_vs_predicted.png).
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(
    y_true: pd.Series, y_pred: np.ndarray, model_label: str = "Random Forest"
) -> Dict[str, Any]:
    """Calculate MAE, RMSE, and R-squared for given actual and predicted vectors."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "Model": model_label,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2_Score": round(r2, 4),
        "Sample_Count": int(len(y_true)),
    }


def evaluate_model_performance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_train_mean: float,
    output_dir: Path = Path("outputs"),
) -> pd.DataFrame:
    """
    Evaluate Random Forest model and compare with a baseline (training mean predictor).
    Exports metrics to CSV and creates actual vs. predicted visualization.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Random Forest Predictions
    y_pred_rf = model.predict(X_test)
    rf_metrics = calculate_metrics(y_test, y_pred_rf, model_label="Random Forest")

    # 2. Simple Mean Baseline Predictions
    y_pred_baseline = np.full(shape=len(y_test), fill_value=y_train_mean)
    baseline_metrics = calculate_metrics(y_test, y_pred_baseline, model_label="Baseline (Mean)")

    metrics_df = pd.DataFrame([baseline_metrics, rf_metrics])
    metrics_csv_path = output_dir / "model_evaluation_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    # 3. Generate Actual vs Predicted Plot
    plot_path = output_dir / "actual_vs_predicted.png"
    plt.figure(figsize=(8, 6))

    # Downsample points for scatter plot if test set is large
    if len(y_test) > 2000:
        sample_indices = np.random.choice(len(y_test), size=2000, replace=False)
        pts_true = y_test.iloc[sample_indices]
        pts_pred = y_pred_rf[sample_indices]
    else:
        pts_true = y_test
        pts_pred = y_pred_rf

    plt.scatter(pts_true, pts_pred, alpha=0.35, color="#1f77b4", edgecolors="none", s=25, label="Test Points")

    # 45-degree reference line
    min_val = min(float(pts_true.min()), float(pts_pred.min()))
    max_val = max(float(pts_true.max()), float(pts_pred.max()))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal Fit (y = x)")

    plt.title(f"Actual vs. Predicted Internet Activity (R² = {rf_metrics['R2_Score']})", fontsize=14, weight="bold")
    plt.xlabel("Actual Internet Activity", fontsize=12)
    plt.ylabel("Predicted Internet Activity", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return metrics_df
