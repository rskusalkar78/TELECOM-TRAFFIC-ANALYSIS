"""
Inference / Prediction Module for Telecom Traffic Analysis.

Provides reusable prediction functions:
- Loads serialized Random Forest model from models/.
- Validates input features against expected columns.
- Accepts raw dictionary, single record, or batch dataframe.
- Returns predicted cellular Internet activity estimates.
"""

from pathlib import Path
from typing import Dict, Any, Union, List
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import FEATURE_COLUMNS

DEFAULT_MODEL_PATH = Path("models") / "telecom_rf_model.joblib"


class TelecomTrafficPredictor:
    """Class wrapper for loading and scoring with the trained telecom model."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.bundle: Dict[str, Any] = {}
        self.model = None
        self.feature_names: List[str] = []
        self._load_model()

    def _load_model(self) -> None:
        """Load model bundle from disk."""
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. "
                f"Please train the model first by running: python main.py"
            )
        self.bundle = joblib.load(self.model_path)
        self.model = self.bundle["model"]
        self.feature_names = self.bundle.get("feature_names", FEATURE_COLUMNS)

    def predict(self, input_data: Union[pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        """
        Estimate Internet activity from input telecom features.

        Accepts:
        - A pandas DataFrame containing the required feature columns.
        - A dict representing a single sample (e.g., {'SMS-in': 1.2, 'SMS-out': 0.8, ...}).
        """
        if isinstance(input_data, dict):
            # Calculate derived totals if omitted
            data = input_data.copy()
            if "total_sms" not in data and "SMS-in" in data and "SMS-out" in data:
                data["total_sms"] = data["SMS-in"] + data["SMS-out"]
            if "total_calls" not in data and "Call-in" in data and "Call-out" in data:
                data["total_calls"] = data["Call-in"] + data["Call-out"]
            if "hour" in data:
                h = data["hour"]
                if "hour_sin" not in data:
                    data["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
                if "hour_cos" not in data:
                    data["hour_cos"] = np.cos(2 * np.pi * h / 24.0)

            df_input = pd.DataFrame([data])
        elif isinstance(input_data, pd.DataFrame):
            df_input = input_data.copy()
        else:
            raise TypeError("input_data must be a pandas DataFrame or a dict.")

        # Validate that all required feature columns exist
        missing = [col for col in self.feature_names if col not in df_input.columns]
        if missing:
            raise ValueError(
                f"Input data is missing required feature columns: {missing}\n"
                f"Expected features are: {self.feature_names}"
            )

        X = df_input[self.feature_names]
        predictions = self.model.predict(X)
        return np.maximum(0.0, predictions)


def predict_single_sample(
    sms_in: float,
    sms_out: float,
    call_in: float,
    call_out: float,
    hour: int,
    day_of_week: int = 4,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> float:
    """
    Convenience function to estimate Internet activity for a single observation.
    """
    predictor = TelecomTrafficPredictor(model_path=model_path)
    sample = {
        "SMS-in": float(sms_in),
        "SMS-out": float(sms_out),
        "Call-in": float(call_in),
        "Call-out": float(call_out),
        "hour": int(hour),
        "day_of_week": int(day_of_week),
    }
    pred = predictor.predict(sample)
    return float(pred[0])
