"""
Preprocessing Module for Telecom Traffic Analysis.

Responsible for:
- Converting timestamps (epoch milliseconds -> datetime).
- Cleaning numeric activity measurements (SMS, Call, Internet).
- Handling missing, NaN, and negative measurement values cleanly.
- Extracting temporal features (hour, day of week, cyclical sin/cos encodings).
- Preparing feature matrix (X) and target variable (y) without data leakage.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd


ACTIVITY_COLUMNS: List[str] = ["SMS-in", "SMS-out", "Call-in", "Call-out", "Internet"]
FEATURE_COLUMNS: List[str] = [
    "SMS-in",
    "SMS-out",
    "Call-in",
    "Call-out",
    "total_sms",
    "total_calls",
    "hour",
    "day_of_week",
    "hour_sin",
    "hour_cos",
]
TARGET_COLUMN: str = "Internet"


def convert_timestamp(df: pd.DataFrame, time_col: str = "TimeInterval") -> pd.DataFrame:
    """
    Convert Unix timestamps in milliseconds to a datetime column named 'datetime'.
    Handles string, float, or integer inputs cleanly.
    """
    df = df.copy()
    if time_col not in df.columns:
        raise KeyError(f"Timestamp column '{time_col}' not found in dataframe.")

    # Convert to numeric, coercing bad entries to NaN
    numeric_series = pd.to_numeric(df[time_col], errors="coerce")
    # Milan Telecom dataset timestamps are Unix epoch in milliseconds (~1.38e12)
    df["datetime"] = pd.to_datetime(numeric_series, unit="ms", errors="coerce")
    return df


def clean_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean numeric activity columns:
    - Coerce strings/bad values to numeric.
    - Missing activity values in the raw dataset typically represent zero activity
      or omitted records; fill NaNs in communication channels with 0.0.
    - Target 'Internet' rows with NaNs will be dropped during feature preparation.
    - Set negative values (if any artifacts exist) to 0.0.
    """
    df = df.copy()
    for col in ACTIVITY_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # If predictor, zero-fill; if target, keep as is for targeted handling
            if col != TARGET_COLUMN:
                df[col] = df[col].fillna(0.0)
                df[col] = df[col].clip(lower=0.0)
    return df


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract hour of day, day of week, and cyclical sine/cosine representations.
    Cyclical features preserve continuity between 23:59 and 00:00.
    """
    df = df.copy()
    if "datetime" not in df.columns:
        df = convert_timestamp(df)

    # Fill any null datetimes with forward-fill or drop
    df = df.dropna(subset=["datetime"])

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.day_of_week

    # Cyclical hour encoding (period = 24 hours)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # Derived domain metrics
    df["total_sms"] = df["SMS-in"] + df["SMS-out"]
    df["total_calls"] = df["Call-in"] + df["Call-out"]

    return df


def preprocess_telecom_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Clean activity measurements.
    2. Convert timestamps.
    3. Extract temporal and aggregated activity features.
    4. Drop invalid rows.
    """
    if df.empty:
        return pd.DataFrame()

    df_cleaned = clean_measurements(df)
    df_temporal = extract_temporal_features(df_cleaned)

    # Drop records where target Internet activity is missing or null
    if TARGET_COLUMN in df_temporal.columns:
        df_temporal = df_temporal.dropna(subset=[TARGET_COLUMN])
        df_temporal[TARGET_COLUMN] = df_temporal[TARGET_COLUMN].clip(lower=0.0)

    return df_temporal.reset_index(drop=True)


def prepare_features_and_target(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature matrix X and target vector y.
    Ensures non-predictive identifiers (SquareID, CountryCode, raw TimeInterval)
    are excluded from direct regression inputs.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Features missing from dataframe: {missing_cols}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from dataframe.")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    return X, y
