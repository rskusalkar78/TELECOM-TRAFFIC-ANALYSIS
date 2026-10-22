"""
Model Training Module for Telecom Traffic Analysis.

Responsible for:
- Splitting features and target without data leakage.
- Implementing GroupShuffleSplit or GroupKFold based on TimeInterval
  (so multiple cellular grid records from the same time interval stay in the same split).
- Training a Random Forest Regressor to estimate Internet activity
  based on telecom activity indicators and cyclical temporal variables.
- Serializing the trained model and feature metadata using Joblib under models/.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

from src.preprocessing import prepare_features_and_target, FEATURE_COLUMNS, TARGET_COLUMN


def split_data_leakage_free(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing partitions.
    Uses GroupShuffleSplit grouped by 'TimeInterval' (or 'datetime') if present
    to prevent temporal leakage across spatial grid cells observed in the exact same time window.
    """
    X, y = prepare_features_and_target(df)

    if "TimeInterval" in df.columns:
        groups = df["TimeInterval"]
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        # Fallback standard train/test split with shuffle
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 15,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """
    Fit a Random Forest Regressor on the training telecom features.
    """
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    rf_model.fit(X_train, y_train)
    return rf_model


def save_model_artifacts(
    model: RandomForestRegressor,
    feature_names: list,
    model_dir: Path = Path("models"),
    model_filename: str = "telecom_rf_model.joblib",
) -> Path:
    """
    Persist the trained model bundle along with feature expectations.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    target_path = model_dir / model_filename

    payload: Dict[str, Any] = {
        "model": model,
        "feature_names": feature_names,
        "target_name": TARGET_COLUMN,
        "model_type": "RandomForestRegressor",
        "description": "Estimates cellular Internet activity from call/SMS counts and temporal features.",
    }
    joblib.dump(payload, target_path)
    return target_path


def run_training_pipeline(
    df: pd.DataFrame,
    model_dir: Path = Path("models"),
    random_state: int = 42,
) -> Tuple[RandomForestRegressor, pd.DataFrame, pd.Series, Path]:
    """
    End-to-end training pipeline:
    - Split dataset into train and test sets.
    - Train Random Forest Regressor.
    - Serialize model artifact to models/.
    - Return model, test set, and artifact location.
    """
    X_train, X_test, y_train, y_test = split_data_leakage_free(
        df, test_size=0.2, random_state=random_state
    )

    model = train_random_forest(
        X_train, y_train, n_estimators=100, max_depth=15, random_state=random_state
    )

    artifact_path = save_model_artifacts(
        model=model,
        feature_names=list(X_train.columns),
        model_dir=model_dir,
    )

    return model, X_test, y_test, artifact_path
