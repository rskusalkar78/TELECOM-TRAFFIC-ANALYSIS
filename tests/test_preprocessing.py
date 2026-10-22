"""
Unit tests for data preprocessing module.

Covers:
- Timestamp conversion from epoch milliseconds.
- Missing and NaN value handling for telecom measurements.
- Negative value clipping.
- Temporal feature engineering (hour, cyclical sin/cos).
- Empty dataframe and malformed input handling.
- Input validation and separation of features and targets.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    convert_timestamp,
    clean_measurements,
    extract_temporal_features,
    preprocess_telecom_data,
    prepare_features_and_target,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """Fixture providing a small in-memory telecom dataframe."""
    return pd.DataFrame(
        {
            "SquareID": [1, 2, 3],
            "TimeInterval": [1383264000000, 1383264600000, 1383265200000],  # 2013-11-01 00:00, 00:10, 00:20
            "CountryCode": [39, 39, 44],
            "SMS-in": [1.5, np.nan, 2.0],
            "SMS-out": [0.5, 1.0, -0.5],
            "Call-in": [3.0, 2.5, 0.0],
            "Call-out": [np.nan, 1.2, 4.1],
            "Internet": [50.2, 60.1, 75.3],
        }
    )


def test_timestamp_conversion(sample_raw_dataframe):
    """Verify that millisecond epoch integers correctly convert to UTC datetime."""
    df_converted = convert_timestamp(sample_raw_dataframe)
    assert "datetime" in df_converted.columns
    assert pd.api.types.is_datetime64_any_dtype(df_converted["datetime"])
    assert df_converted["datetime"].iloc[0] == pd.Timestamp("2013-11-01 00:00:00")


def test_missing_and_invalid_measurement_handling(sample_raw_dataframe):
    """Verify missing telecom activity values are zero-filled and negative values clipped."""
    df_cleaned = clean_measurements(sample_raw_dataframe)

    # NaN in SMS-in and Call-out should be 0.0
    assert df_cleaned["SMS-in"].iloc[1] == 0.0
    assert df_cleaned["Call-out"].iloc[0] == 0.0

    # Negative measurement -0.5 in SMS-out should be clipped to 0.0
    assert df_cleaned["SMS-out"].iloc[2] == 0.0
    assert (df_cleaned["SMS-out"] >= 0.0).all()


def test_extract_temporal_features(sample_raw_dataframe):
    """Verify hour extraction, cyclical sin/cos coordinates, and total aggregates."""
    df_converted = convert_timestamp(sample_raw_dataframe)
    df_feat = extract_temporal_features(df_converted)

    assert "hour" in df_feat.columns
    assert "day_of_week" in df_feat.columns
    assert "hour_sin" in df_feat.columns
    assert "hour_cos" in df_feat.columns
    assert "total_sms" in df_feat.columns
    assert "total_calls" in df_feat.columns

    # Hour 0 sin is 0.0 and cos is 1.0
    assert np.isclose(df_feat["hour_sin"].iloc[0], 0.0, atol=1e-5)
    assert np.isclose(df_feat["hour_cos"].iloc[0], 1.0, atol=1e-5)


def test_empty_dataframe_handling():
    """Verify that passing an empty DataFrame returns an empty DataFrame without raising uncaught errors."""
    empty_df = pd.DataFrame()
    processed = preprocess_telecom_data(empty_df)
    assert processed.empty


def test_malformed_timestamp_coercion():
    """Verify that malformed or non-convertible timestamps are handled gracefully without crash."""
    malformed_df = pd.DataFrame(
        {
            "SquareID": [99],
            "TimeInterval": ["invalid_string_not_a_timestamp"],
            "SMS-in": [1.0],
            "SMS-out": [1.0],
            "Call-in": [1.0],
            "Call-out": [1.0],
            "Internet": [10.0],
        }
    )
    processed = preprocess_telecom_data(malformed_df)
    # The malformed timestamp row has NaT datetime and should be safely dropped
    assert len(processed) == 0


def test_prepare_features_and_target(sample_raw_dataframe):
    """Verify feature matrix X and target y extraction."""
    processed = preprocess_telecom_data(sample_raw_dataframe)
    X, y = prepare_features_and_target(processed)

    for col in FEATURE_COLUMNS:
        assert col in X.columns

    assert y.name == TARGET_COLUMN
    assert len(X) == len(y)
    # Non-predictive identifiers must be absent from feature matrix
    assert "SquareID" not in X.columns
    assert "CountryCode" not in X.columns
    assert "TimeInterval" not in X.columns
