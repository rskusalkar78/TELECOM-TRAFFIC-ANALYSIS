"""
Data Loader Module for Telecom Traffic Analysis.

Responsible for:
- Validating the dataset location and structure.
- Reading tab-separated records in chunks to conserve memory.
- Safely handling missing and malformed records.
- Providing reproducible downsampling to avoid memory crashes on large datasets.
- Optional synthetic data generation exclusively for local testing and CI when
  the raw ~100MB-1GB Milan telecom dataset is not yet downloaded.
"""

from pathlib import Path
from typing import Optional, Generator, List
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Expected standard schema from the Milan Telecom Dataset
# Source: https://doi.org/10.7910/DVN/EGZHFV
EXPECTED_COLUMNS: List[str] = [
    "SquareID",
    "TimeInterval",
    "CountryCode",
    "SMS-in",
    "SMS-out",
    "Call-in",
    "Call-out",
    "Internet",
]

DEFAULT_DATA_PATH = Path("data") / "sms-call-internet-mi-2013-11-01.txt"


def check_dataset_exists(data_path: Path = DEFAULT_DATA_PATH) -> bool:
    """Check if the expected dataset file exists on disk."""
    return data_path.is_file() and data_path.stat().st_size > 0


def get_missing_dataset_instructions(data_path: Path = DEFAULT_DATA_PATH) -> str:
    """Return friendly, clear instructions for downloading and placing the dataset."""
    return (
        f"\n{'='*70}\n"
        f"DATASET NOT FOUND: '{data_path}'\n"
        f"{'='*70}\n"
        f"The primary Milan Telecommunications Activity dataset was not found.\n\n"
        f"To obtain the real dataset:\n"
        f"1. Visit the Harvard Dataverse repository:\n"
        f"   https://doi.org/10.7910/DVN/EGZHFV\n"
        f"2. Download the file named:\n"
        f"   'sms-call-internet-mi-2013-11-01.txt' (or .gz equivalent)\n"
        f"3. Place the extracted text file in the project directory:\n"
        f"   {data_path.resolve()}\n\n"
        f"For testing purposes or demonstration without the full dataset,\n"
        f"you can pass `--use-synthetic` to `python main.py`.\n"
        f"{'='*70}\n"
    )


def read_data_chunks(
    data_path: Path = DEFAULT_DATA_PATH,
    chunksize: int = 50_000,
) -> Generator[pd.DataFrame, None, None]:
    """
    Read tab-separated telecom data in chunks.

    Handles headerless or header-present data safely, validates column count,
    and yields sanitized pandas DataFrames chunk by chunk.
    """
    if not check_dataset_exists(data_path):
        raise FileNotFoundError(get_missing_dataset_instructions(data_path))

    # Inspect the first line to check if headers are already included
    with open(data_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip().split("\t")

    has_header = first_line == EXPECTED_COLUMNS

    reader = pd.read_csv(
        data_path,
        sep="\t",
        names=EXPECTED_COLUMNS if not has_header else None,
        header=0 if has_header else None,
        chunksize=chunksize,
        low_memory=False,
        on_bad_lines="skip",
    )

    for chunk in reader:
        # Enforce column layout match
        if list(chunk.columns) != EXPECTED_COLUMNS:
            chunk.columns = EXPECTED_COLUMNS
        yield chunk


def load_and_sample_data(
    data_path: Path = DEFAULT_DATA_PATH,
    sample_size: int = 100_000,
    random_state: int = 42,
    save_sample_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Read the dataset in chunks and create a reproducible sample of records.
    Prevents memory exhaustion while preserving statistical properties.

    Any sampled copy is written under outputs/ if requested, NEVER overwriting
    the raw data file.
    """
    if not check_dataset_exists(data_path):
        raise FileNotFoundError(get_missing_dataset_instructions(data_path))

    accumulated_chunks: List[pd.DataFrame] = []
    total_rows = 0

    # Read chunks until sample_size * 2 or EOF to allow random sampling
    target_reservoir = sample_size * 2
    for chunk in read_data_chunks(data_path=data_path, chunksize=50_000):
        accumulated_chunks.append(chunk)
        total_rows += len(chunk)
        if total_rows >= target_reservoir:
            break

    combined_df = pd.concat(accumulated_chunks, ignore_index=True)

    if len(combined_df) > sample_size:
        sampled_df = combined_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    else:
        sampled_df = combined_df.copy()

    if save_sample_path is not None:
        save_sample_path.parent.mkdir(parents=True, exist_ok=True)
        sampled_df.to_csv(save_sample_path, index=False)
        logger.info(f"Saved reproducible sampled dataset to: {save_sample_path}")

    return sampled_df


def generate_synthetic_data(
    n_records: int = 2500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate an in-memory synthetic telecom DataFrame strictly for automated tests
    and dry-run verification when the real dataset is not downloaded.

    NOTE: This data is purely synthetic. It MUST NOT be presented as real experimental
    results or used in the final mini-project report.
    """
    np.random.seed(random_state)

    # 2013-11-01 00:00:00 GMT in milliseconds: 1383264000000
    base_timestamp_ms = 1383264000000
    ten_min_ms = 600_000  # 10 minutes interval

    square_ids = np.random.choice([1000, 1001, 1002, 1003, 5000, 5001], size=n_records)
    time_intervals = [
        base_timestamp_ms + int(np.random.randint(0, 144) * ten_min_ms)
        for _ in range(n_records)
    ]
    country_codes = np.random.choice([39, 0, 44, 33, 49], size=n_records)

    sms_in = np.maximum(0.0, np.random.exponential(scale=1.5, size=n_records))
    sms_out = np.maximum(0.0, np.random.exponential(scale=1.2, size=n_records))
    call_in = np.maximum(0.0, np.random.exponential(scale=2.0, size=n_records))
    call_out = np.maximum(0.0, np.random.exponential(scale=2.2, size=n_records))

    # Internet traffic exhibits diurnal patterns correlated with calls & SMS
    noise = np.random.normal(0, 5.0, size=n_records)
    internet = np.maximum(
        0.0,
        5.0 * sms_in + 3.0 * sms_out + 12.0 * call_in + 15.0 * call_out + 20.0 + noise,
    )

    # Inject a few missing values to test preprocessing resilience
    mask_sms = np.random.rand(n_records) < 0.05
    sms_in[mask_sms] = np.nan
    mask_internet = np.random.rand(n_records) < 0.02
    internet[mask_internet] = np.nan

    df = pd.DataFrame(
        {
            "SquareID": square_ids,
            "TimeInterval": time_intervals,
            "CountryCode": country_codes,
            "SMS-in": sms_in,
            "SMS-out": sms_out,
            "Call-in": call_in,
            "Call-out": call_out,
            "Internet": internet,
        }
    )
    return df
