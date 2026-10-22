# Telecom Traffic Analysis and Prediction Using Machine Learning

An academic mini-project designed for **Third-Year Electronics & Telecommunication Engineering (ETE / ECE)** students. This repository provides a modular, reproducible, and robust Python pipeline to analyze cellular telecommunications traffic patterns (SMS, Voice Calls, Internet data) and estimate mobile data activity using machine learning.

---

## 1. Project Overview & Problem Statement

Mobile telecommunications networks generate massive volumes of continuous operational activity. Network operators face fluctuating demand throughout the day, creating network congestion, channel bottlenecks, and unbalanced base station power consumption.

### Objectives:
- **Analyze telecommunication activity patterns** across urban cellular grid cells.
- **Uncover diurnal (hourly) cyclical patterns** in Call, SMS, and Mobile Internet traffic.
- **Establish cross-modality correlations** between voice/SMS traffic and cellular internet activity.
- **Estimate cellular Internet activity** using a trained Random Forest Regressor while preventing temporal data leakage.
- Provide a clean, testable codebase meeting academic mini-project demonstration standards.

> **Academic Note on Modeling Scope**:
> This project builds a model that **estimates observed cellular internet activity** from contemporaneous telecommunications indicators (SMS, Calls) and time-of-day cyclical features. It deliberately distinguishes *activity estimation* from *future time-series forecasting* (which requires sequential autoregressive lag features evaluated on chronological holdout sets).

---

## 2. Dataset Information

- **Dataset Title**: Telecom Italia Open Big Data Challenge - Milano Telecommunications Activity
- **Harvard Dataverse DOI**: [https://doi.org/10.7910/DVN/EGZHFV](https://doi.org/10.7910/DVN/EGZHFV)
- **Expected Filename**: `sms-call-internet-mi-2013-11-01.txt`
- **Expected Path**: `data/sms-call-internet-mi-2013-11-01.txt`

### Dataset Attributes:
| Column | Description |
| :--- | :--- |
| **SquareID** | Grid cell identifier representing a spatial square in Milan |
| **TimeInterval** | Epoch timestamp in milliseconds indicating the start of a 10-minute window |
| **CountryCode** | Country phone prefix code (e.g., 39 for Italy) |
| **SMS-in** | Ingoing SMS activity counter for the square |
| **SMS-out** | Outgoing SMS activity counter for the square |
| **Call-in** | Ingoing voice call activity counter for the square |
| **Call-out** | Outgoing voice call activity counter for the square |
| **Internet** | Cellular Internet traffic activity index (normalized CDR aggregation) |

> **Important**: The traffic measurements represent anonymized, aggregated cellular activity indicators, not individual call logs or raw megabytes.

---

## 3. Project Directory Structure

```text
Telecom-Traffic-Analysis/
├── data/
│   └── .gitkeep               # Place sms-call-internet-mi-2013-11-01.txt here
├── outputs/
│   ├── .gitkeep
│   ├── summary_statistics.txt
│   ├── eda_summary.txt
│   ├── hourly_internet_activity.png
│   ├── activity_comparison.png
│   ├── correlation_heatmap.png
│   ├── model_evaluation_metrics.csv
│   └── actual_vs_predicted.png
├── models/
│   ├── .gitkeep
│   └── telecom_rf_model.joblib
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Chunked reader, validation, synthetic data generator
│   ├── preprocessing.py       # Timestamp conversion, cyclical encoding, sanitization
│   ├── eda.py                 # Visualizations and descriptive statistics
│   ├── train.py               # Leakage-free GroupShuffleSplit & Random Forest training
│   ├── evaluate.py            # MAE, RMSE, R² & baseline comparison
│   └── predict.py             # Inference class & sample predictor
├── tests/
│   └── test_preprocessing.py  # In-memory pytest test suite
├── main.py                    # Master execution CLI
├── requirements.txt           # Minimal dependencies
├── .gitignore                 # Excludes caches, venv, models, and raw datasets
├── README.md                  # Setup & usage instructions
└── PROJECT_REPORT.md          # Academic report template
```

---

## 4. Setup and Installation (Windows PowerShell)

### Step 1: Open Terminal in Project Folder
Open Windows PowerShell and navigate to the project directory:
```powershell
cd c:\Users\ROHAN\Desktop\Telecom-Traffic-Analysis
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 5. Running the Project

### Option A: Running with Dataset Check (Real Data)
If the real dataset is present in `data/sms-call-internet-mi-2013-11-01.txt`:
```powershell
python main.py
```
If the file is not yet placed in `data/`, running `python main.py` prints clear download instructions without crashing.

### Option B: Quick Verification & Dry Run (Synthetic Mode)
To test and verify the entire machine learning pipeline without downloading the full ~100MB+ dataset:
```powershell
python main.py --use-synthetic
```
*(Synthetic results are clearly marked in terminal logs and outputs as synthetic).*

### Option C: Running Unit Tests
Execute the automated test suite using `pytest`:
```powershell
pytest -v
```

---

## 6. Generated Outputs and Evaluation Metrics

All outputs are automatically generated under the `outputs/` folder:

| File Name | Description |
| :--- | :--- |
| `summary_statistics.txt` | Complete descriptive statistics (mean, std, percentiles) for telecom counters |
| `eda_summary.txt` | Textual summary of peak diurnal hours and cross-channel correlations |
| `hourly_internet_activity.png` | Diurnal activity curve showing hourly mean traffic and variance band |
| `activity_comparison.png` | Normalized trajectories of Calls vs. SMS vs. Internet across the day |
| `correlation_heatmap.png` | Pearson correlation matrix between telecommunication features |
| `model_evaluation_metrics.csv` | Comparative metrics table: Random Forest vs. Mean Baseline |
| `actual_vs_predicted.png` | Scatter plot comparing predicted vs actual internet traffic with 45° reference line |

### Metrics Evaluated:
1. **MAE (Mean Absolute Error)**: Average absolute magnitude of errors in traffic prediction units.
2. **RMSE (Root Mean Squared Error)**: Standard deviation of residuals, penalizing larger traffic estimation spikes.
3. **R² Score (Coefficient of Determination)**: Proportion of variance in Internet traffic explained by telecom activity features compared to the mean baseline.

---

## 7. Model Reusability & Inference (`src/predict.py`)

You can import the trained model and make predictions in your own Python scripts:

```python
from src.predict import TelecomTrafficPredictor

predictor = TelecomTrafficPredictor()

sample_reading = {
    "SMS-in": 3.2,
    "SMS-out": 2.1,
    "Call-in": 5.4,
    "Call-out": 4.0,
    "hour": 15,          # 3:00 PM
    "day_of_week": 4      # Friday (0=Monday, 6=Sunday)
}

predicted_activity = predictor.predict(sample_reading)[0]
print(f"Estimated Cellular Internet Activity: {predicted_activity:.2f}")
```

---

## 8. Limitations and Future Improvements

1. **Spatial Aggregation**: Current models estimate activity at grid-level independently. Graph Neural Networks (GNN) or Spatial Lag models could incorporate spatial autocorrelation across neighboring cells.
2. **Chronological Time-Series Forecasting**: Future extensions can engineer lagged rolling features ($t-1$, $t-2$, $t-24$) with chronological split to forecast traffic ahead of time for dynamic power saving and base station sleep modes.
3. **Multi-Day Trends**: Integrating multi-week data would allow modeling of weekend vs. weekday traffic shifts and public holiday dynamics.
