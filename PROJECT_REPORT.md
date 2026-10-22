# TELECOM TRAFFIC ANALYSIS AND PREDICTION USING MACHINE LEARNING
## Academic Mini-Project Report
**Department of Electronics & Telecommunication Engineering**

---

### Abstract
The continuous expansion of cellular subscriber bases and data-intensive mobile applications has placed significant load on cellular radio access networks (RAN). Efficient cellular network operation requires an empirical understanding of how cellular traffic behaves across temporal cycles and how disparate communication modalities interact. In this project, an end-to-end Python pipeline is developed to analyze multi-modal telecommunication activity (SMS, voice calls, and mobile Internet) utilizing the Telecom Italia Open Big Data Challenge benchmark dataset for the metropolitan area of Milan (`sms-call-internet-mi-2013-11-01.txt`). 

Exploratory data analysis demonstrates pronounced diurnal patterns, exhibiting a nighttime trough at 04:00 and peak cellular data usage at 16:00, along with strong cross-modality Pearson correlations ($r \approx 0.75 - 0.80$) between voice/SMS traffic and cellular internet activity. A Random Forest regression model was trained to estimate contemporaneous mobile Internet activity from cellular communication counters and cyclical time features ($\sin$ and $\cos$ 24-hour transforms). To prevent temporal data leakage across concurrent spatial cells, a `GroupShuffleSplit` partitioned on 10-minute time intervals was enforced. Evaluated on an independent holdout set of 8,011 records, the Random Forest model achieved a Mean Absolute Error (MAE) of 4.2142, a Root Mean Squared Error (RMSE) of 9.0434, and a Coefficient of Determination ($R^2$) of 0.8035, significantly outperforming a naive mean baseline predictor (MAE 13.1289, RMSE 20.4035, $R^2 \approx 0.0$). The pipeline is modular, fully reproducible, accompanied by unit test suites, and provides a solid foundation for cellular load estimation.

---

### 1. Introduction
Modern cellular communications infrastructure (LTE, LTE-Advanced, and 5G NR) operates in dynamic environments where subscriber demand varies widely over time. Base station transceivers (eNodeB / gNodeB) consume substantial electrical power, accounting for more than 50-70% of total cellular operator energy budgets. Without data-driven insights into cellular demand, cellular operators risk either service degradation from radio channel congestion or excessive operational expenditure from continuously powering underutilized carrier frequencies.

Telecommunications activity logs provide valuable insight into urban mobility and communication trends. By analyzing Call Detail Records (CDRs) and cellular traffic counters across distinct hours of the day, network planners can identify demand patterns, uncover relationships between different cellular services, and build predictive estimation models to support dynamic load balancing and energy-aware carrier activation.

---

### 2. Problem Statement
Cellular network traffic exhibits strong spatial and temporal non-stationarity driven by human activity schedules. Network engineers face several operational challenges:
1. **Multi-Modality Ingestion**: Cellular traffic comprises disparate channels (ingoing/outgoing SMS, ingoing/outgoing voice calls, and packet-switched Internet access) that must be ingested, cleaned, and analyzed at scale without memory exhaustion.
2. **Diurnal Non-Linearity**: Simple linear thresholding fails to capture non-linear diurnal cycles and abrupt transitions between peak and off-peak hours.
3. **Data Leakage in Spatial-Temporal Datasets**: In cellular grid datasets where hundreds of spatial cells are logged simultaneously within identical 10-minute time slices, naive random splitting causes temporal data leakage between train and test sets.
4. **Need for Reliable Estimation**: Operators require dependable methods to estimate cellular internet demand based on available telecommunications counters and time indicators.

---

### 3. Objectives
- **Robust Data Pipeline**: Implement memory-efficient, chunked processing of tab-separated cellular activity records with schema validation and reproducible sampling.
- **Exploratory Traffic Characterization**: Quantify diurnal activity curves and determine empirical Pearson correlations between voice, SMS, and data traffic.
- **Temporal Feature Engineering**: Implement continuous cyclical encodings ($\sin$ and $\cos$ 24-hour periodic transformations) to preserve midnight boundary continuity.
- **Leakage-Free Model Training**: Formulate a Random Forest regression model trained with `GroupShuffleSplit` grouped by time intervals.
- **Quantitative Benchmarking**: Evaluate estimation accuracy using MAE, RMSE, and $R^2$ against a naive mean baseline model.
- **Academic & Engineering Standards**: Provide a well-documented, testable codebase with automated unit testing for college project demonstration.

---

### 4. Literature Review
Cellular traffic modeling has evolved from classical queuing theory toward statistical machine learning and data-driven pattern analysis:

1. **Traffic Characterization in Urban Cellular Networks**:
   Barlacchi et al. [1] published the multi-source urban life dataset for the city of Milan and Trentino, releasing aggregated telecommunications activity logs that have become the primary open benchmark for urban mobile computing research. Their work established standard spatial tessellation (grid squares) and 10-minute aggregation intervals for cellular indicators.

2. **Temporal Dynamics and Cross-Service Correlations**:
   Empirical studies on urban cellular networks by Naboulsi et al. [2] categorize mobile network data usage and reveal that human mobility and societal schedules govern cellular traffic patterns. Their findings confirm that cellular voice and text messaging correlate strongly with data consumption during peak commuting and working hours, supporting the use of multi-service indicators for traffic estimation.

3. **Machine Learning for Cellular Traffic Estimation**:
   Non-linear ensemble learning, specifically Random Forest regression proposed by Breiman [3], has been widely adopted in network performance modeling due to its resilience against multicollinearity, capacity to capture complex feature interactions, and lack of sensitivity to unscaled feature ranges. Pedregosa et al. [4] provided the foundational Scikit-learn architecture used to implement robust ensemble regression and group-based validation.

4. **Group Validation vs. Time-Series Splitting**:
   In spatial-temporal data mining, Roberts et al. [5] demonstrated that cross-validation strategies must respect grouping structures to prevent optimistic bias caused by spatial or temporal cross-talk. When evaluating simultaneous cellular grid observations, group-based holdout schemes ensure unbiased performance estimates.

---

### 5. Dataset Description
- **Source**: Harvard Dataverse / Telecom Italia Open Big Data Challenge (Milan).
- **Reference URL / DOI**: https://doi.org/10.7910/DVN/EGZHFV
- **Primary Data File**: `sms-call-internet-mi-2013-11-01.txt`
- **Spatial Domain**: Metropolitan Milan partitioned into a grid of 10,000 squares ($100 \times 100$, approx. $235\text{ m} \times 235\text{ m}$ per cell).
- **Temporal Domain**: 10-minute aggregation windows spanning 2013-10-31 23:00:00 to 2013-11-01 22:50:00 UTC (144 time intervals).
- **Tab-Separated Attributes**:
  1. `SquareID`: Integer identifier of the spatial grid cell.
  2. `TimeInterval`: Epoch timestamp in milliseconds indicating the start of the 10-minute window.
  3. `CountryCode`: Country calling code prefix (e.g., 39 for Italy).
  4. `SMS-in`: Normalized incoming SMS activity counter.
  5. `SMS-out`: Normalized outgoing SMS activity counter.
  6. `Call-in`: Normalized incoming voice call activity counter.
  7. `Call-out`: Normalized outgoing voice call activity counter.
  8. `Internet`: Normalized cellular data activity counter (target variable).

*Nature of Activity Values*: The values represent anonymized, aggregated cellular operational transaction indices rather than raw megabytes, active IP connections, or personal subscriber logs.

---

### 6. System Architecture and Methodology

#### 6.1 Architectural Workflow

```text
[Raw Telecom Dataset: data/sms-call-internet-mi-2013-11-01.txt]
                         │
                         ▼
          [Chunked Ingestion & Schema Validation]
         (50,000 rows/chunk, check 8 expected cols)
                         │
                         ▼
        [Reproducible Sampling: 75,000 raw records]
                         │
                         ▼
        [Preprocessing & Feature Engineering]
         ├── Epoch ms -> UTC Datetime conversion
         ├── Communication channels NaN fill (0.0) & clip (>= 0.0)
         ├── Drop missing target records (Internet NaN)
         ├── Cyclical Hour Encodings: sin(2π*h/24), cos(2π*h/24)
         └── Aggregates: total_sms, total_calls
                         │ (38,876 valid records remaining)
         ┌───────────────┴────────────────────────┐
         ▼                                        ▼
[Exploratory Data Analysis]            [Feature Matrix & Target]
 ├── Diurnal curves (hourly mean)       ├── Features (X): 10 columns
 ├── Cross-modality comparison          └── Target (y): Internet activity
 ├── Correlation heatmap                          │
 └── Statistical summary                          ▼
                                       [GroupShuffleSplit on TimeInterval]
                                        (Train: 80% / Test: 20%, 8,011 samples)
                                                  │
                                                  ▼
                                       [Random Forest Regressor Fitting]
                                        (100 estimators, max_depth=15)
                                                  │
                                                  ▼
                                       [Quantitative Evaluation]
                                        ├── Compare vs. Mean Baseline
                                        ├── Compute MAE, RMSE, R²
                                        └── Actual vs. Predicted Plot
                                                  │
                                                  ▼
                                       [Joblib Serialization: models/]
```

#### 6.2 Data Cleaning and Record Filtering Explanation
In the initial data ingestion step, 75,000 raw records were loaded into the reproducible sample. After preprocessing, 38,876 valid records remained. The reduction of 36,124 records occurs because:
1. **Missing Target (`Internet` is NaN)**: In the raw Milan dataset, several country-specific grid records log voice/SMS events without corresponding cellular internet activity for that country code, resulting in `NaN` in the `Internet` column. Since supervised regression requires a ground-truth target, records with missing target values are safely dropped.
2. **Communication Missingness**: In contrast to the target variable, missing values in predictor columns (`SMS-in`, `SMS-out`, `Call-in`, `Call-out`) indicate that zero transactions occurred in that channel during that 10-minute window. These are zero-filled and clipped to $\ge 0.0$.
3. **Identifier Exclusion**: Spatial identifiers (`SquareID`, `CountryCode`) and raw epoch strings (`TimeInterval`) are excluded from the feature matrix to avoid spurious memorization of non-generalizable categorical keys.

#### 6.3 Temporal Feature Engineering
To enable the regression model to understand diurnal human rhythm without artificial boundary discontinuities between 23:59 and 00:00, cyclical trigonometric transformations of the hour variable $h \in [0, 23]$ were computed:
$$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot h}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot h}{24}\right)$$
Additionally, aggregate interaction indicators $\text{total\_sms} = \text{SMS-in} + \text{SMS-out}$ and $\text{total\_calls} = \text{Call-in} + \text{Call-out}$ were constructed.

#### 6.4 Preventing Data Leakage: GroupShuffleSplit
In telecommunications grid data, multiple grid squares are recorded at the exact same 10-minute timestamp (`TimeInterval`). Standard random splitting would assign records from the same timestamp to both training and test sets, causing temporal data leakage.

To prevent this, `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` was configured with `groups = df['TimeInterval']`. This guarantees that all grid cells observed in a given 10-minute window are held out exclusively in either the training set or the testing set.

> **Methodological Clarification**: Holding out complete time-interval groups evaluates how well the model generalizes across unobserved temporal windows. However, because `GroupShuffleSplit` selects time slices randomly across the 24-hour cycle rather than holding out the final chronological portion of the day with lagged autoregressive features ($y_{t-1}, y_{t-2}$), this model is strictly an **activity estimation model** and does not claim to perform future time-series forecasting.

---

### 7. Implementation Details
The project is organized into modular Python components:
- `src/data_loader.py`: Handles chunked CSV reading (`chunksize=50_000`), schema verification, reproducible reservoir sampling, and synthetic fallback generation.
- `src/preprocessing.py`: Implements timestamp conversions (`unit='ms'`), data cleaning, cyclical encodings, and feature matrix separation.
- `src/eda.py`: Generates publication-ready Matplotlib and Seaborn figures and summary files.
- `src/train.py`: Executes the `GroupShuffleSplit`, trains the `RandomForestRegressor`, and serializes the model to `models/telecom_rf_model.joblib`.
- `src/evaluate.py`: Computes MAE, RMSE, and $R^2$ against a mean baseline and generates actual-versus-predicted plots.
- `src/predict.py`: Houses the `TelecomTrafficPredictor` class for standalone inference.
- `main.py`: Master CLI driving the automated workflow.
- `tests/test_preprocessing.py`: Pytest suite containing 6 test cases verifying timestamp parsing, missing-value imputation, clipping, and feature matrix structure.

---

### 8. Results and Discussion

#### 8.1 Descriptive Statistics
Summary statistics computed across the 38,876 processed records (`outputs/summary_statistics.txt`):

| Variable | Count | Mean | Std Dev | Min | 25% | 50% (Median) | 75% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SMS-in** | 38,876 | 0.4103 | 1.0606 | 0.0000 | 0.0000 | 0.0781 | 0.3412 | 45.1247 |
| **SMS-out** | 38,876 | 0.3537 | 0.8576 | 0.0000 | 0.0000 | 0.0729 | 0.2922 | 27.6891 |
| **Call-in** | 38,876 | 0.3762 | 1.0262 | 0.0000 | 0.0000 | 0.0401 | 0.2604 | 20.3444 |
| **Call-out** | 38,876 | 0.4134 | 1.0879 | 0.0000 | 0.0000 | 0.0520 | 0.3014 | 23.0809 |
| **Internet** | 38,876 | 11.7472 | 21.4747 | 0.000005 | 0.0882 | 3.9518 | 12.6213 | 402.5366 |
| **total_sms** | 38,876 | 0.7640 | 1.8238 | 0.0000 | 0.0000 | 0.1698 | 0.6544 | 64.3015 |
| **total_calls** | 38,876 | 0.7896 | 2.0741 | 0.0000 | 0.0000 | 0.1020 | 0.5730 | 38.4004 |

The distribution shows marked positive skewness: while the median Internet activity is 3.9518, localized high-demand squares reach up to 402.5366, typical of heavy commercial centers and transport hubs in Milan.

#### 8.2 Exploratory Findings & Correlations
From `outputs/eda_summary.txt` and generated plots:
1. **Diurnal Pattern**: Mobile Internet traffic exhibits a clear human behavioral cycle. Activity drops to its minimum trough at **04:00**, begins climbing steeply during the morning commute (07:00-09:00), and peaks in the afternoon at **16:00** before stabilizing during evening hours.
2. **Modality Correlations**: Pearson correlation coefficients between cellular Internet activity and telephony indicators are strongly positive:
   - Correlation with `Call-out`: **0.799**
   - Correlation with `Call-in`: **0.795**
   - Correlation with `SMS-in`: **0.788**
   - Correlation with `SMS-out`: **0.750**
   
   This strong co-occurrence demonstrates that voice calls and text messaging provide high explanatory signal for predicting cellular internet activity.

#### 8.3 Quantitative Model Evaluation
The Random Forest Regressor was tested on 8,011 holdout samples from unobserved time-intervals and compared against a Baseline Mean Predictor (which always outputs the training set mean $\bar{y}_{\text{train}}$):

| Model | MAE | RMSE | $R^2$ Score | Sample Count |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Mean)** | 13.1289 | 20.4035 | -0.0000 | 8,011 |
| **Random Forest Regressor** | **4.2142** | **9.0434** | **0.8035** | 8,011 |

#### 8.4 Discussion
- **Error Reduction**: The Random Forest regressor achieves a **67.9% reduction in MAE** (from 13.1289 down to 4.2142) and a **55.7% reduction in RMSE** (from 20.4035 down to 9.0434) compared to the baseline.
- **Goodness of Fit ($R^2 = 0.8035$)**: Over 80% of the variance in cellular Internet traffic is explained by telephony activity levels and cyclical time features.
- **Residual Distribution**: The actual-versus-predicted plot (`outputs/actual_vs_predicted.png`) demonstrates close alignment along the $y = x$ ideal line for normal traffic ranges ($0 - 100$), with expected variance increasing at extreme values ($> 250$) where extreme burst events occur.

---

### 9. Test Verification & Code Integrity
The software implementation was validated through automated tests (`tests/test_preprocessing.py`). All 6 unit tests passed successfully:
1. `test_timestamp_conversion`: Confirmed correct parsing of millisecond epochs to UTC datetime.
2. `test_missing_and_invalid_measurement_handling`: Verified zero-imputation of telephony features and lower-bound clipping at zero.
3. `test_extract_temporal_features`: Verified hour extraction and trigonometric identity values ($\sin(0)=0, \cos(0)=1$).
4. `test_empty_dataframe_handling`: Verified graceful empty DataFrame handling without exceptions.
5. `test_malformed_timestamp_coercion`: Verified that malformed records with invalid timestamps are cleanly removed.
6. `test_prepare_features_and_target`: Confirmed that non-predictive keys (`SquareID`, `CountryCode`, `TimeInterval`) are excluded from feature matrices.

---

### 10. Limitations
1. **Contemporaneous Estimation vs. Future Forecasting**: The current model estimates current Internet traffic given concurrent telephony and temporal indicators. It cannot forecast future traffic hours in advance without autoregressive time-lagged features ($t - k$).
2. **Spatial Autocorrelation**: The current architecture treats each grid cell independently, without explicitly modeling spatial dependencies between adjacent cell towers.
3. **Single-Day Scope**: The processed dataset represents a 24-hour observation window (2013-11-01), meaning multi-day weekly trends (weekday vs. weekend patterns) are not captured in this single-day slice.
4. **Normalized Counters**: Telecom Italia provides normalized indicators rather than raw bitrates or physical bandwidth in Mbps, preventing direct translation to physical link capacity without operator calibration factors.

---

### 11. Future Scope
1. **Time-Series Lag Forecasting**: Implement rolling autoregressive features ($y_{t-1}, y_{t-2}, y_{t-24}$) with chronological forward-chaining splits to forecast cellular load 1 to 6 hours ahead.
2. **Spatial Graph Neural Networks (GNNs)**: Formulate cell grid adjacencies as an undirected spatial graph, applying Spatio-Temporal Graph Convolutional Networks (ST-GCN) to capture geographic traffic handovers.
3. **Green Cellular Base Station Sleep Scheduling**: Integrate load estimation into base station sleep-cycle controllers to dynamically power down secondary carriers during low-traffic night hours (e.g., 02:00 - 05:00) to reduce energy consumption.

---

### 12. References
1. Barlacchi, G., De Nadai, M., Larcher, R., Pentland, A., Lepri, B., & Oliver, N. (2015). A multi-source dataset of urban life in the city of Milan and the Province of Trentino. *Scientific Data*, 2, 150055. https://doi.org/10.1038/sdata.2015.55
2. Naboulsi, D., Fiore, M., Ribot, S., & Stanica, R. (2016). Large-scale mobile traffic analysis: A survey. *IEEE Communications Surveys & Tutorials*, 18(1), 124-161.
3. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
4. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., & Vanderplas, J. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
5. Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., & Warton, D. I. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure. *Ecography*, 40(8), 913-929.
