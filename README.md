# Stock Price Time Series Analysis & Forecasting

## Problem Statement

Stock price prediction is one of the most challenging and high-stakes problems in financial data science. Raw price data is noisy, non-stationary, and influenced by countless external factors — making it difficult to extract meaningful signals. This project tackles three core questions:

1. **Trend & Seasonality** — Does the stock exhibit long-term trends or recurring seasonal patterns (monthly, weekly)?
2. **Feature Importance** — Which temporal and statistical features (lag values, rolling statistics, calendar features) are most predictive of future price?
3. **Forecasting** — Can we build a model that accurately forecasts future stock prices using historical patterns?

---

## Dataset Details

| Property | Details |
|---|---|
| **Source** | `Stock Price Dataset.zip` (CSV inside) |
| **Format** | CSV with a `date` column and numeric price/volume fields |
| **Key Columns** | `date`, `last_value` (closing/last price), `turnover` (trading volume proxy) |
| **Frequency** | Daily trading data |
| **Preprocessing** | Date parsed → set as index → sorted chronologically → rows with missing values dropped |

The dataset captures daily stock price snapshots. Non-trading days are forward-filled during ARIMA modeling to maintain a continuous daily frequency.

---

## Approach

### 1. Exploratory Visualization
- Plotted `last_value` over time to visually identify trends, volatility periods, and structural breaks.

### 2. Time Series Decomposition
- Applied **additive seasonal decomposition** (`statsmodels.seasonal_decompose`) with a period of **252 trading days** (approximate annual cycle) to separate the series into:
  - **Trend** — long-term direction
  - **Seasonal** — repeating periodic pattern
  - **Residual** — unexplained noise

### 3. Feature Engineering
Created a rich feature set to capture temporal structure:

| Feature Type | Features |
|---|---|
| Calendar | `year`, `month`, `day_of_week`, `day_of_year`, `week_of_year`, `quarter` |
| Lag | `last_value_lag_1` (1 day), `last_value_lag_5` (≈1 week), `last_value_lag_22` (≈1 month) |
| Rolling Window | `rolling_mean_5` (5-day mean), `rolling_std_5` (5-day std dev) |

Rows with NaN values introduced by shifting/rolling were dropped.

### 4. Random Forest — Feature Importance
- Trained a **`RandomForestRegressor`** (100 trees, `random_state=42`) on an 80/20 time-based train-test split.
- Extracted and ranked feature importances to identify which signals most influence price prediction.

### 5. ARIMA Forecasting
- Resampled the series to daily frequency and forward-filled non-trading days.
- Fit an **ARIMA(5, 1, 0)** model (AR order 5, first-order differencing, no MA term) on the training set.
- Generated predictions on the test set and evaluated using **RMSE**.

### 6. Seasonality Pattern Analysis
- **Monthly analysis**: Computed average `last_value` per calendar month to reveal intra-year price tendencies.
- **Weekly analysis**: Computed average `turnover` per day of the week to detect activity patterns across the trading week.

---

## Results

### Time Series Decomposition
The decomposition successfully isolated the underlying trend from periodic fluctuations, revealing whether price movements are driven by long-term growth/decline or by cyclical seasonal effects.

### Feature Importance (Random Forest)
Lag features — particularly `last_value_lag_1` (previous day's price) — ranked as the most important predictors, consistent with the well-known autocorrelation in stock price series. Rolling statistics (`rolling_mean_5`) also contributed significantly, while pure calendar features (e.g., `day_of_week`) ranked lower.

### ARIMA Forecasting
- The ARIMA(5, 1, 0) model was fit on 80% of the daily series.
- Performance was evaluated via **RMSE** on the held-out test set.
- Predicted values were plotted against actuals, showing the model's ability to capture directional trends, though short-term volatility remained difficult to predict — a known limitation of ARIMA on financial data.

### Seasonal Patterns
- **Monthly averages** revealed periods of relatively higher or lower average prices across the year.
- **Daily turnover analysis** showed that trading activity varies across days of the week, suggesting behavioral patterns among market participants.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, preprocessing, feature engineering |
| `matplotlib` / `seaborn` | Visualization |
| `statsmodels` | Time series decomposition, ARIMA modeling |
| `scikit-learn` | Random Forest, train-test split, RMSE evaluation |
| `numpy` | Numerical operations |

---

## Project Structure

```
├── Untitled20.ipynb        # Main analysis notebook
├── Stock Price Dataset.zip # Raw dataset
└── README.md               # This file
```

---

## Limitations & Future Work

- **ARIMA hyperparameters** were set manually; using `auto_arima` (pmdarima) could yield better-tuned `(p, d, q)` values.
- **LSTM or Transformer models** may outperform ARIMA on capturing complex non-linear patterns in stock data.
- **External signals** (sentiment, macroeconomic indicators, sector data) are not included and could meaningfully improve forecasts.
- The dataset covers a single stock/instrument; extending to a portfolio would provide broader validation.
