# Sales Forecasting System

A rolling sales forecaster that beats a naive "same as last week" baseline using lag features,
calendar features, and XGBoost -- evaluated honestly with a metric that doesn't break on sparse
retail data.

## Problem
A retail business needs a forecast that accounts for trend, seasonality, and day-of-week
effects instead of just assuming next period looks like last period.

## What It Does
- Aggregates transaction-level sales to a daily series per category/region
- Engineers lag (7/14-day), rolling-mean, and calendar (day-of-week, month, weekend) features
- Splits chronologically, not randomly, so the model is never trained on future data
- Trains an XGBoost regressor and compares it against a naive lag-7 baseline
- Evaluates with MAE and **WAPE** (weighted absolute percentage error) instead of standard MAPE,
  since MAPE breaks down (returns nonsensical values well over 100%) whenever actual sales for a
  day are zero or near-zero, which is common once sales are sliced down to daily granularity

## Real Results (real dataset, 9,994 Superstore order-line rows)
- **Model MAE: 520.45** vs naive baseline MAE: 698.72
- **Model WAPE: 115.0%** vs naive baseline WAPE: 154.5%
- **The model beats the naive baseline by 25.5%** -- consistent across both metrics, which is a
  good sign the improvement is real rather than a metric artifact

## Notes
WAPE above 100% means the model's total error still exceeds total actual sales in absolute
terms -- this is a genuinely noisy dataset to forecast at daily granularity (~10K order lines
spread across many category/region/day combinations). The honest takeaway isn't "production-
ready forecast," it's "a real, reproducible 25% improvement over a naive baseline using proper
time-based validation." Aggregating to weekly instead of daily would likely tighten this
further, and is the natural next iteration.

## Tech Stack
Python, Pandas, XGBoost, Scikit-learn, Matplotlib

## How to Run
Open in Google Colab, run all cells. Dataset auto-downloads via `kagglehub` with a synthetic
fallback if it ever fails.
