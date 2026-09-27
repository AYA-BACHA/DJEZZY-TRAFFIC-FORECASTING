"""
src/models/baselines.py
Standard time-series baselines required for telecom forecasting benchmarks:
  1. Naive: Last observed value (y_{T-1})
  2. Seasonal Naive Yesterday: Value at same hour yesterday (y_{t-24})
  3. Seasonal Naive Last Week: Value at same hour last week (y_{t-168})
  4. Moving Average: Rolling mean of recent 24 hours prior to origin
"""

from typing import Dict
import numpy as np
import pandas as pd


def predict_naive(history: pd.Series, horizon_hours: int) -> np.ndarray:
    """Repeat the last available historical observation for all horizon hours."""
    last_val = history.dropna().iloc[-1] if len(history.dropna()) > 0 else 0.0
    return np.full(horizon_hours, float(last_val))


def predict_seasonal_naive_yesterday(
    history_hourly: pd.Series,
    horizon_hours: int,
) -> np.ndarray:
    """
    Seasonal naive using same hour yesterday.
    For horizon <= 24, takes history[-24:].
    For horizon > 24, tiles the last 24h pattern recursively.
    """
    clean_hist = history_hourly.dropna()
    if len(clean_hist) < 24:
        last_val = clean_hist.iloc[-1] if len(clean_hist) > 0 else 0.0
        return np.full(horizon_hours, float(last_val))

    pattern_24 = clean_hist.iloc[-24:].values
    repeats = int(np.ceil(horizon_hours / 24))
    tiled = np.tile(pattern_24, repeats)[:horizon_hours]
    return np.asarray(tiled, dtype=np.float64)


def predict_seasonal_naive_last_week(
    history_hourly: pd.Series,
    horizon_hours: int,
) -> np.ndarray:
    """
    Seasonal naive using same hour last week.
    For horizon <= 168, takes history[-168:].
    For horizon > 168, tiles the last 168h (weekly) pattern.
    """
    clean_hist = history_hourly.dropna()
    if len(clean_hist) < 168:
        # Fall back to 24h or last val
        return predict_seasonal_naive_yesterday(history_hourly, horizon_hours)

    pattern_168 = clean_hist.iloc[-168:].values
    repeats = int(np.ceil(horizon_hours / 168))
    tiled = np.tile(pattern_168, repeats)[:horizon_hours]
    return np.asarray(tiled, dtype=np.float64)


def predict_moving_average(
    history: pd.Series,
    horizon_hours: int,
    window: int = 24,
) -> np.ndarray:
    """Predict the rolling mean of the recent `window` hours prior to origin."""
    clean_hist = history.dropna()
    if len(clean_hist) == 0:
        return np.zeros(horizon_hours, dtype=np.float64)
    avg_val = float(clean_hist.iloc[-window:].mean()) if len(clean_hist) >= window else float(clean_hist.mean())
    return np.full(horizon_hours, avg_val)


def generate_all_baselines(
    history: pd.Series,
    horizon_hours: int,
) -> Dict[str, np.ndarray]:
    """Generate predictions from all standard baselines for a given cell time series."""
    return {
        "naive": predict_naive(history, horizon_hours),
        "seasonal_naive_yesterday": predict_seasonal_naive_yesterday(history, horizon_hours),
        "seasonal_naive_last_week": predict_seasonal_naive_last_week(history, horizon_hours),
        "moving_average_24h": predict_moving_average(history, horizon_hours, window=24),
    }
