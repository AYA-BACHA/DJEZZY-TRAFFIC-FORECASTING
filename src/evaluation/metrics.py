"""
src/evaluation/metrics.py
Standard metrics for time-series forecasting in telecom networks:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error, zero-guarded)
  - sMAPE (Symmetric Mean Absolute Percentage Error)
  - WAPE (Weighted Absolute Percentage Error)
"""

from typing import Dict
import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "",
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, MAPE, sMAPE, and WAPE between ground truth and predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Model predicted values.
    name : str
        Identifier name for the model/baseline.
    eps : float
        Numerical epsilon to avoid division by zero.
        
    Returns
    -------
    dict with keys: model, MAE, RMSE, MAPE, sMAPE, WAPE
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(y_t) == 0:
        return {
            "model": name,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "sMAPE": np.nan,
            "WAPE": np.nan,
        }

    # MAE
    mae = float(np.mean(np.abs(y_t - y_p)))

    # RMSE
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))

    # MAPE (zero-guarded: only over observations with meaningful magnitude)
    mask = np.abs(y_t) > 0.01
    if np.sum(mask) > 0:
        mape = float(np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100.0)
    else:
        mape = 0.0

    # sMAPE: 200 * |y - y_hat| / (|y| + |y_hat| + eps)
    denom = np.abs(y_t) + np.abs(y_p) + eps
    smape = float(np.mean(200.0 * np.abs(y_t - y_p) / denom))

    # WAPE: sum(|y - y_hat|) / sum(|y|) * 100
    sum_true = float(np.sum(np.abs(y_t)))
    if sum_true > eps:
        wape = float(np.sum(np.abs(y_t - y_p)) / sum_true * 100.0)
    else:
        wape = 0.0

    return {
        "model": name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 2),
        "sMAPE": round(smape, 2),
        "WAPE": round(wape, 2),
    }
