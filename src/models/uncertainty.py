"""
src/models/uncertainty.py
Horizon-aware empirical conformal prediction interval calibration and evaluation for Djezzy traffic forecasting.

Calibrates a central 75% prediction interval:
  - Lower quantile: 12.5% (alpha/2 with alpha = 0.25)
  - Upper quantile: 87.5% (1 - alpha/2)

Strict Leakage Protection:
  - Calibration residuals are estimated strictly from pre-holdout validation origins (before 2025-11-01).
  - The final holdout (Nov-Dec 2025) is NEVER used for calibration.
  - Physical boundary clamping (DL >= 0, PRB in [0, 100]) is enforced.
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from src.evaluation.backtest import run_vectorized_simulation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calibrate_prediction_intervals(
    config_path: str = "config/config.yaml",
    val_origins: List[str] = None,
    alpha: float = 0.25,
) -> Dict:
    """
    Calibrate horizon-aware empirical prediction intervals on pre-holdout validation origins.
    For alpha = 0.25 (75% interval), computes 12.5% and 87.5% quantiles of residuals per horizon step.
    """
    cfg = load_config(config_path)
    feat_path = cfg["paths"]["features_data"]
    models_dir = Path(cfg["paths"]["models_dir"])
    tz = cfg["forecast"]["origin_tz"]
    targets = cfg["forecast"]["targets"]
    horizons = {"24h": 24, "7d": 168}

    if val_origins is None:
        # Strictly pre-holdout validation origins (prior to 2025-11-01)
        val_origins = ["2025-07-01 00:00:00", "2025-08-01 00:00:00", "2025-09-01 00:00:00", "2025-10-01 00:00:00"]

    log.info("Loading feature dataset for uncertainty calibration from %s", feat_path)
    df = pd.read_parquet(feat_path, engine="pyarrow")
    cells = sorted(df["cell_id"].unique())

    # Pre-split cell histories for fast filtering
    cells_pre_dict = {}
    for orig_str in val_origins:
        orig_ts = pd.Timestamp(orig_str, tz=tz)
        df_pre = df[df["timestamp"] < orig_ts]
        cells_pre_dict[orig_str] = {cid: df_pre[df_pre["cell_id"] == cid].sort_values("timestamp") for cid in cells}

    calibration_results = {
        "alpha": alpha,
        "nominal_coverage_pct": (1.0 - alpha) * 100.0,
        "val_origins": val_origins,
        "quantiles": {},
    }

    q_lower = alpha / 2.0         # 0.125 (12.5%)
    q_upper = 1.0 - (alpha / 2.0) # 0.875 (87.5%)

    for target in targets:
        # Load primary model
        model_file = models_dir / f"lgb_{target}.pkl"
        if not model_file.exists():
            model_file = models_dir / f"rf_{target}.pkl"
        with open(model_file, "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]
        fcols = bundle["feature_cols"]

        calibration_results["quantiles"][target] = {}

        for h_name, h_hours in horizons.items():
            log.info("Calibrating %s for horizon %s (%d steps)...", target, h_name, h_hours)
            step_residuals = [[] for _ in range(h_hours)]

            for orig_str in val_origins:
                orig_ts = pd.Timestamp(orig_str, tz=tz)
                end_ts = orig_ts + pd.Timedelta(hours=h_hours)
                df_post = df[(df["timestamp"] >= orig_ts) & (df["timestamp"] < end_ts)]

                valid_cells = []
                histories = {}
                target_frames = {}

                for cid in cells:
                    c_pre = cells_pre_dict[orig_str][cid]
                    c_post = df_post[df_post["cell_id"] == cid].sort_values("timestamp")
                    if len(c_post) >= h_hours and len(c_pre) >= 168:
                        valid_cells.append(cid)
                        histories[cid] = list(c_pre[target].dropna().values)
                        target_frames[cid] = c_post.iloc[:h_hours].copy()

                preds_dict = run_vectorized_simulation(
                    valid_cells, histories, target_frames, model, fcols, target, h_hours
                )

                for cid in valid_cells:
                    y_true = target_frames[cid][target].values[:h_hours]
                    y_pred = preds_dict[cid]
                    for s in range(h_hours):
                        step_residuals[s].append(float(y_true[s] - y_pred[s]))

            # Compute horizon-aware conformal 75% absolute residual quantile
            q_abs_list = [float(np.quantile(np.abs(step_residuals[s]), 1.0 - alpha)) for s in range(h_hours)]

            calibration_results["quantiles"][target][h_name] = {
                "horizon_hours": h_hours,
                "q_abs_offset": q_abs_list,
                "mean_width": float(np.mean(2.0 * np.array(q_abs_list))),
            }
            log.info("  %s %s: Step 0 margin=±%.2f, Final step margin=±%.2f (mean width=%.2f)",
                     target, h_name, q_abs_list[0], q_abs_list[-1], 2.0 * float(np.mean(q_abs_list)))

    out_file = models_dir / "calibration_intervals.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(calibration_results, f, indent=2)
    log.info("Saved calibration intervals to %s", out_file)
    return calibration_results


def load_calibration_intervals(models_dir: Path = Path("models")) -> Dict:
    path = models_dir / "calibration_intervals.json"
    if not path.exists():
        log.warning("Calibration intervals file not found at %s. Running calibration...", path)
        return calibrate_prediction_intervals()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_prediction_intervals(
    df_forecast: pd.DataFrame,
    calibration_config: Dict,
    target: str,
    horizon_key: str,
) -> pd.DataFrame:
    """
    Apply calibrated prediction intervals to a forecast dataframe and enforce physical bounds.
    """
    df_out = df_forecast.copy()
    h_cfg = calibration_config.get("quantiles", {}).get(target, {}).get(horizon_key, {})
    if not h_cfg:
        df_out["predicted"] = df_out["predicted"].round(4)
        df_out["lower_75"] = df_out["predicted"]
        df_out["upper_75"] = df_out["predicted"]
        return df_out

    q_abs_offsets = np.array(h_cfg["q_abs_offset"], dtype=np.float64)
    h_hours = len(q_abs_offsets)

    preds_vals = []
    lower_vals = []
    upper_vals = []

    # Map by cell and step
    for _, group in df_out.groupby("cell_id", sort=False):
        n_steps = len(group)
        for s in range(n_steps):
            pred = float(group.iloc[s]["predicted"])
            margin = q_abs_offsets[s % h_hours]

            raw_low = pred - margin
            raw_high = pred + margin

            # Physical bounding
            if "dl" in target.lower():
                low = max(0.0, raw_low)
                high = max(pred, max(0.0, raw_high))
            else:
                low = float(np.clip(raw_low, 0.0, 100.0))
                high = float(np.clip(raw_high, 0.0, 100.0))

            pred_r = round(pred, 4)
            low_r = min(pred_r, round(low, 4))
            high_r = max(pred_r, round(high, 4))

            preds_vals.append(pred_r)
            lower_vals.append(low_r)
            upper_vals.append(high_r)

    df_out["predicted"] = preds_vals
    df_out["lower_75"] = lower_vals
    df_out["upper_75"] = upper_vals
    return df_out


def compute_empirical_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Calculate empirical coverage percentage."""
    inside = (np.asarray(y_true) >= np.asarray(lower)) & (np.asarray(y_true) <= np.asarray(upper))
    return float(np.mean(inside) * 100.0)


def compute_mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Calculate average interval width."""
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def compute_winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.25) -> float:
    """Calculate mean Winkler interval score."""
    y_t = np.asarray(y_true, dtype=np.float64)
    l_b = np.asarray(lower, dtype=np.float64)
    u_b = np.asarray(upper, dtype=np.float64)
    width = u_b - l_b
    under = np.maximum(0.0, l_b - y_t) * (2.0 / alpha)
    over = np.maximum(0.0, y_t - u_b) * (2.0 / alpha)
    return float(np.mean(width + under + over))


def evaluate_prediction_intervals(
    y_true: np.ndarray,
    lower_75: np.ndarray,
    upper_75: np.ndarray,
    alpha: float = 0.25,
) -> Dict[str, float]:
    """
    Evaluate interval empirical coverage, average width, and Winkler score.
    """
    coverage_pct = compute_empirical_coverage(y_true, lower_75, upper_75)
    mean_width = compute_mean_interval_width(lower_75, upper_75)
    winkler_score = compute_winkler_score(y_true, lower_75, upper_75, alpha=alpha)

    return {
        "nominal_coverage_pct": (1.0 - alpha) * 100.0,
        "empirical_coverage_pct": round(coverage_pct, 2),
        "mean_interval_width": round(mean_width, 4),
        "winkler_score": round(winkler_score, 4),
    }


if __name__ == "__main__":
    calibrate_prediction_intervals()
