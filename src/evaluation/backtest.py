"""
src/evaluation/backtest.py
High-performance vectorized chronological walk-forward backtesting on untouched holdout data (Nov-Dec 2025).

Evaluates:
  - Baselines: Naive, Seasonal Naive Yesterday, Seasonal Naive Last Week, Moving Average
  - ML Models: Ridge, Random Forest, LightGBM, XGBoost
  
Computes across 24h and 7d horizons:
  - MAE
  - RMSE
  - MAPE
  - sMAPE
  - WAPE

Ensures strict zero-leakage recursive forecasting where t-1 is not peeked from the future.
Outputs: reports/metrics/metrics_summary.csv
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

from src.evaluation.metrics import compute_metrics
from src.models.baselines import generate_all_baselines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_bundle(model_path: Path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def run_vectorized_simulation(
    cells: List[str],
    history_by_cell: Dict[str, List[float]],
    target_frames_by_cell: Dict[str, pd.DataFrame],
    model,
    feature_cols: List[str],
    target: str,
    horizon_hours: int,
) -> Dict[str, np.ndarray]:
    """
    Simulate multi-step recursive forecasting across all cells in vectorized batches per step.
    Zero leakage: future target ground truth is never accessed during future steps.
    """
    # Clone histories to avoid mutating caller
    histories = {cid: list(history_by_cell[cid]) for cid in cells}
    cell_preds = {cid: [] for cid in cells}

    for h in range(horizon_hours):
        batch_rows = []
        for cid in cells:
            row_dict = target_frames_by_cell[cid].iloc[h].to_dict()
            h_vals = histories[cid]

            lag1 = h_vals[-1] if len(h_vals) >= 1 else 0.0
            lag2 = h_vals[-2] if len(h_vals) >= 2 else lag1
            lag3 = h_vals[-3] if len(h_vals) >= 3 else lag2
            lag6 = h_vals[-6] if len(h_vals) >= 6 else lag3
            lag12 = h_vals[-12] if len(h_vals) >= 12 else lag6
            lag24 = h_vals[-24] if len(h_vals) >= 24 else lag12
            lag48 = h_vals[-48] if len(h_vals) >= 48 else lag24
            lag168 = h_vals[-168] if len(h_vals) >= 168 else lag24

            roll6_mean = float(np.mean(h_vals[-6:])) if len(h_vals) >= 6 else lag1
            roll6_std = float(np.std(h_vals[-6:])) if len(h_vals) >= 6 else 0.0
            roll24_mean = float(np.mean(h_vals[-24:])) if len(h_vals) >= 24 else lag1
            roll24_std = float(np.std(h_vals[-24:])) if len(h_vals) >= 24 else 0.0
            roll168_mean = float(np.mean(h_vals[-168:])) if len(h_vals) >= 168 else lag1
            roll168_std = float(np.std(h_vals[-168:])) if len(h_vals) >= 168 else 0.0

            row_dict[f"{target}_lag1h"] = lag1
            row_dict[f"{target}_lag2h"] = lag2
            row_dict[f"{target}_lag3h"] = lag3
            row_dict[f"{target}_lag6h"] = lag6
            row_dict[f"{target}_lag12h"] = lag12
            row_dict[f"{target}_lag24h"] = lag24
            row_dict[f"{target}_lag48h"] = lag48
            row_dict[f"{target}_lag168h"] = lag168

            row_dict[f"{target}_roll_mean_6h"] = roll6_mean
            row_dict[f"{target}_roll_std_6h"] = roll6_std
            row_dict[f"{target}_roll_mean_24h"] = roll24_mean
            row_dict[f"{target}_roll_std_24h"] = roll24_std
            row_dict[f"{target}_roll_mean_168h"] = roll168_mean
            row_dict[f"{target}_roll_std_168h"] = roll168_std
            row_dict[f"{target}_same_hour_1d"] = lag24
            row_dict[f"{target}_same_hour_1w"] = lag168

            feat_vector = [row_dict.get(c, 0.0) for c in feature_cols]
            batch_rows.append(feat_vector)

        X_batch = np.nan_to_num(np.array(batch_rows, dtype=np.float64), nan=0.0)
        y_batch = model.predict(X_batch)

        for i, cid in enumerate(cells):
            pred_val = float(y_batch[i])
            if "dl" in target:
                pred_val = max(0.0, pred_val)
            else:
                pred_val = float(np.clip(pred_val, 0.0, 100.0))
            cell_preds[cid].append(pred_val)
            histories[cid].append(pred_val)

    return {cid: np.array(cell_preds[cid], dtype=np.float64) for cid in cells}


def run_holdout_backtest(
    config_path: str = "config/config.yaml",
    test_origins: List[str] = None,
) -> pd.DataFrame:
    """
    Run full holdout backtest across baselines and trained models in fast vectorized mode.
    """
    cfg = load_config(config_path)
    feat_path = cfg["paths"]["features_data"]
    models_dir = Path(cfg["paths"]["models_dir"])
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading feature dataset for backtest from %s", feat_path)
    df = pd.read_parquet(feat_path, engine="pyarrow")
    
    if test_origins is None:
        test_origins = ["2025-11-03 00:00:00", "2025-12-01 00:00:00"]

    tz = cfg["forecast"]["origin_tz"]
    horizons = {"24h": 24, "7d": 168}
    targets = cfg["forecast"]["targets"]

    all_metric_rows = []

    for target in targets:
        log.info("\n" + "=" * 50)
        log.info("Backtesting target: %s", target)
        log.info("=" * 50)
        
        # Load available models
        loaded_models = {}
        for mname in ["ridge", "rf", "lgb", "xgb"]:
            mfile = models_dir / f"{mname}_{target}.pkl"
            if mfile.exists():
                bundle = load_model_bundle(mfile)
                loaded_models[mname] = (bundle["model"], bundle["feature_cols"])
                log.info("  Loaded model: %s", mname)

        for h_name, h_hours in horizons.items():
            log.info("\n--- Evaluating Horizon: %s (%d hours) ---", h_name, h_hours)
            
            ground_truth_all = []
            predictions_by_model = {
                "naive": [],
                "seasonal_naive_yesterday": [],
                "seasonal_naive_last_week": [],
                "moving_average_24h": [],
            }
            for mname in loaded_models.keys():
                predictions_by_model[mname] = []

            for origin_str in test_origins:
                origin_ts = pd.Timestamp(origin_str).tz_localize(tz)
                end_ts = origin_ts + pd.Timedelta(hours=h_hours)

                df_pre = df[df["timestamp"] < origin_ts]
                df_post = df[(df["timestamp"] >= origin_ts) & (df["timestamp"] < end_ts)]

                cell_ids = sorted(df["cell_id"].unique())
                valid_cells = []
                histories_for_cells = {}
                target_frames_for_cells = {}

                for cid in cell_ids:
                    c_pre = df_pre[df_pre["cell_id"] == cid].sort_values("timestamp")
                    c_post = df_post[df_post["cell_id"] == cid].sort_values("timestamp")

                    if len(c_post) >= h_hours and len(c_pre) >= 168:
                        valid_cells.append(cid)
                        hist_series = c_pre[target].dropna()
                        histories_for_cells[cid] = list(hist_series.values)
                        target_frames_for_cells[cid] = c_post.iloc[:h_hours].copy()

                        y_true = c_post[target].values[:h_hours]
                        ground_truth_all.extend(y_true)

                        # Compute baselines
                        baselines = generate_all_baselines(hist_series, h_hours)
                        for bname, b_preds in baselines.items():
                            predictions_by_model[bname].extend(b_preds)

                # Vectorized batch simulation for each ML model
                for mname, (model, fcols) in loaded_models.items():
                    sim_results = run_vectorized_simulation(
                        valid_cells, histories_for_cells, target_frames_for_cells,
                        model, fcols, target, h_hours
                    )
                    for cid in valid_cells:
                        predictions_by_model[mname].extend(sim_results[cid])

            # Compute and log metrics
            y_true_arr = np.array(ground_truth_all, dtype=np.float64)
            for mname, preds_list in predictions_by_model.items():
                if len(preds_list) == len(y_true_arr) and len(y_true_arr) > 0:
                    y_pred_arr = np.array(preds_list, dtype=np.float64)
                    m = compute_metrics(y_true_arr, y_pred_arr, name=mname)
                    m_row = {
                        "target": target,
                        "horizon": h_name,
                        **m
                    }
                    all_metric_rows.append(m_row)
                    log.info("  [%s - %s] %-25s: MAE=%7.4f, RMSE=%7.4f, MAPE=%6.2f%%, sMAPE=%6.2f%%, WAPE=%6.2f%%",
                             target, h_name, mname, m["MAE"], m["RMSE"], m["MAPE"], m["sMAPE"], m["WAPE"])

    metrics_df = pd.DataFrame(all_metric_rows)
    out_csv = metrics_dir / "metrics_summary.csv"
    metrics_df.to_csv(out_csv, index=False)
    log.info("\nSaved complete metrics summary table to %s", out_csv)
    return metrics_df


if __name__ == "__main__":
    run_holdout_backtest()
