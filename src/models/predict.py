"""
src/models/predict.py
Official multi-horizon forecast generation with zero temporal leakage.

Official Forecast Origin:
  2026-01-01 00:00:00 (Africa/Algiers)

Horizons:
  - 24-hour: 2026-01-01 00:00 to 2026-01-01 23:00 (1,872 cell-hours for 78 cells)
  - 7-day:   2026-01-01 00:00 to 2026-01-07 23:00 (13,104 cell-hours for 78 cells)

Uses authentic multi-target recursive forecasting:
  - t-1 and other lags are never peeked from the future.
  - Previous predictions are fed back into lag and rolling statistics for both DL and PRB.

Usage:
    python -m src.models.predict
"""

import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from src.features.build_features import (
    add_algerian_calendar_features,
    add_metadata_features,
    add_temporal_features,
)
from src.models.uncertainty import (
    apply_prediction_intervals,
    load_calibration_intervals,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(model_path: str) -> dict:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def generate_forecast_timestamps(
    origin: str, horizon_hours: int, tz: str
) -> pd.DatetimeIndex:
    """
    Generate hourly forecast timestamps starting exactly at origin.
    24h: origin 00:00 to 23:00 (24 periods)
    7d:  origin 00:00 to day 7 23:00 (168 periods)
    """
    start = pd.Timestamp(origin).tz_localize(tz)
    return pd.date_range(start=start, periods=horizon_hours, freq="h")


def run_joint_cell_forecast(
    cell_id: str,
    cell_history: pd.DataFrame,
    ts_calendar_df: pd.DataFrame,
    forecast_ts: pd.DatetimeIndex,
    model_dl,
    fcols_dl: List[str],
    model_prb,
    fcols_prb: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate multi-step recursive predictions for both DL and PRB simultaneously.
    """
    history_dl = list(cell_history["dl_traffic_volume_gb"].dropna().values)
    history_prb = list(cell_history["prb_utilization_pct"].dropna().values)

    # Base static features from cell
    last_row = cell_history.iloc[-1].to_dict()
    static_dict = {
        k: v for k, v in last_row.items()
        if "lag" not in k and "roll" not in k and "same_hour" not in k and k not in ["timestamp", "predicted"]
    }

    records_dl = []
    records_prb = []

    for h, ts in enumerate(forecast_ts):
        row = dict(static_dict)
        row["cell_id"] = cell_id
        row["timestamp"] = ts

        # Add precomputed calendar features for this step
        cal_row = ts_calendar_df.iloc[h].to_dict()
        row.update(cal_row)

        # 1. DL Lags & Rolling
        lag1_dl = history_dl[-1] if len(history_dl) >= 1 else 0.0
        lag2_dl = history_dl[-2] if len(history_dl) >= 2 else lag1_dl
        lag3_dl = history_dl[-3] if len(history_dl) >= 3 else lag2_dl
        lag6_dl = history_dl[-6] if len(history_dl) >= 6 else lag3_dl
        lag12_dl = history_dl[-12] if len(history_dl) >= 12 else lag6_dl
        lag24_dl = history_dl[-24] if len(history_dl) >= 24 else lag12_dl
        lag48_dl = history_dl[-48] if len(history_dl) >= 48 else lag24_dl
        lag168_dl = history_dl[-168] if len(history_dl) >= 168 else lag24_dl

        roll6_mean_dl = float(np.mean(history_dl[-6:])) if len(history_dl) >= 6 else lag1_dl
        roll6_std_dl = float(np.std(history_dl[-6:])) if len(history_dl) >= 6 else 0.0
        roll24_mean_dl = float(np.mean(history_dl[-24:])) if len(history_dl) >= 24 else lag1_dl
        roll24_std_dl = float(np.std(history_dl[-24:])) if len(history_dl) >= 24 else 0.0
        roll168_mean_dl = float(np.mean(history_dl[-168:])) if len(history_dl) >= 168 else lag1_dl
        roll168_std_dl = float(np.std(history_dl[-168:])) if len(history_dl) >= 168 else 0.0

        row["dl_traffic_volume_gb_lag1h"] = lag1_dl
        row["dl_traffic_volume_gb_lag2h"] = lag2_dl
        row["dl_traffic_volume_gb_lag3h"] = lag3_dl
        row["dl_traffic_volume_gb_lag6h"] = lag6_dl
        row["dl_traffic_volume_gb_lag12h"] = lag12_dl
        row["dl_traffic_volume_gb_lag24h"] = lag24_dl
        row["dl_traffic_volume_gb_lag48h"] = lag48_dl
        row["dl_traffic_volume_gb_lag168h"] = lag168_dl
        row["dl_traffic_volume_gb_roll_mean_6h"] = roll6_mean_dl
        row["dl_traffic_volume_gb_roll_std_6h"] = roll6_std_dl
        row["dl_traffic_volume_gb_roll_mean_24h"] = roll24_mean_dl
        row["dl_traffic_volume_gb_roll_std_24h"] = roll24_std_dl
        row["dl_traffic_volume_gb_roll_mean_168h"] = roll168_mean_dl
        row["dl_traffic_volume_gb_roll_std_168h"] = roll168_std_dl
        row["dl_traffic_volume_gb_same_hour_1d"] = lag24_dl
        row["dl_traffic_volume_gb_same_hour_1w"] = lag168_dl

        # 2. PRB Lags & Rolling
        lag1_prb = history_prb[-1] if len(history_prb) >= 1 else 0.0
        lag2_prb = history_prb[-2] if len(history_prb) >= 2 else lag1_prb
        lag3_prb = history_prb[-3] if len(history_prb) >= 3 else lag2_prb
        lag6_prb = history_prb[-6] if len(history_prb) >= 6 else lag3_prb
        lag12_prb = history_prb[-12] if len(history_prb) >= 12 else lag6_prb
        lag24_prb = history_prb[-24] if len(history_prb) >= 24 else lag12_prb
        lag48_prb = history_prb[-48] if len(history_prb) >= 48 else lag24_prb
        lag168_prb = history_prb[-168] if len(history_prb) >= 168 else lag24_prb

        roll6_mean_prb = float(np.mean(history_prb[-6:])) if len(history_prb) >= 6 else lag1_prb
        roll6_std_prb = float(np.std(history_prb[-6:])) if len(history_prb) >= 6 else 0.0
        roll24_mean_prb = float(np.mean(history_prb[-24:])) if len(history_prb) >= 24 else lag1_prb
        roll24_std_prb = float(np.std(history_prb[-24:])) if len(history_prb) >= 24 else 0.0
        roll168_mean_prb = float(np.mean(history_prb[-168:])) if len(history_prb) >= 168 else lag1_prb
        roll168_std_prb = float(np.std(history_prb[-168:])) if len(history_prb) >= 168 else 0.0

        row["prb_utilization_pct_lag1h"] = lag1_prb
        row["prb_utilization_pct_lag2h"] = lag2_prb
        row["prb_utilization_pct_lag3h"] = lag3_prb
        row["prb_utilization_pct_lag6h"] = lag6_prb
        row["prb_utilization_pct_lag12h"] = lag12_prb
        row["prb_utilization_pct_lag24h"] = lag24_prb
        row["prb_utilization_pct_lag48h"] = lag48_prb
        row["prb_utilization_pct_lag168h"] = lag168_prb
        row["prb_utilization_pct_roll_mean_6h"] = roll6_mean_prb
        row["prb_utilization_pct_roll_std_6h"] = roll6_std_prb
        row["prb_utilization_pct_roll_mean_24h"] = roll24_mean_prb
        row["prb_utilization_pct_roll_std_24h"] = roll24_std_prb
        row["prb_utilization_pct_roll_mean_168h"] = roll168_mean_prb
        row["prb_utilization_pct_roll_std_168h"] = roll168_std_prb
        row["prb_utilization_pct_same_hour_1d"] = lag24_prb
        row["prb_utilization_pct_same_hour_1w"] = lag168_prb

        # 3. Predict DL
        x_dl = np.array([row.get(f, 0.0) for f in fcols_dl], dtype=np.float64).reshape(1, -1)
        pred_dl = float(model_dl.predict(np.nan_to_num(x_dl, nan=0.0))[0])
        pred_dl = max(0.0, pred_dl)

        # 4. Predict PRB
        x_prb = np.array([row.get(f, 0.0) for f in fcols_prb], dtype=np.float64).reshape(1, -1)
        pred_prb = float(model_prb.predict(np.nan_to_num(x_prb, nan=0.0))[0])
        pred_prb = float(np.clip(pred_prb, 0.0, 100.0))

        # Feed predictions back into recursive histories
        history_dl.append(pred_dl)
        history_prb.append(pred_prb)

        # Format output records
        r_dl = {
            "cell_id": cell_id,
            "timestamp": ts,
            "site_id": row.get("site_id", "N/A"),
            "wilaya_name": row.get("wilaya_name", "N/A"),
            "technology": row.get("technology", "N/A"),
            "area_type": row.get("area_type", "N/A"),
            "target": "dl_traffic_volume_gb",
            "predicted": pred_dl,
        }
        r_prb = {
            "cell_id": cell_id,
            "timestamp": ts,
            "site_id": row.get("site_id", "N/A"),
            "wilaya_name": row.get("wilaya_name", "N/A"),
            "technology": row.get("technology", "N/A"),
            "area_type": row.get("area_type", "N/A"),
            "target": "prb_utilization_pct",
            "predicted": pred_prb,
        }
        records_dl.append(r_dl)
        records_prb.append(r_prb)

    return pd.DataFrame(records_dl), pd.DataFrame(records_prb)


def run_prediction_pipeline(config_path: str = "config/config.yaml") -> dict:
    cfg = load_config(config_path)
    feat_path = cfg["paths"]["features_data"]
    models_dir = Path(cfg["paths"]["models_dir"])
    forecasts_dir = Path(cfg["paths"]["forecasts_dir"])
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Djezzy Traffic Forecasting — Official Prediction Pipeline")
    log.info("=" * 60)
    log.info("Loading feature dataset from %s", feat_path)
    df = pd.read_parquet(feat_path, engine="pyarrow")
    log.info("Loaded shape: %s", df.shape)

    origin = cfg["forecast"]["origin"]
    tz = cfg["forecast"]["origin_tz"]
    horizons = {"24h": 24, "7d": 168}

    # Load DL Model (preference: LightGBM > XGBoost > Ridge)
    bundle_dl = None
    for mname in ["lgb", "xgb", "rf", "ridge"]:
        p = models_dir / f"{mname}_dl_traffic_volume_gb.pkl"
        if p.exists():
            bundle_dl = load_model(str(p))
            log.info("Loaded DL model: %s", p.name)
            break

    # Load PRB Model
    bundle_prb = None
    for mname in ["lgb", "xgb", "rf", "ridge"]:
        p = models_dir / f"{mname}_prb_utilization_pct.pkl"
        if p.exists():
            bundle_prb = load_model(str(p))
            log.info("Loaded PRB model: %s", p.name)
            break

    if not bundle_dl or not bundle_prb:
        log.error("Could not find trained models in %s", models_dir)
        return {}

    intervals_cfg = load_calibration_intervals(models_dir)

    cell_ids = sorted(df["cell_id"].unique())
    log.info("Generating forecasts across %d cells starting from %s (%s)", len(cell_ids), origin, tz)

    results = {"dl_traffic_volume_gb": {}, "prb_utilization_pct": {}}

    for h_name, h_hours in horizons.items():
        t0 = time.time()
        log.info("\n--- Generating %s horizon (%d hours) ---", h_name, h_hours)
        forecast_ts = generate_forecast_timestamps(origin, h_hours, tz)

        # Pre-build calendar features for the horizon once
        ts_df = pd.DataFrame({"timestamp": forecast_ts})
        ts_df = add_temporal_features(ts_df)
        ts_df = add_algerian_calendar_features(ts_df, cfg)
        cal_cols_df = ts_df.drop(columns=["timestamp"])

        dl_frames = []
        prb_frames = []

        for cid in cell_ids:
            c_hist = df[df["cell_id"] == cid].sort_values("timestamp")
            f_dl, f_prb = run_joint_cell_forecast(
                cid, c_hist, cal_cols_df, forecast_ts,
                bundle_dl["model"], bundle_dl["feature_cols"],
                bundle_prb["model"], bundle_prb["feature_cols"],
            )
            dl_frames.append(f_dl)
            prb_frames.append(f_prb)

        df_dl_all = pd.concat(dl_frames, ignore_index=True)
        df_prb_all = pd.concat(prb_frames, ignore_index=True)

        df_dl_all["horizon"] = h_name
        df_prb_all["horizon"] = h_name

        # Apply 75% empirical prediction intervals calibrated on pre-holdout validation
        df_dl_all = apply_prediction_intervals(df_dl_all, intervals_cfg, "dl_traffic_volume_gb", h_name)
        df_prb_all = apply_prediction_intervals(df_prb_all, intervals_cfg, "prb_utilization_pct", h_name)

        col_order = [
            "timestamp", "cell_id", "site_id", "wilaya_name", "technology",
            "area_type", "target", "predicted", "lower_75", "upper_75", "horizon"
        ]
        df_dl_all = df_dl_all[[c for c in col_order if c in df_dl_all.columns]]
        df_prb_all = df_prb_all[[c for c in col_order if c in df_prb_all.columns]]

        # Save DL files
        alias = "short" if h_name == "24h" else "long"
        for suffix in [h_name, alias]:
            p_dl = forecasts_dir / f"forecast_dl_traffic_volume_gb_{suffix}.parquet"
            c_dl = forecasts_dir / f"forecast_dl_traffic_volume_gb_{suffix}.csv"
            df_dl_all.to_parquet(p_dl, index=False, engine="pyarrow")
            df_dl_all.to_csv(c_dl, index=False)

            p_prb = forecasts_dir / f"forecast_prb_utilization_pct_{suffix}.parquet"
            c_prb = forecasts_dir / f"forecast_prb_utilization_pct_{suffix}.csv"
            df_prb_all.to_parquet(p_prb, index=False, engine="pyarrow")
            df_prb_all.to_csv(c_prb, index=False)

        mean_w_dl = float((df_dl_all["upper_75"] - df_dl_all["lower_75"]).mean())
        mean_w_prb = float((df_prb_all["upper_75"] - df_prb_all["lower_75"]).mean())
        log.info("Saved %d DL rows (mean 75%% width: %.2f GB) and %d PRB rows (mean 75%% width: %.2f%%) in %.2fs",
                 len(df_dl_all), mean_w_dl, len(df_prb_all), mean_w_prb, time.time() - t0)

        results["dl_traffic_volume_gb"][h_name] = {
            "n_rows": len(df_dl_all),
            "n_cells": len(cell_ids),
            "forecast_start": str(forecast_ts[0]),
            "forecast_end": str(forecast_ts[-1]),
            "mean_predicted": float(df_dl_all["predicted"].mean()),
            "max_predicted": float(df_dl_all["predicted"].max()),
            "min_predicted": float(df_dl_all["predicted"].min()),
            "mean_interval_width_75": mean_w_dl,
        }
        results["prb_utilization_pct"][h_name] = {
            "n_rows": len(df_prb_all),
            "n_cells": len(cell_ids),
            "forecast_start": str(forecast_ts[0]),
            "forecast_end": str(forecast_ts[-1]),
            "mean_predicted": float(df_prb_all["predicted"].mean()),
            "max_predicted": float(df_prb_all["predicted"].max()),
            "min_predicted": float(df_prb_all["predicted"].min()),
            "mean_interval_width_75": mean_w_prb,
        }

    summary_path = forecasts_dir / "forecast_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Forecast summary saved to %s", summary_path)
    return results


if __name__ == "__main__":
    run_prediction_pipeline()
