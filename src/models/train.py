"""
src/models/train.py
Training pipeline for the Djezzy network traffic forecasting models.

Models trained:
  - Ridge regression (baseline)
  - Random Forest
  - LightGBM (primary)
  - XGBoost (gradient boosted trees)

Each model is trained per target variable:
  - dl_traffic_volume_gb  (DL traffic forecasting)
  - prb_utilization_pct   (congestion forecasting)

Uses chronological walk-forward cross-validation.
Evaluates on untouched final holdout period (Nov-Dec 2025).

Run as:  python -m src.models.train
"""

import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logging.warning("LightGBM not available — skipping LGB model")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logging.warning("XGBoost not available — skipping XGB model")

from src.evaluation.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
def get_feature_cols(df: pd.DataFrame, target: str) -> List[str]:
    """Return feature columns to use, strictly excluding contemporaneous targets and metadata."""
    exclude = {
        "timestamp", "timestamp_text", "site_id", "cell_id",
        "enodeb_id", "gnodeb_id", "lac_tac", "mcc", "mnc",
        "technology", "area_type", "wilaya_name",
        "time_segment",
        # exclude contemporaneous KPIs to prevent future leakage
        "dl_traffic_volume_gb", "ul_traffic_volume_gb", "voice_traffic_erlang",
        "rrc_connected_users", "active_users", "prb_utilization_pct",
        "avg_user_throughput_mbps", "latency_ms", "cell_availability_pct",
        "call_drop_rate_pct", "handover_success_rate_pct",
        "congestion_flag", "cum_dl_traffic_gb",
        "outage_flag", "site_total_dl_gb", "site_avg_prb_pct", "site_any_congested",
        "congestion_level",
    }
    exclude.add(target)
    imputed_cols = {c for c in df.columns if c.endswith("_imputed")}
    exclude.update(imputed_cols)

    feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return feat_cols


# ---------------------------------------------------------------------------
# Walk-forward chronological CV splits
# ---------------------------------------------------------------------------
def make_cv_splits(
    df: pd.DataFrame,
    n_folds: int = 3,
    test_months: int = 2,
    holdout_start: str = "2025-11-01",
) -> List[Tuple[pd.Series, pd.Series]]:
    """Return list of (train_mask, val_mask) boolean series for walk-forward CV."""
    holdout_ts = pd.Timestamp(holdout_start).tz_localize("Africa/Algiers")
    train_df = df[df["timestamp"] < holdout_ts]
    ts_max = train_df["timestamp"].max()

    splits = []
    for fold in range(n_folds):
        val_end = ts_max - pd.DateOffset(months=fold * test_months)
        val_start = val_end - pd.DateOffset(months=test_months)
        train_mask = df["timestamp"] < val_start
        val_mask = (df["timestamp"] >= val_start) & (df["timestamp"] < val_end)
        if train_mask.sum() < 1000 or val_mask.sum() < 100:
            log.warning("Fold %d has insufficient data; skipping", fold)
            continue
        splits.append((train_mask, val_mask))
        log.info("  Fold %d: train < %s | val [%s, %s)", fold, val_start, val_start, val_end)

    return splits


# ---------------------------------------------------------------------------
# Model Training Routines
# ---------------------------------------------------------------------------
def train_ridge(X_train: np.ndarray, y_train: np.ndarray, cfg: dict) -> Pipeline:
    alpha = cfg["models"]["ridge"]["alpha"]
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_rf(X_train: np.ndarray, y_train: np.ndarray, cfg: dict) -> Pipeline:
    params = cfg["models"].get("random_forest", {})
    n_est = min(50, params.get("n_estimators", 50))
    max_d = min(12, params.get("max_depth", 12))
    rf = RandomForestRegressor(
        n_estimators=n_est,
        max_depth=max_d,
        min_samples_leaf=params.get("min_samples_leaf", 5),
        max_samples=0.25,
        n_jobs=params.get("n_jobs", -1),
        random_state=cfg.get("random_seed", 42),
    )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", rf),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_lgb(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, cfg: dict):
    if not HAS_LGB:
        return None
    params = cfg["models"]["lightgbm"]
    model = lgb.LGBMRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        num_leaves=params["num_leaves"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        n_jobs=params["n_jobs"],
        random_state=cfg["random_seed"],
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(params.get("early_stopping_rounds", 50), verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    return model


def train_xgb(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, cfg: dict):
    if not HAS_XGB:
        return None
    params = cfg["models"].get("xgboost", {})
    model = xgb.XGBRegressor(
        n_estimators=min(250, params.get("n_estimators", 250)),
        learning_rate=params.get("learning_rate", 0.06),
        max_depth=min(6, params.get("max_depth", 6)),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        tree_method="hist",
        n_jobs=params.get("n_jobs", -1),
        random_state=cfg.get("random_seed", 42),
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------
def run_training_pipeline(config_path: str = "config/config.yaml") -> Dict:
    cfg = load_config(config_path)
    feat_path = cfg["paths"]["features_data"]
    models_dir = Path(cfg["paths"]["models_dir"])
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Djezzy Traffic Forecasting — Training Pipeline")
    log.info("=" * 60)
    log.info("Loading features from %s", feat_path)
    df = pd.read_parquet(feat_path, engine="pyarrow")
    log.info("Loaded shape: %s", df.shape)

    holdout_start = cfg["validation"]["final_holdout_start"]
    n_folds = cfg["validation"]["n_folds"]
    test_months = cfg["validation"]["test_months"]
    targets = cfg["forecast"]["targets"]

    all_results = {}

    for target in targets:
        log.info("\n" + "=" * 50)
        log.info("TARGET: %s", target)
        log.info("=" * 50)

        feat_cols = get_feature_cols(df, target)
        log.info("Using %d features", len(feat_cols))

        mask_valid = df[target].notna()
        df_model = df[mask_valid].copy()

        splits = make_cv_splits(df_model, n_folds, test_months, holdout_start)

        cv_results = []
        for fold_idx, (train_mask, val_mask) in enumerate(splits):
            X_tr = df_model.loc[train_mask, feat_cols].values
            y_tr = df_model.loc[train_mask, target].values
            X_val = df_model.loc[val_mask, feat_cols].values
            y_val = df_model.loc[val_mask, target].values

            log.info("  Fold %d: train=%d, val=%d", fold_idx, len(y_tr), len(y_val))
            fold_metrics = {}

            # Ridge
            ridge = train_ridge(X_tr, y_tr, cfg)
            fold_metrics["ridge"] = compute_metrics(y_val, ridge.predict(X_val), "ridge")

            # LightGBM
            if HAS_LGB:
                lgb_model = train_lgb(X_tr, y_tr, X_val, y_val, cfg)
                if lgb_model:
                    fold_metrics["lightgbm"] = compute_metrics(y_val, lgb_model.predict(X_val), "lightgbm")

            # XGBoost
            if HAS_XGB:
                xgb_model = train_xgb(X_tr, y_tr, X_val, y_val, cfg)
                if xgb_model:
                    fold_metrics["xgboost"] = compute_metrics(y_val, xgb_model.predict(X_val), "xgboost")

            cv_results.append({"fold": fold_idx, "metrics": fold_metrics})
            log.info("  Fold %d metrics: %s", fold_idx, fold_metrics)

        # ----------------------------------------------------------------
        # Final models trained on all pre-holdout data
        # ----------------------------------------------------------------
        log.info("Training FINAL models on all pre-holdout data for target=%s", target)
        holdout_ts = pd.Timestamp(holdout_start).tz_localize("Africa/Algiers")
        pre_holdout_mask = df_model["timestamp"] < holdout_ts
        X_final_tr = df_model.loc[pre_holdout_mask, feat_cols].values
        y_final_tr = df_model.loc[pre_holdout_mask, target].values

        holdout_mask = df_model["timestamp"] >= holdout_ts
        X_holdout = df_model.loc[holdout_mask, feat_cols].values
        y_holdout = df_model.loc[holdout_mask, target].values
        log.info("  Holdout size: %d rows", len(y_holdout))

        holdout_metrics = {}

        # 1. Ridge
        log.info("  Fitting Ridge...")
        ridge_final = train_ridge(X_final_tr, y_final_tr, cfg)
        if len(y_holdout) > 0:
            holdout_metrics["ridge"] = compute_metrics(y_holdout, ridge_final.predict(X_holdout), "ridge")
        ridge_path = models_dir / f"ridge_{target}.pkl"
        with open(ridge_path, "wb") as f:
            pickle.dump({"model": ridge_final, "feature_cols": feat_cols, "target": target}, f)

        # 2. Random Forest
        log.info("  Fitting Random Forest...")
        rf_final = train_rf(X_final_tr, y_final_tr, cfg)
        if len(y_holdout) > 0:
            holdout_metrics["random_forest"] = compute_metrics(y_holdout, rf_final.predict(X_holdout), "random_forest")
        rf_path = models_dir / f"rf_{target}.pkl"
        with open(rf_path, "wb") as f:
            pickle.dump({"model": rf_final, "feature_cols": feat_cols, "target": target}, f)

        # 3. LightGBM
        lgb_final = None
        if HAS_LGB and len(y_holdout) > 0:
            log.info("  Fitting LightGBM...")
            lgb_final = train_lgb(X_final_tr, y_final_tr, X_holdout, y_holdout, cfg)
            if lgb_final:
                holdout_metrics["lightgbm"] = compute_metrics(y_holdout, lgb_final.predict(X_holdout), "lightgbm")
                lgb_path = models_dir / f"lgb_{target}.pkl"
                with open(lgb_path, "wb") as f:
                    pickle.dump({"model": lgb_final, "feature_cols": feat_cols, "target": target}, f)

        # 4. XGBoost
        xgb_final = None
        if HAS_XGB and len(y_holdout) > 0:
            log.info("  Fitting XGBoost...")
            xgb_final = train_xgb(X_final_tr, y_final_tr, X_holdout, y_holdout, cfg)
            if xgb_final:
                holdout_metrics["xgboost"] = compute_metrics(y_holdout, xgb_final.predict(X_holdout), "xgboost")
                xgb_path = models_dir / f"xgb_{target}.pkl"
                with open(xgb_path, "wb") as f:
                    pickle.dump({"model": xgb_final, "feature_cols": feat_cols, "target": target}, f)

        # Save feature importance from LightGBM
        if lgb_final is not None:
            importance_df = pd.DataFrame({
                "feature": feat_cols,
                "importance": lgb_final.feature_importances_,
            }).sort_values("importance", ascending=False)
            importance_path = metrics_dir / f"feature_importance_{target}.csv"
            importance_df.to_csv(importance_path, index=False)
            log.info("  Feature importance saved to %s", importance_path)

        all_results[target] = {
            "cv_results": cv_results,
            "holdout_metrics": holdout_metrics,
            "n_features": len(feat_cols),
            "feature_cols": feat_cols,
        }
        log.info("Holdout metrics for %s: %s", target, holdout_metrics)

    metrics_path = metrics_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("All metrics saved to %s", metrics_path)

    return all_results


if __name__ == "__main__":
    run_training_pipeline()
