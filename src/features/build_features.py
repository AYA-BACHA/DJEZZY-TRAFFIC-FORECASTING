"""
src/features/build_features.py
Feature engineering pipeline for the Djezzy network traffic dataset.

Run as:  python -m src.features.build_features

Features generated:
  - Temporal: hour, day-of-week, month, quarter, etc.
  - Algerian calendar: Ramadan, Eid, BAC exams, weekend (Fri/Sat), holidays
  - Network patterns: rolling traffic means/stds, lag features
  - Congestion: PRB + derived congestion levels
  - Cell metadata: technology, area_type, wilaya encodings
"""

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 1. Temporal features
# ---------------------------------------------------------------------------
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day-of-week, month, and cyclical encodings."""
    log.info("Adding temporal features...")
    df = df.copy()
    ts = df["timestamp"].dt

    df["hour"] = ts.hour.astype("int8")
    df["day_of_week"] = ts.dayofweek.astype("int8")   # 0=Mon ... 6=Sun
    df["day_of_month"] = ts.day.astype("int8")
    df["month"] = ts.month.astype("int8")
    df["quarter"] = ts.quarter.astype("int8")
    df["week_of_year"] = ts.isocalendar().week.astype("int8")
    df["year"] = ts.year.astype("int16")
    df["day_of_year"] = ts.dayofyear.astype("int16")

    # Cyclical encodings (sin/cos) for hour, day_of_week, month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Time of day segments
    df["time_segment"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 13, 17, 20, 23],
        labels=["night", "morning", "midday", "afternoon", "evening", "late_evening"],
    ).astype("category")

    log.info("  Added %d temporal feature columns.", 15)
    return df


# ---------------------------------------------------------------------------
# 2. Algerian calendar features
# ---------------------------------------------------------------------------
def _date_in_ranges(dates: pd.Series, ranges: List[dict]) -> pd.Series:
    """Return bool Series: True where date falls in any of the ranges."""
    result = pd.Series(False, index=dates.index)
    for r in ranges:
        start = pd.Timestamp(r["start"]).normalize()
        end = pd.Timestamp(r["end"]).normalize()
        result |= (dates >= start) & (dates <= end)
    return result


def add_algerian_calendar_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add Ramadan, Eid, exams, weekend, and public holiday flags."""
    log.info("Adding Algerian calendar features...")
    df = df.copy()
    cal = cfg["calendar"]
    dates = df["timestamp"].dt.normalize().dt.tz_localize(None)

    # Algerian weekend: Friday (4) or Saturday (5)
    dow = df["day_of_week"] if "day_of_week" in df.columns else df["timestamp"].dt.dayofweek
    df["is_weekend"] = dow.isin([4, 5]).astype("int8")

    # Ramadan
    df["is_ramadan"] = _date_in_ranges(dates, cal["ramadan"]).astype("int8")

    # Eid days
    eid_fitr_dates = [pd.Timestamp(d).normalize() for d in cal["eid_al_fitr"]]
    eid_adha_dates = [pd.Timestamp(d).normalize() for d in cal["eid_al_adha"]]
    df["is_eid_al_fitr"] = dates.isin(eid_fitr_dates).astype("int8")
    df["is_eid_al_adha"] = dates.isin(eid_adha_dates).astype("int8")
    df["is_eid"] = ((df["is_eid_al_fitr"] == 1) | (df["is_eid_al_adha"] == 1)).astype("int8")

    # BAC & BEM exams
    df["is_bac_exam"] = _date_in_ranges(dates, cal["bac_exams"]).astype("int8")
    df["is_bem_exam"] = _date_in_ranges(dates, cal["bem_exams"]).astype("int8")
    df["is_exam_period"] = ((df["is_bac_exam"] == 1) | (df["is_bem_exam"] == 1)).astype("int8")

    # Summer season
    df["is_summer"] = _date_in_ranges(dates, cal["summer_season"]).astype("int8")

    # Football matches (±2 hours window)
    df["is_football_match"] = 0
    for m in cal["football_matches"]:
        center = pd.Timestamp(m["datetime"]).tz_localize("Africa/Algiers")
        window_start = center - pd.Timedelta(hours=1)
        window_end = center + pd.Timedelta(hours=3)
        mask = (df["timestamp"] >= window_start) & (df["timestamp"] <= window_end)
        df.loc[mask, "is_football_match"] = 1
    df["is_football_match"] = df["is_football_match"].astype("int8")

    # 5G launch flag: after Dec 2025
    g5_date = pd.Timestamp(cal["g5_launch"]).tz_localize("Africa/Algiers")
    df["post_5g_launch"] = (df["timestamp"] >= g5_date).astype("int8")

    # Fixed public holidays (month-day)
    holiday_month_days = [(int(d.split("-")[0]), int(d.split("-")[1]))
                          for d in cal["fixed_holidays"]]
    df["is_public_holiday"] = dates.apply(
        lambda d: (d.month, d.day) in holiday_month_days
    ).astype("int8")

    # Composite: is special day
    df["is_special_day"] = (
        (df["is_weekend"] == 1) | (df["is_eid"] == 1) |
        (df["is_public_holiday"] == 1) | (df["is_ramadan"] == 1)
    ).astype("int8")

    # Ramadan hour interaction
    hr = df["hour"] if "hour" in df.columns else df["timestamp"].dt.hour
    df["ramadan_hour"] = (df["is_ramadan"] * hr).astype("int8")

    log.info("  Added Algerian calendar features.")
    return df


# ---------------------------------------------------------------------------
# 3. Cell-level lag and rolling features
# ---------------------------------------------------------------------------
def add_lag_rolling_features(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    """Add lag and rolling window features per cell."""
    log.info("Adding lag and rolling features...")
    df = df.copy()
    df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)

    lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h to 1 week
    windows = [6, 24, 168]                        # 6h, 24h, 1 week

    for col in target_cols:
        log.info("  Processing lags/rolling for '%s'", col)
        grp = df.groupby("cell_id")[col]

        for lag in lag_hours:
            df[f"{col}_lag{lag}h"] = grp.shift(lag)

        for w in windows:
            df[f"{col}_roll_mean_{w}h"] = grp.transform(
                lambda s: s.shift(1).rolling(window=w, min_periods=max(1, w // 2)).mean()
            )
            df[f"{col}_roll_std_{w}h"] = grp.transform(
                lambda s: s.shift(1).rolling(window=w, min_periods=max(1, w // 2)).std()
            )

        # Same hour last day
        df[f"{col}_same_hour_1d"] = grp.shift(24)
        # Same hour last week
        df[f"{col}_same_hour_1w"] = grp.shift(168)

    log.info("  Done with lags and rolling features.")
    return df


# ---------------------------------------------------------------------------
# 4. Metadata encoding
# ---------------------------------------------------------------------------
def add_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode cell metadata as numeric features."""
    log.info("Adding metadata features...")
    df = df.copy()

    # Technology ordinal
    tech_order = {"2G": 0, "3G": 1, "4G": 2, "5G": 3}
    df["technology_ord"] = df["technology"].map(tech_order).astype("int8")

    # Area type ordinal (population density proxy)
    area_order = {"rural": 0, "highway": 1, "suburban": 2, "urban": 3, "dense_urban": 4}
    df["area_type_ord"] = df["area_type"].map(area_order).astype("int8")

    # Wilaya code stays as-is (numeric)

    # Band as numeric already

    # Congestion level: 0=normal, 1=warning, 2=critical
    df["congestion_level"] = pd.cut(
        df["prb_utilization_pct"],
        bins=[-0.1, 80, 90, 100],
        labels=[0, 1, 2],
    ).astype("float32")
    # Where PRB is 100 (capped), keep as level 2
    df.loc[df["prb_utilization_pct"] >= 100, "congestion_level"] = 2

    log.info("  Added metadata features.")
    return df


# ---------------------------------------------------------------------------
# 5. Site-level aggregate features
# ---------------------------------------------------------------------------
def add_site_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-site aggregations to provide network-wide context to each row."""
    log.info("Adding site-level aggregate features...")
    df = df.copy()
    df = df.sort_values(["site_id", "timestamp"]).reset_index(drop=True)

    site_ts_dl = df.groupby(["site_id", "timestamp"])["dl_traffic_volume_gb"].transform("sum")
    df["site_total_dl_gb"] = site_ts_dl

    site_ts_prb = df.groupby(["site_id", "timestamp"])["prb_utilization_pct"].transform("mean")
    df["site_avg_prb_pct"] = site_ts_prb

    site_ts_cong = df.groupby(["site_id", "timestamp"])["congestion_flag"].transform("max")
    df["site_any_congested"] = site_ts_cong.astype("int8")

    log.info("  Added site aggregate features.")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_feature_pipeline(config_path: str = "config/config.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)
    in_path = cfg["paths"]["processed_data"]
    out_path = cfg["paths"]["features_data"]

    log.info("=" * 60)
    log.info("Djezzy Traffic Forecasting — Feature Engineering Pipeline")
    log.info("=" * 60)
    log.info("Loading cleaned data from %s", in_path)
    df = pd.read_parquet(in_path, engine="pyarrow")
    log.info("Loaded shape: %s", df.shape)

    df = add_temporal_features(df)
    df = add_algerian_calendar_features(df, cfg)
    df = add_site_aggregate_features(df)
    df = add_metadata_features(df)

    # Lag/rolling only for main targets
    target_cols = ["dl_traffic_volume_gb", "prb_utilization_pct"]
    df = add_lag_rolling_features(df, target_cols)

    # Final sort and save
    df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")
    log.info("Feature dataset saved to %s", out_path)
    log.info("Final shape: %s", df.shape)
    log.info("Memory usage: %.1f MB", df.memory_usage(deep=True).sum() / 1e6)

    return df


if __name__ == "__main__":
    run_feature_pipeline()
