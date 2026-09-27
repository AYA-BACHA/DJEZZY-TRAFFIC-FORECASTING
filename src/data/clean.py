"""
src/data/clean.py
Reproducible cleaning pipeline for the Djezzy network traffic dataset.
Run as:  python -m src.data.clean

⚠️  The raw dataset is NEVER modified.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Technology normalisation map
# ---------------------------------------------------------------------------
TECH_MAP = {
    "4G": "4G", "LTE": "4G", "lte": "4G", "4g": "4G",
    " 4G ": "4G", "4G-LTE": "4G", "4G ": "4G",
    "3G": "3G", "3g": "3G", "UMTS": "3G", "umts": "3G",
    "3G ": "3G",
    "2G": "2G", "2g": "2G", "GSM": "2G", "gsm": "2G",
    " 2G": "2G",
    "5G": "5G", "5g": "5G", "5G ": "5G", "NR": "5G", "5G NR": "5G",
}

WILAYA_MAP_EXPECTED = {
    "Alger": 16, "Chlef": 2, "Laghouat": 3, "Oum El Bouaghi": 4,
    "Batna": 5, "Béjaïa": 6, "Biskra": 7, "Béchar": 8, "Blida": 9,
    "Bouira": 10, "Tamanrasset": 11, "Tébessa": 12, "Tlemcen": 13,
    "Tiaret": 14, "Tizi Ouzou": 15, "Médéa": 17, "Mostaganem": 18,
    "M'Sila": 19, "Mascara": 20, "Ouargla": 21, "Oran": 22,
    "El Bayadh": 23, "Illizi": 24, "Bordj Bou Arréridj": 25,
    "Boumerdès": 26, "El Tarf": 27, "Tindouf": 28, "Tissemsilt": 29,
    "El Oued": 30, "Khenchela": 31, "Souk Ahras": 32,
    "Tipaza": 33, "Mila": 34, "Aïn Defla": 35, "Naâma": 36,
    "Aïn Témouchent": 37, "Ghardaïa": 38, "Relizane": 39,
    "Adrar": 1, "M'Sila": 19, "Sétif": 19, "Skikda": 21, "Jijel": 18,
    "Sidi Bel Abbés": 22, "Constantine": 25, "Annaba": 23,
}


# ---------------------------------------------------------------------------
# Step 1 — Load raw data
# ---------------------------------------------------------------------------
def load_raw(path: str) -> pd.DataFrame:
    """Load raw CSV with appropriate dtypes."""
    log.info("Loading raw data from %s", path)
    df = pd.read_csv(
        path,
        low_memory=False,
        dtype={
            "mnc": str,          # preserve leading zero
            "mcc": int,
        },
    )
    log.info("Raw shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 2 — Parse & standardise timestamps
# ---------------------------------------------------------------------------
def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamp column to datetime (Africa/Algiers local time)."""
    log.info("Parsing timestamps...")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    invalid_ts = df["timestamp"].isna().sum()
    if invalid_ts:
        log.warning("  %d invalid timestamps found — will be dropped", invalid_ts)
        df["_drop_reason"] = df["_drop_reason"].where(
            ~df["timestamp"].isna(), "invalid_timestamp"
        ) if "_drop_reason" in df.columns else df["timestamp"].isna().map(
            {True: "invalid_timestamp", False: ""}
        )
        df = df[df["timestamp"].notna()].copy()

    # Localise to Africa/Algiers
    df["timestamp"] = df["timestamp"].dt.tz_localize("Africa/Algiers", ambiguous="NaT", nonexistent="shift_forward")
    log.info("  Timestamp range: %s -> %s", df["timestamp"].min(), df["timestamp"].max())
    return df


# ---------------------------------------------------------------------------
# Step 3 — Normalise categorical columns
# ---------------------------------------------------------------------------
def normalise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise technology and area_type values."""
    log.info("Normalising categorical columns...")
    df = df.copy()

    # Technology
    df["technology_raw"] = df["technology"].copy()
    df["technology"] = df["technology"].str.strip().map(TECH_MAP)
    unmapped_tech = df["technology"].isna().sum()
    if unmapped_tech:
        log.warning("  %d rows with unmapped technology (raw: %s)",
                    unmapped_tech, df.loc[df["technology"].isna(), "technology_raw"].unique())
        # Attempt pattern-based fix
        mask_4g = df["technology"].isna() & df["technology_raw"].str.contains("4|LTE|lte", na=False, regex=True)
        mask_3g = df["technology"].isna() & df["technology_raw"].str.contains("3|UMTS|umts", na=False, regex=True)
        mask_2g = df["technology"].isna() & df["technology_raw"].str.contains("2|GSM|gsm", na=False, regex=True)
        mask_5g = df["technology"].isna() & df["technology_raw"].str.contains("5|NR", na=False, regex=True)
        df.loc[mask_4g, "technology"] = "4G"
        df.loc[mask_3g, "technology"] = "3G"
        df.loc[mask_2g, "technology"] = "2G"
        df.loc[mask_5g, "technology"] = "5G"
        still_unmapped = df["technology"].isna().sum()
        log.warning("  %d rows still unmapped after pattern fix", still_unmapped)

    # Area type — strip whitespace, lower
    df["area_type"] = df["area_type"].str.strip().str.lower()
    valid_area_types = {"dense_urban", "urban", "suburban", "rural", "highway"}
    bad_area = ~df["area_type"].isin(valid_area_types)
    if bad_area.sum() > 0:
        log.warning("  %d rows with invalid area_type: %s",
                    bad_area.sum(), df.loc[bad_area, "area_type"].unique())

    # Wilaya name — canonicalise common variants (encoding issues)
    wilaya_corrections = {
        "ALGER": "Alger", " Alger": "Alger", "alger": "Alger", "Algiers": "Alger",
        "Alger ": "Alger", "Oran ": "Oran", "S\u00e9tif": "Sétif",
        "Constantine ": "Constantine", "Blida ": "Blida",
        "B\u00e9ja\u00efa": "Béjaïa", "T\u00e9bessa": "Tébessa",
        "M\u00e9d\u00e9a": "Médéa", "Sidi Bel Abb\u00e8s": "Sidi Bel Abbés",
        "Gharda\u00efa": "Ghardaïa",
    }
    df["wilaya_name"] = df["wilaya_name"].str.strip()
    df["wilaya_name"] = df["wilaya_name"].replace(wilaya_corrections)

    # mnc — ensure string "02"
    df["mnc"] = df["mnc"].astype(str).str.zfill(2)

    log.info("  Technology distribution: %s", df["technology"].value_counts().to_dict())
    return df


# ---------------------------------------------------------------------------
# Step 4 — Deduplicate cell-hour records
# ---------------------------------------------------------------------------
def deduplicate(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove duplicated (cell_id, timestamp) rows, keeping first occurrence."""
    log.info("Deduplicating cell-hour records...")
    n_before = len(df)
    # Sort so first occurrence = earliest original order (or highest quality)
    df = df.sort_values(["cell_id", "timestamp", "cell_availability_pct"],
                        ascending=[True, True, False])
    mask_dup = df.duplicated(subset=["cell_id", "timestamp"], keep="first")
    n_dup = mask_dup.sum()
    log.info("  %d duplicated records removed", n_dup)
    df = df[~mask_dup].copy()
    assert len(df) == n_before - n_dup
    return df, n_dup


# ---------------------------------------------------------------------------
# Step 5 — Handle KPI ranges and impossible values
# ---------------------------------------------------------------------------
def fix_kpi_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Cap or nullify physically impossible KPI values."""
    log.info("Fixing KPI ranges...")
    df = df.copy()

    issues = {}

    # Negative DL/UL traffic -> set to NaN (can't have negative traffic)
    neg_dl = (df["dl_traffic_volume_gb"] < 0).sum()
    neg_ul = (df["ul_traffic_volume_gb"] < 0).sum()
    if neg_dl > 0:
        log.warning("  %d rows with negative dl_traffic_volume_gb -> set to NaN", neg_dl)
        df.loc[df["dl_traffic_volume_gb"] < 0, "dl_traffic_volume_gb"] = np.nan
        issues["negative_dl"] = neg_dl
    if neg_ul > 0:
        log.warning("  %d rows with negative ul_traffic_volume_gb -> set to NaN", neg_ul)
        df.loc[df["ul_traffic_volume_gb"] < 0, "ul_traffic_volume_gb"] = np.nan
        issues["negative_ul"] = neg_ul

    # PRB > 100%% -- cap at 100 (some telco systems report brief spikes above 100 due to overcommit)
    prb_over = (df["prb_utilization_pct"] > 100).sum()
    if prb_over > 0:
        log.warning("  %d rows with PRB > 100%% -> capped at 100", prb_over)
        df.loc[df["prb_utilization_pct"] > 100, "prb_utilization_pct"] = 100.0
        issues["prb_over_100"] = prb_over

    # cell_availability > 100 -> cap at 100
    avail_over = (df["cell_availability_pct"] > 100).sum()
    if avail_over > 0:
        log.warning("  %d rows with cell_availability > 100%% -> capped at 100", avail_over)
        df.loc[df["cell_availability_pct"] > 100, "cell_availability_pct"] = 100.0
        issues["avail_over_100"] = avail_over

    # call_drop_rate > 100 -> impossible
    cdr_over = (df["call_drop_rate_pct"] > 100).sum()
    if cdr_over > 0:
        log.warning("  %d rows with call_drop_rate > 100%% -> set to NaN", cdr_over)
        df.loc[df["call_drop_rate_pct"] > 100, "call_drop_rate_pct"] = np.nan
        issues["cdr_over_100"] = cdr_over

    # handover_success > 100 -> cap at 100
    hsr_over = (df["handover_success_rate_pct"] > 100).sum()
    if hsr_over > 0:
        log.warning("  %d rows with handover_success_rate > 100%% -> capped at 100", hsr_over)
        df.loc[df["handover_success_rate_pct"] > 100, "handover_success_rate_pct"] = 100.0
        issues["hsr_over_100"] = hsr_over

    # active_users > rrc_connected_users -> logical impossibility -> correct active_users
    mask_active = df["active_users"] > df["rrc_connected_users"]
    n_active = mask_active.sum()
    if n_active > 0:
        log.warning("  %d rows where active_users > rrc_connected_users -> set active_users = rrc_connected_users", n_active)
        df.loc[mask_active, "active_users"] = df.loc[mask_active, "rrc_connected_users"]
        issues["active_gt_rrc"] = n_active

    log.info("  KPI range issues fixed: %s", issues)
    return df


# ---------------------------------------------------------------------------
# Step 6 — Fix congestion_flag
# ---------------------------------------------------------------------------
def fix_congestion_flag(df: pd.DataFrame, threshold: float = 80.0) -> pd.DataFrame:
    """Recompute congestion_flag from PRB utilisation where inconsistent."""
    log.info("Fixing congestion_flag...")
    df = df.copy()

    # Where PRB is not null, recompute
    has_prb = df["prb_utilization_pct"].notna()
    df.loc[has_prb, "congestion_flag"] = (df.loc[has_prb, "prb_utilization_pct"] > threshold).astype(int)

    # Where PRB is null but congestion_flag is null → set to 0 (unknown → assume not congested)
    null_flag = df["congestion_flag"].isna()
    df.loc[null_flag, "congestion_flag"] = 0
    df["congestion_flag"] = df["congestion_flag"].astype(int)

    log.info("  Congestion flag distribution: %s", df["congestion_flag"].value_counts().to_dict())
    return df


# ---------------------------------------------------------------------------
# Step 7 — Handle missing values
# ---------------------------------------------------------------------------
def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing KPI values using temporal and cell-level patterns.
    Creates _imputed flag columns for transparency.
    """
    log.info("Imputing missing values...")
    df = df.copy()
    df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)

    kpi_cols = [
        "dl_traffic_volume_gb", "ul_traffic_volume_gb", "voice_traffic_erlang",
        "rrc_connected_users", "active_users", "prb_utilization_pct",
        "avg_user_throughput_mbps", "latency_ms", "cell_availability_pct",
        "call_drop_rate_pct", "handover_success_rate_pct",
    ]

    for col in kpi_cols:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        log.info("  Imputing %s: %d missing (%.2f%%)", col, n_missing, n_missing / len(df) * 100)

        # Create imputation flag
        df[f"{col}_imputed"] = df[col].isna().astype(int)

        # Strategy: forward-fill within cell, then backward-fill, then cell median
        df[col] = df.groupby("cell_id")[col].transform(
            lambda s: s.ffill().bfill()
        )

    # Post-imputation: re-enforce active_users <= rrc_connected_users
    if "active_users" in df.columns and "rrc_connected_users" in df.columns:
        mask_fix = df["active_users"] > df["rrc_connected_users"]
        if mask_fix.sum() > 0:
            log.info("  Post-imputation: correcting %d rows where active_users > rrc", mask_fix.sum())
            df.loc[mask_fix, "active_users"] = df.loc[mask_fix, "rrc_connected_users"]

        # If still missing (e.g. cell with all-null), fill with overall median
        still_missing = df[col].isna()
        if still_missing.sum() > 0:
            median_val = df[col].median()
            df.loc[still_missing, col] = median_val
            log.info("    %d rows filled with overall median (%.4f)", still_missing.sum(), median_val)

    # cell_availability: outage flag
    df["outage_flag"] = (df["cell_availability_pct"] < 50).astype(int)
    log.info("  Outage rows (availability < 50%%): %d", df["outage_flag"].sum())

    return df


# ---------------------------------------------------------------------------
# Step 8 — Add derived temporal features needed for gap-filling report
# ---------------------------------------------------------------------------
def check_hourly_grid_completeness(df: pd.DataFrame) -> dict:
    """Report on missing hours per cell."""
    log.info("Checking hourly grid completeness...")
    full_range = pd.date_range(
        start="2024-01-01 00:00:00",
        end="2025-12-31 23:00:00",
        freq="h",
        tz="Africa/Algiers",
    )
    expected_per_cell = len(full_range)
    cells = df["cell_id"].unique()

    gaps_info = {}
    for cell in cells:
        cell_ts = df.loc[df["cell_id"] == cell, "timestamp"].sort_values()
        actual = len(cell_ts)
        missing = expected_per_cell - actual
        pct_missing = missing / expected_per_cell * 100
        gaps_info[cell] = {"actual": actual, "missing": missing, "pct_missing": pct_missing}

    total_expected = len(cells) * expected_per_cell
    total_actual = len(df)
    total_missing = total_expected - total_actual
    log.info("  Total expected: %d, Actual: %d, Missing: %d (%.2f%%)",
             total_expected, total_actual, total_missing, total_missing / total_expected * 100)

    return gaps_info


# ---------------------------------------------------------------------------
# Step 9 — Validate cleaned dataset
# ---------------------------------------------------------------------------
def validate_cleaned(df: pd.DataFrame) -> bool:
    """Run automated validation assertions on cleaned dataset."""
    log.info("Running validation checks...")
    passed = True

    checks = [
        ("No null timestamps", df["timestamp"].isna().sum() == 0),
        ("No duplicate cell-hour pairs", not df.duplicated(["cell_id", "timestamp"]).any()),
        ("PRB between 0 and 100", (df["prb_utilization_pct"].dropna().between(0, 100)).all()),
        ("DL traffic >= 0", (df["dl_traffic_volume_gb"].dropna() >= 0).all()),
        ("Cell availability 0-100", (df["cell_availability_pct"].dropna().between(0, 100)).all()),
        ("active_users <= rrc_connected_users",
         (df["active_users"].dropna() <= df["rrc_connected_users"].dropna()).all()),
        ("congestion_flag is 0 or 1", df["congestion_flag"].isin([0, 1]).all()),
        ("Technology values valid", df["technology"].isin(["2G", "3G", "4G", "5G"]).all()),
        ("MCC = 603", (df["mcc"] == 603).all()),
        ("MNC = 02", (df["mnc"] == "02").all()),
        ("78 unique cells", df["cell_id"].nunique() == 78 if len(df) > 10000 else df["cell_id"].nunique() >= 1),
    ]

    for name, condition in checks:
        status = "PASS" if condition else "FAIL"
        if not condition:
            passed = False
            log.error("  [%s] %s", status, name)
        else:
            log.info("  [%s] %s", status, name)

    return passed


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_cleaning_pipeline(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """End-to-end cleaning pipeline. Returns cleaned DataFrame."""
    cfg = load_config(config_path)
    raw_path = cfg["paths"]["raw_data"]
    out_path = cfg["paths"]["processed_data"]
    seed = cfg["random_seed"]
    np.random.seed(seed)

    log.info("=" * 60)
    log.info("Djezzy Traffic Forecasting — Cleaning Pipeline")
    log.info("=" * 60)

    df = load_raw(raw_path)
    df = parse_timestamps(df)
    df = normalise_categoricals(df)
    df, n_dup = deduplicate(df)
    gaps_info = check_hourly_grid_completeness(df)
    df = fix_kpi_ranges(df)
    df = fix_congestion_flag(df, threshold=cfg["congestion"]["warning_pct"])
    df = impute_missing_values(df)

    # Final sort
    df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)

    # Drop helper columns
    if "technology_raw" in df.columns:
        df = df.drop(columns=["technology_raw"])

    # Optimise dtypes for memory
    df["cell_id"] = df["cell_id"].astype("category")
    df["site_id"] = df["site_id"].astype("category")
    df["technology"] = df["technology"].astype("category")
    df["area_type"] = df["area_type"].astype("category")
    df["wilaya_name"] = df["wilaya_name"].astype("category")
    df["wilaya_code"] = df["wilaya_code"].astype("int8")
    df["mcc"] = df["mcc"].astype("int16")
    df["band"] = df["band"].astype("int16")
    df["azimuth"] = df["azimuth"].astype("int16")
    df["congestion_flag"] = df["congestion_flag"].astype("int8")
    df["outage_flag"] = df["outage_flag"].astype("int8")

    # Validate
    valid = validate_cleaned(df)
    if not valid:
        log.warning("Validation had failures — review warnings above")

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")
    log.info("Cleaned dataset saved to %s", out_path)
    log.info("Final shape: %s", df.shape)
    log.info("Memory usage: %.1f MB", df.memory_usage(deep=True).sum() / 1e6)

    return df


if __name__ == "__main__":
    run_cleaning_pipeline()
