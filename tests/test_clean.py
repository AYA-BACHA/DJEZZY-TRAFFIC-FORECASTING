"""
tests/test_clean.py
Unit tests for the data cleaning pipeline.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.clean import (
    normalise_categoricals,
    deduplicate,
    fix_kpi_ranges,
    fix_congestion_flag,
    validate_cleaned,
    TECH_MAP,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def minimal_df():
    """Minimal valid DataFrame simulating cleaned structure."""
    n = 24
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="Africa/Algiers")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "site_id": ["DZ-16-0001"] * n,
        "cell_id": ["DZ-16-0001-L18-1"] * n,
        "enodeb_id": [1001] * n,
        "gnodeb_id": [float("nan")] * n,
        "lac_tac": [100] * n,
        "mcc": [603] * n,
        "mnc": ["02"] * n,
        "technology": ["4G"] * n,
        "band": [1800] * n,
        "azimuth": [120] * n,
        "wilaya_code": [16] * n,
        "wilaya_name": ["Alger"] * n,
        "area_type": ["urban"] * n,
        "dl_traffic_volume_gb": np.random.uniform(0.5, 5.0, n),
        "ul_traffic_volume_gb": np.random.uniform(0.05, 0.5, n),
        "voice_traffic_erlang": np.random.uniform(0, 10, n),
        "rrc_connected_users": np.random.randint(10, 100, n),
        "active_users": np.random.randint(1, 10, n),
        "prb_utilization_pct": np.random.uniform(20, 75, n),
        "avg_user_throughput_mbps": np.random.uniform(5, 50, n),
        "latency_ms": np.random.uniform(5, 50, n),
        "cell_availability_pct": [100.0] * n,
        "call_drop_rate_pct": np.random.uniform(0.1, 1.0, n),
        "handover_success_rate_pct": np.random.uniform(96, 99, n),
        "congestion_flag": [0] * n,
        "cum_dl_traffic_gb": np.cumsum(np.random.uniform(0.5, 5.0, n)),
    })
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestTechMap:
    def test_all_4g_variants_map_to_4G(self):
        variants = ["4G", "LTE", "lte", "4g", " 4G ", "4G-LTE", "4G ", "LTE"]
        for v in variants:
            assert TECH_MAP.get(v.strip() if v != " 4G " else v) == "4G" or \
                   TECH_MAP.get(v) == "4G", f"Failed for {v!r}"

    def test_all_2g_variants_map_to_2G(self):
        for v in ["2G", "2g", "GSM", "gsm", " 2G"]:
            assert TECH_MAP.get(v) == "2G", f"Failed for {v!r}"

    def test_5g_maps_correctly(self):
        for v in ["5G", "5g", "NR", "5G NR"]:
            assert TECH_MAP.get(v) == "5G", f"Failed for {v!r}"


class TestNormaliseCategoricals:
    def test_technology_normalised(self, minimal_df):
        df_in = minimal_df.copy()
        df_in["technology"] = ["LTE", " 4G ", "4g", "4G-LTE"] + ["4G"] * 20
        df_out = normalise_categoricals(df_in)
        assert df_out["technology"].isin(["2G", "3G", "4G", "5G"]).all()

    def test_area_type_stripped(self, minimal_df):
        df_in = minimal_df.copy()
        df_in["area_type"] = " urban "
        df_out = normalise_categoricals(df_in)
        assert (df_out["area_type"] == "urban").all()

    def test_mnc_zero_padded(self, minimal_df):
        df_in = minimal_df.copy()
        df_in["mnc"] = 2
        df_out = normalise_categoricals(df_in)
        assert (df_out["mnc"] == "02").all()


class TestDeduplicate:
    def test_removes_exact_duplicates(self, minimal_df):
        df_dup = pd.concat([minimal_df, minimal_df.iloc[:5]], ignore_index=True)
        df_out, n_dup = deduplicate(df_dup)
        assert n_dup == 5
        assert not df_out.duplicated(["cell_id", "timestamp"]).any()

    def test_no_duplicates_unchanged(self, minimal_df):
        df_out, n_dup = deduplicate(minimal_df)
        assert n_dup == 0
        assert len(df_out) == len(minimal_df)


class TestFixKPIRanges:
    def test_negative_dl_becomes_nan(self, minimal_df):
        df_in = minimal_df.copy()
        df_in.loc[0, "dl_traffic_volume_gb"] = -5.0
        df_out = fix_kpi_ranges(df_in)
        assert pd.isna(df_out.loc[0, "dl_traffic_volume_gb"])

    def test_prb_over_100_capped(self, minimal_df):
        df_in = minimal_df.copy()
        df_in.loc[0, "prb_utilization_pct"] = 150.0
        df_out = fix_kpi_ranges(df_in)
        assert df_out.loc[0, "prb_utilization_pct"] == 100.0

    def test_availability_over_100_capped(self, minimal_df):
        df_in = minimal_df.copy()
        df_in.loc[0, "cell_availability_pct"] = 120.0
        df_out = fix_kpi_ranges(df_in)
        assert df_out.loc[0, "cell_availability_pct"] == 100.0

    def test_active_gt_rrc_corrected(self, minimal_df):
        df_in = minimal_df.copy()
        df_in.loc[0, "rrc_connected_users"] = 5
        df_in.loc[0, "active_users"] = 10
        df_out = fix_kpi_ranges(df_in)
        assert df_out.loc[0, "active_users"] <= df_out.loc[0, "rrc_connected_users"]


class TestFixCongestionFlag:
    def test_prb_above_threshold_flagged(self, minimal_df):
        df_in = minimal_df.copy()
        df_in.loc[0, "prb_utilization_pct"] = 85.0
        df_out = fix_congestion_flag(df_in, threshold=80.0)
        assert df_out.loc[0, "congestion_flag"] == 1

    def test_prb_below_threshold_not_flagged(self, minimal_df):
        df_in = minimal_df.copy()
        df_in.loc[0, "prb_utilization_pct"] = 75.0
        df_out = fix_congestion_flag(df_in, threshold=80.0)
        assert df_out.loc[0, "congestion_flag"] == 0

    def test_flag_is_binary(self, minimal_df):
        df_out = fix_congestion_flag(minimal_df)
        assert df_out["congestion_flag"].isin([0, 1]).all()


class TestValidateCleaned:
    def test_valid_df_passes(self, minimal_df):
        df_in = minimal_df.copy()
        df_in["outage_flag"] = 0
        assert validate_cleaned(df_in) is True

    def test_invalid_technology_fails(self, minimal_df):
        df_in = minimal_df.copy()
        df_in["outage_flag"] = 0
        df_in.loc[0, "technology"] = "LTE"  # not canonical
        result = validate_cleaned(df_in)
        assert result is False
