"""
tests/test_features.py
Unit tests for the feature engineering pipeline.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.build_features import (
    add_temporal_features,
    add_algerian_calendar_features,
    add_metadata_features,
)


@pytest.fixture
def sample_df():
    """Sample cleaned DataFrame for feature tests."""
    n = 48
    timestamps = pd.date_range("2024-03-15", periods=n, freq="h", tz="Africa/Algiers")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "cell_id": pd.Categorical(["cell_A"] * n),
        "site_id": pd.Categorical(["site_1"] * n),
        "technology": pd.Categorical(["4G"] * n),
        "area_type": pd.Categorical(["urban"] * n),
        "wilaya_name": pd.Categorical(["Alger"] * n),
        "wilaya_code": np.int8(16),
        "dl_traffic_volume_gb": np.random.uniform(1, 10, n),
        "prb_utilization_pct": np.random.uniform(20, 70, n),
        "congestion_flag": np.zeros(n, dtype=int),
        "outage_flag": np.zeros(n, dtype=int),
    })
    return df


@pytest.fixture
def sample_cfg():
    import yaml
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestTemporalFeatures:
    def test_hour_range(self, sample_df):
        df_out = add_temporal_features(sample_df)
        assert df_out["hour"].between(0, 23).all()

    def test_day_of_week_range(self, sample_df):
        df_out = add_temporal_features(sample_df)
        assert df_out["day_of_week"].between(0, 6).all()

    def test_cyclical_features_range(self, sample_df):
        df_out = add_temporal_features(sample_df)
        assert df_out["hour_sin"].between(-1, 1).all()
        assert df_out["hour_cos"].between(-1, 1).all()

    def test_time_segment_categories(self, sample_df):
        df_out = add_temporal_features(sample_df)
        valid = {"night", "morning", "midday", "afternoon", "evening", "late_evening"}
        actual = set(df_out["time_segment"].dropna().astype(str).unique())
        assert actual.issubset(valid)


class TestAlgerianCalendarFeatures:
    def test_ramadan_flag_set(self, sample_df, sample_cfg):
        # sample_df starts 2024-03-15 (in Ramadan 2024: Mar 11 - Apr 9)
        df_out = add_algerian_calendar_features(sample_df, sample_cfg)
        assert df_out["is_ramadan"].sum() > 0

    def test_algerian_weekend(self, sample_df, sample_cfg):
        df_out = add_temporal_features(sample_df)
        df_out = add_algerian_calendar_features(df_out, sample_cfg)
        # Check that Friday (4) and Saturday (5) have is_weekend=1
        fri_sat = df_out["day_of_week"].isin([4, 5])
        if fri_sat.any():
            assert (df_out.loc[fri_sat, "is_weekend"] == 1).all()

    def test_flags_are_binary(self, sample_df, sample_cfg):
        df_out = add_algerian_calendar_features(sample_df, sample_cfg)
        binary_cols = ["is_weekend", "is_ramadan", "is_eid", "is_bac_exam",
                       "is_bem_exam", "is_football_match", "post_5g_launch",
                       "is_public_holiday", "is_special_day", "is_summer"]
        for col in binary_cols:
            if col in df_out.columns:
                assert df_out[col].isin([0, 1]).all(), f"Column {col} has non-binary values"


class TestMetadataFeatures:
    def test_technology_ordinal_valid(self, sample_df):
        df_out = add_metadata_features(sample_df)
        assert "technology_ord" in df_out.columns
        assert df_out["technology_ord"].isin([0, 1, 2, 3]).all()

    def test_area_type_ordinal_valid(self, sample_df):
        df_out = add_metadata_features(sample_df)
        assert "area_type_ord" in df_out.columns
        assert df_out["area_type_ord"].isin([0, 1, 2, 3, 4]).all()
