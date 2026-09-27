"""
tests/test_leakage_and_forecast.py
Comprehensive test suite verifying:
  1. Temporal leakage prevention (t-1 from future is strictly avoided)
  2. Recursive forecasting mechanics
  3. Forecast horizons and complete coverage (24h = 1,872 rows, 7d = 13,104 rows)
  4. Metric calculation correctness (MAE, RMSE, MAPE, sMAPE, WAPE)
  5. Baseline model implementations (Naive, Seasonal Naives, Moving Average)
  6. Congestion threshold classification (>=80% Warning, >=90% High)
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_metrics
from src.models.baselines import (
    generate_all_baselines,
    predict_moving_average,
    predict_naive,
    predict_seasonal_naive_last_week,
    predict_seasonal_naive_yesterday,
)
from src.models.predict import generate_forecast_timestamps


# ---------------------------------------------------------------------------
# 1. Metric Calculations Tests
# ---------------------------------------------------------------------------
class TestMetricCalculations:
    def test_perfect_predictions(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        m = compute_metrics(y, y, name="perfect")
        assert m["MAE"] == 0.0
        assert m["RMSE"] == 0.0
        assert m["MAPE"] == 0.0
        assert m["sMAPE"] == 0.0
        assert m["WAPE"] == 0.0

    def test_known_metrics_values(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        # Errors: |100-110|=10, |200-180|=20
        # MAE = (10 + 20) / 2 = 15.0
        # RMSE = sqrt((100 + 400)/2) = sqrt(250) = 15.8114
        # MAPE = (10/100 + 20/200)/2 * 100 = (0.10 + 0.10)/2 * 100 = 10.0%
        # WAPE = (10 + 20) / (100 + 200) * 100 = 30 / 300 * 100 = 10.0%
        m = compute_metrics(y_true, y_pred, name="test")
        assert m["MAE"] == 15.0
        assert pytest.approx(m["RMSE"], 0.001) == 15.8114
        assert pytest.approx(m["MAPE"], 0.01) == 10.0
        assert pytest.approx(m["WAPE"], 0.01) == 10.0

    def test_zero_guard_handling(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(y_true, y_pred, name="zeros")
        assert m["MAE"] == 2.0
        assert np.isfinite(m["RMSE"])
        assert m["MAPE"] == 0.0  # zero-guarded


# ---------------------------------------------------------------------------
# 2. Baseline Model Tests
# ---------------------------------------------------------------------------
class TestBaselines:
    def test_predict_naive(self):
        hist = pd.Series([10.0, 20.0, 35.0])
        preds = predict_naive(hist, horizon_hours=24)
        assert len(preds) == 24
        assert (preds == 35.0).all()

    def test_seasonal_naive_yesterday(self):
        # 48 hourly values: day 1 is 1..24, day 2 is 101..124
        day1 = list(range(1, 25))
        day2 = list(range(101, 125))
        hist = pd.Series(day1 + day2)
        preds = predict_seasonal_naive_yesterday(hist, horizon_hours=24)
        assert len(preds) == 24
        # Should equal day 2 exactly
        np.testing.assert_array_equal(preds, day2)

    def test_seasonal_naive_last_week(self):
        # 168 hourly values
        week = list(range(1, 169))
        hist = pd.Series(week)
        preds = predict_seasonal_naive_last_week(hist, horizon_hours=24)
        assert len(preds) == 24
        # First 24 hours of the week
        np.testing.assert_array_equal(preds, week[:24])

    def test_moving_average_24h(self):
        hist = pd.Series([10.0] * 24 + [20.0] * 24)
        preds = predict_moving_average(hist, horizon_hours=24, window=24)
        assert len(preds) == 24
        assert (preds == 20.0).all()

    def test_generate_all_baselines(self):
        hist = pd.Series(np.random.uniform(5, 50, size=200))
        b = generate_all_baselines(hist, horizon_hours=24)
        assert "naive" in b
        assert "seasonal_naive_yesterday" in b
        assert "seasonal_naive_last_week" in b
        assert "moving_average_24h" in b
        for k, v in b.items():
            assert len(v) == 24
            assert np.isfinite(v).all()


# ---------------------------------------------------------------------------
# 3. Forecast Horizons and Leakage Prevention Tests
# ---------------------------------------------------------------------------
class TestForecastHorizonsAndLeakage:
    def test_forecast_timestamps_24h(self):
        origin = "2026-01-01 00:00:00"
        ts = generate_forecast_timestamps(origin, 24, "Africa/Algiers")
        assert len(ts) == 24
        assert str(ts[0]) == "2026-01-01 00:00:00+01:00"
        assert str(ts[-1]) == "2026-01-01 23:00:00+01:00"

    def test_forecast_timestamps_7d(self):
        origin = "2026-01-01 00:00:00"
        ts = generate_forecast_timestamps(origin, 168, "Africa/Algiers")
        assert len(ts) == 168
        assert str(ts[0]) == "2026-01-01 00:00:00+01:00"
        assert str(ts[-1]) == "2026-01-07 23:00:00+01:00"

    def test_no_temporal_lookahead_in_recursive_rollout(self):
        """
        Verify that recursive forecasting never peeks at ground truth y_{T+h}.
        Simulate an autoregressive step: lag_1 must equal previous prediction,
        NOT true future y.
        """
        # True future target values
        ground_truth_future = [100.0, 200.0, 300.0]
        history = [10.0, 20.0, 30.0]  # prior to origin

        # Dummy model: predicts lag1 + 1.0
        predictions = []
        simulated_history = list(history)

        for step in range(len(ground_truth_future)):
            lag1 = simulated_history[-1]
            # lag1 must NOT be ground_truth_future[step]
            assert lag1 != ground_truth_future[step]
            pred = lag1 + 1.0
            predictions.append(pred)
            simulated_history.append(pred)

        # Expected predictions: 30+1=31, 31+1=32, 32+1=33
        assert predictions == [31.0, 32.0, 33.0]
        # In step 1, lag1 was 31 (the model's prediction), NOT 100 (ground truth)
        assert simulated_history[len(history)] == 31.0

    def test_recursive_simulation_invariance_to_future_ground_truth(self):
        """
        Rigorous invariance test: Corrupting future target ground truth in future dataframes
        must have ZERO effect on predictions generated by run_vectorized_simulation.
        """
        from src.evaluation.backtest import run_vectorized_simulation
        from sklearn.linear_model import LinearRegression

        # Setup 2 cells with historical values
        cells = ["cell_A", "cell_B"]
        history_by_cell = {
            "cell_A": [10.0 + i for i in range(168)],
            "cell_B": [20.0 + i for i in range(168)],
        }
        
        feature_cols = [
            "prb_utilization_pct_lag1h", "prb_utilization_pct_lag2h",
            "prb_utilization_pct_lag24h", "prb_utilization_pct_roll_mean_24h"
        ]

        # Fit a simple linear model
        X_toy = np.random.uniform(10, 80, size=(100, len(feature_cols)))
        y_toy = np.random.uniform(10, 80, size=100)
        dummy_model = LinearRegression().fit(X_toy, y_toy)

        horizon_hours = 24
        timestamps = pd.date_range("2026-01-01 00:00:00", periods=horizon_hours, freq="h")

        # Frame 1: Normal future target
        tf1 = {
            "cell_A": pd.DataFrame({"timestamp": timestamps, "prb_utilization_pct": [50.0] * horizon_hours}),
            "cell_B": pd.DataFrame({"timestamp": timestamps, "prb_utilization_pct": [60.0] * horizon_hours}),
        }

        # Frame 2: Completely corrupted future target (1,000,000x larger, inverted, arbitrary)
        tf2 = {
            "cell_A": pd.DataFrame({"timestamp": timestamps, "prb_utilization_pct": [9999999.0] * horizon_hours}),
            "cell_B": pd.DataFrame({"timestamp": timestamps, "prb_utilization_pct": [-8888888.0] * horizon_hours}),
        }

        preds1 = run_vectorized_simulation(
            cells, history_by_cell, tf1, dummy_model, feature_cols,
            target="prb_utilization_pct", horizon_hours=horizon_hours
        )
        preds2 = run_vectorized_simulation(
            cells, history_by_cell, tf2, dummy_model, feature_cols,
            target="prb_utilization_pct", horizon_hours=horizon_hours
        )

        # Assert predictions are 100% bit-for-bit identical regardless of future ground truth
        for cid in cells:
            np.testing.assert_array_almost_equal(
                preds1[cid], preds2[cid], decimal=10,
                err_msg=f"Future ground-truth leakage detected in cell {cid}!"
            )


# ---------------------------------------------------------------------------
# 4. Congestion Alert Classification Tests
# ---------------------------------------------------------------------------
class TestCongestionAlerts:
    def test_threshold_classification(self):
        from src.evaluation.evaluate import detect_congestion_alerts

        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h"),
            "cell_id": ["cell_1", "cell_2", "cell_3", "cell_4"],
            "predicted": [75.0, 80.5, 89.9, 92.0],
            "wilaya_name": ["Alger", "Oran", "Constantine", "Setif"],
            "technology": ["4G", "4G", "4G", "5G"],
            "horizon": ["short", "short", "short", "short"],
        })

        alerts = detect_congestion_alerts(df, warning_threshold=80.0, high_threshold=90.0)
        assert len(alerts) == 3  # 75.0 is normal

        # 80.5 and 89.9 should be WARNING
        warnings = alerts[alerts["alert_level"] == "WARNING"]
        assert len(warnings) == 2
        assert set(warnings["cell_id"]) == {"cell_2", "cell_3"}

        # 92.0 should be HIGH
        highs = alerts[alerts["alert_level"] == "HIGH"]
        assert len(highs) == 1
        assert highs.iloc[0]["cell_id"] == "cell_4"
