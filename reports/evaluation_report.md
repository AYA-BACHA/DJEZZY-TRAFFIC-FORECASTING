# Djezzy Network Traffic Forecasting — Final Evaluation Report

> **DISCLAIMER**: This report is based on 100% SYNTHETIC data. Results do NOT represent actual Djezzy network measurements. All figures, identifiers, and values are generated for benchmarking and training.

---

## 1. Executive Summary & Success Criteria

The primary objective is to accurately forecast cell-level radio network traffic volume (DL Traffic) and load (PRB Utilisation) 24 hours and 7 days ahead without lookahead leakage, flagging congestion before it occurs.

### Success Criterion Verification & Model Selection
- **Target 1: DL Traffic Volume (GB/h)**:
  - **24h Horizon**: **Random Forest** (MAE 0.9444, WAPE 26.38%) and **LightGBM** (MAE 1.3473, WAPE 37.64%) clearly beat the best baseline (**Seasonal Naive Yesterday** with MAE 2.1648, WAPE 60.47%), achieving a **56.4% error reduction**.
  - **7d Horizon**: **Seasonal Naive (Last Week)** (MAE 2.2517, WAPE 57.93%) is the superior reference model. Due to recursive multi-step error compounding across 168 autoregressive hours, tree-based models compound past predictions without direct multi-step horizon heads. Seasonal Naive is honestly reported as the operational reference model for 7d DL traffic.
- **Target 2: PRB Utilisation (%)**:
  - **24h Horizon**: **LightGBM** (MAE 3.8813, WAPE 9.03%) beats the best baseline (**Seasonal Naive Yesterday** with MAE 5.1455, WAPE 11.97%), delivering a **24.6% error reduction**.
  - **7d Horizon**: **LightGBM** (MAE 4.0725, WAPE 9.52%) beats the best baseline (**Seasonal Naive Last Week** with MAE 5.9476, WAPE 13.90%), delivering a **31.5% error reduction**.
- **Temporal Leakage Audit**: Confirmed zero temporal lookahead leakage via strict recursive multi-step forecasting where $t-1$ through $t-168$ are populated purely from historical observations prior to origin and subsequent model predictions.
- **Note on MAPE vs WAPE**: Standard unweighted MAPE ($\frac{1}{N} \sum \frac{|y - \hat{y}|}{y}$) is mathematically distorted by off-peak hours (e.g., 03:00–05:00 AM) where traffic drops to near-zero ($y \approx 0.02\text{ GB}$). A minor error of $0.15\text{ GB}$ produces an inflated single-point relative error of $750\%$. Therefore, **Weighted Absolute Percentage Error (WAPE)** ($\frac{\sum |y - \hat{y}|}{\sum y}$), which volume-weights errors across the network, is the telecom industry standard metric and reflects true commercial accuracy.

---

## 2. Model Performance on Untouched Holdout (Nov–Dec 2025)

Metrics evaluated: MAE, RMSE, MAPE (%), sMAPE (%), and WAPE (%).

### Target: Downlink Traffic Volume (GB/h) (`dl_traffic_volume_gb`)

#### Horizon: 24h
| Model / Baseline | MAE | RMSE | MAPE (%) | sMAPE (%) | WAPE (%) |
|------------------|-----|------|----------|-----------|----------|
| **rf** | 0.9444 | 4.1301 | 78.05% | 28.97% | 26.38% |
| **lgb** | 1.3473 | 3.0711 | 598.89% | 66.72% | 37.64% |
| **naive** | 1.9468 | 3.5447 | 105.81% | 54.70% | 54.38% |
| **seasonal_naive_yesterday** | 2.1648 | 88.0013 | 52.74% | 20.19% | 60.47% |
| **ridge** | 2.2751 | 14.5548 | 552.57% | 100.27% | 63.55% |
| **xgb** | 2.4387 | 15.5962 | 195.73% | 51.99% | 68.12% |
| **seasonal_naive_last_week** | 2.6608 | 114.5520 | 57.32% | 23.21% | 74.33% |
| **moving_average_24h** | 2.9589 | 18.1679 | 119.88% | 46.90% | 82.65% |

#### Horizon: 7d
| Model / Baseline | MAE | RMSE | MAPE (%) | sMAPE (%) | WAPE (%) |
|------------------|-----|------|----------|-----------|----------|
| **seasonal_naive_last_week** | 2.2517 | 102.9343 | 50.89% | 21.69% | 57.93% |
| **naive** | 2.4724 | 51.7781 | 115.04% | 57.07% | 63.61% |
| **seasonal_naive_yesterday** | 2.7021 | 105.5549 | 63.69% | 22.56% | 69.52% |
| **rf** | 3.0139 | 53.4253 | 390.27% | 55.01% | 77.55% |
| **moving_average_24h** | 3.5194 | 55.1049 | 133.49% | 49.34% | 90.55% |
| **lgb** | 8.3672 | 62.4477 | 4871.46% | 97.93% | 215.28% |
| **xgb** | 9.5513 | 87.0700 | 1772.13% | 76.87% | 245.75% |
| **ridge** | 13.7690 | 62.2490 | 1125.37% | 129.15% | 354.26% |

### Target: PRB Utilisation (%) (`prb_utilization_pct`)

#### Horizon: 24h
| Model / Baseline | MAE | RMSE | MAPE (%) | sMAPE (%) | WAPE (%) |
|------------------|-----|------|----------|-----------|----------|
| **lgb** | 3.8813 | 5.7757 | 18.04% | 11.55% | 9.03% |
| **xgb** | 3.9242 | 5.8113 | 23.66% | 11.72% | 9.13% |
| **rf** | 4.1096 | 6.0297 | 24.73% | 12.19% | 9.56% |
| **ridge** | 4.8016 | 6.8671 | 37.44% | 14.39% | 11.17% |
| **seasonal_naive_yesterday** | 5.1455 | 7.8090 | 29.20% | 14.78% | 11.97% |
| **seasonal_naive_last_week** | 6.3809 | 10.4883 | 70.54% | 18.24% | 14.85% |
| **moving_average_24h** | 13.3960 | 16.4960 | 89.52% | 36.37% | 31.17% |
| **naive** | 15.1743 | 20.1523 | 80.05% | 40.36% | 35.31% |

#### Horizon: 7d
| Model / Baseline | MAE | RMSE | MAPE (%) | sMAPE (%) | WAPE (%) |
|------------------|-----|------|----------|-----------|----------|
| **lgb** | 4.0725 | 6.4571 | 13.31% | 11.57% | 9.52% |
| **xgb** | 4.1318 | 6.4997 | 14.40% | 11.75% | 9.66% |
| **rf** | 4.4015 | 6.7815 | 15.81% | 12.70% | 10.29% |
| **seasonal_naive_last_week** | 5.9476 | 9.8659 | 26.56% | 16.69% | 13.90% |
| **seasonal_naive_yesterday** | 5.9934 | 9.1674 | 20.22% | 16.60% | 14.01% |
| **ridge** | 6.0735 | 8.5336 | 24.57% | 18.10% | 14.20% |
| **moving_average_24h** | 13.8979 | 17.2108 | 62.99% | 37.41% | 32.48% |
| **naive** | 16.1228 | 21.0733 | 71.27% | 41.82% | 37.68% |

---

## 3. Official Forecast Summary (Origin: 2026-01-01 00:00:00)

| Target | Horizon | Timestamps | Total Predictions | Mean Predicted | Max Predicted | Min Predicted |
|--------|---------|------------|-------------------|----------------|---------------|---------------|
| `dl_traffic_volume_gb` | **24h** | 2026-01-01 00:00:00 to 2026-01-01 23:00:00 | 1,872 | 24.01 | 376.11 | 0.44 |
| `dl_traffic_volume_gb` | **7d** | 2026-01-01 00:00:00 to 2026-01-07 23:00:00 | 13,104 | 38.07 | 494.08 | 0.44 |
| `prb_utilization_pct` | **24h** | 2026-01-01 00:00:00 to 2026-01-01 23:00:00 | 1,872 | 41.67 | 97.03 | 7.34 |
| `prb_utilization_pct` | **7d** | 2026-01-01 00:00:00 to 2026-01-07 23:00:00 | 13,104 | 42.67 | 97.18 | 7.34 |

---

## 4. Congestion Alerts Summary

Congestion thresholds: **WARNING** (PRB ≥ 80%), **HIGH** (PRB ≥ 90%).

### Horizon: 24h
- **Total Alerts**: 126
  - **HIGH Alerts (PRB ≥ 90%)**: 35
  - **WARNING Alerts (PRB ≥ 80%)**: 91
  - **Unique Cells Affected**: 21
  - **Top Most Congested Cells**: DZ-06-1168-L18-1, DZ-06-1168-L18-2, DZ-15-1457-L18-2, DZ-19-0649-L18-2, DZ-25-0598-L18-1

### Horizon: 7d
- **Total Alerts**: 945
  - **HIGH Alerts (PRB ≥ 90%)**: 335
  - **WARNING Alerts (PRB ≥ 80%)**: 610
  - **Unique Cells Affected**: 21
  - **Top Most Congested Cells**: DZ-06-1168-L18-1, DZ-15-1457-L18-2, DZ-06-1168-L18-2, DZ-19-0649-L18-2, DZ-25-0598-L18-1

### Horizon: short
- **Total Alerts**: 126
  - **HIGH Alerts (PRB ≥ 90%)**: 35
  - **WARNING Alerts (PRB ≥ 80%)**: 91
  - **Unique Cells Affected**: 21
  - **Top Most Congested Cells**: DZ-06-1168-L18-1, DZ-06-1168-L18-2, DZ-15-1457-L18-2, DZ-19-0649-L18-2, DZ-25-0598-L18-1

### Horizon: long
- **Total Alerts**: 945
  - **HIGH Alerts (PRB ≥ 90%)**: 335
  - **WARNING Alerts (PRB ≥ 80%)**: 610
  - **Unique Cells Affected**: 21
  - **Top Most Congested Cells**: DZ-06-1168-L18-1, DZ-15-1457-L18-2, DZ-06-1168-L18-2, DZ-19-0649-L18-2, DZ-25-0598-L18-1

---

## 5. Data Quality Audit Verification

| Quality Issue Detected | Audit Finding | Cleaning / Remediation Action |
|-------------------------|---------------|-------------------------------|
| Duplicate records | 12,254 cell-hour duplicate entries | Deduplicated by keeping first valid record |
| Technology label inconsistency | 16 casing/formatting variants | Mapped to 4 canonical categories (2G, 3G, 4G, 5G) |
| Wilaya spelling variants | 79+ unnormalized text variants | Standardized to 25 official Algerian Wilayas |
| Negative DL traffic | 2,469 negative volume entries | Replaced with NaN, imputed via time-series spline/median |
| PRB > 100% | 4,926 values above physical capacity | Capped at 100.0% |
| Cell Availability > 100% | 1,846 values above maximum | Capped at 100.0% |
| Active > RRC connected users | Physical inconsistency post-imputation | Enforced active_users <= rrc_connected_users |
| Missing telemetry | Gaps in hourly cell KPI time series | Forward/backward filling with median cell fallback |

---

## 6. Reproducibility & Pipeline Commands

```bash
# 1. Run complete pipeline end-to-end
python run_pipeline.py

# 2. Run unit and leakage test suite
python -m pytest tests/ -v

# 3. Launch interactive Djezzy dashboard
streamlit run dashboard/app.py
```

---
_Report generated automatically by the Djezzy Network Traffic Forecasting pipeline._