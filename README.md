# Djezzy Network Traffic Forecasting

Hourly cell-level radio KPI forecasting (24h and 7-day horizons) with congestion alerting.
Internship project — Data & AI programme 2026. **Dataset is 100% synthetic.**

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the raw CSV at `data/raw/djezzy_network_traffic_hourly_2024_2025.csv` (not committed to Git).

## Pipeline

```bash
python -m src.data.clean            # raw -> data/processed/cleaned_hourly.parquet
python -m src.features.build_features
python -m src.models.train
python -m src.models.predict        # forecasts from the 1 Jan 2026 origin
streamlit run dashboard/app.py
```

## Project structure

See `config/config.yaml` for paths and parameters. Reusable code lives in `src/`;
`notebooks/` is for exploration only — code that's needed more than once is moved to `src/`.

## Testing

```bash
pytest tests/
```

## Team

- Supervisor: zakaria sedda
