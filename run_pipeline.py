"""
run_pipeline.py
Master pipeline runner for the Djezzy Traffic Forecasting project.

Executes all stages end-to-end:
  1. Data cleaning
  2. Feature engineering
  3. Model training
  4. Forecast generation
  5. Evaluation & reports
  6. Visualization

Usage:
    python run_pipeline.py [--skip-clean] [--skip-features] [--skip-train]
                           [--skip-forecast] [--skip-eval] [--skip-plots]
"""

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"


def main():
    parser = argparse.ArgumentParser(description="Djezzy Traffic Forecasting Pipeline")
    parser.add_argument("--skip-clean", action="store_true", help="Skip data cleaning")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature engineering")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-forecast", action="store_true", help="Skip forecast generation")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    pipeline_start = time.time()
    log.info("=" * 70)
    log.info("DJEZZY NETWORK TRAFFIC FORECASTING — FULL PIPELINE")
    log.info("=" * 70)

    # ----------------------------------------------------------------
    # Stage 1: Data Cleaning
    # ----------------------------------------------------------------
    if not args.skip_clean:
        log.info("\n[Stage 1/6] DATA CLEANING")
        t0 = time.time()
        from src.data.clean import run_cleaning_pipeline
        run_cleaning_pipeline()
        log.info("[Stage 1] Done in %s\n", elapsed(t0))
    else:
        log.info("[Stage 1] Skipped.")

    # ----------------------------------------------------------------
    # Stage 2: Feature Engineering
    # ----------------------------------------------------------------
    if not args.skip_features:
        log.info("\n[Stage 2/6] FEATURE ENGINEERING")
        t0 = time.time()
        from src.features.build_features import run_feature_pipeline
        run_feature_pipeline()
        log.info("[Stage 2] Done in %s\n", elapsed(t0))
    else:
        log.info("[Stage 2] Skipped.")

    # ----------------------------------------------------------------
    # Stage 3: Model Training
    # ----------------------------------------------------------------
    if not args.skip_train:
        log.info("\n[Stage 3/6] MODEL TRAINING")
        t0 = time.time()
        from src.models.train import run_training_pipeline
        run_training_pipeline()
        log.info("[Stage 3] Done in %s\n", elapsed(t0))
    else:
        log.info("[Stage 3] Skipped.")

    # ----------------------------------------------------------------
    # Stage 4: Forecast Generation
    # ----------------------------------------------------------------
    if not args.skip_forecast:
        log.info("\n[Stage 4/6] FORECAST GENERATION")
        t0 = time.time()
        from src.models.predict import run_prediction_pipeline
        run_prediction_pipeline()
        log.info("[Stage 4] Done in %s\n", elapsed(t0))
    else:
        log.info("[Stage 4] Skipped.")

    # ----------------------------------------------------------------
    # Stage 5: Evaluation & Reports
    # ----------------------------------------------------------------
    if not args.skip_eval:
        log.info("\n[Stage 5/6] EVALUATION & REPORTING")
        t0 = time.time()
        from src.evaluation.evaluate import run_evaluation_pipeline
        run_evaluation_pipeline()
        log.info("[Stage 5] Done in %s\n", elapsed(t0))
    else:
        log.info("[Stage 5] Skipped.")

    # ----------------------------------------------------------------
    # Stage 6: Visualization
    # ----------------------------------------------------------------
    if not args.skip_plots:
        log.info("\n[Stage 6/6] VISUALIZATION")
        t0 = time.time()
        from src.visualization.plots import generate_plots
        generate_plots()
        log.info("[Stage 6] Done in %s\n", elapsed(t0))
    else:
        log.info("[Stage 6] Skipped.")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE in %s", elapsed(pipeline_start))
    log.info("=" * 70)
    log.info("")
    log.info("Outputs:")
    log.info("  Cleaned data:    data/processed/djezzy_cleaned.parquet")
    log.info("  Feature data:    data/processed/djezzy_features.parquet")
    log.info("  Models:          models/")
    log.info("  Forecasts:       reports/forecasts/")
    log.info("  Metrics:         reports/metrics/")
    log.info("  Figures:         reports/figures/")
    log.info("  Report:          reports/evaluation_report.md")
    log.info("")
    log.info("To launch the dashboard:")
    log.info("  streamlit run dashboard/app.py")
    log.info("")
    log.info("⚠️  REMINDER: All data is 100%% SYNTHETIC.")


if __name__ == "__main__":
    main()
