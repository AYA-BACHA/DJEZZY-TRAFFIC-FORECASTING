"""
src/visualization/plots.py
Static visualization generation with Djezzy red/charcoal/white visual identity.
Saves HTML figures to reports/figures/.

Run as:  python -m src.visualization.plots
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Djezzy visual identity constants
DJEZZY_RED = "#E02B20"
DJEZZY_DARK = "#1E2024"
DJEZZY_GRAY = "#5A626A"
DJEZZY_LIGHT = "#F4F6F8"
WARNING_AMBER = "#FF9800"
CRITICAL_RED = "#D32F2F"
DJEZZY_PALETTE = ["#E02B20", "#2B2F36", "#616A75", "#9AA3AB"]


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_plots(config_path: str = "config/config.yaml") -> None:
    """Generate all static analysis and forecast plots using Djezzy styling."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.error("plotly not installed — skipping plots")
        return

    cfg = load_config(config_path)
    figs_dir = Path(cfg["paths"]["figures_dir"])
    figs_dir.mkdir(parents=True, exist_ok=True)
    processed_path = cfg["paths"]["processed_data"]
    forecasts_dir = Path(cfg["paths"]["forecasts_dir"])

    log.info("=" * 60)
    log.info("Djezzy Traffic Forecasting — Plot Generation (Djezzy Theme)")
    log.info("=" * 60)

    # 1. Load cleaned data
    log.info("Loading cleaned data...")
    df = pd.read_parquet(processed_path, engine="pyarrow")
    log.info("Loaded %d rows", len(df))

    # 2. Hourly DL traffic heatmap (hour vs day-of-week)
    log.info("Generating hourly traffic heatmap...")
    df_4g = df[df["technology"] == "4G"].copy()
    df_4g["hour"] = df_4g["timestamp"].dt.hour
    df_4g["day_of_week"] = df_4g["timestamp"].dt.dayofweek
    pivot = (
        df_4g.groupby(["day_of_week", "hour"], observed=True)["dl_traffic_volume_gb"]
        .mean()
        .reset_index()
        .pivot(index="day_of_week", columns="hour", values="dl_traffic_volume_gb")
    )
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri (Weekend)", "Sat (Weekend)", "Sun"]
    fig_heatmap = px.imshow(
        pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="Avg DL (GB)"),
        x=list(range(24)),
        y=day_labels[: len(pivot)],
        color_continuous_scale="Reds",
        title="Average DL Traffic Volume by Hour & Day of Week (4G) [SYNTHETIC DATA]",
    )
    fig_heatmap.update_layout(width=900, height=450, template="plotly_white")
    fig_heatmap.write_html(str(figs_dir / "heatmap_dl_traffic_hour_dow.html"))
    log.info("Saved heatmap.")

    # 3. Technology split — DL traffic distribution
    log.info("Generating technology distribution plot...")
    tech_dl = (
        df.groupby("technology", observed=True)["dl_traffic_volume_gb"].mean().reset_index()
    )
    fig_tech = px.bar(
        tech_dl, x="technology", y="dl_traffic_volume_gb",
        color="technology",
        title="Average DL Traffic by Technology [SYNTHETIC DATA]",
        labels={"dl_traffic_volume_gb": "Avg DL Traffic (GB/h)", "technology": "Technology"},
        color_discrete_sequence=DJEZZY_PALETTE,
    )
    fig_tech.update_layout(template="plotly_white")
    fig_tech.write_html(str(figs_dir / "bar_dl_traffic_by_technology.html"))
    log.info("Saved technology distribution.")

    # 4. PRB utilisation boxplot by area type
    log.info("Generating PRB by area type boxplot...")
    fig_prb_area = px.box(
        df.sample(min(50000, len(df)), random_state=42),
        x="area_type", y="prb_utilization_pct",
        color="area_type",
        title="PRB Utilisation by Area Type [SYNTHETIC DATA]",
        labels={"prb_utilization_pct": "PRB Utilisation (%)", "area_type": "Area Type"},
        color_discrete_sequence=["#D32F2F", "#1E2024", "#5A626A", "#8C95A0", "#B0B8C0"],
    )
    fig_prb_area.add_hline(y=80, line_dash="dash", line_color=WARNING_AMBER, annotation_text="Warning (80%)")
    fig_prb_area.add_hline(y=90, line_dash="dash", line_color=CRITICAL_RED, annotation_text="High (90%)")
    fig_prb_area.update_layout(template="plotly_white")
    fig_prb_area.write_html(str(figs_dir / "boxplot_prb_by_area_type.html"))
    log.info("Saved PRB boxplot.")

    # 5. Monthly traffic trend
    log.info("Generating monthly traffic trend...")
    df["month_period"] = df["timestamp"].dt.to_period("M").astype(str)
    monthly = (
        df.groupby("month_period")
        .agg(dl_total=("dl_traffic_volume_gb", "sum"),
             prb_mean=("prb_utilization_pct", "mean"))
        .reset_index()
    )
    fig_monthly = make_subplots(specs=[[{"secondary_y": True}]])
    fig_monthly.add_trace(
        go.Bar(x=monthly["month_period"], y=monthly["dl_total"],
               name="Total DL Traffic (GB)", marker_color=DJEZZY_RED),
        secondary_y=False,
    )
    fig_monthly.add_trace(
        go.Scatter(x=monthly["month_period"], y=monthly["prb_mean"],
                   name="Avg PRB (%)", line=dict(color=DJEZZY_DARK, width=2.5)),
        secondary_y=True,
    )
    fig_monthly.update_layout(
        title="Monthly DL Traffic & PRB Utilisation [SYNTHETIC DATA]",
        xaxis_title="Month",
        legend=dict(x=0.01, y=0.99),
        width=1000, height=500,
        template="plotly_white",
    )
    fig_monthly.update_yaxes(title_text="Total DL Traffic (GB)", secondary_y=False)
    fig_monthly.update_yaxes(title_text="Avg PRB (%)", secondary_y=True)
    fig_monthly.write_html(str(figs_dir / "trend_monthly_dl_prb.html"))
    log.info("Saved monthly trend.")

    # 6. Ramadan vs normal traffic
    log.info("Generating Ramadan impact plot...")
    df["is_ramadan"] = 0
    for r in cfg["calendar"]["ramadan"]:
        mask = (df["timestamp"].dt.date >= pd.Timestamp(r["start"]).date()) & \
               (df["timestamp"].dt.date <= pd.Timestamp(r["end"]).date())
        df.loc[mask, "is_ramadan"] = 1

    df["hour"] = df["timestamp"].dt.hour
    ram_hourly = (
        df.groupby(["is_ramadan", "hour"])["dl_traffic_volume_gb"].mean().reset_index()
    )
    ram_hourly["period"] = ram_hourly["is_ramadan"].map({0: "Non-Ramadan", 1: "Ramadan"})

    fig_ramadan = px.line(
        ram_hourly, x="hour", y="dl_traffic_volume_gb",
        color="period",
        title="Hourly DL Traffic: Ramadan vs Non-Ramadan [SYNTHETIC DATA]",
        labels={"dl_traffic_volume_gb": "Avg DL Traffic (GB/h)", "hour": "Hour of Day"},
        line_shape="spline",
        color_discrete_sequence=[DJEZZY_DARK, DJEZZY_RED],
    )
    fig_ramadan.update_layout(template="plotly_white")
    fig_ramadan.write_html(str(figs_dir / "ramadan_hourly_traffic.html"))
    log.info("Saved Ramadan plot.")

    # 7. Forecast visualization
    log.info("Generating forecast plots...")
    for target in cfg["forecast"]["targets"]:
        for horizon_name in ["24h", "7d", "short", "long"]:
            forecast_path = forecasts_dir / f"forecast_{target}_{horizon_name}.parquet"
            if not forecast_path.exists():
                continue
            fdf = pd.read_parquet(forecast_path, engine="pyarrow")
            if "cell_id" in fdf.columns:
                fdf_agg = fdf.groupby("timestamp")["predicted"].agg(["mean", "max", "min"]).reset_index()
            else:
                fdf_agg = fdf[["timestamp", "predicted"]].copy()
                fdf_agg.columns = ["timestamp", "mean"]
                fdf_agg["max"] = fdf_agg["mean"]
                fdf_agg["min"] = fdf_agg["mean"]

            fdf_agg = fdf_agg.sort_values("timestamp")
            line_color = DJEZZY_RED if "dl" in target else DJEZZY_DARK

            fig_fore = go.Figure()
            fig_fore.add_trace(go.Scatter(
                x=fdf_agg["timestamp"], y=fdf_agg["mean"],
                name=f"Predicted {target} (mean)",
                line=dict(color=line_color, width=2.5),
            ))
            fig_fore.add_trace(go.Scatter(
                x=pd.concat([fdf_agg["timestamp"], fdf_agg["timestamp"][::-1]]),
                y=pd.concat([fdf_agg["max"], fdf_agg["min"][::-1]]),
                fill="toself",
                fillcolor="rgba(224, 43, 32, 0.12)" if "dl" in target else "rgba(30, 32, 36, 0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Min/Max range",
            ))
            if "prb" in target:
                fig_fore.add_hline(y=80, line_dash="dash", line_color=WARNING_AMBER, annotation_text="Warning (80%)")
                fig_fore.add_hline(y=90, line_dash="dash", line_color=CRITICAL_RED, annotation_text="High (90%)")

            target_label = "DL Traffic (GB/h)" if "dl" in target else "PRB Utilisation (%)"
            fig_fore.update_layout(
                title=f"Forecast — {target_label} — {horizon_name} horizon [SYNTHETIC DATA]",
                xaxis_title="Timestamp (Africa/Algiers)",
                yaxis_title=target_label,
                width=1000, height=450,
                template="plotly_white",
            )
            fig_fore.write_html(str(figs_dir / f"forecast_{target}_{horizon_name}.html"))
            log.info("Saved forecast plot for %s/%s", target, horizon_name)

    # 8. Wilaya heatmap
    log.info("Generating wilaya traffic summary...")
    wilaya_summary = (
        df.groupby("wilaya_name", observed=True).agg(
            avg_dl=("dl_traffic_volume_gb", "mean"),
            avg_prb=("prb_utilization_pct", "mean"),
            congestion_rate=("congestion_flag", "mean"),
        ).reset_index().sort_values("avg_dl", ascending=False)
    )
    fig_wilaya = px.bar(
        wilaya_summary.head(15), x="wilaya_name", y="avg_dl",
        color="avg_prb",
        title="Top 15 Wilayas by Average DL Traffic [SYNTHETIC DATA]",
        labels={"avg_dl": "Avg DL (GB/h)", "wilaya_name": "Wilaya", "avg_prb": "Avg PRB (%)"},
        color_continuous_scale="Reds",
    )
    fig_wilaya.update_layout(template="plotly_white")
    fig_wilaya.write_html(str(figs_dir / "bar_wilaya_dl_traffic.html"))
    log.info("Saved wilaya traffic summary.")

    log.info("All plots generated and saved to %s", figs_dir)


if __name__ == "__main__":
    generate_plots()
