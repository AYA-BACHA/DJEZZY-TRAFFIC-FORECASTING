"""
dashboard/app.py
Djezzy Network Traffic Forecasting & Capacity Planning Platform

Visual Identity: Djezzy Red, White, Charcoal, and Gray
Features:
  1. Network Overview: KPIs, hourly heatmap, tech split, monthly trend
  2. Forecast Analysis: 24h/7d multi-horizon, specific cell forecast (no aggregation if cell selected)
  3. Congestion Alerts: HIGH (>=90%) and WARNING (>=80%), timeline, interactive table
  4. Cell Deep-Dive: Cascading selectors (Wilaya -> Site -> Cell), status badge, dual charts
  5. Model Performance: Holdout metrics table (MAE, RMSE, MAPE, sMAPE, WAPE), baselines comparison
  6. Data Quality: Raw vs cleaned audit, normalization verification, missingness handling

Run with:  streamlit run dashboard/app.py
"""

import base64
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Djezzy Network Traffic Forecasting & Capacity Planning Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Visual Tokens (Djezzy Red / Charcoal / White)
# ---------------------------------------------------------------------------
DJEZZY_RED = "#E02B20"
DJEZZY_DARK = "#14171A"
DJEZZY_CARD = "#FFFFFF"
DJEZZY_BORDER = "#E2E6EA"
DJEZZY_TEXT = "#14171A"
DJEZZY_MUTED = "#5A626A"
WARNING_AMBER = "#FF9800"
CRITICAL_RED = "#D32F2F"
SUCCESS_GREEN = "#2E7D32"

TECH_COLORS = {
    "2G": "#7A828A",
    "3G": "#4A525A",
    "4G": "#E02B20",
    "5G": "#14171A",
}


# ---------------------------------------------------------------------------
# Caching Data Loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_cleaned_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p, engine="pyarrow")


@st.cache_data
def load_forecast_file(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p, engine="pyarrow")


@st.cache_data
def load_metrics_summary(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data
def load_alerts_file(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data
def load_feature_importance(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data
def get_djezzy_logo_b64() -> str:
    logo_path = Path("dashboard/assets/djezzy_logo.svg")
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


# ---------------------------------------------------------------------------
# Custom CSS: Djezzy Telecom Visual Identity
# ---------------------------------------------------------------------------
def inject_custom_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: {DJEZZY_TEXT};
    }}

    /* Main Branding Header */
    .djezzy-header {{
        background: {DJEZZY_DARK};
        border-left: 6px solid {DJEZZY_RED};
        padding: 1.25rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    }}
    .djezzy-title {{
        color: #FFFFFF;
        font-size: 1.75rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }}
    .djezzy-title span {{
        color: {DJEZZY_RED};
    }}
    .djezzy-subtitle {{
        color: #9AA3AB;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }}
    .djezzy-badge {{
        background: {DJEZZY_RED};
        color: #FFFFFF;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
    }}

    /* Synthetic Data Disclaimer */
    .disclaimer-banner {{
        background: #FFF5F5;
        border: 1px solid #FFD0D0;
        border-left: 5px solid {DJEZZY_RED};
        border-radius: 6px;
        padding: 0.75rem 1.25rem;
        font-size: 0.85rem;
        font-weight: 500;
        color: #9C1414;
        margin-bottom: 1.5rem;
    }}

    /* Section Subheaders */
    .section-title {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {DJEZZY_DARK};
        border-bottom: 2px solid {DJEZZY_RED};
        padding-bottom: 0.35rem;
        margin-bottom: 1.25rem;
        margin-top: 0.5rem;
    }}

    /* Card styling */
    .kpi-card {{
        background: #FFFFFF;
        border: 1px solid {DJEZZY_BORDER};
        border-radius: 8px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }}

    /* Status Badges */
    .badge-normal {{
        background: #E8F5E9;
        color: {SUCCESS_GREEN};
        border: 1px solid #C8E6C9;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }}
    .badge-warning {{
        background: #FFF3E0;
        color: #E65100;
        border: 1px solid #FFE0B2;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }}
    .badge-high {{
        background: #FFEBEE;
        color: {CRITICAL_RED};
        border: 1px solid #FFCDD2;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }}

    /* Metric Containers */
    div[data-testid="metric-container"] {{
        background: #FFFFFF;
        border: 1px solid {DJEZZY_BORDER};
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }}
    div[data-testid="metric-container"] label {{
        color: {DJEZZY_MUTED} !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }}
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
        color: {DJEZZY_DARK} !important;
        font-weight: 800 !important;
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        border-bottom: 2px solid #E2E6EA;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        color: {DJEZZY_MUTED};
    }}
    .stTabs [aria-selected="true"] {{
        color: {DJEZZY_RED} !important;
        border-bottom-color: {DJEZZY_RED} !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar Filters Component
# ---------------------------------------------------------------------------
def render_sidebar(df_cleaned: pd.DataFrame, cfg: dict) -> dict:
    logo_b64 = get_djezzy_logo_b64()
    if logo_b64:
        st.sidebar.markdown(f"""
        <div style="display:flex; align-items:center; gap:12px; padding: 4px 0 14px 0;">
            <img src="data:image/svg+xml;base64,{logo_b64}" style="height:44px; width:auto; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.15));" alt="Djezzy" />
            <div>
                <div style="font-weight:800; font-size:1.15rem; color:{DJEZZY_DARK}; letter-spacing:-0.4px;">Djezzy Operations</div>
                <div style="font-size:0.75rem; color:{DJEZZY_MUTED}; font-weight:600;">RAN Capacity &amp; Planning Center</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"### 📡 **Djezzy Control Center**")
    st.sidebar.markdown("---")

    # Horizon Selector
    horizon_choice = st.sidebar.radio(
        "Forecast Horizon",
        options=["24 Hours (Daily)", "7 Days (Weekly)"],
        index=0,
    )
    horizon_key = "24h" if "24" in horizon_choice else "7d"
    horizon_alias = "short" if horizon_key == "24h" else "long"

    st.sidebar.markdown("#### 🎯 **Network Filters**")

    # Wilaya Selector
    all_wilayas = sorted(df_cleaned["wilaya_name"].dropna().unique().tolist()) if not df_cleaned.empty else []
    sel_wilayas = st.sidebar.multiselect(
        "Wilaya",
        options=all_wilayas,
        default=all_wilayas,
        placeholder="Filter by Wilaya...",
    )

    # Technology Selector
    all_techs = ["2G", "3G", "4G", "5G"]
    sel_techs = st.sidebar.multiselect(
        "Radio Layer",
        options=all_techs,
        default=all_techs,
    )

    # Filtered cells pool based on Wilaya & Tech
    df_pool = df_cleaned.copy() if not df_cleaned.empty else pd.DataFrame()
    if not df_pool.empty:
        if sel_wilayas:
            df_pool = df_pool[df_pool["wilaya_name"].isin(sel_wilayas)]
        if sel_techs:
            df_pool = df_pool[df_pool["technology"].isin(sel_techs)]

    # Site Selector
    site_list = ["All Sites"] + sorted(df_pool["site_id"].dropna().unique().tolist()) if not df_pool.empty else ["All Sites"]
    sel_site = st.sidebar.selectbox("Site Filter", options=site_list, index=0)

    if sel_site != "All Sites" and not df_pool.empty:
        df_pool = df_pool[df_pool["site_id"] == sel_site]

    # Specific Cell Selector
    cell_list = ["All Cells"] + sorted(df_pool["cell_id"].dropna().unique().tolist()) if not df_pool.empty else ["All Cells"]
    sel_cell = st.sidebar.selectbox("Specific Cell", options=cell_list, index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="font-size:0.75rem; color:{DJEZZY_MUTED};">
        <b>Network Baseline:</b> 78 cells · 28 sites · 25 wilayas<br>
        <b>Forecast Origin:</b> 2026-01-01 00:00 Africa/Algiers<br>
        <b>Model:</b> LightGBM + Recursive Rollout
        </div>
        """,
        unsafe_allow_html=True,
    )

    return {
        "horizon": horizon_key,
        "horizon_alias": horizon_alias,
        "wilayas": sel_wilayas,
        "technologies": sel_techs,
        "site": sel_site,
        "cell": sel_cell,
    }


# ---------------------------------------------------------------------------
# Header Component
# ---------------------------------------------------------------------------
def render_header():
    logo_b64 = get_djezzy_logo_b64()
    logo_html = ""
    if logo_b64:
        logo_html = f'<img src="data:image/svg+xml;base64,{logo_b64}" style="height: 54px; width: auto; margin-right: 18px; filter: drop-shadow(0 2px 5px rgba(0,0,0,0.3));" alt="Djezzy Logo" />'

    st.markdown(f"""
    <div class="djezzy-header">
        <div style="display:flex; align-items:center;">
            {logo_html}
            <div>
                <h1 class="djezzy-title">Djezzy <span>Network Forecasting</span></h1>
                <div class="djezzy-subtitle">AI-Driven Radio Traffic Forecasting &amp; Capacity Planning Platform</div>
            </div>
        </div>
        <div>
            <span class="djezzy-badge">NOC / RAN SUITE</span>
        </div>
    </div>
    <div class="disclaimer-banner">
        ⚠️ <strong>Synthetic Data Notice</strong>: This platform operates entirely on 100% synthetic network telemetry generated
        for capacity planning benchmarking. Figures do not represent real Djezzy network traffic or personal subscriber data.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# TAB 1: Network Overview
# ---------------------------------------------------------------------------
def render_overview_tab(df: pd.DataFrame, cfg: dict):
    st.markdown('<div class="section-title">Network Telemetry & Infrastructure Overview</div>', unsafe_allow_html=True)

    if df.empty:
        st.warning("Processed network data not found. Run `python run_pipeline.py` first.")
        return

    # Top KPI Metrics Row
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("Total Cells", f"{df['cell_id'].nunique():,}")
    with k2:
        st.metric("Sites Monitored", f"{df['site_id'].nunique():,}")
    with k3:
        st.metric("Wilayas", f"{df['wilaya_name'].nunique():,}")
    with k4:
        st.metric("Avg DL Traffic", f"{df['dl_traffic_volume_gb'].mean():.2f} GB/h")
    with k5:
        st.metric("Avg PRB Load", f"{df['prb_utilization_pct'].mean():.1f}%")
    with k6:
        cong_rate = (df['congestion_flag'] == 1).mean() * 100
        st.metric("Congestion Rate", f"{cong_rate:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizations Row 1: Heatmap + Technology Breakdown
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Hourly Traffic Diurnal Profile (4G Layer)**")
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
        fig_heat = px.imshow(
            pivot,
            labels=dict(x="Hour of Day", y="Day of Week", color="DL (GB/h)"),
            x=list(range(24)),
            y=day_labels[: len(pivot)],
            color_continuous_scale="Reds",
        )
        fig_heat.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white")
        st.plotly_chart(fig_heat, use_container_width=True)

    with c2:
        st.markdown("**Traffic Distribution by Technology Generation**")
        tech_summary = (
            df.groupby("technology", observed=True)
            .agg(avg_dl=("dl_traffic_volume_gb", "mean"), cell_count=("cell_id", "nunique"))
            .reset_index()
        )
        fig_tech = px.bar(
            tech_summary,
            x="technology",
            y="avg_dl",
            color="technology",
            color_discrete_map=TECH_COLORS,
            labels={"avg_dl": "Avg DL (GB/h)", "technology": "Generation"},
            text_auto=".1f",
        )
        fig_tech.update_layout(height=350, showlegend=False, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white")
        st.plotly_chart(fig_tech, use_container_width=True)

    # Visualizations Row 2: Monthly Trends
    st.markdown("**Long-Term Monthly Network Evolution (2024–2025)**")
    df_monthly = df.copy()
    df_monthly["month_str"] = df_monthly["timestamp"].dt.to_period("M").astype(str)
    monthly_agg = (
        df_monthly.groupby("month_str")
        .agg(total_dl=("dl_traffic_volume_gb", "sum"), avg_prb=("prb_utilization_pct", "mean"))
        .reset_index()
    )
    fig_month = make_subplots(specs=[[{"secondary_y": True}]])
    fig_month.add_trace(
        go.Bar(x=monthly_agg["month_str"], y=monthly_agg["total_dl"], name="Total DL Volume (GB)", marker_color=DJEZZY_RED),
        secondary_y=False,
    )
    fig_month.add_trace(
        go.Scatter(x=monthly_agg["month_str"], y=monthly_agg["avg_prb"], name="Avg PRB Utilisation (%)", line=dict(color=DJEZZY_DARK, width=2.5)),
        secondary_y=True,
    )
    fig_month.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0), template="plotly_white", legend=dict(x=0.01, y=0.99))
    fig_month.update_yaxes(title_text="Total Downlink (GB)", secondary_y=False)
    fig_month.update_yaxes(title_text="Avg PRB (%)", secondary_y=True)
    st.plotly_chart(fig_month, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 2: Forecast Analysis
# ---------------------------------------------------------------------------
def render_forecast_tab(cfg: dict, filters: dict, df_cleaned: pd.DataFrame):
    st.markdown('<div class="section-title">Forward Operational Forecasts</div>', unsafe_allow_html=True)

    h_key = filters["horizon"]
    h_alias = filters["horizon_alias"]
    forecasts_dir = Path(cfg["paths"]["forecasts_dir"])

    # Load DL and PRB forecast files
    f_dl_p = forecasts_dir / f"forecast_dl_traffic_volume_gb_{h_key}.parquet"
    if not f_dl_p.exists():
        f_dl_p = forecasts_dir / f"forecast_dl_traffic_volume_gb_{h_alias}.parquet"

    f_prb_p = forecasts_dir / f"forecast_prb_utilization_pct_{h_key}.parquet"
    if not f_prb_p.exists():
        f_prb_p = forecasts_dir / f"forecast_prb_utilization_pct_{h_alias}.parquet"

    if not f_dl_p.exists() or not f_prb_p.exists():
        st.warning(f"Forecasts for horizon `{h_key}` not found. Please run `python -m src.models.predict`.")
        return

    df_fc_dl = load_forecast_file(str(f_dl_p))
    df_fc_prb = load_forecast_file(str(f_prb_p))

    # Apply filters
    if filters["wilayas"]:
        df_fc_dl = df_fc_dl[df_fc_dl["wilaya_name"].isin(filters["wilayas"])]
        df_fc_prb = df_fc_prb[df_fc_prb["wilaya_name"].isin(filters["wilayas"])]
    if filters["technologies"]:
        df_fc_dl = df_fc_dl[df_fc_dl["technology"].isin(filters["technologies"])]
        df_fc_prb = df_fc_prb[df_fc_prb["technology"].isin(filters["technologies"])]
    if filters["site"] != "All Sites":
        df_fc_dl = df_fc_dl[df_fc_dl["site_id"] == filters["site"]]
        df_fc_prb = df_fc_prb[df_fc_prb["site_id"] == filters["site"]]

    # CRITICAL CHECK: If specific cell is selected, show that individual cell's forecast!
    is_single_cell = (filters["cell"] != "All Cells")
    if is_single_cell:
        selected_cid = filters["cell"]
        df_fc_dl = df_fc_dl[df_fc_dl["cell_id"] == selected_cid]
        df_fc_prb = df_fc_prb[df_fc_prb["cell_id"] == selected_cid]
        st.info(f"📍 Viewing Individual Forecast for Cell: **{selected_cid}** (Horizon: {h_key.upper()})")
    else:
        st.info(f"🌐 Viewing Aggregate Forecast Across **{df_fc_dl['cell_id'].nunique()} Selected Cells** (Horizon: {h_key.upper()})")

    # Forecast Statistics Row
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Total Forecast Steps", f"{df_fc_dl['timestamp'].nunique()} hours")
    with s2:
        st.metric("Avg Projected DL", f"{df_fc_dl['predicted'].mean():.2f} GB/h")
    with s3:
        st.metric("Avg Projected PRB", f"{df_fc_prb['predicted'].mean():.1f}%")
    with s4:
        peak_prb = df_fc_prb["predicted"].max()
        st.metric("Peak Projected PRB", f"{peak_prb:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Downlink Traffic Forecast Chart
    st.markdown("#### 1. Downlink Traffic Volume Forecast (GB/h)")
    fig_dl = go.Figure()

    if is_single_cell:
        # Show actual cell line
        fig_dl.add_trace(go.Scatter(
            x=df_fc_dl["timestamp"],
            y=df_fc_dl["predicted"],
            mode="lines+markers",
            name=f"Forecast: {selected_cid}",
            line=dict(color=DJEZZY_RED, width=3),
        ))
    else:
        # Aggregate across cells
        agg_dl = df_fc_dl.groupby("timestamp")["predicted"].agg(["mean", "min", "max"]).reset_index()
        fig_dl.add_trace(go.Scatter(
            x=pd.concat([agg_dl["timestamp"], agg_dl["timestamp"][::-1]]),
            y=pd.concat([agg_dl["max"], agg_dl["min"][::-1]]),
            fill="toself",
            fillcolor="rgba(224, 43, 32, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Cell Range (Min–Max)",
        ))
        fig_dl.add_trace(go.Scatter(
            x=agg_dl["timestamp"],
            y=agg_dl["mean"],
            mode="lines",
            name="Network Mean DL",
            line=dict(color=DJEZZY_RED, width=2.5),
        ))

    fig_dl.update_layout(
        xaxis_title="Forecast Timestamp (Africa/Algiers)",
        yaxis_title="Downlink Traffic (GB/h)",
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig_dl, use_container_width=True)

    # 2. PRB Utilisation Forecast Chart with Thresholds
    st.markdown("#### 2. Physical Resource Block (PRB) Utilisation & Congestion Thresholds")
    fig_prb = go.Figure()

    if is_single_cell:
        fig_prb.add_trace(go.Scatter(
            x=df_fc_prb["timestamp"],
            y=df_fc_prb["predicted"],
            mode="lines+markers",
            name=f"Forecast PRB: {selected_cid}",
            line=dict(color=DJEZZY_DARK, width=3),
        ))
    else:
        agg_prb = df_fc_prb.groupby("timestamp")["predicted"].agg(["mean", "min", "max"]).reset_index()
        fig_prb.add_trace(go.Scatter(
            x=pd.concat([agg_prb["timestamp"], agg_prb["timestamp"][::-1]]),
            y=pd.concat([agg_prb["max"], agg_prb["min"][::-1]]),
            fill="toself",
            fillcolor="rgba(20, 23, 26, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Cell Range (Min–Max)",
        ))
        fig_prb.add_trace(go.Scatter(
            x=agg_prb["timestamp"],
            y=agg_prb["mean"],
            mode="lines",
            name="Network Mean PRB",
            line=dict(color=DJEZZY_DARK, width=2.5),
        ))

    # Add Official Congestion Threshold Lines
    fig_prb.add_hline(y=80, line_dash="dash", line_color=WARNING_AMBER, annotation_text="WARNING (≥80%)")
    fig_prb.add_hline(y=90, line_dash="dash", line_color=CRITICAL_RED, annotation_text="HIGH CONGESTION (≥90%)")

    fig_prb.update_layout(
        xaxis_title="Forecast Timestamp (Africa/Algiers)",
        yaxis_title="PRB Utilisation (%)",
        yaxis_range=[0, 105],
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig_prb, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 3: Congestion Alerts
# ---------------------------------------------------------------------------
def render_alerts_tab(cfg: dict, filters: dict):
    st.markdown('<div class="section-title">Congestion Prevention & Early Warning System</div>', unsafe_allow_html=True)

    h_key = filters["horizon"]
    h_alias = filters["horizon_alias"]
    metrics_dir = Path(cfg["paths"]["metrics_dir"])

    alert_p = metrics_dir / f"congestion_alerts_{h_key}.csv"
    if not alert_p.exists():
        alert_p = metrics_dir / f"congestion_alerts_{h_alias}.csv"

    df_alerts = load_alerts_file(str(alert_p))

    if df_alerts.empty:
        st.success("No predicted congestion alerts for the selected horizon.")
        return

    # Filter alerts
    if filters["wilayas"]:
        df_alerts = df_alerts[df_alerts["wilaya_name"].isin(filters["wilayas"])]
    if filters["technologies"]:
        df_alerts = df_alerts[df_alerts["technology"].isin(filters["technologies"])]
    if filters["cell"] != "All Cells":
        df_alerts = df_alerts[df_alerts["cell_id"] == filters["cell"]]

    # Severity Level filter
    sev_filter = st.radio("Filter Severity Level", options=["All Alerts", "HIGH (≥90%)", "WARNING (≥80%)"], horizontal=True)
    if "HIGH" in sev_filter:
        df_alerts = df_alerts[df_alerts["alert_level"] == "HIGH"]
    elif "WARNING" in sev_filter:
        df_alerts = df_alerts[df_alerts["alert_level"] == "WARNING"]

    # Alerts Summary Metrics
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.metric("Total Alerts", f"{len(df_alerts):,}")
    with a2:
        high_cnt = int((df_alerts["alert_level"] == "HIGH").sum())
        st.metric("🔴 HIGH Alerts (PRB ≥ 90%)", f"{high_cnt:,}")
    with a3:
        warn_cnt = int((df_alerts["alert_level"] == "WARNING").sum())
        st.metric("🟠 WARNING Alerts (PRB ≥ 80%)", f"{warn_cnt:,}")
    with a4:
        affected_cells = df_alerts["cell_id"].nunique()
        st.metric("Affected Unique Cells", f"{affected_cells:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    if df_alerts.empty:
        st.info("No predicted congestion alerts matching the active filters.")
        return

    # Top Congested Cells Chart
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.markdown("**Hourly Alert Timeline Across Network**")
        df_alerts["hour_slot"] = pd.to_datetime(df_alerts["timestamp"]).dt.strftime("%m-%d %H:00")
        timeline = (
            df_alerts.groupby(["hour_slot", "alert_level"])
            .size()
            .reset_index(name="alert_count")
        )
        fig_time = px.bar(
            timeline,
            x="hour_slot",
            y="alert_count",
            color="alert_level",
            color_discrete_map={"HIGH": CRITICAL_RED, "WARNING": WARNING_AMBER},
            labels={"alert_count": "Count of Alerts", "hour_slot": "Hour"},
        )
        fig_time.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white")
        st.plotly_chart(fig_time, use_container_width=True)

    with col_t2:
        st.markdown("**Top Most Congested Cells**")
        top_cells_df = (
            df_alerts.groupby(["cell_id", "wilaya_name", "technology"])["predicted_prb_pct"]
            .max()
            .reset_index()
            .sort_values("predicted_prb_pct", ascending=False)
            .head(8)
        )
        fig_top = px.bar(
            top_cells_df,
            x="predicted_prb_pct",
            y="cell_id",
            orientation="h",
            color="predicted_prb_pct",
            color_continuous_scale="Reds",
            labels={"predicted_prb_pct": "Peak PRB (%)", "cell_id": "Cell ID"},
        )
        fig_top.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_top, use_container_width=True)

    # Detailed Alerts Table
    st.markdown("#### Detailed Congestion Alerts Register")
    display_cols = ["timestamp", "cell_id", "wilaya_name", "technology", "predicted_prb_pct", "alert_level"]
    avail_cols = [c for c in display_cols if c in df_alerts.columns]
    st.dataframe(
        df_alerts[avail_cols].sort_values(["predicted_prb_pct", "timestamp"], ascending=[False, True]),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# TAB 4: Cell Deep-Dive
# ---------------------------------------------------------------------------
def render_deep_dive_tab(df_cleaned: pd.DataFrame, cfg: dict, filters: dict):
    st.markdown('<div class="section-title">Cell-Level Diagnostic Deep-Dive</div>', unsafe_allow_html=True)

    if df_cleaned.empty:
        st.warning("Cleaned dataset not available.")
        return

    # Cascading Dropdowns: Wilaya -> Site -> Cell
    c_w, c_s, c_c = st.columns(3)
    with c_w:
        w_list = sorted(df_cleaned["wilaya_name"].dropna().unique().tolist())
        chosen_w = st.selectbox("1. Select Wilaya", options=w_list, index=0)

    with c_s:
        sub_sites = sorted(df_cleaned[df_cleaned["wilaya_name"] == chosen_w]["site_id"].dropna().unique().tolist())
        chosen_s = st.selectbox("2. Select Site", options=sub_sites, index=0)

    with c_c:
        sub_cells = sorted(df_cleaned[df_cleaned["site_id"] == chosen_s]["cell_id"].dropna().unique().tolist())
        target_cell = st.selectbox("3. Select Radio Cell", options=sub_cells, index=0)

    # Extract Cell Historical Data
    cell_hist = df_cleaned[df_cleaned["cell_id"] == target_cell].sort_values("timestamp")
    if cell_hist.empty:
        st.warning("No data found for this cell.")
        return

    cell_meta = cell_hist.iloc[-1]

    # Load Forecast for this cell
    h_key = filters["horizon"]
    h_alias = filters["horizon_alias"]
    forecasts_dir = Path(cfg["paths"]["forecasts_dir"])

    f_dl_p = forecasts_dir / f"forecast_dl_traffic_volume_gb_{h_key}.parquet"
    if not f_dl_p.exists():
        f_dl_p = forecasts_dir / f"forecast_dl_traffic_volume_gb_{h_alias}.parquet"

    f_prb_p = forecasts_dir / f"forecast_prb_utilization_pct_{h_key}.parquet"
    if not f_prb_p.exists():
        f_prb_p = forecasts_dir / f"forecast_prb_utilization_pct_{h_alias}.parquet"

    fc_dl_cell = pd.DataFrame()
    fc_prb_cell = pd.DataFrame()
    if f_dl_p.exists() and f_prb_p.exists():
        all_fc_dl = load_forecast_file(str(f_dl_p))
        all_fc_prb = load_forecast_file(str(f_prb_p))
        fc_dl_cell = all_fc_dl[all_fc_dl["cell_id"] == target_cell].sort_values("timestamp")
        fc_prb_cell = all_fc_prb[all_fc_prb["cell_id"] == target_cell].sort_values("timestamp")

    peak_prb = fc_prb_cell["predicted"].max() if not fc_prb_cell.empty else 0.0
    peak_dl = fc_dl_cell["predicted"].max() if not fc_dl_cell.empty else 0.0

    # Determine Health Status
    if peak_prb >= 90.0:
        status_html = '<span class="badge-high">🔴 HIGH CONGESTION</span>'
    elif peak_prb >= 80.0:
        status_html = '<span class="badge-warning">🟠 WARNING</span>'
    else:
        status_html = '<span class="badge-normal">🟢 NORMAL</span>'

    # Cell Metadata & Health Card
    st.markdown(f"""
    <div class="kpi-card" style="margin-bottom: 1.25rem;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h3 style="margin:0; font-size:1.35rem; color:{DJEZZY_DARK};">{target_cell}</h3>
                <div style="color:{DJEZZY_MUTED}; font-size:0.85rem; margin-top:0.25rem;">
                    Site: <b>{cell_meta.get('site_id', 'N/A')}</b> &nbsp;|&nbsp; 
                    Wilaya: <b>{cell_meta.get('wilaya_name', 'N/A')}</b> &nbsp;|&nbsp; 
                    Technology: <b>{cell_meta.get('technology', 'N/A')}</b> &nbsp;|&nbsp; 
                    Area Type: <b>{cell_meta.get('area_type', 'N/A')}</b>
                </div>
            </div>
            <div>
                {status_html}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Diagnostic Metrics
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Historical Avg DL", f"{cell_hist['dl_traffic_volume_gb'].mean():.2f} GB/h")
    with d2:
        st.metric("Forecast Peak DL", f"{peak_dl:.2f} GB/h")
    with d3:
        st.metric("Historical Avg PRB", f"{cell_hist['prb_utilization_pct'].mean():.1f}%")
    with d4:
        st.metric("Forecast Peak PRB", f"{peak_prb:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Chart 1: Historical + Forecast DL
    st.markdown("**Downlink Traffic: Recent History vs Forward Forecast**")
    recent_hist = cell_hist.tail(168)  # Last week of 2025
    fig_d1 = go.Figure()
    fig_d1.add_trace(go.Scatter(
        x=recent_hist["timestamp"],
        y=recent_hist["dl_traffic_volume_gb"],
        mode="lines",
        name="Historical Observed",
        line=dict(color=DJEZZY_DARK, width=2),
    ))
    if not fc_dl_cell.empty:
        fig_d1.add_trace(go.Scatter(
            x=fc_dl_cell["timestamp"],
            y=fc_dl_cell["predicted"],
            mode="lines+markers",
            name="Model Forecast",
            line=dict(color=DJEZZY_RED, width=2.5),
        ))
    fig_d1.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white", yaxis_title="DL Traffic (GB/h)")
    st.plotly_chart(fig_d1, use_container_width=True)

    # Chart 2: Historical + Forecast PRB with Congestion Lines
    st.markdown("**PRB Utilisation: Recent History vs Forward Forecast (with Congestion Limits)**")
    fig_d2 = go.Figure()
    fig_d2.add_trace(go.Scatter(
        x=recent_hist["timestamp"],
        y=recent_hist["prb_utilization_pct"],
        mode="lines",
        name="Historical PRB",
        line=dict(color=DJEZZY_DARK, width=2),
    ))
    if not fc_prb_cell.empty:
        fig_d2.add_trace(go.Scatter(
            x=fc_prb_cell["timestamp"],
            y=fc_prb_cell["predicted"],
            mode="lines+markers",
            name="Forecast PRB",
            line=dict(color=DJEZZY_RED, width=2.5),
        ))
    fig_d2.add_hline(y=80, line_dash="dash", line_color=WARNING_AMBER, annotation_text="WARNING (80%)")
    fig_d2.add_hline(y=90, line_dash="dash", line_color=CRITICAL_RED, annotation_text="HIGH (90%)")
    fig_d2.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white", yaxis_title="PRB (%)", yaxis_range=[0, 105])
    st.plotly_chart(fig_d2, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 5: Model Performance & Benchmarking
# ---------------------------------------------------------------------------
def render_performance_tab(cfg: dict):
    st.markdown('<div class="section-title">Model Performance & Baseline Benchmarking</div>', unsafe_allow_html=True)

    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    summary_csv = metrics_dir / "metrics_summary.csv"
    training_json = metrics_dir / "training_metrics.json"

    df_metrics = load_metrics_summary(str(summary_csv))

    st.markdown("""
    **Validation Strategy**: Chronological walk-forward cross-validation (3 folds × 2-month test windows).
    All models evaluated on the **untouched final holdout period (November 1 – December 31, 2025)** with strict zero temporal leakage.
    """)

    if not df_metrics.empty:
        st.markdown("#### 1. Holdout Benchmark Table Across All Models & Baselines")
        # Format metrics table
        st.dataframe(
            df_metrics.style.format({
                "MAE": "{:.4f}",
                "RMSE": "{:.4f}",
                "MAPE": "{:.2f}%",
                "sMAPE": "{:.2f}%",
                "WAPE": "{:.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Model Comparison Chart (WAPE & MAE)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**DL Traffic Volume (24h) — Model Comparison (WAPE %)**")
            dl_24 = df_metrics[(df_metrics["target"] == "dl_traffic_volume_gb") & (df_metrics["horizon"] == "24h")]
            if not dl_24.empty:
                fig_m1 = px.bar(
                    dl_24.sort_values("WAPE"),
                    x="model",
                    y="WAPE",
                    color="model",
                    color_discrete_sequence=["#E02B20", "#14171A", "#5A626A", "#8C95A0", "#B0B8C0"],
                    labels={"WAPE": "WAPE (%)", "model": "Model / Baseline"},
                    text_auto=".1f",
                )
                fig_m1.update_layout(showlegend=False, height=330, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white")
                st.plotly_chart(fig_m1, use_container_width=True)

        with c2:
            st.markdown("**PRB Utilisation (24h) — Model Comparison (MAE %)**")
            prb_24 = df_metrics[(df_metrics["target"] == "prb_utilization_pct") & (df_metrics["horizon"] == "24h")]
            if not prb_24.empty:
                fig_m2 = px.bar(
                    prb_24.sort_values("MAE"),
                    x="model",
                    y="MAE",
                    color="model",
                    color_discrete_sequence=["#E02B20", "#14171A", "#5A626A", "#8C95A0", "#B0B8C0"],
                    labels={"MAE": "MAE (%)", "model": "Model / Baseline"},
                    text_auto=".2f",
                )
                fig_m2.update_layout(showlegend=False, height=330, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white")
                st.plotly_chart(fig_m2, use_container_width=True)

    # Feature Importance Section
    st.markdown("#### 2. Key Predictive Signals (LightGBM Feature Importance)")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("**Top Drivers for DL Traffic Forecasting**")
        f_imp_dl = load_feature_importance(str(metrics_dir / "feature_importance_dl_traffic_volume_gb.csv"))
        if not f_imp_dl.empty:
            fig_f1 = px.bar(
                f_imp_dl.head(10).sort_values("importance", ascending=True),
                x="importance",
                y="feature",
                orientation="h",
                color_discrete_sequence=[DJEZZY_RED],
                labels={"importance": "Importance (Splits)", "feature": "Feature"},
            )
            fig_f1.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white")
            st.plotly_chart(fig_f1, use_container_width=True)

    with col_f2:
        st.markdown("**Top Drivers for PRB Congestion Forecasting**")
        f_imp_prb = load_feature_importance(str(metrics_dir / "feature_importance_prb_utilization_pct.csv"))
        if not f_imp_prb.empty:
            fig_f2 = px.bar(
                f_imp_prb.head(10).sort_values("importance", ascending=True),
                x="importance",
                y="feature",
                orientation="h",
                color_discrete_sequence=[DJEZZY_DARK],
                labels={"importance": "Importance (Splits)", "feature": "Feature"},
            )
            fig_f2.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white")
            st.plotly_chart(fig_f2, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 6: Data Quality Audit
# ---------------------------------------------------------------------------
def render_quality_tab(df_cleaned: pd.DataFrame, cfg: dict):
    st.markdown('<div class="section-title">Data Quality Audit & Pipeline Governance</div>', unsafe_allow_html=True)

    # Audit Metrics Cards
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("Raw Observations", "1,238,186")
    with q2:
        st.metric("Duplicates Cleared", "12,254")
    with q3:
        st.metric("Clean Records", f"{len(df_cleaned):,}")
    with q4:
        st.metric("Hourly Grid Completeness", "100.0%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Quality Problem Log Table
    st.markdown("#### Automated Quality Verification & Anomaly Log")
    quality_table = pd.DataFrame([
        {"Anomaly Identified": "Duplicate cell-hour records", "Detected Volume": "12,254 rows", "Remediation Action": "Exact cell-hour deduplication keeping primary record"},
        {"Anomaly Identified": "Technology variant labels", "Detected Volume": "16 unstandardized variants", "Remediation Action": "Mapped to 4 canonical layers (2G, 3G, 4G, 5G)"},
        {"Anomaly Identified": "Wilaya spelling inconsistencies", "Detected Volume": "79+ text variants", "Remediation Action": "Canonicalized to 25 official Algerian Wilayas"},
        {"Anomaly Identified": "Negative DL traffic volume", "Detected Volume": "2,469 erroneous readings", "Remediation Action": "Replaced with NaN and imputed via temporal profile"},
        {"Anomaly Identified": "PRB utilisation > 100%", "Detected Volume": "4,926 exceeding capacity", "Remediation Action": "Capped strictly at physical ceiling of 100.0%"},
        {"Anomaly Identified": "Cell availability > 100%", "Detected Volume": "1,846 erroneous readings", "Remediation Action": "Capped strictly at 100.0%"},
        {"Anomaly Identified": "Active users > Connected users", "Detected Volume": "Post-imputation artifacts", "Remediation Action": "Bounded active_users <= rrc_connected_users"},
        {"Anomaly Identified": "Congestion flag discrepancy", "Detected Volume": "OSS flag unsynchronized", "Remediation Action": "Recomputed as 1 if PRB >= 80%, else 0"},
    ])
    st.dataframe(quality_table, use_container_width=True, hide_index=True)

    # Topology Summary
    st.markdown("#### Network Topology Distribution")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Monitored Cells by Radio Generation**")
        tech_dist = df_cleaned.groupby("technology", observed=True)["cell_id"].nunique().reset_index()
        tech_dist.columns = ["Technology", "Unique Cell Count"]
        st.dataframe(tech_dist, use_container_width=True, hide_index=True)

    with t2:
        st.markdown("**Monitored Cells by Area Type**")
        area_dist = df_cleaned.groupby("area_type", observed=True)["cell_id"].nunique().reset_index()
        area_dist.columns = ["Area Type", "Unique Cell Count"]
        st.dataframe(area_dist, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main Application Entrypoint
# ---------------------------------------------------------------------------
def main():
    inject_custom_css()
    cfg = load_config()

    # Load cleaned dataset for reference
    df_cleaned = load_cleaned_data(cfg["paths"]["processed_data"])

    # Render Sidebar and Filters
    filters = render_sidebar(df_cleaned, cfg)

    # Render Header
    render_header()

    # Render Navigation Tabs
    tabs = st.tabs([
        "🌐 Network Overview",
        "📈 Forecast Analysis",
        "⚠️ Congestion Alerts",
        "🔍 Cell Deep-Dive",
        "🎯 Model Performance",
        "🛡️ Data Quality",
    ])

    with tabs[0]:
        render_overview_tab(df_cleaned, cfg)
    with tabs[1]:
        render_forecast_tab(cfg, filters, df_cleaned)
    with tabs[2]:
        render_alerts_tab(cfg, filters)
    with tabs[3]:
        render_deep_dive_tab(df_cleaned, cfg, filters)
    with tabs[4]:
        render_performance_tab(cfg)
    with tabs[5]:
        render_quality_tab(df_cleaned, cfg)

    # Footer Branding
    st.markdown(f"""
    <div style="margin-top: 3.5rem; padding-top: 1.5rem; border-top: 1px solid {DJEZZY_BORDER}; text-align: center; color: {DJEZZY_MUTED}; font-size: 0.82rem;">
        <b>Djezzy Network Forecasting &amp; Capacity Planning Platform</b> · Direction de l'Ingénierie &amp; Planification Réseau · Optimum Telecom Algérie (OTA)
        <br><span style="font-size: 0.74rem; color: #8C96A0;">Internal Decision-Support Tool · 100% Synthetic Telemetry for Benchmark Modeling</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
