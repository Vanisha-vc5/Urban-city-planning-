"""
coverage_visualizer.py
======================
Phase 8 - Visualization: Infrastructure Coverage Analysis Charts

Coverage gap analysis, service area maps, and before/after comparisons.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

log = logging.getLogger(__name__)
BASE_DIR      = Path(__file__).resolve().parents[3]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"

INFRA_COLORS = {
    "hospital":    "#E74C3C",
    "school":      "#3498DB",
    "ev_station":  "#2ECC71",
    "fire_station":"#F39C12",
}


def plot_coverage_gap_map(
    df: pd.DataFrame,
    gap_col: str = "coverage_gap",
    infra_type: str = "hospital",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """
    Scatter map showing coverage gap intensity per zone.

    Requires lat/lon columns. Color-coded by gap severity.
    """
    if "lat" not in df.columns or "lon" not in df.columns:
        df = df.copy()
        df["lat"] = np.random.normal(19.0, 0.1, len(df))
        df["lon"] = np.random.normal(72.85, 0.1, len(df))

    col = gap_col if gap_col in df.columns else "coverage_gap"
    if col not in df.columns:
        df[col] = 0.5

    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color=col,
        color_continuous_scale=["#27AE60", "#F39C12", "#E74C3C"],
        size_max=15,
        zoom=11,
        center={"lat": df["lat"].mean(), "lon": df["lon"].mean()},
        mapbox_style="carto-darkmatter",
        title=f"{infra_type.replace('_',' ').title()} Coverage Gap Map",
        hover_data={"lat": False, "lon": False, col: ":.3f"},
        opacity=0.75,
    )

    fig.update_layout(
        coloraxis_colorbar=dict(title="Coverage Gap", tickformat=".0%"),
        template="plotly_dark",
        paper_bgcolor="#0f0f23",
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        font=dict(color="white"),
    )

    if save_path:
        fig.write_html(str(save_path))
        log.info(f"Coverage map saved → {save_path}")
    return fig


def plot_multi_infra_coverage(
    df: pd.DataFrame,
    infra_types: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
) -> go.Figure:
    """
    Grouped bar chart comparing coverage gaps across infrastructure types.
    Shows: % zones with High gap, Medium gap, Low gap per infra type.
    """
    if infra_types is None:
        infra_types = ["hospital", "school", "ev_station", "fire_station"]

    categories = ["High Gap (>60%)", "Medium Gap (30-60%)", "Low Gap (<30%)"]
    fig = go.Figure()

    for infra in infra_types:
        col = f"{infra}_coverage_gap" if f"{infra}_coverage_gap" in df.columns else "coverage_gap"
        if col not in df.columns:
            continue

        gaps = df[col].dropna()
        high_pct   = (gaps > 0.6).mean() * 100
        medium_pct = ((gaps > 0.3) & (gaps <= 0.6)).mean() * 100
        low_pct    = (gaps <= 0.3).mean() * 100

        fig.add_trace(go.Bar(
            name=infra.replace("_", " ").title(),
            x=categories,
            y=[high_pct, medium_pct, low_pct],
            marker_color=INFRA_COLORS.get(infra, "#9B59B6"),
            text=[f"{v:.1f}%" for v in [high_pct, medium_pct, low_pct]],
            textposition="outside",
        ))

    fig.update_layout(
        title="Multi-Infrastructure Coverage Gap Analysis",
        xaxis_title="Gap Category",
        yaxis_title="% of Zones",
        barmode="group",
        template="plotly_dark",
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f23",
        font=dict(color="white", family="Inter"),
        yaxis=dict(range=[0, 110]),
        legend=dict(bgcolor="#1a1a2e"),
    )

    if save_path:
        fig.write_html(str(save_path))
    return fig


def plot_before_after_coverage(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    gap_col: str = "coverage_gap",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """
    Before/after comparison showing coverage improvement post-recommendation.

    df_before: current state (from create_zones.py)
    df_after:  simulated state with recommended sites added
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Current Coverage Gaps", "After Recommendations"])

    for col_idx, (df, title) in enumerate([(df_before, "Before"), (df_after, "After")], 1):
        col = gap_col if gap_col in df.columns else "coverage_gap"
        if col not in df.columns:
            continue

        gaps = df[col].dropna()
        counts, bins = np.histogram(gaps, bins=20, range=(0, 1))
        bin_centers  = (bins[:-1] + bins[1:]) / 2

        fig.add_trace(
            go.Bar(x=bin_centers, y=counts,
                   marker_color=["#E74C3C" if b > 0.5 else "#F39C12" if b > 0.3 else "#27AE60" for b in bin_centers],
                   name=title, showlegend=False, opacity=0.8),
            row=1, col=col_idx
        )

    fig.update_layout(
        title="Coverage Gap: Before vs After Recommendations",
        template="plotly_dark",
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f23",
        font=dict(color="white"),
        height=400,
    )
    fig.update_xaxes(title_text="Coverage Gap", range=[0, 1])
    fig.update_yaxes(title_text="Number of Zones")

    if save_path:
        fig.write_html(str(save_path))
    return fig


def plot_coverage_improvement_summary(
    ranked_df: pd.DataFrame,
    df: pd.DataFrame,
    max_sites: int = 10,
    save_path: Optional[Path] = None,
) -> go.Figure:
    """
    Cumulative coverage improvement as more sites are added.
    X-axis: Number of recommended sites built.
    Y-axis: Cumulative % of previously uncovered population now covered.
    """
    pops = []
    total_uncovered = df.get("population_total", pd.Series(np.ones(len(df)) * 10000)).sum()

    fig = go.Figure()
    cum_coverage = []
    covered_so_far = 0.0

    for n in range(1, min(max_sites + 1, len(ranked_df) + 1)):
        site = ranked_df.iloc[n - 1]
        pop = site.get("population_total", 10000)
        gap = site.get("coverage_gap", 0)
        newly_covered = pop * gap * 0.7  # 70% capture rate
        covered_so_far += newly_covered
        cum_coverage.append(covered_so_far / (total_uncovered + 1) * 100)

    if cum_coverage:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cum_coverage) + 1)),
            y=cum_coverage,
            mode="lines+markers",
            line=dict(color="#3498DB", width=3),
            marker=dict(size=8, color="#3498DB"),
            fill="tozeroy",
            fillcolor="rgba(52, 152, 219, 0.15)",
            name="Cumulative Coverage Gain",
        ))

        # Diminishing returns annotation
        if len(cum_coverage) >= 3:
            marginal_gains = np.diff(cum_coverage)
            elbow = np.argmin(marginal_gains) + 1
            fig.add_vline(
                x=elbow + 1,
                line_dash="dash", line_color="#F39C12",
                annotation_text=f"Diminishing returns ({elbow+1} sites)",
                annotation_font_color="#F39C12",
            )

    fig.update_layout(
        title="Cumulative Coverage Improvement vs Number of Sites Built",
        xaxis_title="Number of New Sites",
        yaxis_title="Cumulative Population Gain (%)",
        template="plotly_dark",
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f23",
        font=dict(color="white", family="Inter"),
        height=400,
    )

    if save_path:
        fig.write_html(str(save_path))
    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    csv = PROCESSED_DIR / "zone_features.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        plot_multi_infra_coverage(df, save_path=MODELS_DIR / "coverage_comparison.html")
        print("Coverage visualizations saved.")
    else:
        print("Run create_zones.py first.")
