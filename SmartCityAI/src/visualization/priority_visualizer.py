"""
priority_visualizer.py
======================
Phase 8 - Visualization: Priority Score Charts

Bar charts, rank plots, gauge charts, and feature importance for priority scores.
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

log = logging.getLogger(__name__)
BASE_DIR      = Path(__file__).resolve().parents[3]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"

PRIORITY_COLORS = {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#27AE60"}


def plot_top_priority_sites(
    ranked_df: pd.DataFrame,
    infra_type: str = "hospital",
    top_n: int = 15,
    save_path: Optional[Path] = None,
) -> go.Figure:
    """
    Horizontal bar chart of top-N sites by priority score.
    Uses Plotly for interactive tooltips.
    """
    df = ranked_df.head(top_n).copy()
    df["label"] = [f"#{r:.0f} — {h3[:12]}..." for r, h3 in zip(df.get("rank", range(1, top_n+1)), df.get("h3_id", [""]*top_n))]
    df["color"] = df.get("priority_score", 50).apply(
        lambda s: "#E74C3C" if s >= 75 else ("#F39C12" if s >= 50 else "#27AE60")
    )

    fig = go.Figure(go.Bar(
        x=df.get("priority_score", [50]*len(df)),
        y=df["label"],
        orientation="h",
        marker_color=df["color"],
        text=[f"{s:.1f}" for s in df.get("priority_score", [50]*len(df))],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Priority: %{x:.1f}/100<br>Coverage Gap: %{customdata[0]:.1%}",
        customdata=np.column_stack([df.get("coverage_gap", np.zeros(len(df)))]),
    ))

    fig.update_layout(
        title=f"Top {top_n} Priority Sites — {infra_type.replace('_',' ').title()}",
        xaxis_title="Priority Score (0–100)",
        yaxis_title="Site Rank",
        template="plotly_dark",
        height=max(400, top_n * 35),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f23",
        font=dict(color="white", family="Inter"),
        xaxis=dict(range=[0, 110]),
        yaxis=dict(autorange="reversed"),  # Rank 1 at top
    )

    if save_path:
        fig.write_html(str(save_path))
        log.info(f"Priority chart saved → {save_path}")
    return fig


def plot_priority_distribution(
    df: pd.DataFrame,
    priority_col: str = "priority_composite_100",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """
    Histogram of priority score distribution across all zones.
    Shows High/Medium/Low bands with color coding.
    """
    if priority_col not in df.columns:
        priority_col = next((c for c in ["priority_score", "infrastructure_need_score"] if c in df.columns), None)
        if not priority_col:
            return go.Figure()

    scores = df[priority_col].dropna()

    fig = go.Figure()

    # Three colored bands
    for band_name, (low, high), color in [
        ("Low Priority (0–33)",    (0, 33),   "#27AE60"),
        ("Medium Priority (33–66)",(33, 66),  "#F39C12"),
        ("High Priority (66–100)", (66, 100), "#E74C3C"),
    ]:
        mask = (scores >= low) & (scores < high)
        fig.add_trace(go.Histogram(
            x=scores[mask],
            name=band_name,
            marker_color=color,
            opacity=0.85,
            nbinsx=20,
        ))

    # Add vertical lines for band boundaries
    for x_val in [33, 66]:
        fig.add_vline(x=x_val, line_dash="dash", line_color="white", line_width=1.5, opacity=0.5)

    fig.update_layout(
        title="Infrastructure Priority Score Distribution",
        xaxis_title="Priority Score (0–100)",
        yaxis_title="Number of Zones",
        template="plotly_dark",
        barmode="overlay",
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f23",
        font=dict(color="white", family="Inter"),
        legend=dict(bgcolor="#1a1a2e", bordercolor="#444"),
    )

    if save_path:
        fig.write_html(str(save_path))
    return fig


def plot_priority_gauge(
    priority_score: float,
    zone_id: str = "Selected Zone",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """
    Gauge/speedometer chart for a single zone's priority score.
    Used in the Streamlit dashboard explanation panel.
    """
    if priority_score >= 66:
        color = "#E74C3C"
        label = "HIGH PRIORITY"
    elif priority_score >= 33:
        color = "#F39C12"
        label = "MEDIUM PRIORITY"
    else:
        color = "#27AE60"
        label = "LOW PRIORITY"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=priority_score,
        title={"text": f"Priority Score<br><span style='font-size:0.8em;color:{color}'>{label}</span>",
               "font": {"size": 16, "color": "white"}},
        delta={"reference": 50, "increasing": {"color": "#E74C3C"}, "decreasing": {"color": "#27AE60"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white", "tickfont": {"color": "white"}},
            "bar": {"color": color},
            "bgcolor": "#1a1a2e",
            "borderwidth": 2,
            "bordercolor": "#444",
            "steps": [
                {"range": [0, 33],   "color": "#1a3a1a"},
                {"range": [33, 66],  "color": "#3a2a1a"},
                {"range": [66, 100], "color": "#3a1a1a"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.75,
                "value": priority_score,
            },
        },
        number={"font": {"size": 48, "color": color}, "suffix": "/100"},
    ))

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0f0f23",
        paper_bgcolor="#0f0f23",
        height=300,
        margin=dict(l=20, r=20, t=80, b=20),
    )

    if save_path:
        fig.write_html(str(save_path))
    return fig


def plot_feature_radar(
    zone_row: pd.Series,
    feature_cols: List[str],
    city_avg: Optional[pd.Series] = None,
    save_path: Optional[Path] = None,
) -> go.Figure:
    """
    Radar/spider chart comparing zone features vs city average.
    """
    available = [c for c in feature_cols if c in zone_row.index][:8]
    if not available:
        return go.Figure()

    from sklearn.preprocessing import MinMaxScaler
    vals = [float(zone_row.get(c, 0)) for c in available]
    labels = [c.replace("_", " ").title()[:15] for c in available]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name="This Zone",
        fillcolor="rgba(231, 76, 60, 0.25)",
        line=dict(color="#E74C3C", width=2),
    ))

    if city_avg is not None:
        avg_vals = [float(city_avg.get(c, 0)) for c in available]
        fig.add_trace(go.Scatterpolar(
            r=avg_vals + [avg_vals[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="City Average",
            fillcolor="rgba(52, 152, 219, 0.15)",
            line=dict(color="#3498DB", width=2, dash="dash"),
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#1a1a2e",
            radialaxis=dict(visible=True, range=[0, 1], color="white"),
            angularaxis=dict(color="white"),
        ),
        showlegend=True,
        template="plotly_dark",
        paper_bgcolor="#0f0f23",
        title="Zone Feature Profile vs City Average",
        font=dict(color="white", family="Inter"),
        height=400,
    )

    if save_path:
        fig.write_html(str(save_path))
    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ranked = PROCESSED_DIR / "ranked_sites_hospital.csv"
    if ranked.exists():
        df = pd.read_csv(ranked)
        plot_top_priority_sites(df, save_path=MODELS_DIR / "priority_chart.html")
        print("Priority charts saved.")
    else:
        print("Run best_first.py first.")
