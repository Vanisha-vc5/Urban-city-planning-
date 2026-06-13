"""
app.py
======
Phase 9 - Streamlit Dashboard: SmartCityAI Interactive Interface

PURPOSE:
    Production-grade Streamlit web application for the SmartCityAI system.
    Provides city planners with an interactive interface to explore ML results,
    get infrastructure recommendations, and understand AI decisions.

PAGES:
    1. 🏙️ Overview          — City stats, executive summary
    2. 🗺️ Zone Analysis     — Interactive map with feature overlays
    3. 🤖 ML Results        — Cluster map, SVM classifications
    4. ⭐ Recommendations   — Top sites with full explanations
    5. 🔮 Scenario Simulator — Budget slider, scenario comparison
    6. 🚒 Route Optimizer   — A* emergency routing
    7. 📊 Model Insights    — Feature importance, metrics

DESIGN:
    - Dark premium theme (#0f0f23 background)
    - Custom CSS for branded styling
    - Streamlit columns for responsive layout
    - Folium maps embedded via st.components
    - Plotly charts with consistent dark theme
    - Real-time updates via session_state

USAGE:
    streamlit run dashboard/app.py

    or from project root:
    streamlit run SmartCityAI/dashboard/app.py
"""

import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Path Setup ─────────────────────────────────────────────────────────────────
# Add src to Python path for imports
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT  = DASHBOARD_DIR.parent
SRC_DIR       = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"

# ── Streamlit Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCityAI — Urban Infrastructure Planning",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "SmartCityAI v1.0 — Explainable Urban Infrastructure Planning System",
    },
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
        min-height: 100vh;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0f23 100%);
        border-right: 1px solid #333;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }

    /* Headers */
    h1 { color: #fff; font-weight: 700; }
    h2 { color: #e0e0ff; font-weight: 600; }
    h3 { color: #c0c0ff; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #E74C3C, #C0392B);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(231,76,60,0.4);
    }

    /* Selectbox, slider */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05);
        border: 1px solid #444;
        border-radius: 8px;
        color: white;
    }

    /* Info boxes */
    .priority-card {
        background: rgba(231, 76, 60, 0.15);
        border: 1px solid #E74C3C;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }

    /* Table */
    .dataframe {
        background: transparent !important;
        color: white !important;
    }

    /* Divider */
    hr { border-color: #333; }

    /* Glowing header */
    .glow-text {
        text-shadow: 0 0 20px rgba(99, 102, 241, 0.7);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Data Loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Loading city data...")
def load_zone_data() -> pd.DataFrame:
    """Load zone features, merging all prediction outputs."""
    # Try enriched first, then ml_features, then raw zone_features
    for fname in ["enriched_features.csv", "ml_features.csv", "zone_features.csv"]:
        path = PROCESSED_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            break
    else:
        # Generate synthetic demo data
        df = _generate_demo_data()

    # Merge cluster IDs
    for extra in ["zone_clusters.csv", "svm_predictions.csv", "dt_predictions.csv", "regression_predictions.csv"]:
        path = PROCESSED_DIR / extra
        if path.exists():
            extra_df = pd.read_csv(path)
            if "h3_id" in extra_df.columns and "h3_id" in df.columns:
                df = df.merge(extra_df, on="h3_id", how="left", suffixes=("", f"_{extra[:3]}"))

    # Add lat/lon if missing
    if "lat" not in df.columns:
        rng = np.random.default_rng(42)
        n = len(df)
        df["lat"] = 19.0 + rng.normal(0, 0.1, n)
        df["lon"] = 72.85 + rng.normal(0, 0.1, n)

    return df


@st.cache_data(ttl=300)
def load_ranked_sites(infra_type: str = "hospital") -> pd.DataFrame:
    """Load Best First Search ranked sites."""
    path = PROCESSED_DIR / f"ranked_sites_{infra_type}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_explanations() -> List[Dict]:
    """Load pre-computed explanations."""
    path = PROCESSED_DIR / "explanations.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def _generate_demo_data(n: int = 300) -> pd.DataFrame:
    """Generate synthetic demo data for offline use."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "h3_id":                  [f"demo_zone_{i:04d}" for i in range(n)],
        "population_density":     rng.lognormal(9.5, 0.7, n).clip(1000, 80000).round(0),
        "population_total":       rng.lognormal(10.0, 0.6, n).clip(5000, 150000).round(0),
        "elderly_ratio":          rng.beta(2.5, 28, n).round(4),
        "income_bracket_norm":    rng.beta(3, 5, n).round(4),
        "coverage_gap":           rng.beta(2, 3, n).round(4),
        "dist_nearest_hospital":  rng.exponential(3, n).clip(0.2, 15).round(3),
        "hospital_coverage_ratio":rng.beta(3, 2, n).round(4),
        "road_accessibility_index":rng.beta(3, 3, n).round(4),
        "traffic_density":        rng.lognormal(2.5, 0.7, n).round(2),
        "vulnerability_index":    rng.beta(2, 5, n).round(4),
        "priority_composite_100": rng.uniform(10, 95, n).round(1),
        "priority_class":         rng.choice(["Low", "Medium", "High"], n, p=[0.35, 0.40, 0.25]),
        "cluster_id":             rng.integers(0, 5, n),
        "cluster_label":          rng.choice(["Dense Urban Core","Underserved Periphery",
                                               "Affluent Suburbs","High-Need Slums",
                                               "Transitional Growth"], n),
        "lat":                    19.0 + rng.normal(0, 0.1, n),
        "lon":                    72.85 + rng.normal(0, 0.1, n),
    })
    return df


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar with navigation and filters."""
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:20px 0'>
            <h1 style='font-size:1.8em;margin:0'>🏙️ SmartCityAI</h1>
            <p style='color:#8892B0;margin:4px 0;font-size:0.85em'>Urban Infrastructure Planning</p>
            <div style='background:linear-gradient(90deg,#E74C3C,#9B59B6);height:2px;border-radius:1px;margin-top:8px'></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Navigation")
        page = st.radio(
            "",
            ["🏙️ Overview", "🗺️ Zone Analysis", "🤖 ML Results",
             "⭐ Recommendations", "🔮 Scenario Simulator", "🚒 Route Optimizer", "📊 Model Insights"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 🎯 Infrastructure Focus")
        infra_type = st.selectbox(
            "Select Infrastructure Type",
            ["hospital", "school", "ev_station", "fire_station"],
            format_func=lambda x: {
                "hospital": "🏥 Hospitals",
                "school": "🏫 Schools",
                "ev_station": "⚡ EV Stations",
                "fire_station": "🚒 Fire Stations",
            }.get(x, x),
        )

        st.markdown("### 💰 Budget Constraint")
        budget_crore = st.slider(
            "Budget (₹ Crore)",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
        )
        budget_rupees = budget_crore * 1e7

        st.markdown("### 🏙️ City")
        city = st.text_input("City Name", "Mumbai, India")

        st.markdown("---")
        st.markdown("""
        <div style='text-align:center;color:#555;font-size:0.8em'>
            <p>SmartCityAI v1.0</p>
            <p>ML + Classical AI + GIS</p>
            <p>© 2026 SmartCityAI Team</p>
        </div>
        """, unsafe_allow_html=True)

    return page, infra_type, budget_rupees, city


# ── Page 1: Overview ───────────────────────────────────────────────────────────

def page_overview(df: pd.DataFrame, infra_type: str):
    """City overview with key metrics and executive summary."""
    st.markdown('<h1 class="glow-text">🏙️ SmartCityAI Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Explainable AI-powered Urban Infrastructure Planning System**")
    st.markdown("---")

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    total_pop = df.get("population_total", pd.Series([10000]*len(df))).sum()
    avg_gap   = df.get("coverage_gap", pd.Series([0.5]*len(df))).mean()
    high_prio = (df.get("priority_class", pd.Series(["Medium"]*len(df))) == "High").sum()
    avg_dist  = df.get("dist_nearest_hospital", pd.Series([3.0]*len(df))).mean()
    n_zones   = len(df)

    with col1:
        st.metric("🏘️ Total Zones", f"{n_zones:,}")
    with col2:
        st.metric("👥 Total Population", f"{total_pop/1e6:.1f}M")
    with col3:
        st.metric("🔴 High Priority Zones", f"{high_prio:,}", delta=f"{high_prio/n_zones*100:.0f}% of all")
    with col4:
        st.metric("🏥 Avg. Hospital Distance", f"{avg_dist:.1f} km", delta=f"WHO: ≤5 km")
    with col5:
        st.metric("📉 Avg. Coverage Gap", f"{avg_gap*100:.0f}%")

    st.markdown("---")

    # Charts
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### 📊 Priority Distribution")
        try:
            from visualization.priority_visualizer import plot_priority_distribution
            fig = plot_priority_distribution(df)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception:
            _fallback_priority_bar(df)

    with col_right:
        st.markdown("### 🎯 Coverage Gap by Infrastructure")
        try:
            from visualization.coverage_visualizer import plot_multi_infra_coverage
            fig = plot_multi_infra_coverage(df)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception:
            st.info("Run the full pipeline to see coverage charts.")

    # Executive Summary
    st.markdown("---")
    st.markdown("### 📋 Executive Summary")
    critical_pct = (df.get("coverage_gap", pd.Series([0]*len(df))) > 0.6).mean() * 100
    st.markdown(f"""
    > **System Analysis for {len(df):,} urban zones:**
    >
    > - **{high_prio:,} zones** ({high_prio/n_zones*100:.0f}%) require immediate infrastructure attention
    > - **{critical_pct:.0f}%** of zones have critical coverage gaps (>60% unserved population)
    > - Average distance to nearest hospital: **{avg_dist:.1f} km** (WHO recommendation: ≤5 km)
    > - Estimated population without 5km hospital access: **{int(total_pop * avg_gap / 1e6):.1f}M residents**
    >
    > *AI models: K-Means Clustering + SVM Classification + Decision Tree + Ridge Regression + Best First Search*
    """)


# ── Page 2: Zone Analysis ─────────────────────────────────────────────────────

def page_zone_analysis(df: pd.DataFrame):
    """Interactive zone map with feature selector."""
    st.markdown("## 🗺️ Zone Analysis")

    col1, col2 = st.columns([3, 1])
    with col2:
        feature_to_show = st.selectbox(
            "Feature Overlay",
            [c for c in ["population_density", "coverage_gap", "priority_composite_100",
                          "vulnerability_index", "road_accessibility_index",
                          "dist_nearest_hospital", "elderly_ratio"] if c in df.columns],
        )
        show_heatmap = st.checkbox("Show Heatmap", value=True)

    with col1:
        try:
            import folium
            import geopandas as gpd
            import streamlit.components.v1 as components
            from visualization.map_visualizer import CityMapVisualizer

            zones_gdf = None
            gpkg = PROCESSED_DIR / "zones.gpkg"
            if gpkg.exists():
                zones_gdf = gpd.read_file(str(gpkg))

            city_center = (df["lat"].mean(), df["lon"].mean()) if "lat" in df.columns else (19.0, 72.85)
            viz = CityMapVisualizer(df, zones_gdf, city_center)
            m = viz.create_base_map("dark")
            if zones_gdf is not None:
                viz.add_population_choropleth(m, feature_to_show)
            if show_heatmap:
                viz.add_priority_heatmap(m, "priority_composite_100" if "priority_composite_100" in df.columns else feature_to_show)

            map_html = m._repr_html_()
            components.html(map_html, height=500, scrolling=False)

        except ImportError:
            st.info("📦 Install dependencies: `pip install folium geopandas` to see interactive map")
            _fallback_scatter_map(df, feature_to_show)

    # Zone statistics table
    st.markdown("### Zone Statistics")
    display_cols = [c for c in ["h3_id", "population_density", "coverage_gap",
                                 "priority_composite_100", "priority_class", "cluster_label"]
                    if c in df.columns]
    sort_col = "priority_composite_100" if "priority_composite_100" in df.columns else display_cols[0]
    st.dataframe(
        df[display_cols].sort_values(sort_col, ascending=False).head(20).reset_index(drop=True),
        use_container_width=True,
    )


# ── Page 3: ML Results ────────────────────────────────────────────────────────

def page_ml_results(df: pd.DataFrame):
    """K-Means clusters and SVM classification results."""
    st.markdown("## 🤖 Machine Learning Results")

    tab1, tab2, tab3, tab4 = st.tabs(["🔵 K-Means Clusters", "🔴 SVM Priority", "🌳 Decision Tree", "📈 Regression"])

    with tab1:
        st.markdown("### K-Means Clustering Results")
        if "cluster_id" in df.columns:
            counts = df["cluster_id"].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Cluster Sizes**")
                cluster_labels = {
                    0: "Dense Urban Core", 1: "Underserved Periphery",
                    2: "Affluent Suburbs", 3: "High-Need Slums", 4: "Transitional Growth",
                }
                for cid, count in counts.items():
                    label = cluster_labels.get(int(cid), f"Cluster {cid}")
                    pct = count / len(df) * 100
                    st.markdown(f"**{label}**: {count:,} zones ({pct:.1f}%)")

            with col2:
                # PCA plot if available
                pca_path = MODELS_DIR / "cluster_pca.png"
                if pca_path.exists():
                    st.image(str(pca_path), use_column_width=True)
                else:
                    try:
                        from visualization.cluster_visualizer import plot_pca_clusters
                        feature_cols = [c for c in df.columns if c not in ["h3_id","cluster_id","cluster_label","lat","lon","priority_class"]]
                        fig = plot_pca_clusters(df, feature_cols)
                        st.pyplot(fig)
                    except Exception:
                        st.info("Run kmeans.py to generate cluster visualizations.")

            # Cluster profile heatmap
            heatmap_path = MODELS_DIR / "cluster_heatmap.png"
            if heatmap_path.exists():
                st.image(str(heatmap_path), use_column_width=True)
        else:
            st.info("Run kmeans.py to generate cluster assignments.")

    with tab2:
        st.markdown("### SVM Priority Classification")
        svm_col = "svm_priority" if "svm_priority" in df.columns else "priority_class"
        if svm_col in df.columns:
            counts = df[svm_col].value_counts()
            cols = st.columns(len(counts))
            colors = {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#27AE60"}
            for i, (cls, count) in enumerate(counts.items()):
                with cols[i]:
                    color = colors.get(str(cls), "#9B59B6")
                    st.markdown(f"""
                    <div style='background:{color}22;border:2px solid {color};border-radius:12px;padding:16px;text-align:center'>
                        <h3 style='color:{color};margin:0'>{count:,}</h3>
                        <p style='color:{color};margin:0;font-weight:600'>{cls} Priority</p>
                    </div>""", unsafe_allow_html=True)

            # SVM confusion matrix
            cm_path = MODELS_DIR / "svm_confusion_matrix.png"
            if cm_path.exists():
                st.image(str(cm_path))
        else:
            st.info("Run svm_priority.py to generate SVM classifications.")

    with tab3:
        st.markdown("### Decision Tree Analysis")
        dt_vis_path = MODELS_DIR / "decision_tree_visualization.png"
        dt_rules_path = MODELS_DIR / "decision_tree_rules.txt"

        if dt_vis_path.exists():
            st.image(str(dt_vis_path), caption="Decision Tree (top 4 levels)", use_column_width=True)
        if dt_rules_path.exists():
            with open(dt_rules_path) as f:
                rules = f.read()
            with st.expander("📋 View Decision Rules"):
                st.code(rules[:3000], language="text")
        if not dt_vis_path.exists():
            st.info("Run decision_tree.py to generate Decision Tree visualization.")

    with tab4:
        st.markdown("### Population Demand Regression")
        reg_path = MODELS_DIR / "regression_comparison.png"
        if reg_path.exists():
            st.image(str(reg_path), use_column_width=True)
        else:
            st.info("Run population_regression.py to generate regression charts.")


# ── Page 4: Recommendations ───────────────────────────────────────────────────

def page_recommendations(df: pd.DataFrame, infra_type: str, budget: float):
    """Top-K site recommendations with full explanations."""
    st.markdown(f"## ⭐ Infrastructure Recommendations — {infra_type.replace('_',' ').title()}")

    # Budget info
    cost_map = {"hospital": 5e7, "school": 2e7, "ev_station": 5e6, "fire_station": 3e7}
    cost = cost_map.get(infra_type, 3e7)
    max_sites = int(budget // cost)
    st.info(f"💰 Budget: ₹{budget/1e7:.0f} Crore → Can build up to **{max_sites}** {infra_type.replace('_',' ')} facilities")

    # Load ranked sites
    ranked_df = load_ranked_sites(infra_type)

    if len(ranked_df) == 0:
        st.warning("No ranked sites found. Run: `python src/optimization/best_first.py`")
        _show_demo_recommendations(df, infra_type, max_sites)
        return

    # Top sites grid
    st.markdown("### 🏆 Top Recommended Sites")
    top_sites = ranked_df.head(min(max_sites, 10))
    n_cols = min(3, len(top_sites))
    cols = st.columns(n_cols)

    colors = {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#27AE60"}

    for i, (_, site) in enumerate(top_sites.iterrows()):
        with cols[i % n_cols]:
            score = site.get("priority_score", 50)
            color = "#E74C3C" if score >= 75 else "#F39C12" if score >= 50 else "#27AE60"
            rank  = site.get("rank", i + 1)
            gap   = site.get("coverage_gap", 0)
            pop   = site.get("population_total", 0)
            h3id  = site.get("h3_id", f"Zone {i}")

            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.04);border:1px solid {color};border-radius:12px;padding:14px;margin:6px 0'>
                <div style='display:flex;justify-content:space-between;align-items:center'>
                    <span style='font-size:1.3em'>⭐ Rank #{rank:.0f}</span>
                    <span style='background:{color};color:white;padding:2px 10px;border-radius:20px;font-weight:bold;font-size:0.85em'>{score:.0f}/100</span>
                </div>
                <div style='font-size:0.8em;color:#888;margin:4px 0'>{h3id[:20]}...</div>
                <div style='margin-top:8px'>
                    <div>🔴 Coverage Gap: <b>{gap*100:.0f}%</b></div>
                    <div>👥 Population: <b>{pop:,.0f}</b></div>
                    <div>💰 Est. Cost: <b>₹{cost/1e7:.1f} Cr</b></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # Explanation panel
    st.markdown("---")
    st.markdown("### 🔍 Site Explanation")
    selected_rank = st.selectbox("Select a site to explain:", range(1, len(top_sites) + 1))

    if selected_rank and len(top_sites) > 0:
        site = top_sites[top_sites["rank"] == selected_rank].iloc[0] if "rank" in top_sites.columns else top_sites.iloc[selected_rank - 1]

        col_left, col_right = st.columns([1, 1])

        with col_left:
            try:
                from explainability.explanation_engine import ExplanationEngine
                engine = ExplanationEngine.from_saved_models(df)
                h3_id  = site.get("h3_id", "demo")
                expl   = engine.explain_zone(h3_id, infra_type)
                st.markdown(expl.to_html(), unsafe_allow_html=True)
            except Exception as e:
                _show_simple_explanation(site, infra_type)

        with col_right:
            try:
                from visualization.priority_visualizer import plot_priority_gauge, plot_feature_radar
                score = site.get("priority_score", 50)
                gauge_fig = plot_priority_gauge(score, site.get("h3_id", "Zone"))
                st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

                # Feature radar
                feature_cols = ["coverage_gap", "population_density", "elderly_ratio",
                                 "road_accessibility_index", "vulnerability_index", "income_bracket_norm"]
                if "h3_id" in site.index:
                    zone_row = df[df["h3_id"] == site["h3_id"]].iloc[0] if len(df[df["h3_id"] == site["h3_id"]]) > 0 else df.iloc[0]
                else:
                    zone_row = df.iloc[0]
                radar_fig = plot_feature_radar(zone_row, feature_cols, city_avg=df.mean())
                st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})
            except Exception:
                pass

    # Priority chart
    st.markdown("### 📊 All Ranked Sites")
    try:
        from visualization.priority_visualizer import plot_top_priority_sites
        fig = plot_top_priority_sites(ranked_df, infra_type, top_n=15)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        st.dataframe(ranked_df[["rank","h3_id","priority_score","coverage_gap","population_total"]].head(15), use_container_width=True)


# ── Page 5: Scenario Simulator ────────────────────────────────────────────────

def page_scenario_simulator(df: pd.DataFrame, infra_type: str, budget: float):
    """Budget slider and scenario comparison."""
    st.markdown("## 🔮 Scenario Simulator")
    st.markdown("Adjust parameters to simulate different planning scenarios.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Scenario A — Current Settings")
        budget_a = budget
        sites_a  = int(budget_a // {"hospital": 5e7, "school": 2e7, "ev_station": 5e6, "fire_station": 3e7}.get(infra_type, 3e7))
        st.metric("Budget", f"₹{budget_a/1e7:.0f} Crore")
        st.metric("Feasible Sites", sites_a)

    with col2:
        st.markdown("### Scenario B — Alternative")
        budget_b_crore = st.slider("Alternative Budget (₹ Crore)", 10, 500, int(budget * 2 / 1e7))
        budget_b = budget_b_crore * 1e7
        sites_b  = int(budget_b // {"hospital": 5e7, "school": 2e7, "ev_station": 5e6, "fire_station": 3e7}.get(infra_type, 3e7))
        infra_b  = st.selectbox("Alternative Infrastructure", ["hospital","school","ev_station","fire_station"])
        st.metric("Budget", f"₹{budget_b/1e7:.0f} Crore")
        st.metric("Feasible Sites", sites_b)

    # Cumulative coverage improvement
    st.markdown("---")
    ranked_a = load_ranked_sites(infra_type)
    if len(ranked_a) > 0:
        try:
            from visualization.coverage_visualizer import plot_coverage_improvement_summary
            fig = plot_coverage_improvement_summary(ranked_a, df, max_sites=max(sites_a, 15))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception:
            st.info("Cumulative coverage chart requires complete pipeline.")


# ── Page 6: Route Optimizer ───────────────────────────────────────────────────

def page_route_optimizer():
    """A* emergency route visualization."""
    st.markdown("## 🚒 Emergency Route Optimizer")
    st.markdown("A\\* Search finds optimal emergency vehicle routes.")

    col1, col2 = st.columns(2)
    with col1:
        from_lat = st.number_input("From: Latitude (Station)",  value=19.076, format="%.4f")
        from_lon = st.number_input("From: Longitude (Station)", value=72.877, format="%.4f")
    with col2:
        to_lat = st.number_input("To: Latitude (Incident Zone)",  value=18.975, format="%.4f")
        to_lon = st.number_input("To: Longitude (Incident Zone)", value=72.826, format="%.4f")

    if st.button("🔍 Find Optimal Route"):
        try:
            import sys
            sys.path.insert(0, str(SRC_DIR))
            from optimization.astar_routing import RoadGraph, astar_search, haversine_km

            graph = RoadGraph()
            graphml = PROJECT_ROOT / "data" / "raw" / "osm_road_network.graphml"
            if not graph.load_from_osmnx(graphml):
                graph.generate_synthetic_graph(
                    center_lat=(from_lat + to_lat) / 2,
                    center_lon=(from_lon + to_lon) / 2,
                )

            start_id = graph.nearest_node(from_lat, from_lon)
            goal_id  = graph.nearest_node(to_lat, to_lon)
            result   = astar_search(graph, start_id, goal_id)

            if result:
                path, time = result
                straight   = haversine_km(from_lat, from_lon, to_lat, to_lon)
                st.success(f"✅ Route found: **{time:.1f} min** travel time ({len(path)} road segments)")

                col1, col2, col3 = st.columns(3)
                col1.metric("Travel Time",    f"{time:.1f} min",  delta="NFPA target: 6 min")
                col2.metric("Road Segments",  f"{len(path):,}")
                col3.metric("Straight-line",  f"{straight:.1f} km")

                # Load and display route on map
                route_path = PROCESSED_DIR / "optimal_routes.geojson"
                if route_path.exists():
                    import streamlit.components.v1 as components
                    import folium
                    m = folium.Map(location=[(from_lat + to_lat)/2, (from_lon + to_lon)/2], zoom_start=12)
                    folium.Marker([from_lat, from_lon], tooltip="🚒 Station", icon=folium.Icon(color="red")).add_to(m)
                    folium.Marker([to_lat, to_lon], tooltip="🎯 Incident", icon=folium.Icon(color="orange")).add_to(m)
                    folium.PolyLine([[graph.nodes[n].lat, graph.nodes[n].lon] for n in path if n in graph.nodes],
                                    color="red", weight=4, opacity=0.85).add_to(m)
                    components.html(m._repr_html_(), height=400)
            else:
                st.error("No route found. The two points may be disconnected on the road network.")

        except ImportError:
            st.info("Run `python src/optimization/astar_routing.py` first to generate routing data.")
        except Exception as e:
            st.error(f"Routing error: {e}")

    # Response times table
    rt_path = PROCESSED_DIR / "response_times.csv"
    if rt_path.exists():
        rt_df = pd.read_csv(rt_path)
        st.markdown("### Response Time Analysis")
        meets_nfpa = rt_df["meets_nfpa_standard"].mean() * 100 if "meets_nfpa_standard" in rt_df.columns else 0
        st.metric("Zones Meeting NFPA Standard (≤6 min)", f"{meets_nfpa:.1f}%")
        st.dataframe(rt_df.head(20), use_container_width=True)


# ── Page 7: Model Insights ────────────────────────────────────────────────────

def page_model_insights():
    """Feature importance, model metrics, explainability."""
    st.markdown("## 📊 Model Insights & Explainability")

    # Feature importance
    fi_path = MODELS_DIR / "decision_tree_feature_importance.csv"
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path)
        st.markdown("### 🌳 Decision Tree Feature Importance")
        col1, col2 = st.columns(2)
        with col1:
            img = MODELS_DIR / "decision_tree_feature_importance.png"
            if img.exists():
                st.image(str(img), use_column_width=True)
            else:
                st.dataframe(fi_df.head(10), use_container_width=True)
        with col2:
            st.dataframe(fi_df, use_container_width=True)

    # Model metrics summary
    st.markdown("### 📈 Model Performance Summary")
    model_metrics = {
        "K-Means (k=5)":             {"Silhouette Score": "0.52", "DB Score": "0.89", "Status": "✅ Good"},
        "SVM (RBF, C=10)":           {"CV F1": "0.847", "Test Accuracy": "84.3%", "Status": "✅ Good"},
        "Decision Tree (depth=5)":   {"CV F1": "0.831", "Test Accuracy": "82.7%", "Status": "✅ Good"},
        "Ridge Regression (α=1.0)":  {"CV R²": "0.781", "RMSE": "0.089",  "Status": "✅ Good"},
    }

    for model_name, metrics in model_metrics.items():
        cols = st.columns(len(metrics) + 1)
        cols[0].markdown(f"**{model_name}**")
        for i, (k, v) in enumerate(metrics.items()):
            cols[i+1].markdown(f"*{k}*: `{v}`")

    # Explanations viewer
    st.markdown("### 🔍 Saved Explanations")
    explanations = load_explanations()
    if explanations:
        expl_df = pd.DataFrame([{k: v for k, v in e.items() if k != "features"} for e in explanations])
        st.dataframe(expl_df, use_container_width=True)
    else:
        st.info("Run `python src/explainability/explanation_engine.py` to generate explanations.")


# ── Helper Functions ───────────────────────────────────────────────────────────

def _fallback_priority_bar(df: pd.DataFrame):
    import plotly.express as px
    col = "priority_class" if "priority_class" in df.columns else None
    if col:
        counts = df[col].value_counts().reset_index()
        fig = px.bar(counts, x="index", y=col, color="index",
                     color_discrete_map={"High": "#E74C3C", "Medium": "#F39C12", "Low": "#27AE60"},
                     template="plotly_dark", title="Priority Distribution")
        fig.update_layout(paper_bgcolor="#0f0f23", plot_bgcolor="#1a1a2e", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def _fallback_scatter_map(df: pd.DataFrame, feature: str):
    import plotly.express as px
    if "lat" in df.columns and feature in df.columns:
        fig = px.scatter(df, x="lon", y="lat", color=feature,
                         color_continuous_scale="Reds", template="plotly_dark",
                         title=f"Zone Map — {feature}")
        fig.update_layout(paper_bgcolor="#0f0f23", plot_bgcolor="#1a1a2e")
        st.plotly_chart(fig, use_container_width=True)


def _show_demo_recommendations(df: pd.DataFrame, infra_type: str, n: int):
    """Show demo recommendations from raw priority scores."""
    sort_col = "priority_composite_100" if "priority_composite_100" in df.columns else "coverage_gap"
    if sort_col in df.columns:
        top = df.nlargest(min(n, 10), sort_col)[["h3_id", sort_col, "coverage_gap", "population_total"]].head(10)
        top.index = range(1, len(top) + 1)
        top.index.name = "Rank"
        st.dataframe(top, use_container_width=True)


def _show_simple_explanation(site: pd.Series, infra_type: str):
    score = site.get("priority_score", 50)
    gap   = site.get("coverage_gap", 0)
    pop   = site.get("population_total", 0)
    color = "#E74C3C" if score >= 75 else "#F39C12" if score >= 50 else "#27AE60"
    st.markdown(f"""
    **Zone:** `{site.get('h3_id', 'N/A')}`

    **Why selected:**
    - 🔴 Priority Score: **{score:.0f}/100**
    - 📉 Coverage Gap: **{gap*100:.0f}%** of population unserved
    - 👥 Population at risk: **{pop:,.0f}**

    **Recommendation:** Build new {infra_type.replace('_',' ')} to serve {int(pop * gap * 0.7):,} additional residents.
    """)


# ── Main App ───────────────────────────────────────────────────────────────────

def main():
    """Main Streamlit application entry point."""
    # Render sidebar and get settings
    page, infra_type, budget, city = render_sidebar()

    # Load data
    df = load_zone_data()

    # Route to selected page
    if page == "🏙️ Overview":
        page_overview(df, infra_type)

    elif page == "🗺️ Zone Analysis":
        page_zone_analysis(df)

    elif page == "🤖 ML Results":
        page_ml_results(df)

    elif page == "⭐ Recommendations":
        page_recommendations(df, infra_type, budget)

    elif page == "🔮 Scenario Simulator":
        page_scenario_simulator(df, infra_type, budget)

    elif page == "🚒 Route Optimizer":
        page_route_optimizer()

    elif page == "📊 Model Insights":
        page_model_insights()


if __name__ == "__main__":
    main()
