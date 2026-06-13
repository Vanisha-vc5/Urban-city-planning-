"""
cluster_visualizer.py
=====================
Phase 8 - Visualization: K-Means Cluster Analysis Charts

Generates: scatter plots, radar charts, cluster maps, feature distributions.
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)
BASE_DIR      = Path(__file__).resolve().parents[3]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"

CLUSTER_COLORS = ["#E74C3C", "#E67E22", "#3498DB", "#9B59B6", "#2ECC71",
                  "#1ABC9C", "#F39C12", "#2C3E50", "#8E44AD", "#16A085"]

CLUSTER_LABELS = {
    0: "Dense Urban Core",
    1: "Underserved Periphery",
    2: "Affluent Suburbs",
    3: "High-Need Slums",
    4: "Transitional Growth Areas",
}


def plot_pca_clusters(
    df: pd.DataFrame,
    feature_cols: List[str],
    cluster_col: str = "cluster_id",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    2D PCA scatter plot of K-Means clusters.

    Reduces n-dimensional feature space to 2D for visualization.
    Color-coded by cluster. Size by population density.

    PCA axes labels explain % variance retained.
    """
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        return plt.figure()

    X = df[available].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f23")

    clusters = df[cluster_col].fillna(0).astype(int) if cluster_col in df.columns else np.zeros(len(df))

    # Size: population density
    sizes = 20
    if "population_density" in df.columns:
        sizes = np.clip(df["population_density"].fillna(1000) / 500, 5, 80)

    for c_id in sorted(clusters.unique()):
        mask = clusters == c_id
        label = CLUSTER_LABELS.get(c_id, f"Cluster {c_id}")
        color = CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)]
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=color, s=sizes[mask] if hasattr(sizes, '__len__') else sizes,
            alpha=0.75, label=label, edgecolors="none",
        )

    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)", color="white", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)", color="white", fontsize=12)
    ax.set_title("K-Means Cluster Analysis — PCA Projection", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    legend = ax.legend(loc="upper right", framealpha=0.3, labelcolor="white",
                       facecolor="#1a1a2e", edgecolor="#444")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        log.info(f"PCA plot saved → {save_path}")
    return fig


def plot_cluster_feature_heatmap(
    df: pd.DataFrame,
    feature_cols: List[str],
    cluster_col: str = "cluster_id",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Heatmap of normalized cluster means per feature.

    Rows = clusters, Columns = features.
    Color = normalized mean (0=blue, 1=red).
    Useful to characterize each cluster archetype.
    """
    available = [c for c in feature_cols if c in df.columns and c != cluster_col][:12]
    if not available or cluster_col not in df.columns:
        return plt.figure()

    cluster_means = df.groupby(cluster_col)[available].mean()
    # Normalize each feature to [0,1] for heatmap scale
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    heatmap_vals = scaler.fit_transform(cluster_means.T).T  # shape: (n_clusters, n_features)

    fig, ax = plt.subplots(figsize=(max(12, len(available)), max(4, len(cluster_means) * 1.2)))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#1a1a2e")

    im = ax.imshow(heatmap_vals, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    short_names = [c.replace("_", "\n")[:15] for c in available]
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(short_names, color="white", fontsize=8, rotation=45, ha="right")
    cluster_labels = [CLUSTER_LABELS.get(c, f"C{c}") for c in cluster_means.index]
    ax.set_yticks(range(len(cluster_means)))
    ax.set_yticklabels(cluster_labels, color="white", fontsize=10)
    ax.set_title("Cluster Feature Profile Heatmap (normalized means)", color="white", fontsize=13, fontweight="bold")

    for i in range(len(cluster_means)):
        for j in range(len(available)):
            val = cluster_means.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white", fontsize=7)

    plt.colorbar(im, ax=ax, label="Normalized Value", shrink=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        log.info(f"Heatmap saved → {save_path}")
    return fig


def plot_cluster_distribution(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Donut chart of cluster sizes."""
    if cluster_col not in df.columns:
        return plt.figure()

    counts = df[cluster_col].value_counts().sort_index()
    labels = [CLUSTER_LABELS.get(c, f"C{c}") for c in counts.index]
    colors = [CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in counts.index]

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#0f0f23")

    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"linewidth": 2, "edgecolor": "#0f0f23"},
    )
    # Donut hole
    centre_circle = plt.Circle((0, 0), 0.55, fc="#0f0f23")
    ax.add_patch(centre_circle)

    for text in texts:
        text.set_color("white")
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(9)

    ax.text(0, 0, f"{counts.sum():,}\nZones", ha="center", va="center",
            color="white", fontsize=14, fontweight="bold")
    ax.set_title("Cluster Size Distribution", color="white", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    csv = PROCESSED_DIR / "zone_clusters.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        features = PROCESSED_DIR / "ml_features.csv"
        if features.exists():
            df = df.merge(pd.read_csv(features), on="h3_id", how="left")
        feature_cols = [c for c in df.columns if c not in ["h3_id","cluster_id","cluster_label"]]
        plot_pca_clusters(df, feature_cols, save_path=MODELS_DIR / "cluster_pca.png")
        plot_cluster_feature_heatmap(df, feature_cols, save_path=MODELS_DIR / "cluster_heatmap.png")
        plot_cluster_distribution(df, save_path=MODELS_DIR / "cluster_donut.png")
        print("Cluster visualizations saved.")
    else:
        print("Run kmeans.py first.")
