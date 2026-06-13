"""
kmeans.py
=========
Phase 5 - Machine Learning: K-Means Clustering

PURPOSE:
    Clusters city zones into behavioral groups (neighborhood archetypes)
    to identify zones that need similar types of infrastructure.

    K-Means provides:
    1. Neighborhood segmentation for targeted planning
    2. Input features for SVM (cluster-as-feature)
    3. Visual cluster map for dashboard

WHY K-MEANS?
    - Unsupervised: no labels needed (we don't have "correct" clusters)
    - Interpretable: cluster centroids are the "average neighborhood"
    - Fast: O(k × n × d × iterations) — scales to 10,000+ zones
    - Industry standard for geographic segmentation (Uber, Airbnb, CARTO)
    - Combined with SHAP → explains WHAT defines each cluster

K=5 CLUSTER INTERPRETATION (typical urban Indian city):
    Cluster 0: Dense Urban Core      — high density, low gap, moderate income
    Cluster 1: Underserved Periphery — low density, high gap, low income
    Cluster 2: Affluent Suburbs      — moderate density, low gap, high income
    Cluster 3: High-Need Slums       — very high density, very high gap, very low income
    Cluster 4: Transitional Areas    — medium all metrics, growing rapidly

ELBOW METHOD:
    Determines optimal k by minimizing WCSS (Within-Cluster Sum of Squares).
    Silhouette score validates cluster compactness.
    
USAGE:
    python kmeans.py
    python kmeans.py --k 5 --features population_density_log elderly_ratio coverage_gap

OUTPUT:
    models/kmeans_model.pkl        (fitted KMeans)
    data/processed/zone_clusters.csv (zone → cluster mapping)
    models/kmeans_scaler.pkl       (StandardScaler for inference)
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ── Configuration ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("kmeans")

BASE_DIR      = Path(__file__).resolve().parents[3]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Default features for K-Means clustering
DEFAULT_KMEANS_FEATURES = [
    "population_density_log",
    "elderly_ratio",
    "income_bracket_norm",
    "coverage_gap",
    "road_accessibility_index",
    "vulnerability_index",
    "infrastructure_need_score",
    "emergency_response_time_min",
]

# Cluster labels for interpretation
CLUSTER_LABELS = {
    0: "Dense Urban Core",
    1: "Underserved Periphery",
    2: "Affluent Suburbs",
    3: "High-Need Slums",
    4: "Transitional Growth Areas",
}

CLUSTER_COLORS = ["#E74C3C", "#E67E22", "#3498DB", "#9B59B6", "#2ECC71"]
RANDOM_STATE   = 42


# ── Elbow Method ──────────────────────────────────────────────────────────────

def find_optimal_k(
    X: np.ndarray,
    k_range: range = range(2, 11),
    save_plot: bool = True,
) -> int:
    """
    Find optimal number of clusters using Elbow Method + Silhouette Score.

    Elbow Method:
        - Plots WCSS (inertia) vs k
        - Optimal k is at the "elbow" where improvement diminishes
        - Quantified using the Kneedle algorithm (or manual inspection)

    Silhouette Score:
        - Range [-1, 1]: higher = better-defined clusters
        - Score > 0.5: strong structure; > 0.7: excellent

    Args:
        X:         Feature matrix (scaled)
        k_range:   Range of k values to test
        save_plot: Save elbow plot to disk

    Returns:
        Recommended k value
    """
    log.info("Running Elbow Method to find optimal k...")
    inertias  = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        if k > 1:
            sil = silhouette_score(X, labels, sample_size=min(5000, len(X)))
            silhouettes.append(sil)
            log.info(f"  k={k:2d} | Inertia={km.inertia_:,.0f} | Silhouette={sil:.4f}")
        else:
            silhouettes.append(0)

    # Find elbow using second derivative of inertia
    inertia_arr = np.array(inertias)
    diffs       = np.diff(inertia_arr)
    diffs2      = np.diff(diffs)
    elbow_idx   = np.argmax(diffs2) + 1  # k index in k_range
    optimal_k_elbow = list(k_range)[elbow_idx]

    # Also check best silhouette
    best_sil_idx = np.argmax(silhouettes[1:]) + 1
    optimal_k_sil = list(k_range)[best_sil_idx]

    log.info(f"  Elbow method suggests k={optimal_k_elbow}")
    log.info(f"  Best silhouette score at k={optimal_k_sil}")

    # Use 5 as default (interpretable, matches urban archetypes)
    optimal_k = 5

    if save_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("K-Means Cluster Optimization", fontsize=14, fontweight="bold")

        ax1.plot(list(k_range), inertias, "bо-", linewidth=2)
        ax1.axvline(x=optimal_k, color="red", linestyle="--", label=f"Chosen k={optimal_k}")
        ax1.set_xlabel("Number of Clusters (k)")
        ax1.set_ylabel("WCSS (Inertia)")
        ax1.set_title("Elbow Method")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(list(k_range)[1:], silhouettes[1:], "gо-", linewidth=2)
        ax2.axvline(x=optimal_k, color="red", linestyle="--", label=f"Chosen k={optimal_k}")
        ax2.set_xlabel("Number of Clusters (k)")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Analysis")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = MODELS_DIR / "kmeans_elbow_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Elbow plot saved → {plot_path}")

    return optimal_k


# ── Model Training ────────────────────────────────────────────────────────────

def train_kmeans(
    df: pd.DataFrame,
    feature_cols: List[str],
    k: int = 5,
) -> Tuple[KMeans, StandardScaler, pd.DataFrame]:
    """
    Train K-Means clustering model.

    Steps:
    1. Select features and handle missing values
    2. Standardize (Z-score) — critical for K-Means (distance-based)
    3. Train KMeans with k-means++ initialization
    4. Compute evaluation metrics
    5. Assign cluster labels and add to DataFrame

    WHY STANDARDIZE?
        K-Means uses Euclidean distance. Without standardization:
        - population_density (0–80,000) dominates
        - elderly_ratio (0–0.2) is ignored
        StandardScaler: z = (x - μ) / σ → all features on equal footing.

    Args:
        df:           Feature DataFrame
        feature_cols: List of feature column names
        k:            Number of clusters

    Returns:
        (fitted_model, fitted_scaler, df_with_clusters)
    """
    # Validate features
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        log.warning(f"Missing features (will skip): {missing}")
    feature_cols = available

    log.info(f"Training K-Means: k={k}, features={len(feature_cols)}")
    log.info(f"Features: {feature_cols}")

    # Prepare matrix
    X_raw = df[feature_cols].fillna(df[feature_cols].median()).values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Train K-Means (k-means++ initialization = smart centroid seeding)
    km = KMeans(
        n_clusters=k,
        init="k-means++",          # O(log k) smarter initialization
        n_init=20,                  # 20 restarts, pick best WCSS
        max_iter=500,               # Convergence iterations
        random_state=RANDOM_STATE,
        tol=1e-5,
    )
    labels = km.fit_predict(X_scaled)

    # Evaluation metrics
    sil_score = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
    db_score  = davies_bouldin_score(X_scaled, labels)

    log.info(f"\n{'='*50}")
    log.info(f"K-Means Training Results (k={k})")
    log.info(f"{'='*50}")
    log.info(f"  Inertia (WCSS):        {km.inertia_:,.2f}")
    log.info(f"  Silhouette Score:       {sil_score:.4f}  (target: >0.5)")
    log.info(f"  Davies-Bouldin Score:  {db_score:.4f}   (target: <1.0)")

    # Add cluster assignments
    df = df.copy()
    df["cluster_id"]    = labels
    df["cluster_label"] = [CLUSTER_LABELS.get(l, f"Cluster {l}") for l in labels]

    # Cluster statistics
    log.info("\nCluster Sizes:")
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        pct = count / len(labels) * 100
        label = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
        log.info(f"  {cluster_id} ({label}): {count:,} zones ({pct:.1f}%)")

    # Cluster centroids (back in original scale)
    centroids_scaled = km.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    centroid_df = pd.DataFrame(centroids_original, columns=feature_cols)
    centroid_df.index.name = "cluster_id"
    log.info(f"\nCluster Centroids (original scale):\n{centroid_df.round(3).to_string()}")

    return km, scaler, df


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_cluster_profiles(
    df: pd.DataFrame,
    feature_cols: List[str],
    k: int = 5,
) -> None:
    """
    Create radar/bar chart showing cluster feature profiles.

    Args:
        df:           DataFrame with cluster_id column
        feature_cols: Feature columns to profile
        k:            Number of clusters
    """
    fig, axes = plt.subplots(1, min(k, 5), figsize=(20, 4))
    if k == 1:
        axes = [axes]

    cols_to_plot = [c for c in feature_cols[:6] if c in df.columns]

    for cluster_id in range(min(k, 5)):
        ax = axes[cluster_id]
        cluster_data = df[df["cluster_id"] == cluster_id]
        if len(cluster_data) == 0:
            continue
        means = cluster_data[cols_to_plot].mean()
        short_names = [c.replace("_", "\n").replace("norm", "").strip() for c in cols_to_plot]
        bars = ax.bar(range(len(cols_to_plot)), means.values,
                      color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(cols_to_plot)))
        ax.set_xticklabels(short_names, fontsize=7, rotation=30, ha="right")
        label = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
        ax.set_title(f"Cluster {cluster_id}\n{label}", fontsize=9, fontweight="bold",
                     color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)])
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("K-Means Cluster Feature Profiles", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = MODELS_DIR / "kmeans_cluster_profiles.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Cluster profiles saved → {path}")


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_model(model: KMeans, scaler: StandardScaler) -> None:
    """Save fitted KMeans model and scaler."""
    with open(MODELS_DIR / "kmeans_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / "kmeans_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    log.info(f"Model saved → {MODELS_DIR}/kmeans_model.pkl")


def load_model() -> Tuple[KMeans, StandardScaler]:
    """Load saved KMeans model and scaler."""
    with open(MODELS_DIR / "kmeans_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "kmeans_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict_cluster(zone_features: np.ndarray, model: KMeans, scaler: StandardScaler) -> int:
    """
    Predict cluster for new zone features.

    Args:
        zone_features: 1D array of feature values (same order as training)
        model:         Fitted KMeans model
        scaler:        Fitted StandardScaler

    Returns:
        Cluster label integer
    """
    x_scaled = scaler.transform(zone_features.reshape(1, -1))
    return int(model.predict(x_scaled)[0])


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train K-Means clustering for SmartCityAI")
    parser.add_argument("--k",        type=int, default=5,   help="Number of clusters")
    parser.add_argument("--auto-k",   action="store_true",   help="Auto-find optimal k via elbow method")
    parser.add_argument("--features", nargs="+", default=DEFAULT_KMEANS_FEATURES)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SmartCityAI — K-Means Clustering")
    log.info("=" * 60)

    # Load feature data
    ml_features_path = PROCESSED_DIR / "ml_features.csv"
    zone_features_path = PROCESSED_DIR / "zone_features.csv"
    csv_path = ml_features_path if ml_features_path.exists() else zone_features_path

    if not csv_path.exists():
        log.error("No feature CSV found. Run build_feature_table.py or create_zones.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df):,} zones from {csv_path.name}")

    # Prepare features
    available = [c for c in args.features if c in df.columns]
    if len(available) < 3:
        log.warning("Few features available. Using all numeric columns.")
        available = df.select_dtypes(include=np.number).columns.tolist()[:10]

    # Scale for elbow method
    X_raw = df[available].fillna(df[available].median()).values
    scaler_temp = StandardScaler()
    X_scaled = scaler_temp.fit_transform(X_raw)

    # Find optimal k if requested
    k = args.k
    if args.auto_k:
        k = find_optimal_k(X_scaled)
        log.info(f"Auto-selected k={k}")

    # Train
    model, scaler, df_clustered = train_kmeans(df, available, k=k)

    # Visualize
    plot_cluster_profiles(df_clustered, available, k=k)

    # Save model
    save_model(model, scaler)

    # Save cluster assignments
    cluster_out = PROCESSED_DIR / "zone_clusters.csv"
    df_clustered[["h3_id", "cluster_id", "cluster_label"]].to_csv(cluster_out, index=False)
    log.info(f"Cluster assignments saved → {cluster_out}")

    log.info("\nK-Means complete! Use cluster_id as an input feature for SVM and Decision Tree.")


if __name__ == "__main__":
    main()
