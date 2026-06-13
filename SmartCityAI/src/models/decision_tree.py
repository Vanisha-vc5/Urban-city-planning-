"""
decision_tree.py
================
Phase 5 - Machine Learning: Decision Tree Classification & Rule Extraction

PURPOSE:
    Trains an interpretable Decision Tree for priority classification.
    Unlike SVM (black box), the Decision Tree produces human-readable
    IF-THEN rules that city planners can understand and trust.

WHY DECISION TREE?
    - Fully interpretable: generates explicit rules
    - No feature scaling required (uses thresholds, not distances)
    - Feature importance: ranks which features drive decisions
    - Visualizable: export as image or text
    - Foundation for XGBoost/Random Forest explanations
    - Perfect for stakeholder communication in urban planning

EXPLAINABILITY (Urban Planner perspective):
    The tree says:
    "IF coverage_gap > 0.6 AND population_density > 15000 AND elderly_ratio > 0.12
     THEN priority = HIGH (98% of such zones)"

    This is more trustworthy than SVM saying "probability=0.91" without explanation.

DEPTH SELECTION:
    - depth=3: Easy to visualize, may underfit
    - depth=5: Good balance (chosen for production)
    - depth=8: May overfit on noisy GIS data
    - depth=None: Full tree (diagnostic only)

    5-fold CV on max_depth [3,4,5,6,7,8] to select optimal.

FEATURE IMPORTANCE:
    Uses Gini importance (MDI — Mean Decrease in Impurity).
    Validated against Permutation Importance (removes bias for high-cardinality features).

USAGE:
    python decision_tree.py
    python decision_tree.py --max-depth 5 --extract-rules

OUTPUT:
    models/decision_tree_model.pkl
    models/decision_tree_visualization.png
    models/decision_tree_rules.txt
    data/processed/dt_predictions.csv
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    export_text,
    plot_tree,
)

# ── Configuration ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("decision_tree")

BASE_DIR      = Path(__file__).resolve().parents[3]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

DEFAULT_DT_FEATURES = [
    "coverage_gap",
    "population_density_log",
    "elderly_ratio",
    "vulnerability_index",
    "dist_nearest_hospital",
    "road_accessibility_index",
    "income_bracket_norm",
    "infrastructure_need_score",
    "density_gap_interaction",
    "emergency_response_time_min",
    "multi_coverage_gap",
]

CLASS_NAMES = ["High", "Low", "Medium"]  # Alphabetical (sklearn convention)


# ── Data Preparation ──────────────────────────────────────────────────────────

def prepare_dt_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "priority_class",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare Decision Tree training data.

    Args:
        df:           Feature DataFrame
        feature_cols: Input features
        target_col:   Target column

    Returns:
        (X_train, X_test, y_train, y_test, available_features)
    """
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        available = df.select_dtypes(include=np.number).columns[:8].tolist()
    log.info(f"DT features ({len(available)}): {available}")

    # Generate target if missing
    if target_col not in df.columns:
        df = df.copy()
        if "priority_composite_100" in df.columns:
            df[target_col] = pd.cut(df["priority_composite_100"], [0,33,66,100], labels=["Low","Medium","High"], include_lowest=True)
        elif "priority_score" in df.columns:
            df[target_col] = pd.cut(df["priority_score"], [0,33,66,100], labels=["Low","Medium","High"], include_lowest=True)
        else:
            rng = np.random.default_rng(42)
            df[target_col] = rng.choice(["Low","Medium","High"], len(df))

    X = df[available].fillna(df[available].median()).values
    y = df[target_col].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, available


# ── Depth Selection ───────────────────────────────────────────────────────────

def select_optimal_depth(
    X_train: np.ndarray,
    y_train: np.ndarray,
    depth_range: range = range(2, 10),
) -> int:
    """
    Select optimal tree depth via cross-validated F1 score.

    Plots accuracy vs depth to show the bias-variance tradeoff:
    - Too shallow (depth<3): high bias, underfits (misses real patterns)
    - Too deep (depth>7): high variance, overfits (memorizes training noise)

    Args:
        X_train:     Training features
        y_train:     Training labels
        depth_range: Range of depths to evaluate

    Returns:
        Optimal max_depth integer
    """
    log.info("Selecting optimal tree depth...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for depth in depth_range:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE, class_weight="balanced")
        scores = cross_val_score(dt, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
        results.append((depth, scores.mean(), scores.std()))
        log.info(f"  depth={depth}: F1={scores.mean():.4f} ± {scores.std():.4f}")

    # Find best
    best = max(results, key=lambda x: x[1])
    optimal_depth = best[0]
    log.info(f"  Optimal depth: {optimal_depth} (F1={best[1]:.4f})")

    # Plot depth vs accuracy
    depths = [r[0] for r in results]
    means  = [r[1] for r in results]
    stds   = [r[2] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(depths, means, "bо-", linewidth=2, markersize=8, label="CV F1 Score")
    ax.fill_between(depths,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color="blue")
    ax.axvline(x=optimal_depth, color="red", linestyle="--", linewidth=2, label=f"Optimal depth={optimal_depth}")
    ax.set_xlabel("Tree Max Depth", fontsize=12)
    ax.set_ylabel("CV Weighted F1 Score", fontsize=12)
    ax.set_title("Decision Tree: Depth vs Performance (Bias-Variance Tradeoff)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    path = MODELS_DIR / "decision_tree_depth_selection.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Depth selection plot → {path}")

    return optimal_depth


# ── Model Training ────────────────────────────────────────────────────────────

def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int = 5,
) -> DecisionTreeClassifier:
    """
    Train the Decision Tree classifier.

    Parameters:
        criterion="gini":     Gini impurity (vs entropy — equivalent results, slightly faster)
        class_weight="balanced": Handles class imbalance without oversampling
        min_samples_leaf=5:   Prevents very small leaves (reduces overfitting on GIS noise)
        min_impurity_decrease: Splits only if purity gain > threshold (regularization)

    Args:
        X_train:   Training features
        y_train:   Training labels
        max_depth: Maximum tree depth

    Returns:
        Fitted DecisionTreeClassifier
    """
    log.info(f"Training Decision Tree (max_depth={max_depth})...")

    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion="gini",
        class_weight="balanced",
        min_samples_leaf=5,
        min_impurity_decrease=0.001,
        random_state=RANDOM_STATE,
    )
    dt.fit(X_train, y_train)

    log.info(f"  Actual depth:    {dt.get_depth()}")
    log.info(f"  Number of nodes: {dt.tree_.node_count}")
    log.info(f"  Number of leaves:{dt.get_n_leaves()}")

    return dt


# ── Feature Importance ────────────────────────────────────────────────────────

def analyze_feature_importance(
    model: DecisionTreeClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Compute and compare Gini importance vs Permutation importance.

    Gini Importance (MDI):
        Σ (n_samples_reaching_node / n_samples) × impurity_decrease
        Fast, but biased toward high-cardinality features.

    Permutation Importance:
        Shuffles each feature; measures accuracy drop.
        Unbiased — recommended for final feature ranking.

    Args:
        model:         Fitted DecisionTreeClassifier
        X_test:        Test features
        y_test:        Test labels
        feature_names: Feature name list

    Returns:
        DataFrame with both importance measures
    """
    # Gini importance
    gini_imp = model.feature_importances_

    # Permutation importance
    perm_result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    perm_imp = perm_result.importances_mean

    imp_df = pd.DataFrame({
        "feature":               feature_names,
        "gini_importance":       gini_imp.round(4),
        "permutation_importance": perm_imp.round(4),
        "rank_gini":             pd.Series(gini_imp).rank(ascending=False).astype(int).values,
        "rank_perm":             pd.Series(perm_imp).rank(ascending=False).astype(int).values,
    }).sort_values("permutation_importance", ascending=False)

    log.info(f"\nFeature Importance (Top 10 by Permutation):\n{imp_df.head(10).to_string(index=False)}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    top_n = min(12, len(feature_names))
    top_gini = imp_df.sort_values("gini_importance", ascending=False).head(top_n)
    ax1.barh(top_gini["feature"], top_gini["gini_importance"], color="#3498DB", alpha=0.85)
    ax1.set_xlabel("Gini Importance (MDI)", fontsize=11)
    ax1.set_title("Decision Tree — Gini Feature Importance", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="x", alpha=0.3)

    top_perm = imp_df.head(top_n)
    ax2.barh(top_perm["feature"], top_perm["permutation_importance"], color="#E74C3C", alpha=0.85)
    ax2.set_xlabel("Permutation Importance", fontsize=11)
    ax2.set_title("Decision Tree — Permutation Feature Importance", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = MODELS_DIR / "decision_tree_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Feature importance plot → {path}")

    return imp_df


# ── Rule Extraction ───────────────────────────────────────────────────────────

def extract_decision_rules(
    model: DecisionTreeClassifier,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Extract human-readable IF-THEN rules from the Decision Tree.

    This is the KEY EXPLAINABILITY feature of Decision Trees.
    Output format:

    |--- coverage_gap <= 0.500
    |   |--- population_density_log <= 9.210
    |   |   |--- class: Low
    |   |--- population_density_log > 9.210
    |   |   |--- elderly_ratio <= 0.100
    |   |   |   |--- class: Medium
    |   |   |--- elderly_ratio > 0.100
    |   |   |   |--- class: High

    Args:
        model:         Fitted DecisionTreeClassifier
        feature_names: Feature names for labeling
        class_names:   Class names

    Returns:
        Text rule representation
    """
    if class_names is None:
        class_names = model.classes_

    rules_text = export_text(
        model,
        feature_names=feature_names,
        class_names=list(class_names),
        show_weights=True,
        max_depth=model.get_depth(),
    )

    log.info(f"\nDecision Rules (depth {model.get_depth()}):\n{rules_text[:2000]}...")

    # Save rules
    rules_path = MODELS_DIR / "decision_tree_rules.txt"
    with open(rules_path, "w") as f:
        f.write("SMARTCITYAI — DECISION TREE RULES\n")
        f.write("=" * 60 + "\n")
        f.write("Generated by: decision_tree.py\n\n")
        f.write("HOW TO READ:\n")
        f.write("  Each path from root to leaf is one decision rule.\n")
        f.write("  The class at each leaf is the predicted priority.\n")
        f.write("  weights[class] shows how many training samples follow this rule.\n\n")
        f.write("=" * 60 + "\n\n")
        f.write(rules_text)

    log.info(f"Decision rules saved → {rules_path}")
    return rules_text


def visualize_tree(
    model: DecisionTreeClassifier,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Create high-resolution Decision Tree visualization.

    Args:
        model:         Fitted DecisionTreeClassifier
        feature_names: Feature names
        class_names:   Class names
    """
    if class_names is None:
        class_names = model.classes_

    fig_width  = max(20, 5 * 2 ** min(model.get_depth(), 4))
    fig_height = max(10, 3 * model.get_depth())
    fig, ax = plt.subplots(figsize=(min(fig_width, 40), min(fig_height, 25)))

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=list(class_names),
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        fontsize=9,
        ax=ax,
        max_depth=min(4, model.get_depth()),  # Limit depth for readability
    )

    ax.set_title("SmartCityAI — Infrastructure Priority Decision Tree",
                 fontsize=14, fontweight="bold", pad=20)
    path = MODELS_DIR / "decision_tree_visualization.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"Tree visualization saved → {path}")


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train Decision Tree for SmartCityAI")
    parser.add_argument("--max-depth",     type=int, default=None, help="Tree depth (auto if not set)")
    parser.add_argument("--extract-rules", action="store_true",    help="Print decision rules")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SmartCityAI — Decision Tree Classifier")
    log.info("=" * 60)

    # Load data
    for path in [PROCESSED_DIR / "ml_features.csv", PROCESSED_DIR / "zone_features.csv"]:
        if path.exists():
            df = pd.read_csv(path)
            break
    else:
        log.error("No feature CSV found.")
        sys.exit(1)

    log.info(f"Loaded {len(df):,} zones")

    # Merge cluster IDs
    cluster_path = PROCESSED_DIR / "zone_clusters.csv"
    if cluster_path.exists():
        df = df.merge(pd.read_csv(cluster_path)[["h3_id","cluster_id"]], on="h3_id", how="left")

    # Prepare data
    X_train, X_test, y_train, y_test, available_features = prepare_dt_data(df, DEFAULT_DT_FEATURES)

    # Select depth
    max_depth = args.max_depth
    if max_depth is None:
        max_depth = select_optimal_depth(X_train, y_train)

    # Train
    model = train_decision_tree(X_train, y_train, max_depth)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    log.info(f"\nTest Accuracy: {accuracy:.4f}")
    log.info(f"\n{classification_report(y_test, y_pred)}")

    # Cross-validate
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, df[available_features].fillna(0).values,
                                df["priority_class"].astype(str).values if "priority_class" in df.columns
                                else np.zeros(len(df)),
                                cv=cv, scoring="f1_weighted", n_jobs=-1)
    log.info(f"CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    imp_df = analyze_feature_importance(model, X_test, y_test, available_features)

    # Visualize
    visualize_tree(model, available_features)

    # Extract rules
    if args.extract_rules or True:
        extract_decision_rules(model, available_features)

    # Predictions on full dataset
    X_full = df[available_features].fillna(0).values
    df["dt_priority"] = model.predict(X_full)
    df[["h3_id", "dt_priority"]].to_csv(PROCESSED_DIR / "dt_predictions.csv", index=False)

    # Save model
    with open(MODELS_DIR / "decision_tree_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": available_features, "classes": list(model.classes_)}, f)
    log.info(f"Model saved → {MODELS_DIR}/decision_tree_model.pkl")

    imp_df.to_csv(MODELS_DIR / "decision_tree_feature_importance.csv", index=False)
    log.info("Decision Tree training complete!")


if __name__ == "__main__":
    main()
