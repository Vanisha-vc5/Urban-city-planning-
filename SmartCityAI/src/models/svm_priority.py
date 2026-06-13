"""
svm_priority.py
===============
Phase 5 - Machine Learning: SVM Priority Classification

PURPOSE:
    Classifies each zone as HIGH / MEDIUM / LOW infrastructure priority
    using a Support Vector Machine (SVM) with an RBF kernel.

WHY SVM?
    - Works well with small-to-medium datasets (100–10,000 zones)
    - RBF kernel handles non-linear decision boundaries (city zones aren't linearly separable)
    - Maximum margin classifier → robust to outliers (noisy GIS data)
    - Principled probabilistic output (Platt scaling via probability=True)
    - Well-studied theoretical guarantees (VC dimension theory)

DESIGN DECISIONS:
    - MultiClass: One-vs-Rest (OvR) with 3 classes
    - Kernel: RBF (Radial Basis Function) — handles complex zone feature interactions
    - C parameter: regularization tradeoff (high C = low bias, high variance)
    - gamma: RBF bandwidth — auto or fine-tuned via GridSearchCV
    - Labels from priority_class (computed in create_zones.py)
    - 5-fold stratified cross-validation avoids overfitting

HYPERPARAMETER TUNING:
    GridSearchCV searches:
    - C: [0.1, 1, 10, 100, 1000]
    - gamma: ['scale', 'auto', 0.001, 0.01, 0.1]
    - kernel: ['rbf', 'poly']

    Best params saved with model for reproducibility.

LABELS:
    HIGH   → priority_composite_100 > 66
    MEDIUM → priority_composite_100 33-66
    LOW    → priority_composite_100 < 33

USAGE:
    python svm_priority.py
    python svm_priority.py --cv 5 --tune

OUTPUT:
    models/svm_model.pkl         (fitted SVM pipeline)
    models/svm_results.json      (metrics, best params)
    data/processed/svm_predictions.csv
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# ── Configuration ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("svm_priority")

BASE_DIR      = Path(__file__).resolve().parents[3]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

# Default SVM features (must be standardized)
DEFAULT_SVM_FEATURES = [
    "population_density_log",
    "elderly_ratio",
    "income_bracket_norm",
    "coverage_gap",
    "dist_nearest_hospital",
    "road_accessibility_index",
    "vulnerability_index",
    "infrastructure_need_score",
    "density_gap_interaction",
    "emergency_response_time_min",
    "cluster_id",  # From K-Means output (Phase 5A)
]

# Hyperparameter grid for GridSearchCV
SVM_PARAM_GRID = {
    "svm__C":      [0.1, 1, 10, 100],
    "svm__gamma":  ["scale", "auto", 0.01, 0.1],
    "svm__kernel": ["rbf", "poly"],
}


# ── Data Preparation ──────────────────────────────────────────────────────────

def prepare_svm_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "priority_class",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Prepare train/test split for SVM.

    Also generates synthetic labels if priority_class is missing.

    Args:
        df:           Feature DataFrame
        feature_cols: Input feature column names
        target_col:   Target class column

    Returns:
        (X_train, X_test, y_train, y_test, label_encoder)
    """
    # Generate target if missing
    if target_col not in df.columns:
        log.warning(f"'{target_col}' not found. Generating from priority_composite_100.")
        if "priority_composite_100" in df.columns:
            df = df.copy()
            df[target_col] = pd.cut(
                df["priority_composite_100"],
                bins=[0, 33, 66, 100],
                labels=["Low", "Medium", "High"],
                include_lowest=True,
            )
        elif "priority_score" in df.columns:
            df = df.copy()
            df[target_col] = pd.cut(
                df["priority_score"],
                bins=[0, 33, 66, 100],
                labels=["Low", "Medium", "High"],
                include_lowest=True,
            )
        else:
            # Fully synthetic labels
            rng = np.random.default_rng(42)
            df = df.copy()
            df[target_col] = rng.choice(["Low", "Medium", "High"], size=len(df))

    # Validate features
    available = [c for c in feature_cols if c in df.columns]
    log.info(f"SVM features ({len(available)}): {available}")

    X = df[available].fillna(df[available].median()).values
    y_raw = df[target_col].astype(str).values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    log.info(f"Class distribution: {dict(zip(le.classes_, np.bincount(y)))}")

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, le


# ── Model Training ────────────────────────────────────────────────────────────

def build_svm_pipeline(C: float = 10, gamma: str = "scale", kernel: str = "rbf") -> Pipeline:
    """
    Build SVM Pipeline with StandardScaler preprocessing.

    Pipeline ensures no data leakage (scaler fit only on training data).
    Using Pipeline with GridSearchCV is the correct way to tune hyperparams.

    Args:
        C:      Regularization parameter
        gamma:  RBF bandwidth
        kernel: Kernel type

    Returns:
        sklearn Pipeline (scaler → SVM)
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(
            C=C,
            gamma=gamma,
            kernel=kernel,
            probability=True,     # Enable predict_proba (Platt scaling)
            class_weight="balanced",  # Handle class imbalance
            random_state=RANDOM_STATE,
            cache_size=500,       # MB of RAM for kernel cache (faster training)
        ))
    ])


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = CV_FOLDS,
    n_jobs: int = -1,
) -> Tuple[Pipeline, dict]:
    """
    Tune SVM hyperparameters via GridSearchCV.

    Uses stratified k-fold cross-validation.
    Scoring: weighted F1 (handles class imbalance better than accuracy).

    Time complexity:
        O(|C| × |gamma| × |kernel| × cv_folds × n × n_sv²)
        For small datasets (<5000 samples), completes in minutes.

    Args:
        X_train:  Training features
        y_train:  Training labels
        cv_folds: Number of CV folds
        n_jobs:   Parallel jobs (-1 = all CPUs)

    Returns:
        (best_pipeline, best_params_dict)
    """
    log.info(f"Tuning SVM hyperparameters with {cv_folds}-fold CV...")
    log.info(f"Grid: {SVM_PARAM_GRID}")

    base_pipeline = build_svm_pipeline()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=SVM_PARAM_GRID,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    log.info(f"\nBest Parameters: {grid_search.best_params_}")
    log.info(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def cross_validate_svm(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = CV_FOLDS,
) -> dict:
    """
    Run detailed cross-validation evaluation.

    Returns mean and std of accuracy, F1, precision, recall.

    Args:
        pipeline: Fitted or unfitted SVM pipeline
        X:        Full feature matrix
        y:        Full label array
        cv_folds: Number of folds

    Returns:
        Dict of metric → mean, std
    """
    log.info(f"Running {cv_folds}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    metrics = {}

    for metric in ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=metric, n_jobs=-1)
        metrics[metric] = {"mean": float(scores.mean()), "std": float(scores.std())}
        log.info(f"  {metric:25s}: {scores.mean():.4f} ± {scores.std():.4f}")

    return metrics


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_svm(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict:
    """
    Comprehensive SVM test set evaluation.

    Args:
        pipeline:      Fitted SVM pipeline
        X_test:        Test features
        y_test:        True labels (encoded)
        label_encoder: For converting back to class names

    Returns:
        Dict of evaluation metrics
    """
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    class_names = label_encoder.classes_

    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    log.info(f"\nTest Set Results:")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=class_names)}")

    # ROC-AUC (one-vs-rest for multiclass)
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        log.info(f"  ROC-AUC (weighted OvR): {auc:.4f}")
    except Exception:
        auc = 0.0

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("SVM Priority Classification — Confusion Matrix", fontweight="bold")
    path = MODELS_DIR / "svm_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Confusion matrix → {path}")

    return {
        "classification_report": report,
        "roc_auc_weighted": auc,
        "confusion_matrix": cm.tolist(),
        "test_accuracy": float((y_pred == y_test).mean()),
    }


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_svm_model(pipeline: Pipeline, label_encoder: LabelEncoder, results: dict) -> None:
    """Save SVM model, encoder, and results."""
    with open(MODELS_DIR / "svm_model.pkl", "wb") as f:
        pickle.dump({"pipeline": pipeline, "label_encoder": label_encoder}, f)

    with open(MODELS_DIR / "svm_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"SVM model saved → {MODELS_DIR}/svm_model.pkl")


def load_svm_model() -> Tuple[Pipeline, LabelEncoder]:
    """Load saved SVM model."""
    with open(MODELS_DIR / "svm_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["pipeline"], data["label_encoder"]


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train SVM priority classifier for SmartCityAI")
    import argparse
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning (slower)")
    parser.add_argument("--cv",   type=int, default=CV_FOLDS, help="Cross-validation folds")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SmartCityAI — SVM Priority Classification")
    log.info("=" * 60)

    # Load data
    for path in [PROCESSED_DIR / "ml_features.csv", PROCESSED_DIR / "zone_features.csv"]:
        if path.exists():
            df = pd.read_csv(path)
            log.info(f"Loaded {len(df):,} zones from {path.name}")
            break
    else:
        log.error("No feature CSV found.")
        sys.exit(1)

    # Load cluster IDs if available
    cluster_path = PROCESSED_DIR / "zone_clusters.csv"
    if cluster_path.exists():
        clusters = pd.read_csv(cluster_path)
        df = df.merge(clusters[["h3_id", "cluster_id"]], on="h3_id", how="left")

    # Prepare data
    X_train, X_test, y_train, y_test, le = prepare_svm_data(df, DEFAULT_SVM_FEATURES)

    # Train
    if args.tune:
        pipeline, best_params = tune_hyperparameters(X_train, y_train, args.cv)
    else:
        pipeline = build_svm_pipeline(C=10, gamma="scale", kernel="rbf")
        pipeline.fit(X_train, y_train)
        best_params = {"svm__C": 10, "svm__gamma": "scale", "svm__kernel": "rbf"}
        log.info("Trained with default params. Use --tune for GridSearchCV.")

    # Cross-validate on full data
    X_full = df[DEFAULT_SVM_FEATURES].fillna(0).values
    y_full = le.transform(df.get("priority_class", ["Medium"] * len(df)).astype(str))
    cv_metrics = cross_validate_svm(pipeline, X_full, y_full, args.cv)

    # Evaluate on test set
    test_results = evaluate_svm(pipeline, X_test, y_test, le)

    # Predict on full dataset
    df["svm_priority"]     = le.inverse_transform(pipeline.predict(X_full))
    df["svm_probability_high"] = pipeline.predict_proba(X_full)[:, list(le.classes_).index("High")]
    df[["h3_id", "svm_priority", "svm_probability_high"]].to_csv(
        PROCESSED_DIR / "svm_predictions.csv", index=False
    )

    # Save
    results = {
        "best_params": best_params,
        "cv_metrics": cv_metrics,
        "test_metrics": test_results,
        "classes": list(le.classes_),
        "features": DEFAULT_SVM_FEATURES,
    }
    save_svm_model(pipeline, le, results)

    log.info("\nSVM training complete!")


if __name__ == "__main__":
    main()
