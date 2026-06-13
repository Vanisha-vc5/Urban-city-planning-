"""
population_regression.py
========================
Phase 5 - Machine Learning: Population Demand Regression

PURPOSE:
    Predicts future infrastructure demand using regression models.
    Answers: "How many hospital beds/school seats will Zone X need in 5 years?"

WHY REGRESSION?
    - Classification tells us WHERE to build (High/Medium/Low priority)
    - Regression tells us HOW MUCH capacity to build
    - Enables budget planning: "Build X beds if demand grows Y%"
    - Multi-model comparison: Ridge vs SVR vs Polynomial

MODELS COMPARED:
    1. Ridge Regression (L2 regularization)
       - Linear baseline
       - Handles correlated features (population features are correlated)
       - λ (alpha) tuned via RidgeCV

    2. Support Vector Regression (SVR, RBF kernel)
       - Non-linear regression
       - Handles feature interactions
       - Consistent with SVM Classification Phase

    3. Polynomial Regression (degree=2)
       - Captures quadratic effects (density²)
       - Useful for urban growth saturation modeling

TARGET VARIABLE:
    future_demand_5yr_norm:
        = MinMaxNorm(population_density × (1 + growth_rate)^5)
    This represents normalized 5-year projected demand.

FEATURES:
    Same as classification models (demographic + coverage + accessibility)
    plus temporal features (growth_rate_annual).

EVALUATION METRICS:
    - MAE  (Mean Absolute Error): interpretable as "how many people wrong"
    - RMSE (Root Mean Squared Error): penalizes large errors
    - R²   (Coefficient of Determination): variance explained (target: >0.7)
    - MAPE (Mean Absolute Percentage Error): relative accuracy

USAGE:
    python population_regression.py
    python population_regression.py --model ridge --alpha 1.0

OUTPUT:
    models/regression_ridge_model.pkl
    models/regression_svr_model.pkl
    models/regression_comparison.png
    data/processed/regression_predictions.csv
"""

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
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

# ── Configuration ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("population_regression")

BASE_DIR      = Path(__file__).resolve().parents[3]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

REGRESSION_FEATURES = [
    "population_density_log",
    "elderly_ratio",
    "youth_ratio",
    "income_bracket_norm",
    "coverage_gap",
    "road_accessibility_index",
    "growth_rate_annual",
    "vulnerability_index",
    "dist_nearest_hospital",
    "emergency_response_time_min",
]

TARGET_COL = "future_demand_5yr_norm"


# ── Data Preparation ──────────────────────────────────────────────────────────

def prepare_regression_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = TARGET_COL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare regression dataset.

    Generates synthetic target if not in data.

    Args:
        df:           Feature DataFrame
        feature_cols: Input feature column names
        target_col:   Target variable column

    Returns:
        (X_train, X_test, y_train, y_test, available_features)
    """
    available = [c for c in feature_cols if c in df.columns]
    log.info(f"Regression features ({len(available)}): {available}")

    # Generate target if missing
    if target_col not in df.columns:
        log.warning(f"'{target_col}' not found. Generating synthetic target.")
        df = df.copy()

        # Synthetic future demand = weighted feature combination + noise
        rng = np.random.default_rng(42)
        n = len(df)

        pop_log = df.get("population_density_log", np.full(n, 9.0))
        growth  = df.get("growth_rate_annual", np.full(n, 0.02))
        vuln    = df.get("vulnerability_index", np.full(n, 0.5))

        raw = pop_log * (1 + growth * 5) * (1 + vuln * 0.3) + rng.normal(0, 0.5, n)
        scaler = StandardScaler()
        raw_norm = scaler.fit_transform(raw.values.reshape(-1, 1) if hasattr(raw, 'values') else raw.reshape(-1, 1)).flatten()
        df[target_col] = np.clip(raw_norm, 0, 1).round(4)

    X = df[available].fillna(df[available].median()).values
    y = df[target_col].fillna(0).values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    log.info(f"Target range: {y.min():.4f} – {y.max():.4f} | Mean: {y.mean():.4f}")
    return X_train, X_test, y_train, y_test, available


# ── Model Builders ────────────────────────────────────────────────────────────

def build_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """
    Build Ridge Regression pipeline.

    Ridge (L2 regularization):
        Objective: minimize ||Xw - y||² + α||w||²
        Effect: shrinks all coefficients toward zero (no feature selection)
        Choice: α tuned via RidgeCV on 5 values [0.01, 0.1, 1, 10, 100]

    Args:
        alpha: L2 regularization strength

    Returns:
        Pipeline: scaler → Ridge
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=alpha, fit_intercept=True, random_state=RANDOM_STATE)),
    ])


def build_svr_pipeline() -> Pipeline:
    """
    Build SVR (Support Vector Regression) pipeline.

    SVR with RBF kernel:
        Finds a tube of width ε around the function.
        Only support vectors (points outside tube) affect the model.
        Robust to outliers in GIS-derived population data.

    Returns:
        Pipeline: scaler → SVR
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(
            kernel="rbf",
            C=10,
            epsilon=0.05,
            gamma="scale",
            cache_size=500,
        )),
    ])


def build_polynomial_pipeline(degree: int = 2) -> Pipeline:
    """
    Build Polynomial Regression pipeline.

    Polynomial features (degree=2):
        Adds: x₁², x₂², x₁×x₂ for all feature pairs.
        Captures quadratic urban growth saturation effects.
        Note: degree>2 leads to combinatorial explosion + overfitting.

    Args:
        degree: Polynomial degree (typically 2)

    Returns:
        Pipeline: scaler → PolyFeatures → Ridge
    """
    return Pipeline([
        ("scaler",  StandardScaler()),
        ("poly",    PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)),
        ("ridge",   Ridge(alpha=1.0)),
    ])


# ── Alpha Tuning ──────────────────────────────────────────────────────────────

def tune_ridge_alpha(X_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Find optimal Ridge alpha using RidgeCV.

    RidgeCV is more efficient than GridSearchCV for alpha selection:
    Uses LOO (Leave-One-Out) or generalized CV for ridge regression.

    Args:
        X_train: Training features (already scaled)
        y_train: Training targets

    Returns:
        Best alpha value
    """
    alphas = np.logspace(-3, 3, 50)  # 50 values from 0.001 to 1000
    ridge_cv = RidgeCV(alphas=alphas, cv=CV_FOLDS, scoring="neg_mean_squared_error")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    ridge_cv.fit(X_scaled, y_train)

    log.info(f"  RidgeCV optimal alpha: {ridge_cv.alpha_:.4f}")
    return ridge_cv.alpha_


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_regression_model(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Dict:
    """
    Compute regression evaluation metrics.

    Args:
        model:      Fitted regression pipeline
        X_test:     Test features
        y_test:     Test targets
        model_name: Label for logging

    Returns:
        Dict of metric → value
    """
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # MAPE (safe: avoid division by zero)
    mask = y_test != 0
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100 if mask.any() else float("nan")

    log.info(f"\n{model_name} Test Results:")
    log.info(f"  MAE:  {mae:.4f}")
    log.info(f"  RMSE: {rmse:.4f}")
    log.info(f"  R²:   {r2:.4f}  (target: >0.7)")
    log.info(f"  MAPE: {mape:.2f}%")

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "model": model_name}


def cross_validate_regression(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
) -> Dict:
    """
    Run k-fold cross-validation for regression model.

    Args:
        model:      Pipeline (unfitted)
        X:          Full feature matrix
        y:          Full target vector
        model_name: Label

    Returns:
        Dict of CV metrics
    """
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
    mae_scores = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)

    log.info(f"\n{model_name} {CV_FOLDS}-fold CV:")
    log.info(f"  R²  = {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    log.info(f"  MAE = {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")

    return {
        "cv_r2_mean": float(r2_scores.mean()),
        "cv_r2_std":  float(r2_scores.std()),
        "cv_mae_mean": float(mae_scores.mean()),
        "cv_mae_std":  float(mae_scores.std()),
    }


def plot_model_comparison(results: List[Dict], X_test_list: List, y_test: np.ndarray) -> None:
    """
    Plot actual vs predicted values and residuals for all models.

    Args:
        results:     List of result dicts
        X_test_list: List of (pipeline, X_test) tuples
        y_test:      Test targets
    """
    n_models = len(X_test_list)
    fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 10))

    for i, ((name, pipeline, X_test), result) in enumerate(zip(X_test_list, results)):
        y_pred = pipeline.predict(X_test)

        # Actual vs Predicted scatter
        ax1 = axes[0][i] if n_models > 1 else axes[0]
        ax1.scatter(y_test, y_pred, alpha=0.5, s=20, color=["#3498DB","#E74C3C","#2ECC71"][i % 3])
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", linewidth=2, label="Perfect")
        ax1.set_xlabel("Actual Demand", fontsize=11)
        ax1.set_ylabel("Predicted Demand", fontsize=11)
        ax1.set_title(f"{name}\nR²={result['r2']:.3f}, RMSE={result['rmse']:.4f}", fontsize=11, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Residuals
        ax2 = axes[1][i] if n_models > 1 else axes[1]
        residuals = y_test - y_pred
        ax2.hist(residuals, bins=40, color=["#3498DB","#E74C3C","#2ECC71"][i % 3], alpha=0.8, edgecolor="white")
        ax2.axvline(0, color="black", linestyle="--", linewidth=2)
        ax2.set_xlabel("Residuals (Actual - Predicted)", fontsize=11)
        ax2.set_ylabel("Frequency", fontsize=11)
        ax2.set_title(f"{name} — Residual Distribution", fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.3)

    plt.suptitle("SmartCityAI — Regression Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = MODELS_DIR / "regression_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Regression comparison plot → {path}")


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train regression models for SmartCityAI")
    parser.add_argument("--model",  default="all", choices=["ridge", "svr", "poly", "all"])
    parser.add_argument("--alpha",  type=float, default=None, help="Ridge alpha (auto-tuned if not set)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SmartCityAI — Population Demand Regression")
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

    X_train, X_test, y_train, y_test, features = prepare_regression_data(df, REGRESSION_FEATURES)

    results_list = []
    models_to_compare = []

    # Ridge
    if args.model in ("ridge", "all"):
        alpha = args.alpha or tune_ridge_alpha(X_train, y_train)
        ridge_pipeline = build_ridge_pipeline(alpha)
        ridge_pipeline.fit(X_train, y_train)
        r = evaluate_regression_model(ridge_pipeline, X_test, y_test, "Ridge")
        cv_r = cross_validate_regression(build_ridge_pipeline(alpha), X_train, y_train, "Ridge")
        results_list.append({**r, **cv_r})
        models_to_compare.append(("Ridge", ridge_pipeline, X_test))

        with open(MODELS_DIR / "regression_ridge_model.pkl", "wb") as f:
            pickle.dump({"pipeline": ridge_pipeline, "features": features}, f)

    # SVR
    if args.model in ("svr", "all"):
        svr_pipeline = build_svr_pipeline()
        svr_pipeline.fit(X_train, y_train)
        r = evaluate_regression_model(svr_pipeline, X_test, y_test, "SVR")
        results_list.append(r)
        models_to_compare.append(("SVR", svr_pipeline, X_test))

        with open(MODELS_DIR / "regression_svr_model.pkl", "wb") as f:
            pickle.dump({"pipeline": svr_pipeline, "features": features}, f)

    # Polynomial
    if args.model in ("poly", "all"):
        poly_pipeline = build_polynomial_pipeline(degree=2)
        poly_pipeline.fit(X_train, y_train)
        r = evaluate_regression_model(poly_pipeline, X_test, y_test, "Polynomial (deg=2)")
        results_list.append(r)
        models_to_compare.append(("Polynomial", poly_pipeline, X_test))

    # Plot comparison
    if models_to_compare:
        plot_model_comparison(results_list, models_to_compare, y_test)

    # Save best model predictions
    best_result = max(results_list, key=lambda x: x.get("r2", 0))
    log.info(f"\nBest model: {best_result['model']} (R²={best_result['r2']:.4f})")

    # Predict on full dataset with Ridge (most stable)
    if models_to_compare:
        best_pipeline = models_to_compare[0][1]  # Ridge
        X_full = df[features].fillna(0).values
        df["demand_prediction"] = best_pipeline.predict(X_full).clip(0, 1)
        df[["h3_id", "demand_prediction"]].to_csv(PROCESSED_DIR / "regression_predictions.csv", index=False)
        log.info(f"Predictions saved → {PROCESSED_DIR}/regression_predictions.csv")

    log.info("Regression models training complete!")


if __name__ == "__main__":
    main()
