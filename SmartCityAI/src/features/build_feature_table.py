"""
build_feature_table.py
======================
Phase 4 - Feature Engineering: Master Feature Table Builder

PURPOSE:
    Orchestrates the complete feature engineering pipeline.
    Runs all feature modules in order and produces the final
    enriched feature table used by all ML models.

PIPELINE:
    zone_features.csv (raw from create_zones.py)
        → demographic_features.py    (+10 features)
        → coverage_features.py       (+12 features)
        → accessibility_features.py  (+7 features)
        → composite_features.py      (+8 features)
        → [output] enriched_features.csv    (all ~40 features)
        → [output] ml_features.csv          (clean ML-ready subset)
        → [output] feature_summary.txt      (documentation)

ML-READY FEATURES (ml_features.csv):
    Core ML features used by K-Means, SVM, Decision Tree, Regression:
    1.  population_density_log
    2.  elderly_ratio
    3.  youth_ratio
    4.  income_bracket_norm
    5.  vulnerability_index
    6.  dist_nearest_hospital
    7.  hospital_coverage_ratio
    8.  coverage_gap
    9.  road_accessibility_index
    10. traffic_density
    11. emergency_response_time_min
    12. infrastructure_need_score
    13. equity_adjusted_priority
    14. density_gap_interaction
    15. composite_risk_score
    16. future_demand_5yr_norm  (regression target)
    17. priority_composite_100  (main priority label)

USAGE:
    python build_feature_table.py
    python build_feature_table.py --city "Mumbai, India" --output-dir data/processed

OUTPUT:
    data/processed/enriched_features.csv  (all features, for analysis)
    data/processed/ml_features.csv        (ML-ready subset, no geometry)
    data/processed/feature_summary.txt    (column documentation)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from features.demographic_features  import compute_all_demographic_features
from features.coverage_features     import compute_all_coverage_features
from features.accessibility_features import compute_all_accessibility_features
from features.composite_features    import compute_all_composite_features

# ── Configuration ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_feature_table")

BASE_DIR      = Path(__file__).resolve().parents[3]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Core ML features (no leakage, no IDs, model-ready)
ML_FEATURE_COLUMNS = [
    # Demographics
    "population_density_log",
    "elderly_ratio",
    "youth_ratio",
    "income_bracket_norm",
    "vulnerability_index",
    "demand_pressure_norm",
    # Coverage
    "dist_nearest_hospital",
    "hospital_coverage_ratio",
    "coverage_gap",
    "dist_nearest_school",
    "dist_nearest_ev_station",
    "dist_nearest_fire_station",
    "multi_coverage_gap",
    # Accessibility
    "road_accessibility_index",
    "traffic_density",
    "walkability_index",
    "emergency_response_time_min",
    "nearest_arterial_dist_km",
    # Composite (engineered)
    "infrastructure_need_score",
    "site_suitability_score",
    "equity_adjusted_priority",
    "density_gap_interaction",
    "elderly_distance_product",
    "composite_risk_score",
    "growth_rate_annual",
    "future_demand_5yr_norm",
]

# Target columns for different models
TARGET_COLUMNS = {
    "clustering":   None,          # K-Means: unsupervised (no target)
    "classification": "priority_class",   # SVM + DT: High/Medium/Low
    "regression":   "future_demand_5yr_norm",  # Ridge Regression
    "ranking":      "priority_composite_100",  # Best First Search
}

# Feature documentation
FEATURE_DOCS = {
    "population_density_log":      ("demographic", "log(1 + pop/km²)", "K-Means, SVM, DT, Regression"),
    "elderly_ratio":               ("demographic", "pop_60+/pop_total", "Hospital priority weighting"),
    "youth_ratio":                 ("demographic", "pop_0-14/pop_total", "School demand proxy"),
    "income_bracket_norm":         ("demographic", "MinMax(median_income)", "Equity weighting"),
    "vulnerability_index":         ("demographic", "0.5×elderly + 0.3×poverty + 0.2×youth", "All models (strong predictor)"),
    "dist_nearest_hospital":       ("coverage",    "Haversine distance to nearest hospital (km)", "All models (strongest signal)"),
    "hospital_coverage_ratio":     ("coverage",    "Fraction of zone within 5km of hospital", "SVM classification input"),
    "coverage_gap":                ("coverage",    "1 - hospital_coverage_ratio", "Best First Search heuristic"),
    "road_accessibility_index":    ("accessibility","Σ(weight×length)/area_km²", "Site suitability"),
    "traffic_density":             ("accessibility","road_count/area_km²", "EV station demand proxy"),
    "emergency_response_time_min": ("accessibility","distance/speed × 60 + 1.5 dispatch", "Fire station placement"),
    "infrastructure_need_score":   ("composite",   "Weighted coverage+demand+vulnerability", "Primary ranking score"),
    "equity_adjusted_priority":    ("composite",   "need × equity_factor(poverty, elderly)", "Equity-aware ranking"),
    "density_gap_interaction":     ("composite",   "pop_density_norm × coverage_gap", "Non-linear interaction for DT"),
    "future_demand_5yr_norm":      ("composite",   "pop × (1+growth_rate)^5", "Regression target"),
    "priority_composite_100":      ("composite",   "Weighted final score [0-100]", "Dashboard display & ranking"),
}


def load_base_features(csv_path: Path) -> pd.DataFrame:
    """
    Load base zone features from create_zones.py output.

    Args:
        csv_path: Path to zone_features.csv

    Returns:
        DataFrame or raises FileNotFoundError
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"zone_features.csv not found at {csv_path}.\n"
            "Run `python src/preprocessing/create_zones.py` first."
        )

    df = pd.read_csv(csv_path)
    log.info(f"Loaded base features: {len(df):,} zones, {len(df.columns)} columns")
    return df


def validate_features(df: pd.DataFrame) -> None:
    """
    Validate feature table for ML readiness.

    Checks:
    - No infinite values
    - Missing value rate < 10% per column
    - All ML features present
    - No negative values in distance columns

    Args:
        df: Enriched feature DataFrame

    Raises:
        ValueError if validation fails
    """
    log.info("Validating feature table...")
    errors = []

    # Check for infinities
    inf_mask = np.isinf(df.select_dtypes(include=np.number))
    if inf_mask.any().any():
        inf_cols = inf_mask.columns[inf_mask.any()].tolist()
        errors.append(f"Infinite values found in: {inf_cols}")

    # Check missing rates
    missing_rate = df.isnull().mean()
    high_missing = missing_rate[missing_rate > 0.10].index.tolist()
    if high_missing:
        errors.append(f"High missing rate (>10%) in: {high_missing}")

    # Check distance columns are non-negative
    dist_cols = [c for c in df.columns if "dist_" in c]
    for col in dist_cols:
        if col in df.columns and (df[col] < 0).any():
            errors.append(f"Negative distances found in: {col}")

    if errors:
        log.warning("Feature validation warnings:\n  " + "\n  ".join(errors))
    else:
        log.info("  Feature validation passed.")


def write_feature_summary(df: pd.DataFrame, output_path: Path) -> None:
    """
    Write a human-readable feature documentation file.

    Args:
        df:          Feature DataFrame
        output_path: Output .txt path
    """
    lines = [
        "=" * 70,
        "SMARTCITYAI — FEATURE SUMMARY",
        "=" * 70,
        f"Total zones:    {len(df):,}",
        f"Total features: {len(df.columns)}",
        "",
        "FEATURE DOCUMENTATION:",
        "-" * 70,
    ]

    numeric_df = df.select_dtypes(include=np.number)

    for col in sorted(df.columns):
        if col in ("h3_id",):
            continue
        category, formula, usage = FEATURE_DOCS.get(col, ("other", "—", "—"))
        missing = df[col].isnull().sum()
        if col in numeric_df.columns:
            min_v = numeric_df[col].min()
            max_v = numeric_df[col].max()
            mean_v = numeric_df[col].mean()
            stats = f"min={min_v:.4f}, mean={mean_v:.4f}, max={max_v:.4f}"
        else:
            stats = f"values={df[col].unique()[:5]}"

        lines += [
            f"\n{col}",
            f"  Category:  {category}",
            f"  Formula:   {formula}",
            f"  Usage:     {usage}",
            f"  Stats:     {stats}",
            f"  Missing:   {missing} ({missing/len(df)*100:.1f}%)",
        ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    log.info(f"Feature summary written → {output_path}")


def build_ml_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the ML-ready feature matrix.

    - Selects only ML_FEATURE_COLUMNS (plus h3_id, priority columns)
    - Fills remaining NaN with column median
    - Keeps h3_id for spatial join back to zones GeoPackage

    Args:
        df: Enriched features DataFrame

    Returns:
        ML-ready DataFrame
    """
    keep_cols = ["h3_id"] + [c for c in ML_FEATURE_COLUMNS if c in df.columns]

    # Add target columns
    for target in TARGET_COLUMNS.values():
        if target and target in df.columns and target not in keep_cols:
            keep_cols.append(target)

    ml_df = df[keep_cols].copy()

    # Fill NaN with median
    numeric_cols = ml_df.select_dtypes(include=np.number).columns
    ml_df[numeric_cols] = ml_df[numeric_cols].fillna(ml_df[numeric_cols].median())

    # Add standardized versions for SVM/distance-based models
    features_to_scale = [c for c in ML_FEATURE_COLUMNS if c in ml_df.columns]
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(ml_df[features_to_scale])
    for i, col in enumerate(features_to_scale):
        ml_df[f"{col}_scaled"] = scaled_values[:, i].round(4)

    log.info(f"ML feature matrix: {len(ml_df):,} rows × {len(ml_df.columns)} columns")
    return ml_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SmartCityAI feature table")
    parser.add_argument("--input",       default=str(PROCESSED_DIR / "zone_features.csv"))
    parser.add_argument("--output-dir",  default=str(PROCESSED_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("SmartCityAI — Feature Engineering Pipeline")
    log.info("=" * 60)

    # Load base
    df = load_base_features(Path(args.input))

    # Run all feature engineering modules
    log.info("\n[1/4] Demographic features...")
    df = compute_all_demographic_features(df)

    log.info("\n[2/4] Coverage features...")
    df = compute_all_coverage_features(df)

    log.info("\n[3/4] Accessibility features...")
    df = compute_all_accessibility_features(df)

    log.info("\n[4/4] Composite features...")
    df = compute_all_composite_features(df)

    # Validate
    validate_features(df)

    # Save enriched features (all columns)
    enriched_path = output_dir / "enriched_features.csv"
    df.to_csv(enriched_path, index=False)
    log.info(f"\nEnriched features saved → {enriched_path} ({len(df):,} rows, {len(df.columns)} cols)")

    # Save ML-ready features
    ml_df = build_ml_feature_matrix(df)
    ml_path = output_dir / "ml_features.csv"
    ml_df.to_csv(ml_path, index=False)
    log.info(f"ML features saved → {ml_path} ({len(ml_df.columns)} cols)")

    # Save feature summary
    summary_path = output_dir / "feature_summary.txt"
    write_feature_summary(df, summary_path)

    log.info("\n" + "=" * 60)
    log.info("Feature Engineering Complete!")
    log.info(f"  Enriched features: {enriched_path}")
    log.info(f"  ML features:       {ml_path}")
    log.info(f"  Feature summary:   {summary_path}")

    # Print top features by variance
    numeric_df = df.select_dtypes(include=np.number)
    variances = numeric_df.var().sort_values(ascending=False)
    log.info(f"\nTop 10 features by variance:\n{variances.head(10).to_string()}")


if __name__ == "__main__":
    main()
