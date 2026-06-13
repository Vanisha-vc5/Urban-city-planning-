"""
demographic_features.py
=======================
Phase 4 - Feature Engineering: Demographic Features

PURPOSE:
    Computes demographic-based features for each H3 zone from census data.
    These features capture population pressure and vulnerability,
    which are the primary demand drivers for hospitals and schools.

FEATURES GENERATED:
┌────────────────────────────┬────────────────────────────────┬─────────────────────────────────────────┐
│ Feature                    │ Formula                        │ Model Usage                             │
├────────────────────────────┼────────────────────────────────┼─────────────────────────────────────────┤
│ population_density         │ pop_total / area_km²           │ K-Means cluster, SVM, DT, Regression    │
│ population_density_norm    │ MinMax scaled                  │ SVM input (required for RBF kernel)     │
│ elderly_ratio              │ pop_60+ / pop_total            │ Hospital priority (elderly use more)    │
│ youth_ratio                │ pop_0_14 / pop_total           │ School demand proxy                     │
│ working_ratio              │ pop_15_59 / pop_total          │ EV charging demand (commuters)          │
│ income_bracket_norm        │ MinMax(median_income)          │ Equity weighting in priority score      │
│ vulnerability_index        │ 0.5×elderly + 0.3×poverty_norm │ Composite vulnerability metric          │
│ demand_pressure            │ log(pop_density × pop_total)  │ Regression target variable proxy        │
└────────────────────────────┴────────────────────────────────┴─────────────────────────────────────────┘

WHY THESE FEATURES?
    - Elderly ratio is the single strongest predictor of hospital demand
      (elderly visit hospitals 3-4× more than working-age adults)
    - Youth ratio predicts school demand
    - Income normalization enables equity-based planning (not just demand)
    - Vulnerability index captures compound disadvantage
    - Log-transform on demand_pressure handles extreme Mumbai-style density

DESIGN DECISIONS:
    - All features bounded [0, 1] after normalization for model compatibility
    - Features computed from census CSV + optional WorldPop raster
    - Fallback synthetic generation for offline/demo mode
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

log = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).resolve().parents[3]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR       = BASE_DIR / "data" / "raw"


def load_census_data(census_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    Load census CSV data.

    Args:
        census_path: Path to census CSV file. Auto-discovers if None.

    Returns:
        DataFrame or None if not found.
    """
    if census_path and census_path.exists():
        return pd.read_csv(census_path)

    # Auto-discover
    candidates = list(RAW_DIR.glob("census_*.csv"))
    if candidates:
        log.info(f"Auto-discovered census file: {candidates[0].name}")
        return pd.read_csv(candidates[0])

    log.warning("No census file found. Using synthetic demographics.")
    return None


def compute_population_density(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute population density and its log-transformed variant.

    population_density:
        Source: zone_features.csv (from create_zones.py)
        Formula: population_total / area_km²
        Units: persons per km²
        Purpose: Primary demand indicator for all infrastructure types

    population_density_log:
        Formula: log(1 + population_density)
        Purpose: Handles extreme skew in dense urban cores (Mumbai: 100k/km²)
        Model Usage: Regression input (normalizes distribution)

    Args:
        df: Zone features DataFrame (must have population_density)

    Returns:
        df with population_density_log added
    """
    df = df.copy()
    if "population_density" not in df.columns:
        log.error("population_density column missing!")
        df["population_density"] = 0
    df["population_density_log"] = np.log1p(df["population_density"]).round(4)
    return df


def compute_age_ratios(df: pd.DataFrame, census_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute age structure ratios per zone.

    elderly_ratio:
        Source: Census age bands (pop_60+ / pop_total)
        Formula: (pop_age_60_69 + pop_age_70_79 + pop_age_80plus) / pop_total
        Range: [0, 1]
        Purpose: Weights hospital priority (WHO: 60+ use 3.4× more hospital beds)

    youth_ratio:
        Source: Census age bands (pop_0_14 / pop_total)
        Formula: (pop_age_0_4 + pop_age_5_9 + pop_age_10_14) / pop_total
        Range: [0, 1]
        Purpose: School demand driver

    working_ratio:
        Source: Derived (1 - elderly - youth)
        Formula: pop_15_59 / pop_total
        Range: [0, 1]
        Purpose: EV charging demand, fire station response capacity

    Args:
        df:         Zone features DataFrame
        census_df:  Census data with age breakdown

    Returns:
        df with age ratio columns
    """
    df = df.copy()

    if census_df is not None and "elderly_ratio" in census_df.columns:
        # Map from census to zones (simplified: use city-wide average per zone)
        df["elderly_ratio"] = census_df["elderly_ratio"].mean()
        df["youth_ratio"]   = census_df.get("youth_ratio", pd.Series([0.22])).mean()
    elif "elderly_ratio" not in df.columns:
        # Synthetic: Mumbai-calibrated defaults with variance
        rng = np.random.default_rng(42)
        n = len(df)
        df["elderly_ratio"] = rng.beta(2.5, 28, n).round(4)
        df["youth_ratio"]   = rng.beta(5.0, 18, n).round(4)

    # Ensure bounds
    df["elderly_ratio"] = df["elderly_ratio"].clip(0, 1)
    df["youth_ratio"]   = df["youth_ratio"].clip(0, 1)
    df["working_ratio"] = (1.0 - df["elderly_ratio"] - df["youth_ratio"]).clip(0, 1).round(4)

    return df


def compute_income_features(df: pd.DataFrame, census_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute income normalization and poverty proxy features.

    income_bracket_norm:
        Source: Census median household income
        Formula: MinMaxScaler(median_income) → [0, 1]
        Purpose: Equity weighting — low income areas prioritized for public infrastructure
        Note: In priority score formula, this is INVERTED (low income = high priority)

    poverty_proxy:
        Formula: 1 - income_bracket_norm
        Purpose: Explicit poverty signal for SVM classification

    Args:
        df:        Zone features DataFrame
        census_df: Census data

    Returns:
        df with income features
    """
    df = df.copy()

    if census_df is not None and "income_bracket_norm" in census_df.columns:
        df["income_bracket_norm"] = census_df["income_bracket_norm"].mean()
    elif "income_bracket_norm" not in df.columns:
        rng = np.random.default_rng(99)
        df["income_bracket_norm"] = rng.beta(3, 5, len(df)).round(4)

    df["income_bracket_norm"] = df["income_bracket_norm"].clip(0, 1)
    df["poverty_proxy"]       = (1.0 - df["income_bracket_norm"]).round(4)

    return df


def compute_vulnerability_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite vulnerability index.

    vulnerability_index:
        Formula: 0.5 × elderly_ratio_norm + 0.3 × poverty_proxy + 0.2 × youth_ratio_norm
        Range: [0, 1]
        Purpose: Single composite score representing social vulnerability
        Note: High vulnerability → infrastructure should be prioritized regardless of
              current coverage (equity lens, not just efficiency)

        Used by: Decision Tree (top split feature), SVM, Explanation Engine

    Args:
        df: DataFrame with age and income features

    Returns:
        df with vulnerability_index column
    """
    df = df.copy()
    scaler = MinMaxScaler()

    # Normalize components
    for col in ["elderly_ratio", "poverty_proxy", "youth_ratio"]:
        if col not in df.columns:
            df[col] = 0.0

    elderly_n = scaler.fit_transform(df[["elderly_ratio"]]).flatten()
    poverty_n = scaler.fit_transform(df[["poverty_proxy"]]).flatten()
    youth_n   = scaler.fit_transform(df[["youth_ratio"]]).flatten()

    df["vulnerability_index"] = (
        0.5 * elderly_n + 0.3 * poverty_n + 0.2 * youth_n
    ).round(4)

    return df


def compute_demand_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute demand pressure: a compound metric combining density and vulnerability.

    demand_pressure:
        Formula: log(1 + pop_density) × vulnerability_index
        Purpose: Combines raw demand (density) with equity need (vulnerability)
        Range: Unbounded positive (log-scaled)
        Normalization: MinMax applied after computation
        Model Usage: Regression target variable proxy, K-Means clustering

    Args:
        df: DataFrame with population_density_log and vulnerability_index

    Returns:
        df with demand_pressure and demand_pressure_norm
    """
    df = df.copy()

    pop_log = df.get("population_density_log", np.log1p(df.get("population_density", 0)))
    vuln    = df.get("vulnerability_index", 0.5)

    dp = pop_log * vuln
    scaler = MinMaxScaler()
    df["demand_pressure"]      = dp.round(4)
    df["demand_pressure_norm"] = scaler.fit_transform(dp.values.reshape(-1, 1)).flatten().round(4)

    return df


def compute_all_demographic_features(
    df: pd.DataFrame,
    census_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Master function: compute all demographic features.

    Pipeline:
    1. Load census data
    2. Population density + log transform
    3. Age ratios (elderly, youth, working)
    4. Income normalization + poverty proxy
    5. Vulnerability index
    6. Demand pressure

    Args:
        df:          Input DataFrame (from zone_features.csv)
        census_path: Optional path to census CSV

    Returns:
        DataFrame with all demographic features
    """
    log.info("Computing demographic features...")
    census_df = load_census_data(census_path)

    df = compute_population_density(df)
    df = compute_age_ratios(df, census_df)
    df = compute_income_features(df, census_df)
    df = compute_vulnerability_index(df)
    df = compute_demand_pressure(df)

    demographic_cols = [
        "population_density", "population_density_log",
        "elderly_ratio", "youth_ratio", "working_ratio",
        "income_bracket_norm", "poverty_proxy",
        "vulnerability_index", "demand_pressure", "demand_pressure_norm",
    ]
    log.info(f"Demographic features computed: {[c for c in demographic_cols if c in df.columns]}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick test
    csv = PROCESSED_DIR / "zone_features.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        df = compute_all_demographic_features(df)
        print(df[[c for c in df.columns if any(k in c for k in ["density", "ratio", "index", "pressure"])]].describe())
    else:
        print(f"Run create_zones.py first to generate {csv}")
