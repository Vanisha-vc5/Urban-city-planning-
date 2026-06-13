"""
composite_features.py
=====================
Phase 4 - Feature Engineering: Composite & Derived Features

PURPOSE:
    Combines demographic, coverage, and accessibility features into
    higher-order composite metrics that capture complex interactions.
    These are the most powerful features for ML models.

COMPOSITE FEATURES:
┌─────────────────────────────────┬───────────────────────────────────────────────────────┐
│ Feature                         │ Formula & Purpose                                     │
├─────────────────────────────────┼───────────────────────────────────────────────────────┤
│ infrastructure_need_score       │ Combines coverage gap + demand + vulnerability         │
│ site_suitability_score          │ Combines accessibility + coverage + land availability  │
│ equity_adjusted_priority        │ Amplifies need in low-income/high-vulnerability zones  │
│ temporal_demand_trend           │ Estimated annual population growth rate for zone       │
│ cluster_interaction_feature     │ Cross-feature: pop_density × coverage_gap             │
│ composite_risk_score            │ Fire + emergency response risk metric                  │
│ normalized_priority_composite   │ Final combined score [0, 100] for ranking              │
└─────────────────────────────────┴───────────────────────────────────────────────────────┘

WHY COMPOSITE FEATURES?
    - Pure individual features miss interactions (high density + no hospital = crisis)
    - Cross-features capture non-linear effects (ML trees can split on these)
    - Composite scores enable interpretable ranking for city planners
    - Equity adjustment ensures poor/vulnerable areas aren't overlooked
    - These features drove 15-20% accuracy improvement in urban AI literature
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


def compute_infrastructure_need_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite infrastructure need score.

    Formula:
        infrastructure_need_score =
            0.40 × coverage_gap
          + 0.25 × demand_pressure_norm
          + 0.20 × vulnerability_index
          + 0.15 × (1 - hospital_coverage_ratio)

    All components normalized [0,1]. Result is [0,1].
    This score says: "How urgently does THIS zone need a new facility?"

    Args:
        df: Feature DataFrame

    Returns:
        df with infrastructure_need_score [0,1]
    """
    df = df.copy()
    scaler = MinMaxScaler()

    def safe_col(name: str, default: float = 0.5) -> pd.Series:
        return df.get(name, pd.Series(np.full(len(df), default)))

    coverage_gap    = safe_col("coverage_gap")
    demand_pressure = safe_col("demand_pressure_norm")
    vulnerability   = safe_col("vulnerability_index")
    hosp_coverage   = safe_col("hospital_coverage_ratio")

    score = (
        0.40 * coverage_gap.clip(0, 1)
      + 0.25 * demand_pressure.clip(0, 1)
      + 0.20 * vulnerability.clip(0, 1)
      + 0.15 * (1.0 - hosp_coverage.clip(0, 1))
    )

    df["infrastructure_need_score"] = score.round(4)
    return df


def compute_site_suitability_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute site suitability score — how BUILDABLE is this zone?

    A high-need zone may not be suitable for construction if it is:
    - Poorly accessible (no roads nearby)
    - Already densely built up

    Formula:
        site_suitability_score =
            0.50 × road_accessibility_index
          + 0.30 × (1 - population_density_norm)   # less dense = more land available
          + 0.20 × (1 - nearest_arterial_dist_norm) # close to arterial road

    Args:
        df: Feature DataFrame

    Returns:
        df with site_suitability_score [0,1]
    """
    df = df.copy()
    scaler = MinMaxScaler()

    road_access = df.get("road_accessibility_index", pd.Series(np.full(len(df), 0.5)))
    pop_density = df.get("population_density", pd.Series(np.ones(len(df)) * 10000))
    arterial_d  = df.get("nearest_arterial_dist_km", pd.Series(np.full(len(df), 1.0)))

    pop_norm = scaler.fit_transform(pop_density.values.reshape(-1, 1)).flatten()
    art_norm = scaler.fit_transform(arterial_d.values.reshape(-1, 1)).flatten()

    score = (
        0.50 * road_access.clip(0, 1)
      + 0.30 * (1.0 - pop_norm)
      + 0.20 * (1.0 - art_norm)
    )

    df["site_suitability_score"] = score.round(4)
    return df


def compute_equity_adjusted_priority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply equity adjustment to infrastructure need.

    Standard need scoring prioritizes high-density areas (more people = more need).
    Equity adjustment ensures low-income + vulnerable areas get amplified priority,
    even if their absolute population is smaller.

    Formula:
        equity_factor = 1 + poverty_proxy × elderly_ratio × 2
        equity_adjusted_priority = infrastructure_need_score × equity_factor

    Then normalized to [0, 1].

    Policy rationale:
        A slum with moderate population density but NO hospital for 6 km and
        high elderly proportion should rank ABOVE a rich neighborhood with the
        same population but has hospitals nearby.

    Args:
        df: Feature DataFrame (needs infrastructure_need_score)

    Returns:
        df with equity_adjusted_priority [0,1]
    """
    df = df.copy()
    scaler = MinMaxScaler()

    need     = df.get("infrastructure_need_score", pd.Series(np.full(len(df), 0.5)))
    poverty  = df.get("poverty_proxy", pd.Series(np.full(len(df), 0.3)))
    elderly  = df.get("elderly_ratio", pd.Series(np.full(len(df), 0.08)))

    equity_factor = 1.0 + (poverty.clip(0,1) * elderly.clip(0,1) * 2.0)
    raw = need * equity_factor

    normalized = scaler.fit_transform(raw.values.reshape(-1, 1)).flatten()
    df["equity_adjusted_priority"] = normalized.round(4)
    return df


def compute_cluster_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-product interaction features for tree-based models.

    Decision Trees and Random Forests can identify thresholds but cannot
    inherently capture multiplicative interactions. Explicit cross-features
    expose these relationships.

    Features:
        density_gap_interaction:   population_density_norm × coverage_gap
            → "Dense area with no hospital" — pure urgency signal
        elderly_distance_product:  elderly_ratio × dist_nearest_hospital
            → "Elderly people far from hospital" — mortality risk signal
        income_coverage_gap:       poverty_proxy × coverage_gap
            → "Poor area with no coverage" — equity signal
        road_hospital_interaction: road_accessibility_index × dist_nearest_hospital
            → "Accessible zone with distant hospital" — ideal new site signal

    Args:
        df: Feature DataFrame

    Returns:
        df with interaction features
    """
    df = df.copy()
    scaler = MinMaxScaler()

    def norm(col: str, default: float = 0.5) -> np.ndarray:
        vals = df.get(col, pd.Series(np.full(len(df), default))).fillna(default)
        return scaler.fit_transform(vals.values.reshape(-1, 1)).flatten()

    pop_n      = norm("population_density")
    gap_n      = norm("coverage_gap")
    elderly_n  = norm("elderly_ratio")
    dist_h_n   = norm("dist_nearest_hospital")
    poverty_n  = norm("poverty_proxy")
    road_n     = norm("road_accessibility_index")

    df["density_gap_interaction"]    = (pop_n * gap_n).round(4)
    df["elderly_distance_product"]   = (elderly_n * dist_h_n).round(4)
    df["income_coverage_gap"]        = (poverty_n * gap_n).round(4)
    df["road_hospital_interaction"]  = (road_n * dist_h_n).round(4)

    return df


def compute_temporal_demand_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate annual population demand growth for regression modeling.

    Uses a simplified urban growth model:
        - Core areas: slow growth (saturated)
        - Peri-urban areas: fast growth (suburbanization)
        - Growth correlates with road accessibility (development follows roads)

    Formula:
        growth_rate = 0.02 + 0.03 × road_access × (1 - pop_density_norm) + noise
        future_demand = population_density × (1 + growth_rate)^5

    The 5-year projection matches typical urban infrastructure planning cycles.

    Args:
        df: Feature DataFrame

    Returns:
        df with growth_rate_annual and future_demand_5yr features
    """
    df = df.copy()
    scaler = MinMaxScaler()
    rng = np.random.default_rng(2024)

    pop_density = df.get("population_density", pd.Series(np.ones(len(df)) * 10000))
    road_access = df.get("road_accessibility_index", pd.Series(np.full(len(df), 0.5)))

    pop_norm = scaler.fit_transform(pop_density.values.reshape(-1, 1)).flatten()

    # Peri-urban areas (low density + good roads) grow fastest
    base_growth = 0.02  # 2% annual urban growth (India average)
    location_bonus = 0.03 * road_access.clip(0, 1) * (1.0 - pop_norm)
    noise = rng.normal(0, 0.005, len(df))
    growth_rate = (base_growth + location_bonus + noise).clip(0, 0.10)

    df["growth_rate_annual"]   = growth_rate.round(4)
    df["future_demand_5yr"]    = (pop_density * (1 + growth_rate) ** 5).round(0)
    df["future_demand_5yr_norm"] = scaler.fit_transform(
        df["future_demand_5yr"].values.reshape(-1, 1)
    ).flatten().round(4)

    return df


def compute_composite_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute emergency/disaster response risk score.

    High risk = many people, poor emergency access, far from fire stations.

    Formula:
        composite_risk = (
            0.40 × emergency_response_time_norm
          + 0.35 × vulnerability_index
          + 0.25 × traffic_density_norm   # high traffic → ambulance delays
        )

    Used for: Fire station and emergency response center placement priority.

    Args:
        df: Feature DataFrame

    Returns:
        df with composite_risk_score [0,1]
    """
    df = df.copy()
    scaler = MinMaxScaler()

    def norm_col(col: str, default: float = 0.5) -> np.ndarray:
        vals = df.get(col, pd.Series(np.full(len(df), default))).fillna(default)
        return scaler.fit_transform(vals.values.reshape(-1, 1)).flatten()

    resp_time  = norm_col("emergency_response_time_min", 8.0)
    vuln       = norm_col("vulnerability_index")
    traffic    = norm_col("traffic_density")

    df["composite_risk_score"] = (
        0.40 * resp_time
      + 0.35 * vuln
      + 0.25 * traffic
    ).round(4)

    return df


def compute_normalized_priority_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute final normalized priority composite score [0, 100].

    This is the top-level decision metric displayed on the dashboard.

    Formula:
        raw_priority = (
            0.35 × equity_adjusted_priority
          + 0.25 × infrastructure_need_score
          + 0.20 × composite_risk_score
          + 0.20 × (1 - site_suitability_score)   # inverse: hard to build = lower priority
        )
        priority_composite_100 = raw_priority × 100

    Args:
        df: Feature DataFrame (needs all component scores)

    Returns:
        df with priority_composite_100 and priority_rank columns
    """
    df = df.copy()

    def safe(col: str, default: float = 0.5) -> pd.Series:
        return df.get(col, pd.Series(np.full(len(df), default))).clip(0, 1)

    raw = (
        0.35 * safe("equity_adjusted_priority")
      + 0.25 * safe("infrastructure_need_score")
      + 0.20 * safe("composite_risk_score")
      + 0.20 * (1.0 - safe("site_suitability_score"))
    )

    df["priority_composite_100"] = (raw * 100).round(1)
    df["priority_rank"] = df["priority_composite_100"].rank(ascending=False, method="first").astype(int)

    return df


def compute_all_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: run all composite feature computations in order.

    Order matters — later functions depend on outputs of earlier ones.

    Args:
        df: Feature DataFrame (should already have demographic, coverage, accessibility features)

    Returns:
        df with all composite features
    """
    log.info("Computing composite features...")
    df = compute_infrastructure_need_score(df)
    df = compute_site_suitability_score(df)
    df = compute_equity_adjusted_priority(df)
    df = compute_cluster_interaction_features(df)
    df = compute_temporal_demand_trend(df)
    df = compute_composite_risk_score(df)
    df = compute_normalized_priority_composite(df)

    composite_cols = [
        "infrastructure_need_score", "site_suitability_score",
        "equity_adjusted_priority", "density_gap_interaction",
        "elderly_distance_product", "composite_risk_score",
        "priority_composite_100", "priority_rank",
    ]
    log.info(f"Composite features: {[c for c in composite_cols if c in df.columns]}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    csv = PROCESSED_DIR / "zone_features.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        df = compute_all_composite_features(df)
        print(df[["infrastructure_need_score", "equity_adjusted_priority",
                   "priority_composite_100", "priority_rank"]].describe())
    else:
        print("Run create_zones.py first.")
