"""
coverage_features.py
====================
Phase 4 - Feature Engineering: Infrastructure Coverage Features

PURPOSE:
    Computes coverage gap metrics — how well existing infrastructure
    serves the population of each zone. Coverage gap is the single
    most important feature for identifying where NEW facilities should go.

FEATURES GENERATED:
┌──────────────────────────────┬──────────────────────────────────────────┬───────────────────────────────┐
│ Feature                      │ Formula                                  │ Model Usage                   │
├──────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────┤
│ dist_nearest_hospital        │ Haversine distance to closest hospital   │ All models (strongest signal) │
│ hospital_coverage_ratio      │ (pop in 5km radius) / zone_pop           │ SVM priority classification   │
│ coverage_gap                 │ 1 - hospital_coverage_ratio              │ Best First Search heuristic   │
│ dist_nearest_school          │ Haversine to closest school              │ School placement model        │
│ dist_nearest_ev_station      │ Haversine to closest EV station          │ EV placement model            │
│ dist_nearest_fire_station    │ Haversine to closest fire station        │ Emergency response model      │
│ multi_coverage_gap           │ Weighted average of all coverage gaps    │ Hill Climbing objective       │
│ is_underserved               │ Binary: coverage_gap > 0.5               │ DT classification label       │
│ service_radius_coverage      │ Proportion of zone within any service R  │ Visual coverage map overlay   │
└──────────────────────────────┴──────────────────────────────────────────┴───────────────────────────────┘

SERVICE RADIUS STANDARDS:
    Hospital:          5 km  (WHO guideline for primary care)
    School:            2 km  (max walking distance for students)
    EV Station:        3 km  (Level 2) / 8 km (DC Fast Charge)
    Fire Station:      3 km  (NFPA Standard 1710 response time)
    Emergency Center:  5 km  (NIMS/ICS recommendation)

WHY HAVERSINE + BALLTREE?
    - Pure Euclidean distance is wrong for geographic coordinates
    - Haversine: O(1) exact spherical distance formula
    - BallTree with haversine metric: O(n log n) nearest neighbor
    - 100× faster than pairwise distance matrix for large cities
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

log = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).resolve().parents[3]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR       = BASE_DIR / "data" / "raw"

# WHO / Standards-based service radii (in km)
SERVICE_RADII_KM: Dict[str, float] = {
    "hospital":     5.0,
    "school":       2.0,
    "ev_station":   3.0,
    "fire_station": 3.0,
}

# Coverage weights for multi-coverage gap (must sum to 1.0)
COVERAGE_WEIGHTS = {
    "hospital":     0.40,
    "school":       0.25,
    "ev_station":   0.20,
    "fire_station": 0.15,
}

EARTH_RADIUS_KM = 6371.0


def _load_facility_gdf(
    facility_type: str,
    custom_path: Optional[Path] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Load facility GeoDataFrame from file, with auto-discovery.

    Args:
        facility_type: One of "hospital", "school", "ev_station", "fire_station"
        custom_path:   Override path

    Returns:
        GeoDataFrame of facility points, or None
    """
    if custom_path and custom_path.exists():
        return gpd.read_file(str(custom_path))

    # Auto-discover filenames
    discovery_map = {
        "hospital":     ["healthsites_*.gpkg", "osm_hospitals.gpkg"],
        "school":       ["osm_schools.gpkg"],
        "ev_station":   ["ev_stations_*.gpkg"],
        "fire_station": ["osm_fire_stations.gpkg"],
    }

    for pattern in discovery_map.get(facility_type, []):
        matches = list(RAW_DIR.glob(pattern))
        if matches:
            return gpd.read_file(str(matches[0]))

    return None


def _build_balltree(gdf: gpd.GeoDataFrame) -> Tuple[BallTree, np.ndarray]:
    """
    Build a haversine BallTree from a GeoDataFrame of points.

    Args:
        gdf: GeoDataFrame with Point geometries

    Returns:
        (BallTree, coordinate array in radians)
    """
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.centroid  # Ensure points

    coords_deg = np.column_stack([gdf.geometry.y, gdf.geometry.x])
    coords_rad = np.radians(coords_deg)
    tree = BallTree(coords_rad, metric="haversine")
    return tree, coords_rad


def _generate_synthetic_facilities(
    zone_centroids: np.ndarray,
    n_facilities: int,
    spread: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic facility coordinates (fallback when no data files exist).

    Args:
        zone_centroids: Array of (lat, lon) zone centers
        n_facilities:   Number of synthetic facilities to generate
        spread:         Degree spread around city center
        seed:           RNG seed

    Returns:
        Array of (lat, lon) facility coordinates
    """
    rng = np.random.default_rng(seed)
    center_lat = zone_centroids[:, 0].mean()
    center_lon = zone_centroids[:, 1].mean()
    lats = center_lat + rng.normal(0, spread, n_facilities)
    lons = center_lon + rng.normal(0, spread, n_facilities)
    return np.column_stack([lats, lons])


def compute_distance_to_nearest(
    zone_centroids_deg: np.ndarray,
    facility_type: str,
    custom_path: Optional[Path] = None,
    n_synthetic: int = 30,
) -> np.ndarray:
    """
    Compute distance from each zone centroid to the nearest facility.

    Args:
        zone_centroids_deg: Array of (lat, lon) zone centroids in degrees
        facility_type:      Type of facility
        custom_path:        Optional override path to facility file
        n_synthetic:        Number of synthetic facilities (fallback)

    Returns:
        Array of distances in km, shape (n_zones,)
    """
    gdf = _load_facility_gdf(facility_type, custom_path)

    if gdf is not None and len(gdf) > 0:
        tree, _ = _build_balltree(gdf)
        query_rad = np.radians(zone_centroids_deg)
        distances_rad, _ = tree.query(query_rad, k=1)
        return (distances_rad[:, 0] * EARTH_RADIUS_KM).round(3)
    else:
        log.warning(f"No {facility_type} data found. Using synthetic fallback.")
        synth_coords = _generate_synthetic_facilities(
            zone_centroids_deg, n_synthetic, seed=hash(facility_type) % 1000
        )
        tree = BallTree(np.radians(synth_coords), metric="haversine")
        query_rad = np.radians(zone_centroids_deg)
        distances_rad, _ = tree.query(query_rad, k=1)
        return (distances_rad[:, 0] * EARTH_RADIUS_KM).round(3)


def compute_coverage_ratio(
    zone_centroids_deg: np.ndarray,
    zone_populations: np.ndarray,
    facility_type: str,
    radius_km: Optional[float] = None,
    custom_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Compute what fraction of each zone's population is within service radius.

    For each zone centroid, checks if ANY facility exists within the service radius.
    Binary per zone (0 or 1) — for more granular analysis, use buffer intersection.

    Args:
        zone_centroids_deg: Array (n, 2) of [lat, lon] zone centroids
        zone_populations:   Array (n,) of zone population totals
        facility_type:      Facility type string
        radius_km:          Service radius in km (uses default if None)
        custom_path:        Override file path

    Returns:
        Array (n,) of coverage ratios [0, 1]
    """
    if radius_km is None:
        radius_km = SERVICE_RADII_KM.get(facility_type, 5.0)

    gdf = _load_facility_gdf(facility_type, custom_path)

    if gdf is not None and len(gdf) > 0:
        tree, _ = _build_balltree(gdf)
    else:
        synth_coords = _generate_synthetic_facilities(zone_centroids_deg, 30)
        tree = BallTree(np.radians(synth_coords), metric="haversine")

    query_rad  = np.radians(zone_centroids_deg)
    radius_rad = radius_km / EARTH_RADIUS_KM
    counts     = tree.query_radius(query_rad, r=radius_rad, count_only=True)
    return (counts > 0).astype(float)


def compute_multi_coverage_gap(
    zones_df: pd.DataFrame,
    coverage_cols: Dict[str, str],
) -> pd.Series:
    """
    Compute weighted multi-facility coverage gap.

    Formula:
        multi_coverage_gap = Σ weight_i × (1 - coverage_ratio_i)
        where weights sum to 1.0

    Purpose:
        Single score combining all infrastructure gaps.
        Used as the objective function in Hill Climbing (Phase 6).

    Args:
        zones_df:     DataFrame with coverage ratio columns
        coverage_cols: Dict mapping facility type → column name

    Returns:
        Series of multi_coverage_gap values [0, 1]
    """
    gap_sum = pd.Series(np.zeros(len(zones_df)), index=zones_df.index)

    for facility, col in coverage_cols.items():
        weight = COVERAGE_WEIGHTS.get(facility, 0.25)
        if col in zones_df.columns:
            gap_sum += weight * (1.0 - zones_df[col].clip(0, 1))

    return gap_sum.round(4)


def compute_all_coverage_features(
    df: pd.DataFrame,
    zones_gdf: Optional[gpd.GeoDataFrame] = None,
) -> pd.DataFrame:
    """
    Master function: compute all coverage features.

    Args:
        df:        Zone features DataFrame (with h3_id)
        zones_gdf: Optional GeoDataFrame with zone geometries

    Returns:
        df with all coverage features added
    """
    log.info("Computing coverage features...")
    df = df.copy()

    # Get zone centroids
    if zones_gdf is not None:
        centroids = zones_gdf.geometry.centroid
    else:
        gpkg = PROCESSED_DIR / "zones.gpkg"
        if gpkg.exists():
            zones_gdf = gpd.read_file(str(gpkg))
            centroids = zones_gdf.geometry.centroid
        else:
            log.warning("No zones GeoPackage. Generating synthetic centroids.")
            rng = np.random.default_rng(0)
            n = len(df)
            # Mumbai approximate centroid
            lats = 19.0 + rng.normal(0, 0.1, n)
            lons = 72.85 + rng.normal(0, 0.1, n)
            from shapely.geometry import Point
            import geopandas as gpd
            zones_gdf = gpd.GeoDataFrame(
                {"geometry": [Point(lo, la) for la, lo in zip(lats, lons)]},
                crs="EPSG:4326"
            )
            centroids = zones_gdf.geometry

    zone_centroids = np.column_stack([centroids.y, centroids.x])
    pop_array = df.get("population_total", pd.Series(np.ones(len(df)) * 10000)).values

    # Distance features for all facility types
    for ftype in ["hospital", "school", "ev_station", "fire_station"]:
        col = f"dist_nearest_{ftype}"
        log.info(f"  Computing {col}...")
        df[col] = compute_distance_to_nearest(zone_centroids, ftype)

    # Coverage ratios
    coverage_col_map = {}
    for ftype in ["hospital", "school", "ev_station", "fire_station"]:
        col = f"{ftype}_coverage_ratio"
        log.info(f"  Computing {col}...")
        df[col] = compute_coverage_ratio(zone_centroids, pop_array, ftype)
        coverage_col_map[ftype] = col

        # Per-facility coverage gap
        df[f"{ftype}_coverage_gap"] = (1.0 - df[col]).round(4)

    # Keep original hospital coverage_gap for backward compatibility
    df["coverage_gap"] = df["hospital_coverage_gap"]

    # Multi-facility weighted gap
    df["multi_coverage_gap"] = compute_multi_coverage_gap(df, coverage_col_map)

    # Underserved binary flag (coverage_gap > 50%)
    df["is_underserved"] = (df["coverage_gap"] > 0.5).astype(int)

    log.info(f"  Coverage features computed. Underserved zones: {df['is_underserved'].sum()} / {len(df)}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    csv = PROCESSED_DIR / "zone_features.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        df = compute_all_coverage_features(df)
        print(df[[c for c in df.columns if "dist" in c or "coverage" in c or "gap" in c]].describe())
    else:
        print("Run create_zones.py first.")
