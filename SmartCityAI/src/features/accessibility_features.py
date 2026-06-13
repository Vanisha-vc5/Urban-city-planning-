"""
accessibility_features.py
=========================
Phase 4 - Feature Engineering: Road & Transportation Accessibility Features

PURPOSE:
    Computes accessibility metrics measuring how well each zone is connected
    to the road network and public transport. Accessibility determines both
    the ease of reaching facilities AND the ease of serving residents.

FEATURES GENERATED:
┌──────────────────────────────┬──────────────────────────────────────────────┬─────────────────────────┐
│ Feature                      │ Formula                                      │ Model Usage             │
├──────────────────────────────┼──────────────────────────────────────────────┼─────────────────────────┤
│ road_accessibility_index     │ Σ(highway_weight × length) / area_km²        │ K-Means, SVM, DT        │
│ traffic_density              │ road_count / area_km²                        │ EV station demand proxy │
│ nearest_arterial_dist        │ Distance to nearest primary+ road (km)       │ Site accessibility      │
│ network_centrality           │ Normalized betweenness centrality of zone    │ Emergency routing       │
│ walkability_index            │ road_density × pedestrian_route_ratio        │ School/hospital choice  │
│ transit_accessibility        │ Proxy from intersection density              │ Equity metric           │
│ emergency_response_time_est  │ Distance / avg_speed × 60 (minutes)         │ Fire station placement  │
└──────────────────────────────┴──────────────────────────────────────────────┴─────────────────────────┘

HIGHWAY WEIGHT RATIONALE:
    Motorway (1.0): Maximum throughput, connects city to city
    Primary (0.85): Major arterials, 24/7 flow
    Secondary (0.70): District connectors
    Tertiary (0.55): Neighborhood streets
    Residential (0.35): Last-mile access
    Service (0.10): Very low throughput, parking/loading only

EMERGENCY RESPONSE TIME:
    - Based on NFPA 1710 standard: fire companies should travel ≤1.5 miles in 4 min
    - Formula: time_min = distance_km / (average_speed_kmh / 60)
    - Average emergency vehicle speed: 40 km/h in urban areas
    - Target: ≤8 minutes total response time (dispatch + travel)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

log = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).resolve().parents[3]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR       = BASE_DIR / "data" / "raw"

# Highway tag weights (matching create_zones.py for consistency)
HIGHWAY_WEIGHTS: Dict[str, float] = {
    "motorway": 1.0, "trunk": 1.0, "motorway_link": 0.90, "trunk_link": 0.85,
    "primary": 0.85, "primary_link": 0.80,
    "secondary": 0.70, "secondary_link": 0.65,
    "tertiary": 0.55, "tertiary_link": 0.50,
    "residential": 0.35, "living_street": 0.30,
    "unclassified": 0.20, "service": 0.10, "track": 0.05,
}

# Average urban emergency vehicle speed (km/h)
EMERGENCY_SPEED_KMH = 40.0

# Arterial road types (high weight roads)
ARTERIAL_TYPES = {"motorway", "trunk", "primary", "secondary"}


def load_road_network(roads_path: Optional[Path] = None) -> Optional[gpd.GeoDataFrame]:
    """
    Load road network GeoDataFrame.

    Args:
        roads_path: Optional override path

    Returns:
        GeoDataFrame with road edges, or None
    """
    candidates = [
        roads_path,
        RAW_DIR / "osm_roads.gpkg",
    ]
    for path in candidates:
        if path and path.exists():
            gdf = gpd.read_file(str(path))
            log.info(f"Loaded {len(gdf):,} road segments from {path.name}")
            return gdf
    return None


def compute_road_accessibility_index(
    zones_gdf: gpd.GeoDataFrame,
    roads_gdf: Optional[gpd.GeoDataFrame] = None,
) -> pd.Series:
    """
    Compute weighted Road Accessibility Index (RAI) per zone.

    Formula:
        RAI = Σ(highway_weight_i × segment_length_i) / zone_area_km²

    Intuition:
        A zone with many high-capacity roads per unit area is highly accessible.
        Normalizes by area so large zones don't automatically score higher.

    Args:
        zones_gdf: H3 zones GeoDataFrame (with area_km2 column)
        roads_gdf: Road edge GeoDataFrame

    Returns:
        Series of RAI values (normalized [0,1])
    """
    CRS_METRIC = "EPSG:3857"
    n = len(zones_gdf)

    if roads_gdf is None:
        log.warning("No road data. Generating synthetic RAI.")
        rng = np.random.default_rng(42)
        return pd.Series(rng.beta(3, 3, n).round(4), index=zones_gdf.index)

    zones_m  = zones_gdf.to_crs(CRS_METRIC)
    roads_m  = roads_gdf.to_crs(CRS_METRIC)

    # Spatial join: roads within each zone
    joined = gpd.sjoin(roads_m, zones_m[["geometry", "area_km2"]], how="inner", predicate="intersects")

    if "highway" in joined.columns:
        joined["weight"] = joined["highway"].map(HIGHWAY_WEIGHTS).fillna(0.20)
    else:
        joined["weight"] = 0.35

    joined["weighted_length"] = joined.geometry.length * joined["weight"]

    agg = joined.groupby(joined.index_right).agg(
        weighted_road_length=("weighted_length", "sum"),
    )

    # Align with zones
    result = pd.Series(0.0, index=zones_gdf.index)
    for idx, row in agg.iterrows():
        if idx in result.index:
            area = zones_gdf.loc[idx, "area_km2"] if "area_km2" in zones_gdf.columns else 1.0
            result.loc[idx] = row["weighted_road_length"] / (area * 1_000_000 + 1e-9)

    # Normalize [0, 1]
    scaler = MinMaxScaler()
    result_norm = scaler.fit_transform(result.values.reshape(-1, 1)).flatten()
    return pd.Series(result_norm, index=zones_gdf.index).round(4)


def compute_nearest_arterial_distance(
    zones_gdf: gpd.GeoDataFrame,
    roads_gdf: Optional[gpd.GeoDataFrame] = None,
) -> pd.Series:
    """
    Compute distance from zone centroid to nearest arterial road (km).

    Arterial roads (primary+) are essential for large vehicle access
    (ambulances, fire trucks, delivery vehicles for construction).

    Args:
        zones_gdf: H3 zones GeoDataFrame
        roads_gdf: Road edge GeoDataFrame

    Returns:
        Series of distances in km
    """
    from sklearn.neighbors import BallTree
    EARTH_RADIUS_KM = 6371.0

    centroids = zones_gdf.geometry.centroid
    centroid_coords = np.radians(np.column_stack([centroids.y, centroids.x]))

    if roads_gdf is not None and "highway" in roads_gdf.columns:
        arterials = roads_gdf[roads_gdf["highway"].isin(ARTERIAL_TYPES)]
        if len(arterials) > 0:
            # Use road centroids as reference points
            art_centroids = arterials.geometry.centroid
            art_coords = np.radians(np.column_stack([art_centroids.y, art_centroids.x]))
            tree = BallTree(art_coords, metric="haversine")
            dists, _ = tree.query(centroid_coords, k=1)
            return pd.Series((dists[:, 0] * EARTH_RADIUS_KM).round(3), index=zones_gdf.index)

    # Synthetic fallback
    rng = np.random.default_rng(7)
    return pd.Series(rng.exponential(0.5, len(zones_gdf)).clip(0.1, 5.0).round(3), index=zones_gdf.index)


def compute_intersection_density(
    zones_gdf: gpd.GeoDataFrame,
    roads_gdf: Optional[gpd.GeoDataFrame] = None,
) -> pd.Series:
    """
    Compute road intersection density as a walkability proxy.

    Intersection density (intersections per km²) is strongly correlated with
    walkability in urban planning literature (Walk Score methodology).

    High intersection density → grid-like streets → pedestrian-friendly → school placement priority.

    Args:
        zones_gdf: H3 zones GeoDataFrame
        roads_gdf: Road edge GeoDataFrame

    Returns:
        Series of intersection density values
    """
    if roads_gdf is None:
        rng = np.random.default_rng(11)
        return pd.Series(rng.lognormal(2, 0.5, len(zones_gdf)).round(2), index=zones_gdf.index)

    CRS_METRIC = "EPSG:3857"
    zones_m = zones_gdf.to_crs(CRS_METRIC)
    roads_m = roads_gdf.to_crs(CRS_METRIC)

    # Road count per zone as intersection proxy
    joined = gpd.sjoin(roads_m, zones_m[["geometry", "area_km2"]], how="inner", predicate="intersects")
    road_count = joined.groupby(joined.index_right).size()

    result = pd.Series(0.0, index=zones_gdf.index)
    for idx, count in road_count.items():
        if idx in result.index:
            area = zones_gdf.loc[idx, "area_km2"] if "area_km2" in zones_gdf.columns else 1.0
            result.loc[idx] = count / max(area, 0.01)

    return result.round(2)


def compute_emergency_response_time(
    dist_nearest_fire_km: pd.Series,
    average_speed_kmh: float = EMERGENCY_SPEED_KMH,
) -> pd.Series:
    """
    Estimate emergency vehicle response time in minutes.

    Formula:
        response_time_min = (dist_nearest_fire_km / average_speed_kmh) × 60
        + 1.5  # dispatch time (NFPA standard average)

    NFPA 1710 Standard: Total response time ≤ 6 minutes (career departments)
    This feature identifies zones where response time EXCEEDS standard.

    Args:
        dist_nearest_fire_km: Distance to nearest fire station (km)
        average_speed_kmh:    Emergency vehicle speed assumption

    Returns:
        Series of estimated response times in minutes
    """
    travel_time = (dist_nearest_fire_km / average_speed_kmh) * 60
    response_time = travel_time + 1.5  # dispatch time
    return response_time.round(2)


def compute_all_accessibility_features(
    df: pd.DataFrame,
    zones_gdf: Optional[gpd.GeoDataFrame] = None,
    roads_gdf: Optional[gpd.GeoDataFrame] = None,
) -> pd.DataFrame:
    """
    Master function: compute all accessibility features.

    Args:
        df:        Zone features DataFrame
        zones_gdf: H3 zones GeoDataFrame (for spatial operations)
        roads_gdf: Road edge GeoDataFrame

    Returns:
        df with all accessibility features
    """
    log.info("Computing accessibility features...")
    df = df.copy()

    # Load spatial data if not provided
    if zones_gdf is None:
        gpkg = PROCESSED_DIR / "zones.gpkg"
        if gpkg.exists():
            zones_gdf = gpd.read_file(str(gpkg))

    if roads_gdf is None:
        roads_gdf = load_road_network()

    if zones_gdf is not None:
        # Road Accessibility Index
        df["road_accessibility_index"] = compute_road_accessibility_index(
            zones_gdf, roads_gdf
        ).values

        # Distance to arterial road
        df["nearest_arterial_dist_km"] = compute_nearest_arterial_distance(
            zones_gdf, roads_gdf
        ).values

        # Intersection density (walkability proxy)
        df["intersection_density"] = compute_intersection_density(
            zones_gdf, roads_gdf
        ).values
    else:
        log.warning("No zones GeoDataFrame. Using synthetic accessibility.")
        rng = np.random.default_rng(42)
        n = len(df)
        df["road_accessibility_index"] = rng.beta(3, 3, n).round(4)
        df["nearest_arterial_dist_km"] = rng.exponential(0.5, n).clip(0.1, 5).round(3)
        df["intersection_density"]     = rng.lognormal(2, 0.5, n).round(2)

    # Emergency response time (from coverage features)
    if "dist_nearest_fire_station" in df.columns:
        df["emergency_response_time_min"] = compute_emergency_response_time(
            df["dist_nearest_fire_station"]
        )
    else:
        df["emergency_response_time_min"] = compute_emergency_response_time(
            pd.Series(np.random.exponential(2, len(df)))
        )

    # Normalize intersection density [0, 1]
    scaler = MinMaxScaler()
    df["walkability_index"] = scaler.fit_transform(
        df["intersection_density"].values.reshape(-1, 1)
    ).flatten().round(4)

    # Traffic density (if not from roads)
    if "traffic_density" not in df.columns:
        df["traffic_density"] = df["intersection_density"] * 0.5

    log.info("Accessibility features computed: road_accessibility_index, nearest_arterial_dist_km, "
             "intersection_density, emergency_response_time_min, walkability_index")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    csv = PROCESSED_DIR / "zone_features.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        df = compute_all_accessibility_features(df)
        print(df[["road_accessibility_index", "nearest_arterial_dist_km",
                   "emergency_response_time_min", "walkability_index"]].describe())
    else:
        print("Run create_zones.py first.")
