"""
create_zones.py
===============
Phase 3 - Data Processing: Urban Zone Creation & Feature Aggregation

PURPOSE:
    Divides a city into H3 hexagonal zones and performs spatial joins
    to aggregate all downloaded data sources into a single feature table.

    This produces the master zone_features.csv used by all ML models.

WHY H3 HEXAGONS OVER ADMINISTRATIVE WARDS?
    ┌─────────────────────────────────────────────────────────────────┐
    │ H3 HEXAGONS (CHOSEN)           ADMIN WARDS                     │
    ├─────────────────────────────────────────────────────────────────┤
    │ Uniform area per cell          Highly variable ward sizes       │
    │ Resolution configurable (6-12) Fixed by government boundaries   │
    │ No Modifiable Areal Unit       MAUP problem is severe           │
    │ Hierarchical (zoom in/out)     No hierarchy                     │
    │ Globally consistent            India-specific only              │
    │ Vectorized H3 math             Slow polygon operations          │
    │ Used by Uber, WHO, UN          Legacy government format         │
    │ No boundary bias               Boundary artifacts in analysis   │
    └─────────────────────────────────────────────────────────────────┘

    Resolution 8 chosen: ~0.74 km² per cell, ~9 km edge-to-edge.
    Perfect granularity for "where should a hospital go?" questions.
    A hospital serves ~3–5 km radius → 1–3 H3 cells at res 8.

SPATIAL OPERATIONS:
    1. Generate H3 grid covering city bounding box
    2. Spatial join H3 ← Population rasters (rasterio zonal stats)
    3. Spatial join H3 ← Hospital points (count + min distance)
    4. Spatial join H3 ← OSM road edges (density calculation)
    5. Spatial join H3 ← Census ward polygons (demographic interpolation)
    6. Compute coverage gaps and composite metrics
    7. Output: zone_features.csv

COMPLEXITY:
    O(n_cells × n_facilities) for nearest-neighbor joins
    Vectorized with GeoPandas spatial index (STRtree) for O(n log n)

USAGE:
    python create_zones.py --city "Mumbai, India" --resolution 8
    python create_zones.py --city "Delhi, India" --resolution 9

OUTPUT:
    data/processed/zone_features.csv    (master feature table)
    data/processed/zones.gpkg           (H3 hexagon geometries)
    data/processed/zone_features_viz.gpkg  (joined for visualization)
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, mapping
from sklearn.preprocessing import MinMaxScaler

# Conditionally import rasterio (may not be installed in all envs)
try:
    import rasterio
    from rasterio.mask import mask as rasterio_mask
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("create_zones")

BASE_DIR      = Path(__file__).resolve().parents[3]
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

H3_RESOLUTION = 8       # ~0.74 km² per cell
CRS_WGS84     = "EPSG:4326"
CRS_METRIC    = "EPSG:3857"   # Web Mercator for metric calculations

# Hospital service radius for coverage calculation (meters)
HOSPITAL_SERVICE_RADIUS_M = 5000  # 5 km (WHO recommendation)


# ── H3 Grid Generation ────────────────────────────────────────────────────────

def get_city_bbox(city: str) -> Tuple[float, float, float, float]:
    """
    Get bounding box for a city using OSMnx geocoder.

    Falls back to Mumbai coordinates if geocoding fails.

    Args:
        city: City name string (e.g., "Mumbai, India")

    Returns:
        (min_lat, min_lon, max_lat, max_lon) tuple
    """
    try:
        import osmnx as ox
        gdf = ox.geocode_to_gdf(city)
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy] = [min_lon, min_lat, max_lon, max_lat]
        return bounds[1], bounds[0], bounds[3], bounds[2]  # min_lat, min_lon, max_lat, max_lon
    except Exception:
        log.warning("Geocoding failed. Using Mumbai default bbox.")
        return 18.87, 72.77, 19.27, 73.00  # Mumbai


def generate_h3_grid(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    resolution: int = H3_RESOLUTION,
) -> gpd.GeoDataFrame:
    """
    Generate H3 hexagonal grid covering the bounding box.

    Algorithm:
    1. Create a rectangular polygon from bbox
    2. Find all H3 cells at given resolution that intersect this polygon
    3. Convert each H3 cell to a Shapely polygon
    4. Return as GeoDataFrame

    H3 cell area at various resolutions:
    - Res 6: ~36.1 km²   (district scale)
    - Res 7: ~5.16 km²   (neighborhood scale)
    - Res 8: ~0.74 km²   (block scale) ← CHOSEN
    - Res 9: ~0.10 km²   (parcel scale)

    Args:
        min_lat, min_lon, max_lat, max_lon: Bounding box
        resolution: H3 resolution (6–10)

    Returns:
        GeoDataFrame with columns: h3_id, geometry, area_km2, resolution
    """
    log.info(f"Generating H3 grid (resolution {resolution}) for bbox: "
             f"({min_lat:.3f}, {min_lon:.3f}) → ({max_lat:.3f}, {max_lon:.3f})")

    # Cover bbox with H3 cells (polyfill)
    # H3 polyfill expects GeoJSON polygon format
    bbox_polygon = {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]]
    }

    # h3-py 4.0+ converts GeoJSON to H3Shape, then extracts cells
    h3shape = h3.geo_to_h3shape(bbox_polygon)
    cell_ids = h3.h3shape_to_cells(h3shape, resolution)

    log.info(f"  Generated {len(cell_ids):,} H3 cells")

    # Convert to Shapely polygons
    records = []
    for cell_id in cell_ids:
        # h3-py 4.0+ uses cell_to_boundary()
        boundary = h3.cell_to_boundary(cell_id)
        coords = [(lon, lat) for lat, lon in boundary]  # H3 returns lat,lon → convert to lon,lat
        polygon = Polygon(coords)

        records.append({
            "h3_id": cell_id,
            "geometry": polygon,
            "resolution": resolution,
        })

    gdf = gpd.GeoDataFrame(records, crs=CRS_WGS84)

    # Calculate area in km²
    gdf_proj = gdf.to_crs(CRS_METRIC)
    gdf["area_km2"] = (gdf_proj.geometry.area / 1_000_000).round(4)

    log.info(f"  Mean cell area: {gdf['area_km2'].mean():.3f} km²")
    return gdf


# ── Spatial Join: Population ──────────────────────────────────────────────────

def add_population_features(
    zones: gpd.GeoDataFrame,
    population_raster_path: Optional[Path] = None,
    census_csv_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """
    Add population density and demographic features to H3 zones.

    Method:
    - If raster available: zonal statistics (sum population pixels per hex)
    - Else: spatial join with census CSV (ward-level interpolation)
    - Population density = total_pop / area_km²

    Args:
        zones:                  H3 GeoDataFrame
        population_raster_path: Path to WorldPop GeoTIFF
        census_csv_path:        Path to census CSV (fallback)

    Returns:
        zones with added columns: population_total, population_density,
                                  elderly_ratio, income_bracket_norm
    """
    log.info("Adding population features...")
    zones = zones.copy()

    # Method 1: WorldPop raster zonal statistics
    if population_raster_path and population_raster_path.exists() and RASTERIO_AVAILABLE:
        log.info(f"  Using WorldPop raster: {population_raster_path.name}")
        populations = []

        with rasterio.open(population_raster_path) as src:
            for _, row in zones.iterrows():
                try:
                    geom = [mapping(row.geometry)]
                    out_image, _ = rasterio_mask(src, geom, crop=True, nodata=0)
                    pop_sum = float(np.nansum(out_image[out_image > 0]))
                    populations.append(max(pop_sum, 0))
                except Exception:
                    populations.append(0.0)

        zones["population_total"] = populations

    # Method 2: Census CSV areal interpolation
    elif census_csv_path and census_csv_path.exists():
        log.info(f"  Using census CSV: {census_csv_path.name}")
        census_df = pd.read_csv(census_csv_path)
        # Distribute census population across zones proportionally
        total_pop = census_df["population_total"].sum() if "population_total" in census_df.columns else 1_000_000
        zones["population_total"] = total_pop / len(zones)  # Simple uniform distribution
        if "elderly_ratio" in census_df.columns:
            zones["elderly_ratio"] = census_df["elderly_ratio"].mean()
        if "income_bracket_norm" in census_df.columns:
            zones["income_bracket_norm"] = census_df["income_bracket_norm"].mean()

    # Method 3: Synthetic fallback (for demo/testing)
    else:
        log.info("  Generating synthetic population data (no raster/census file found)")
        rng = np.random.default_rng(42)
        n = len(zones)

        # Realistic urban population distribution
        center_lat = zones.geometry.centroid.y.mean()
        center_lon = zones.geometry.centroid.x.mean()
        centroids = zones.geometry.centroid

        # Distance from city center affects density (monocentric city model)
        dist_from_center = np.sqrt(
            (centroids.x - center_lon) ** 2 + (centroids.y - center_lat) ** 2
        )
        dist_norm = (dist_from_center - dist_from_center.min()) / (dist_from_center.max() - dist_from_center.min() + 1e-6)

        # Dense core (30,000/km²), sparse periphery (5,000/km²) — Mumbai-like
        base_density = 30000 * np.exp(-2 * dist_norm) + 5000
        noise = rng.normal(0, 2000, n)
        pop_density = np.clip(base_density + noise, 1000, 80000)

        zones["population_density"]  = pop_density.round(0)
        zones["population_total"]    = (pop_density * zones["area_km2"]).round(0)
        zones["elderly_ratio"]       = rng.beta(2.5, 28, n).round(4)
        zones["income_bracket_norm"] = rng.beta(3, 5, n).round(4)

    # Calculate density if not done via raster
    if "population_density" not in zones.columns:
        zones["population_density"] = (
            zones["population_total"] / zones["area_km2"].clip(lower=0.01)
        ).round(0)

    # Fill demographic defaults if missing
    if "elderly_ratio" not in zones.columns:
        zones["elderly_ratio"] = 0.085
    if "income_bracket_norm" not in zones.columns:
        zones["income_bracket_norm"] = 0.5

    log.info(f"  Pop density range: {zones['population_density'].min():.0f} – {zones['population_density'].max():.0f} /km²")
    return zones


# ── Spatial Join: Hospitals ───────────────────────────────────────────────────

def add_hospital_features(
    zones: gpd.GeoDataFrame,
    hospitals_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """
    Add hospital accessibility features to H3 zones.

    Computed features:
    - dist_nearest_hospital: Haversine distance to nearest hospital (km)
    - hospitals_in_radius:   Count of hospitals within 5 km
    - hospital_coverage_ratio: fraction of zone population within 5km of a hospital
    - coverage_gap: 1 - hospital_coverage_ratio

    Uses sklearn.neighbors.BallTree for efficient nearest-neighbor search.
    BallTree with haversine metric is O(n log n) — essential for large cities.

    Args:
        zones:          H3 GeoDataFrame
        hospitals_path: Path to hospitals GeoPackage

    Returns:
        zones with hospital feature columns added
    """
    from sklearn.neighbors import BallTree

    log.info("Adding hospital accessibility features...")
    zones = zones.copy()
    zone_centroids = zones.geometry.centroid

    # Load hospital data
    hospitals_gdf = None
    for candidate_path in [
        hospitals_path,
        RAW_DIR / "healthsites_IN.gpkg",
        RAW_DIR / "osm_hospitals.gpkg",
    ]:
        if candidate_path and candidate_path.exists():
            hospitals_gdf = gpd.read_file(str(candidate_path))
            log.info(f"  Loaded {len(hospitals_gdf):,} hospitals from {candidate_path.name}")
            break

    # Fallback: synthetic hospitals
    if hospitals_gdf is None or len(hospitals_gdf) == 0:
        log.info("  No hospital file found. Using synthetic hospital locations.")
        rng = np.random.default_rng(42)
        n_hospitals = 40  # Realistic for a major Indian city subdivision

        center_lat = zone_centroids.y.mean()
        center_lon = zone_centroids.x.mean()

        hosp_lats = center_lat + rng.normal(0, 0.06, n_hospitals)
        hosp_lons = center_lon + rng.normal(0, 0.06, n_hospitals)

        from shapely.geometry import Point
        hospitals_gdf = gpd.GeoDataFrame(
            {"geometry": [Point(lon, lat) for lon, lat in zip(hosp_lons, hosp_lats)],
             "name": [f"Hospital {i+1}" for i in range(n_hospitals)]},
            crs=CRS_WGS84,
        )

    # Ensure point geometry
    hospitals_gdf = hospitals_gdf.copy()
    hospitals_gdf["geometry"] = hospitals_gdf.geometry.centroid

    # Build BallTree on hospital coordinates (radians for haversine)
    hosp_coords = np.radians(
        np.column_stack([hospitals_gdf.geometry.y, hospitals_gdf.geometry.x])
    )
    tree = BallTree(hosp_coords, metric="haversine")

    # Query BallTree for each zone centroid
    zone_coords = np.radians(
        np.column_stack([zone_centroids.y, zone_centroids.x])
    )

    # Nearest neighbor distance
    distances_rad, _ = tree.query(zone_coords, k=1)
    EARTH_RADIUS_KM = 6371.0
    zones["dist_nearest_hospital"] = (distances_rad[:, 0] * EARTH_RADIUS_KM).round(3)

    # Count hospitals within service radius
    radius_rad = HOSPITAL_SERVICE_RADIUS_M / (EARTH_RADIUS_KM * 1000)
    counts = tree.query_radius(zone_coords, r=radius_rad, count_only=True)
    zones["hospitals_in_5km"] = counts

    # Coverage ratio: whether this zone is within service radius of ANY hospital
    zones["hospital_coverage_ratio"] = (counts > 0).astype(float)
    zones["coverage_gap"] = (1.0 - zones["hospital_coverage_ratio"]).round(4)

    log.info(f"  Dist nearest hospital: {zones['dist_nearest_hospital'].min():.2f} – {zones['dist_nearest_hospital'].max():.2f} km")
    log.info(f"  Zones with no hospital coverage: {(zones['coverage_gap'] == 1.0).sum()} / {len(zones)}")
    return zones


# ── Spatial Join: Road Accessibility ─────────────────────────────────────────

def add_road_features(
    zones: gpd.GeoDataFrame,
    roads_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """
    Add road accessibility features to H3 zones.

    Road Accessibility Index formula:
        RAI = (Σ highway_weight_i × length_i) / area_km²

    Highway weights (by OSM highway tag):
        motorway/trunk:   1.0  (highest connectivity)
        primary:          0.85
        secondary:        0.70
        tertiary:         0.55
        residential:      0.35
        unclassified:     0.20
        service:          0.10

    Higher RAI = better connected zone = easier infrastructure access.

    Args:
        zones:      H3 GeoDataFrame
        roads_path: Path to roads GeoPackage

    Returns:
        zones with road_accessibility_index, traffic_density columns
    """
    log.info("Adding road accessibility features...")
    zones = zones.copy()

    HIGHWAY_WEIGHTS = {
        "motorway": 1.0, "trunk": 1.0,
        "primary": 0.85, "primary_link": 0.80,
        "secondary": 0.70, "secondary_link": 0.65,
        "tertiary": 0.55, "tertiary_link": 0.50,
        "residential": 0.35, "living_street": 0.30,
        "unclassified": 0.20, "service": 0.10,
    }

    # Load roads
    roads_gdf = None
    road_file = roads_path or RAW_DIR / "osm_roads.gpkg"
    if road_file.exists():
        roads_gdf = gpd.read_file(str(road_file))
        log.info(f"  Loaded {len(roads_gdf):,} road segments")
    else:
        log.info("  No roads file. Using synthetic accessibility scores.")

    if roads_gdf is not None and len(roads_gdf) > 0:
        # Reproject to metric CRS for length calculation
        roads_metric = roads_gdf.to_crs(CRS_METRIC)
        zones_metric  = zones.to_crs(CRS_METRIC)

        # Spatial join: roads within each zone
        joined = gpd.sjoin(roads_metric, zones_metric[["h3_id", "geometry", "area_km2"]], how="inner", predicate="intersects")

        # Calculate weighted road length per zone
        if "highway" in joined.columns:
            joined["weight"] = joined["highway"].map(HIGHWAY_WEIGHTS).fillna(0.20)
        else:
            joined["weight"] = 0.35

        joined["weighted_length"] = joined.geometry.length * joined["weight"]

        agg = joined.groupby("h3_id").agg(
            total_road_length=("geometry", lambda g: g.length.sum()),
            weighted_road_length=("weighted_length", "sum"),
            road_count=("geometry", "count"),
        ).reset_index()

        zones = zones.merge(agg, on="h3_id", how="left")
        zones["total_road_length"]    = zones["total_road_length"].fillna(0)
        zones["weighted_road_length"] = zones["weighted_road_length"].fillna(0)

        # Road Accessibility Index
        zones["road_accessibility_index"] = (
            zones["weighted_road_length"] / (zones["area_km2"].clip(lower=0.01) * 1_000_000)
        ).round(6)

        # Traffic density proxy: road count per km²
        zones["traffic_density"] = (
            zones["road_count"].fillna(0) / zones["area_km2"].clip(lower=0.01)
        ).round(2)

    else:
        # Synthetic fallback
        rng = np.random.default_rng(123)
        n = len(zones)
        zones["road_accessibility_index"] = rng.beta(3, 3, n).round(4)
        zones["traffic_density"] = (rng.lognormal(2.5, 0.7, n)).round(2)
        zones["total_road_length"] = 0
        zones["weighted_road_length"] = 0

    # Normalize road accessibility to [0, 1]
    rai = zones["road_accessibility_index"]
    zones["road_accessibility_index"] = (
        (rai - rai.min()) / (rai.max() - rai.min() + 1e-9)
    ).round(4)

    log.info(f"  RAI range: {zones['road_accessibility_index'].min():.4f} – {zones['road_accessibility_index'].max():.4f}")
    return zones


# ── Priority Score ────────────────────────────────────────────────────────────

def compute_priority_score(zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute a composite priority score for infrastructure need.

    Formula (weighted sum, range 0–100):
        priority_score = (
            0.35 × coverage_gap_norm +
            0.25 × pop_density_norm +
            0.20 × elderly_ratio_norm +
            0.10 × (1 - income_norm)    +   # inverse: poor areas need more
            0.10 × (1 - road_access_norm)   # inverse: inaccessible areas need more
        ) × 100

    This is not the ML output — it is the labeled target variable
    used to generate synthetic labels for supervised models (SVM, DT).

    Args:
        zones: GeoDataFrame with all feature columns

    Returns:
        zones with priority_score and priority_class columns
    """
    log.info("Computing priority scores...")
    scaler = MinMaxScaler()

    feature_cols = [
        "coverage_gap",
        "population_density",
        "elderly_ratio",
        "income_bracket_norm",
        "road_accessibility_index",
    ]

    # Fill missing columns with defaults
    for col in feature_cols:
        if col not in zones.columns:
            zones[col] = 0.5

    X = zones[feature_cols].fillna(0).values
    X_norm = scaler.fit_transform(X)

    weights = np.array([0.35, 0.25, 0.20, 0.10, 0.10])
    inverted = np.array([False, False, False, True, True])  # True = invert for priority

    X_weighted = np.where(inverted, 1 - X_norm, X_norm) * weights
    priority_raw = X_weighted.sum(axis=1)
    zones["priority_score"] = (priority_raw * 100).round(1)

    # Priority classes for SVM/DT classification
    zones["priority_class"] = pd.cut(
        zones["priority_score"],
        bins=[0, 33, 66, 100],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    log.info(f"  Priority class distribution:\n{zones['priority_class'].value_counts().to_string()}")
    return zones


# ── Master Output ─────────────────────────────────────────────────────────────

def build_zone_features_csv(zones: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Extract the feature table from the GeoDataFrame.

    The CSV contains only numeric/categorical features — no geometry.
    Geometry is saved separately in the GeoPackage.

    Returns:
        DataFrame ready for ML training
    """
    feature_columns = [
        "h3_id",
        "area_km2",
        "population_total",
        "population_density",
        "elderly_ratio",
        "income_bracket_norm",
        "dist_nearest_hospital",
        "hospitals_in_5km",
        "hospital_coverage_ratio",
        "coverage_gap",
        "road_accessibility_index",
        "traffic_density",
        "priority_score",
        "priority_class",
    ]

    available = [c for c in feature_columns if c in zones.columns]
    df = zones[available].copy()
    df = df.fillna(0)
    return df


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Create H3 zone feature table for SmartCityAI")
    parser.add_argument("--city",       type=str, default="Mumbai, India", help="Target city")
    parser.add_argument("--resolution", type=int, default=8, help="H3 resolution (6–10)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SmartCityAI — Zone Creation Pipeline")
    log.info(f"City: {args.city} | H3 Resolution: {args.resolution}")
    log.info("=" * 60)

    # Step 1: Get bounding box
    min_lat, min_lon, max_lat, max_lon = get_city_bbox(args.city)

    # Step 2: Generate H3 grid
    zones = generate_h3_grid(min_lat, min_lon, max_lat, max_lon, args.resolution)

    # Step 3: Add population features
    pop_raster = next(RAW_DIR.glob("worldpop_population_*.tif"), None)
    census_csv = next(RAW_DIR.glob("census_IN_*.csv"), None)
    zones = add_population_features(zones, pop_raster, census_csv)

    # Step 4: Add hospital features
    zones = add_hospital_features(zones)

    # Step 5: Add road features
    zones = add_road_features(zones)

    # Step 6: Compute priority score
    zones = compute_priority_score(zones)

    # Step 7: Save GeoPackage (with geometry for visualization)
    gpkg_path = PROCESSED_DIR / "zones.gpkg"
    zones.to_file(str(gpkg_path), driver="GPKG", layer="zones")
    log.info(f"Zones GeoPackage saved → {gpkg_path}")

    # Step 8: Save feature CSV (without geometry for ML)
    feature_df = build_zone_features_csv(zones)
    csv_path   = PROCESSED_DIR / "zone_features.csv"
    feature_df.to_csv(csv_path, index=False)
    log.info(f"Feature CSV saved → {csv_path} ({len(feature_df):,} zones, {len(feature_df.columns)} features)")
    log.info(f"\nFeature table preview:\n{feature_df.describe().round(3).to_string()}")


if __name__ == "__main__":
    main()
