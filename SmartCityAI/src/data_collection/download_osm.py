"""
download_osm.py
===============
Phase 2 - Dataset Collection: OpenStreetMap

PURPOSE:
    Downloads roads, buildings, hospitals, schools, and fire stations
    for a given city using OSMnx (OpenStreetMap + NetworkX wrapper).

WHY OSMnx?
    - Completely free and open-source
    - Direct API to OpenStreetMap's Overpass API
    - Returns GeoDataFrames natively
    - No API key required
    - Production-quality data used globally

DESIGN DECISIONS:
    - City is configurable via CLI argument or config file
    - Each layer saved as separate GeoPackage (.gpkg) for compatibility
    - GeoPackage preferred over Shapefile (no 10-char column limit)
    - Retry logic handles transient Overpass API timeouts
    - Bounding box approach avoids polygon-based query complexity

USAGE:
    python download_osm.py --city "Mumbai, India"
    python download_osm.py --city "Delhi, India"

OUTPUT:
    data/raw/osm_roads.gpkg
    data/raw/osm_buildings.gpkg
    data/raw/osm_hospitals.gpkg
    data/raw/osm_schools.gpkg
    data/raw/osm_fire_stations.gpkg
    data/raw/osm_road_network.graphml

DATASET INFO:
    Source:  OpenStreetMap (© OpenStreetMap contributors)
    URL:     https://www.openstreetmap.org
    API:     OSMnx wraps the Overpass API (https://overpass-api.de)
    Format:  GeoJSON → GeoDataFrame → GeoPackage
    License: ODbL (Open Database License)
"""

import argparse
import logging
import time
from pathlib import Path

import geopandas as gpd
import osmnx as ox

# ── Configuration ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_osm")

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# OSMnx settings: cache requests locally for reproducibility
ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.timeout = 300           # 5-minute timeout per query
ox.settings.max_query_area_size = 50_000_000_000  # Large cities


# ── Layer Definitions ──────────────────────────────────────────────────────────
AMENITY_LAYERS = {
    "hospitals": {"amenity": ["hospital", "clinic", "health_centre"]},
    "schools":   {"amenity": ["school", "university", "college"]},
    "fire_stations": {"amenity": "fire_station"},
}

BUILDING_TAGS = {"building": True}  # All building footprints


# ── Download Functions ─────────────────────────────────────────────────────────

def download_road_network(city: str) -> None:
    """
    Download the drivable road network graph and save as GraphML + edges GeoPackage.

    State representation used in A* routing (Phase 6) comes from this graph.

    Args:
        city: City name string (e.g., "Mumbai, India")
    """
    log.info(f"Downloading road network for: {city}")
    try:
        G = ox.graph_from_place(city, network_type="drive", simplify=True)

        # Save graph for A* routing (Phase 6)
        graphml_path = RAW_DIR / "osm_road_network.graphml"
        ox.save_graphml(G, filepath=str(graphml_path))
        log.info(f"  Road network graph saved → {graphml_path}")

        # Also save edges as GeoDataFrame for spatial analysis
        _, edges = ox.graph_to_gdfs(G)
        edges_path = RAW_DIR / "osm_roads.gpkg"
        edges.to_file(str(edges_path), driver="GPKG", layer="roads")
        log.info(f"  Road edges saved → {edges_path} ({len(edges):,} edges)")

    except Exception as exc:
        log.error(f"Road network download failed: {exc}")
        raise


def download_amenity_layer(city: str, layer_name: str, tags: dict) -> None:
    """
    Download Points of Interest (POI) for a given amenity tag set.

    Args:
        city:       City place name
        layer_name: Output layer identifier (hospitals, schools, etc.)
        tags:       OSM tag filter dict
    """
    log.info(f"Downloading amenity layer: {layer_name}")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            gdf = ox.features_from_place(city, tags=tags)

            # Normalize geometry to Point (use centroid for polygons)
            gdf = gdf.copy()
            gdf["geometry"] = gdf["geometry"].centroid

            # Keep only useful columns
            keep_cols = [
                "geometry", "name", "amenity", "addr:street",
                "addr:city", "opening_hours", "operator"
            ]
            available = [c for c in keep_cols if c in gdf.columns]
            gdf = gdf[available].reset_index(drop=True)

            out_path = RAW_DIR / f"osm_{layer_name}.gpkg"
            gdf.to_file(str(out_path), driver="GPKG", layer=layer_name)
            log.info(f"  {layer_name}: {len(gdf):,} features → {out_path}")
            return

        except Exception as exc:
            log.warning(f"  Attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                time.sleep(5 * attempt)  # exponential backoff
            else:
                log.error(f"  All retries exhausted for {layer_name}")
                raise


def download_buildings(city: str) -> None:
    """
    Download building footprints.

    Used to estimate built-up density and identify vacant land for new sites.

    Args:
        city: City place name
    """
    log.info("Downloading building footprints...")
    try:
        gdf = ox.features_from_place(city, tags=BUILDING_TAGS)
        gdf = gdf[["geometry", "building"]].copy()
        gdf = gdf[gdf.geometry.is_valid]

        out_path = RAW_DIR / "osm_buildings.gpkg"
        gdf.to_file(str(out_path), driver="GPKG", layer="buildings")
        log.info(f"  Buildings: {len(gdf):,} footprints → {out_path}")

    except Exception as exc:
        log.error(f"Buildings download failed: {exc}")
        raise


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OpenStreetMap data for SmartCityAI"
    )
    parser.add_argument(
        "--city",
        type=str,
        default="Mumbai, India",
        help='Target city (e.g., "Mumbai, India")',
    )
    args = parser.parse_args()
    city = args.city

    log.info(f"{'='*60}")
    log.info(f"SmartCityAI — OSM Data Downloader")
    log.info(f"City: {city}")
    log.info(f"Output: {RAW_DIR}")
    log.info(f"{'='*60}")

    # 1. Road network (used for routing + accessibility features)
    download_road_network(city)

    # 2. Buildings (density estimation)
    download_buildings(city)

    # 3. Amenity layers (hospitals, schools, fire stations)
    for layer_name, tags in AMENITY_LAYERS.items():
        download_amenity_layer(city, layer_name, tags)

    log.info("All OSM downloads complete!")
    log.info(f"Files in {RAW_DIR}:")
    for f in sorted(RAW_DIR.glob("osm_*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        log.info(f"  {f.name:40s} {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
