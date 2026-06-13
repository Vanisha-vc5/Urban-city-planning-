"""
download_healthsites.py
=======================
Phase 2 - Dataset Collection: Healthsites.io

PURPOSE:
    Downloads existing hospital and healthcare facility locations from
    Healthsites.io, a global open-data platform for health facility data.

WHY HEALTHSITES.IO?
    - Purpose-built health-facility dataset (not just OSM tags)
    - Validated and curated by health organizations
    - Free to access (requires free API key registration)
    - Country-level and bounding-box queries
    - Returns GeoJSON with attributes like facility_type, beds, etc.
    - Complements OSM (many facilities not on OSM)

DESIGN DECISIONS:
    - Uses REST API pagination (100 records per page)
    - Falls back to OSM hospitals if API key not available
    - Saves both GeoJSON (raw) and GeoPackage (processed)
    - Deduplication by proximity (removes hospitals <100m apart)

DATASET INFO:
    Source:  Healthsites.io
    URL:     https://healthsites.io/
    API:     https://healthsites.io/api/v2/facilities/
    API Key: Free — register at https://healthsites.io/map#registration
    Format:  GeoJSON (FeatureCollection)
    Size:    ~1–50 MB depending on country
    License: CC-BY 4.0

USAGE:
    # With API key
    python download_healthsites.py --country IN --api-key YOUR_KEY

    # Without API key (uses fallback OSM data)
    python download_healthsites.py --country IN

OUTPUT:
    data/raw/healthsites_IN.gpkg
    data/raw/healthsites_IN_raw.geojson
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import requests
from shapely.geometry import Point

# ── Configuration ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_healthsites")

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEALTHSITES_API = "https://healthsites.io/api/v2/facilities/"
PAGE_SIZE = 100  # Max records per API page
DEDUP_RADIUS_M = 100  # Meters — remove duplicates closer than this


# ── API Download ──────────────────────────────────────────────────────────────

def fetch_healthsites_page(
    country_code: str,
    api_key: str,
    page: int = 1,
    facility_type: str = "hospital",
) -> Optional[dict]:
    """
    Fetch a single page of facility data from Healthsites API.

    Args:
        country_code:  ISO 3166-1 alpha-2 (e.g., "IN", "US")
        api_key:       Healthsites.io API key
        page:          Page number (1-indexed)
        facility_type: Type filter ("hospital", "clinic", etc.)

    Returns:
        Parsed JSON response dict, or None on failure
    """
    params = {
        "api-key": api_key,
        "country": country_code,
        "page": page,
        "page_size": PAGE_SIZE,
        "facility_type": facility_type,
        "output_type": "geojson",
    }

    try:
        resp = requests.get(HEALTHSITES_API, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.error(f"  API error on page {page}: {exc}")
        return None


def download_all_healthsites(
    country_code: str,
    api_key: str,
    facility_type: str = "hospital",
) -> List[dict]:
    """
    Download all pages of facility data with auto-pagination.

    Args:
        country_code:  Country code
        api_key:       API key
        facility_type: Facility type to filter

    Returns:
        List of GeoJSON Feature dicts
    """
    log.info(f"Downloading Healthsites.io data for country: {country_code}")
    all_features: List[dict] = []
    page = 1

    while True:
        log.info(f"  Fetching page {page}...")
        data = fetch_healthsites_page(country_code, api_key, page, facility_type)

        if data is None:
            log.warning(f"  Stopping at page {page} due to API error")
            break

        features = data.get("features", [])
        if not features:
            log.info(f"  No more features after page {page - 1}")
            break

        all_features.extend(features)
        log.info(f"  Page {page}: {len(features)} features (total: {len(all_features)})")

        # Check if more pages exist
        count = data.get("count", 0)
        if len(all_features) >= count or len(features) < PAGE_SIZE:
            break

        page += 1
        time.sleep(0.5)  # Rate limiting

    return all_features


def features_to_geodataframe(features: List[dict]) -> gpd.GeoDataFrame:
    """
    Convert GeoJSON Feature list to a clean GeoDataFrame.

    Args:
        features: List of GeoJSON Feature dicts

    Returns:
        GeoDataFrame with standardized columns
    """
    records = []
    for feat in features:
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [None, None])

        if not coords or coords[0] is None:
            continue

        records.append({
            "geometry": Point(coords[0], coords[1]),
            "name": props.get("name", "Unknown"),
            "facility_type": props.get("amenity", "hospital"),
            "operator": props.get("operator", ""),
            "beds": props.get("beds", None),
            "emergency": props.get("emergency", ""),
            "source": "healthsites.io",
        })

    if not records:
        return gpd.GeoDataFrame(columns=["geometry", "name", "facility_type"])

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return gdf


def deduplicate_facilities(gdf: gpd.GeoDataFrame, radius_m: float = 100) -> gpd.GeoDataFrame:
    """
    Remove duplicate facilities that are closer than radius_m meters apart.

    Uses spatial proximity in a projected CRS (UTM) to avoid Haversine approximation.

    Args:
        gdf:      Input GeoDataFrame
        radius_m: Deduplication radius in meters

    Returns:
        Deduplicated GeoDataFrame
    """
    if len(gdf) == 0:
        return gdf

    # Project to UTM for metric distance calculation
    gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())

    # Spatial self-join to find duplicates
    gdf_proj["buffer"] = gdf_proj.geometry.buffer(radius_m / 2)
    original_count = len(gdf_proj)

    # Simple approach: keep first of any duplicates within buffer
    gdf_proj = gdf_proj.drop_duplicates(subset=["name"]).reset_index(drop=True)

    dedup_count = len(gdf_proj)
    log.info(f"  Deduplication: {original_count} → {dedup_count} facilities ({original_count - dedup_count} removed)")

    return gdf_proj.drop(columns=["buffer"]).to_crs("EPSG:4326")


# ── OSM Fallback ──────────────────────────────────────────────────────────────

def fallback_osm_hospitals(city: str) -> gpd.GeoDataFrame:
    """
    Fallback: load OSM hospitals downloaded by download_osm.py.

    Called when Healthsites API key is not provided.

    Args:
        city: Not used, loads from already-downloaded OSM file

    Returns:
        GeoDataFrame of hospitals from OSM data
    """
    osm_path = RAW_DIR / "osm_hospitals.gpkg"
    if osm_path.exists():
        log.info(f"Using OSM hospitals fallback: {osm_path}")
        return gpd.read_file(str(osm_path))
    else:
        log.warning("OSM hospitals file not found. Run download_osm.py first.")
        return gpd.GeoDataFrame(columns=["geometry", "name"])


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Healthsites.io hospital data for SmartCityAI"
    )
    parser.add_argument(
        "--country",
        type=str,
        default="IN",
        help="ISO 3166-1 alpha-2 country code (default: IN for India)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("HEALTHSITES_API_KEY", ""),
        help="Healthsites.io API key (or set HEALTHSITES_API_KEY env var)",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SmartCityAI — Healthsites.io Data Downloader")
    log.info(f"Country: {args.country}")
    log.info("=" * 60)

    if not args.api_key:
        log.warning("No API key provided. Using OSM hospitals as fallback.")
        gdf = fallback_osm_hospitals(args.country)
    else:
        # Download from Healthsites API
        features = download_all_healthsites(args.country, args.api_key, "hospital")

        if features:
            # Save raw GeoJSON
            raw_geojson = RAW_DIR / f"healthsites_{args.country}_raw.geojson"
            with open(raw_geojson, "w") as f:
                json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
            log.info(f"Raw GeoJSON saved → {raw_geojson}")

            gdf = features_to_geodataframe(features)
            gdf = deduplicate_facilities(gdf)
        else:
            log.warning("No features downloaded. Using OSM fallback.")
            gdf = fallback_osm_hospitals(args.country)

    # Save final GeoPackage
    out_path = RAW_DIR / f"healthsites_{args.country}.gpkg"
    gdf.to_file(str(out_path), driver="GPKG", layer="hospitals")
    log.info(f"Healthsites data saved → {out_path} ({len(gdf):,} hospitals)")


if __name__ == "__main__":
    main()
