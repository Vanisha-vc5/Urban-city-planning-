"""
download_ev_data.py
===================
Phase 2 - Dataset Collection: Open Charge Map (EV Charging Stations)

PURPOSE:
    Downloads existing EV (Electric Vehicle) charging station locations
    from Open Charge Map — the world's largest open-source EV charging
    point dataset.

WHY OPEN CHARGE MAP?
    - Completely free and open (requires free API key)
    - Global coverage, regularly updated by community
    - Rich metadata: charger type, connector type, power (kW), status
    - REST API with bounding-box and country-level queries
    - Actively maintained with verified locations

DESIGN DECISIONS:
    - Queries by country code + bounding box
    - Filters to operational chargers only (StatusType.IsOperational)
    - Extracts power level (kW) to distinguish fast/slow chargers
    - Calculates coverage_radius based on charging speed:
        * Level 1 (AC 120V): 1 km service radius
        * Level 2 (AC 240V): 3 km service radius
        * DC Fast Charging:  8 km service radius
    - Saves with all metadata for EV demand analysis

DATASET INFO:
    Source:  Open Charge Map
    URL:     https://openchargemap.org/
    API:     https://api.openchargemap.io/v3/poi/
    API Key: Free — register at https://openchargemap.org/site/developerinfo
    Format:  JSON → GeoDataFrame → GeoPackage
    Size:    ~5–50 MB depending on country
    License: Open Data Commons Open Database License (ODbL)

USAGE:
    python download_ev_data.py --country IN --api-key YOUR_KEY
    python download_ev_data.py --country US --bbox "25,-90,50,-60"

OUTPUT:
    data/raw/ev_stations_IN.gpkg
    data/raw/ev_stations_IN_raw.json
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import requests
from shapely.geometry import Point

# ── Configuration ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_ev_data")

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OCM_API = "https://api.openchargemap.io/v3/poi/"
MAX_RESULTS = 5000  # Max per API request (documented limit)

# Coverage radius by charging level (meters)
CHARGING_LEVEL_RADIUS = {
    1: 1000,   # Level 1: ~1 km
    2: 3000,   # Level 2: ~3 km
    3: 8000,   # DC Fast Charging: ~8 km (Tesla Supercharger class)
}


# ── API Functions ─────────────────────────────────────────────────────────────

def build_api_params(
    country_code: str,
    api_key: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    max_results: int = MAX_RESULTS,
) -> dict:
    """
    Build Open Charge Map API query parameters.

    Args:
        country_code: ISO 3166-1 alpha-2 code (e.g., "IN")
        api_key:      OCM API key
        bbox:         Optional (min_lat, min_lon, max_lat, max_lon) bounding box
        max_results:  Max POIs to fetch

    Returns:
        Parameters dict for requests.get()
    """
    params = {
        "key": api_key,
        "countrycode": country_code,
        "maxresults": max_results,
        "compact": False,      # Include full metadata
        "verbose": False,      # Exclude verbose lookup tables
        "output": "json",
        "statustypeid": 50,    # 50 = Operational only
    }

    if bbox:
        min_lat, min_lon, max_lat, max_lon = bbox
        params["boundingbox"] = f"({min_lat},{min_lon}),({max_lat},{max_lon})"

    return params


def fetch_ev_stations(
    country_code: str,
    api_key: str,
    bbox: Optional[Tuple] = None,
) -> List[dict]:
    """
    Fetch EV charging stations from Open Charge Map API.

    The API returns a flat JSON array of POI objects, each containing:
    - AddressInfo: lat, lon, address, town, state
    - Connections: list of connector specs (powerKW, level, connectorType)
    - StatusType: operational status

    Args:
        country_code: Country code
        api_key:      API key
        bbox:         Optional bounding box tuple

    Returns:
        List of station dicts
    """
    log.info(f"Fetching EV stations from Open Charge Map — country: {country_code}")
    params = build_api_params(country_code, api_key, bbox)

    try:
        resp = requests.get(OCM_API, params=params, timeout=60)
        resp.raise_for_status()
        stations = resp.json()
        log.info(f"  Fetched {len(stations):,} stations")
        return stations
    except Exception as exc:
        log.error(f"  API request failed: {exc}")
        return []


# ── Data Processing ───────────────────────────────────────────────────────────

def determine_charging_level(connections: List[dict]) -> int:
    """
    Determine highest charging level from connection list.

    Level 3 (DC Fast Charge) > Level 2 (AC 240V) > Level 1 (AC 120V)

    Args:
        connections: List of connection spec dicts from OCM

    Returns:
        Charging level integer (1, 2, or 3)
    """
    max_level = 1
    for conn in connections:
        level_info = conn.get("Level", {})
        level_id = level_info.get("ID", 1)
        if isinstance(level_id, int):
            max_level = max(max_level, level_id)
    return max_level


def determine_max_power_kw(connections: List[dict]) -> float:
    """
    Get maximum power (kW) across all connections at this station.

    Args:
        connections: List of connection spec dicts

    Returns:
        Maximum power in kW
    """
    powers = [
        c.get("PowerKW", 0) or 0
        for c in connections
        if isinstance(c.get("PowerKW"), (int, float))
    ]
    return max(powers) if powers else 0.0


def stations_to_geodataframe(stations: List[dict]) -> gpd.GeoDataFrame:
    """
    Convert raw OCM JSON to a clean GeoDataFrame with engineered attributes.

    Attributes extracted:
    - geometry: Point geometry from lat/lon
    - name: Station name
    - charging_level: 1, 2, or 3
    - max_power_kw: Maximum power output
    - coverage_radius_m: Estimated service radius
    - num_connections: Total charge points
    - is_operational: Boolean
    - town/state: Location info

    Args:
        stations: List of raw OCM station dicts

    Returns:
        Clean GeoDataFrame
    """
    records = []

    for station in stations:
        addr = station.get("AddressInfo", {})
        connections = station.get("Connections", []) or []

        lat = addr.get("Latitude")
        lon = addr.get("Longitude")

        if lat is None or lon is None:
            continue

        charging_level = determine_charging_level(connections)
        max_power_kw = determine_max_power_kw(connections)
        coverage_radius = CHARGING_LEVEL_RADIUS.get(charging_level, 1000)

        records.append({
            "geometry": Point(lon, lat),
            "name": addr.get("Title", "Unknown Charging Station"),
            "town": addr.get("Town", ""),
            "state": addr.get("StateOrProvince", ""),
            "country": addr.get("Country", {}).get("ISOCode", ""),
            "charging_level": charging_level,
            "max_power_kw": max_power_kw,
            "coverage_radius_m": coverage_radius,
            "num_connections": len(connections),
            "is_operational": True,  # Filtered to operational in API query
            "source": "open_charge_map",
        })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    log.info(f"  Converted {len(gdf):,} stations to GeoDataFrame")

    # Log charging level distribution
    if len(gdf) > 0:
        level_counts = gdf["charging_level"].value_counts().to_dict()
        log.info(f"  Charging levels: {level_counts}")

    return gdf


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download EV charging station data for SmartCityAI"
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
        default=os.environ.get("OCM_API_KEY", ""),
        help="Open Charge Map API key (or set OCM_API_KEY env var)",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help='Bounding box "min_lat,min_lon,max_lat,max_lon" (e.g., "18.5,72.7,19.2,73.0" for Mumbai)',
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SmartCityAI — Open Charge Map EV Data Downloader")
    log.info(f"Country: {args.country}")
    log.info("=" * 60)

    # Parse bounding box
    bbox = None
    if args.bbox:
        parts = [float(x.strip()) for x in args.bbox.split(",")]
        bbox = tuple(parts[:4])
        log.info(f"Bounding box: {bbox}")

    if not args.api_key:
        log.error("No API key provided. Get a free key at https://openchargemap.org/site/developerinfo")
        log.info("Generating synthetic EV station data for demonstration...")
        # Generate minimal synthetic data so pipeline doesn't break
        import numpy as np
        np.random.seed(42)
        n = 50
        records = [
            {
                "geometry": Point(72.8 + np.random.randn() * 0.1, 19.0 + np.random.randn() * 0.1),
                "name": f"Demo EV Station {i+1}",
                "charging_level": np.random.choice([1, 2, 3]),
                "max_power_kw": float(np.random.choice([7.4, 22.0, 50.0, 150.0])),
                "coverage_radius_m": 3000,
                "num_connections": int(np.random.randint(2, 8)),
                "is_operational": True,
                "source": "synthetic_demo",
            }
            for i in range(n)
        ]
        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    else:
        stations = fetch_ev_stations(args.country, args.api_key, bbox)

        # Save raw JSON
        raw_path = RAW_DIR / f"ev_stations_{args.country}_raw.json"
        with open(raw_path, "w") as f:
            json.dump(stations, f, indent=2)
        log.info(f"Raw JSON saved → {raw_path}")

        gdf = stations_to_geodataframe(stations)

    # Save GeoPackage
    out_path = RAW_DIR / f"ev_stations_{args.country}.gpkg"
    gdf.to_file(str(out_path), driver="GPKG", layer="ev_stations")
    log.info(f"EV stations saved → {out_path} ({len(gdf):,} stations)")


if __name__ == "__main__":
    main()
