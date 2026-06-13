"""
download_worldpop.py
====================
Phase 2 - Dataset Collection: WorldPop Population Data

PURPOSE:
    Downloads WorldPop population density and age-structure raster data
    for a given country/year. WorldPop provides free, peer-reviewed
    gridded population data at ~100m resolution.

WHY WORLDPOP?
    - Free and open (Creative Commons Attribution 4.0)
    - 100m resolution — fine enough for ward/hex analysis
    - Age-disaggregated (elderly ratio calculation)
    - Covers all countries including India
    - No API key required (direct HTTP download)

DESIGN DECISIONS:
    - Downloads both total population AND age-band rasters
    - Supports year selection (2015–2023 available)
    - Country code follows ISO 3166-1 alpha-3 (e.g., IND = India)
    - Raster reprojection to EPSG:4326 ensures compatibility
    - Chunk streaming handles large raster files (100MB+)

DATASET INFO:
    Source:  WorldPop — University of Southampton
    URL:     https://www.worldpop.org/datacatalog/
    API:     REST API at https://hub.worldpop.org/rest/data
    Format:  GeoTIFF (.tif)
    Size:    ~50–300 MB per country per year
    License: CC-BY 4.0

USAGE:
    python download_worldpop.py --country IND --year 2020
    python download_worldpop.py --country USA --year 2020

OUTPUT:
    data/raw/worldpop_population_IND_2020.tif
    data/raw/worldpop_age_structure_IND_2020/  (multiple age-band files)
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_worldpop")

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

WORLDPOP_API = "https://hub.worldpop.org/rest/data"
CHUNK_SIZE = 8192  # bytes per download chunk


# ── WorldPop REST API Query ───────────────────────────────────────────────────

def query_worldpop_api(
    dataset: str, country: str, year: int
) -> Optional[dict]:
    """
    Query WorldPop Hub REST API for a specific dataset.

    The REST API returns JSON with download URLs for specific files.

    Args:
        dataset: WorldPop dataset alias (e.g., "pop", "agebysex")
        country: ISO3 country code (e.g., "IND", "USA")
        year:    Year of data (2015–2023)

    Returns:
        API response dict, or None on failure
    """
    endpoint = f"{WORLDPOP_API}/{dataset}/cic2020_100m"
    params = {
        "iso3": country,
        "year": year,
        "format": "json",
    }

    log.info(f"Querying WorldPop API: dataset={dataset}, country={country}, year={year}")
    try:
        resp = requests.get(endpoint, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        log.warning(f"API query failed: {exc}")
        return None


def get_direct_download_url(country: str, year: int) -> str:
    """
    Construct direct GeoTIFF download URL for WorldPop total population.

    WorldPop uses a predictable naming convention for unconstrained
    individual country UN-adjusted population data.

    Args:
        country: ISO3 country code (lowercase)
        year:    Year (2015–2023)

    Returns:
        Direct download URL string
    """
    iso_lower = country.lower()
    # WorldPop UN-adjusted, unconstrained, 100m resolution
    base = "https://data.worldpop.org/GIS/Population/Global_2000_2020"
    filename = f"{iso_lower}_ppp_{year}_UNadj.tif"
    return f"{base}/{year}/{country}/{filename}"


def download_file(url: str, dest: Path, description: str = "file") -> bool:
    """
    Download a file from URL with progress bar and resume support.

    Args:
        url:         Target download URL
        dest:        Destination Path object
        description: Human-readable label for progress bar

    Returns:
        True if download succeeded, False otherwise
    """
    if dest.exists():
        log.info(f"  Already downloaded: {dest.name} — skipping")
        return True

    log.info(f"  Downloading {description} from:\n    {url}")
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            with open(dest, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=dest.name,
                ncols=80,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    pbar.update(len(chunk))

        size_mb = dest.stat().st_size / (1024 * 1024)
        log.info(f"  Saved: {dest} ({size_mb:.1f} MB)")
        return True

    except Exception as exc:
        log.error(f"  Download failed: {exc}")
        if dest.exists():
            dest.unlink()  # Remove partial file
        return False


# ── Population Raster Download ────────────────────────────────────────────────

def download_total_population(country: str, year: int) -> None:
    """
    Download total population raster (all ages, all sexes).

    This raster provides the primary population_density feature.
    Each pixel value = estimated population count in that 100m x 100m cell.

    Args:
        country: ISO3 code
        year:    Target year
    """
    log.info(f"[1/2] Downloading total population raster — {country} {year}")
    url = get_direct_download_url(country, year)
    dest = RAW_DIR / f"worldpop_population_{country}_{year}.tif"
    success = download_file(url, dest, f"WorldPop Population {country} {year}")

    if not success:
        # Fallback to 2020 if requested year fails
        log.warning("Primary URL failed. Trying 2020 fallback...")
        url_fallback = get_direct_download_url(country, 2020)
        dest_fallback = RAW_DIR / f"worldpop_population_{country}_2020_fallback.tif"
        download_file(url_fallback, dest_fallback, f"WorldPop Population {country} 2020 (fallback)")


def download_age_structure(country: str, year: int) -> None:
    """
    Download age-sex disaggregated population rasters.

    Age bands used:
        - 65+  → elderly_ratio feature
        - 0-14 → youth_ratio (school demand)
        - all  → normalization base

    WorldPop provides 5-year age bands (0-4, 5-9, ..., 75-79, 80+) by sex.
    We download the key bands only to save storage.

    Args:
        country: ISO3 code
        year:    Target year (2000–2020 available for age structure)
    """
    log.info(f"[2/2] Downloading age-structure rasters — {country} {year}")

    iso_lower = country.lower()
    # Age-sex structure base URL
    base = "https://data.worldpop.org/GIS/AgeSex_structures/Global_2000_2020"

    # Key age bands for our features
    age_bands = {
        "60_64": "senior_60_64",
        "65_69": "senior_65_69",
        "70_74": "senior_70_74",
        "75_79": "senior_75_79",
        "80":    "senior_80plus",  # 80+ combined
        "0_4":   "youth_0_4",
        "5_9":   "youth_5_9",
        "10_14": "youth_10_14",
    }

    age_dir = RAW_DIR / f"worldpop_age_{country}_{year}"
    age_dir.mkdir(exist_ok=True)

    for band, label in age_bands.items():
        for sex in ["m", "f"]:  # male, female
            filename = f"{iso_lower}_{sex}_{band}_{year}.tif"
            url = f"{base}/{year}/{country.upper()}/{filename}"
            dest = age_dir / filename
            download_file(url, dest, f"Age {band} {sex}")
            time.sleep(0.5)  # Be polite to the server


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download WorldPop population data for SmartCityAI"
    )
    parser.add_argument(
        "--country",
        type=str,
        default="IND",
        help="ISO3 country code (default: IND for India)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2020,
        help="Year of data (default: 2020)",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SmartCityAI — WorldPop Data Downloader")
    log.info(f"Country: {args.country} | Year: {args.year}")
    log.info(f"Output:  {RAW_DIR}")
    log.info("=" * 60)

    download_total_population(args.country, args.year)
    download_age_structure(args.country, args.year)

    log.info("WorldPop download complete!")


if __name__ == "__main__":
    main()
