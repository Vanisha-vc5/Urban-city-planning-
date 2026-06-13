"""
download_census.py
==================
Phase 2 - Dataset Collection: Census / Demographic Data

PURPOSE:
    Downloads demographic data including population, age structure,
    income levels, and household statistics for urban planning analysis.

    Supports two primary sources:
    1. India Census (census.gov.in / data.gov.in) — for Indian cities
    2. US Census Bureau API — for US cities

WHY CENSUS DATA?
    - Authoritative demographic baseline (government source)
    - Ward/block level granularity for Indian cities
    - Age brackets → elderly_ratio (hospital demand proxy)
    - Per-capita income → income_bracket_norm (equity analysis)
    - Household density → combined with WorldPop for accuracy

DESIGN DECISIONS:
    - Generates realistic synthetic data when API unavailable
      (common for Indian census — not all data is API-accessible)
    - Synthetic data uses statistical distributions matching real cities
    - Always saves both raw CSV and a processed version
    - Income normalized to [0, 1] for model compatibility

DATASET INFO:
    Source A: Census of India 2011/2021 — https://censusindia.gov.in
    Source B: data.gov.in Open Government Data — https://data.gov.in
    Source C: US Census Bureau API — https://api.census.gov
    Format:   CSV
    License:  Government Open Data License (India), CC0 (US Census)

USAGE:
    python download_census.py --country IN --city Mumbai
    python download_census.py --country US --state CA --year 2020

OUTPUT:
    data/raw/census_IN_Mumbai.csv
    data/raw/census_ward_demographics.csv
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ── Configuration ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_census")

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# US Census API base
US_CENSUS_API = "https://api.census.gov/data"

# India data.gov.in resources (district-level population)
INDIA_DATGOV_BASE = "https://data.gov.in/resource"


# ── India Census (data.gov.in) ────────────────────────────────────────────────

def download_india_census(city: str = "Mumbai") -> pd.DataFrame:
    """
    Download or generate India census demographic data.

    India's census API has limited programmatic access.
    We try data.gov.in API; fall back to realistic synthetic data.

    Synthetic data parameters derived from published Mumbai Census 2011:
    - Population: 12.4M (Greater Mumbai)
    - Elderly (60+): ~8.5%
    - Youth (0-14): ~22%
    - Working age (15-59): ~69.5%
    - Per capita income: ₹1,70,000/year (2011)
    - Literacy rate: 88%

    Args:
        city: City name for labeling

    Returns:
        DataFrame with demographic columns per ward
    """
    log.info(f"Downloading India census data for: {city}")

    # Try data.gov.in API
    resource_id = "6176ee09-3d56-4a3b-8115-21841576d7ac"  # Population by district
    api_key = os.environ.get("DATAGOV_IN_API_KEY", "")
    n_wards = 227  # Mumbai has 227 wards (K-North, K-West, etc.)

    if api_key:
        try:
            url = f"{INDIA_DATGOV_BASE}/{resource_id}"
            params = {
                "api-key": api_key,
                "limit": 1000,
                "filters[District]": city,
            }
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                records = data.get("records", [])
                if records:
                    df = pd.DataFrame(records)
                    log.info(f"  Fetched {len(df)} records from data.gov.in")
                    return df
        except Exception as exc:
            log.warning(f"  data.gov.in API failed: {exc}")

    # Generate realistic synthetic data
    log.info(f"  Generating synthetic census data for {city} ({n_wards} wards)...")
    return _generate_synthetic_census(city, n_wards)


def _generate_synthetic_census(city: str, n_wards: int) -> pd.DataFrame:
    """
    Generate statistically realistic synthetic census data.

    Based on published urban demographics for Indian metros:
    - Mumbai: dense inner wards (50,000+ pop), sparse outer suburbs
    - Age structure follows standard urban Indian demographic pyramid
    - Income follows log-normal distribution (inequality = Gini ~0.37)

    Args:
        city:    City name
        n_wards: Number of administrative wards

    Returns:
        Synthetic ward-level DataFrame
    """
    rng = np.random.default_rng(42)  # Reproducible

    # Ward IDs using BBMP/MCGM numbering convention
    ward_ids = [f"{city[:3].upper()}-W{str(i+1).zfill(3)}" for i in range(n_wards)]

    # Population: Log-normal (dense core, sparser periphery)
    pop_total = rng.lognormal(mean=10.2, sigma=0.6, size=n_wards).astype(int)
    pop_total = np.clip(pop_total, 5000, 150000)

    # Age groups (proportions sum to 1.0)
    # Based on Census of India 2011 age structure
    p_elderly = rng.beta(a=2.5, b=28, size=n_wards)        # ~8% ± 3%
    p_youth   = rng.beta(a=5.0, b=18, size=n_wards)        # ~22% ± 4%
    p_working = 1.0 - p_elderly - p_youth                  # remainder
    p_working = np.clip(p_working, 0.5, 0.8)

    # Renormalize
    p_sum = p_elderly + p_youth + p_working
    p_elderly /= p_sum
    p_youth   /= p_sum
    p_working /= p_sum

    # Per-capita income: Log-normal (₹ per year)
    # Range: ₹60,000 (slum) to ₹8,00,000 (South Mumbai premium)
    income_pc = rng.lognormal(mean=11.8, sigma=0.7, size=n_wards).astype(int)
    income_pc = np.clip(income_pc, 60_000, 800_000)

    # Normalize income to [0, 1] for ML features
    income_norm = (income_pc - income_pc.min()) / (income_pc.max() - income_pc.min())

    # Literacy rate
    literacy = rng.beta(a=18, b=4, size=n_wards)           # ~82% ± 5%
    literacy = np.clip(literacy, 0.65, 0.98)

    # Household size (persons per household)
    household_size = rng.normal(loc=4.2, scale=0.8, size=n_wards)
    household_size = np.clip(household_size, 2.5, 8.0)

    df = pd.DataFrame({
        "ward_id":            ward_ids,
        "city":               city,
        "population_total":   pop_total,
        "pop_elderly_60plus": (pop_total * p_elderly).astype(int),
        "pop_youth_0_14":     (pop_total * p_youth).astype(int),
        "pop_working_15_59":  (pop_total * p_working).astype(int),
        "elderly_ratio":      p_elderly.round(4),
        "youth_ratio":        p_youth.round(4),
        "working_ratio":      p_working.round(4),
        "income_per_capita":  income_pc,
        "income_bracket_norm": income_norm.round(4),
        "literacy_rate":      literacy.round(4),
        "avg_household_size": household_size.round(2),
        "data_source":        "synthetic_census_2011_based",
    })

    log.info(f"  Generated {len(df)} ward records")
    log.info(f"  Pop range: {pop_total.min():,} – {pop_total.max():,}")
    log.info(f"  Income range: ₹{income_pc.min():,} – ₹{income_pc.max():,}")

    return df


# ── US Census Bureau API ──────────────────────────────────────────────────────

def download_us_census(
    state_fips: str = "06",
    year: int = 2020,
    api_key: str = "",
) -> pd.DataFrame:
    """
    Download US Census Bureau ACS 5-year estimates.

    Variables:
    - B01003_001E: Total population
    - B01002_001E: Median age
    - B19013_001E: Median household income
    - B17001_002E: Population below poverty line
    - B01001_020E...B01001_025E: Female population 65+

    Args:
        state_fips: State FIPS code (e.g., "06" = California)
        year:       ACS 5-year estimate year
        api_key:    US Census API key (optional, limits to 500 req/day without)

    Returns:
        DataFrame with census variables per tract
    """
    log.info(f"Downloading US Census ACS data — State FIPS: {state_fips}, Year: {year}")

    variables = ",".join([
        "NAME",
        "B01003_001E",   # Total population
        "B01002_001E",   # Median age
        "B19013_001E",   # Median household income
        "B17001_002E",   # Population in poverty
        "B25001_001E",   # Total housing units
    ])

    url = f"{US_CENSUS_API}/{year}/acs/acs5"
    params = {
        "get": variables,
        "for": "tract:*",
        "in": f"state:{state_fips}",
    }
    if api_key:
        params["key"] = api_key

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()

        headers = rows[0]
        data = rows[1:]
        df = pd.DataFrame(data, columns=headers)

        # Type conversion
        numeric_cols = [c for c in headers if c not in ["NAME", "state", "county", "tract"]]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.rename(columns={
            "B01003_001E": "population_total",
            "B01002_001E": "median_age",
            "B19013_001E": "income_per_capita",
            "B17001_002E": "poverty_population",
        }, inplace=True)

        # Derive elderly ratio (approximation using median age)
        df["elderly_ratio"] = (df["median_age"] / 100.0).clip(0, 1)

        # Normalize income
        df["income_bracket_norm"] = (
            (df["income_per_capita"] - df["income_per_capita"].min())
            / (df["income_per_capita"].max() - df["income_per_capita"].min())
        )

        log.info(f"  US Census: {len(df):,} tracts downloaded")
        return df

    except Exception as exc:
        log.error(f"  US Census API failed: {exc}")
        return pd.DataFrame()


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download census demographic data for SmartCityAI"
    )
    parser.add_argument("--country", default="IN", choices=["IN", "US"], help="Country code")
    parser.add_argument("--city",    default="Mumbai", help="City name (for India)")
    parser.add_argument("--state",   default="06", help="US State FIPS code")
    parser.add_argument("--year",    type=int, default=2020, help="Census year")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("US_CENSUS_API_KEY", ""),
        help="US Census API key",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SmartCityAI — Census Data Downloader")
    log.info("=" * 60)

    if args.country == "IN":
        df = download_india_census(args.city)
        out_path = RAW_DIR / f"census_IN_{args.city}.csv"
    else:
        df = download_us_census(args.state, args.year, args.api_key)
        out_path = RAW_DIR / f"census_US_state{args.state}_{args.year}.csv"

    if not df.empty:
        df.to_csv(out_path, index=False)
        log.info(f"Census data saved → {out_path} ({len(df):,} rows, {len(df.columns)} columns)")
        log.info(f"Columns: {list(df.columns)}")
    else:
        log.error("No data to save!")


if __name__ == "__main__":
    main()
