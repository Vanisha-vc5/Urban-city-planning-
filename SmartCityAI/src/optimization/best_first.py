"""
best_first.py
=============
Phase 6 - Classical AI: Best First Search for Site Ranking

PURPOSE:
    Uses Best First Search to rank all candidate sites for new infrastructure
    by priority score, exploring the most promising candidates first.

ALGORITHM: Best First Search (Greedy)
    - State space: All H3 zones × infrastructure types
    - Start state: (any zone, budget=B)
    - Goal state: Exhaustively ranked list of top-K sites
    - Frontier: Priority queue (min-heap) ordered by heuristic value
    - Heuristic h(n): priority_composite_100 + coverage_gap × population_density

WHY BEST FIRST SEARCH?
    - Pure greedy exploration of most-promising candidates
    - O(b × n log n) where b=branching factor — efficient for 1000s of zones
    - No backtracking needed (ranking problem, not path problem)
    - Naturally produces a ranked list — perfect for "top-10 sites" query
    - Demonstrates classical AI in an ML-heavy project

STATE REPRESENTATION:
    ZoneState = namedtuple('ZoneState', [
        'h3_id',          # Zone identifier
        'infra_type',     # "hospital" | "school" | "ev_station" | "fire_station"
        'priority_score', # h(n): heuristic priority
        'coverage_gap',   # Current gap this zone has
        'population',     # Zone population
        'lat', 'lon'      # Location for map display
    ])

HEURISTIC FUNCTION:
    h(zone, infra_type) =
        w1 × coverage_gap
      + w2 × population_density_norm
      + w3 × elderly_ratio           (hospitals only)
      + w4 × equity_adjusted_priority

    Weights per infrastructure type:
    Hospital:    w1=0.40, w2=0.30, w3=0.20, w4=0.10
    School:      w1=0.30, w2=0.35, w3=0.00 (youth instead), w4=0.15
    EV Station:  w1=0.25, w2=0.40, w3=0.00, w4=0.15
    Fire Station:w1=0.35, w2=0.25, w3=0.05, w4=0.20, +emergency_time×0.15

COMPLEXITY:
    Time:  O(n × log n) — heap operations per zone
    Space: O(n) — all zones stored in heap

USAGE:
    python best_first.py --infra hospital --top 20
    python best_first.py --infra school --budget 50000000

OUTPUT:
    data/processed/ranked_sites_{infra_type}.csv
"""

import heapq
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Configuration ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("best_first")

BASE_DIR      = Path(__file__).resolve().parents[3]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Infrastructure type configurations
INFRA_CONFIGS: Dict[str, Dict] = {
    "hospital": {
        "weights": {
            "coverage_gap":             0.40,
            "population_density_norm":  0.25,
            "elderly_ratio":            0.20,
            "equity_adjusted_priority": 0.15,
        },
        "cost_per_unit": 5_00_00_000,  # ₹5 Crore
        "unit_label": "hospital bed",
    },
    "school": {
        "weights": {
            "coverage_gap":             0.30,
            "population_density_norm":  0.30,
            "youth_ratio":              0.25,
            "road_accessibility_index": 0.15,
        },
        "cost_per_unit": 2_00_00_000,  # ₹2 Crore
        "unit_label": "school",
    },
    "ev_station": {
        "weights": {
            "coverage_gap":             0.25,
            "population_density_norm":  0.35,
            "road_accessibility_index": 0.25,
            "traffic_density":          0.15,
        },
        "cost_per_unit": 50_00_000,    # ₹50 Lakh
        "unit_label": "charging point",
    },
    "fire_station": {
        "weights": {
            "coverage_gap":                 0.30,
            "population_density_norm":      0.20,
            "emergency_response_time_min":  0.30,
            "equity_adjusted_priority":     0.20,
        },
        "cost_per_unit": 3_00_00_000,  # ₹3 Crore
        "unit_label": "fire station",
    },
}


# ── Zone State ────────────────────────────────────────────────────────────────

@dataclass(order=True)
class ZoneState:
    """
    Represents a candidate site state in Best First Search.

    sort_index is the NEGATIVE priority score (min-heap = highest priority first).
    Other fields are not used for comparison (compare_ignore).

    Attributes:
        sort_index:        -priority_score (for min-heap ordering)
        h3_id:             H3 zone identifier
        infra_type:        Infrastructure type being considered
        priority_score:    Computed heuristic score [0, 100]
        coverage_gap:      Infrastructure gap value [0, 1]
        population_total:  Zone population
        lat:               Zone centroid latitude
        lon:               Zone centroid longitude
        features:          Dict of feature values for explanation
    """
    sort_index:    float
    h3_id:         str       = field(compare=False)
    infra_type:    str       = field(compare=False)
    priority_score: float    = field(compare=False)
    coverage_gap:  float     = field(compare=False)
    population_total: float  = field(compare=False)
    lat:           float     = field(compare=False)
    lon:           float     = field(compare=False)
    features:      dict      = field(compare=False, default_factory=dict)


# ── Heuristic Function ────────────────────────────────────────────────────────

def compute_heuristic(
    row: pd.Series,
    infra_type: str,
    weights: Dict[str, float],
    normalized_df: pd.DataFrame,
    idx: int,
) -> float:
    """
    Compute Best First Search heuristic for a zone.

    The heuristic h(n) guides which zones to explore first.
    Unlike A*, Best First Search uses ONLY h(n) — no path cost g(n).

    Formula:
        h(n) = Σ weight_i × feature_i_normalized × 100

    All features normalized [0, 1] before weighting.
    Final score in [0, 100].

    Args:
        row:           Raw zone data row
        infra_type:    Infrastructure type
        weights:       Feature weight dict
        normalized_df: DataFrame with normalized features (all [0,1])
        idx:           Row index in normalized_df

    Returns:
        Heuristic value (higher = higher priority)
    """
    score = 0.0
    total_weight = 0.0

    for feature, weight in weights.items():
        # Map feature names to column names
        col_map = {
            "coverage_gap":                 "coverage_gap",
            "population_density_norm":      "population_density",  # Will be normalized
            "elderly_ratio":               "elderly_ratio",
            "youth_ratio":                 "youth_ratio",
            "equity_adjusted_priority":    "equity_adjusted_priority",
            "road_accessibility_index":    "road_accessibility_index",
            "traffic_density":             "traffic_density",
            "emergency_response_time_min": "emergency_response_time_min",
        }

        col = col_map.get(feature, feature)

        if col in normalized_df.columns:
            val = float(normalized_df.iloc[idx][col])
            score += weight * val * 100
            total_weight += weight
        elif col in row.index:
            val = float(row[col])
            score += weight * val * 100
            total_weight += weight

    if total_weight > 0:
        score = score / total_weight * sum(weights.values())

    return round(score, 2)


# ── Best First Search ─────────────────────────────────────────────────────────

class BestFirstSearch:
    """
    Best First Search algorithm for infrastructure site ranking.

    Maintains a priority queue (min-heap) of ZoneStates.
    Explores zones in order of decreasing priority score.

    Exploration strategy:
        1. Push all candidate zones to heap with their heuristic scores
        2. Pop top-priority zone
        3. Add to ranked list
        4. Continue until budget exhausted or all zones explored

    Complexity Analysis:
        Time:  O(n log n) — initial heap build + n × log n extractions
        Space: O(n) — stores all n zone states in heap
        n = number of candidate zones (typically 200–5000 for a city)

    Usage:
        bfs = BestFirstSearch(df, infra_type="hospital")
        ranked = bfs.run(top_k=20, budget=500_000_000)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        infra_type: str = "hospital",
    ):
        """
        Initialize Best First Search.

        Args:
            df:          Zone features DataFrame
            infra_type:  Target infrastructure type
        """
        self.df = df.copy()
        self.infra_type = infra_type
        self.config = INFRA_CONFIGS.get(infra_type, INFRA_CONFIGS["hospital"])
        self.weights = self.config["weights"]
        self.heap: List[ZoneState] = []
        self._normalized_df = self._normalize_features()

        log.info(f"BestFirstSearch initialized: infra_type={infra_type}, zones={len(df):,}")
        log.info(f"Weights: {self.weights}")

    def _normalize_features(self) -> pd.DataFrame:
        """
        Normalize all numeric features to [0, 1].

        Returns:
            Normalized DataFrame
        """
        scaler = MinMaxScaler()
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()

        # Don't normalize already-normalized features
        already_norm = [c for c in numeric_cols if "_norm" in c or "ratio" in c]
        to_normalize = [c for c in numeric_cols if c not in already_norm]

        norm_df = self.df.copy()
        if to_normalize:
            norm_df[to_normalize] = scaler.fit_transform(self.df[to_normalize].fillna(0))

        return norm_df

    def build_heap(self) -> None:
        """
        Build priority heap from all candidate zones.

        Each zone becomes a ZoneState with its computed priority.
        Uses Python's heapq (min-heap) — negate score for max-heap behavior.

        Time: O(n log n)
        """
        log.info("Building priority heap...")
        self.heap = []

        # Get zone centroids if available
        zones_gdf = None
        gpkg_path = PROCESSED_DIR / "zones.gpkg"
        if gpkg_path.exists():
            try:
                import geopandas as gpd
                zones_gdf = gpd.read_file(str(gpkg_path))
                centroids = zones_gdf.set_index("h3_id").geometry.centroid
            except Exception:
                zones_gdf = None

        for idx, row in self.df.iterrows():
            h3_id = str(row.get("h3_id", f"zone_{idx}"))

            # Compute heuristic
            score = compute_heuristic(row, self.infra_type, self.weights, self._normalized_df, idx)

            # Get coordinates
            lat, lon = 19.0, 72.85  # Default (Mumbai)
            if zones_gdf is not None and h3_id in centroids.index:
                lat = centroids[h3_id].y
                lon = centroids[h3_id].x

            # Build feature explanation dict
            features_dict = {
                col: round(float(row[col]), 4)
                for col in self.weights.keys()
                if col in row.index or col.replace("_norm", "") in row.index
            }

            state = ZoneState(
                sort_index   = -score,  # Negate for max-heap (min-heap pops smallest first)
                h3_id        = h3_id,
                infra_type   = self.infra_type,
                priority_score = score,
                coverage_gap = float(row.get("coverage_gap", 0)),
                population_total = float(row.get("population_total", 0)),
                lat          = lat,
                lon          = lon,
                features     = features_dict,
            )
            heapq.heappush(self.heap, state)

        log.info(f"  Heap built with {len(self.heap):,} candidate zones")

    def run(
        self,
        top_k: int = 20,
        budget: Optional[float] = None,
    ) -> List[ZoneState]:
        """
        Run Best First Search to get top-k ranked sites.

        Args:
            top_k:  Number of top sites to return
            budget: Optional budget constraint (stops when cost > budget)

        Returns:
            Ordered list of ZoneStates (highest priority first)
        """
        if not self.heap:
            self.build_heap()

        log.info(f"Running Best First Search: top_k={top_k}, budget={budget}")
        ranked = []
        total_cost = 0
        cost_per_unit = self.config["cost_per_unit"]

        # Pop from heap in priority order
        heap_copy = list(self.heap)  # Non-destructive search
        heapq.heapify(heap_copy)

        while heap_copy and len(ranked) < top_k:
            state = heapq.heappop(heap_copy)

            # Budget check
            if budget and (total_cost + cost_per_unit > budget):
                log.info(f"  Budget exhausted after {len(ranked)} sites")
                break

            ranked.append(state)
            total_cost += cost_per_unit
            log.debug(f"  Rank {len(ranked):2d}: {state.h3_id} | Score={state.priority_score:.1f} | Gap={state.coverage_gap:.3f}")

        log.info(f"Search complete: {len(ranked)} sites ranked")
        log.info(f"Total estimated cost: ₹{total_cost:,.0f}")

        return ranked

    def get_results_dataframe(self, ranked: List[ZoneState]) -> pd.DataFrame:
        """
        Convert ranked ZoneState list to a results DataFrame.

        Args:
            ranked: Output of run()

        Returns:
            DataFrame with rank, zone info, and scores
        """
        records = []
        cost_per_unit = self.config["cost_per_unit"]

        for rank, state in enumerate(ranked, 1):
            record = {
                "rank":            rank,
                "h3_id":           state.h3_id,
                "infra_type":      state.infra_type,
                "priority_score":  state.priority_score,
                "coverage_gap":    state.coverage_gap,
                "population_total": state.population_total,
                "lat":             state.lat,
                "lon":             state.lon,
                "estimated_cost":  cost_per_unit,
                "cumulative_cost": rank * cost_per_unit,
            }
            record.update({f"feat_{k}": v for k, v in state.features.items()})
            records.append(record)

        return pd.DataFrame(records)


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Best First Search site ranking for SmartCityAI")
    parser.add_argument("--infra",  default="hospital",
                        choices=list(INFRA_CONFIGS.keys()) + ["all"])
    parser.add_argument("--top",    type=int, default=20, help="Top K sites to return")
    parser.add_argument("--budget", type=float, default=None, help="Budget constraint (₹)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SmartCityAI — Best First Search Site Ranking")
    log.info("=" * 60)

    # Load data
    for path in [PROCESSED_DIR / "ml_features.csv", PROCESSED_DIR / "zone_features.csv"]:
        if path.exists():
            df = pd.read_csv(path)
            break
    else:
        log.error("No feature CSV found.")
        sys.exit(1)

    log.info(f"Loaded {len(df):,} zones")

    # Merge predictions if available
    for pred_file in ["svm_predictions.csv", "dt_predictions.csv"]:
        pred_path = PROCESSED_DIR / pred_file
        if pred_path.exists():
            preds = pd.read_csv(pred_path)
            df = df.merge(preds, on="h3_id", how="left")

    # Run search
    infra_types = list(INFRA_CONFIGS.keys()) if args.infra == "all" else [args.infra]

    for infra_type in infra_types:
        log.info(f"\n{'─'*50}")
        log.info(f"Ranking sites for: {infra_type.upper()}")
        log.info(f"{'─'*50}")

        bfs = BestFirstSearch(df, infra_type)
        ranked = bfs.run(top_k=args.top, budget=args.budget)
        results_df = bfs.get_results_dataframe(ranked)

        out_path = PROCESSED_DIR / f"ranked_sites_{infra_type}.csv"
        results_df.to_csv(out_path, index=False)
        log.info(f"Ranked sites saved → {out_path}")

        # Print top 5
        log.info(f"\nTop 5 {infra_type} sites:")
        for _, row in results_df.head(5).iterrows():
            log.info(f"  #{row['rank']:2.0f} | Zone {row['h3_id']} | Score={row['priority_score']:.1f} | Gap={row['coverage_gap']:.3f}")

    log.info("\nBest First Search complete!")


if __name__ == "__main__":
    main()
