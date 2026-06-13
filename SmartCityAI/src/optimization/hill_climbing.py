"""
hill_climbing.py
================
Phase 6 - Classical AI: Hill Climbing Optimization for Site Placement

PURPOSE:
    Uses Hill Climbing to optimize placement of multiple infrastructure
    units under a budget constraint — finding the best combination of sites
    that maximizes total coverage gain.

    Answers: "Given ₹50 Crore and 5 hospital slots — which 5 zones maximize
             total population coverage gain?"

ALGORITHM: Stochastic Hill Climbing with Restart
    - State: Set of selected sites (binary vector over all zones)
    - Initial state: Greedy initialization (pick top-K from BFS)
    - Neighbor generation: Swap one selected zone for a non-selected zone
    - Objective: Maximize coverage_gain(selected_set) - λ × budget_penalty
    - Termination: No improving neighbor found OR max_iterations exceeded
    - Restart: Random restart to escape local optima

WHY HILL CLIMBING vs GENETIC ALGORITHM?
    - Simpler to explain in interviews (fundamental AI concept)
    - Sufficient for site counts < 20 (manageable local optima)
    - Faster than GA for small search spaces
    - With restarts: competitive with SA for urban planning scale
    - Perfect showcase of classical AI search

PROBLEM FORMULATION:
    Variables:
        x_i ∈ {0, 1} for each zone i (0=not selected, 1=selected)
    Objective:
        maximize Σ_i x_i × marginal_coverage_gain_i
    Constraints:
        Σ_i x_i × cost_per_site ≤ budget
        Σ_i x_i ≤ max_sites

    This is a VARIANT of the 0-1 Knapsack problem (NP-hard),
    but Hill Climbing finds good approximate solutions quickly.

STATE EVALUATION:
    coverage_gain(S) = Population newly covered if all sites in S are built
    marginal gain: accounts for overlapping service areas (double-counted if sites overlap)

COMPLEXITY:
    Time:  O(max_iter × n) per restart, O(n_restarts × max_iter × n)
    Space: O(n) — only current state + best state stored
    n = number of candidate zones (200–5000)

USAGE:
    python hill_climbing.py --infra hospital --budget 500000000 --sites 5
    python hill_climbing.py --infra school --budget 200000000 --sites 8 --restarts 10

OUTPUT:
    data/processed/optimized_sites_{infra_type}.csv
    models/hill_climbing_results.json
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hill_climbing")

BASE_DIR      = Path(__file__).resolve().parents[3]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

INFRA_COSTS = {
    "hospital":     5_00_00_000,   # ₹5 Crore per facility
    "school":       2_00_00_000,   # ₹2 Crore
    "ev_station":     50_00_000,   # ₹50 Lakh
    "fire_station": 3_00_00_000,   # ₹3 Crore
}

SERVICE_RADII = {
    "hospital":     5.0,   # km
    "school":       2.0,
    "ev_station":   3.0,
    "fire_station": 3.0,
}


# ── Objective Function ────────────────────────────────────────────────────────

def compute_coverage_gain(
    selected_indices: List[int],
    df: pd.DataFrame,
    infra_type: str,
) -> float:
    """
    Compute total coverage gain for a selected set of sites.

    Handles overlapping service areas:
    - Zones within radius of MULTIPLE selected sites are counted only ONCE
    - This is the "union area" computation (set coverage problem)

    Simplified approximation:
    - For each candidate zone not currently covered → check if ANY selected site covers it
    - Compute weighted coverage: Σ (pop_i × is_newly_covered_i) / total_uncovered_pop

    Args:
        selected_indices: Indices in df of selected sites
        df:               Zone features DataFrame
        infra_type:       Infrastructure type

    Returns:
        Coverage gain score (higher = better selection)
    """
    if not selected_indices:
        return 0.0

    selected = df.iloc[selected_indices]
    radius_km = SERVICE_RADII.get(infra_type, 5.0)
    EARTH_R = 6371.0

    # Zones not currently covered (coverage_gap > 0)
    gap_col = f"{infra_type}_coverage_gap" if f"{infra_type}_coverage_gap" in df.columns else "coverage_gap"
    uncovered_mask = df[gap_col] > 0.3 if gap_col in df.columns else pd.Series([True] * len(df))
    uncovered_df = df[uncovered_mask]

    if len(uncovered_df) == 0:
        return 0.0

    # Get selected site centroids (use lat/lon if available, else synthetic)
    sel_lats = selected.get("lat", pd.Series(np.full(len(selected), 19.0))).values
    sel_lons = selected.get("lon", pd.Series(np.full(len(selected), 72.85))).values

    total_gain = 0.0
    total_pop = uncovered_df.get("population_total", pd.Series(np.ones(len(uncovered_df)) * 10000)).sum()

    if total_pop == 0:
        return 0.0

    for _, zone in uncovered_df.iterrows():
        zone_lat = zone.get("lat", 19.0)
        zone_lon = zone.get("lon", 72.85)
        zone_pop = zone.get("population_total", 10000)

        # Check if any selected site covers this zone
        covered_by_any = False
        for s_lat, s_lon in zip(sel_lats, sel_lons):
            # Haversine distance (vectorized)
            dlat = np.radians(zone_lat - s_lat)
            dlon = np.radians(zone_lon - s_lon)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(s_lat)) * np.cos(np.radians(zone_lat)) * np.sin(dlon/2)**2
            dist_km = 2 * EARTH_R * np.arcsin(np.sqrt(a))
            if dist_km <= radius_km:
                covered_by_any = True
                break

        if covered_by_any:
            total_gain += zone_pop

    return total_gain / total_pop  # Fraction of uncovered population now covered


def evaluate_state(
    selected: Set[int],
    df: pd.DataFrame,
    infra_type: str,
    budget: float,
    cost_per_site: float,
    penalty_weight: float = 0.5,
) -> float:
    """
    Evaluate a state (set of selected sites).

    Objective function:
        score = coverage_gain - penalty_weight × budget_violation_fraction
        budget_violation = max(0, total_cost - budget) / budget

    Args:
        selected:       Set of selected zone indices
        df:             Zone features DataFrame
        infra_type:     Infrastructure type
        budget:         Total available budget
        cost_per_site:  Cost per facility
        penalty_weight: Penalty for exceeding budget

    Returns:
        Scalar objective value (maximize this)
    """
    selected_list = list(selected)
    total_cost = len(selected_list) * cost_per_site

    coverage = compute_coverage_gain(selected_list, df, infra_type)

    # Budget penalty
    if total_cost > budget:
        violation_fraction = (total_cost - budget) / budget
        coverage -= penalty_weight * violation_fraction

    return coverage


# ── Neighbor Generation ───────────────────────────────────────────────────────

def generate_neighbors(
    current: Set[int],
    all_indices: List[int],
    n_swaps: int = 1,
) -> List[Set[int]]:
    """
    Generate neighboring states by swapping selected/unselected sites.

    Neighbor types:
    1. Swap: Remove one selected, add one unselected (most common)
    2. Add: Add one more site (if budget allows)
    3. Remove: Remove one site (cost reduction)

    Args:
        current:    Current set of selected indices
        all_indices: All available zone indices
        n_swaps:    Number of swap neighbors to generate

    Returns:
        List of neighboring state sets
    """
    current = set(current)
    unselected = [i for i in all_indices if i not in current]
    neighbors = []

    # Swap neighbors
    selected_list = list(current)
    if selected_list and unselected:
        for _ in range(n_swaps):
            to_remove = random.choice(selected_list)
            to_add    = random.choice(unselected)
            neighbor  = (current - {to_remove}) | {to_add}
            neighbors.append(neighbor)

    # Add neighbor
    if unselected:
        neighbors.append(current | {random.choice(unselected)})

    # Remove neighbor
    if selected_list:
        neighbors.append(current - {random.choice(selected_list)})

    return neighbors


# ── Hill Climbing Algorithm ───────────────────────────────────────────────────

class HillClimber:
    """
    Stochastic Hill Climbing optimizer for infrastructure placement.

    Uses random restarts to avoid local optima.

    Attributes:
        df:             Zone features DataFrame
        infra_type:     Infrastructure type
        budget:         Total budget constraint
        max_sites:      Maximum number of sites
        max_iterations: Max iterations per restart
        n_restarts:     Number of random restarts
    """

    def __init__(
        self,
        df: pd.DataFrame,
        infra_type: str = "hospital",
        budget: float = 5e8,
        max_sites: int = 5,
        max_iterations: int = 500,
        n_restarts: int = 5,
    ):
        self.df           = df
        self.infra_type   = infra_type
        self.budget       = budget
        self.max_sites    = max_sites
        self.max_iter     = max_iterations
        self.n_restarts   = n_restarts
        self.cost_per_site = INFRA_COSTS.get(infra_type, 3_00_00_000)
        self.all_indices  = list(range(len(df)))
        self.history      = []  # For convergence plot

        log.info(f"HillClimber initialized:")
        log.info(f"  infra_type:    {infra_type}")
        log.info(f"  budget:        ₹{budget:,.0f}")
        log.info(f"  max_sites:     {max_sites}")
        log.info(f"  cost/site:     ₹{self.cost_per_site:,.0f}")
        log.info(f"  affordable:    {int(budget // self.cost_per_site)} sites")

    def _greedy_init(self, n_sites: int) -> Set[int]:
        """
        Greedy initialization: pick top-n zones by priority_composite_100.

        Provides a strong starting point (better than random).

        Args:
            n_sites: Number of sites to initialize

        Returns:
            Set of initial selected indices
        """
        priority_col = next(
            (c for c in ["priority_composite_100", "priority_score", "infrastructure_need_score"]
             if c in self.df.columns),
            None
        )

        if priority_col:
            top_indices = self.df.nlargest(n_sites, priority_col).index.tolist()
            return set(top_indices[:n_sites])
        else:
            return set(random.sample(self.all_indices, min(n_sites, len(self.all_indices))))

    def _random_init(self, n_sites: int) -> Set[int]:
        """Random initialization for restarts."""
        n = min(n_sites, len(self.all_indices))
        return set(random.sample(self.all_indices, n))

    def climb(self, initial_state: Set[int]) -> Tuple[Set[int], float, List[float]]:
        """
        Run one hill climbing ascent from initial state.

        At each step:
        1. Generate neighbors
        2. Find best neighbor
        3. Move if better (strict improvement)
        4. Terminate if no improvement (local maximum)

        Args:
            initial_state: Starting set of selected indices

        Returns:
            (best_state, best_score, score_history)
        """
        current = set(initial_state)
        current_score = evaluate_state(
            current, self.df, self.infra_type, self.budget, self.cost_per_site
        )

        score_history = [current_score]
        no_improvement_count = 0

        for iteration in range(self.max_iter):
            # Generate neighbors (5 swaps + add + remove)
            neighbors = generate_neighbors(current, self.all_indices, n_swaps=5)

            # Find best neighbor
            best_neighbor = None
            best_neighbor_score = current_score

            for neighbor in neighbors:
                # Check site count constraint
                if len(neighbor) > self.max_sites:
                    continue

                score = evaluate_state(
                    neighbor, self.df, self.infra_type, self.budget, self.cost_per_site
                )

                if score > best_neighbor_score:
                    best_neighbor_score = score
                    best_neighbor = neighbor

            # Move to better state
            if best_neighbor is not None and best_neighbor_score > current_score:
                current = best_neighbor
                current_score = best_neighbor_score
                no_improvement_count = 0
                log.debug(f"  iter={iteration}: improved to {current_score:.4f}")
            else:
                no_improvement_count += 1

            score_history.append(current_score)

            # Early termination if stuck
            if no_improvement_count >= 20:
                log.debug(f"  Stuck at local max after {iteration} iterations")
                break

        return current, current_score, score_history

    def optimize(self) -> Tuple[Set[int], float]:
        """
        Run hill climbing with random restarts.

        Greedy init → climb → random restart → climb → ... → best overall

        Returns:
            (best_selected_set, best_score)
        """
        log.info(f"Starting Hill Climbing ({self.n_restarts} restarts, max {self.max_iter} iter each)...")
        n_sites = min(self.max_sites, int(self.budget // self.cost_per_site))
        log.info(f"Budget allows {n_sites} sites")

        global_best: Set[int] = set()
        global_best_score: float = -1.0
        all_histories = []

        # First restart: greedy initialization (usually best)
        init_state = self._greedy_init(n_sites)
        state, score, history = self.climb(init_state)
        all_histories.append(history)

        log.info(f"  Restart 1 (greedy init): score={score:.4f}")
        if score > global_best_score:
            global_best = state
            global_best_score = score

        # Remaining restarts: random initialization
        for restart in range(1, self.n_restarts):
            random.seed(restart)
            init_state = self._random_init(n_sites)
            state, score, history = self.climb(init_state)
            all_histories.append(history)

            log.info(f"  Restart {restart+1} (random init): score={score:.4f}")
            if score > global_best_score:
                global_best = state
                global_best_score = score

        self.history = all_histories

        log.info(f"\nBest solution: score={global_best_score:.4f}, sites={len(global_best)}")
        selected_zones = self.df.iloc[list(global_best)]
        total_cost = len(global_best) * self.cost_per_site
        log.info(f"Total cost: ₹{total_cost:,.0f} (budget: ₹{self.budget:,.0f})")

        return global_best, global_best_score

    def plot_convergence(self) -> None:
        """Plot convergence curves for all restarts."""
        if not self.history:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.history)))

        for i, hist in enumerate(self.history):
            label = "Greedy init" if i == 0 else f"Random restart {i}"
            ax.plot(hist, color=colors[i], alpha=0.8, linewidth=2, label=label)

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Coverage Gain Score", fontsize=12)
        ax.set_title(f"Hill Climbing Convergence — {self.infra_type.title()} Placement", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        path = MODELS_DIR / f"hill_climbing_convergence_{self.infra_type}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Convergence plot → {path}")


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Hill Climbing optimization for SmartCityAI")
    parser.add_argument("--infra",     default="hospital", choices=list(INFRA_COSTS.keys()))
    parser.add_argument("--budget",    type=float, default=5e8, help="Budget in ₹")
    parser.add_argument("--sites",     type=int, default=5, help="Max sites to place")
    parser.add_argument("--restarts",  type=int, default=5, help="Random restarts")
    parser.add_argument("--max-iter",  type=int, default=500, help="Max iterations per restart")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SmartCityAI — Hill Climbing Site Optimization")
    log.info("=" * 60)

    # Load data
    for path in [PROCESSED_DIR / "ml_features.csv", PROCESSED_DIR / "zone_features.csv"]:
        if path.exists():
            df = pd.read_csv(path)
            break
    else:
        log.error("No feature CSV.")
        sys.exit(1)

    log.info(f"Loaded {len(df):,} zones")

    # Add lat/lon if available
    gpkg = PROCESSED_DIR / "zones.gpkg"
    if gpkg.exists():
        try:
            import geopandas as gpd
            zones_gdf = gpd.read_file(str(gpkg))
            centroids = zones_gdf.geometry.centroid
            df = df.merge(
                pd.DataFrame({
                    "h3_id": zones_gdf["h3_id"],
                    "lat": centroids.y,
                    "lon": centroids.x,
                }), on="h3_id", how="left"
            )
        except Exception as e:
            log.warning(f"Could not load zone geometry: {e}")
            df["lat"] = 19.0
            df["lon"] = 72.85

    if "lat" not in df.columns:
        df["lat"] = 19.0
        df["lon"] = 72.85

    # Run optimizer
    climber = HillClimber(
        df          = df,
        infra_type  = args.infra,
        budget      = args.budget,
        max_sites   = args.sites,
        max_iterations = args.max_iter,
        n_restarts  = args.restarts,
    )

    best_set, best_score = climber.optimize()
    climber.plot_convergence()

    # Save results
    best_df = df.iloc[list(best_set)].copy()
    best_df["optimization_score"] = best_score
    best_df["infra_type"] = args.infra

    out_path = PROCESSED_DIR / f"optimized_sites_{args.infra}.csv"
    best_df.to_csv(out_path, index=False)
    log.info(f"Optimized sites saved → {out_path}")

    # Save JSON results
    results = {
        "infra_type": args.infra,
        "budget": args.budget,
        "max_sites": args.sites,
        "best_score": best_score,
        "selected_count": len(best_set),
        "total_cost": len(best_set) * INFRA_COSTS[args.infra],
        "selected_zone_ids": list(df.iloc[list(best_set)]["h3_id"].values),
    }
    with open(MODELS_DIR / "hill_climbing_results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\nHill Climbing complete!")
    log.info(f"  Best coverage gain: {best_score:.4f}")
    log.info(f"  Sites selected: {len(best_set)}")


if __name__ == "__main__":
    main()
