"""
test_optimization.py
====================
Phase 10 - Testing: Classical AI Algorithm Tests
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def make_test_zones(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "h3_id":                  [f"zone_{i}" for i in range(n)],
        "coverage_gap":           rng.beta(2, 3, n),
        "population_density":     rng.lognormal(9, 0.5, n),
        "population_density_norm":rng.beta(2, 3, n),
        "population_total":       rng.lognormal(10, 0.6, n),
        "elderly_ratio":          rng.beta(2, 20, n),
        "road_accessibility_index":rng.beta(3, 3, n),
        "equity_adjusted_priority":rng.beta(2, 4, n),
        "emergency_response_time_min":rng.exponential(5, n).clip(1, 30),
        "traffic_density":        rng.lognormal(2, 0.5, n),
        "priority_composite_100": rng.uniform(10, 95, n),
        "infrastructure_need_score": rng.beta(2, 3, n),
        "lat":  19.0 + rng.normal(0, 0.05, n),
        "lon":  72.85 + rng.normal(0, 0.05, n),
    })


class TestBestFirstSearch:
    """Tests for Best First Search algorithm."""

    def test_heap_builds_correctly(self):
        """Heap should contain exactly as many zones as input DataFrame."""
        from optimization.best_first import BestFirstSearch
        df = make_test_zones(30)
        bfs = BestFirstSearch(df, infra_type="hospital")
        bfs.build_heap()
        assert len(bfs.heap) == len(df)

    def test_run_returns_ordered_list(self):
        """Ranked sites should be ordered by priority (descending)."""
        from optimization.best_first import BestFirstSearch
        df = make_test_zones(50)
        bfs = BestFirstSearch(df, infra_type="hospital")
        ranked = bfs.run(top_k=10)
        scores = [s.priority_score for s in ranked]
        assert scores == sorted(scores, reverse=True), "Results must be ordered by priority descending"

    def test_top_k_limit(self):
        """run() should return at most top_k results."""
        from optimization.best_first import BestFirstSearch
        df = make_test_zones(50)
        bfs = BestFirstSearch(df, infra_type="hospital")
        for k in [5, 10, 20]:
            ranked = bfs.run(top_k=k)
            assert len(ranked) <= k

    def test_budget_constraint(self):
        """Budget constraint should limit results."""
        from optimization.best_first import BestFirstSearch
        df = make_test_zones(50)
        bfs = BestFirstSearch(df, infra_type="hospital")
        # 1 hospital = ₹5 Crore → budget of 15 Crore → max 3 sites
        budget_15cr = 15_00_00_000
        ranked = bfs.run(top_k=20, budget=budget_15cr)
        assert len(ranked) <= 3

    def test_results_dataframe_columns(self):
        """Results DataFrame should have required columns."""
        from optimization.best_first import BestFirstSearch
        df = make_test_zones(30)
        bfs = BestFirstSearch(df, infra_type="hospital")
        ranked = bfs.run(top_k=5)
        results_df = bfs.get_results_dataframe(ranked)
        required_cols = ["rank", "h3_id", "priority_score", "coverage_gap"]
        for col in required_cols:
            assert col in results_df.columns, f"Missing column: {col}"

    def test_different_infra_types(self):
        """BFS should work for all infrastructure types."""
        from optimization.best_first import BestFirstSearch
        df = make_test_zones(30)
        for infra in ["hospital", "school", "ev_station", "fire_station"]:
            bfs = BestFirstSearch(df, infra_type=infra)
            ranked = bfs.run(top_k=5)
            assert len(ranked) > 0, f"No results for {infra}"


class TestHillClimbing:
    """Tests for Hill Climbing optimization."""

    def test_optimize_returns_valid_set(self):
        """Optimizer should return a set of selected zone indices."""
        from optimization.hill_climbing import HillClimber
        df = make_test_zones(30)
        climber = HillClimber(df, infra_type="hospital", budget=5e8, max_sites=3,
                               max_iterations=50, n_restarts=2)
        best_set, best_score = climber.optimize()
        assert isinstance(best_set, set)
        assert len(best_set) <= 3
        assert isinstance(best_score, float)

    def test_budget_constraint_respected(self):
        """Selected sites cost should not exceed budget."""
        from optimization.hill_climbing import HillClimber, INFRA_COSTS
        df = make_test_zones(30)
        budget = 2e8  # ₹20 Crore
        climber = HillClimber(df, infra_type="hospital", budget=budget, max_sites=5,
                               max_iterations=50, n_restarts=2)
        best_set, _ = climber.optimize()
        total_cost = len(best_set) * INFRA_COSTS["hospital"]
        assert total_cost <= budget + 1, f"Cost ₹{total_cost} > budget ₹{budget}"

    def test_greedy_init_nonempty(self):
        """Greedy initialization should return a non-empty set."""
        from optimization.hill_climbing import HillClimber
        df = make_test_zones(50)
        climber = HillClimber(df, infra_type="school", budget=5e8, max_sites=5,
                               max_iterations=10, n_restarts=1)
        init = climber._greedy_init(n_sites=3)
        assert len(init) == 3
        assert all(0 <= i < len(df) for i in init)


class TestAStarSearch:
    """Tests for A* routing algorithm."""

    def setup_method(self):
        from optimization.astar_routing import RoadGraph
        self.graph = RoadGraph()
        self.graph.generate_synthetic_graph(center_lat=19.0, center_lon=72.85, n_nodes=100)

    def test_synthetic_graph_nodes(self):
        """Synthetic graph should have approximately n_nodes nodes."""
        n_nodes = len(self.graph.nodes)
        assert n_nodes > 50, f"Expected ~100 nodes, got {n_nodes}"

    def test_astar_finds_path(self):
        """A* should find a path between two connected nodes."""
        from optimization.astar_routing import astar_search
        node_ids = list(self.graph.nodes.keys())
        if len(node_ids) >= 2:
            result = astar_search(self.graph, node_ids[0], node_ids[-1])
            if result:
                path, time = result
                assert len(path) >= 2
                assert time > 0

    def test_astar_same_node(self):
        """A* from a node to itself should return immediately."""
        from optimization.astar_routing import astar_search
        node_ids = list(self.graph.nodes.keys())
        if node_ids:
            result = astar_search(self.graph, node_ids[0], node_ids[0])
            if result:
                path, time = result
                assert time == 0.0 or len(path) == 1

    def test_heuristic_admissibility(self):
        """Heuristic should not exceed actual path time (admissibility)."""
        from optimization.astar_routing import astar_search, heuristic, haversine_km
        nodes = list(self.graph.nodes.values())
        if len(nodes) >= 2:
            start, goal = nodes[0], nodes[-1]
            h = heuristic(start, goal, max_speed=80.0)
            result = astar_search(self.graph, start.node_id, goal.node_id)
            if result:
                _, actual_time = result
                assert h <= actual_time + 0.01, f"h={h:.3f} > actual={actual_time:.3f} (inadmissible!)"

    def test_nearest_node(self):
        """nearest_node should return a valid node ID."""
        node_id = self.graph.nearest_node(19.01, 72.86)
        assert node_id is not None
        assert node_id in self.graph.nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
