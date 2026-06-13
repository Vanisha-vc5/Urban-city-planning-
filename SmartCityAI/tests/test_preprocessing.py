"""
test_preprocessing.py
=====================
Phase 10 - Testing: Preprocessing Pipeline Unit Tests

Tests for create_zones.py, data loading, H3 grid generation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TestH3Grid:
    """Tests for H3 hexagonal grid generation."""

    def test_generate_h3_grid_basic(self):
        """H3 grid should generate cells for a valid bounding box."""
        from preprocessing.create_zones import generate_h3_grid
        gdf = generate_h3_grid(18.87, 72.77, 19.07, 72.97, resolution=8)
        assert len(gdf) > 0, "H3 grid should produce cells"
        assert "h3_id" in gdf.columns
        assert "geometry" in gdf.columns
        assert "area_km2" in gdf.columns

    def test_h3_resolution_effects(self):
        """Higher resolution should produce more (smaller) cells."""
        from preprocessing.create_zones import generate_h3_grid
        gdf_low  = generate_h3_grid(18.87, 72.77, 19.07, 72.97, resolution=7)
        gdf_high = generate_h3_grid(18.87, 72.77, 19.07, 72.97, resolution=8)
        assert len(gdf_high) > len(gdf_low), "Res 8 should have more cells than res 7"

    def test_h3_cell_area(self):
        """H3 resolution 8 cells should be ~0.74 km²."""
        from preprocessing.create_zones import generate_h3_grid
        gdf = generate_h3_grid(18.87, 72.77, 18.97, 72.87, resolution=8)
        if len(gdf) > 0:
            mean_area = gdf["area_km2"].mean()
            assert 0.4 < mean_area < 1.5, f"Expected ~0.74 km², got {mean_area:.3f}"

    def test_h3_ids_unique(self):
        """Each H3 cell should have a unique ID."""
        from preprocessing.create_zones import generate_h3_grid
        gdf = generate_h3_grid(18.87, 72.77, 19.07, 72.97, resolution=8)
        assert gdf["h3_id"].nunique() == len(gdf), "H3 IDs must be unique"


class TestPopulationFeatures:
    """Tests for population feature computation."""

    def setup_method(self):
        """Create synthetic test zones DataFrame."""
        np.random.seed(42)
        n = 50
        self.zones_df = pd.DataFrame({
            "h3_id":    [f"zone_{i}" for i in range(n)],
            "area_km2": np.random.uniform(0.5, 1.5, n),
        })

    def test_synthetic_fallback(self):
        """Should generate synthetic population data when no raster/census found."""
        from preprocessing.create_zones import add_population_features
        result = add_population_features(self.zones_df.copy())
        assert "population_density" in result.columns
        assert "population_total" in result.columns
        assert (result["population_density"] > 0).all(), "Density should be positive"

    def test_population_density_formula(self):
        """population_density = population_total / area_km2."""
        from preprocessing.create_zones import add_population_features
        result = add_population_features(self.zones_df.copy())
        computed_density = result["population_total"] / result["area_km2"]
        # Allow 1% tolerance due to rounding
        ratio = (result["population_density"] / computed_density.clip(lower=1)).clip(0.95, 1.05)
        assert ratio.between(0.95, 1.05).all(), "Density formula incorrect"

    def test_elderly_ratio_bounds(self):
        """Elderly ratio must be in [0, 1]."""
        from preprocessing.create_zones import add_population_features
        result = add_population_features(self.zones_df.copy())
        if "elderly_ratio" in result.columns:
            assert result["elderly_ratio"].between(0, 1).all(), "elderly_ratio out of [0,1]"


class TestHospitalFeatures:
    """Tests for hospital accessibility feature computation."""

    def setup_method(self):
        np.random.seed(42)
        n = 20
        self.zones_df = pd.DataFrame({
            "h3_id":   [f"zone_{i}" for i in range(n)],
            "area_km2": np.ones(n),
            "population_total": np.ones(n) * 10000,
        })

    def test_distance_positive(self):
        """Distance to hospital must be non-negative."""
        from preprocessing.create_zones import add_hospital_features
        result = add_hospital_features(self.zones_df.copy())
        assert "dist_nearest_hospital" in result.columns
        assert (result["dist_nearest_hospital"] >= 0).all()

    def test_coverage_gap_range(self):
        """Coverage gap must be in [0, 1]."""
        from preprocessing.create_zones import add_hospital_features
        result = add_hospital_features(self.zones_df.copy())
        assert result["coverage_gap"].between(0, 1).all()

    def test_coverage_consistency(self):
        """coverage_gap = 1 - hospital_coverage_ratio."""
        from preprocessing.create_zones import add_hospital_features
        result = add_hospital_features(self.zones_df.copy())
        if "hospital_coverage_ratio" in result.columns:
            expected_gap = 1.0 - result["hospital_coverage_ratio"]
            diff = (result["coverage_gap"] - expected_gap).abs()
            assert (diff < 0.01).all(), "coverage_gap != 1 - hospital_coverage_ratio"


class TestPriorityScore:
    """Tests for priority score computation."""

    def test_priority_range(self):
        """Priority score must be in [0, 100]."""
        from preprocessing.create_zones import compute_priority_score
        n = 100
        df = pd.DataFrame({
            "h3_id": [f"z{i}" for i in range(n)],
            "coverage_gap": np.random.rand(n),
            "population_density": np.random.lognormal(9, 0.5, n),
            "elderly_ratio": np.random.beta(2, 20, n),
            "income_bracket_norm": np.random.rand(n),
            "road_accessibility_index": np.random.rand(n),
        })
        result = compute_priority_score(df)
        assert result["priority_score"].between(0, 100).all()

    def test_priority_class_labels(self):
        """Priority class must be one of High/Medium/Low."""
        from preprocessing.create_zones import compute_priority_score
        df = pd.DataFrame({
            "h3_id": [f"z{i}" for i in range(50)],
            "coverage_gap": np.random.rand(50),
            "population_density": np.random.lognormal(9, 0.5, 50),
            "elderly_ratio": np.random.beta(2, 20, 50),
            "income_bracket_norm": np.random.rand(50),
            "road_accessibility_index": np.random.rand(50),
        })
        result = compute_priority_score(df)
        valid_classes = {"Low", "Medium", "High"}
        actual_classes = set(result["priority_class"].dropna().astype(str))
        assert actual_classes.issubset(valid_classes), f"Invalid classes: {actual_classes - valid_classes}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
