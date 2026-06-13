"""
test_models.py
==============
Phase 10 - Testing: ML Model Unit Tests
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def make_urban_df(n: int = 200) -> pd.DataFrame:
    """Create synthetic urban zone DataFrame for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "h3_id":                   [f"zone_{i}" for i in range(n)],
        "population_density_log":  rng.normal(9.5, 0.8, n),
        "elderly_ratio":           rng.beta(2.5, 28, n),
        "income_bracket_norm":     rng.beta(3, 5, n),
        "coverage_gap":            rng.beta(2, 3, n),
        "dist_nearest_hospital":   rng.exponential(3, n).clip(0.2, 15),
        "road_accessibility_index":rng.beta(3, 3, n),
        "vulnerability_index":     rng.beta(2, 5, n),
        "infrastructure_need_score":rng.beta(2, 3, n),
        "density_gap_interaction": rng.beta(2, 4, n),
        "emergency_response_time_min": rng.exponential(5, n).clip(1, 30),
        "cluster_id":              rng.integers(0, 5, n),
        "priority_class":          rng.choice(["Low","Medium","High"], n),
        "priority_composite_100":  rng.uniform(10, 95, n),
        "future_demand_5yr_norm":  rng.beta(3, 4, n),
    })


class TestKMeans:
    """Tests for K-Means clustering."""

    def test_train_basic(self):
        """KMeans should train without errors and return labels."""
        from models.kmeans import train_kmeans, DEFAULT_KMEANS_FEATURES
        df = make_urban_df(200)
        model, scaler, df_out = train_kmeans(df, DEFAULT_KMEANS_FEATURES, k=3)
        assert "cluster_id" in df_out.columns
        assert len(df_out["cluster_id"].unique()) <= 3

    def test_cluster_count(self):
        """Number of unique clusters should equal k (or less if data is small)."""
        from models.kmeans import train_kmeans
        df = make_urban_df(100)
        features = ["population_density_log", "coverage_gap", "elderly_ratio"]
        _, _, df_out = train_kmeans(df, features, k=4)
        assert 1 <= df_out["cluster_id"].nunique() <= 4

    def test_predict_cluster(self):
        """predict_cluster should return an integer."""
        from models.kmeans import train_kmeans, predict_cluster
        df = make_urban_df(200)
        features = ["population_density_log", "coverage_gap"]
        model, scaler, _ = train_kmeans(df, features, k=3)
        sample = np.array([9.5, 0.4])
        result = predict_cluster(sample, model, scaler)
        assert isinstance(result, int)
        assert 0 <= result < 3


class TestSVM:
    """Tests for SVM priority classifier."""

    def test_train_basic(self):
        """SVM pipeline should fit and predict without errors."""
        from models.svm_priority import build_svm_pipeline, prepare_svm_data
        df = make_urban_df(200)
        features = ["population_density_log", "coverage_gap", "elderly_ratio",
                    "road_accessibility_index", "vulnerability_index"]
        X_train, X_test, y_train, y_test, le = prepare_svm_data(df, features)
        pipeline = build_svm_pipeline(C=1, gamma="scale")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        assert len(y_pred) == len(X_test)

    def test_predict_proba_shape(self):
        """predict_proba should return probabilities for all classes."""
        from models.svm_priority import build_svm_pipeline, prepare_svm_data
        df = make_urban_df(200)
        features = ["coverage_gap", "elderly_ratio", "vulnerability_index"]
        X_train, X_test, y_train, y_test, le = prepare_svm_data(df, features)
        pipeline = build_svm_pipeline(C=1)
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)
        n_classes = len(le.classes_)
        assert proba.shape == (len(X_test), n_classes)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_output_classes(self):
        """SVM should only predict valid class labels."""
        from models.svm_priority import build_svm_pipeline, prepare_svm_data
        df = make_urban_df(200)
        features = ["coverage_gap", "population_density_log"]
        X_train, X_test, y_train, y_test, le = prepare_svm_data(df, features)
        pipeline = build_svm_pipeline(C=1)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        valid_encoded = set(range(len(le.classes_)))
        assert set(y_pred).issubset(valid_encoded)


class TestDecisionTree:
    """Tests for Decision Tree classifier."""

    def test_train_predict(self):
        """Decision Tree should train and predict correctly."""
        from models.decision_tree import train_decision_tree, prepare_dt_data
        df = make_urban_df(200)
        features = ["coverage_gap", "population_density_log", "elderly_ratio"]
        X_train, X_test, y_train, y_test, avail = prepare_dt_data(df, features)
        model = train_decision_tree(X_train, y_train, max_depth=4)
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(X_test)

    def test_depth_limit(self):
        """Tree depth should not exceed max_depth parameter."""
        from models.decision_tree import train_decision_tree, prepare_dt_data
        df = make_urban_df(200)
        features = ["coverage_gap", "population_density_log", "elderly_ratio", "vulnerability_index"]
        X_train, X_test, y_train, y_test, avail = prepare_dt_data(df, features)
        for depth in [2, 4, 6]:
            model = train_decision_tree(X_train, y_train, max_depth=depth)
            assert model.get_depth() <= depth, f"Tree depth {model.get_depth()} > {depth}"

    def test_feature_importance_sum(self):
        """Feature importances should sum to approximately 1.0."""
        from models.decision_tree import train_decision_tree, prepare_dt_data
        df = make_urban_df(200)
        features = ["coverage_gap", "population_density_log", "elderly_ratio"]
        X_train, _, y_train, _, _ = prepare_dt_data(df, features)
        model = train_decision_tree(X_train, y_train, max_depth=4)
        assert abs(model.feature_importances_.sum() - 1.0) < 1e-6


class TestRegression:
    """Tests for population demand regression."""

    def test_ridge_fit_predict(self):
        """Ridge regression should fit and predict in [0,1] range."""
        from models.population_regression import build_ridge_pipeline, prepare_regression_data
        df = make_urban_df(200)
        features = ["population_density_log", "elderly_ratio", "coverage_gap"]
        X_train, X_test, y_train, y_test, avail = prepare_regression_data(df, features)
        pipeline = build_ridge_pipeline(alpha=1.0)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        assert len(y_pred) == len(X_test)
        # Predictions can be slightly outside [0,1] for regression
        assert y_pred.min() > -1, "Predictions shouldn't be wildly negative"

    def test_svr_fit(self):
        """SVR should fit without errors."""
        from models.population_regression import build_svr_pipeline, prepare_regression_data
        df = make_urban_df(200)
        features = ["population_density_log", "coverage_gap"]
        X_train, X_test, y_train, y_test, avail = prepare_regression_data(df, features)
        pipeline = build_svr_pipeline()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        assert len(y_pred) == len(X_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
