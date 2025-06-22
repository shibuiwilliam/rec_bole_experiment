import pytest

from src.recbole_experiment.models.registry import ModelRegistry
from src.recbole_experiment.training.metrics import MetricsManager


class TestMetricsManager:
    """Test class for MetricsManager"""

    def test_ranking_metrics_constants(self):
        """Test that ranking metrics are correctly defined"""
        expected_ranking = ["Recall", "MRR", "NDCG", "Hit", "Precision"]
        assert MetricsManager.RANKING_METRICS == expected_ranking

    def test_value_metrics_constants(self):
        """Test that value metrics are correctly defined"""
        expected_value = ["AUC", "LogLoss", "MAE", "RMSE"]
        assert MetricsManager.VALUE_METRICS == expected_value

    def test_get_ranking_metrics(self):
        """Test get_ranking_metrics class method"""
        ranking_metrics = MetricsManager.get_ranking_metrics()
        assert ranking_metrics == ["Recall", "MRR", "NDCG", "Hit", "Precision"]
        assert isinstance(ranking_metrics, list)

    def test_get_value_metrics(self):
        """Test get_value_metrics class method"""
        value_metrics = MetricsManager.get_value_metrics()
        assert value_metrics == ["AUC", "LogLoss", "MAE", "RMSE"]
        assert isinstance(value_metrics, list)

    def test_get_all_metrics(self):
        """Test get_all_metrics class method"""
        all_metrics = MetricsManager.get_all_metrics()
        expected = [
            "Recall",
            "MRR",
            "NDCG",
            "Hit",
            "Precision",
            "AUC",
            "LogLoss",
            "MAE",
            "RMSE",
        ]
        assert all_metrics == expected
        assert len(all_metrics) == 9

    def test_determine_metrics_for_context_aware_models(self):
        """Test metrics determination for context-aware models"""
        context_aware_models = ["LR", "FM", "DeepFM", "WideDeep"]

        for model in context_aware_models:
            # Context-aware models should always use value metrics
            metrics_labeled = MetricsManager.determine_metrics_for_model(
                model, "labeled"
            )
            metrics_full = MetricsManager.determine_metrics_for_model(model, "full")

            assert metrics_labeled == MetricsManager.VALUE_METRICS
            assert metrics_full == MetricsManager.VALUE_METRICS

    def test_determine_metrics_for_general_models_labeled_mode(self):
        """Test metrics determination for general models in labeled mode"""
        general_models = ["Pop", "BPR", "NeuMF"]

        for model in general_models:
            metrics = MetricsManager.determine_metrics_for_model(model, "labeled")
            assert metrics == MetricsManager.VALUE_METRICS

    def test_determine_metrics_for_general_models_full_mode(self):
        """Test metrics determination for general models in full mode"""
        general_models = ["Pop", "BPR", "NeuMF"]

        for model in general_models:
            metrics = MetricsManager.determine_metrics_for_model(model, "full")
            assert metrics == MetricsManager.RANKING_METRICS

    def test_determine_metrics_for_sequential_models_labeled_mode(self):
        """Test metrics determination for sequential models in labeled mode"""
        sequential_models = ["GRU4Rec", "SASRec", "BERT4Rec"]

        for model in sequential_models:
            metrics = MetricsManager.determine_metrics_for_model(model, "labeled")
            assert metrics == MetricsManager.VALUE_METRICS

    def test_determine_metrics_for_sequential_models_full_mode(self):
        """Test metrics determination for sequential models in full mode"""
        sequential_models = ["GRU4Rec", "SASRec", "BERT4Rec"]

        for model in sequential_models:
            metrics = MetricsManager.determine_metrics_for_model(model, "full")
            assert metrics == MetricsManager.RANKING_METRICS

    def test_determine_metrics_unknown_model(self):
        """Test metrics determination for unknown model"""
        # Unknown models should be treated as non-context-aware
        metrics_labeled = MetricsManager.determine_metrics_for_model(
            "UnknownModel", "labeled"
        )
        metrics_full = MetricsManager.determine_metrics_for_model(
            "UnknownModel", "full"
        )

        assert metrics_labeled == MetricsManager.VALUE_METRICS
        assert metrics_full == MetricsManager.RANKING_METRICS

    def test_determine_metrics_default_eval_mode(self):
        """Test metrics determination with default eval_mode"""
        # Should default to "labeled" mode behavior
        metrics = MetricsManager.determine_metrics_for_model("Pop")
        assert metrics == MetricsManager.VALUE_METRICS

    def test_metrics_lists_are_immutable(self):
        """Test that modifying returned lists doesn't affect constants"""
        # Store original state
        original_ranking = ["Recall", "MRR", "NDCG", "Hit", "Precision"]
        original_value = ["AUC", "LogLoss", "MAE", "RMSE"]

        # Get fresh copies for modification
        ranking_metrics = MetricsManager.get_ranking_metrics()
        value_metrics = MetricsManager.get_value_metrics()
        all_metrics = MetricsManager.get_all_metrics()

        # Modify the returned lists
        ranking_metrics.append("NewMetric")
        value_metrics.append("NewMetric")
        all_metrics.append("NewMetric")

        # Original constants should be unchanged
        assert MetricsManager.RANKING_METRICS == original_ranking
        assert MetricsManager.VALUE_METRICS == original_value
        assert "NewMetric" not in MetricsManager.get_ranking_metrics()
        assert "NewMetric" not in MetricsManager.get_value_metrics()

    def test_all_model_categories_covered(self):
        """Test that all model categories are properly handled"""
        # Test all context-aware models
        for model in ModelRegistry.CONTEXT_AWARE_MODELS:
            metrics = MetricsManager.determine_metrics_for_model(model, "labeled")
            assert metrics == MetricsManager.VALUE_METRICS

        # Test all general models
        for model in ModelRegistry.GENERAL_MODELS:
            metrics_labeled = MetricsManager.determine_metrics_for_model(
                model, "labeled"
            )
            metrics_full = MetricsManager.determine_metrics_for_model(model, "full")
            assert metrics_labeled == MetricsManager.VALUE_METRICS
            assert metrics_full == MetricsManager.RANKING_METRICS

        # Test all sequential models
        for model in ModelRegistry.SEQUENTIAL_MODELS:
            metrics_labeled = MetricsManager.determine_metrics_for_model(
                model, "labeled"
            )
            metrics_full = MetricsManager.determine_metrics_for_model(model, "full")
            assert metrics_labeled == MetricsManager.VALUE_METRICS
            assert metrics_full == MetricsManager.RANKING_METRICS

    @pytest.mark.parametrize("eval_mode", ["labeled", "full"])
    def test_determine_metrics_case_sensitivity(self, eval_mode):
        """Test that model name comparison is case sensitive"""
        # DeepFM should be recognized as context-aware
        metrics = MetricsManager.determine_metrics_for_model("DeepFM", eval_mode)
        assert metrics == MetricsManager.VALUE_METRICS

        # deepfm should not be recognized (case sensitive)
        metrics = MetricsManager.determine_metrics_for_model("deepfm", eval_mode)
        if eval_mode == "labeled":
            assert metrics == MetricsManager.VALUE_METRICS
        else:
            assert metrics == MetricsManager.RANKING_METRICS

    def test_no_duplicate_metrics(self):
        """Test that there are no duplicate metrics across categories"""
        # Use constants directly to ensure we're testing the original state
        ranking = set(MetricsManager.RANKING_METRICS)
        value = set(MetricsManager.VALUE_METRICS)

        # No overlaps between ranking and value metrics
        assert len(ranking.intersection(value)) == 0

        # All metrics combined should equal sum of individual sets
        all_metrics = MetricsManager.get_all_metrics()
        assert len(all_metrics) == len(ranking) + len(value)
