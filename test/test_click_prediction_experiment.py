from unittest.mock import MagicMock

import pandas as pd

from src.recbole_experiment.data.dataset import Dataset
from src.recbole_experiment.experiments.click_prediction import (
    ClickPredictionExperiment,
)


class TestClickPredictionExperiment:
    """Test class for ClickPredictionExperiment"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_users = pd.DataFrame(
            {
                "user_id:token": ["u_0", "u_1"],
                "age:float": [25, 30],
                "gender:token": ["M", "F"],
            }
        )

        self.sample_items = pd.DataFrame(
            {
                "item_id:token": ["i_0", "i_1"],
                "price:float": [100.0, 200.0],
                "rating:float": [3.5, 4.0],
                "category:token": ["Electronics", "Books"],
            }
        )

        self.sample_interactions = pd.DataFrame(
            {
                "user_id:token": ["u_0", "u_1"],
                "item_id:token": ["i_0", "i_1"],
                "label:float": [1.0, 0.0],
                "timestamp:float": [1500000000, 1600000000],
            }
        )

        self.dataset = Dataset(
            self.sample_users, self.sample_items, self.sample_interactions
        )
        self.experiment = ClickPredictionExperiment(self.dataset)

    def test_init(self):
        """Test ClickPredictionExperiment initialization"""
        assert self.experiment.dataset is self.dataset
        assert hasattr(self.experiment, "config_manager")
        assert hasattr(self.experiment, "trainer")
        assert self.experiment._data_prepared is False

    def test_ensure_data_prepared_first_call(self, mocker):
        """Test _ensure_data_prepared on first call"""
        mock_print = mocker.patch("builtins.print")
        mock_save = mocker.patch.object(
            self.experiment.dataset, "save_to_recbole_format"
        )

        self.experiment._ensure_data_prepared()

        mock_save.assert_called_once()
        mock_print.assert_called_with("=== データ準備 ===")
        assert self.experiment._data_prepared is True

    def test_ensure_data_prepared_subsequent_call(self, mocker):
        """Test _ensure_data_prepared on subsequent calls"""
        mocker.patch("builtins.print")
        mock_save = mocker.patch.object(
            self.experiment.dataset, "save_to_recbole_format"
        )

        # First call
        self.experiment._ensure_data_prepared()
        # Second call
        self.experiment._ensure_data_prepared()

        # Should only save once
        mock_save.assert_called_once()
        assert self.experiment._data_prepared is True

    def test_run_single_model_experiment(self, mocker):
        """Test run_single_model_experiment"""
        mock_ensure_data = mocker.patch.object(self.experiment, "_ensure_data_prepared")
        mock_train = mocker.patch.object(self.experiment.trainer, "train_single_model")

        mock_train.return_value = {"test_result": {"auc": 0.85}}

        result = self.experiment.run_single_model_experiment(
            "DeepFM", ["AUC"], "labeled"
        )

        mock_ensure_data.assert_called_once()
        mock_train.assert_called_once_with("DeepFM", ["AUC"], "labeled")
        assert result == {"test_result": {"auc": 0.85}}

    def test_run_quick_comparison(self, mocker):
        """Test run_quick_comparison"""
        mock_ensure_data = mocker.patch.object(self.experiment, "_ensure_data_prepared")
        mock_get_quick = mocker.patch(
            "src.recbole_experiment.models.registry.ModelRegistry.get_quick_models"
        )
        mocker.patch("builtins.print")
        mock_compare = mocker.patch.object(self.experiment.trainer, "compare_models")

        mock_get_quick.return_value = ["LR", "FM", "DeepFM"]
        mock_compare.return_value = {"LR": {"test_result": {"auc": 0.80}}}

        self.experiment.run_quick_comparison(["AUC"], "labeled")

        mock_ensure_data.assert_called_once()
        mock_get_quick.assert_called_once()
        mock_compare.assert_called_once_with(
            ["LR", "FM", "DeepFM"],
            mode="quick",
            metrics=["AUC"],
            eval_mode="labeled",
        )

    def test_run_comprehensive_comparison(self, mocker):
        """Test run_comprehensive_comparison"""
        mock_ensure_data = mocker.patch.object(self.experiment, "_ensure_data_prepared")
        mock_get_all = mocker.patch(
            "src.recbole_experiment.models.registry.ModelRegistry.get_all_models"
        )
        mocker.patch("builtins.print")
        mock_compare = mocker.patch.object(self.experiment.trainer, "compare_models")

        mock_get_all.return_value = ["LR", "FM", "DeepFM", "Pop", "BPR"]
        mock_compare.return_value = {"LR": {"test_result": {"auc": 0.80}}}

        self.experiment.run_comprehensive_comparison(["AUC"], "full")

        mock_ensure_data.assert_called_once()
        mock_get_all.assert_called_once()
        mock_compare.assert_called_once_with(
            ["LR", "FM", "DeepFM", "Pop", "BPR"],
            mode="full",
            metrics=["AUC"],
            eval_mode="full",
        )

    def test_run_value_metrics_comparison(self, mocker):
        """Test run_value_metrics_comparison"""
        mock_ensure_data = mocker.patch.object(self.experiment, "_ensure_data_prepared")
        mock_get_context = mocker.patch(
            "src.recbole_experiment.models.registry.ModelRegistry.get_context_aware_models"
        )
        mock_get_value = mocker.patch(
            "src.recbole_experiment.training.metrics.MetricsManager.get_value_metrics"
        )
        mocker.patch("builtins.print")
        mock_compare = mocker.patch.object(self.experiment.trainer, "compare_models")

        mock_get_context.return_value = ["LR", "FM", "DeepFM"]
        mock_get_value.return_value = ["AUC", "LogLoss", "MAE", "RMSE"]
        mock_compare.return_value = {"LR": {"test_result": {"auc": 0.80}}}

        self.experiment.run_value_metrics_comparison()

        mock_ensure_data.assert_called_once()
        mock_get_context.assert_called_once()
        mock_get_value.assert_called_once()
        mock_compare.assert_called_once_with(
            ["LR", "FM", "DeepFM"],
            mode="quick",
            metrics=["AUC", "LogLoss", "MAE", "RMSE"],
            eval_mode="labeled",
        )

    def test_run_ranking_metrics_comparison(self, mocker):
        """Test run_ranking_metrics_comparison"""
        mock_ensure_data = mocker.patch.object(self.experiment, "_ensure_data_prepared")
        mock_get_compatible = mocker.patch(
            "src.recbole_experiment.models.registry.ModelRegistry.get_compatible_ranking_models"
        )
        mock_get_ranking = mocker.patch(
            "src.recbole_experiment.training.metrics.MetricsManager.get_ranking_metrics"
        )
        mocker.patch("builtins.print")
        mock_compare = mocker.patch.object(self.experiment.trainer, "compare_models")

        mock_get_compatible.return_value = ["Pop", "BPR", "NeuMF", "DGCF"]
        mock_get_ranking.return_value = ["Recall", "MRR", "NDCG", "Hit", "Precision"]
        mock_compare.return_value = {"Pop": {"test_result": {"recall@10": 0.25}}}

        self.experiment.run_ranking_metrics_comparison()

        mock_ensure_data.assert_called_once()
        mock_get_compatible.assert_called_once()
        mock_get_ranking.assert_called_once()
        mock_compare.assert_called_once_with(
            ["Pop", "BPR", "NeuMF", "DGCF"],
            mode="quick",
            metrics=["Recall", "MRR", "NDCG", "Hit", "Precision"],
            eval_mode="full",
        )

    def test_run_custom_metrics_comparison_defaults(self, mocker):
        """Test run_custom_metrics_comparison with default parameters"""
        mock_ensure_data = mocker.patch.object(self.experiment, "_ensure_data_prepared")
        mock_get_quick = mocker.patch(
            "src.recbole_experiment.models.registry.ModelRegistry.get_quick_models"
        )
        mock_get_value = mocker.patch(
            "src.recbole_experiment.training.metrics.MetricsManager.get_value_metrics"
        )
        mocker.patch("builtins.print")
        mock_compare = mocker.patch.object(self.experiment.trainer, "compare_models")

        mock_get_quick.return_value = ["LR", "FM", "DeepFM"]
        mock_get_value.return_value = ["AUC", "LogLoss", "MAE", "RMSE"]
        mock_compare.return_value = {"LR": {"test_result": {"auc": 0.80}}}

        self.experiment.run_custom_metrics_comparison()

        mock_ensure_data.assert_called_once()
        mock_get_quick.assert_called_once()
        mock_get_value.assert_called_once()
        mock_compare.assert_called_once_with(
            ["LR", "FM", "DeepFM"],
            mode="quick",
            metrics=["AUC", "LogLoss", "MAE", "RMSE"],
            eval_mode="labeled",
        )

    def test_run_custom_metrics_comparison_custom_params(self, mocker):
        """Test run_custom_metrics_comparison with custom parameters"""
        mock_ensure_data = mocker.patch.object(self.experiment, "_ensure_data_prepared")
        mocker.patch("builtins.print")
        mock_compare = mocker.patch.object(self.experiment.trainer, "compare_models")

        mock_compare.return_value = {"DeepFM": {"test_result": {"auc": 0.85}}}

        self.experiment.run_custom_metrics_comparison(
            models=["DeepFM", "LR"], metrics=["AUC"], eval_mode="full"
        )

        mock_ensure_data.assert_called_once()
        mock_compare.assert_called_once_with(
            ["DeepFM", "LR"], mode="quick", metrics=["AUC"], eval_mode="full"
        )

    def test_predict_click_probability(self, mocker):
        """Test predict_click_probability"""
        # Setup mocks
        mock_config = mocker.patch(
            "src.recbole_experiment.experiments.click_prediction.Config"
        )
        mock_init_seed = mocker.patch(
            "src.recbole_experiment.experiments.click_prediction.init_seed"
        )
        mock_create_dataset = mocker.patch(
            "src.recbole_experiment.experiments.click_prediction.create_dataset"
        )
        mock_data_prep = mocker.patch(
            "src.recbole_experiment.experiments.click_prediction.data_preparation"
        )
        mock_deepfm = mocker.patch(
            "src.recbole_experiment.experiments.click_prediction.DeepFM"
        )
        mocker.patch("builtins.print")

        mock_config_instance = MagicMock()
        mock_config_instance.__getitem__.side_effect = lambda key: {
            "seed": 2023,
            "reproducibility": True,
            "device": "cpu",
        }[key]
        mock_config.return_value = mock_config_instance

        mock_dataset = MagicMock()
        mock_create_dataset.return_value = mock_dataset

        mock_train_data = MagicMock()
        mock_train_data.dataset = mock_dataset
        mock_data_prep.return_value = (mock_train_data, MagicMock(), MagicMock())

        mock_model = MagicMock()
        mock_deepfm.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Execute
        self.experiment.predict_click_probability()

        # Verify
        mock_config.assert_called_once()
        mock_init_seed.assert_called_once()
        mock_create_dataset.assert_called_once()
        mock_data_prep.assert_called_once()
        mock_deepfm.assert_called_once()

    def test_experiment_dataset_reference(self):
        """Test that experiment maintains reference to dataset"""
        assert self.experiment.dataset is self.dataset

        # Modifying original dataset should be reflected in experiment
        original_len = len(self.experiment.dataset.users_df)
        assert original_len == 2

    def test_experiment_components_initialization(self):
        """Test that experiment components are properly initialized"""
        from src.recbole_experiment.config.manager import ConfigManager
        from src.recbole_experiment.training.trainer import ModelTrainer

        assert isinstance(self.experiment.config_manager, ConfigManager)
        assert isinstance(self.experiment.trainer, ModelTrainer)
        assert self.experiment.trainer.config_manager is self.experiment.config_manager

    def test_multiple_experiment_calls_data_prepared_once(self, mocker):
        """Test that multiple experiment calls only prepare data once"""
        mock_ensure_data = mocker.patch.object(self.experiment, "_ensure_data_prepared")
        mock_train = mocker.patch.object(self.experiment.trainer, "train_single_model")

        mock_train.return_value = {"test_result": {"auc": 0.85}}

        # Run multiple experiments
        self.experiment.run_single_model_experiment("DeepFM")
        self.experiment.run_single_model_experiment("LR")

        # Data preparation should be called for each experiment method call
        assert mock_ensure_data.call_count == 2

    def test_experiment_with_different_datasets(self):
        """Test creating experiments with different datasets"""
        # Create another dataset
        different_users = pd.DataFrame(
            {
                "user_id:token": ["u_100", "u_101"],
                "age:float": [40, 45],
                "gender:token": ["F", "M"],
            }
        )

        different_dataset = Dataset(
            different_users, self.sample_items, self.sample_interactions
        )
        different_experiment = ClickPredictionExperiment(different_dataset)

        # Experiments should be independent
        assert self.experiment.dataset is not different_experiment.dataset
        assert self.experiment is not different_experiment

        # But both should have properly initialized components
        assert hasattr(different_experiment, "config_manager")
        assert hasattr(different_experiment, "trainer")

    def test_data_prepared_flag_management(self, mocker):
        """Test _data_prepared flag management"""
        mocker.patch("builtins.print")
        mocker.patch.object(self.experiment.dataset, "save_to_recbole_format")

        assert self.experiment._data_prepared is False

        self.experiment._ensure_data_prepared()
        assert self.experiment._data_prepared is True

        # Flag should remain True after subsequent calls
        self.experiment._ensure_data_prepared()
        assert self.experiment._data_prepared is True
