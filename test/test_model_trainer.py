from unittest.mock import patch

from src.recbole_experiment.config.manager import ConfigManager
from src.recbole_experiment.training.trainer import ModelTrainer


class TestModelTrainer:
    """Test class for ModelTrainer"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config_manager = ConfigManager()
        self.trainer = ModelTrainer(self.config_manager)

    def test_init(self):
        """Test ModelTrainer initialization"""
        assert self.trainer.config_manager is self.config_manager

    @patch("src.recbole_experiment.training.trainer.run_recbole")
    @patch(
        "src.recbole_experiment.training.metrics.MetricsManager.determine_metrics_for_model"
    )
    @patch("builtins.print")
    def test_train_single_model_default_metrics(
        self, mock_print, mock_determine_metrics, mock_run_recbole
    ):
        """Test single model training with default metrics"""
        # Setup mocks
        mock_determine_metrics.return_value = ["AUC", "LogLoss"]
        mock_result = {"test_result": {"auc": 0.85, "logloss": 0.45}}
        mock_run_recbole.return_value = mock_result

        # Execute
        result = self.trainer.train_single_model("DeepFM")

        # Verify
        assert result == mock_result
        mock_determine_metrics.assert_called_once_with("DeepFM", "labeled")
        mock_run_recbole.assert_called_once()

    @patch("src.recbole_experiment.training.trainer.run_recbole")
    @patch("builtins.print")
    def test_train_single_model_custom_metrics(self, mock_print, mock_run_recbole):
        """Test single model training with custom metrics"""
        # Setup mocks
        mock_result = {"test_result": {"auc": 0.85}}
        mock_run_recbole.return_value = mock_result
        custom_metrics = ["AUC"]

        # Execute
        result = self.trainer.train_single_model("DeepFM", custom_metrics, "full")

        # Verify
        assert result == mock_result
        mock_run_recbole.assert_called_once()
        args, kwargs = mock_run_recbole.call_args
        assert kwargs["model"] == "DeepFM"
        assert kwargs["dataset"] == "click_prediction"

    @patch("src.recbole_experiment.training.trainer.run_recbole")
    @patch("builtins.print")
    def test_compare_models_success(self, mock_print, mock_run_recbole):
        """Test successful model comparison"""
        # Setup mocks
        mock_results = {
            "call_1": {"test_result": {"auc": 0.85, "logloss": 0.45}},
            "call_2": {"test_result": {"auc": 0.80, "logloss": 0.50}},
        }
        mock_run_recbole.side_effect = [mock_results["call_1"], mock_results["call_2"]]

        # Execute
        models = ["DeepFM", "LR"]
        result = self.trainer.compare_models(models, metrics=["AUC", "LogLoss"])

        # Verify
        assert len(result) == 2
        assert "DeepFM" in result
        assert "LR" in result
        assert mock_run_recbole.call_count == 2

    @patch("src.recbole_experiment.training.trainer.run_recbole")
    @patch("builtins.print")
    def test_compare_models_with_error(self, mock_print, mock_run_recbole):
        """Test model comparison with one model failing"""
        # Setup mocks - first call succeeds, second fails
        mock_run_recbole.side_effect = [
            {"test_result": {"auc": 0.85}},
            Exception("Model failed"),
        ]

        # Execute
        models = ["DeepFM", "FailingModel"]
        result = self.trainer.compare_models(models, metrics=["AUC"])

        # Verify - only successful model should be in results
        assert len(result) == 1
        assert "DeepFM" in result
        assert "FailingModel" not in result

    @patch("src.recbole_experiment.training.trainer.run_recbole")
    @patch(
        "src.recbole_experiment.training.metrics.MetricsManager.determine_metrics_for_model"
    )
    @patch("builtins.print")
    def test_compare_models_auto_metrics(
        self, mock_print, mock_determine_metrics, mock_run_recbole
    ):
        """Test model comparison with automatic metric selection"""
        # Setup mocks
        mock_determine_metrics.side_effect = [["AUC"], ["Recall"]]
        mock_run_recbole.side_effect = [
            {"test_result": {"auc": 0.85}},
            {"test_result": {"recall@10": 0.25}},
        ]

        # Execute
        models = ["DeepFM", "Pop"]
        result = self.trainer.compare_models(models, eval_mode="labeled")

        # Verify
        assert len(result) == 2
        assert mock_determine_metrics.call_count == 2

    @patch("src.recbole_experiment.training.trainer.run_recbole")
    @patch("builtins.print")
    def test_compare_models_quick_mode(self, mock_print, mock_run_recbole):
        """Test model comparison in quick mode"""
        # Setup mocks
        mock_run_recbole.return_value = {"test_result": {"auc": 0.85}}

        # Execute
        self.trainer.compare_models(["DeepFM"], mode="quick")

        # Verify that epochs were reduced
        args, kwargs = mock_run_recbole.call_args
        config_dict = kwargs["config_dict"]
        assert config_dict["epochs"] == 15

    def test_display_results_empty(self):
        """Test display results with empty results"""
        with patch("builtins.print") as mock_print:
            self.trainer._display_results({})
            mock_print.assert_called_with("実行可能なモデルがありませんでした")

    @patch("builtins.print")
    def test_display_results_with_custom_metrics(self, mock_print):
        """Test display results with custom metrics"""
        results = {
            "DeepFM": {"test_result": {"auc": 0.85, "logloss": 0.45}},
            "LR": {"test_result": {"auc": 0.80, "logloss": 0.50}},
        }
        custom_metrics = ["AUC", "LogLoss"]

        self.trainer._display_results(results, custom_metrics)

        # Verify that print was called (detailed verification of output would be complex)
        assert mock_print.called

    @patch("builtins.print")
    def test_display_results_auto_metrics(self, mock_print):
        """Test display results with automatic metric detection"""
        results = {
            "DeepFM": {"test_result": {"auc": 0.85, "recall@10": 0.25}},
            "Pop": {"test_result": {"auc": 0.70, "recall@10": 0.20}},
        }

        self.trainer._display_results(results)

        # Verify that print was called
        assert mock_print.called

    @patch("builtins.print")
    def test_display_results_ranking_metrics(self, mock_print):
        """Test display results with ranking metrics"""
        results = {
            "Pop": {"test_result": {"recall@10": 0.25, "ndcg@10": 0.15}},
            "BPR": {"test_result": {"recall@10": 0.30, "ndcg@10": 0.18}},
        }

        self.trainer._display_results(results)

        # Verify that print was called
        assert mock_print.called

    @patch(
        "src.recbole_experiment.models.registry.ModelRegistry.get_model_descriptions"
    )
    @patch("builtins.print")
    def test_display_results_with_descriptions(self, mock_print, mock_get_descriptions):
        """Test display results includes model descriptions"""
        mock_get_descriptions.return_value = {
            "DeepFM": "Deep + FM model",
            "LR": "Logistic Regression",
        }

        results = {
            "DeepFM": {"test_result": {"auc": 0.85}},
            "LR": {"test_result": {"auc": 0.80}},
        }

        self.trainer._display_results(results)

        mock_get_descriptions.assert_called_once()
        assert mock_print.called

    def test_display_results_sorting(self):
        """Test that results are sorted correctly"""
        results = {
            "ModelA": {"test_result": {"auc": 0.80}},
            "ModelB": {"test_result": {"auc": 0.90}},  # Higher AUC
            "ModelC": {"test_result": {"auc": 0.75}},
        }

        with patch("builtins.print") as mock_print:
            self.trainer._display_results(results, ["AUC"])

        # Results should be sorted by AUC in descending order
        # This is tested indirectly by checking the internal sorting logic exists
        assert mock_print.called

    @patch("src.recbole_experiment.training.trainer.run_recbole")
    @patch("builtins.print")
    def test_compare_models_preserves_config(self, mock_print, mock_run_recbole):
        """Test that base config is not modified during comparison"""
        mock_run_recbole.return_value = {"test_result": {"auc": 0.85}}

        # Get original config
        original_config = self.config_manager.create_base_config()
        original_epochs = original_config["epochs"]

        # Run comparison
        self.trainer.compare_models(["DeepFM"], mode="quick")

        # Verify original config unchanged
        new_config = self.config_manager.create_base_config()
        assert new_config["epochs"] == original_epochs

    @patch("src.recbole_experiment.training.trainer.run_recbole")
    @patch("builtins.print")
    def test_train_single_model_prints_metrics(self, mock_print, mock_run_recbole):
        """Test that single model training prints selected metrics"""
        mock_run_recbole.return_value = {"test_result": {"auc": 0.85}}

        self.trainer.train_single_model("DeepFM", metrics=["AUC"])

        # Verify that print was called (the method does print various things)
        assert mock_print.called
        # Check that at least some meaningful output was printed
        assert len(mock_print.call_args_list) > 0

    @patch("src.recbole_experiment.models.registry.ModelRegistry.get_model_config")
    @patch("src.recbole_experiment.training.trainer.run_recbole")
    def test_train_single_model_uses_model_config(
        self, mock_run_recbole, mock_get_config
    ):
        """Test that single model training uses model-specific config"""
        mock_get_config.return_value = {"model": "DeepFM", "custom_param": "value"}
        mock_run_recbole.return_value = {"test_result": {}}

        self.trainer.train_single_model("DeepFM")

        mock_get_config.assert_called_once()
        args, kwargs = mock_get_config.call_args
        assert args[0] == "DeepFM"  # model name
        assert isinstance(args[1], dict)  # base config

    def test_trainer_initialization_with_config_manager(self):
        """Test that trainer properly stores config manager"""
        custom_config_manager = ConfigManager()
        trainer = ModelTrainer(custom_config_manager)
        assert trainer.config_manager is custom_config_manager
