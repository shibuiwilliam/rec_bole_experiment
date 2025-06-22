import pytest

from src.recbole_experiment.config.manager import ConfigManager
from src.recbole_experiment.training.metrics import MetricsManager


class TestConfigManager:
    """Test class for ConfigManager"""

    def test_create_base_config_default(self):
        """Test base config creation with defaults"""
        config = ConfigManager.create_base_config()

        # Check basic structure
        assert isinstance(config, dict)
        assert config["dataset"] == "click_prediction"
        assert config["data_path"] == "dataset/"
        assert config["model"] == "DeepFM"
        assert config["loss_type"] == "BCE"

    def test_create_base_config_field_mappings(self):
        """Test field mappings in base config"""
        config = ConfigManager.create_base_config()

        assert config["USER_ID_FIELD"] == "user_id"
        assert config["ITEM_ID_FIELD"] == "item_id"
        assert config["RATING_FIELD"] == "label"
        assert config["TIME_FIELD"] == "timestamp"

    def test_create_base_config_data_loading(self):
        """Test data loading configuration"""
        config = ConfigManager.create_base_config()

        expected_load_col = {
            "inter": ["user_id", "item_id", "label", "timestamp"],
            "user": ["user_id", "age", "gender"],
            "item": ["item_id", "price", "rating", "category"],
        }
        assert config["load_col"] == expected_load_col

    def test_create_base_config_features(self):
        """Test feature configuration"""
        config = ConfigManager.create_base_config()

        assert config["numerical_features"] == ["age", "price", "rating"]
        assert config["categorical_features"] == ["gender", "category"]

    def test_create_base_config_training_params(self):
        """Test training parameters"""
        config = ConfigManager.create_base_config()

        assert config["epochs"] == 30
        assert config["learning_rate"] == 0.001
        assert config["train_batch_size"] == 2048
        assert config["eval_batch_size"] == 2048
        assert config["stopping_step"] == 5
        assert config["eval_step"] == 5

    def test_create_base_config_model_params(self):
        """Test model parameters"""
        config = ConfigManager.create_base_config()

        assert config["embedding_size"] == 64
        assert config["mlp_hidden_size"] == [256, 128, 64]
        assert config["dropout_prob"] == 0.2

    def test_create_base_config_reproducibility(self):
        """Test reproducibility settings"""
        config = ConfigManager.create_base_config()

        assert config["seed"] == 2023
        assert config["reproducibility"] is True
        assert config["state"] == "INFO"

    def test_create_base_config_with_custom_metrics(self):
        """Test config creation with custom metrics"""
        custom_metrics = ["AUC", "LogLoss"]
        config = ConfigManager.create_base_config(metrics=custom_metrics)

        assert config["metrics"] == custom_metrics
        assert config["metric_decimal_place"] == 4

    def test_create_base_config_labeled_eval_mode(self):
        """Test config with labeled evaluation mode"""
        config = ConfigManager.create_base_config(eval_mode="labeled")

        assert config["eval_args"]["mode"] == "labeled"
        assert config["eval_args"]["split"] == {"RS": [0.8, 0.1, 0.1]}
        assert config["eval_args"]["group_by"] is None
        assert config["eval_args"]["order"] == "TO"

    def test_create_base_config_full_eval_mode(self):
        """Test config with full evaluation mode"""
        config = ConfigManager.create_base_config(eval_mode="full")

        assert config["eval_args"]["mode"] == "full"

    def test_create_base_config_ranking_metrics_adds_topk(self):
        """Test that ranking metrics add topk configuration"""
        ranking_metrics = ["Recall", "NDCG"]
        config = ConfigManager.create_base_config(metrics=ranking_metrics)

        assert "topk" in config
        assert config["topk"] == [10, 20]

    def test_create_base_config_value_metrics_no_topk(self):
        """Test that value metrics don't add topk configuration"""
        value_metrics = ["AUC", "LogLoss"]
        config = ConfigManager.create_base_config(metrics=value_metrics)

        # topk should not be present for value metrics only
        assert "topk" not in config

    def test_create_base_config_mixed_metrics_adds_topk(self):
        """Test that mixed metrics add topk configuration"""
        mixed_metrics = ["AUC", "Recall", "LogLoss"]
        config = ConfigManager.create_base_config(metrics=mixed_metrics)

        assert "topk" in config
        assert config["topk"] == [10, 20]

    def test_create_base_config_ranking_metrics_adds_neg_sampling(self):
        """Test that ranking metrics add negative sampling configuration"""
        ranking_metrics = ["Recall", "NDCG"]
        config = ConfigManager.create_base_config(metrics=ranking_metrics)

        assert "train_neg_sample_args" in config
        expected_neg_args = {
            "distribution": "uniform",
            "sample_num": 1,
            "alpha": 1.0,
            "dynamic": False,
            "candidate_num": 0,
        }
        assert config["train_neg_sample_args"] == expected_neg_args

    def test_create_base_config_value_metrics_no_neg_sampling(self):
        """Test that value metrics don't add negative sampling configuration"""
        value_metrics = ["AUC", "LogLoss"]
        config = ConfigManager.create_base_config(metrics=value_metrics)

        # neg sampling should not be present for value metrics only
        assert "train_neg_sample_args" not in config

    def test_create_base_config_default_metrics(self):
        """Test that default metrics are value metrics"""
        config = ConfigManager.create_base_config()

        assert config["metrics"] == MetricsManager.VALUE_METRICS

    @pytest.mark.parametrize("eval_mode", ["labeled", "full", "invalid"])
    def test_create_base_config_eval_mode_mapping(self, eval_mode):
        """Test eval_mode mapping to config format"""
        config = ConfigManager.create_base_config(eval_mode=eval_mode)

        if eval_mode == "labeled":
            assert config["eval_args"]["mode"] == "labeled"
        elif eval_mode == "full":
            assert config["eval_args"]["mode"] == "full"
        else:
            # Invalid eval_mode should default to "full"
            assert config["eval_args"]["mode"] == "full"

    def test_create_base_config_immutability(self):
        """Test that modifying returned config doesn't affect subsequent calls"""
        config1 = ConfigManager.create_base_config()
        config1["epochs"] = 999
        config1["new_key"] = "new_value"

        config2 = ConfigManager.create_base_config()

        # Second config should not be affected by first config modifications
        assert config2["epochs"] == 30
        assert "new_key" not in config2

    def test_create_base_config_all_ranking_metrics(self):
        """Test config with all ranking metrics"""
        all_ranking = MetricsManager.get_ranking_metrics()
        config = ConfigManager.create_base_config(metrics=all_ranking)

        assert config["metrics"] == all_ranking
        assert "topk" in config
        assert "train_neg_sample_args" in config

    def test_create_base_config_all_value_metrics(self):
        """Test config with all value metrics"""
        all_value = MetricsManager.get_value_metrics()
        config = ConfigManager.create_base_config(metrics=all_value)

        assert config["metrics"] == all_value
        assert "topk" not in config
        assert "train_neg_sample_args" not in config

    def test_create_base_config_empty_metrics_list(self):
        """Test config with empty metrics list"""
        config = ConfigManager.create_base_config(metrics=[])

        assert config["metrics"] == []
        assert "topk" not in config
        assert "train_neg_sample_args" not in config

    def test_create_base_config_none_metrics(self):
        """Test config with None metrics defaults to value metrics"""
        config = ConfigManager.create_base_config(metrics=None)

        assert config["metrics"] == MetricsManager.VALUE_METRICS

    def test_config_data_split_settings(self):
        """Test data split configuration"""
        config = ConfigManager.create_base_config()

        assert config["eval_args"]["split"] == {"RS": [0.8, 0.1, 0.1]}
        assert config["eval_args"]["group_by"] is None
        assert config["eval_args"]["order"] == "TO"
