import pytest

from src.recbole_experiment.models.registry import ModelRegistry


class TestModelRegistry:
    """Test class for ModelRegistry"""

    def test_context_aware_models_list(self):
        """Test context-aware models list"""
        expected_models = [
            "LR",
            "FM",
            "FFM",
            "FNN",
            "DeepFM",
            "NFM",
            "AFM",
            "PNN",
            "WideDeep",
            "DCN",
            "DCNV2",
            "xDeepFM",
            "AutoInt",
            "FwFM",
            "FiGNN",
            "DIN",
            "DIEN",
            "DSSM",
        ]
        assert ModelRegistry.CONTEXT_AWARE_MODELS == expected_models
        assert len(ModelRegistry.CONTEXT_AWARE_MODELS) == 18

    def test_general_models_list(self):
        """Test general models list"""
        expected_models = ["Pop", "ItemKNN", "BPR", "NeuMF", "LightGCN", "NGCF", "DGCF"]
        assert ModelRegistry.GENERAL_MODELS == expected_models
        assert len(ModelRegistry.GENERAL_MODELS) == 7

    def test_sequential_models_list(self):
        """Test sequential models list"""
        expected_models = ["GRU4Rec", "SASRec", "BERT4Rec", "Caser", "NARM"]
        assert ModelRegistry.SEQUENTIAL_MODELS == expected_models
        assert len(ModelRegistry.SEQUENTIAL_MODELS) == 5

    def test_quick_models_list(self):
        """Test quick models list"""
        expected_models = [
            "LR",
            "FM",
            "DeepFM",
            "WideDeep",
            "DCN",
            "AutoInt",
            "Pop",
            "BPR",
            "SASRec",
        ]
        assert ModelRegistry.QUICK_MODELS == expected_models
        assert len(ModelRegistry.QUICK_MODELS) == 9

    def test_get_all_models(self):
        """Test get_all_models method"""
        all_models = ModelRegistry.get_all_models()
        expected_total = 18 + 7 + 5  # 30 models total
        assert len(all_models) == expected_total

        # Check that all category models are included
        for model in ModelRegistry.CONTEXT_AWARE_MODELS:
            assert model in all_models
        for model in ModelRegistry.GENERAL_MODELS:
            assert model in all_models
        for model in ModelRegistry.SEQUENTIAL_MODELS:
            assert model in all_models

    def test_get_quick_models(self):
        """Test get_quick_models method"""
        quick_models = ModelRegistry.get_quick_models()
        assert quick_models == ModelRegistry.QUICK_MODELS
        assert isinstance(quick_models, list)

    def test_get_context_aware_models(self):
        """Test get_context_aware_models method"""
        context_models = ModelRegistry.get_context_aware_models()
        assert context_models == ModelRegistry.CONTEXT_AWARE_MODELS
        assert isinstance(context_models, list)

    def test_get_general_models(self):
        """Test get_general_models method"""
        general_models = ModelRegistry.get_general_models()
        assert general_models == ModelRegistry.GENERAL_MODELS
        assert isinstance(general_models, list)

    def test_get_sequential_models(self):
        """Test get_sequential_models method"""
        sequential_models = ModelRegistry.get_sequential_models()
        assert sequential_models == ModelRegistry.SEQUENTIAL_MODELS
        assert isinstance(sequential_models, list)

    def test_get_model_config_basic(self):
        """Test basic model config generation"""
        base_config = {"epochs": 30, "learning_rate": 0.001}
        config = ModelRegistry.get_model_config("DeepFM", base_config)

        assert config["model"] == "DeepFM"
        assert config["epochs"] == 30  # from base config
        assert config["learning_rate"] == 0.001  # from base config

    def test_get_model_config_lr_specific(self):
        """Test LR specific configuration"""
        base_config = {"epochs": 30}
        config = ModelRegistry.get_model_config("LR", base_config)

        assert config["model"] == "LR"
        assert config["epochs"] == 20  # LR specific override

    def test_get_model_config_fm_specific(self):
        """Test FM specific configuration"""
        base_config = {"embedding_size": 64}
        config = ModelRegistry.get_model_config("FM", base_config)

        assert config["model"] == "FM"
        assert config["embedding_size"] == 32  # FM specific override

    def test_get_model_config_deepfm_default(self):
        """Test DeepFM uses base config when no specific config"""
        base_config = {"epochs": 30, "embedding_size": 64}
        config = ModelRegistry.get_model_config("DeepFM", base_config)

        assert config["model"] == "DeepFM"
        assert config["epochs"] == 30  # no override for DeepFM
        assert config["embedding_size"] == 64  # no override for DeepFM

    def test_get_model_config_dcn_specific(self):
        """Test DCN specific configuration"""
        base_config = {}
        config = ModelRegistry.get_model_config("DCN", base_config)

        assert config["model"] == "DCN"
        assert config["cross_layer_num"] == 3
        assert config["mlp_hidden_size"] == [128, 64]

    def test_get_model_config_sasrec_specific(self):
        """Test SASRec specific configuration"""
        base_config = {}
        config = ModelRegistry.get_model_config("SASRec", base_config)

        assert config["model"] == "SASRec"
        assert config["n_layers"] == 2
        assert config["n_heads"] == 2
        assert config["hidden_size"] == 64
        assert config["inner_size"] == 256
        assert config["dropout_prob"] == 0.2

    def test_get_model_config_unknown_model(self):
        """Test configuration for unknown model"""
        base_config = {"epochs": 30}
        config = ModelRegistry.get_model_config("UnknownModel", base_config)

        assert config["model"] == "UnknownModel"
        assert config["epochs"] == 30  # should preserve base config

    def test_get_model_config_preserves_base_config(self):
        """Test that original base_config is not modified"""
        base_config = {"epochs": 30, "learning_rate": 0.001}
        original_base = base_config.copy()

        ModelRegistry.get_model_config("LR", base_config)

        # Original base_config should be unchanged
        assert base_config == original_base

    def test_get_model_descriptions(self):
        """Test model descriptions"""
        descriptions = ModelRegistry.get_model_descriptions()

        assert isinstance(descriptions, dict)

        # Check some specific descriptions
        assert "DeepFM" in descriptions
        assert "Deep + FM" in descriptions["DeepFM"]

        assert "Pop" in descriptions
        assert "人気度" in descriptions["Pop"]

        assert "SASRec" in descriptions
        assert "Self-Attention" in descriptions["SASRec"]

    def test_model_descriptions_completeness(self):
        """Test that all models have descriptions"""
        descriptions = ModelRegistry.get_model_descriptions()
        all_models = ModelRegistry.get_all_models()

        for model in all_models:
            assert model in descriptions, f"Model {model} missing description"
            assert descriptions[model].strip() != "", (
                f"Model {model} has empty description"
            )

    def test_no_duplicate_models_across_categories(self):
        """Test that no model appears in multiple categories"""
        context_set = set(ModelRegistry.CONTEXT_AWARE_MODELS)
        general_set = set(ModelRegistry.GENERAL_MODELS)
        sequential_set = set(ModelRegistry.SEQUENTIAL_MODELS)

        # No overlaps between categories
        assert len(context_set.intersection(general_set)) == 0
        assert len(context_set.intersection(sequential_set)) == 0
        assert len(general_set.intersection(sequential_set)) == 0

    def test_quick_models_subset_of_all_models(self):
        """Test that quick models are subset of all models"""
        quick_models = set(ModelRegistry.QUICK_MODELS)
        all_models = set(ModelRegistry.get_all_models())

        assert quick_models.issubset(all_models)

    def test_quick_models_represent_all_categories(self):
        """Test that quick models include representatives from all categories"""
        quick_models = set(ModelRegistry.QUICK_MODELS)

        # Should have context-aware models
        context_in_quick = quick_models.intersection(
            set(ModelRegistry.CONTEXT_AWARE_MODELS)
        )
        assert len(context_in_quick) > 0

        # Should have general models
        general_in_quick = quick_models.intersection(set(ModelRegistry.GENERAL_MODELS))
        assert len(general_in_quick) > 0

        # Should have sequential models
        sequential_in_quick = quick_models.intersection(
            set(ModelRegistry.SEQUENTIAL_MODELS)
        )
        assert len(sequential_in_quick) > 0

    @pytest.mark.parametrize(
        "model_name", ["LR", "FM", "DeepFM", "Pop", "BPR", "SASRec"]
    )
    def test_specific_model_configs(self, model_name):
        """Test specific model configurations"""
        base_config = {"default_param": "default_value"}
        config = ModelRegistry.get_model_config(model_name, base_config)

        assert config["model"] == model_name
        assert "default_param" in config  # base config preserved

    def test_model_config_types(self):
        """Test that model configs have correct types"""
        base_config = {}

        # Test numeric configs
        lr_config = ModelRegistry.get_model_config("LR", base_config)
        if "epochs" in lr_config:
            assert isinstance(lr_config["epochs"], int)

        # Test list configs
        dcn_config = ModelRegistry.get_model_config("DCN", base_config)
        if "mlp_hidden_size" in dcn_config:
            assert isinstance(dcn_config["mlp_hidden_size"], list)

        # Test boolean configs
        pnn_config = ModelRegistry.get_model_config("PNN", base_config)
        if "use_inner" in pnn_config:
            assert isinstance(pnn_config["use_inner"], bool)
