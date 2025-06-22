"""Model registry and configuration management."""

from typing import Any, Dict, List


class ModelRegistry:
    """モデル登録・設定管理クラス"""

    # モデルカテゴリ定義
    CONTEXT_AWARE_MODELS = [
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

    GENERAL_MODELS = ["Pop", "ItemKNN", "BPR", "NeuMF", "LightGCN", "NGCF", "DGCF"]

    SEQUENTIAL_MODELS = ["GRU4Rec", "SASRec", "BERT4Rec", "Caser", "NARM"]

    # 代表的なモデル
    QUICK_MODELS = [
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

    @classmethod
    def get_all_models(cls) -> List[str]:
        """全モデルのリストを取得"""
        return cls.CONTEXT_AWARE_MODELS + cls.GENERAL_MODELS + cls.SEQUENTIAL_MODELS

    @classmethod
    def get_quick_models(cls) -> List[str]:
        """代表的なモデルのリストを取得"""
        return cls.QUICK_MODELS

    @classmethod
    def get_context_aware_models(cls) -> List[str]:
        """コンテキスト対応モデルのリストを取得"""
        return cls.CONTEXT_AWARE_MODELS

    @classmethod
    def get_general_models(cls) -> List[str]:
        """一般的な推薦モデルのリストを取得"""
        return cls.GENERAL_MODELS

    @classmethod
    def get_sequential_models(cls) -> List[str]:
        """系列推薦モデルのリストを取得"""
        return cls.SEQUENTIAL_MODELS

    @staticmethod
    def get_model_config(
        model_name: str, base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """モデル固有の設定を適用"""
        config = base_config.copy()
        config["model"] = model_name

        # モデル固有設定の適用
        model_configs = {
            # Context-aware models
            "LR": {"epochs": 20},
            "FM": {"embedding_size": 32},
            "FFM": {"embedding_size": 32},
            "FNN": {"mlp_hidden_size": [128, 64], "dropout_prob": 0.2},
            "NFM": {"mlp_hidden_size": [128, 64], "dropout_prob": 0.2},
            "AFM": {"attention_size": 32, "dropout_prob": 0.2},
            "PNN": {
                "use_inner": True,
                "use_outer": False,
                "mlp_hidden_size": [128, 64],
            },
            "WideDeep": {"mlp_hidden_size": [128, 64]},
            "DCN": {"cross_layer_num": 3, "mlp_hidden_size": [128, 64]},
            "DCNV2": {"cross_layer_num": 3, "mlp_hidden_size": [128, 64]},
            "xDeepFM": {
                "cin_layer_size": [128, 128],
                "direct": True,
                "mlp_hidden_size": [128, 64],
            },
            "AutoInt": {"attention_layers": 3, "num_heads": 2},
            "FwFM": {"embedding_size": 32},
            "FiGNN": {"num_layers": 3, "embedding_size": 32},
            "DIN": {"mlp_hidden_size": [128, 64], "attention_mlp_layers": [64, 32]},
            "DIEN": {"mlp_hidden_size": [128, 64], "gru_hidden_size": 64},
            "DSSM": {"mlp_hidden_size": [128, 64]},
            # General recommender models
            "Pop": {"epochs": 1},
            "ItemKNN": {"k": 50},
            "BPR": {"embedding_size": 64},
            "NeuMF": {
                "mf_embedding_size": 32,
                "mlp_embedding_size": 32,
                "mlp_hidden_size": [128, 64, 32],
            },
            "LightGCN": {"n_layers": 3, "embedding_size": 64},
            "NGCF": {
                "n_layers": 3,
                "embedding_size": 64,
                "node_dropout": 0.1,
                "message_dropout": 0.1,
            },
            "DGCF": {"n_layers": 3, "embedding_size": 64, "n_factors": 4},
            # Sequential models
            "GRU4Rec": {"embedding_size": 64, "hidden_size": 128, "num_layers": 1},
            "SASRec": {
                "n_layers": 2,
                "n_heads": 2,
                "hidden_size": 64,
                "inner_size": 256,
                "dropout_prob": 0.2,
            },
            "BERT4Rec": {
                "n_layers": 2,
                "n_heads": 2,
                "hidden_size": 64,
                "inner_size": 256,
                "dropout_prob": 0.2,
            },
            "Caser": {"embedding_size": 64, "nv": 8, "nh": 16, "dropout_prob": 0.2},
            "NARM": {"embedding_size": 64, "hidden_size": 128},
        }

        config.update(model_configs.get(model_name, {}))
        return config

    @staticmethod
    def get_model_descriptions() -> Dict[str, str]:
        """モデルの説明を取得"""
        return {
            # Context-aware models
            "LR": "Logistic Regression (線形ベースライン)",
            "FM": "Factorization Machines (特徴量交互作用)",
            "FFM": "Field-aware FM (フィールド別特徴量交互作用)",
            "FNN": "Factorization NN (FM初期化+深層学習)",
            "DeepFM": "Deep + FM (深層学習とFMの組み合わせ)",
            "NFM": "Neural FM (高次特徴量交互作用)",
            "AFM": "Attentional FM (アテンション機構付きFM)",
            "PNN": "Product-based NN (積ベース深層学習)",
            "WideDeep": "Wide & Deep (記憶と汎化の両立)",
            "DCN": "Deep & Cross Network (自動特徴量クロス)",
            "DCNV2": "DCN V2 (改良版Deep & Cross)",
            "xDeepFM": "eXtreme DeepFM (明示的・暗黙的交互作用)",
            "AutoInt": "AutoInt (自動特徴量交互作用学習)",
            "FwFM": "Field-weighted FM (フィールド重み付きFM)",
            "FiGNN": "Field-matrixed FM + GNN (グラフニューラル)",
            "DIN": "Deep Interest Network (動的注意機構)",
            "DIEN": "Deep Interest Evolution Network (興味進化)",
            "DSSM": "Deep Structured Semantic Model (意味マッチング)",
            # General recommender models
            "Pop": "Popularity (人気度ベースライン)",
            "ItemKNN": "Item-based KNN (アイテム協調フィルタリング)",
            "BPR": "Bayesian Personalized Ranking (ペアワイズ学習)",
            "NeuMF": "Neural Matrix Factorization (深層学習MF)",
            "LightGCN": "Light Graph Convolutional Network (軽量GCN)",
            "NGCF": "Neural Graph Collaborative Filtering (グラフ協調)",
            "DGCF": "Disentangled Graph Collaborative Filtering (分離グラフ)",
            # Sequential models
            "GRU4Rec": "GRU for Recommendation (RNN系列推薦)",
            "SASRec": "Self-Attention Sequential Rec (自己注意系列)",
            "BERT4Rec": "BERT for Recommendation (BERT系列推薦)",
            "Caser": "Convolutional Sequence Embedding (畳み込み系列)",
            "NARM": "Neural Attention Recommendation (注意系列)",
        }
