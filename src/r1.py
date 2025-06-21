# RecBoleを使ったアイテムクリック予測モデル

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.context_aware_recommender import DeepFM
from recbole.quick_start import run_recbole
from recbole.utils import init_seed

# PyTorch 2.6 compatibility fix
original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)


torch.load = patched_torch_load


class DataGenerator:
    """サンプルデータ生成クラス"""

    @staticmethod
    def create_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """サンプルデータの作成"""
        # ユーザーデータの作成
        n_users = 1000
        users_data = {
            "user_id:token": [f"u_{i}" for i in range(n_users)],
            "age:float": np.random.randint(18, 65, n_users),
            "gender:token": np.random.choice(["M", "F"], n_users),
        }
        users_df = pd.DataFrame(users_data)

        # アイテムデータの作成
        n_items = 500
        categories = ["Electronics", "Clothing", "Books", "Sports", "Home"]
        items_data = {
            "item_id:token": [f"i_{i}" for i in range(n_items)],
            "price:float": np.random.uniform(10, 1000, n_items),
            "rating:float": np.random.uniform(1, 5, n_items),
            "category:token": np.random.choice(categories, n_items),
        }
        items_df = pd.DataFrame(items_data)

        # インタラクションデータの作成（PVとクリック）
        n_interactions = 10000
        interactions_data = {
            "user_id:token": np.random.choice(
                users_df["user_id:token"], n_interactions
            ),
            "item_id:token": np.random.choice(
                items_df["item_id:token"], n_interactions
            ),
            "label:float": np.random.choice([0, 1], n_interactions, p=[0.8, 0.2]),
            "timestamp:float": np.random.randint(
                1000000000, 1700000000, n_interactions
            ),
        }
        interactions_df = pd.DataFrame(interactions_data)

        return users_df, items_df, interactions_df

    @staticmethod
    def prepare_recbole_data(
        users_df: pd.DataFrame,
        items_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        data_path: str = "dataset/click_prediction/",
    ) -> None:
        """RecBole用のデータファイル作成"""
        os.makedirs(data_path, exist_ok=True)

        # データファイルの保存
        users_df.to_csv(
            os.path.join(data_path, "click_prediction.user"), sep="\t", index=False
        )
        items_df.to_csv(
            os.path.join(data_path, "click_prediction.item"), sep="\t", index=False
        )
        interactions_df.to_csv(
            os.path.join(data_path, "click_prediction.inter"), sep="\t", index=False
        )

        print(f"データファイルを {data_path} に保存しました")


class ConfigManager:
    """設定管理クラス"""

    @staticmethod
    def create_base_config() -> Dict[str, Any]:
        """RecBoleの基本設定"""
        return {
            # データセット設定
            "dataset": "click_prediction",
            "data_path": "dataset/",
            # モデル設定
            "model": "DeepFM",
            "loss_type": "BCE",
            # 特徴量設定
            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": "item_id",
            "RATING_FIELD": "label",
            "TIME_FIELD": "timestamp",
            # ユーザー・アイテム特徴量
            "load_col": {
                "inter": ["user_id", "item_id", "label", "timestamp"],
                "user": ["user_id", "age", "gender"],
                "item": ["item_id", "price", "rating", "category"],
            },
            # 数値・カテゴリ特徴量の設定
            "numerical_features": ["age", "price", "rating"],
            "categorical_features": ["gender", "category"],
            # 学習設定
            "epochs": 30,
            "learning_rate": 0.001,
            "train_batch_size": 2048,
            "eval_batch_size": 2048,
            # 評価設定
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "group_by": None,
                "order": "TO",
                "mode": "labeled",
            },
            # 評価指標
            "metrics": ["AUC", "LogLoss", "MAE", "RMSE"],
            "metric_decimal_place": 4,
            # デフォルトモデル設定
            "embedding_size": 64,
            "mlp_hidden_size": [256, 128, 64],
            "dropout_prob": 0.2,
            # その他
            "seed": 2023,
            "reproducibility": True,
            "state": "INFO",
        }


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


class ModelTrainer:
    """モデル訓練・評価クラス"""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    def train_single_model(self, model_name: str = "DeepFM") -> Dict[str, Any]:
        """単一モデルの学習と評価"""
        print(f"=== {model_name} モデルの学習開始 ===")

        config_dict = ModelRegistry.get_model_config(
            model_name, self.config_manager.create_base_config()
        )

        result = run_recbole(
            model=model_name, dataset="click_prediction", config_dict=config_dict
        )

        print("=== 学習完了 ===")
        print(f"テスト結果: {result}")
        return result

    def compare_models(self, models: List[str], mode: str = "full") -> Dict[str, Any]:
        """複数モデルの比較"""
        results = {}
        base_config = self.config_manager.create_base_config()

        # クイックモードでは少ないエポック数で実行
        if mode == "quick":
            base_config["epochs"] = 15

        print(f"=== {len(models)} モデルの比較実行 ===")

        for model_name in models:
            print(f"\n--- {model_name} の学習 ---")

            try:
                config_dict = ModelRegistry.get_model_config(model_name, base_config)
                result = run_recbole(
                    model=model_name,
                    dataset="click_prediction",
                    config_dict=config_dict,
                )
                results[model_name] = result

                test_result = result.get("test_result", {})
                metrics = {
                    k: test_result.get(k, "N/A")
                    for k in ["auc", "logloss", "mae", "rmse"]
                }
                print(
                    f"{model_name} 完了: "
                    + ", ".join([f"{k.upper()}={v}" for k, v in metrics.items()])
                )

            except Exception as e:
                print(f"{model_name} でエラー: {str(e)}")
                continue

        self._display_results(results)
        return results

    def _display_results(self, results: Dict[str, Any]) -> None:
        """結果の表示"""
        if not results:
            print("実行可能なモデルがありませんでした")
            return

        print("\n=== モデル比較結果 ===")
        header = f"{'Model':<12} {'AUC':<8} {'LogLoss':<10} {'MAE':<10} {'RMSE':<10} {'Description':<50}"
        print(header)
        print("-" * len(header))

        # AUCでソート（降順）
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].get("test_result", {}).get("auc", 0),
            reverse=True,
        )

        descriptions = ModelRegistry.get_model_descriptions()

        for model_name, result in sorted_results:
            test_result = result.get("test_result", {})
            metrics = {
                "auc": test_result.get("auc", "N/A"),
                "logloss": test_result.get("logloss", "N/A"),
                "mae": test_result.get("mae", "N/A"),
                "rmse": test_result.get("rmse", "N/A"),
            }

            formatted_metrics = {
                k: f"{v:.4f}" if isinstance(v, float) else str(v)
                for k, v in metrics.items()
            }

            description = descriptions.get(model_name, "")
            print(
                f"{model_name:<12} "
                f"{formatted_metrics['auc']:<8} "
                f"{formatted_metrics['logloss']:<10} "
                f"{formatted_metrics['mae']:<10} "
                f"{formatted_metrics['rmse']:<10} "
                f"{description:<50}"
            )


class ClickPredictionExperiment:
    """クリック予測実験の統合クラス"""

    def __init__(self):
        self.data_generator = DataGenerator()
        self.config_manager = ConfigManager()
        self.trainer = ModelTrainer(self.config_manager)

    def setup_data(self) -> None:
        """データ準備"""
        print("=== データ準備 ===")
        users_df, items_df, interactions_df = self.data_generator.create_sample_data()
        self.data_generator.prepare_recbole_data(users_df, items_df, interactions_df)

    def run_single_model_experiment(self, model_name: str = "DeepFM") -> Dict[str, Any]:
        """単一モデル実験"""
        self.setup_data()
        return self.trainer.train_single_model(model_name)

    def run_quick_comparison(self) -> Dict[str, Any]:
        """主要モデルの高速比較"""
        print("=== 主要モデルの高速比較 ===")
        self.setup_data()
        models = ModelRegistry.get_quick_models()
        return self.trainer.compare_models(models, mode="quick")

    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """全モデルの包括的比較"""
        print("=== 全モデルの包括的比較 ===")
        self.setup_data()
        models = ModelRegistry.get_all_models()
        return self.trainer.compare_models(models, mode="full")

    def predict_click_probability(self) -> None:
        """クリック確率予測の例（概念的な実装）"""
        print("=== クリック確率予測の例 ===")
        config_dict = self.config_manager.create_base_config()

        # データセットとモデルの準備
        config = Config(
            model="DeepFM", dataset="click_prediction", config_dict=config_dict
        )
        init_seed(config["seed"], config["reproducibility"])

        # データセットの作成
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        # モデルの初期化
        model = DeepFM(config, train_data.dataset).to(config["device"])

        print("モデルの構造:")
        print(model)
        print("\n注意: 実際の予測には学習済みモデルが必要です")


def main():
    """メイン実行関数"""
    print("RecBoleを使ったアイテムクリック予測モデル")
    print("=" * 50)

    experiment = ClickPredictionExperiment()

    # 基本的な学習と評価
    experiment.run_single_model_experiment("DeepFM")

    # モデル比較の選択
    print("\n" + "=" * 50)
    print("モデル比較を実行します...")

    # 主要モデルの高速比較（代表的な9モデル）
    experiment.run_quick_comparison()

    # 詳細比較も実行する場合（全32+モデル - 時間がかかるのでコメントアウト推奨）
    # comprehensive_results = experiment.run_comprehensive_comparison()

    # 予測の準備（概念的な例）
    # experiment.predict_click_probability()


if __name__ == "__main__":
    main()
