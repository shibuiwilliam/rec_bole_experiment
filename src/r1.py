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
    def create_sample_data(
        eval_mode: str = "labeled",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

        # インタラクションデータの作成（ランキング評価も考慮した形式）
        n_interactions = 10000

        # ユーザーごとに少なくとも数個のインタラクションを持つようにする
        interactions_data = {
            "user_id:token": [],
            "item_id:token": [],
            "label:float": [],
            "timestamp:float": [],
        }

        # 評価モードに応じてデータ形式を調整
        if eval_mode == "full":
            # ランキング評価用：各ユーザーに正例のみ
            for user_id in users_df["user_id:token"]:
                n_user_interactions = np.random.randint(5, 15)  # ユーザーあたり5-15個
                if n_user_interactions > len(items_df):
                    n_user_interactions = len(items_df) // 2
                user_items = np.random.choice(
                    items_df["item_id:token"], n_user_interactions, replace=False
                )
                user_ratings = np.ones(n_user_interactions)  # 全て正例（1）
                user_timestamps = np.random.randint(
                    1000000000, 1700000000, n_user_interactions
                )

                interactions_data["user_id:token"].extend(
                    [user_id] * n_user_interactions
                )
                interactions_data["item_id:token"].extend(user_items)
                interactions_data["label:float"].extend(user_ratings)
                interactions_data["timestamp:float"].extend(user_timestamps)
        else:
            # CTR予測用：正例・負例の混在
            users_sample = np.random.choice(users_df["user_id:token"], n_interactions)
            items_sample = np.random.choice(items_df["item_id:token"], n_interactions)
            labels = np.random.choice([0, 1], n_interactions, p=[0.8, 0.2])
            timestamps = np.random.randint(1000000000, 1700000000, n_interactions)

            interactions_data["user_id:token"] = users_sample.tolist()
            interactions_data["item_id:token"] = items_sample.tolist()
            interactions_data["label:float"] = labels.tolist()
            interactions_data["timestamp:float"] = timestamps.tolist()
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


class MetricsManager:
    """評価指標管理クラス"""

    # 評価指標の定義
    RANKING_METRICS = ["Recall", "MRR", "NDCG", "Hit", "Precision"]
    VALUE_METRICS = ["AUC", "LogLoss", "MAE", "RMSE"]

    @classmethod
    def get_ranking_metrics(cls) -> List[str]:
        """ランキング指標を取得"""
        return cls.RANKING_METRICS

    @classmethod
    def get_value_metrics(cls) -> List[str]:
        """値ベース指標を取得"""
        return cls.VALUE_METRICS

    @classmethod
    def get_all_metrics(cls) -> List[str]:
        """全指標を取得"""
        return cls.RANKING_METRICS + cls.VALUE_METRICS

    @staticmethod
    def determine_metrics_for_model(
        model_name: str, eval_mode: str = "labeled"
    ) -> List[str]:
        """モデルと評価モードに基づいて適切な指標を決定"""
        context_aware_models = ModelRegistry.CONTEXT_AWARE_MODELS

        # Context-aware modelsは通常value metricsを使用
        if model_name in context_aware_models:
            return MetricsManager.VALUE_METRICS

        # General/Sequential modelsの場合、評価モードに依存
        if eval_mode == "labeled":
            return MetricsManager.VALUE_METRICS
        else:
            return MetricsManager.RANKING_METRICS


class ConfigManager:
    """設定管理クラス"""

    @staticmethod
    def create_base_config(
        metrics: List[str] = None, eval_mode: str = "labeled"
    ) -> Dict[str, Any]:
        """RecBoleの基本設定"""
        # デフォルトの指標設定
        if metrics is None:
            metrics = MetricsManager.VALUE_METRICS

        # 評価モードの設定
        eval_mode_config = "labeled" if eval_mode == "labeled" else "full"

        config = {
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
            "stopping_step": 5,
            "eval_step": 5,
            # 評価設定
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "group_by": None,
                "order": "TO",
                "mode": eval_mode_config,
            },
            # 評価指標（動的設定）
            "metrics": metrics,
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

        # ランキング指標の場合はtopkを追加
        if any(metric in MetricsManager.RANKING_METRICS for metric in metrics):
            config["topk"] = [10, 20]

        # ランキング指標の場合は負サンプリング設定を調整
        if any(metric in MetricsManager.RANKING_METRICS for metric in metrics):
            config["train_neg_sample_args"] = {
                "distribution": "uniform",
                "sample_num": 1,
                "alpha": 1.0,
                "dynamic": False,
                "candidate_num": 0,
            }

        return config


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


class ModelTrainer:
    """モデル訓練・評価クラス"""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    def train_single_model(
        self,
        model_name: str = "DeepFM",
        metrics: List[str] = None,
        eval_mode: str = "labeled",
    ) -> Dict[str, Any]:
        """単一モデルの学習と評価"""
        print(f"=== {model_name} モデルの学習開始 ===")

        # 指標が指定されていない場合は自動選択
        if metrics is None:
            metrics = MetricsManager.determine_metrics_for_model(model_name, eval_mode)
            print(f"使用する評価指標: {metrics}")

        config_dict = ModelRegistry.get_model_config(
            model_name, self.config_manager.create_base_config(metrics, eval_mode)
        )

        result = run_recbole(
            model=model_name, dataset="click_prediction", config_dict=config_dict
        )

        print("=== 学習完了 ===")
        print(f"テスト結果: {result}")
        return result

    def compare_models(
        self,
        models: List[str],
        mode: str = "full",
        metrics: List[str] = None,
        eval_mode: str = "labeled",
    ) -> Dict[str, Any]:
        """複数モデルの比較"""
        results = {}

        print(f"=== {len(models)} モデルの比較実行 ===")

        for model_name in models:
            print(f"\n--- {model_name} の学習 ---")

            try:
                # モデルごとに適切な指標を選択
                model_metrics = (
                    metrics
                    if metrics
                    else MetricsManager.determine_metrics_for_model(
                        model_name, eval_mode
                    )
                )

                base_config = self.config_manager.create_base_config(
                    model_metrics, eval_mode
                )

                # クイックモードでは少ないエポック数で実行
                if mode == "quick":
                    base_config["epochs"] = 15

                config_dict = ModelRegistry.get_model_config(model_name, base_config)
                result = run_recbole(
                    model=model_name,
                    dataset="click_prediction",
                    config_dict=config_dict,
                )
                results[model_name] = result

                test_result = result.get("test_result", {})
                # 動的に使用されている指標を表示（@topk付きも考慮）
                metrics_display = {}
                for metric in model_metrics:
                    metric_lower = metric.lower()
                    # 直接一致を探す
                    if metric_lower in test_result:
                        metrics_display[metric_lower] = test_result[metric_lower]
                    else:
                        # @topk付きの指標を探す（例: recall@10, ndcg@10）
                        for key, value in test_result.items():
                            if key.startswith(metric_lower + "@"):
                                metrics_display[key] = value
                                break
                        else:
                            metrics_display[metric_lower] = "N/A"
                print(
                    f"{model_name} 完了: "
                    + ", ".join(
                        [f"{k.upper()}={v}" for k, v in metrics_display.items()]
                    )
                )

            except Exception as e:
                print(f"{model_name} でエラー: {str(e)}")
                continue

        self._display_results(results, metrics)
        return results

    def _display_results(
        self, results: Dict[str, Any], custom_metrics: List[str] = None
    ) -> None:
        """結果の表示"""
        if not results:
            print("実行可能なモデルがありませんでした")
            return

        # 表示する指標を決定
        if custom_metrics:
            # カスタム指標の場合、@topk付きも考慮
            display_metrics = []
            for metric in custom_metrics:
                metric_lower = metric.lower()
                display_metrics.append(metric_lower)
        else:
            # 結果から利用可能な指標を抽出
            all_metrics = set()
            for result in results.values():
                test_result = result.get("test_result", {})
                all_metrics.update(test_result.keys())

            # 優先順位に従って表示指標を選択（@topk付きも考慮）
            priority_metrics = [
                "auc",
                "recall@10",
                "ndcg@10",
                "mrr@10",
                "precision@10",
                "hit@10",
                "logloss",
                "mae",
                "rmse",
            ]
            display_metrics = []
            for metric in priority_metrics:
                if metric in all_metrics:
                    display_metrics.append(metric)
                elif any(m.startswith(metric.split("@")[0] + "@") for m in all_metrics):
                    # @topk付きの指標がある場合は最初に見つかったものを使用
                    for m in all_metrics:
                        if m.startswith(metric.split("@")[0] + "@"):
                            display_metrics.append(m)
                            break
                if len(display_metrics) >= 4:
                    break

        print("\n=== モデル比較結果 ===")

        # 動的ヘッダー生成
        metric_headers = [m.upper() for m in display_metrics]
        header_parts = (
            ["Model"] + [f"{m:<10}" for m in metric_headers] + ["Description"]
        )
        header = "".join(
            [f"{header_parts[0]:<12}"]
            + header_parts[1:-1]
            + [f"{header_parts[-1]:<50}"]
        )
        print(header)
        print("-" * len(header))

        # ソート用の主要指標を決定（AUC > Recall@10 > NDCG@10 > その他の順）
        sort_metric = display_metrics[0] if display_metrics else "auc"
        for preferred in ["auc", "recall@10", "ndcg@10", "mrr@10"]:
            if preferred in display_metrics:
                sort_metric = preferred
                break
            # @topk付きの指標もチェック
            for metric in display_metrics:
                if metric.startswith(preferred.split("@")[0] + "@"):
                    sort_metric = metric
                    break
            if sort_metric != display_metrics[0]:
                break

        # 結果をソート
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].get("test_result", {}).get(sort_metric, 0),
            reverse=True,
        )

        descriptions = ModelRegistry.get_model_descriptions()

        for model_name, result in sorted_results:
            test_result = result.get("test_result", {})

            # 指標値を取得してフォーマット
            metric_values = []
            for metric in display_metrics:
                value = test_result.get(metric, "N/A")
                # @topk付きの指標も確認
                if value == "N/A" and "@" not in metric:
                    # @10付きのバージョンを探す
                    alt_metric = f"{metric}@10"
                    value = test_result.get(alt_metric, "N/A")

                formatted_value = (
                    f"{value:.4f}" if isinstance(value, float) else str(value)
                )
                metric_values.append(f"{formatted_value:<10}")

            description = descriptions.get(model_name, "")

            # 行を構築
            row_parts = [f"{model_name:<12}"] + metric_values + [f"{description:<50}"]
            print("".join(row_parts))


class ClickPredictionExperiment:
    """クリック予測実験の統合クラス"""

    def __init__(self):
        self.data_generator = DataGenerator()
        self.config_manager = ConfigManager()
        self.trainer = ModelTrainer(self.config_manager)

    def setup_data(self, eval_mode: str = "labeled") -> None:
        """データ準備"""
        print("=== データ準備 ===")
        users_df, items_df, interactions_df = self.data_generator.create_sample_data(
            eval_mode
        )
        self.data_generator.prepare_recbole_data(users_df, items_df, interactions_df)

    def run_single_model_experiment(
        self,
        model_name: str = "DeepFM",
        metrics: List[str] = None,
        eval_mode: str = "labeled",
    ) -> Dict[str, Any]:
        """単一モデル実験"""
        self.setup_data(eval_mode)
        return self.trainer.train_single_model(model_name, metrics, eval_mode)

    def run_quick_comparison(
        self, metrics: List[str] = None, eval_mode: str = "labeled"
    ) -> Dict[str, Any]:
        """主要モデルの高速比較"""
        print("=== 主要モデルの高速比較 ===")
        self.setup_data(eval_mode)
        models = ModelRegistry.get_quick_models()
        return self.trainer.compare_models(
            models, mode="quick", metrics=metrics, eval_mode=eval_mode
        )

    def run_comprehensive_comparison(
        self, metrics: List[str] = None, eval_mode: str = "labeled"
    ) -> Dict[str, Any]:
        """全モデルの包括的比較"""
        print("=== 全モデルの包括的比較 ===")
        self.setup_data(eval_mode)
        models = ModelRegistry.get_all_models()
        return self.trainer.compare_models(
            models, mode="full", metrics=metrics, eval_mode=eval_mode
        )

    def run_value_metrics_comparison(self) -> Dict[str, Any]:
        """Value指標での比較"""
        print("=== Value指標での比較 ===")
        self.setup_data("labeled")
        models = ModelRegistry.get_quick_models()
        metrics = MetricsManager.get_value_metrics()
        return self.trainer.compare_models(
            models, mode="quick", metrics=metrics, eval_mode="labeled"
        )

    def run_ranking_metrics_comparison(self) -> Dict[str, Any]:
        """Ranking指標での比較"""
        print("=== Ranking指標での比較 ===")
        self.setup_data("full")  # ランキング評価用のデータ形式
        # より互換性の高いモデルを選択
        models = ["Pop", "BPR", "NeuMF"]  # 確実に動作するモデル
        metrics = MetricsManager.get_ranking_metrics()
        return self.trainer.compare_models(
            models, mode="quick", metrics=metrics, eval_mode="full"
        )

    def run_custom_metrics_comparison(
        self,
        models: List[str] = None,
        metrics: List[str] = None,
        eval_mode: str = "labeled",
    ) -> Dict[str, Any]:
        """カスタム指標・モデルでの比較"""
        print(f"=== カスタム比較 (指標: {metrics}, モード: {eval_mode}) ===")
        self.setup_data(eval_mode)

        if models is None:
            models = ModelRegistry.get_quick_models()
        if metrics is None:
            metrics = MetricsManager.get_value_metrics()

        return self.trainer.compare_models(
            models, mode="quick", metrics=metrics, eval_mode=eval_mode
        )

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
    print("利用可能な評価指標:")
    print(f"  • Value指標: {MetricsManager.get_value_metrics()}")
    print(f"  • Ranking指標: {MetricsManager.get_ranking_metrics()}")

    experiment = ClickPredictionExperiment()

    # 基本的な学習と評価（自動指標選択）
    # experiment.run_single_model_experiment("DeepFM")

    # モデル比較の選択
    print("\n" + "=" * 50)
    print("異なる評価指標での比較を実行します...")

    # 1. Value指標での比較（CTR予測に適している）
    # print("\n" + "-" * 30)
    # experiment.run_value_metrics_comparison()

    # 2. Ranking指標での比較（推薦システムに適している）
    print("\n" + "-" * 30)
    experiment.run_ranking_metrics_comparison()

    # 3. カスタム指標での比較例
    # print("\n" + "-" * 30)
    # experiment.run_custom_metrics_comparison(
    #     models=["LR", "FM", "DeepFM"], metrics=["AUC", "LogLoss"], eval_mode="labeled"
    # )

    # 詳細比較も実行する場合（全32+モデル - 時間がかかるのでコメントアウト推奨）
    # comprehensive_results = experiment.run_comprehensive_comparison(
    #     metrics=["AUC", "LogLoss", "MAE"],
    #     eval_mode="labeled"
    # )

    # 予測の準備（概念的な例）
    # experiment.predict_click_probability()


if __name__ == "__main__":
    main()
