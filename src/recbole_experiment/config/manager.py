"""Configuration management for RecBole experiments."""

from typing import Any, Dict, List

from src.recbole_experiment.training.metrics import MetricsManager


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
