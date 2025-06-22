"""Metrics management for RecBole experiments."""

from typing import List

from src.recbole_experiment.models.registry import ModelRegistry


class MetricsManager:
    """評価指標管理クラス"""

    # 評価指標の定義
    RANKING_METRICS = ["Recall", "MRR", "NDCG", "Hit", "Precision"]
    VALUE_METRICS = ["AUC", "LogLoss", "MAE", "RMSE"]

    @classmethod
    def get_ranking_metrics(cls) -> List[str]:
        """ランキング指標を取得"""
        return cls.RANKING_METRICS.copy()

    @classmethod
    def get_value_metrics(cls) -> List[str]:
        """値ベース指標を取得"""
        return cls.VALUE_METRICS.copy()

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
