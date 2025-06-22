"""Click prediction experiment orchestration."""

from typing import Any, Dict, List

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.context_aware_recommender import DeepFM
from recbole.utils import init_seed

from src.recbole_experiment.config.manager import ConfigManager
from src.recbole_experiment.data.dataset import Dataset
from src.recbole_experiment.models.registry import ModelRegistry
from src.recbole_experiment.training.metrics import MetricsManager
from src.recbole_experiment.training.trainer import ModelTrainer


class ClickPredictionExperiment:
    """クリック予測実験の統合クラス"""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.config_manager = ConfigManager()
        self.trainer = ModelTrainer(self.config_manager)
        self._data_prepared = False

    def _ensure_data_prepared(self) -> None:
        """データが準備されていることを確認"""
        if not self._data_prepared:
            print("=== データ準備 ===")
            self.dataset.save_to_recbole_format()
            self._data_prepared = True

    def run_single_model_experiment(
        self,
        model_name: str = "DeepFM",
        metrics: List[str] = None,
        eval_mode: str = "labeled",
    ) -> Dict[str, Any]:
        """単一モデル実験"""
        self._ensure_data_prepared()
        return self.trainer.train_single_model(model_name, metrics, eval_mode)

    def run_quick_comparison(
        self, metrics: List[str] = None, eval_mode: str = "labeled"
    ) -> Dict[str, Any]:
        """主要モデルの高速比較"""
        print("=== 主要モデルの高速比較 ===")
        self._ensure_data_prepared()
        models = ModelRegistry.get_quick_models()
        return self.trainer.compare_models(
            models, mode="quick", metrics=metrics, eval_mode=eval_mode
        )

    def run_comprehensive_comparison(
        self, metrics: List[str] = None, eval_mode: str = "labeled"
    ) -> Dict[str, Any]:
        """全モデルの包括的比較"""
        print("=== 全モデルの包括的比較 ===")
        self._ensure_data_prepared()
        models = ModelRegistry.get_all_models()
        return self.trainer.compare_models(
            models, mode="full", metrics=metrics, eval_mode=eval_mode
        )

    def run_value_metrics_comparison(self) -> Dict[str, Any]:
        """Value指標での比較"""
        print("=== Value指標での比較 ===")
        self._ensure_data_prepared()
        models = ModelRegistry.get_quick_models()
        metrics = MetricsManager.get_value_metrics()
        return self.trainer.compare_models(
            models, mode="quick", metrics=metrics, eval_mode="labeled"
        )

    def run_ranking_metrics_comparison(self) -> Dict[str, Any]:
        """Ranking指標での比較"""
        print("=== Ranking指標での比較 ===")
        self._ensure_data_prepared()
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
        self._ensure_data_prepared()

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
