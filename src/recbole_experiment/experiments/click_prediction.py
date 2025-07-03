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
from src.recbole_experiment.utils.feature_extractor import FNNFeatureExtractor


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
        models = ModelRegistry.get_context_aware_models()
        metrics = MetricsManager.get_value_metrics()
        return self.trainer.compare_models(
            models, mode="quick", metrics=metrics, eval_mode="labeled"
        )

    def run_ranking_metrics_comparison(self) -> Dict[str, Any]:
        """Ranking指標での比較"""
        print("=== Ranking指標での比較 ===")
        self._ensure_data_prepared()
        # ランキング指標と互換性のあるモデルを選択
        models = ModelRegistry.get_compatible_ranking_models()
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

    def extract_model_features(
        self,
        model_path: str,
        data_type: str = "test",
        save_path: str = None,
        analyze_similarity: bool = False,
    ) -> Dict[str, Any]:
        """
        学習済みFNNモデルからuser x item特徴量を抽出

        Args:
            model_path: 学習済みモデルのパス
            data_type: 特徴量を抽出するデータ種別 ("train", "valid", "test")
            save_path: 特徴量保存パス (optional)
            analyze_similarity: 類似度分析を実行するかどうか

        Returns:
            抽出された特徴量の辞書
        """
        print("=== FNNモデル特徴量抽出 ===")
        print(f"モデルパス: {model_path}")
        print(f"データ種別: {data_type}")

        try:
            # 特徴量抽出器を初期化
            extractor = FNNFeatureExtractor(model_path)

            # モデル情報を表示
            model_info = extractor.get_model_info()
            print("\nモデル情報:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")

            # 特徴量を抽出
            features = extractor.extract_user_item_features(
                data_type=data_type, save_path=save_path
            )

            # 類似度分析を実行
            if analyze_similarity:
                print("\n類似度分析を実行中...")
                similarity_results = extractor.analyze_feature_similarity(
                    features, top_k=5
                )

                print(f"平均類似度: {similarity_results['mean_similarity']:.4f}")
                print(f"類似度標準偏差: {similarity_results['std_similarity']:.4f}")
                print("\n最も類似度の高いuser-itemペア (Top 5):")
                for i, pair in enumerate(similarity_results["top_similar_pairs"]):
                    print(
                        f"  {i+1}. User {pair['user1']}-Item {pair['item1']} ~ "
                        f"User {pair['user2']}-Item {pair['item2']} "
                        f"(類似度: {pair['similarity']:.4f})"
                    )

                features["similarity_analysis"] = similarity_results

            print(f"\n特徴量抽出完了: {features['num_samples']} サンプル")
            return features

        except Exception as e:
            print(f"特徴量抽出でエラーが発生しました: {str(e)}")
            raise
