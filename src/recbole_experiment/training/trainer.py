"""Model training and evaluation functionality."""

from typing import Any, Dict, List

from recbole.quick_start import run_recbole

from src.recbole_experiment.models.registry import ModelRegistry
from src.recbole_experiment.training.metrics import MetricsManager
from src.recbole_experiment.utils.model_saver import ModelSaver


class ModelTrainer:
    """モデル訓練・評価クラス"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.model_saver = ModelSaver()

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

        # Save model to outputs directory
        saved_path = self.model_saver.save_model(model_name)
        if saved_path:
            result["saved_model_path"] = saved_path

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

                # Save model to outputs directory
                saved_path = self.model_saver.save_model(model_name)
                if saved_path:
                    result["saved_model_path"] = saved_path

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
