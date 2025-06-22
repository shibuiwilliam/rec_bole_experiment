# RecBoleを使ったアイテムクリック予測モデル

import click

from src.recbole_experiment import (
    ClickPredictionExperiment,
    DataGenerator,
    Dataset,
    MetricsManager,
)
from src.recbole_experiment.utils.torch_compat import apply_torch_compatibility_patch

# Apply PyTorch compatibility patches
apply_torch_compatibility_patch()


@click.group()
def main():
    """RecBoleを使ったアイテムクリック予測モデル"""
    pass


@main.command("single_model")
@click.option("--model", default="DeepFM", help="モデル名")
@click.option("--metrics", multiple=True, help="評価指標（複数指定可）")
@click.option(
    "--eval-mode",
    default="labeled",
    type=click.Choice(["labeled", "full"]),
    help="評価モード",
)
def run_single_model(model, metrics, eval_mode):
    """単一モデルの学習と評価を実行"""
    print("RecBoleを使ったアイテムクリック予測モデル")
    print("=" * 50)
    print("利用可能な評価指標:")
    print(f"  • Value指標: {MetricsManager.get_value_metrics()}")
    print(f"  • Ranking指標: {MetricsManager.get_ranking_metrics()}")

    # データセットの事前生成
    print("\n=== データセット準備 ===")
    users_df, items_df, interactions_df = DataGenerator.create_sample_data(
        eval_mode=eval_mode
    )
    dataset = Dataset(users_df, items_df, interactions_df)

    # 実験インスタンスの作成（データセットを渡す）
    experiment = ClickPredictionExperiment(dataset)

    # 指標リストの変換
    metrics_list = list(metrics) if metrics else None

    # 単一モデルの実行
    experiment.run_single_model_experiment(model, metrics_list, eval_mode)


@main.command("quick_comparison")
@click.option("--metrics", multiple=True, help="評価指標（複数指定可）")
@click.option(
    "--eval-mode",
    default="labeled",
    type=click.Choice(["labeled", "full"]),
    help="評価モード",
)
def run_quick_comparison(metrics, eval_mode):
    """主要モデルの高速比較を実行"""
    print("RecBoleを使ったアイテムクリック予測モデル")
    print("=" * 50)

    # データセットの事前生成
    print("\n=== データセット準備 ===")
    users_df, items_df, interactions_df = DataGenerator.create_sample_data(
        eval_mode=eval_mode
    )
    dataset = Dataset(users_df, items_df, interactions_df)

    # 実験インスタンスの作成（データセットを渡す）
    experiment = ClickPredictionExperiment(dataset)

    # 指標リストの変換
    metrics_list = list(metrics) if metrics else None

    # 主要モデルの高速比較
    experiment.run_quick_comparison(metrics_list, eval_mode)


@main.command("comprehensive_comparison")
@click.option("--metrics", multiple=True, help="評価指標（複数指定可）")
@click.option(
    "--eval-mode",
    default="labeled",
    type=click.Choice(["labeled", "full"]),
    help="評価モード",
)
def run_comprehensive_comparison(metrics, eval_mode):
    """全モデルの包括的比較を実行"""
    print("RecBoleを使ったアイテムクリック予測モデル")
    print("=" * 50)

    # データセットの事前生成
    print("\n=== データセット準備 ===")
    users_df, items_df, interactions_df = DataGenerator.create_sample_data(
        eval_mode=eval_mode
    )
    dataset = Dataset(users_df, items_df, interactions_df)

    # 実験インスタンスの作成（データセットを渡す）
    experiment = ClickPredictionExperiment(dataset)

    # 指標リストの変換
    metrics_list = list(metrics) if metrics else None

    # 全モデルの包括的比較
    experiment.run_comprehensive_comparison(metrics_list, eval_mode)


@main.command("value_metrics")
def run_value_metrics():
    """Value指標での比較を実行"""
    print("RecBoleを使ったアイテムクリック予測モデル")
    print("=" * 50)

    # データセットの事前生成
    print("\n=== データセット準備 ===")
    users_df, items_df, interactions_df = DataGenerator.create_sample_data(
        eval_mode="labeled"
    )
    dataset = Dataset(users_df, items_df, interactions_df)

    # 実験インスタンスの作成（データセットを渡す）
    experiment = ClickPredictionExperiment(dataset)

    # Value指標での比較
    experiment.run_value_metrics_comparison()


@main.command("ranking_metrics")
def run_ranking_metrics():
    """Ranking指標での比較を実行"""
    print("RecBoleを使ったアイテムクリック予測モデル")
    print("=" * 50)

    # データセットの事前生成
    print("\n=== データセット準備 ===")
    users_df, items_df, interactions_df = DataGenerator.create_sample_data(
        eval_mode="full"
    )
    dataset = Dataset(users_df, items_df, interactions_df)

    # 実験インスタンスの作成（データセットを渡す）
    experiment = ClickPredictionExperiment(dataset)

    # Ranking指標での比較
    experiment.run_ranking_metrics_comparison()


@main.command("custom_metrics")
@click.option("--models", multiple=True, help="モデル名（複数指定可）")
@click.option("--metrics", multiple=True, help="評価指標（複数指定可）")
@click.option(
    "--eval-mode",
    default="labeled",
    type=click.Choice(["labeled", "full"]),
    help="評価モード",
)
def run_custom_metrics(models, metrics, eval_mode):
    """カスタム指標・モデルでの比較を実行"""
    print("RecBoleを使ったアイテムクリック予測モデル")
    print("=" * 50)

    # データセットの事前生成
    print("\n=== データセット準備 ===")
    users_df, items_df, interactions_df = DataGenerator.create_sample_data(
        eval_mode=eval_mode
    )
    dataset = Dataset(users_df, items_df, interactions_df)

    # 実験インスタンスの作成（データセットを渡す）
    experiment = ClickPredictionExperiment(dataset)

    # リストの変換
    models_list = list(models) if models else None
    metrics_list = list(metrics) if metrics else None

    # カスタム指標での比較
    experiment.run_custom_metrics_comparison(models_list, metrics_list, eval_mode)


@main.command("predict")
def predict_click():
    """クリック確率予測の例を実行"""
    print("RecBoleを使ったアイテムクリック予測モデル")
    print("=" * 50)

    # データセットの事前生成
    print("\n=== データセット準備 ===")
    users_df, items_df, interactions_df = DataGenerator.create_sample_data(
        eval_mode="labeled"
    )
    dataset = Dataset(users_df, items_df, interactions_df)

    # 実験インスタンスの作成（データセットを渡す）
    experiment = ClickPredictionExperiment(dataset)

    # クリック確率予測
    experiment.predict_click_probability()


if __name__ == "__main__":
    main()
