# RecBoleを使ったアイテムクリック予測モデル

import os

import numpy as np
import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.context_aware_recommender import (
    DeepFM,
)
from recbole.quick_start import run_recbole
from recbole.utils import init_seed

# PyTorch 2.6 compatibility fix
original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)


torch.load = patched_torch_load


def create_sample_data():
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
        "user_id:token": np.random.choice(users_df["user_id:token"], n_interactions),
        "item_id:token": np.random.choice(items_df["item_id:token"], n_interactions),
        "label:float": np.random.choice(
            [0, 1], n_interactions, p=[0.8, 0.2]
        ),  # 0: PV, 1: Click
        "timestamp:float": np.random.randint(1000000000, 1700000000, n_interactions),
    }
    interactions_df = pd.DataFrame(interactions_data)

    return users_df, items_df, interactions_df


def prepare_recbole_data(
    users_df, items_df, interactions_df, data_path="dataset/click_prediction/"
):
    """RecBole用のデータファイル作成"""

    # データディレクトリの作成
    os.makedirs(data_path, exist_ok=True)

    # .user ファイル（ユーザー属性）
    users_df.to_csv(
        os.path.join(data_path, "click_prediction.user"), sep="\t", index=False
    )

    # .item ファイル（アイテム属性）
    items_df.to_csv(
        os.path.join(data_path, "click_prediction.item"), sep="\t", index=False
    )

    # .inter ファイル（インタラクション）
    interactions_df.to_csv(
        os.path.join(data_path, "click_prediction.inter"), sep="\t", index=False
    )

    print(f"データファイルを {data_path} に保存しました")


def create_config():
    """RecBoleの設定"""
    config_dict = {
        # データセット設定
        "dataset": "click_prediction",
        "data_path": "dataset/",
        # モデル設定
        "model": "DeepFM",
        "loss_type": "BCE",  # Binary Cross Entropy for CTR prediction
        # 特徴量設定
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "RATING_FIELD": "label",  # クリック（1）かPV（0）か
        "TIME_FIELD": "timestamp",
        # ユーザー・アイテム特徴量
        "load_col": {
            "inter": ["user_id", "item_id", "label", "timestamp"],
            "user": ["user_id", "age", "gender"],
            "item": ["item_id", "price", "rating", "category"],
        },
        # 数値特徴量の設定
        "numerical_features": ["age", "price", "rating"],
        # カテゴリ特徴量の設定
        "categorical_features": ["gender", "category"],
        # 学習設定
        "epochs": 30,  # 複数モデル比較のため短縮
        "learning_rate": 0.001,
        "train_batch_size": 2048,
        "eval_batch_size": 2048,
        # 評価設定
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},  # train:valid:test = 8:1:1
            "group_by": None,
            "order": "TO",  # Time Order
            "mode": "labeled",  # CTR予測用
        },
        # 評価指標 (CTR予測用のvalue metrics)
        "metrics": ["AUC", "LogLoss"],
        "metric_decimal_place": 4,
        # DeepFMの設定
        "embedding_size": 64,
        "mlp_hidden_size": [256, 128, 64],
        "dropout_prob": 0.2,
        # その他
        "seed": 2023,
        "reproducibility": True,
        "state": "INFO",
    }

    return config_dict


def train_and_evaluate():
    """モデルの学習と評価"""

    print("=== データ準備 ===")
    users_df, items_df, interactions_df = create_sample_data()
    prepare_recbole_data(users_df, items_df, interactions_df)

    print("=== 設定作成 ===")
    config_dict = create_config()

    print("=== モデル学習開始 ===")

    # RecBoleの実行
    result = run_recbole(
        model="DeepFM", dataset="click_prediction", config_dict=config_dict
    )

    print("=== 学習完了 ===")
    print(f"テスト結果: {result}")

    return result


def advanced_model_comparison():
    """複数のモデルを比較"""

    print("=== 複数モデルの比較 ===")

    # CTR予測に適したモデルのリスト
    models = [
        "LR",  # Logistic Regression - シンプルなベースライン
        "FM",  # Factorization Machines - 特徴量交互作用をモデル化
        "DeepFM",  # Deep Factorization Machines - FMとDNNの組み合わせ
        "WideDeep",  # Wide & Deep - 記憶と汎化の両方をカバー
        "xDeepFM",  # eXtreme Deep Factorization Machines - 明示的・暗黙的特徴量交互作用
        "DCN",  # Deep & Cross Network - 特徴量クロスを自動学習
        "NFM",  # Neural Factorization Machines - 高次特徴量交互作用
        "AFM",  # Attentional Factorization Machines - アテンション機構付きFM
        "AutoInt",  # AutoInt - 自動特徴量交互作用学習
        "PNN",  # Product-based Neural Network - 積ベースのネットワーク
    ]
    results = {}

    base_config = create_config()

    for model_name in models:
        print(f"\n--- {model_name} の学習 ---")

        config_dict = base_config.copy()
        config_dict["model"] = model_name

        # モデル固有の設定
        if model_name == "LR":
            # Logistic Regressionは単純な線形モデル
            config_dict["epochs"] = 20
        elif model_name == "FM":
            # Factorization Machinesの埋め込みサイズ
            config_dict["embedding_size"] = 32
        elif model_name == "WideDeep":
            # Wide & Deep固有の設定
            config_dict["mlp_hidden_size"] = [128, 64]
        elif model_name == "xDeepFM":
            # xDeepFM固有の設定
            config_dict["cin_layer_size"] = [128, 128]
            config_dict["direct"] = True
        elif model_name == "DCN":
            # Deep & Cross Network固有の設定
            config_dict["cross_layer_num"] = 3
        elif model_name == "AFM":
            # Attentional FM固有の設定
            config_dict["attention_size"] = 32
        elif model_name == "AutoInt":
            # AutoInt固有の設定
            config_dict["attention_layers"] = 3
            config_dict["num_heads"] = 2
        elif model_name == "PNN":
            # PNN固有の設定
            config_dict["use_inner"] = True
            config_dict["use_outer"] = False

        try:
            result = run_recbole(
                model=model_name, dataset="click_prediction", config_dict=config_dict
            )
            results[model_name] = result
            test_result = result.get("test_result", {})
            auc = test_result.get("auc", "N/A")
            logloss = test_result.get("logloss", "N/A")
            print(f"{model_name} 完了: AUC = {auc}, LogLoss = {logloss}")

        except Exception as e:
            print(f"{model_name} でエラー: {str(e)}")
            continue

    # 結果の比較とランキング
    print("\n=== モデル比較結果 ===")
    print(f"{'Model':<12} {'AUC':<8} {'LogLoss':<10} {'Description':<50}")
    print("-" * 80)

    # AUCでソート（降順）
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get("test_result", {}).get("auc", 0),
        reverse=True,
    )

    model_descriptions = {
        "LR": "Logistic Regression (ベースライン)",
        "FM": "Factorization Machines (特徴量交互作用)",
        "DeepFM": "Deep + FM (深層学習とFMの組み合わせ)",
        "WideDeep": "Wide & Deep (記憶と汎化)",
        "xDeepFM": "eXtreme DeepFM (明示的・暗黙的交互作用)",
        "DCN": "Deep & Cross Network (自動特徴量クロス)",
        "NFM": "Neural FM (高次特徴量交互作用)",
        "AFM": "Attentional FM (アテンション機構)",
        "AutoInt": "AutoInt (自動特徴量交互作用学習)",
        "PNN": "Product-based Neural Network (積ベース)",
    }

    for model_name, result in sorted_results:
        test_result = result.get("test_result", {})
        auc = test_result.get("auc", "N/A")
        logloss = test_result.get("logloss", "N/A")
        description = model_descriptions.get(model_name, "")

        auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
        logloss_str = f"{logloss:.4f}" if isinstance(logloss, float) else str(logloss)

        print(f"{model_name:<12} {auc_str:<8} {logloss_str:<10} {description:<50}")

    return results


def predict_click_probability():
    """新しいユーザー・アイテムペアのクリック確率を予測"""

    print("=== クリック確率予測の例 ===")

    # 実際のプロジェクトでは、学習済みモデルをロードして予測を行います
    # ここでは概念的な例を示します

    config_dict = create_config()

    # データセットとモデルの準備
    config = Config(model="DeepFM", dataset="click_prediction", config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])

    # データセットの作成
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # モデルの初期化
    model = DeepFM(config, train_data.dataset).to(config["device"])

    print("モデルの構造:")
    print(model)

    print("\n注意: 実際の予測には学習済みモデルが必要です")
    print("上記のtrain_and_evaluate()を実行してモデルを学習させてください")


if __name__ == "__main__":
    # メイン実行
    print("RecBoleを使ったアイテムクリック予測モデル")
    print("=" * 50)

    # 基本的な学習と評価
    result = train_and_evaluate()

    # 複数モデルの比較
    print("\n" + "=" * 50)
    advanced_results = advanced_model_comparison()

    # 予測の準備（概念的な例）
    # predict_click_probability()
