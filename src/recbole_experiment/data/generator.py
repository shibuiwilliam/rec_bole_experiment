"""Data generation utilities for RecBole experiments."""

import os
from typing import Tuple

import numpy as np
import pandas as pd


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
