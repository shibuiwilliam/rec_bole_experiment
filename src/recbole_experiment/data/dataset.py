"""Dataset container for RecBole experiments."""

import pandas as pd

from src.recbole_experiment.data.generator import DataGenerator


class Dataset:
    """データセット保持クラス"""

    def __init__(
        self,
        users_df: pd.DataFrame,
        items_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
    ):
        self.users_df = users_df
        self.items_df = items_df
        self.interactions_df = interactions_df

    def save_to_recbole_format(
        self, data_path: str = "dataset/click_prediction/"
    ) -> None:
        """RecBole形式でデータを保存"""
        DataGenerator.prepare_recbole_data(
            self.users_df, self.items_df, self.interactions_df, data_path
        )
