import os
import tempfile

import numpy as np
import pandas as pd

from src.recbole_experiment.data.generator import DataGenerator


class TestDataGenerator:
    """Test class for DataGenerator"""

    def test_create_sample_data_labeled_mode(self):
        """Test sample data creation in labeled mode"""
        users_df, items_df, interactions_df = DataGenerator.create_sample_data(
            eval_mode="labeled"
        )

        # Check users dataframe
        assert len(users_df) == 1000
        assert set(users_df.columns) == {
            "user_id:token",
            "age:float",
            "gender:token",
        }
        assert all(users_df["user_id:token"].str.startswith("u_"))
        assert users_df["age:float"].between(18, 65).all()
        assert set(users_df["gender:token"]) <= {"M", "F"}

        # Check items dataframe
        assert len(items_df) == 500
        assert set(items_df.columns) == {
            "item_id:token",
            "price:float",
            "rating:float",
            "category:token",
        }
        assert all(items_df["item_id:token"].str.startswith("i_"))
        assert items_df["price:float"].between(10, 1000).all()
        assert items_df["rating:float"].between(1, 5).all()

        # Check interactions dataframe
        assert len(interactions_df) == 10000
        assert set(interactions_df.columns) == {
            "user_id:token",
            "item_id:token",
            "label:float",
            "timestamp:float",
        }
        assert set(interactions_df["label:float"]) <= {0.0, 1.0}

    def test_create_sample_data_full_mode(self):
        """Test sample data creation in full mode"""
        users_df, items_df, interactions_df = DataGenerator.create_sample_data(
            eval_mode="full"
        )

        # Check that all labels are 1.0 in full mode
        assert all(interactions_df["label:float"] == 1.0)

        # Check that each user has some interactions
        user_counts = interactions_df["user_id:token"].value_counts()
        assert user_counts.min() >= 5
        assert user_counts.max() <= 15

    def test_create_sample_data_with_mocked_random(self, mocker):
        """Test data creation with mocked random functions"""
        # Setup mocks using pytest-mock
        mock_randint = mocker.patch("numpy.random.randint")
        mock_choice = mocker.patch("numpy.random.choice")
        mock_uniform = mocker.patch("numpy.random.uniform")

        mock_randint.side_effect = [
            25,
            30,
            35,
            8,
            1500000000,
        ]  # ages, interactions, timestamp
        mock_choice.side_effect = [["M"], ["Electronics"], ["u_0"], ["i_0"], [1]]
        mock_uniform.side_effect = [[100.0], [3.5]]  # price, rating

        mock_create = mocker.patch(
            "src.recbole_experiment.data.generator.DataGenerator.create_sample_data"
        )
        mock_create.return_value = (
            pd.DataFrame(
                {"user_id:token": ["u_0"], "age:float": [25], "gender:token": ["M"]}
            ),
            pd.DataFrame(
                {
                    "item_id:token": ["i_0"],
                    "price:float": [100.0],
                    "rating:float": [3.5],
                    "category:token": ["Electronics"],
                }
            ),
            pd.DataFrame(
                {
                    "user_id:token": ["u_0"],
                    "item_id:token": ["i_0"],
                    "label:float": [1.0],
                    "timestamp:float": [1500000000],
                }
            ),
        )

        users_df, items_df, interactions_df = DataGenerator.create_sample_data()

        assert len(users_df) == 1
        assert len(items_df) == 1
        assert len(interactions_df) == 1

    def test_prepare_recbole_data(self):
        """Test RecBole data preparation"""
        # Create sample dataframes
        users_df = pd.DataFrame(
            {
                "user_id:token": ["u_0", "u_1"],
                "age:float": [25, 30],
                "gender:token": ["M", "F"],
            }
        )

        items_df = pd.DataFrame(
            {
                "item_id:token": ["i_0", "i_1"],
                "price:float": [100.0, 200.0],
                "rating:float": [3.5, 4.0],
                "category:token": ["Electronics", "Books"],
            }
        )

        interactions_df = pd.DataFrame(
            {
                "user_id:token": ["u_0", "u_1"],
                "item_id:token": ["i_0", "i_1"],
                "label:float": [1.0, 0.0],
                "timestamp:float": [1500000000, 1600000000],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            DataGenerator.prepare_recbole_data(
                users_df, items_df, interactions_df, temp_dir
            )

            # Check that files were created
            assert os.path.exists(os.path.join(temp_dir, "click_prediction.user"))
            assert os.path.exists(os.path.join(temp_dir, "click_prediction.item"))
            assert os.path.exists(os.path.join(temp_dir, "click_prediction.inter"))

            # Check file contents
            saved_users = pd.read_csv(
                os.path.join(temp_dir, "click_prediction.user"), sep="\t"
            )
            saved_items = pd.read_csv(
                os.path.join(temp_dir, "click_prediction.item"), sep="\t"
            )
            saved_interactions = pd.read_csv(
                os.path.join(temp_dir, "click_prediction.inter"), sep="\t"
            )

            pd.testing.assert_frame_equal(users_df, saved_users)
            pd.testing.assert_frame_equal(items_df, saved_items)
            pd.testing.assert_frame_equal(interactions_df, saved_interactions)

    def test_prepare_recbole_data_creates_directory(self, mocker):
        """Test that prepare_recbole_data creates directory if it doesn't exist"""
        # Setup mocks using pytest-mock
        mock_makedirs = mocker.patch("os.makedirs")
        mocker.patch("pandas.DataFrame.to_csv")
        mock_print = mocker.patch("builtins.print")

        users_df = pd.DataFrame({"user_id:token": ["u_0"]})
        items_df = pd.DataFrame({"item_id:token": ["i_0"]})
        interactions_df = pd.DataFrame(
            {"user_id:token": ["u_0"], "item_id:token": ["i_0"]}
        )

        data_path = "test_path/"
        DataGenerator.prepare_recbole_data(
            users_df, items_df, interactions_df, data_path
        )

        mock_makedirs.assert_called_once_with(data_path, exist_ok=True)
        mock_print.assert_called_once_with(
            f"データファイルを {data_path} に保存しました"
        )

    def test_data_types_and_ranges(self):
        """Test that generated data has correct types and ranges"""
        users_df, items_df, interactions_df = DataGenerator.create_sample_data()

        # Check data types
        assert users_df["age:float"].dtype in [np.float64, np.int64, int]
        assert items_df["price:float"].dtype == np.float64
        assert items_df["rating:float"].dtype == np.float64
        assert interactions_df["label:float"].dtype in [np.float64, int, float]
        assert interactions_df["timestamp:float"].dtype in [np.float64, np.int64, int]

        # Check value ranges
        assert users_df["age:float"].min() >= 18
        assert users_df["age:float"].max() < 65
        assert items_df["price:float"].min() >= 10
        assert items_df["price:float"].max() <= 1000
        assert items_df["rating:float"].min() >= 1
        assert items_df["rating:float"].max() <= 5

    def test_interactions_referential_integrity(self):
        """Test that interactions reference valid users and items"""
        users_df, items_df, interactions_df = DataGenerator.create_sample_data()

        # Check that all user_ids in interactions exist in users
        user_ids = set(users_df["user_id:token"])
        interaction_user_ids = set(interactions_df["user_id:token"])
        assert interaction_user_ids.issubset(user_ids)

        # Check that all item_ids in interactions exist in items
        item_ids = set(items_df["item_id:token"])
        interaction_item_ids = set(interactions_df["item_id:token"])
        assert interaction_item_ids.issubset(item_ids)
