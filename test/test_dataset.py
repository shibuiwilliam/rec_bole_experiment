import tempfile

import pandas as pd

from src.recbole_experiment.data.dataset import Dataset


class TestDataset:
    """Test class for Dataset"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_users = pd.DataFrame(
            {
                "user_id:token": ["u_0", "u_1"],
                "age:float": [25, 30],
                "gender:token": ["M", "F"],
            }
        )

        self.sample_items = pd.DataFrame(
            {
                "item_id:token": ["i_0", "i_1"],
                "price:float": [100.0, 200.0],
                "rating:float": [3.5, 4.0],
                "category:token": ["Electronics", "Books"],
            }
        )

        self.sample_interactions = pd.DataFrame(
            {
                "user_id:token": ["u_0", "u_1"],
                "item_id:token": ["i_0", "i_1"],
                "label:float": [1.0, 0.0],
                "timestamp:float": [1500000000, 1600000000],
            }
        )

        self.dataset = Dataset(
            self.sample_users, self.sample_items, self.sample_interactions
        )

    def test_init(self):
        """Test Dataset initialization"""
        assert self.dataset.users_df is self.sample_users
        assert self.dataset.items_df is self.sample_items
        assert self.dataset.interactions_df is self.sample_interactions

    def test_init_with_copies(self):
        """Test that Dataset stores references, not copies"""
        users_df = self.sample_users.copy()
        items_df = self.sample_items.copy()
        interactions_df = self.sample_interactions.copy()

        dataset = Dataset(users_df, items_df, interactions_df)

        # Should be the same objects
        assert dataset.users_df is users_df
        assert dataset.items_df is items_df
        assert dataset.interactions_df is interactions_df

    def test_save_to_recbole_format_default_path(self, mocker):
        """Test saving to RecBole format with default path"""
        mock_prepare = mocker.patch(
            "src.recbole_experiment.data.generator.DataGenerator.prepare_recbole_data"
        )
        self.dataset.save_to_recbole_format()

        mock_prepare.assert_called_once_with(
            self.sample_users,
            self.sample_items,
            self.sample_interactions,
            "dataset/click_prediction/",
        )

    def test_save_to_recbole_format_custom_path(self, mocker):
        """Test saving to RecBole format with custom path"""
        custom_path = "custom/path/"

        mock_prepare = mocker.patch(
            "src.recbole_experiment.data.generator.DataGenerator.prepare_recbole_data"
        )
        self.dataset.save_to_recbole_format(custom_path)

        mock_prepare.assert_called_once_with(
            self.sample_users,
            self.sample_items,
            self.sample_interactions,
            custom_path,
        )

    def test_save_to_recbole_format_integration(self):
        """Test actual saving to RecBole format (integration test)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.dataset.save_to_recbole_format(temp_dir)

            # Check that files exist and have correct content
            import os

            user_file = os.path.join(temp_dir, "click_prediction.user")
            item_file = os.path.join(temp_dir, "click_prediction.item")
            inter_file = os.path.join(temp_dir, "click_prediction.inter")

            assert os.path.exists(user_file)
            assert os.path.exists(item_file)
            assert os.path.exists(inter_file)

            # Verify content
            saved_users = pd.read_csv(user_file, sep="\t")
            saved_items = pd.read_csv(item_file, sep="\t")
            saved_interactions = pd.read_csv(inter_file, sep="\t")

            pd.testing.assert_frame_equal(self.sample_users, saved_users)
            pd.testing.assert_frame_equal(self.sample_items, saved_items)
            pd.testing.assert_frame_equal(self.sample_interactions, saved_interactions)

    def test_dataframes_are_accessible(self):
        """Test that dataframes can be accessed and modified"""
        # Should be able to access dataframes
        assert len(self.dataset.users_df) == 2
        assert len(self.dataset.items_df) == 2
        assert len(self.dataset.interactions_df) == 2

        # Should be able to modify them (they're references)
        original_user_count = len(self.dataset.users_df)
        new_user = pd.DataFrame(
            {"user_id:token": ["u_2"], "age:float": [35], "gender:token": ["M"]}
        )

        # Append to original dataframe
        self.sample_users = pd.concat([self.sample_users, new_user], ignore_index=True)

        # Dataset should not automatically reflect this change since it holds a reference
        # to the original dataframe, not the new concatenated one
        assert len(self.dataset.users_df) == original_user_count

    def test_dataset_with_empty_dataframes(self, mocker):
        """Test Dataset with empty dataframes"""
        empty_users = pd.DataFrame(
            columns=["user_id:token", "age:float", "gender:token"]
        )
        empty_items = pd.DataFrame(
            columns=["item_id:token", "price:float", "rating:float", "category:token"]
        )
        empty_interactions = pd.DataFrame(
            columns=["user_id:token", "item_id:token", "label:float", "timestamp:float"]
        )

        dataset = Dataset(empty_users, empty_items, empty_interactions)

        assert len(dataset.users_df) == 0
        assert len(dataset.items_df) == 0
        assert len(dataset.interactions_df) == 0

        # Should still be able to save
        mock_prepare = mocker.patch(
            "src.recbole_experiment.data.generator.DataGenerator.prepare_recbole_data"
        )
        dataset.save_to_recbole_format()
        mock_prepare.assert_called_once()

    def test_dataset_with_large_dataframes(self):
        """Test Dataset with larger dataframes"""
        # Create larger test data
        n_users = 100
        n_items = 50
        n_interactions = 1000

        large_users = pd.DataFrame(
            {
                "user_id:token": [f"u_{i}" for i in range(n_users)],
                "age:float": [25 + i % 40 for i in range(n_users)],
                "gender:token": ["M" if i % 2 == 0 else "F" for i in range(n_users)],
            }
        )

        large_items = pd.DataFrame(
            {
                "item_id:token": [f"i_{i}" for i in range(n_items)],
                "price:float": [100.0 + i * 10 for i in range(n_items)],
                "rating:float": [3.0 + (i % 3) for i in range(n_items)],
                "category:token": ["Cat" + str(i % 5) for i in range(n_items)],
            }
        )

        large_interactions = pd.DataFrame(
            {
                "user_id:token": [f"u_{i % n_users}" for i in range(n_interactions)],
                "item_id:token": [f"i_{i % n_items}" for i in range(n_interactions)],
                "label:float": [float(i % 2) for i in range(n_interactions)],
                "timestamp:float": [
                    1500000000 + i * 1000 for i in range(n_interactions)
                ],
            }
        )

        dataset = Dataset(large_users, large_items, large_interactions)

        assert len(dataset.users_df) == n_users
        assert len(dataset.items_df) == n_items
        assert len(dataset.interactions_df) == n_interactions

    def test_dataset_dataframe_properties(self):
        """Test that Dataset properly exposes dataframe properties"""
        # Test that we can access dataframe properties
        assert isinstance(self.dataset.users_df.columns, pd.Index)
        assert isinstance(self.dataset.items_df.dtypes, pd.Series)
        assert isinstance(self.dataset.interactions_df.shape, tuple)

        # Test that we can call dataframe methods
        assert self.dataset.users_df.empty is False
        assert self.dataset.items_df.size > 0
        assert len(self.dataset.interactions_df.index) == 2

    def test_dataset_immutability_concept(self):
        """Test the concept of dataset immutability through references"""
        # Create new dataframes
        new_users = pd.DataFrame(
            {
                "user_id:token": ["u_new"],
                "age:float": [40],
                "gender:token": ["X"],
            }
        )

        # Dataset should maintain its original references
        original_users_id = id(self.dataset.users_df)

        # Even if we reassign the variable we used to create the dataset,
        # the dataset should maintain its reference
        self.sample_users = new_users

        # Dataset should still reference the original dataframe
        assert id(self.dataset.users_df) == original_users_id
        assert "u_new" not in self.dataset.users_df["user_id:token"].values

    def test_multiple_datasets_independence(self):
        """Test that multiple Dataset instances are independent"""
        dataset1 = Dataset(
            self.sample_users.copy(),
            self.sample_items.copy(),
            self.sample_interactions.copy(),
        )

        dataset2 = Dataset(
            self.sample_users.copy(),
            self.sample_items.copy(),
            self.sample_interactions.copy(),
        )

        # Should be different objects
        assert dataset1 is not dataset2
        assert dataset1.users_df is not dataset2.users_df
        assert dataset1.items_df is not dataset2.items_df
        assert dataset1.interactions_df is not dataset2.interactions_df

        # But should have same content
        pd.testing.assert_frame_equal(dataset1.users_df, dataset2.users_df)
        pd.testing.assert_frame_equal(dataset1.items_df, dataset2.items_df)
        pd.testing.assert_frame_equal(
            dataset1.interactions_df, dataset2.interactions_df
        )
