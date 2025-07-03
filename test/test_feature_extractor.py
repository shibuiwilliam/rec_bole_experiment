"""Tests for feature extractor utility."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.recbole_experiment.utils.feature_extractor import FNNFeatureExtractor


class TestFNNFeatureExtractor:
    """Test cases for FNNFeatureExtractor class"""

    def test_init_requires_valid_model_path(self, tmp_path):
        """Test that FNNFeatureExtractor requires valid model path"""
        non_existent_path = tmp_path / "non_existent_model.pth"
        
        with pytest.raises(RuntimeError, match="Failed to load model"):
            FNNFeatureExtractor(str(non_existent_path))

    @patch('src.recbole_experiment.utils.feature_extractor.load_data_and_model')
    def test_init_loads_model_successfully(self, mock_load):
        """Test successful model loading"""
        # Mock the load_data_and_model function
        mock_config = {'device': 'cpu'}
        mock_model = Mock()
        mock_model.__class__.__name__ = "FNN"
        mock_dataset = Mock()
        mock_train_data = Mock()
        mock_valid_data = Mock()
        mock_test_data = Mock()
        
        mock_load.return_value = (
            mock_config, mock_model, mock_dataset, 
            mock_train_data, mock_valid_data, mock_test_data
        )
        
        extractor = FNNFeatureExtractor("dummy_path.pth")
        
        assert extractor.config == mock_config
        assert extractor.model == mock_model
        assert extractor.dataset == mock_dataset
        mock_model.eval.assert_called_once()

    @patch('src.recbole_experiment.utils.feature_extractor.load_data_and_model')
    def test_get_model_info(self, mock_load):
        """Test get_model_info method"""
        # Mock the load_data_and_model function
        mock_config = {
            'device': 'cpu',
            'embedding_size': 64,
            'mlp_hidden_size': [128, 64]
        }
        mock_model = Mock()
        mock_model.__class__.__name__ = "FNN"
        mock_dataset = Mock()
        mock_dataset.user_num = 100
        mock_dataset.item_num = 50
        mock_dataset.__len__ = Mock(return_value=1000)
        
        mock_load.return_value = (
            mock_config, mock_model, mock_dataset, None, None, None
        )
        
        extractor = FNNFeatureExtractor("dummy_path.pth")
        info = extractor.get_model_info()
        
        assert info['model_type'] == "FNN"
        assert info['device'] == "cpu"
        assert info['embedding_size'] == 64
        assert info['num_users'] == 100
        assert info['num_items'] == 50

    @pytest.mark.skip(reason="Complex mocking of PyTorch tensor operations")
    @patch('src.recbole_experiment.utils.feature_extractor.load_data_and_model')
    def test_extract_user_item_embeddings(self, mock_load):
        """Test extract_user_item_embeddings method - skipped due to complex mocking"""
        pass

    @patch('src.recbole_experiment.utils.feature_extractor.load_data_and_model')
    def test_extract_user_item_embeddings_mismatched_lengths(self, mock_load):
        """Test that mismatched user_ids and item_ids raises error"""
        mock_config = {'device': 'cpu'}
        mock_model = Mock()
        mock_dataset = Mock()
        
        mock_load.return_value = (
            mock_config, mock_model, mock_dataset, None, None, None
        )
        
        extractor = FNNFeatureExtractor("dummy_path.pth")
        
        user_ids = [0, 1]
        item_ids = [0, 1, 2]  # Different length
        
        with pytest.raises(ValueError, match="user_ids and item_ids must have the same length"):
            extractor.extract_user_item_embeddings(user_ids, item_ids)

    @patch('src.recbole_experiment.utils.feature_extractor.load_data_and_model')
    def test_analyze_feature_similarity(self, mock_load):
        """Test analyze_feature_similarity method"""
        mock_config = {'device': 'cpu'}
        mock_model = Mock()
        mock_dataset = Mock()
        
        mock_load.return_value = (
            mock_config, mock_model, mock_dataset, None, None, None
        )
        
        extractor = FNNFeatureExtractor("dummy_path.pth")
        
        # Create mock features
        features = {
            'embeddings': np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],  # Similar to first
            ]),
            'user_ids': np.array([1, 2, 3]),
            'item_ids': np.array([101, 102, 103]),
        }
        
        similarity_results = extractor.analyze_feature_similarity(features, top_k=2)
        
        assert 'similarity_matrix' in similarity_results
        assert 'top_similar_pairs' in similarity_results
        assert 'mean_similarity' in similarity_results
        assert 'std_similarity' in similarity_results
        
        # Check that we get the expected number of top pairs
        assert len(similarity_results['top_similar_pairs']) == 2
        
        # The most similar pair should be between indices 0 and 2 (identical embeddings)
        top_pair = similarity_results['top_similar_pairs'][0]
        assert top_pair['similarity'] == 1.0  # Identical vectors

    @patch('src.recbole_experiment.utils.feature_extractor.load_data_and_model')
    def test_save_features_npz(self, mock_load, tmp_path):
        """Test saving features in NPZ format"""
        mock_config = {'device': 'cpu'}
        mock_model = Mock()
        mock_dataset = Mock()
        
        mock_load.return_value = (
            mock_config, mock_model, mock_dataset, None, None, None
        )
        
        extractor = FNNFeatureExtractor("dummy_path.pth")
        
        # Create mock features
        features = {
            'user_ids': np.array([1, 2, 3]),
            'item_ids': np.array([101, 102, 103]),
            'embeddings': np.random.rand(3, 5),
            'mlp_features': np.random.rand(3, 4),
            'predictions': np.array([0.1, 0.8, 0.3]),
            'labels': np.array([0, 1, 0]),
            'embedding_dim': 5,
            'mlp_feature_dim': 4,
            'num_samples': 3
        }
        
        save_path = tmp_path / "features.npz"
        extractor._save_features(features, str(save_path))
        
        assert save_path.exists()
        
        # Load and verify
        loaded = np.load(save_path)
        np.testing.assert_array_equal(loaded['user_ids'], features['user_ids'])
        np.testing.assert_array_equal(loaded['embeddings'], features['embeddings'])

    @patch('src.recbole_experiment.utils.feature_extractor.load_data_and_model')
    def test_save_features_csv(self, mock_load, tmp_path):
        """Test saving features in CSV format"""
        mock_config = {'device': 'cpu'}
        mock_model = Mock()
        mock_dataset = Mock()
        
        mock_load.return_value = (
            mock_config, mock_model, mock_dataset, None, None, None
        )
        
        extractor = FNNFeatureExtractor("dummy_path.pth")
        
        # Create mock features
        features = {
            'user_ids': np.array([1, 2, 3]),
            'item_ids': np.array([101, 102, 103]),
            'embeddings': np.random.rand(3, 2),  # Small dimensions for testing
            'mlp_features': np.random.rand(3, 2),
            'predictions': np.array([0.1, 0.8, 0.3]),
            'labels': np.array([0, 1, 0]),
            'embedding_dim': 2,
            'mlp_feature_dim': 2,
            'num_samples': 3
        }
        
        save_path = tmp_path / "features.csv"
        extractor._save_features(features, str(save_path))
        
        assert save_path.exists()
        
        # Basic verification that file was created and has content
        content = save_path.read_text()
        assert "user_id" in content
        assert "item_id" in content
        assert "embed_0" in content
        assert "mlp_0" in content

    @patch('src.recbole_experiment.utils.feature_extractor.load_data_and_model')
    def test_save_features_unsupported_format(self, mock_load):
        """Test that unsupported file format raises error"""
        mock_config = {'device': 'cpu'}
        mock_model = Mock()
        mock_dataset = Mock()
        
        mock_load.return_value = (
            mock_config, mock_model, mock_dataset, None, None, None
        )
        
        extractor = FNNFeatureExtractor("dummy_path.pth")
        
        features = {'user_ids': np.array([1, 2, 3])}
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            extractor._save_features(features, "features.txt")

    @patch('src.recbole_experiment.utils.feature_extractor.load_data_and_model')
    def test_extract_user_item_features_invalid_data_type(self, mock_load):
        """Test that invalid data_type raises error"""
        mock_config = {'device': 'cpu'}
        mock_model = Mock()
        mock_dataset = Mock()
        
        mock_load.return_value = (
            mock_config, mock_model, mock_dataset, None, None, None
        )
        
        extractor = FNNFeatureExtractor("dummy_path.pth")
        
        with pytest.raises(ValueError, match="Invalid data_type"):
            extractor.extract_user_item_features(data_type="invalid")