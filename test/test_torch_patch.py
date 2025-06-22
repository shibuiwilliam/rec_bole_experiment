from unittest.mock import patch

from src.recbole_experiment.utils.torch_compat import patched_torch_load


class TestTorchPatch:
    """Test class for PyTorch compatibility patch"""

    def test_patched_torch_load_without_weights_only(self):
        """Test patched_torch_load adds weights_only=False when not present"""
        with patch(
            "src.recbole_experiment.utils.torch_compat.original_torch_load"
        ) as mock_original:
            mock_original.return_value = "mock_result"

            result = patched_torch_load("model.pth")

            mock_original.assert_called_once_with("model.pth", weights_only=False)
            assert result == "mock_result"

    def test_patched_torch_load_with_weights_only_true(self):
        """Test patched_torch_load preserves weights_only=True"""
        with patch(
            "src.recbole_experiment.utils.torch_compat.original_torch_load"
        ) as mock_original:
            mock_original.return_value = "mock_result"

            result = patched_torch_load("model.pth", weights_only=True)

            mock_original.assert_called_once_with("model.pth", weights_only=True)
            assert result == "mock_result"

    def test_patched_torch_load_with_weights_only_false(self):
        """Test patched_torch_load preserves weights_only=False"""
        with patch(
            "src.recbole_experiment.utils.torch_compat.original_torch_load"
        ) as mock_original:
            mock_original.return_value = "mock_result"

            result = patched_torch_load("model.pth", weights_only=False)

            mock_original.assert_called_once_with("model.pth", weights_only=False)
            assert result == "mock_result"

    def test_patched_torch_load_with_other_kwargs(self):
        """Test patched_torch_load handles other kwargs"""
        with patch(
            "src.recbole_experiment.utils.torch_compat.original_torch_load"
        ) as mock_original:
            mock_original.return_value = "mock_result"

            result = patched_torch_load(
                "model.pth", map_location="cpu", pickle_module=None
            )

            mock_original.assert_called_once_with(
                "model.pth", map_location="cpu", pickle_module=None, weights_only=False
            )
            assert result == "mock_result"

    def test_patched_torch_load_with_mixed_kwargs(self):
        """Test patched_torch_load with weights_only and other kwargs"""
        with patch(
            "src.recbole_experiment.utils.torch_compat.original_torch_load"
        ) as mock_original:
            mock_original.return_value = "mock_result"

            result = patched_torch_load(
                "model.pth", weights_only=True, map_location="cuda:0"
            )

            mock_original.assert_called_once_with(
                "model.pth", weights_only=True, map_location="cuda:0"
            )
            assert result == "mock_result"

    def test_patched_torch_load_positional_args(self):
        """Test patched_torch_load with positional arguments"""
        with patch(
            "src.recbole_experiment.utils.torch_compat.original_torch_load"
        ) as mock_original:
            mock_original.return_value = "mock_result"

            result = patched_torch_load("model.pth", "cpu")

            mock_original.assert_called_once_with(
                "model.pth", "cpu", weights_only=False
            )
            assert result == "mock_result"
