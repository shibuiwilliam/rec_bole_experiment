"""Tests for model saver utility."""

import os
from unittest.mock import patch

from src.recbole_experiment.utils.model_saver import ModelSaver


class TestModelSaver:
    """Test cases for ModelSaver class"""

    def test_init_creates_outputs_directory(self, tmp_path):
        """Test that ModelSaver creates outputs directory"""
        outputs_dir = tmp_path / "test_outputs"
        ModelSaver(str(outputs_dir))

        assert outputs_dir.exists()
        assert outputs_dir.is_dir()

    def test_save_model_no_files_found(self, tmp_path):
        """Test save_model when no model files exist"""
        outputs_dir = tmp_path / "outputs"
        saved_dir = tmp_path / "saved"
        saved_dir.mkdir()

        saver = ModelSaver(str(outputs_dir))
        result = saver.save_model("NonExistentModel", str(saved_dir))

        assert result is None

    def test_save_model_success(self, tmp_path):
        """Test successful model saving"""
        outputs_dir = tmp_path / "outputs"
        saved_dir = tmp_path / "saved"
        saved_dir.mkdir()

        # Create a mock model file
        model_file = saved_dir / "DeepFM-Jun-29-2025_08-01-12.pth"
        model_file.write_text("mock model data")

        saver = ModelSaver(str(outputs_dir))

        with patch(
            "src.recbole_experiment.utils.model_saver.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20250703_120000"
            result = saver.save_model("DeepFM", str(saved_dir))

        assert result is not None
        assert "DeepFM_20250703_120000.pth" in result

        # Check that file was actually copied
        output_file = outputs_dir / "DeepFM_20250703_120000.pth"
        assert output_file.exists()
        assert output_file.read_text() == "mock model data"

    def test_save_model_multiple_files_selects_latest(self, tmp_path):
        """Test that save_model selects the most recent file"""
        outputs_dir = tmp_path / "outputs"
        saved_dir = tmp_path / "saved"
        saved_dir.mkdir()

        # Create multiple model files with different timestamps
        old_file = saved_dir / "DeepFM-Jun-28-2025_08-01-12.pth"
        old_file.write_text("old model data")

        new_file = saved_dir / "DeepFM-Jun-29-2025_08-01-12.pth"
        new_file.write_text("new model data")

        # Make the new file have a later modification time
        old_time = old_file.stat().st_mtime
        os.utime(new_file, (old_time + 100, old_time + 100))

        saver = ModelSaver(str(outputs_dir))

        with patch(
            "src.recbole_experiment.utils.model_saver.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20250703_120000"
            result = saver.save_model("DeepFM", str(saved_dir))

        assert result is not None
        output_file = outputs_dir / "DeepFM_20250703_120000.pth"
        assert output_file.exists()
        assert output_file.read_text() == "new model data"

    def test_save_model_copy_error(self, tmp_path, mocker):
        """Test save_model handles copy errors gracefully"""
        outputs_dir = tmp_path / "outputs"
        saved_dir = tmp_path / "saved"
        saved_dir.mkdir()

        # Create a model file
        model_file = saved_dir / "DeepFM-Jun-29-2025_08-01-12.pth"
        model_file.write_text("mock model data")

        saver = ModelSaver(str(outputs_dir))

        # Mock shutil.copy2 to raise an exception
        mock_copy = mocker.patch(
            "src.recbole_experiment.utils.model_saver.shutil.copy2"
        )
        mock_copy.side_effect = Exception("Copy failed")

        with patch(
            "src.recbole_experiment.utils.model_saver.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20250703_120000"
            result = saver.save_model("DeepFM", str(saved_dir))

        assert result is None

    def test_get_saved_models_empty(self, tmp_path):
        """Test get_saved_models with no saved models"""
        outputs_dir = tmp_path / "outputs"
        saver = ModelSaver(str(outputs_dir))

        result = saver.get_saved_models()
        assert result == []

    def test_get_saved_models_with_files(self, tmp_path):
        """Test get_saved_models with saved models"""
        outputs_dir = tmp_path / "outputs"
        saver = ModelSaver(str(outputs_dir))

        # Create some model files
        model1 = outputs_dir / "DeepFM_20250703_120000.pth"
        model1.write_text("model1")

        model2 = outputs_dir / "LR_20250703_120001.pth"
        model2.write_text("model2")

        result = saver.get_saved_models()
        assert len(result) == 2
        assert any("DeepFM_20250703_120000.pth" in path for path in result)
        assert any("LR_20250703_120001.pth" in path for path in result)

    def test_save_model_preserves_extension(self, tmp_path):
        """Test that save_model preserves file extension"""
        outputs_dir = tmp_path / "outputs"
        saved_dir = tmp_path / "saved"
        saved_dir.mkdir()

        # Create a model file with .pkl extension
        model_file = saved_dir / "CustomModel-Jun-29-2025_08-01-12.pkl"
        model_file.write_text("pickle model data")

        saver = ModelSaver(str(outputs_dir))

        with patch(
            "src.recbole_experiment.utils.model_saver.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20250703_120000"
            result = saver.save_model("CustomModel", str(saved_dir))

        assert result is not None
        assert "CustomModel_20250703_120000.pkl" in result

        output_file = outputs_dir / "CustomModel_20250703_120000.pkl"
        assert output_file.exists()
