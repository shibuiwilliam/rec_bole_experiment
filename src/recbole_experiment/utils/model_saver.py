"""Model saving utilities."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


class ModelSaver:
    """Model saving utility class"""

    def __init__(self, outputs_dir: str = "outputs"):
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(exist_ok=True)

    def save_model(self, model_name: str, saved_dir: str = "saved") -> Optional[str]:
        """
        Save trained model to outputs directory with datetime format

        Args:
            model_name: Name of the model
            saved_dir: Directory where RecBole saves models

        Returns:
            Path to saved model file if successful, None if failed
        """
        saved_path = Path(saved_dir)

        # Find the latest model file for this model (any extension)
        model_files = list(saved_path.glob(f"{model_name}-*"))
        if not model_files:
            print(f"No saved model found for {model_name}")
            return None

        # Get the most recent model file
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create new filename with datetime format
        extension = latest_model.suffix
        new_filename = f"{model_name}_{timestamp}{extension}"
        output_path = self.outputs_dir / new_filename

        try:
            # Copy the model file to outputs directory
            shutil.copy2(latest_model, output_path)
            print(f"Model saved to: {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"Error saving model: {e}")
            return None

    def get_saved_models(self) -> list[str]:
        """Get list of saved models in outputs directory"""
        model_files = list(self.outputs_dir.glob("*.pth"))
        return [str(f) for f in model_files]
