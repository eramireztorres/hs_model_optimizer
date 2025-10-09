"""
Repository pattern for managing model code file operations.
"""
import logging
from pathlib import Path
from typing import Optional


class ModelCodeRepository:
    """Handles persistence of dynamic model code."""

    def __init__(self, file_path: str):
        """
        Initialize the repository.

        Args:
            file_path (str): Path to the dynamic model file.
        """
        self.file_path = Path(file_path)

    def save(self, code: str) -> bool:
        """
        Save model code to file.

        Args:
            code (str): Python code to save.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            self.file_path.write_text(code)
            logging.info(f"Model code saved to {self.file_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save model code to {self.file_path}: {e}")
            return False

    def load(self) -> Optional[str]:
        """
        Load model code from file.

        Returns:
            Optional[str]: Code if successful, None otherwise.
        """
        try:
            code = self.file_path.read_text()
            logging.debug(f"Model code loaded from {self.file_path}")
            return code
        except Exception as e:
            logging.error(f"Failed to load model code from {self.file_path}: {e}")
            return None

    def backup(self) -> Optional[str]:
        """
        Create a backup of the current model code.

        Returns:
            Optional[str]: Backed up code if successful, None otherwise.
        """
        code = self.load()
        if code:
            logging.info(f"Model code backed up from {self.file_path}")
        return code

    def exists(self) -> bool:
        """
        Check if the model code file exists.

        Returns:
            bool: True if file exists, False otherwise.
        """
        return self.file_path.exists()
