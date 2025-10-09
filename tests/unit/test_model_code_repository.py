"""
Unit tests for ModelCodeRepository.
"""
import pytest
from pathlib import Path
from src.persistence.model_code_repository import ModelCodeRepository


class TestModelCodeRepository:
    """Test ModelCodeRepository class."""

    def test_repository_initialization(self, temp_model_file):
        """Test repository can be initialized."""
        repo = ModelCodeRepository(str(temp_model_file))

        assert repo.file_path == Path(temp_model_file)

    def test_save_code(self, temp_model_file, valid_classification_model_code):
        """Test saving code to file."""
        repo = ModelCodeRepository(str(temp_model_file))

        result = repo.save(valid_classification_model_code)

        assert result is True
        assert temp_model_file.exists()
        content = temp_model_file.read_text()
        assert content == valid_classification_model_code

    def test_load_code(self, temp_model_file, valid_classification_model_code):
        """Test loading code from file."""
        repo = ModelCodeRepository(str(temp_model_file))

        repo.save(valid_classification_model_code)
        loaded_code = repo.load()

        assert loaded_code == valid_classification_model_code

    def test_load_nonexistent_file(self, temp_model_file):
        """Test loading from nonexistent file returns None."""
        repo = ModelCodeRepository(str(temp_model_file))

        loaded_code = repo.load()

        assert loaded_code is None

    def test_backup_code(self, temp_model_file, valid_classification_model_code):
        """Test backing up code."""
        repo = ModelCodeRepository(str(temp_model_file))

        repo.save(valid_classification_model_code)
        backup = repo.backup()

        assert backup == valid_classification_model_code

    def test_exists_check(self, temp_model_file, valid_classification_model_code):
        """Test file existence check."""
        repo = ModelCodeRepository(str(temp_model_file))

        assert repo.exists() is False

        repo.save(valid_classification_model_code)

        assert repo.exists() is True

    def test_save_overwrites_existing(self, temp_model_file):
        """Test that save overwrites existing file."""
        repo = ModelCodeRepository(str(temp_model_file))

        code1 = "def load_model():\n    return 1"
        code2 = "def load_model():\n    return 2"

        repo.save(code1)
        repo.save(code2)

        loaded = repo.load()
        assert loaded == code2

    def test_save_handles_errors_gracefully(self, tmp_path):
        """Test save handles errors (e.g., invalid path)."""
        invalid_path = tmp_path / "nonexistent" / "subdir" / "file.py"
        repo = ModelCodeRepository(str(invalid_path))

        # Should return False on error, not raise exception
        result = repo.save("some code")

        assert result is False
