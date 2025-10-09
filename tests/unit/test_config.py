"""
Unit tests for config module.
"""
import pytest
from src.config import OptimizerConfig, ValidationSplit, TrainingData
from src.constants import MetricsSource, DEFAULT_ITERATIONS


class TestOptimizerConfig:
    """Test OptimizerConfig dataclass."""

    def test_config_creation_with_required_params(self):
        """Test creating config with only required parameters."""
        config = OptimizerConfig(data_path="/path/to/data.csv")

        assert config.data_path == "/path/to/data.csv"
        assert config.model is not None
        assert config.iterations == DEFAULT_ITERATIONS

    def test_config_creation_with_all_params(self):
        """Test creating config with all parameters."""
        config = OptimizerConfig(
            data_path="/path/to/data.csv",
            model="gpt-4o-mini",
            model_provider="openai",
            iterations=5,
            metrics_source=MetricsSource.VALIDATION
        )

        assert config.data_path == "/path/to/data.csv"
        assert config.model == "gpt-4o-mini"
        assert config.model_provider == "openai"
        assert config.iterations == 5
        assert config.metrics_source == MetricsSource.VALIDATION

    def test_config_validates_positive_iterations(self):
        """Test that config validates iterations must be positive."""
        with pytest.raises(ValueError, match="iterations must be at least 1"):
            OptimizerConfig(data_path="/path/to/data.csv", iterations=0)

        with pytest.raises(ValueError, match="iterations must be at least 1"):
            OptimizerConfig(data_path="/path/to/data.csv", iterations=-1)

    def test_config_validates_non_negative_max_retries(self):
        """Test that config validates max_retries must be non-negative."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            OptimizerConfig(data_path="/path/to/data.csv", max_retries=-1)

    def test_config_converts_string_metrics_source_to_enum(self):
        """Test that string metrics_source is converted to enum."""
        config = OptimizerConfig(
            data_path="/path/to/data.csv",
            metrics_source="validation"
        )

        assert isinstance(config.metrics_source, MetricsSource)
        assert config.metrics_source == MetricsSource.VALIDATION

    def test_config_default_values(self):
        """Test default values are set correctly."""
        config = OptimizerConfig(data_path="/path/to/data.csv")

        assert config.model_provider is None
        assert config.error_model is None
        assert config.is_regression is None
        assert config.metrics_source == MetricsSource.VALIDATION


class TestValidationSplit:
    """Test ValidationSplit dataclass."""

    def test_validation_split_creation(self):
        """Test creating a ValidationSplit."""
        import numpy as np

        X_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([0, 1])
        X_val = np.array([[5, 6]])
        y_val = np.array([1])

        split = ValidationSplit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )

        assert len(split.X_train) == 2
        assert len(split.y_train) == 2
        assert len(split.X_val) == 1
        assert len(split.y_val) == 1


class TestTrainingData:
    """Test TrainingData dataclass."""

    def test_training_data_creation(self):
        """Test creating TrainingData."""
        import numpy as np

        X_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([0, 1])
        X_test = np.array([[5, 6]])
        y_test = np.array([1])

        data = TrainingData(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        assert data.is_pre_split is True
        assert len(data.X_train) == 2
        assert len(data.X_test) == 1

    def test_training_data_pre_split_flag(self):
        """Test pre_split flag can be set."""
        import numpy as np

        data = TrainingData(
            X_train=np.array([[1, 2]]),
            y_train=np.array([0]),
            X_test=np.array([[3, 4]]),
            y_test=np.array([1]),
            is_pre_split=False
        )

        assert data.is_pre_split is False
