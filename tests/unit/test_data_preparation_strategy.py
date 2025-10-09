"""
Unit tests for data preparation strategies.
"""
import pytest
import numpy as np
from src.data.data_preparation_strategy import (
    ValidationMetricsStrategy,
    TestMetricsStrategy,
    DataPreparationStrategyFactory
)
from src.data.data_splitter import DataSplitter
from src.constants import MetricsSource
from src.config import ValidationSplit


class TestValidationMetricsStrategy:
    """Test ValidationMetricsStrategy."""

    def test_strategy_with_pre_split_data(self):
        """Test strategy with pre-split data."""
        strategy = ValidationMetricsStrategy()

        np.random.seed(42)
        data = {
            'X_train': np.random.randn(80, 5),
            'y_train': np.random.randint(0, 2, 80),
            'X_test': np.random.randn(20, 5),
            'y_test': np.random.randint(0, 2, 20)
        }

        split = strategy.prepare(data, is_pre_split=True)

        assert isinstance(split, ValidationSplit)
        assert len(split.X_train) < 80  # Further split
        assert len(split.X_val) > 0

    def test_strategy_with_unsplit_data(self):
        """Test strategy with unsplit data."""
        strategy = ValidationMetricsStrategy()

        np.random.seed(42)
        data = {
            'X': np.random.randn(100, 5),
            'y': np.random.randint(0, 2, 100)
        }

        split = strategy.prepare(data, is_pre_split=False)

        assert isinstance(split, ValidationSplit)
        assert len(split.X_train) > 0
        assert len(split.X_val) > 0
        assert len(split.X_train) + len(split.X_val) == 100


class TestTestMetricsStrategy:
    """Test TestMetricsStrategy."""

    def test_strategy_with_pre_split_data(self):
        """Test strategy uses test set when data is pre-split."""
        strategy = TestMetricsStrategy()

        np.random.seed(42)
        X_test = np.random.randn(20, 5)
        y_test = np.random.randint(0, 2, 20)

        data = {
            'X_train': np.random.randn(80, 5),
            'y_train': np.random.randint(0, 2, 80),
            'X_test': X_test,
            'y_test': y_test
        }

        split = strategy.prepare(data, is_pre_split=True)

        assert isinstance(split, ValidationSplit)
        # Should use test data as validation
        np.testing.assert_array_equal(split.X_val, X_test)
        np.testing.assert_array_equal(split.y_val, y_test)

    def test_strategy_falls_back_for_unsplit_data(self):
        """Test strategy falls back to validation split for unsplit data."""
        strategy = TestMetricsStrategy()

        np.random.seed(42)
        data = {
            'X': np.random.randn(100, 5),
            'y': np.random.randint(0, 2, 100)
        }

        # Should log warning and fall back to validation split
        split = strategy.prepare(data, is_pre_split=False)

        assert isinstance(split, ValidationSplit)
        assert len(split.X_train) + len(split.X_val) == 100


class TestDataPreparationStrategyFactory:
    """Test DataPreparationStrategyFactory."""

    def test_factory_creates_validation_strategy(self):
        """Test factory creates ValidationMetricsStrategy."""
        strategy = DataPreparationStrategyFactory.create(MetricsSource.VALIDATION)

        assert isinstance(strategy, ValidationMetricsStrategy)

    def test_factory_creates_test_strategy(self):
        """Test factory creates TestMetricsStrategy."""
        strategy = DataPreparationStrategyFactory.create(MetricsSource.TEST)

        assert isinstance(strategy, TestMetricsStrategy)

    def test_factory_rejects_invalid_source(self):
        """Test factory raises error for invalid metrics source."""
        with pytest.raises(ValueError, match="Invalid metrics_source"):
            DataPreparationStrategyFactory.create("invalid")

    def test_factory_accepts_custom_splitter(self):
        """Test factory can use custom data splitter."""
        custom_splitter = DataSplitter(test_size=0.3)
        strategy = DataPreparationStrategyFactory.create(
            MetricsSource.VALIDATION,
            data_splitter=custom_splitter
        )

        assert strategy.data_splitter.test_size == 0.3
