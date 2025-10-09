"""
Unit tests for DataSplitter.
"""
import pytest
import numpy as np
from src.data.data_splitter import DataSplitter
from src.config import ValidationSplit


class TestDataSplitter:
    """Test DataSplitter class."""

    def test_data_splitter_initialization(self):
        """Test DataSplitter can be initialized."""
        splitter = DataSplitter()
        assert splitter is not None
        assert splitter.test_size > 0
        assert splitter.random_state is not None

    def test_data_splitter_custom_parameters(self):
        """Test DataSplitter with custom parameters."""
        splitter = DataSplitter(test_size=0.3, random_state=123)
        assert splitter.test_size == 0.3
        assert splitter.random_state == 123

    def test_create_train_val_split(self, sample_classification_data):
        """Test creating train/validation split."""
        X, y = sample_classification_data
        splitter = DataSplitter()

        split = splitter.create_train_val_split(X, y)

        assert isinstance(split, ValidationSplit)
        assert len(split.X_train) + len(split.X_val) == len(X)
        assert len(split.y_train) + len(split.y_val) == len(y)
        assert len(split.X_train) > len(split.X_val)  # Train should be larger

    def test_split_training_data(self, sample_classification_data):
        """Test splitting training data."""
        X, y = sample_classification_data
        splitter = DataSplitter()

        split = splitter.split_training_data(X, y)

        assert isinstance(split, ValidationSplit)
        assert len(split.X_train) > 0
        assert len(split.X_val) > 0

    def test_split_deterministic_with_random_state(self):
        """Test that splits are deterministic with same random_state."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        splitter1 = DataSplitter(random_state=42)
        splitter2 = DataSplitter(random_state=42)

        split1 = splitter1.create_train_val_split(X, y)
        split2 = splitter2.create_train_val_split(X, y)

        np.testing.assert_array_equal(split1.X_train, split2.X_train)
        np.testing.assert_array_equal(split1.y_train, split2.y_train)

    def test_split_respects_test_size(self):
        """Test that split respects test_size parameter."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        test_size = 0.3
        splitter = DataSplitter(test_size=test_size)
        split = splitter.create_train_val_split(X, y)

        expected_val_size = int(100 * test_size)
        assert len(split.X_val) == expected_val_size
        assert len(split.X_train) == 100 - expected_val_size
