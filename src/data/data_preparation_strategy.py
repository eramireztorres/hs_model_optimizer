"""
Strategy pattern for preparing data based on metrics source and split state.
"""
from abc import ABC, abstractmethod
import logging
from ..config import ValidationSplit
from .data_splitter import DataSplitter
from ..constants import MetricsSource


class DataPreparationStrategy(ABC):
    """Abstract base class for data preparation strategies."""

    def __init__(self, data_splitter: DataSplitter = None):
        """
        Initialize the strategy.

        Args:
            data_splitter (DataSplitter): Splitter for creating train/val splits.
        """
        self.data_splitter = data_splitter or DataSplitter()

    @abstractmethod
    def prepare(self, data: dict, is_pre_split: bool) -> ValidationSplit:
        """
        Prepare data for training and evaluation.

        Args:
            data (dict): Raw data dictionary.
            is_pre_split (bool): Whether data is already split into train/test.

        Returns:
            ValidationSplit: Prepared data with X_train, y_train, X_val, y_val.
        """
        pass


class ValidationMetricsStrategy(DataPreparationStrategy):
    """Strategy for using validation split from training data."""

    def prepare(self, data: dict, is_pre_split: bool) -> ValidationSplit:
        """
        Prepare data using validation metrics.
        Creates a validation split from training data or from full data.

        Args:
            data (dict): Data dictionary containing X_train, y_train or X, y.
            is_pre_split (bool): Whether data is already split.

        Returns:
            ValidationSplit: Train/validation split.
        """
        if is_pre_split:
            # Pre-split data: further split the training partition for validation
            return self.data_splitter.split_training_data(
                data['X_train'], data['y_train']
            )
        else:
            # Unsplit data: perform a single split to obtain validation set
            return self.data_splitter.create_train_val_split(
                data['X'], data['y']
            )


class TestMetricsStrategy(DataPreparationStrategy):
    """Strategy for using test data for metrics evaluation."""

    def prepare(self, data: dict, is_pre_split: bool) -> ValidationSplit:
        """
        Prepare data using test set for metrics.
        Falls back to validation if data is not pre-split.

        Args:
            data (dict): Data dictionary.
            is_pre_split (bool): Whether data is already split.

        Returns:
            ValidationSplit: Train/test or train/validation split.
        """
        if is_pre_split:
            # Use test data as validation data
            return ValidationSplit(
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_val=data['X_test'],
                y_val=data['y_test']
            )
        else:
            # For unsplit data, override to validation metrics
            logging.warning(
                "Unsplit data provided; overriding metrics_source to 'validation'."
            )
            return self.data_splitter.create_train_val_split(
                data['X'], data['y']
            )


class DataPreparationStrategyFactory:
    """Factory for creating data preparation strategies."""

    @staticmethod
    def create(metrics_source: MetricsSource, data_splitter: DataSplitter = None) -> DataPreparationStrategy:
        """
        Create appropriate data preparation strategy.

        Args:
            metrics_source (MetricsSource): Source for metrics (validation or test).
            data_splitter (DataSplitter): Optional custom data splitter.

        Returns:
            DataPreparationStrategy: Strategy instance.

        Raises:
            ValueError: If metrics_source is invalid.
        """
        if metrics_source == MetricsSource.VALIDATION:
            return ValidationMetricsStrategy(data_splitter)
        elif metrics_source == MetricsSource.TEST:
            return TestMetricsStrategy(data_splitter)
        else:
            raise ValueError(f"Invalid metrics_source: {metrics_source}")
