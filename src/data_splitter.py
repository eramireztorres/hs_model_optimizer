"""
Data splitting strategies for train/validation/test splits.
"""
from sklearn.model_selection import train_test_split
from config import ValidationSplit
from constants import DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE


class DataSplitter:
    """Handles splitting of data for model training and evaluation."""

    def __init__(self, test_size: float = DEFAULT_TEST_SIZE, random_state: int = DEFAULT_RANDOM_STATE):
        """
        Initialize the data splitter.

        Args:
            test_size (float): Proportion of data to use for validation/test.
            random_state (int): Random seed for reproducibility.
        """
        self.test_size = test_size
        self.random_state = random_state

    def create_train_val_split(self, X, y) -> ValidationSplit:
        """
        Create a train/validation split from full data.

        Args:
            X: Feature data.
            y: Target data.

        Returns:
            ValidationSplit: Container with X_train, y_train, X_val, y_val.
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        return ValidationSplit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )

    def split_training_data(self, X_train, y_train) -> ValidationSplit:
        """
        Further split training data into train/validation.

        Args:
            X_train: Training features.
            y_train: Training targets.

        Returns:
            ValidationSplit: Container with X_train, y_train, X_val, y_val.
        """
        return self.create_train_val_split(X_train, y_train)
