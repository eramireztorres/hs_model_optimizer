"""
Metrics calculation service following Dependency Inversion Principle.
"""
from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)


class MetricsCalculator(ABC):
    """Abstract base class for metrics calculation."""

    @abstractmethod
    def calculate(self, y_true, y_pred) -> dict:
        """
        Calculate metrics given true and predicted values.

        Args:
            y_true: True target values.
            y_pred: Predicted target values.

        Returns:
            dict: Dictionary of metrics.
        """


class ClassificationMetricsCalculator(MetricsCalculator):
    """Calculate classification metrics."""

    def calculate(self, y_true, y_pred) -> dict:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            dict: Classification metrics including accuracy, precision, recall, f1.
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": list(precision_score(y_true, y_pred, average=None)),
            "recall": list(recall_score(y_true, y_pred, average=None)),
            "f1_score": list(f1_score(y_true, y_pred, average=None)),
            "global_metrics": {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average='weighted')),
                "recall": float(recall_score(y_true, y_pred, average='weighted')),
                "f1_score": float(f1_score(y_true, y_pred, average='weighted'))
            }
        }
        return metrics


class RegressionMetricsCalculator(MetricsCalculator):
    """Calculate regression metrics."""

    def calculate(self, y_true, y_pred) -> dict:
        """
        Calculate regression metrics.

        Args:
            y_true: True target values.
            y_pred: Predicted target values.

        Returns:
            dict: Regression metrics including MSE and R².
        """
        metrics = {
            "mean_squared_error": float(mean_squared_error(y_true, y_pred)),
            "r2_score": float(r2_score(y_true, y_pred)),
            "mean_absolute_error": float(mean_absolute_error(y_true, y_pred))
        }
        return metrics


class MetricsCalculatorFactory:
    """Factory for creating metrics calculators."""

    @staticmethod
    def create(is_regression: bool = False) -> MetricsCalculator:
        """
        Create appropriate metrics calculator.

        Args:
            is_regression (bool): Whether the task is regression.

        Returns:
            MetricsCalculator: Appropriate calculator instance.
        """
        if is_regression:
            return RegressionMetricsCalculator()
        else:
            return ClassificationMetricsCalculator()
