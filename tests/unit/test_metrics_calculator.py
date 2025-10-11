"""
Unit tests for MetricsCalculator classes.
"""
import numpy as np
from src.models.metrics_calculator import (
    ClassificationMetricsCalculator,
    RegressionMetricsCalculator,
    MetricsCalculatorFactory
)


class TestClassificationMetricsCalculator:
    """Test ClassificationMetricsCalculator."""

    def test_calculate_perfect_predictions(self):
        """Test metrics calculation with perfect predictions."""
        calculator = ClassificationMetricsCalculator()

        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])

        metrics = calculator.calculate(y_true, y_pred)

        assert metrics['accuracy'] == 1.0
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'global_metrics' in metrics
        assert metrics['global_metrics']['accuracy'] == 1.0

    def test_calculate_random_predictions(self):
        """Test metrics calculation with random predictions."""
        np.random.seed(42)
        calculator = ClassificationMetricsCalculator()

        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)

        metrics = calculator.calculate(y_true, y_pred)

        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 'global_metrics' in metrics

    def test_calculate_with_multiclass(self):
        """Test metrics with multiclass classification."""
        calculator = ClassificationMetricsCalculator()

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = calculator.calculate(y_true, y_pred)

        assert metrics['accuracy'] == 1.0
        assert len(metrics['precision']) == 3  # 3 classes
        assert len(metrics['recall']) == 3
        assert len(metrics['f1_score']) == 3


class TestRegressionMetricsCalculator:
    """Test RegressionMetricsCalculator."""

    def test_calculate_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        calculator = RegressionMetricsCalculator()

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = calculator.calculate(y_true, y_pred)

        assert metrics['mean_squared_error'] == 0.0
        assert metrics['r2_score'] == 1.0

    def test_calculate_with_error(self):
        """Test metrics with prediction error."""
        calculator = RegressionMetricsCalculator()

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        metrics = calculator.calculate(y_true, y_pred)

        assert metrics['mean_squared_error'] > 0
        assert metrics['r2_score'] < 1.0
        assert 'mean_absolute_error' in metrics

    def test_calculate_returns_all_metrics(self):
        """Test that all expected regression metrics are returned."""
        calculator = RegressionMetricsCalculator()

        y_true = np.random.randn(50)
        y_pred = np.random.randn(50)

        metrics = calculator.calculate(y_true, y_pred)

        required_metrics = ['mean_squared_error', 'r2_score', 'mean_absolute_error']
        for metric in required_metrics:
            assert metric in metrics


class TestMetricsCalculatorFactory:
    """Test MetricsCalculatorFactory."""

    def test_factory_creates_classification_calculator(self):
        """Test factory creates classification calculator."""
        calculator = MetricsCalculatorFactory.create(is_regression=False)

        assert isinstance(calculator, ClassificationMetricsCalculator)

    def test_factory_creates_regression_calculator(self):
        """Test factory creates regression calculator."""
        calculator = MetricsCalculatorFactory.create(is_regression=True)

        assert isinstance(calculator, RegressionMetricsCalculator)

    def test_factory_default_is_classification(self):
        """Test factory defaults to classification."""
        calculator = MetricsCalculatorFactory.create()

        assert isinstance(calculator, ClassificationMetricsCalculator)
