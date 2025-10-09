"""
Unit tests for ModelTrainer classes.
"""
import pytest
import numpy as np
from src.models.model_trainer import ModelTrainer, RegressionModelTrainer
from src.models.metrics_calculator import ClassificationMetricsCalculator, RegressionMetricsCalculator


class TestModelTrainer:
    """Test ModelTrainer class."""

    def test_trainer_initialization(self, simple_classifier, sample_classification_data):
        """Test ModelTrainer initialization."""
        X, y = sample_classification_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        trainer = ModelTrainer(
            model=simple_classifier,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        assert trainer.model is not None
        assert len(trainer.X_train) == 80
        assert len(trainer.X_test) == 20

    def test_train_model(self, simple_classifier, sample_classification_data):
        """Test training a model."""
        X, y = sample_classification_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        trainer = ModelTrainer(
            model=simple_classifier,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        trainer.train_model()

        # Check model is fitted
        assert hasattr(trainer.model, 'tree_')

    def test_evaluate_model(self, simple_classifier, sample_classification_data, assert_valid_metrics):
        """Test evaluating a model."""
        X, y = sample_classification_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        trainer = ModelTrainer(
            model=simple_classifier,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        trainer.train_model()
        metrics = trainer.evaluate_model()

        assert_valid_metrics(metrics, task_type='classification')

    def test_trainer_raises_error_without_model(self, sample_classification_data):
        """Test trainer raises error when model is None."""
        X, y = sample_classification_data

        trainer = ModelTrainer(
            model=None,
            X_train=X,
            y_train=y,
            X_test=X,
            y_test=y
        )

        with pytest.raises(ValueError, match="No model provided"):
            trainer.train_model()


class TestRegressionModelTrainer:
    """Test RegressionModelTrainer class."""

    def test_regression_trainer_initialization(self, simple_regressor, sample_regression_data):
        """Test RegressionModelTrainer initialization."""
        X, y = sample_regression_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        trainer = RegressionModelTrainer(
            model=simple_regressor,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        assert trainer.model is not None
        assert isinstance(trainer.metrics_calculator, RegressionMetricsCalculator)

    def test_regression_train_model(self, simple_regressor, sample_regression_data):
        """Test training a regression model."""
        X, y = sample_regression_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        trainer = RegressionModelTrainer(
            model=simple_regressor,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        trainer.train_model()

        # Check model is fitted
        assert hasattr(trainer.model, 'tree_')

    def test_regression_evaluate_model(self, simple_regressor, sample_regression_data, assert_valid_metrics):
        """Test evaluating a regression model."""
        X, y = sample_regression_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        trainer = RegressionModelTrainer(
            model=simple_regressor,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        trainer.train_model()
        metrics = trainer.evaluate_model()

        assert_valid_metrics(metrics, task_type='regression')

    @pytest.mark.slow
    def test_catboost_huber_loss_configuration(self, sample_regression_data):
        """Test that CatBoost Huber loss is automatically configured."""
        pytest.importorskip("catboost")
        from catboost import CatBoostRegressor
        from sklearn.ensemble import VotingRegressor
        from xgboost import XGBRegressor

        X, y = sample_regression_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        # Create model with Huber loss (without delta)
        xgb = XGBRegressor(n_estimators=10, max_depth=3)
        catboost = CatBoostRegressor(iterations=10, depth=3, loss_function='Huber', verbose=0)
        voting_model = VotingRegressor([('xgb', xgb), ('catboost', catboost)])

        trainer = RegressionModelTrainer(
            model=voting_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        # This should not raise an error
        trainer.train_model()

        # Verify delta was added to loss_function
        for est in trainer.model.estimators_:
            if isinstance(est, CatBoostRegressor):
                loss_fn = est.get_params().get('loss_function', '')
                assert 'delta=' in str(loss_fn), "Delta parameter should be added to Huber loss"
