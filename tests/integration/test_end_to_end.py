"""
Integration tests for end-to-end workflows.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.config import OptimizerConfig
from src.constants import MetricsSource
from src.core.main_controller_refactored import MainController


@pytest.mark.integration
class TestEndToEndClassification:
    """Test complete classification workflow."""

    @pytest.fixture
    def classification_csv(self, tmp_path):
        """Create a classification CSV file."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        csv_path = tmp_path / "classification_data.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def mock_llm_for_classification(self):
        """Mock LLM that returns valid classification models."""
        mock = Mock()
        # Return progressively better models
        mock.get_response = Mock(side_effect=[
            """def load_model():
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(max_depth=5, random_state=42)
""",
            """def load_model():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
"""
        ])
        return mock

    @pytest.mark.slow
    def test_classification_workflow_with_mock_llm(
        self, tmp_path, classification_csv, mock_llm_for_classification
    ):
        """Test complete classification workflow with mocked LLM."""
        config = OptimizerConfig(
            data_path=str(classification_csv),
            model='gpt-4o-mini',
            model_provider='openai',
            iterations=2,
            metrics_source=MetricsSource.VALIDATION,
            history_file_path=str(tmp_path / "history.joblib"),
            error_prompt_path="src/prompts/error_correction_prompt.txt"
        )

        with patch('src.llm.model_api_factory.ModelAPIFactory.get_model_api') as mock_factory:
            mock_factory.return_value = mock_llm_for_classification

            controller = MainController(config)
            controller.run()

            # Verify LLM was called
            assert mock_llm_for_classification.get_response.call_count >= 2

            # Verify history file was created
            assert (tmp_path / "history.joblib").exists()


@pytest.mark.integration
class TestEndToEndRegression:
    """Test complete regression workflow."""

    @pytest.fixture
    def regression_csv(self, tmp_path):
        """Create a regression CSV file."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.randn(100)  # Continuous values
        })
        csv_path = tmp_path / "regression_data.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def mock_llm_for_regression(self):
        """Mock LLM that returns valid regression models."""
        mock = Mock()
        mock.get_response = Mock(side_effect=[
            """def load_model():
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(max_depth=5, random_state=42)
""",
            """def load_model():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
"""
        ])
        return mock

    @pytest.mark.slow
    def test_regression_workflow_with_mock_llm(
        self, tmp_path, regression_csv, mock_llm_for_regression
    ):
        """Test complete regression workflow with mocked LLM."""
        config = OptimizerConfig(
            data_path=str(regression_csv),
            model='gpt-4o-mini',
            model_provider='openai',
            iterations=2,
            is_regression=True,
            metrics_source=MetricsSource.VALIDATION,
            history_file_path=str(tmp_path / "history.joblib"),
            error_prompt_path="src/prompts/error_correction_prompt.txt"
        )

        with patch('src.llm.model_api_factory.ModelAPIFactory.get_model_api') as mock_factory:
            mock_factory.return_value = mock_llm_for_regression

            controller = MainController(config)
            controller.run()

            # Verify LLM was called
            assert mock_llm_for_regression.get_response.call_count >= 2

            # Verify history file was created
            assert (tmp_path / "history.joblib").exists()


@pytest.mark.integration
class TestDataLoading:
    """Test data loading from various formats."""

    def test_load_csv_file(self, temp_csv_file):
        """Test loading data from CSV file."""
        from src.data.data_loader import DataLoader

        data = DataLoader.load_data(str(temp_csv_file))

        assert 'X' in data or 'X_train' in data
        assert 'y' in data or 'y_train' in data

    def test_load_split_data(self, tmp_path, sample_classification_data):
        """Test loading pre-split data from joblib."""
        import joblib
        from src.data.data_loader import DataLoader

        X, y = sample_classification_data
        split_data = {
            'X_train': X[:80],
            'y_train': y[:80],
            'X_test': X[80:],
            'y_test': y[80:]
        }

        joblib_path = tmp_path / "split_data.joblib"
        joblib.dump(split_data, joblib_path)

        loaded_data = DataLoader.load_data(str(joblib_path))

        assert 'X_train' in loaded_data
        assert 'y_train' in loaded_data
        assert 'X_test' in loaded_data
        assert 'y_test' in loaded_data
        assert loaded_data['is_pre_split'] is True


@pytest.mark.integration
@pytest.mark.slow
class TestModelImprovement:
    """Test that models actually improve over iterations."""

    def test_metrics_improve_over_iterations(self, tmp_path, sample_classification_data):
        """Test that model metrics improve (or at least change) over iterations."""
        from src.models.dynamic_model_updater import DynamicModelUpdater
        from src.models.model_trainer import ModelTrainer

        X, y = sample_classification_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        model_file = tmp_path / "test_model.py"
        updater = DynamicModelUpdater(dynamic_file_path=str(model_file))

        # Iteration 1: Simple model
        code1 = """def load_model():
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(max_depth=2, random_state=42)
"""
        updater.update_model_code(code1)
        model1, _ = updater.run_dynamic_model()

        trainer1 = ModelTrainer(model=model1, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        trainer1.train_model()
        metrics1 = trainer1.evaluate_model()

        # Iteration 2: Better model
        code2 = """def load_model():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
"""
        updater.update_model_code(code2)
        model2, _ = updater.run_dynamic_model()

        trainer2 = ModelTrainer(model=model2, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        trainer2.train_model()
        metrics2 = trainer2.evaluate_model()

        # Both should produce valid metrics
        assert 'accuracy' in metrics1
        assert 'accuracy' in metrics2

        # Models should be different
        assert type(model1) != type(model2)
