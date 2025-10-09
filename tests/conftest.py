"""
Pytest configuration and shared fixtures for hs_model_optimizer tests.
"""
import sys
import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock

# Add project root to path so src can be imported as a package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_classification_data():
    """Generate sample classification dataset."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    return X, y


@pytest.fixture
def sample_regression_data():
    """Generate sample regression dataset."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    return X, y


@pytest.fixture
def sample_train_test_split(sample_classification_data):
    """Generate pre-split train/test data."""
    X, y = sample_classification_data
    split_idx = 80

    return {
        'X_train': X[:split_idx],
        'y_train': y[:split_idx],
        'X_test': X[split_idx:],
        'y_test': y[split_idx:]
    }


@pytest.fixture
def sample_dataframe():
    """Generate sample pandas DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })


# ============================================================================
# Model Code Fixtures
# ============================================================================

@pytest.fixture
def valid_classification_model_code():
    """Valid model code for classification."""
    return """def load_model():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=10, random_state=42)
"""


@pytest.fixture
def valid_regression_model_code():
    """Valid model code for regression."""
    return """def load_model():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=10, random_state=42)
"""


@pytest.fixture
def invalid_model_code():
    """Invalid model code that will fail."""
    return """def load_model():
    from sklearn.ensemble import NonExistentModel
    return NonExistentModel()
"""


@pytest.fixture
def model_code_with_syntax_error():
    """Model code with syntax error."""
    return """def load_model()
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier()
"""


# ============================================================================
# LLM Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_model():
    """Mock LLM model API."""
    mock = Mock()
    mock.get_response = Mock(return_value="def load_model():\n    from sklearn.tree import DecisionTreeClassifier\n    return DecisionTreeClassifier(max_depth=5)")
    return mock


@pytest.fixture
def mock_llm_model_with_error():
    """Mock LLM model that returns None."""
    mock = Mock()
    mock.get_response = Mock(return_value=None)
    return mock


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def basic_optimizer_config(tmp_path):
    """Basic optimizer configuration."""
    from config import OptimizerConfig
    from constants import MetricsSource

    data_file = tmp_path / "test_data.csv"
    data_file.write_text("feature1,feature2,target\n1,2,0\n3,4,1\n")

    return OptimizerConfig(
        data_path=str(data_file),
        model='gpt-4o-mini',
        iterations=2,
        metrics_source=MetricsSource.VALIDATION,
        history_file_path=str(tmp_path / "history.joblib")
    )


# ============================================================================
# File System Fixtures
# ============================================================================

@pytest.fixture
def temp_model_file(tmp_path):
    """Temporary model file path."""
    return tmp_path / "dynamic_model.py"


@pytest.fixture
def temp_history_file(tmp_path):
    """Temporary history file path."""
    return tmp_path / "model_history.joblib"


@pytest.fixture
def temp_csv_file(tmp_path, sample_dataframe):
    """Temporary CSV file with sample data."""
    csv_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def simple_classifier():
    """Simple sklearn classifier."""
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(max_depth=3, random_state=42)


@pytest.fixture
def simple_regressor():
    """Simple sklearn regressor."""
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(max_depth=3, random_state=42)


@pytest.fixture
def trained_classifier(simple_classifier, sample_classification_data):
    """Pre-trained classifier."""
    X, y = sample_classification_data
    simple_classifier.fit(X, y)
    return simple_classifier


# ============================================================================
# Helper Functions
# ============================================================================

@pytest.fixture
def assert_valid_metrics():
    """Helper to validate metrics dictionary."""
    def _assert(metrics, task_type='classification'):
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

        if task_type == 'classification':
            assert 'accuracy' in metrics
            assert 0 <= metrics['accuracy'] <= 1
        elif task_type == 'regression':
            assert 'mean_squared_error' in metrics
            assert metrics['mean_squared_error'] >= 0

    return _assert


# ============================================================================
# Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_imports():
    """Clean up dynamically imported modules after each test."""
    yield
    # Remove any dynamic_model imports
    modules_to_remove = [key for key in sys.modules.keys() if 'dynamic_model' in key]
    for module in modules_to_remove:
        del sys.modules[module]
