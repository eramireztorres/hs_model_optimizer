"""
Unit tests for DynamicModelUpdater.
"""
from src.models.dynamic_model_updater import DynamicModelUpdater, DynamicRegressionModelUpdater


class TestDynamicModelUpdater:
    """Test DynamicModelUpdater class."""

    def test_updater_initialization(self, temp_model_file):
        """Test DynamicModelUpdater can be initialized."""
        updater = DynamicModelUpdater(dynamic_file_path=str(temp_model_file))

        assert updater.dynamic_file_path == str(temp_model_file)
        assert updater.repository is not None

    def test_update_model_code(self, temp_model_file, valid_classification_model_code):
        """Test updating model code."""
        updater = DynamicModelUpdater(dynamic_file_path=str(temp_model_file))

        updater.update_model_code(valid_classification_model_code)

        # Verify file was written
        assert temp_model_file.exists()
        content = temp_model_file.read_text()
        assert "RandomForestClassifier" in content

    def test_run_dynamic_model_success(self, temp_model_file, valid_classification_model_code):
        """Test successfully loading a valid model."""
        updater = DynamicModelUpdater(dynamic_file_path=str(temp_model_file))

        updater.update_model_code(valid_classification_model_code)
        model, error = updater.run_dynamic_model()

        assert model is not None
        assert error is None
        assert model.__class__.__name__ == "RandomForestClassifier"

    def test_run_dynamic_model_with_invalid_code(self, temp_model_file, invalid_model_code):
        """Test loading invalid model code returns error."""
        updater = DynamicModelUpdater(dynamic_file_path=str(temp_model_file))

        updater.update_model_code(invalid_model_code)
        model, error = updater.run_dynamic_model()

        assert model is None
        assert error is not None
        assert isinstance(error, str)

    def test_run_dynamic_model_with_syntax_error(self, temp_model_file, model_code_with_syntax_error):
        """Test loading code with syntax error."""
        updater = DynamicModelUpdater(dynamic_file_path=str(temp_model_file))

        updater.update_model_code(model_code_with_syntax_error)
        model, error = updater.run_dynamic_model()

        assert model is None
        assert error is not None

    def test_model_reload_after_update(self, temp_model_file):
        """Test that model is reloaded after code update."""
        updater = DynamicModelUpdater(dynamic_file_path=str(temp_model_file))

        # Load first model
        code1 = """def load_model():
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(max_depth=3)
"""
        updater.update_model_code(code1)
        model1, _ = updater.run_dynamic_model()
        assert model1.__class__.__name__ == "DecisionTreeClassifier"

        # Update and load second model
        code2 = """def load_model():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=10)
"""
        updater.update_model_code(code2)
        model2, _ = updater.run_dynamic_model()
        assert model2.__class__.__name__ == "RandomForestClassifier"

        # Models should be different classes
        assert type(model1) != type(model2)


class TestDynamicRegressionModelUpdater:
    """Test DynamicRegressionModelUpdater class."""

    def test_regression_updater_initialization(self, temp_model_file):
        """Test DynamicRegressionModelUpdater can be initialized."""
        updater = DynamicRegressionModelUpdater(dynamic_file_path=str(temp_model_file))

        assert updater.dynamic_file_path == str(temp_model_file)
        assert updater.repository is not None

    def test_regression_model_loading(self, temp_model_file, valid_regression_model_code):
        """Test loading a regression model."""
        updater = DynamicRegressionModelUpdater(dynamic_file_path=str(temp_model_file))

        updater.update_model_code(valid_regression_model_code)
        model, error = updater.run_dynamic_model()

        assert model is not None
        assert error is None
        assert model.__class__.__name__ == "RandomForestRegressor"

    def test_regression_model_reload(self, temp_model_file):
        """Test that regression model reloads correctly."""
        updater = DynamicRegressionModelUpdater(dynamic_file_path=str(temp_model_file))

        # Load first model
        code1 = """def load_model():
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(max_depth=5)
"""
        updater.update_model_code(code1)
        model1, _ = updater.run_dynamic_model()

        # Load second model
        code2 = """def load_model():
    from sklearn.linear_model import LinearRegression
    return LinearRegression()
"""
        updater.update_model_code(code2)
        model2, _ = updater.run_dynamic_model()

        # Verify different models loaded
        assert model1.__class__.__name__ == "DecisionTreeRegressor"
        assert model2.__class__.__name__ == "LinearRegression"
