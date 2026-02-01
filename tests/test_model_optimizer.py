
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.core.main_controller import MainController
import pandas as pd
import numpy as np

class TestMainController:
    
    @pytest.fixture
    def mock_data(self):
        return {
            'X_train': pd.DataFrame({'a': [1, 2], 'b': [3, 4]}),
            'y_train': pd.Series([0, 1]),
            'X_test': pd.DataFrame({'a': [5], 'b': [6]}),
            'y_test': pd.Series([1]),
            'is_pre_split': True
        }

    @pytest.fixture
    def controller(self, mock_data, tmp_path):
        with patch('src.core.main_controller.DataLoader.load_data', return_value=mock_data):
            # Mock APIs to avoid initialization errors
            with patch('src.core.main_controller.ModelAPIFactory') as mock_factory:
                mock_factory.get_model_api.return_value = Mock()
                
                ctrl = MainController(
                    joblib_file_path="dummy_path",
                    model_provider="openai",
                    history_file_path=str(tmp_path / "history.joblib"),
                    is_regression_bool=False
                )
                # Ensure we don't actually read/write files during init unless mocked
                ctrl.dynamic_updater = Mock()
                ctrl.dynamic_updater.dynamic_file_path = "dummy_dynamic_path"
                ctrl.llm_improver = Mock()
                ctrl.error_corrector = None
                return ctrl

    def test_prepare_data_splits_validation(self, controller, mock_data):
        """Test data splitting when metrics_source is VALIDATION."""
        controller.metrics_source = "validation"
        
        # We need to mock train_test_split to verify it's called
        with patch('sklearn.model_selection.train_test_split') as mock_split:
            mock_split.return_value = (1, 2, 3, 4) # dummy return
            
            res = controller._prepare_data_splits()
            
            assert res == (1, 2, 3, 4)
            mock_split.assert_called_once()
            # Verify it used X_train/y_train from data
            args, kwargs = mock_split.call_args
            pd.testing.assert_frame_equal(args[0], mock_data['X_train'])

    def test_run_single_iteration_success(self, controller):
        """Test the run loop for one iteration where everything works."""
        
        # Mock dependencies for the loop
        controller._backup_original_model = Mock(return_value="original_code")
        controller._prepare_data_splits = Mock(return_value=(Mock(), Mock(), Mock(), Mock()))
        controller._get_valid_model = Mock(return_value=(Mock(), "valid_code"))
        controller._train_and_evaluate = Mock(return_value={'accuracy': 0.9})
        controller._save_and_log_results = Mock()
        controller._apply_next_improvements = Mock()
        controller._get_dynamic_model_code = Mock(return_value="current_code")
        
        controller.run(iterations=1)
        
        controller._prepare_data_splits.assert_called_once()
        controller._get_valid_model.assert_called_once()
        controller._train_and_evaluate.assert_called_once()
        controller._save_and_log_results.assert_called_once()
        controller._apply_next_improvements.assert_called_once()
    
    def test_run_handles_model_failure(self, controller):
        """Test that run skips iteration if no valid model is found."""
        controller._backup_original_model = Mock(return_value="original_code")
        controller._prepare_data_splits = Mock(return_value=(Mock(), Mock(), Mock(), Mock()))
        
        # Return None for model to simulate failure
        controller._get_valid_model = Mock(return_value=(None, "code")) 
        
        controller.run(iterations=1)
        
        # Ensure we didn't proceed to training
        assert not hasattr(controller, 'model_trainer') or controller.model_trainer is None 

