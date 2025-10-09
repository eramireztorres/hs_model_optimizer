import importlib
import os
import logging
import sys
sys.path.append(os.path.dirname(__file__))

from model_code_repository import ModelCodeRepository

dynamic_file_path = os.path.join(os.path.dirname(__file__), 'dynamic_model.py')
dynamic_regression_file_path = os.path.join(os.path.dirname(__file__), 'dynamic_regression_model.py')

#%%

class DynamicModelUpdater:
    def __init__(self, dynamic_file_path=dynamic_file_path, repository: ModelCodeRepository = None):
        """
        Initialize the DynamicModelUpdater.

        Args:
            dynamic_file_path (str): The path to the Python file that will be dynamically updated.
            repository (ModelCodeRepository): Repository for code persistence.
        """
        self.dynamic_file_path = dynamic_file_path
        self.repository = repository or ModelCodeRepository(dynamic_file_path)

    def update_model_code(self, new_model_code):
        """
        Update the dynamic model file with the new model code provided by the LLM.

        Args:
            new_model_code (str): The Python code for the new model and hyperparameters.
        """
        self.repository.save(new_model_code)

    def run_dynamic_model(self):
        """
        Run the dynamically updated `load_model()` method from the `dynamic_model.py` file.
        
        Returns:
            model: The model returned by the dynamically updated `load_model()` function.
        """
        try:
            # Invalidate cache and reload the module to pick up the latest changes
            module_name = os.path.splitext(os.path.basename(self.dynamic_file_path))[0]
            if module_name in sys.modules:
                del sys.modules[module_name]
            dynamic_module = importlib.import_module(module_name)
            importlib.reload(dynamic_module)
            
            # Execute the `load_model()` function and return the model
            model = dynamic_module.load_model()
            logging.info("Successfully loaded the dynamically updated model.")
            # return model
        
            return model, None # (success case)

        except Exception as e:
            logging.error(f"Failed to run dynamic model: {str(e)}")
            return None, str(e)  # Return both model and error message
    
class DynamicRegressionModelUpdater:
    def __init__(self, dynamic_file_path=dynamic_regression_file_path, repository: ModelCodeRepository = None):
        self.dynamic_file_path = dynamic_file_path
        self.dynamic_directory = os.path.dirname(os.path.abspath(self.dynamic_file_path))
        self.repository = repository or ModelCodeRepository(dynamic_file_path)

    def update_model_code(self, new_model_code):
        self.repository.save(new_model_code)

    def run_dynamic_model(self):
        try:
            if self.dynamic_directory not in sys.path:
                sys.path.append(self.dynamic_directory)

            module_name = os.path.splitext(os.path.basename(self.dynamic_file_path))[0]

            if module_name in sys.modules:
                del sys.modules[module_name]

            dynamic_module = importlib.import_module(module_name)
            importlib.reload(dynamic_module)

            model = dynamic_module.load_model()
            logging.info("Successfully loaded the dynamically updated regression model.")
            # return model
        
            return model, None # (success case)

        except Exception as e:
            logging.error(f"Failed to run dynamic regression model: {str(e)}")
            return None, str(e)  # Return both model and error message
