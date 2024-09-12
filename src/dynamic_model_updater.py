import importlib
import os
import logging
import sys

class DynamicModelUpdater:
    def __init__(self, dynamic_file_path='dynamic_model.py'):
        """
        Initialize the DynamicModelUpdater.

        Args:
            dynamic_file_path (str): The path to the Python file that will be dynamically updated.
        """
        self.dynamic_file_path = dynamic_file_path

    def update_model_code(self, new_model_code):
        """
        Update the `dynamic_model.py` file with the new model code provided by the LLM.

        Args:
            new_model_code (str): The Python code for the new model and hyperparameters.
        """
        try:
            with open(self.dynamic_file_path, 'w') as f:
                f.write(new_model_code)
            logging.info(f"Updated model code in {self.dynamic_file_path}")
        except Exception as e:
            logging.error(f"Failed to update the dynamic model file: {e}")

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
            return model

        except Exception as e:
            logging.error(f"Failed to run the dynamic model: {e}")
            return None
