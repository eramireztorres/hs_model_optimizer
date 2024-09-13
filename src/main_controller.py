import joblib
import logging
import re
import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model_trainer import ModelTrainer, RegressionModelTrainer
from llm_improver import LLMImprover
from model_history_manager import ModelHistoryManager
from dynamic_model_updater import DynamicModelUpdater, DynamicRegressionModelUpdater
from gpt import Gpt4AnswerGenerator



#%%

class MainController:
    def __init__(self, joblib_file_path, llm_improver, history_file_path, is_regression_bool=False):
        """
        Initialize the MainController.

        Args:
            joblib_file_path (str): Path to the joblib file containing the training and test data.
            llm_improver: The LLM model improver to query for model improvements.
            history_file_path (str): Path to the file where model history will be stored.
        """
        self.joblib_file_path = joblib_file_path
        self.llm_improver = llm_improver
        self.data = self._load_data()
        self.history_manager = ModelHistoryManager(history_file_path=history_file_path)
        # self.dynamic_updater = DynamicModelUpdater()
        self.model_trainer = None
        
        # Decide whether to use regression or classification based on target values
        if is_regression_bool:
            self.dynamic_updater = DynamicRegressionModelUpdater()
        else:
            self.dynamic_updater = DynamicModelUpdater()
            
        self.is_regression = is_regression_bool

    def _load_data(self):
        """
        Load the training and test data from the joblib file.

        Returns:
            dict: A dictionary containing X_train, y_train, X_test, and y_test.
        """
        try:
            data = joblib.load(self.joblib_file_path)
            logging.info(f"Data loaded successfully from {self.joblib_file_path}")
            return data
        except Exception as e:
            logging.error(f"Failed to load data from {self.joblib_file_path}: {e}")
            return None

    def run(self, iterations=5):
        """
        Run the training and improvement process for the specified number of iterations.
    
        Args:
            iterations (int): The number of iterations to improve the model.
        """
        # Step 1: Backup the original model code
        original_model_code = self._backup_original_model()
        if not original_model_code:
            logging.error("Failed to backup the original model. Exiting.")
            return
    
        try:
            # Step 2: Run the iterations
            for iteration in range(iterations):
                print(f"\n=== Iteration {iteration + 1} ===")
    
                # Run the dynamically updated model
                model = self.dynamic_updater.run_dynamic_model()
                if model is None:
                    logging.error("No model returned by the dynamic model. Exiting.")
                    break
    
                print(f"Model for iteration {iteration + 1}: {model.__class__.__name__}")
    
                # Train and evaluate the model
                if self.is_regression:
                    self.model_trainer = RegressionModelTrainer(
                        model=model,
                        X_train=self.data['X_train'],
                        y_train=self.data['y_train'],
                        X_test=self.data['X_test'],
                        y_test=self.data['y_test']
                    )
                else:
                    self.model_trainer = ModelTrainer(
                        model=model,
                        X_train=self.data['X_train'],
                        y_train=self.data['y_train'],
                        X_test=self.data['X_test'],
                        y_test=self.data['y_test']
                    )
                    
                self.model_trainer.train_model()
                metrics = self.model_trainer.evaluate_model()
    
                print(f"Metrics for iteration {iteration + 1}: {metrics}")
    
                # Log the model and its performance
                current_model_code = self._get_dynamic_model_code()
                self.history_manager.save_model_history(current_model_code, metrics)
    
                # Log the model history in LLMImprover
                self.llm_improver.log_model_history(current_model_code, metrics)
    
                # Get suggestions from the LLM for improvements
                improved_code = self.llm_improver.get_model_suggestions(current_model_code, metrics)
    
                # Clean up the returned code
                improved_code = re.sub(r'^```.*\n', '', improved_code).strip().strip('```').strip()
                improved_code = re.sub(r'^python\n', '', improved_code).strip()
    
                if improved_code:
                    print(f"Improved model code for iteration {iteration + 1} received from LLM.")
                    self.dynamic_updater.update_model_code(improved_code)
                else:
                    logging.warning("No improvements suggested by the LLM in this iteration.")
                    print("No improvements suggested by the LLM in this iteration.")
        
        finally:
            # Step 3: Restore the original model code
            if original_model_code:
                self.dynamic_updater.update_model_code(original_model_code)
                print("Original model restored after iterations.")
                logging.info("Original model restored after iterations.")



    def _get_dynamic_model_code(self):
        """
        Retrieve the current Python code from the dynamic model file.

        Returns:
            str: The code inside the dynamic model file.
        """
        try:
            with open(self.dynamic_updater.dynamic_file_path, 'r') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Failed to read the dynamic model code: {e}")
            return ""
        
    def _backup_original_model(self):
        """
        Backup the original model code from dynamic_model.py.
        """
        try:
            print(f'DYNAMIC PATH: {self.dynamic_updater.dynamic_file_path}')
            
            with open(self.dynamic_updater.dynamic_file_path, 'r') as f:
                original_model_code = f.read()
            return original_model_code
        except Exception as e:
            logging.error(f"Failed to backup original model: {e}")
            return None


import numpy as np

def is_regression(y_train):
    """
    Check if the target values suggest a regression problem.
    Regression typically has continuous target values (e.g., floats).
    This function checks if all values are exact integers, even if they are of type float.
    
    Args:
        y_train (array-like): The target values from the training set.

    Returns:
        bool: True if the problem is regression, False if it's classification.
    """
    # If the target array contains floats but all values are actually integers
    if np.issubdtype(y_train.dtype, np.floating):
        # Check if all float values are actually integers
        if np.all(np.equal(np.mod(y_train, 1), 0)):
            return False  # This suggests it's a classification problem with integer-like floats

    # Otherwise, treat it as a regression problem if it's not an integer-like float array
    return np.issubdtype(y_train.dtype, np.floating) or np.issubdtype(y_train.dtype, np.integer) and not np.all(np.equal(np.mod(y_train, 1), 0))




if __name__ == "__main__":
    
    api_key = os.getenv('OPENAI_API_KEY')
    generator = Gpt4AnswerGenerator(api_key, model='gpt-4o')
    
    llm_improver = LLMImprover(generator, model_history=None)
    
    history_file_path = 'skdata_model_history.joblib'
    train_data_file = 'classification_data.joblib'
 
    
    iterations = 5
    
    controller = MainController(train_data_file, llm_improver, history_file_path)
    controller.run(iterations=iterations)

