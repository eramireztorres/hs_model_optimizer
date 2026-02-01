import joblib
import logging
import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.model_trainer import ModelTrainer, RegressionModelTrainer
from llm.llm_improver import LLMImprover, LLMRegressionImprover
from persistence.model_history_manager import ModelHistoryManager
from models.dynamic_model_updater import DynamicModelUpdater, DynamicRegressionModelUpdater
from llm.model_api_factory import ModelAPIFactory
from data.data_loader import DataLoader
from llm.llm_code_cleaner import LLMCodeCleaner
from llm.error_corrector import ErrorCorrector


# %%

class MainController:
    def __init__(self, joblib_file_path, model_provider, history_file_path, model=None,
                 is_regression_bool=None, extra_info="Not available", output_models_path=None,
                 metrics_source="validation",
                 error_model: str = None,
                 error_prompt_path: str = None,
                 initial_model_path: str = None):
        """
        Initialize the MainController.

        Args:
            joblib_file_path (str): Path to the joblib file containing training and test data.
            model_provider (str): The provider name for the LLM (e.g., "openai", "llama", "gemini").
            history_file_path: Path to the file where model history will be stored.
            is_regression_bool (bool): Whether the task is regression.
            extra_info (str): Additional information to include in the LLM prompt (e.g., class imbalance, noisy labels).
            output_models_path (str): Directory where the trained models will be saved. If None, models will not be saved.
        """
        self.joblib_file_path = joblib_file_path
        self.history_manager = ModelHistoryManager(history_file_path=history_file_path)
        self.is_regression = is_regression_bool
        if initial_model_path:
            try:
                with open(initial_model_path, 'r') as f:
                    init_code = f.read()
                updater = DynamicRegressionModelUpdater() if is_regression_bool else DynamicModelUpdater()
                updater.update_model_code(init_code)
            except Exception as e:
                print(f"Warning: could not load initial model from {initial_model_path}: {e}")
        self.data = self._load_data()
        self.extra_info = extra_info
        self.model_trainer = None
        self.output_models_path = output_models_path

        # Dynamically initialize the LLM model
        self.llm_improver = self._initialize_llm_improver(model_provider, model)

        # Choose appropriate dynamic updater
        self.dynamic_updater = DynamicRegressionModelUpdater() if is_regression_bool else DynamicModelUpdater()

        self.metrics_source = metrics_source

        # Add these members:
        self.error_corrector = None
        if error_model:
            error_llm = ModelAPIFactory.get_model_api(
                provider=ModelAPIFactory.get_provider_from_model(error_model),
                model=error_model
            )
        else:
            # Use the main model as the error model
            error_llm = ModelAPIFactory.get_model_api(
                provider=ModelAPIFactory.get_provider_from_model(model),  # Use the main model's provider
                model=model  # Use the main model
            )
        if error_prompt_path:
            self.error_corrector = ErrorCorrector(error_llm, error_prompt_path)

    def _initialize_llm_improver(self, model_provider, model):
        """
        Initialize the LLM improver dynamically based on the provider.
        """
        llm_model = ModelAPIFactory.get_model_api(provider=model_provider, model=model)
        if self.is_regression:
            return LLMRegressionImprover(llm_model)
        return LLMImprover(llm_model)

    def _load_data(self):
        try:
            data = DataLoader.load_data(self.joblib_file_path)
            logging.info(f"Data loaded successfully from {self.joblib_file_path}")

            # If the flag is missing (for backward compatibility), assume pre-split.
            self.is_pre_split = data.pop('is_pre_split', True)

            # (Optionally, set self.is_regression if not provided)
            if self.is_regression is None:
                if self.is_pre_split:
                    self.is_regression = is_regression(data['y_train'])
                else:
                    self.is_regression = is_regression(data['y'])

            return data
        except Exception as e:
            logging.error(f"Failed to load data from {self.joblib_file_path}: {e}")
            return None

    def run(self, iterations=5, max_retries=1):
        """
        Run the training and improvement process for the specified number of iterations.
        Retries up to `max_retries` times if the LLM provides invalid code.
        """
        original_model_code = self._backup_original_model()
        last_valid_model_code = original_model_code
        
        if not original_model_code:
            logging.error("Failed to backup the original model. Exiting.")
            return

        try:
            print(f'ITERATIONS: {iterations}')
            for iteration in range(iterations):
                print(f"\n=== Iteration {iteration + 1} ===")
                
                # 1. Prepare data (validation vs test split)
                X_train, X_val, y_train, y_val = self._prepare_data_splits()

                # 2. Get a valid model code (with retries if needed)
                model, last_valid_model_code = self._get_valid_model(
                    max_retries, last_valid_model_code, iteration
                )

                if model is None:
                    continue  # Skip this iteration if we couldn't get a working model

                print(f"Model for iteration {iteration + 1}: {model.__class__.__name__}")

                # 3. Train and Evaluate
                metrics = self._train_and_evaluate(model, X_train, y_train, X_val, y_val)
                print(f"Metrics for iteration {iteration + 1}: {metrics}")

                # 4. Handle errors if training/eval failed
                if "error" in metrics:
                    last_valid_model_code = self._handle_training_error(metrics, iteration)
                    continue

                # 5. Save and Log
                current_model_code = self._get_dynamic_model_code()
                self._save_and_log_results(iteration, model, metrics, current_model_code)
                last_valid_model_code = current_model_code

                # 6. Get Next Improvements (if not the last iteration)
                # Note: We usually want to get improvements for the next round even on the last one, 
                # or we can skip it. The original code gets it every time.
                self._apply_next_improvements(current_model_code, metrics, iteration)

        except ChildProcessError:
            pass
        finally:
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


            with open(self.dynamic_updater.dynamic_file_path, 'r') as f:
                original_model_code = f.read()
            return original_model_code
        except Exception as e:
            logging.error(f"Failed to backup original model: {e}")
            return None

    def _simplify_error(self, error) -> str:
        """
        Simplify exception/error message to a single-line string.
        """
        msg = str(error)
        return msg.splitlines()[-1] if msg else ''

    def _save_model(self, iteration, model):
        """
        Save the trained model to a joblib file after each iteration.
    
        Args:
            iteration (int): The current iteration number.
            model: The trained model to save.
        """
        if not self.output_models_path:
            return  # If no path is provided, we do not save the models

        # Ensure the output directory exists
        if not os.path.exists(self.output_models_path):
            os.makedirs(self.output_models_path)

        # Save the model to a separate file for each iteration
        model_path = os.path.join(self.output_models_path, f"model_{iteration + 1}.joblib")

        try:
            joblib.dump(model, model_path)
            logging.info(f"Model for iteration {iteration + 1} saved to {model_path}")
        except Exception as e:
            logging.error(f"Failed to save model for iteration {iteration + 1}: {e}")



    def _prepare_data_splits(self):
        """Prepare train/validation splits based on configuration."""
        from sklearn.model_selection import train_test_split

        if self.metrics_source == "validation":
            if self.is_pre_split:
                 # Pre-split data: further split the training partition for validation
                return train_test_split(
                    self.data['X_train'], self.data['y_train'], test_size=0.2, random_state=42
                )
            else:
                # Unsplit data: perform a single split now to obtain a validation set
                return train_test_split(
                    self.data['X'], self.data['y'], test_size=0.2, random_state=42
                )
        else:  # metrics_source == "test"
            if self.is_pre_split:
                return (
                    self.data['X_train'], 
                    self.data['X_test'], # using test as validation
                    self.data['y_train'], 
                    self.data['y_test']
                )
            else:
                logging.warning("Unsplit data provided; overriding metrics_source to 'validation'.")
                return train_test_split(
                    self.data['X'], self.data['y'], test_size=0.2, random_state=42
                )

    def _get_valid_model(self, max_retries, last_valid_model_code, iteration):
        """
        Attempt to get a valid model from dynamic_updater, retrying with LLM/ErrorCorrector if needed.
        Returns (model, new_last_valid_code). 
        If model is None, it means we failed after retries.
        """
        retries = 0
        model = None
        
        while retries < max_retries:
            model, error_msg = self.dynamic_updater.run_dynamic_model()

            if model is not None:
                return model, last_valid_model_code

            # Handle failure
            if self.error_corrector:
                current_code = self._get_dynamic_model_code()
                improved_code = self.error_corrector.get_error_fix(current_code, error_msg)
                print("\n=== CODE after ERROR correction ===")
                # print(improved_code)
            else:
                improved_code = self.llm_improver.get_model_suggestions(
                    last_valid_model_code, {}, self.extra_info
                )
            
            if improved_code:
                cleaner = LLMCodeCleaner()
                improved_code = cleaner.clean_code(improved_code)
                self.dynamic_updater.update_model_code(improved_code)
            else:
                logging.warning("No new suggestions received from LLM. Skipping retry.")
                print("No new suggestions received from LLM. Skipping retry.")
                # If no suggestion, we can't really retry meaningfully in the same way, but let's continue loop
                
            retries += 1

        # If we exit loop without model:
        logging.error(f"Exceeded maximum retries ({max_retries}) for iteration {iteration + 1}. Skipping iteration.")
        print(f"Exceeded maximum retries ({max_retries}) for iteration {iteration + 1}. Skipping iteration.")
        
        # Log the error state
        current_model_code = self._get_dynamic_model_code()
        # We don't have error_msg from the last attempt readily available if we didn't save it, 
        # but logic above implies we just failed.
        # Let's try to capture the last error_msg if possible or just generic.
        # For simplicity, we assume the last run failure caused this.
        metrics = {"error": "Exceeded max retries for model creation"}
        self.history_manager.save_model_history(current_model_code, metrics)
        self.llm_improver.log_model_history(current_model_code, metrics)
        
        # new last_valid_code is just the current one (even if broken, it's the latest state)
        return None, current_model_code

    def _train_and_evaluate(self, model, X_train, y_train, X_val, y_val):
        """Train and evaluate the model, catching any exceptions."""
        trainer_class = RegressionModelTrainer if self.is_regression else ModelTrainer
        self.model_trainer = trainer_class(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_val,
            y_test=y_val
        )

        try:
            self.model_trainer.train_model()
            metrics = self.model_trainer.evaluate_model()
            return metrics
        except Exception as e:
            simplified_error = self._simplify_error(e)
            logging.error(f"Training or evaluation failed: {simplified_error}")
            print(f"Training or evaluation failed: {simplified_error}")
            return {"error": simplified_error}

    def _handle_training_error(self, metrics, iteration):
        """Handle case where training/evaluation produced an error."""
        current_model_code = self._get_dynamic_model_code()
        self.history_manager.save_model_history(current_model_code, metrics)
        self.llm_improver.log_model_history(current_model_code, metrics)

        # Try to fix it immediately for next round/retry
        improved_code = self.llm_improver.get_model_suggestions(
            current_model_code, metrics, extra_info=self.extra_info
        )
        if improved_code:
            cleaner = LLMCodeCleaner()
            improved_code = cleaner.clean_code(improved_code)
            self.dynamic_updater.update_model_code(improved_code)
        else:
            logging.warning("No improvements suggested by the LLM after training error.")

        return current_model_code

    def _save_and_log_results(self, iteration, model, metrics, current_model_code):
        """Save results to history and disk."""
        self.history_manager.save_model_history(current_model_code, metrics)
        if self.output_models_path is not None:
            self._save_model(iteration, model)
        self.llm_improver.log_model_history(current_model_code, metrics)

    def _apply_next_improvements(self, current_model_code, metrics, iteration):
        """Ask LLM for improvements for the next iteration."""
        improved_code = self.llm_improver.get_model_suggestions(
            current_model_code, metrics, extra_info=self.extra_info
        )

        if improved_code:
            cleaner = LLMCodeCleaner()
            improved_code = cleaner.clean_code(improved_code)
            print(f"\n=== IMPROVED MODEL ITERATION {iteration + 1} ===")
            print(improved_code)
            self.dynamic_updater.update_model_code(improved_code)
        else:
            logging.warning("No improvements suggested by the LLM in this iteration.")
            print(f"No improvements suggested by the LLM in iteration {iteration}.")


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
