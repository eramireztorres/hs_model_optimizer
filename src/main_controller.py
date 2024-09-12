import joblib
import logging

from model_trainer import ModelTrainer
from llm_improver import LLMImprover
from model_history_manager import ModelHistoryManager
from dynamic_model_updater import DynamicModelUpdater

class MainController:
    def __init__(self, joblib_file_path, llm_model):
        """
        Initialize the MainController.

        Args:
            joblib_file_path (str): Path to the joblib file containing the training and test data.
            llm_model: The LLM model instance to query for model improvements.
        """
        self.joblib_file_path = joblib_file_path
        self.llm_model = llm_model
        self.data = self._load_data()
        self.history_manager = ModelHistoryManager()
        self.dynamic_updater = DynamicModelUpdater()
        self.model_trainer = None

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
        for iteration in range(iterations):
            logging.info(f"Starting iteration {iteration + 1}")

            # Step 1: Run the dynamically updated model
            model = self.dynamic_updater.run_dynamic_model()
            if model is None:
                logging.error("No model returned by the dynamic model. Exiting.")
                break

            # Step 2: Train and evaluate the model
            self.model_trainer = ModelTrainer(
                model=model,
                X_train=self.data['X_train'],
                y_train=self.data['y_train'],
                X_test=self.data['X_test'],
                y_test=self.data['y_test']
            )
            self.model_trainer.train_model()
            metrics = self.model_trainer.evaluate_model()

            # Step 3: Log the model and its performance
            current_model_code = self._get_dynamic_model_code()
            self.history_manager.save_model_history(current_model_code, metrics)

            # Step 4: Get suggestions from the LLM for improvements
            improved_code = self.llm_model.get_model_suggestions(current_model_code, metrics)
            if improved_code:
                # Step 5: Update the dynamic model with the improved code
                self.dynamic_updater.update_model_code(improved_code)
            else:
                logging.warning("No improvements suggested by the LLM in this iteration.")

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


# from src.data_handler import DataHandler
# from src.model_trainer import ModelTrainer
# from src.model_selector import ModelSelector
# from src.llm_improver import LLMImprover
# from src.hyperparameter_tuner import HyperparameterTuner
# from src.experiment_manager import ExperimentManager
# # from google.generativeai import GenerativeModel
# import os

# from src.gpt import Gpt4AnswerGenerator

# api_key = os.getenv('OPENAI_API_KEY')
# generator = Gpt4AnswerGenerator(api_key, model='gpt-4o')

# def main():
#     # Initialize components
#     data_handler = DataHandler()
#     model_trainer = ModelTrainer()
#     model_selector = ModelSelector()
#     llm_improver = LLMImprover(generator)
#     hyperparameter_tuner = HyperparameterTuner()

#     # Experiment Manager
#     experiment_manager = ExperimentManager(
#         data_handler, 
#         model_trainer, 
#         model_selector, 
#         llm_improver, 
#         hyperparameter_tuner,
#         max_iterations=5
#     )

#     # Run the experiment
#     best_model = experiment_manager.run_experiment(filepath='/home/erick.ramirez/repo/hs_model_optimizer/classification_data.joblib')

#     print("Best model trained successfully!")

# if __name__ == "__main__":
#     main()


