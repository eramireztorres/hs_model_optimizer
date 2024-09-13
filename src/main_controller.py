import joblib
import logging
import re
import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model_trainer import ModelTrainer
from llm_improver import LLMImprover
from model_history_manager import ModelHistoryManager
from dynamic_model_updater import DynamicModelUpdater
from gpt import Gpt4AnswerGenerator



#%%

class MainController:
    def __init__(self, joblib_file_path, llm_improver, history_file_path):
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
            with open(self.dynamic_updater.dynamic_file_path, 'r') as f:
                original_model_code = f.read()
            return original_model_code
        except Exception as e:
            logging.error(f"Failed to backup original model: {e}")
            return None



if __name__ == "__main__":
    
    api_key = os.getenv('OPENAI_API_KEY')
    generator = Gpt4AnswerGenerator(api_key, model='gpt-4o')
    
    llm_improver = LLMImprover(generator, model_history=None)
    
    history_file_path = 'skdata_model_history.joblib'
    train_data_file = '/home/erick.ramirez/repo/hs_model_optimizer/classification_data.joblib'
    
    # history_file_path = 'north_data_model_history.joblib'
    # train_data_file = '/home/erick.ramirez/repo/Project-PSI/cli/extracted_features_100_band.joblib'
    
    
    iterations = 5
    
    controller = MainController(train_data_file, llm_improver, history_file_path)
    controller.run(iterations=iterations)


# class MainController:
#     def __init__(self, joblib_file_path, llm_improver, history_file_path):
#         """
#         Initialize the MainController.

#         Args:
#             joblib_file_path (str): Path to the joblib file containing the training and test data.
#             llm_improver: The LLM model improver to query for model improvements.
#             history_file_path (str): Path to the file where model history will be stored.
#         """
#         self.joblib_file_path = joblib_file_path
#         self.llm_improver = llm_improver
#         self.data = self._load_data()
#         self.history_manager = ModelHistoryManager(history_file_path=history_file_path)
#         self.dynamic_updater = DynamicModelUpdater()
#         self.model_trainer = None

#     def _load_data(self):
#         """
#         Load the training and test data from the joblib file.

#         Returns:
#             dict: A dictionary containing X_train, y_train, X_test, and y_test.
#         """
#         try:
#             data = joblib.load(self.joblib_file_path)
#             logging.info(f"Data loaded successfully from {self.joblib_file_path}")
#             return data
#         except Exception as e:
#             logging.error(f"Failed to load data from {self.joblib_file_path}: {e}")
#             return None

#     def run(self, iterations=5):
#         """
#         Run the training and improvement process for the specified number of iterations.
    
#         Args:
#             iterations (int): The number of iterations to improve the model.
#         """
#         for iteration in range(iterations):
#             print(f"\n=== Iteration {iteration + 1} ===")
    
#             # Step 1: Run the dynamically updated model
#             model = self.dynamic_updater.run_dynamic_model()
#             if model is None:
#                 logging.error("No model returned by the dynamic model. Exiting.")
#                 break
    
#             print(f"Model for iteration {iteration + 1}: {model.__class__.__name__}")
    
#             # Step 2: Train and evaluate the model
#             self.model_trainer = ModelTrainer(
#                 model=model,
#                 X_train=self.data['X_train'],
#                 y_train=self.data['y_train'],
#                 X_test=self.data['X_test'],
#                 y_test=self.data['y_test']
#             )
#             self.model_trainer.train_model()
#             metrics = self.model_trainer.evaluate_model()
    
#             print(f"Metrics for iteration {iteration + 1}: {metrics}")
    
#             # Step 3: Log the model and its performance
#             current_model_code = self._get_dynamic_model_code()
#             self.history_manager.save_model_history(current_model_code, metrics)
    
#             # Step 4: Log the model history in LLMImprover
#             self.llm_improver.log_model_history(current_model_code, metrics)
    
#             # Step 5: Get suggestions from the LLM for improvements
#             improved_code = self.llm_improver.get_model_suggestions(current_model_code, metrics)
            
#             # Clean up the returned code
#             improved_code = re.sub(r'^```.*\n', '', improved_code).strip().strip('```').strip()
#             improved_code = re.sub(r'^python\n', '', improved_code).strip()
    
#             if improved_code:
#                 print(f"Improved model code for iteration {iteration + 1} received from LLM.")
#                 # Step 6: Update the dynamic model with the improved code
#                 self.dynamic_updater.update_model_code(improved_code)
#             else:
#                 logging.warning("No improvements suggested by the LLM in this iteration.")
#                 print("No improvements suggested by the LLM in this iteration.")

#     def _get_dynamic_model_code(self):
#         """
#         Retrieve the current Python code from the dynamic model file.

#         Returns:
#             str: The code inside the dynamic model file.
#         """
#         try:
#             with open(self.dynamic_updater.dynamic_file_path, 'r') as f:
#                 return f.read()
#         except Exception as e:
#             logging.error(f"Failed to read the dynamic model code: {e}")
#             return ""


# if __name__ == "__main__":
#     controller = MainController(train_data_file, llm_improver, history_file_path)
#     controller.run(iterations=iterations)
    

    


    # def run(self, iterations=5):
    #     """
    #     Run the training and improvement process for the specified number of iterations.

    #     Args:
    #         iterations (int): The number of iterations to improve the model.
    #     """
    #     for iteration in range(iterations):
    #         print(f"\n=== Iteration {iteration + 1} ===")

    #         # Step 1: Run the dynamically updated model
    #         model = self.dynamic_updater.run_dynamic_model()
    #         if model is None:
    #             logging.error("No model returned by the dynamic model. Exiting.")
    #             break

    #         print(f"Model for iteration {iteration + 1}: {model.__class__.__name__}")

    #         # Step 2: Train and evaluate the model
    #         self.model_trainer = ModelTrainer(
    #             model=model,
    #             X_train=self.data['X_train'],
    #             y_train=self.data['y_train'],
    #             X_test=self.data['X_test'],
    #             y_test=self.data['y_test']
    #         )
    #         self.model_trainer.train_model()
    #         metrics = self.model_trainer.evaluate_model()

    #         print(f"Metrics for iteration {iteration + 1}: {metrics}")

    #         # Step 3: Log the model and its performance
    #         current_model_code = self._get_dynamic_model_code()
    #         self.history_manager.save_model_history(current_model_code, metrics)

    #         # Step 4: Get suggestions from the LLM for improvements
    #         improved_code = self.llm_improver.get_model_suggestions(current_model_code, metrics)
            
    #         # Clean up the returned code
    #         improved_code = re.sub(r'^```.*\n', '', improved_code).strip().strip('```').strip()
    #         improved_code = re.sub(r'^python\n', '', improved_code).strip()

    #         if improved_code:
    #             print(f"Improved model code for iteration {iteration + 1} received from LLM.")
    #             # Step 5: Update the dynamic model with the improved code
    #             self.dynamic_updater.update_model_code(improved_code)
    #         else:
    #             logging.warning("No improvements suggested by the LLM in this iteration.")
    #             print("No improvements suggested by the LLM in this iteration.")

    


# class MainController:
#     def __init__(self, joblib_file_path, llm_improver):
#         """
#         Initialize the MainController.

#         Args:
#             joblib_file_path (str): Path to the joblib file containing the training and test data.
#             llm_improver: The LLM model improver to query for model improvements.
#         """
#         self.joblib_file_path = joblib_file_path
#         self.llm_improver = llm_improver
#         self.data = self._load_data()
#         self.history_manager = ModelHistoryManager(history_file_path=history_file_path)
#         self.dynamic_updater = DynamicModelUpdater()
#         self.model_trainer = None

#     def _load_data(self):
#         """
#         Load the training and test data from the joblib file.

#         Returns:
#             dict: A dictionary containing X_train, y_train, X_test, and y_test.
#         """
#         try:
#             data = joblib.load(self.joblib_file_path)
#             logging.info(f"Data loaded successfully from {self.joblib_file_path}")
#             return data
#         except Exception as e:
#             logging.error(f"Failed to load data from {self.joblib_file_path}: {e}")
#             return None

#     def run(self, iterations=5):
#         """
#         Run the training and improvement process for the specified number of iterations.

#         Args:
#             iterations (int): The number of iterations to improve the model.
#         """
#         for iteration in range(iterations):
#             logging.info(f"Starting iteration {iteration + 1}")

#             # Step 1: Run the dynamically updated model
#             model = self.dynamic_updater.run_dynamic_model()
#             if model is None:
#                 logging.error("No model returned by the dynamic model. Exiting.")
#                 break

#             # Step 2: Train and evaluate the model
#             self.model_trainer = ModelTrainer(
#                 model=model,
#                 X_train=self.data['X_train'],
#                 y_train=self.data['y_train'],
#                 X_test=self.data['X_test'],
#                 y_test=self.data['y_test']
#             )
#             self.model_trainer.train_model()
#             metrics = self.model_trainer.evaluate_model()

#             # Step 3: Log the model and its performance
#             current_model_code = self._get_dynamic_model_code()
#             self.history_manager.save_model_history(current_model_code, metrics)

#             # Step 4: Get suggestions from the LLM for improvements
#             improved_code = self.llm_improver.get_model_suggestions(current_model_code, metrics)
            
#             improved_code = re.sub(r'^```.*\n', '', improved_code).strip().strip('```').strip()
#             improved_code = re.sub(r'^python\n', '', improved_code).strip()
            
#             if improved_code:
#                 # Step 5: Update the dynamic model with the improved code
#                 self.dynamic_updater.update_model_code(improved_code)
#             else:
#                 logging.warning("No improvements suggested by the LLM in this iteration.")

#     def _get_dynamic_model_code(self):
#         """
#         Retrieve the current Python code from the dynamic model file.

#         Returns:
#             str: The code inside the dynamic model file.
#         """
#         try:
#             with open(self.dynamic_updater.dynamic_file_path, 'r') as f:
#                 return f.read()
#         except Exception as e:
#             logging.error(f"Failed to read the dynamic model code: {e}")
#             return ""

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


