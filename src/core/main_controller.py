import joblib
import logging
import re
import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.model_trainer import ModelTrainer, RegressionModelTrainer
from llm.llm_improver import LLMImprover,LLMRegressionImprover
from persistence.model_history_manager import ModelHistoryManager
from models.dynamic_model_updater import DynamicModelUpdater, DynamicRegressionModelUpdater
from llm.model_api_factory import ModelAPIFactory
from data.data_loader import DataLoader  
from llm.llm_code_cleaner import LLMCodeCleaner
from llm.error_corrector import ErrorCorrector



#%%

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
        self.error_corrector = ErrorCorrector(error_llm, error_prompt_path)
            

    def _initialize_llm_improver(self, model_provider, model):
        """
        Initialize the LLM improver dynamically based on the provider.
        """
        llm_model = ModelAPIFactory.get_model_api(provider=model_provider, model=model)
        if self.is_regression:
            return LLMRegressionImprover(llm_model)
        return LLMImprover(llm_model)
          
    # def _load_data(self):
    #     """
    #     Load the training and test data from the joblib file.

    #     Returns:
    #         dict: A dictionary containing X_train, y_train, X_test, and y_test.
    #     """
    #     try:
    #         data = joblib.load(self.joblib_file_path)
    #         logging.info(f"Data loaded successfully from {self.joblib_file_path}")
    #         return data
    #     except Exception as e:
    #         logging.error(f"Failed to load data from {self.joblib_file_path}: {e}")
    #         return None
    
    # def _load_data(self):
    #     """
    #     Load the training and test data using the DataLoader.
    
    #     Returns:
    #         dict: A dictionary containing X_train, y_train, X_test, and y_test.
    #     """
    #     try:
    #         data = DataLoader.load_data(self.joblib_file_path)  # Handle both file and directory inputs
    #         logging.info(f"Data loaded successfully from {self.joblib_file_path}")            
            
    #         if self.is_regression is None:
    #             self.is_regression = is_regression(data['y_train'])
            
    #         return data
    #     except Exception as e:
    #         logging.error(f"Failed to load data from {self.joblib_file_path}: {e}")
    #         return None
    

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
    
        if self.metrics_source == "validation":
            from sklearn.model_selection import train_test_split
    
        try:
            print(f'ITERATIONS: {iterations}')
            for iteration in range(iterations):
                print(f"\n=== Iteration {iteration + 1} ===")
    
                # Decide on metrics source: validation or test
                if self.metrics_source == "validation":
                    if self.is_pre_split:
                        # Pre-split data: further split the training partition for validation
                        X_train, X_val, y_train, y_val = train_test_split(
                            self.data['X_train'], self.data['y_train'], test_size=0.2, random_state=42
                        )
                    else:
                        # Unsplit data: perform a single split now to obtain a validation set
                        X_train, X_val, y_train, y_val = train_test_split(
                            self.data['X'], self.data['y'], test_size=0.2, random_state=42
                        )
                else:  # metrics_source == "test"
                    if self.is_pre_split:
                        X_train, y_train = self.data['X_train'], self.data['y_train']
                        X_val, y_val = self.data['X_test'], self.data['y_test']
                    else:
                        # For unsplit data, override to validation metrics (or raise an error)
                        logging.warning("Unsplit data provided; overriding metrics_source to 'validation'.")
                        X_train, X_val, y_train, y_val = train_test_split(
                            self.data['X'], self.data['y'], test_size=0.2, random_state=42
                        )
    
                retries = 0
                model = None
              
                while retries < max_retries:
                    model, error_msg = self.dynamic_updater.run_dynamic_model()
                    
                    if model is not None:
                        break
                    
                        
                    if self.error_corrector:
                        # Get current faulty code
                        current_code = self._get_dynamic_model_code()
                        
                       
                        # Get error correction
                        improved_code = self.error_corrector.get_error_fix(
                            current_code, error_msg
                        )
                        
                        print("\n=== CODE after ERROR correction ===")
                        print(improved_code)
                        
                    else:
                        # Original fallback                       
                      
                        
                        improved_code = self.llm_improver.get_model_suggestions(
                            last_valid_model_code, {}, self.extra_info
                        )
                        
                        
                    
                    # Log the response from the LLM
                    if improved_code:                     
                        
                        cleaner = LLMCodeCleaner()
                        improved_code = cleaner.clean_code(improved_code)
                        self.dynamic_updater.update_model_code(improved_code)
                        
                        # print(f"\n=== IMPROVED MODEL ITERATION {iteration + 1} ===")
                        # print(improved_code)  # Display the suggested code in the console
                        # logging.info(f"CLEANED code:\n{improved_code}")
                        
                        model, error_msg = self.dynamic_updater.run_dynamic_model()
                        
                    else:
                        logging.warning("No new suggestions received from LLM. Skipping retry.")
                        print("No new suggestions received from LLM. Skipping retry.")
                        continue
                    
                    retries += 1
                
                
                if model is None:
                    # logging.error(f"Exceeded maximum retries ({max_retries}) for iteration {iteration + 1}. Skipping iteration.")
                    # print(f"Exceeded maximum retries ({max_retries}) for iteration {iteration + 1}. Skipping iteration.")
                    # continue

                    logging.error(
                        f"Exceeded maximum retries ({max_retries}) for iteration {iteration + 1}. Skipping iteration.")
                    print(
                        f"Exceeded maximum retries ({max_retries}) for iteration {iteration + 1}. Skipping iteration.")

                    current_model_code = self._get_dynamic_model_code()
                    simplified_error = self._simplify_error(error_msg)
                    metrics = {"error": simplified_error}
                    self.history_manager.save_model_history(current_model_code, metrics)
                    self.llm_improver.log_model_history(current_model_code, metrics)
                    last_valid_model_code = current_model_code
                    continue
    
                print(f"Model for iteration {iteration + 1}: {model.__class__.__name__}")
    
                # Select the appropriate trainer (regression or classification)
                trainer_class = RegressionModelTrainer if self.is_regression else ModelTrainer
                self.model_trainer = trainer_class(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_val,
                    y_test=y_val
                )
    
                # Train and evaluate the model
                # self.model_trainer.train_model()
                # metrics = self.model_trainer.evaluate_model()
                
                try:
                    self.model_trainer.train_model()
                    metrics = self.model_trainer.evaluate_model()
                except Exception as e:
                    simplified_error = self._simplify_error(e)
                    logging.error(f"Training or evaluation failed: {simplified_error}")
                    print(f"Training or evaluation failed: {simplified_error}")
                    metrics = {"error": simplified_error}
            
                    current_model_code = self._get_dynamic_model_code()
                    self.history_manager.save_model_history(current_model_code, metrics)
                    self.llm_improver.log_model_history(current_model_code, metrics)
            
                    improved_code = self.llm_improver.get_model_suggestions(
                        current_model_code, metrics, extra_info=self.extra_info
                    )
                    if improved_code:
                        cleaner = LLMCodeCleaner()
                        improved_code = cleaner.clean_code(improved_code)
                        self.dynamic_updater.update_model_code(improved_code)
                    else:
                        logging.warning("No improvements suggested by the LLM in this iteration.")
            
                    last_valid_model_code = current_model_code
                    continue
    
                print(f"Metrics for iteration {iteration + 1}: {metrics}")
    
                # Save model to history and joblib if output_models_path is provided
                current_model_code = self._get_dynamic_model_code()
                self.history_manager.save_model_history(current_model_code, metrics)
    
                if self.output_models_path is not None:
                    self._save_model(iteration, model)
    
                # Update last valid model code
                last_valid_model_code = current_model_code
    
                self.llm_improver.log_model_history(current_model_code, metrics)
                
                
   
                # Get improved model code from the LLM
                improved_code = self.llm_improver.get_model_suggestions(
                    current_model_code, metrics, extra_info=self.extra_info
                )
    
                if improved_code:
                    # print("\n=== LLM Suggested Code ===")
                    # print(improved_code)  # Display the suggested code in the console
                    # logging.info(f"LLM suggested code:\n{improved_code}")   
                    
                    cleaner = LLMCodeCleaner()
                    improved_code = cleaner.clean_code(improved_code)
                    
                    print(f"\n=== IMPROVED MODEL ITERATION {iteration + 1} ===")
                    print(improved_code)  # Display the suggested code in the console
                    
                    self.dynamic_updater.update_model_code(improved_code)
                else:
                    logging.warning("No improvements suggested by the LLM in this iteration.")
                    print(f"No improvements suggested by the LLM in iteration {iteration}.")
                    
                print(f'FINISHED ITERATION {iteration + 1}')
                    
            
        except ChildProcessError:
            pass # Ignore this error, as it is likely caused by a library that uses multiprocessing
        finally:
            if original_model_code:
                self.dynamic_updater.update_model_code(original_model_code)
                print("Original model restored after iterations.")
                logging.info("Original model restored after iterations.")



    # def run(self, iterations=5, max_retries=1):
    #     """
    #     Run the training and improvement process for the specified number of iterations.
    #     Retries up to `max_retries` times if the LLM provides invalid code.
    #     """
    #     original_model_code = self._backup_original_model()
    #     last_valid_model_code = original_model_code
    #     if not original_model_code:
    #         logging.error("Failed to backup the original model. Exiting.")
    #         return
    
    #     if self.metrics_source == "validation":
    #         from sklearn.model_selection import train_test_split
    
    #     try:
    #         for iteration in range(iterations):
    #             print(f"\n=== Iteration {iteration + 1} ===")
    
    #             # Decide on metrics source: validation or test
    #             if self.metrics_source == "validation":
    #                 X_train, X_val, y_train, y_val = train_test_split(
    #                     self.data['X_train'], self.data['y_train'], test_size=0.2, random_state=42
    #                 )
    #             else:
    #                 X_train, y_train = self.data['X_train'], self.data['y_train']
    #                 X_val, y_val = self.data['X_test'], self.data['y_test']
    
    #             retries = 0
    #             model = None
              
    #             while retries < max_retries:
    #                 model, error_msg = self.dynamic_updater.run_dynamic_model()
                    
    #                 if model is not None:
    #                     break
                    
    #                 print(f'self.error_corrector: {self.error_corrector}')
                        
    #                 if self.error_corrector:
    #                     # Get current faulty code
    #                     current_code = self._get_dynamic_model_code()
                        
                        
    #                     print("\n=== CURRENT Code before ERROR correction ===")
    #                     print(current_code)
                        
    #                     # Get error correction
    #                     improved_code = self.error_corrector.get_error_fix(
    #                         current_code, error_msg
    #                     )
                        
    #                     print("\n=== CURRENT Code after ERROR correction ===")
    #                     print(improved_code)
                        
    #                 else:
    #                     # Original fallback                       
                      
                        
    #                     improved_code = self.llm_improver.get_model_suggestions(
    #                         last_valid_model_code, {}, self.extra_info
    #                     )
                        
    #                     print("\n=== CURRENT Code after retry ===")
    #                     print(improved_code)
                        
                    
    #                 # Log the response from the LLM
    #                 if improved_code:
    #                     print("\n=== LLM Suggested Code ===")
    #                     print(improved_code)  # Display the suggested code in the console
    #                     logging.info(f"LLM suggested code:\n{improved_code}")
    
    #                     # Clean and update the dynamic model code
    #                     # improved_code = re.sub(r'^```.*\n', '', improved_code).strip().strip('```').strip()
    #                     # improved_code = re.sub(r'^python\n', '', improved_code).strip()
                        
                        
    #                     cleaner = LLMCodeCleaner()
    #                     improved_code = cleaner.clean_code(improved_code)
    #                     self.dynamic_updater.update_model_code(improved_code)
                        
    #                     print("\n=== CLEANED Code ===")
    #                     print(improved_code)  # Display the suggested code in the console
    #                     logging.info(f"CLEANED code:\n{improved_code}")
                        
    #                     model, error_msg = self.dynamic_updater.run_dynamic_model()
                        
    #                 else:
    #                     logging.warning("No new suggestions received from LLM. Skipping retry.")
    #                     print("No new suggestions received from LLM. Skipping retry.")
    #                     continue
                    
    #                 retries += 1
                
                
    #             if model is None:
    #                 logging.error(f"Exceeded maximum retries ({max_retries}) for iteration {iteration + 1}. Skipping iteration.")
    #                 print(f"Exceeded maximum retries ({max_retries}) for iteration {iteration + 1}. Skipping iteration.")
    #                 continue
    
    #             print(f"Model for iteration {iteration + 1}: {model.__class__.__name__}")
    
    #             # Select the appropriate trainer (regression or classification)
    #             trainer_class = RegressionModelTrainer if self.is_regression else ModelTrainer
    #             self.model_trainer = trainer_class(
    #                 model=model,
    #                 X_train=X_train,
    #                 y_train=y_train,
    #                 X_test=X_val,
    #                 y_test=y_val
    #             )
    
    #             # Train and evaluate the model
    #             self.model_trainer.train_model()
    #             metrics = self.model_trainer.evaluate_model()
    
    #             print(f"Metrics for iteration {iteration + 1}: {metrics}")
    
    #             # Save model to history and joblib if output_models_path is provided
    #             current_model_code = self._get_dynamic_model_code()
    #             self.history_manager.save_model_history(current_model_code, metrics)
    
    #             if self.output_models_path is not None:
    #                 self._save_model(iteration, model)
    
    #             # Update last valid model code
    #             last_valid_model_code = current_model_code
    
    #             self.llm_improver.log_model_history(current_model_code, metrics)
                
                
    #             print("\n=== CURRENT Code before improvement ===")
    #             print(current_model_code)
    
    #             # Get improved model code from the LLM
    #             improved_code = self.llm_improver.get_model_suggestions(
    #                 current_model_code, metrics, extra_info=self.extra_info
    #             )
    
    #             if improved_code:
    #                 print("\n=== LLM Suggested Code ===")
    #                 print(improved_code)  # Display the suggested code in the console
    #                 logging.info(f"LLM suggested code:\n{improved_code}")
    
    #                 # improved_code = re.sub(r'^```.*\n', '', improved_code).strip().strip('```').strip()
    #                 # improved_code = re.sub(r'^python\n', '', improved_code).strip()
                    
                    
    #                 cleaner = LLMCodeCleaner()
    #                 improved_code = cleaner.clean_code(improved_code)
    #                 print(f'CLEANED CODED: \n {improved_code}')
                    
    #                 self.dynamic_updater.update_model_code(improved_code)
    #             else:
    #                 logging.warning("No improvements suggested by the LLM in this iteration.")
    #                 print("No improvements suggested by the LLM in this iteration.")
    
    #     finally:
    #         if original_model_code:
    #             self.dynamic_updater.update_model_code(original_model_code)
    #             print("Original model restored after iterations.")
    #             logging.info("Original model restored after iterations.")


    # def run(self, iterations=5, max_retries=1):
    #     """
    #     Run the training and improvement process for the specified number of iterations.
    #     Retries up to `max_retries` times if the LLM provides invalid code.
    #     """
    #     original_model_code = self._backup_original_model()
    #     last_valid_model_code = original_model_code
    #     if not original_model_code:
    #         logging.error("Failed to backup the original model. Exiting.")
    #         return
    
    #     if self.metrics_source == "validation":
    #         from sklearn.model_selection import train_test_split
    
    #     try:
    #         for iteration in range(iterations):
    #             print(f"\n=== Iteration {iteration + 1} ===")
    
    #             # Decide on metrics source: validation or test
    #             if self.metrics_source == "validation":
    #                 X_train, X_val, y_train, y_val = train_test_split(
    #                     self.data['X_train'], self.data['y_train'], test_size=0.2, random_state=42
    #                 )
    #             else:
    #                 X_train, y_train = self.data['X_train'], self.data['y_train']
    #                 X_val, y_val = self.data['X_test'], self.data['y_test']
    
    #             retries = 0
    #             model = None
    #             while retries < max_retries:
    #                 model = self.dynamic_updater.run_dynamic_model()
    #                 if model is not None:
    #                     break  # Exit retry loop if valid model is returned
    
    #                 logging.error(f"Invalid model returned by the dynamic model. Retrying... ({retries + 1}/{max_retries})")
    #                 improved_code = self.llm_improver.get_model_suggestions(
    #                     last_valid_model_code, {}, extra_info=self.extra_info
    #                 )
    
    #                 if improved_code:
    #                     improved_code = re.sub(r'^```.*\n', '', improved_code).strip().strip('```').strip()
    #                     improved_code = re.sub(r'^python\n', '', improved_code).strip()
    #                     self.dynamic_updater.update_model_code(improved_code)
    #                 else:
    #                     logging.warning("No new suggestions received from LLM. Skipping retry.")
    #                     break
    
    #                 retries += 1
    
    #             if model is None:
    #                 logging.error(f"Exceeded maximum retries ({max_retries}) for iteration {iteration + 1}. Skipping iteration.")
    #                 continue
    
    #             print(f"Model for iteration {iteration + 1}: {model.__class__.__name__}")
    
    #             # Select the appropriate trainer (regression or classification)
    #             trainer_class = RegressionModelTrainer if self.is_regression else ModelTrainer
    #             self.model_trainer = trainer_class(
    #                 model=model,
    #                 X_train=X_train,
    #                 y_train=y_train,
    #                 X_test=X_val,
    #                 y_test=y_val
    #             )
    
    #             # Train and evaluate the model
    #             self.model_trainer.train_model()
    #             metrics = self.model_trainer.evaluate_model()
    
    #             print(f"Metrics for iteration {iteration + 1}: {metrics}")
    
    #             # Save model to history and joblib if output_models_path is provided
    #             current_model_code = self._get_dynamic_model_code()
    #             self.history_manager.save_model_history(current_model_code, metrics)
    
    #             if self.output_models_path is not None:
    #                 self._save_model(iteration, model)
    
    #             # Update last valid model code
    #             last_valid_model_code = current_model_code
    
    #             self.llm_improver.log_model_history(current_model_code, metrics)
    
    #             # Get improved model code from the LLM
    #             improved_code = self.llm_improver.get_model_suggestions(
    #                 current_model_code, metrics, extra_info=self.extra_info
    #             )
    
    #             if improved_code:
    #                 improved_code = re.sub(r'^```.*\n', '', improved_code).strip().strip('```').strip()
    #                 improved_code = re.sub(r'^python\n', '', improved_code).strip()
    #                 self.dynamic_updater.update_model_code(improved_code)
    #             else:
    #                 logging.warning("No improvements suggested by the LLM in this iteration.")
    #                 print("No improvements suggested by the LLM in this iteration.")
    
    #     finally:
    #         if original_model_code:
    #             self.dynamic_updater.update_model_code(original_model_code)
    #             print("Original model restored after iterations.")
    #             logging.info("Original model restored after iterations.")




    # def run(self, iterations=5):
    #     """
    #     Run the training and improvement process for the specified number of iterations.
    #     """
    #     original_model_code = self._backup_original_model()
    #     if not original_model_code:
    #         logging.error("Failed to backup the original model. Exiting.")
    #         return
        
    #     if self.metrics_source == "validation":
    #         from sklearn.model_selection import train_test_split
    
    #     try:
    #         for iteration in range(iterations):
    #             print(f"\n=== Iteration {iteration + 1} ===")
    
    #             # Decide on metrics source: validation or test
    #             if self.metrics_source == "validation":
    #                 # Create a validation split from training data
    #                 X_train, X_val, y_train, y_val = train_test_split(
    #                     self.data['X_train'], self.data['y_train'], test_size=0.2, random_state=42
    #                 )
    #             else:
    #                 # Use the original test data
    #                 X_train, y_train = self.data['X_train'], self.data['y_train']
    #                 X_val, y_val = self.data['X_test'], self.data['y_test']
    
    #             model = self.dynamic_updater.run_dynamic_model()
    #             if model is None:
    #                 logging.error("No model returned by the dynamic model. Exiting.")
    #                 break
    
    #             print(f"Model for iteration {iteration + 1}: {model.__class__.__name__}")
    
    #             # Select the appropriate trainer (regression or classification)
    #             if self.is_regression:
    #                 self.model_trainer = RegressionModelTrainer(
    #                     model=model,
    #                     X_train=X_train,
    #                     y_train=y_train,
    #                     X_test=X_val,  # Use validation or test data
    #                     y_test=y_val   # Use validation or test data
    #                 )
    #             else:
    #                 self.model_trainer = ModelTrainer(
    #                     model=model,
    #                     X_train=X_train,
    #                     y_train=y_train,
    #                     X_test=X_val,  # Use validation or test data
    #                     y_test=y_val   # Use validation or test data
    #                 )
    
    #             # Train and evaluate the model
    #             self.model_trainer.train_model()
    #             metrics = self.model_trainer.evaluate_model()
    
    #             print(f"Metrics for iteration {iteration + 1}: {metrics}")
    
    #             # Save model to history and joblib if output_models_path is provided
    #             current_model_code = self._get_dynamic_model_code()
    #             self.history_manager.save_model_history(current_model_code, metrics)
    
    #             if self.output_models_path is not None:
    #                 self._save_model(iteration, model)
    
    #             self.llm_improver.log_model_history(current_model_code, metrics)
    
    #             # Get improved model code from the LLM
    #             improved_code = self.llm_improver.get_model_suggestions(
    #                 current_model_code, metrics, extra_info=self.extra_info
    #             )
    
    #             # Clean up the returned code
    #             if improved_code:
    #                 improved_code = re.sub(r'^```.*\n', '', improved_code).strip().strip('```').strip()
    #                 improved_code = re.sub(r'^python\n', '', improved_code).strip()
    #             else:
    #                 logging.warning("Improved code is None. Skipping update.")
    #                 improved_code = ""
    
    #             if improved_code:
    #                 print(f"Improved model code for iteration {iteration + 1} received from LLM.")
    #                 self.dynamic_updater.update_model_code(improved_code)
    #             else:
    #                 logging.warning("No improvements suggested by the LLM in this iteration.")
    #                 print("No improvements suggested by the LLM in this iteration.")
    
    #     finally:
    #         if original_model_code:
    #             self.dynamic_updater.update_model_code(original_model_code)
    #             print("Original model restored after iterations.")
    #             logging.info("Original model restored after iterations.")



    # def run(self, iterations=5):
    #     """
    #     Run the training and improvement process for the specified number of iterations.
    #     """
    #     original_model_code = self._backup_original_model()
    #     if not original_model_code:
    #         logging.error("Failed to backup the original model. Exiting.")
    #         return

    #     try:
    #         for iteration in range(iterations):
    #             print(f"\n=== Iteration {iteration + 1} ===")

    #             model = self.dynamic_updater.run_dynamic_model()
    #             if model is None:
    #                 logging.error("No model returned by the dynamic model. Exiting.")
    #                 break

    #             print(f"Model for iteration {iteration + 1}: {model.__class__.__name__}")

    #             if self.is_regression:
    #                 self.model_trainer = RegressionModelTrainer(
    #                     model=model,
    #                     X_train=self.data['X_train'],
    #                     y_train=self.data['y_train'],
    #                     X_test=self.data['X_test'],
    #                     y_test=self.data['y_test']
    #                 )
    #             else:
    #                 self.model_trainer = ModelTrainer(
    #                     model=model,
    #                     X_train=self.data['X_train'],
    #                     y_train=self.data['y_train'],
    #                     X_test=self.data['X_test'],
    #                     y_test=self.data['y_test']
    #                 )

    #             self.model_trainer.train_model()
    #             metrics = self.model_trainer.evaluate_model()

    #             print(f"Metrics for iteration {iteration + 1}: {metrics}")

    #             # Save model to history and joblib if output_models_path is provided
    #             current_model_code = self._get_dynamic_model_code()
    #             self.history_manager.save_model_history(current_model_code, metrics)

    #             if self.output_models_path is not None:
    #                 self._save_model(iteration, model)

    #             self.llm_improver.log_model_history(current_model_code, metrics)

    #             improved_code = self.llm_improver.get_model_suggestions(current_model_code, metrics, extra_info=self.extra_info)

    #             # Clean up the returned code
    #             # improved_code = re.sub(r'^```.*\n', '', improved_code).strip().strip('```').strip()
    #             # improved_code = re.sub(r'^python\n', '', improved_code).strip()
                
    #             if improved_code:
    #                 improved_code = re.sub(r'^```.*\n', '', improved_code).strip().strip('```').strip()
    #                 improved_code = re.sub(r'^python\n', '', improved_code).strip()
    #             else:
    #                 logging.warning("Improved code is None. Skipping update.")
    #                 improved_code = ""


    #             if improved_code:
    #                 print(f"Improved model code for iteration {iteration + 1} received from LLM.")
    #                 self.dynamic_updater.update_model_code(improved_code)
    #             else:
    #                 logging.warning("No improvements suggested by the LLM in this iteration.")
    #                 print("No improvements suggested by the LLM in this iteration.")

    #     finally:
    #         if original_model_code:
    #             self.dynamic_updater.update_model_code(original_model_code)
    #             print("Original model restored after iterations.")
    #             logging.info("Original model restored after iterations.")


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



