import logging
import inspect
import os

class ExperimentManager:
    def __init__(self, data_handler, model_trainer, model_selector, llm_improver, hyperparameter_tuner, max_iterations=5):
        """
        Initialize the ExperimentManager class.

        Args:
            data_handler (DataHandler): An instance of the DataHandler class.
            model_trainer (ModelTrainer): An instance of the ModelTrainer class.
            model_selector (ModelSelector): An instance of the ModelSelector class.
            llm_improver (LLMImprover): An instance of the LLMImprover class.
            hyperparameter_tuner (HyperparameterTuner): An instance of the HyperparameterTuner class.
            max_iterations (int): The maximum number of iterations for the experiment.
        """
        self.data_handler = data_handler
        self.model_trainer = model_trainer
        self.model_selector = model_selector
        self.llm_improver = llm_improver
        self.hyperparameter_tuner = hyperparameter_tuner
        self.max_iterations = max_iterations
        self.models_history = []
        self.metrics_history = []

    def run_experiment(self, filepath):
        """
        Run the experiment, managing the iterative process of model training, evaluation, and improvement.

        Args:
            filepath (str): Path to the joblib file containing the training and test data.

        Returns:
            object: The best-performing model after all iterations.
        """
        # Load data
        data = self.data_handler.load_data(filepath)
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']

        current_model = None
        for iteration in range(self.max_iterations):
            logging.info(f"Iteration {iteration + 1}/{self.max_iterations}")

            # Train the model in the current iteration (either the initial model or LLM-improved model)
            trained_model = self.model_trainer.train_model(current_model, X_train, y_train)

            # Evaluate the model and store metrics
            metrics = self.model_trainer.evaluate_model(trained_model, X_test, y_test)
            self.metrics_history.append(metrics)
            self.models_history.append(trained_model)

            logging.info(f"Model trained with metrics: {metrics}")

            # Select the best model so far
            best_model, best_metrics = self.model_selector.select_best_model(self.models_history, self.metrics_history)
            logging.info(f"Best model so far with metrics: {best_metrics}")

            # Use LLM to propose improvements
            logging.info("Requesting improvement suggestions from LLM...")
            models_code_history = [inspect.getsource(model.__class__) for model in self.models_history]
            improved_code = self.llm_improver.propose_improvement(models_code_history, self.metrics_history)

            # Apply improvements using the hot-swapping pattern
            if improved_code:
                self.hyperparameter_tuner.apply_corrected_method(improved_code)
                current_model = self.hyperparameter_tuner.tune_hyperparameters(best_model, improved_code)
            else:
                logging.info("No valid improvements from LLM, using the best model from previous iteration.")

        # Save the model history
        history_filepath = os.path.join(os.path.dirname(filepath), "model_history.joblib")
        self.data_handler.save_model_history(self.models_history, self.metrics_history, history_filepath)

        logging.info(f"Experiment completed. Best model saved to {history_filepath}")

        return best_model
