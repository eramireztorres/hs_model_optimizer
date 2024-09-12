import logging
import json

class LLMImprover:
    def __init__(self, llm_model, model_history=None):
        """
        Initialize the LLMImprover.

        Args:
            llm_model: The LLM model instance to query for suggestions.
            model_history: A list of dictionaries containing model information and metrics.
        """
        self.llm_model = llm_model
        self.model_history = model_history if model_history else []

    def get_model_suggestions(self, current_model_code, metrics):
        """
        Ask the LLM for suggestions on model and hyperparameter improvements.

        Args:
            current_model_code: The Python code of the current model.
            metrics: The performance metrics of the current model.

        Returns:
            str: The improved model code proposed by the LLM.
        """
        # Format the prompt with the current model code and the latest metrics
        prompt = self._format_prompt(current_model_code, metrics)
        
        # Query the LLM for improvements
        try:
            improved_code = self.llm_model.get_response(prompt)
            return improved_code
        except Exception as e:
            logging.error(f"Error querying LLM for suggestions: {e}")
            return None

    def log_model_history(self, model_code, metrics):
        """
        Log the current model code and its metrics for future reference.

        Args:
            model_code: The Python code of the model.
            metrics: The performance metrics of the model.
        """
        history_entry = {
            'model_code': model_code,
            'metrics': metrics
        }
        self.model_history.append(history_entry)
        logging.info(f"Logged model history: {history_entry}")

    def _format_prompt(self, current_model_code, metrics):
        """
        Formats the prompt to provide to the LLM with the model code and performance metrics.

        Args:
            current_model_code: The Python code of the current model.
            metrics: The performance metrics of the current model.

        Returns:
            str: The formatted prompt for the LLM.
        """
        history_str = json.dumps(self.model_history, indent=2)
        metrics_str = json.dumps(metrics, indent=2)
        
        prompt = f"""
        You are given a Python model that implements a machine learning classifier. Here is the current model code:

        {current_model_code}

        Below are the classification metrics for this model:
        {metrics_str}

        Previous model versions and their metrics are as follows:
        {history_str}

        Please suggest improvements to the model or its hyperparameters, aiming to improve performance.
        Provide only executable Python code for the new model.
        """
        
        return prompt
