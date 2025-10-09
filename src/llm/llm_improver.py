import logging
import json
from .providers.base_model_api import BaseModelAPI  # Ensure this import exists and points to the correct file.

import os
import sys
sys.path.append(os.path.dirname(__file__))

prompt_file_path = os.path.join(os.path.dirname(__file__), 'prompts/classification_prompt.txt')
prompt_regression_file_path = os.path.join(os.path.dirname(__file__), 'prompts/regression_prompt.txt')


class LLMImprover:
    def __init__(self, llm_model: BaseModelAPI, model_history=None, prompt_file_path=prompt_file_path):
        """
        Initialize the LLMImprover.

        Args:
            llm_model (BaseModelAPI): The LLM model instance to query for suggestions.
            model_history (list): A list of dictionaries containing model information and metrics.
            prompt_file_path (str): Path to the prompt template file.
        """
        self.llm_model = llm_model
        self.model_history = model_history if model_history else []
        self.prompt_file_path = prompt_file_path

    def get_model_suggestions(self, current_model_code, metrics, extra_info="Not available"):
        """
        Ask the LLM for suggestions on model improvements with additional information.

        Args:
            current_model_code (str): The Python code of the current model.
            metrics (dict): The performance metrics of the current model.
            extra_info (str): Additional information for the LLM prompt (e.g., class imbalance, noisy labels).

        Returns:
            str: The improved model code proposed by the LLM.
        """
        prompt = self._format_prompt(current_model_code, metrics, extra_info)

        try:
            # Query the LLM for suggestions using the generic `get_response` method
            improved_code = self.llm_model.get_response(prompt)
            return improved_code.strip() if improved_code else None
        except Exception as e:
            logging.error(f"Error querying LLM for suggestions: {e}")
            return None

    def log_model_history(self, model_code, metrics):
        """
        Log the current model code and its metrics for future reference.

        Args:
            model_code (str): The Python code of the model.
            metrics (dict): The performance metrics of the model.
        """
        history_entry = {
            'model_code': model_code,
            'metrics': metrics
        }
        self.model_history.append(history_entry)
        logging.info(f"Logged model history: {history_entry}")

    def _format_prompt(self, current_model_code, metrics, extra_info):
        """
        Load the prompt template from a file and format it with the current model details and additional information.

        Args:
            current_model_code (str): The Python code of the current model.
            metrics (dict): The performance metrics of the current model.
            extra_info (str): Additional information for the LLM prompt.

        Returns:
            str: The formatted prompt.
        """
        try:
            with open(self.prompt_file_path, 'r') as file:
                prompt_template = file.read()

            history_str = json.dumps(self.model_history, indent=2)
            metrics_str = json.dumps(metrics, indent=2)

            prompt = prompt_template.format(
                current_model_code=current_model_code,
                metrics_str=metrics_str,
                history_str=history_str,
                extra_info=extra_info
            )
            return prompt
        except FileNotFoundError:
            logging.error(f"Prompt file not found: {self.prompt_file_path}")
            return None


class LLMRegressionImprover(LLMImprover):
    def __init__(self, llm_model: BaseModelAPI, model_history=None, prompt_file_path=prompt_regression_file_path):
        """
        Initialize the LLMRegressionImprover.

        Args:
            llm_model (BaseModelAPI): The LLM model instance to query for regression-specific suggestions.
            model_history (list): A list of dictionaries containing regression model information and metrics.
            prompt_file_path (str): Path to the regression prompt template file.
        """
        super().__init__(llm_model, model_history, prompt_file_path)




