import logging
import json

# class LLMImprover:
#     def __init__(self, llm_model, model_history=None):
#         """
#         Initialize the LLMImprover.

#         Args:
#             llm_model: The LLM model instance to query for suggestions.
#             model_history: A list of dictionaries containing model information and metrics.
#         """
#         self.llm_model = llm_model
#         self.model_history = model_history if model_history else []

#     def get_model_suggestions(self, current_model_code, metrics):
#         """
#         Ask the LLM for suggestions on model and hyperparameter improvements.

#         Args:
#             current_model_code: The Python code of the current model.
#             metrics: The performance metrics of the current model.

#         Returns:
#             str: The improved model code proposed by the LLM.
#         """
#         # Format the prompt with the current model code and the latest metrics
#         prompt = self._format_prompt(current_model_code, metrics)
        
#         # Query the LLM for improvements
#         try:
#             improved_code = self.llm_model.get_response(prompt)
#             return improved_code
#         except Exception as e:
#             logging.error(f"Error querying LLM for suggestions: {e}")
#             return None

#     def log_model_history(self, model_code, metrics):
#         """
#         Log the current model code and its metrics for future reference.

#         Args:
#             model_code: The Python code of the model.
#             metrics: The performance metrics of the model.
#         """
#         history_entry = {
#             'model_code': model_code,
#             'metrics': metrics
#         }
#         self.model_history.append(history_entry)
#         logging.info(f"Logged model history: {history_entry}")


#     def _format_prompt(self, current_model_code, metrics):
#         """
#         Formats the prompt to provide to the LLM with the model code and performance metrics.
    
#         Args:
#             current_model_code: The Python code of the current model.
#             metrics: The performance metrics of the current model.
    
#         Returns:
#             str: The formatted prompt for the LLM.
#         """
#         history_str = json.dumps(self.model_history, indent=2)
#         metrics_str = json.dumps(metrics, indent=2)
        
#         prompt = f"""
        # You are provided with the following Python model that implements a machine learning classifier:
    
        # {current_model_code}
    
        # Classification metrics for this model are:
        # {metrics_str}
    
        # Previous models and their performance metrics are:
        # {history_str}
    
        # Task:
        # Based on the given model and its performance, suggest improvements. You may either:
        #     - Change the model to a different classifier (e.g., XGBoost).
        #     - Adjust the hyperparameters of the current model, especially if the metrics are already high.
    
        # **Example 1**:
        # Previous Model:
        # def load_model():
        #     from sklearn.ensemble import ExtraTreesClassifier
        #     return ExtraTreesClassifier(n_estimators=1200, max_depth=60, min_samples_split=2, min_samples_leaf=1)
    
        # Metrics:
        # Accuracy: 0.936
        # Precision: 0.937
        # Recall: 0.936
        # F1 Score: 0.936
    
        # Suggested Improvement:
        # Since the metrics are strong, a small adjustment in hyperparameters:
        # def load_model():
        #     from sklearn.ensemble import ExtraTreesClassifier
        #     return ExtraTreesClassifier(n_estimators=1500, max_depth=70, min_samples_split=2, min_samples_leaf=1)
    
        # **Example 2**:
        # Previous Model:
        # def load_model():
        #     from sklearn.ensemble import RandomForestClassifier
        #     return RandomForestClassifier(n_estimators=800, max_depth=50, min_samples_split=2, min_samples_leaf=1)
    
        # Metrics:
        # Accuracy: 0.92
        # Precision: 0.92
        # Recall: 0.92
        # F1 Score: 0.92
    
        # Suggested Improvement:
        # Switch to a more powerful model like XGBoost for improved performance:
        # def load_model():
        #     from xgboost import XGBClassifier
        #     return XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.05, subsample=0.8)
    
        # Please ensure all necessary imports are included within the function.
        # Provide only executable Python code for the improved model without any comments, explanations, or markdown formatting.
    
        # Output:
        # Provide only the improved Python code that can replace the current model.
#         """
        
#         return prompt


# class LLMRegressionImprover:
#     def __init__(self, llm_model, model_history=None):
#         self.llm_model = llm_model
#         self.model_history = model_history if model_history else []

#     def get_model_suggestions(self, current_model_code, metrics):
#         prompt = self._format_prompt(current_model_code, metrics)
#         try:
#             improved_code = self.llm_model.get_response(prompt)
#             return improved_code
#         except Exception as e:
#             logging.error(f"Error querying LLM for regression suggestions: {e}")
#             return None

#     def log_model_history(self, model_code, metrics):
#         history_entry = {
#             'model_code': model_code,
#             'metrics': metrics
#         }
#         self.model_history.append(history_entry)


#     def _format_prompt(self, current_model_code, metrics):
#         """
#         Formats the prompt to provide to the LLM with the model code and performance metrics.
    
#         Args:
#             current_model_code: The Python code of the current regression model.
#             metrics: The performance metrics of the current regression model.
    
#         Returns:
#             str: The formatted prompt for the LLM.
#         """
#         history_str = json.dumps(self.model_history, indent=2)
#         metrics_str = json.dumps(metrics, indent=2)
    
#         prompt = f"""
        # You are provided with the following Python regression model:
    
        # {current_model_code}
    
        # Regression metrics for this model are:
        # {metrics_str}
    
        # Previous regression models and their performance metrics are:
        # {history_str}
    
        # Task:
        # Based on the given regression model and its performance, suggest improvements. You may either:
        #     - Change the regression model to a different one (e.g., XGBRegressor).
        #     - Adjust the hyperparameters of the current regression model, especially if the metrics are already strong.
    
        # **Example 1**:
        # Previous Model:
        # def load_model():
        #     from sklearn.linear_model import LinearRegression
        #     return LinearRegression()
    
        # Metrics:
        # Mean Squared Error: 250.75
        # R^2 Score: 0.85
    
        # Suggested Improvement:
        # Since the performance is decent but could be improved, switch to a more powerful model like XGBRegressor:
        # def load_model():
        #     from xgboost import XGBRegressor
        #     return XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    
        # **Example 2**:
        # Previous Model:
        # def load_model():
        #     from sklearn.ensemble import RandomForestRegressor
        #     return RandomForestRegressor(n_estimators=500, max_depth=50)
    
        # Metrics:
        # Mean Squared Error: 200.5
        # R^2 Score: 0.90
    
        # Suggested Improvement:
        # Since the model is already performing well, adjust the hyperparameters slightly for potential improvements:
        # def load_model():
        #     from sklearn.ensemble import RandomForestRegressor
        #     return RandomForestRegressor(n_estimators=700, max_depth=60, min_samples_split=5)
    
        # Please ensure all necessary imports are included within the function.
        # Provide only executable Python code for the improved regression model without any comments, explanations, or markdown formatting.
    
        # Output:
        # Provide only the improved Python code that can replace the current model.
#         """
        
#         return prompt

import os
import sys
sys.path.append(os.path.dirname(__file__))

prompt_file_path = os.path.join(os.path.dirname(__file__), 'prompts/classification_prompt.txt')
prompt_regression_file_path = os.path.join(os.path.dirname(__file__), 'prompts/regression_prompt.txt')


class LLMImprover:
    def __init__(self, llm_model, model_history=None, prompt_file_path=prompt_file_path):
        self.llm_model = llm_model
        self.model_history = model_history if model_history else []
        self.prompt_file_path = prompt_file_path

    def get_model_suggestions(self, current_model_code, metrics, extra_info="Not available"):
        """
        Ask the LLM for suggestions on model improvements with additional information.

        Args:
            current_model_code: The Python code of the current model.
            metrics: The performance metrics of the current model.
            extra_info: Additional information for the LLM prompt (e.g., class imbalance, noisy labels).

        Returns:
            str: The improved model code proposed by the LLM.
        """
        prompt = self._format_prompt(current_model_code, metrics, extra_info)

        try:
            improved_code = self.llm_model.get_response(prompt)
            return improved_code
        except Exception as e:
            logging.error(f"Error querying LLM for suggestions: {e}")
            return None
        
    def log_model_history(self, model_code, metrics):
        """
        Log the current model code and its metrics for future reference.
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
            current_model_code: The Python code of the current model.
            metrics: The performance metrics of the current model.
            extra_info: Additional information for the LLM prompt.

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
                extra_info=extra_info  # Include extra info in the prompt
            )
            return prompt
        except FileNotFoundError:
            logging.error(f"Prompt file not found: {self.prompt_file_path}")
            return None


class LLMRegressionImprover:
    def __init__(self, llm_model, model_history=None, prompt_file_path=prompt_regression_file_path):
        self.llm_model = llm_model
        self.model_history = model_history if model_history else []
        self.prompt_file_path = prompt_file_path

    def get_model_suggestions(self, current_model_code, metrics, extra_info="Not available"):
        """
        Ask the LLM for suggestions on regression model improvements with additional information.

        Args:
            current_model_code: The Python code of the current regression model.
            metrics: The performance metrics of the current regression model.
            extra_info: Additional information for the LLM prompt (e.g., noisy labels, outliers).

        Returns:
            str: The improved model code proposed by the LLM.
        """
        prompt = self._format_prompt(current_model_code, metrics, extra_info)

        try:
            improved_code = self.llm_model.get_response(prompt)
            return improved_code
        except Exception as e:
            logging.error(f"Error querying LLM for regression suggestions: {e}")
            return None

    def log_model_history(self, model_code, metrics):
        """
        Log the current model code and its metrics for future reference.
        """
        history_entry = {
            'model_code': model_code,
            'metrics': metrics
        }
        self.model_history.append(history_entry)


    def _format_prompt(self, current_model_code, metrics, extra_info):
        """
        Load the prompt template from a file and format it with the current regression model details and additional information.

        Args:
            current_model_code: The Python code of the current regression model.
            metrics: The performance metrics of the current regression model.
            extra_info: Additional information for the LLM prompt.

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
                extra_info=extra_info  # Include extra info in the prompt
            )
            return prompt
        except FileNotFoundError:
            logging.error(f"Prompt file not found: {self.prompt_file_path}")
            return None


# class LLMImprover:
#     def __init__(self, llm_model, model_history=None, prompt_file_path=prompt_file_path):
#         """
#         Initialize the LLMImprover.

#         Args:
#             llm_model: The LLM model instance to query for suggestions.
#             model_history: A list of dictionaries containing model information and metrics.
#             prompt_file_path: Path to the prompt template file.
#         """
#         self.llm_model = llm_model
#         self.model_history = model_history if model_history else []
#         self.prompt_file_path = prompt_file_path

#     def get_model_suggestions(self, current_model_code, metrics):
#         """
#         Ask the LLM for suggestions on model and hyperparameter improvements.

#         Args:
#             current_model_code: The Python code of the current model.
#             metrics: The performance metrics of the current model.

#         Returns:
#             str: The improved model code proposed by the LLM.
#         """
#         prompt = self._format_prompt(current_model_code, metrics)

#         # Query the LLM for improvements
#         try:
#             improved_code = self.llm_model.get_response(prompt)
#             return improved_code
#         except Exception as e:
#             logging.error(f"Error querying LLM for suggestions: {e}")
#             return None

#     def log_model_history(self, model_code, metrics):
#         """
#         Log the current model code and its metrics for future reference.
#         """
#         history_entry = {
#             'model_code': model_code,
#             'metrics': metrics
#         }
#         self.model_history.append(history_entry)
#         logging.info(f"Logged model history: {history_entry}")

#     def _format_prompt(self, current_model_code, metrics):
#         """
#         Load the prompt template from a file and format it with current model details.

#         Args:
#             current_model_code: The Python code of the current model.
#             metrics: The performance metrics of the current model.

#         Returns:
#             str: The formatted prompt.
#         """
#         try:
#             with open(self.prompt_file_path, 'r') as file:
#                 prompt_template = file.read()

#             history_str = json.dumps(self.model_history, indent=2)
#             metrics_str = json.dumps(metrics, indent=2)

#             # Insert the model code, metrics, and history into the prompt
#             prompt = prompt_template.format(
#                 current_model_code=current_model_code,
#                 metrics_str=metrics_str,
#                 history_str=history_str
#             )
#             return prompt
#         except FileNotFoundError:
#             logging.error(f"Prompt file not found: {self.prompt_file_path}")
#             return None


# class LLMRegressionImprover:
#     def __init__(self, llm_model, model_history=None, prompt_file_path=prompt_regression_file_path):
#         """
#         Initialize the LLMRegressionImprover.

#         Args:
#             llm_model: The LLM model instance to query for suggestions.
#             model_history: A list of dictionaries containing model information and metrics.
#             prompt_file_path: Path to the prompt template file.
#         """
#         self.llm_model = llm_model
#         self.model_history = model_history if model_history else []
#         self.prompt_file_path = prompt_file_path

#     def get_model_suggestions(self, current_model_code, metrics):
#         """
#         Ask the LLM for suggestions on regression model improvements.

#         Args:
#             current_model_code: The Python code of the current model.
#             metrics: The performance metrics of the current model.

#         Returns:
#             str: The improved model code proposed by the LLM.
#         """
#         prompt = self._format_prompt(current_model_code, metrics)

#         try:
#             improved_code = self.llm_model.get_response(prompt)
#             return improved_code
#         except Exception as e:
#             logging.error(f"Error querying LLM for regression suggestions: {e}")
#             return None

#     def log_model_history(self, model_code, metrics):
#         """
#         Log the current model code and its metrics for future reference.
#         """
#         history_entry = {
#             'model_code': model_code,
#             'metrics': metrics
#         }
#         self.model_history.append(history_entry)

#     def _format_prompt(self, current_model_code, metrics):
#         """
#         Load the prompt template from a file and format it with current regression model details.

#         Args:
#             current_model_code: The Python code of the current regression model.
#             metrics: The performance metrics of the current regression model.

#         Returns:
#             str: The formatted prompt.
#         """
#         try:
#             with open(self.prompt_file_path, 'r') as file:
#                 prompt_template = file.read()

#             history_str = json.dumps(self.model_history, indent=2)
#             metrics_str = json.dumps(metrics, indent=2)

#             # Insert the model code, metrics, and history into the prompt
#             prompt = prompt_template.format(
#                 current_model_code=current_model_code,
#                 metrics_str=metrics_str,
#                 history_str=history_str
#             )
#             return prompt
#         except FileNotFoundError:
#             logging.error(f"Prompt file not found: {self.prompt_file_path}")
#             return None
