from joblib import load
import sys
import os
sys.path.append(os.path.dirname(__file__))

from cli_decorator import cli_decorator
from main_controller import MainController, is_regression
from model_api_factory import ModelAPIFactory
# from gpt import Gpt4AnswerGenerator
# from llm_improver import LLMImprover, LLMRegressionImprover

#%%
@cli_decorator
def select_model_cli(data,
                     model: str = 'meta-llama/llama-3.1-405b-instruct:free',
                     model_provider: str = None,
                     history_file_path: str = 'model_history.joblib',
                     iterations: int = 5,
                     extra_info: str = 'Not available',
                     output_models_path: str = None):
    """
    Args:
        model (str, optional): The LLM model name (e.g., 'gpt-4', 'llama-3.1').
        model_provider (str, optional): The LLM provider name ('openai', 'llama', 'gemini').
        ...
    """
    if not model_provider:
        model_provider = ModelAPIFactory.get_provider_from_model(model)

    print(f"Using model: {model} (provider: {model_provider})")

    is_regression_bool = is_regression(load(data)['y_train'])

    controller = MainController(
        joblib_file_path=data,
        model_provider=model_provider,
        model=model,
        history_file_path=history_file_path,
        is_regression_bool=is_regression_bool,
        extra_info=extra_info,
        output_models_path=output_models_path
    )
    controller.run(iterations=iterations)


# api_key = os.getenv('OPENAI_API_KEY')

# @cli_decorator
# def select_model_cli(data,
#                      model: str = 'gpt-4o',
#                      model_provider: str = None,  # Optional if model is provided
#                      history_file_path: str = 'model_history.joblib',
#                      iterations: int = 5,
#                      extra_info: str = 'Not available',
#                      output_models_path: str = None):
#     """
#     Args:
#         model (str, optional): The LLM model name (e.g., 'gpt-4', 'llama-3.1').
#         model_provider (str, optional): The LLM provider name ('openai', 'llama', 'gemini').
#         ...
#     """
#     # Deduce provider if not explicitly provided
#     if not model_provider:
#         try:
#             model_provider = ModelAPIFactory.get_provider_from_model(model)
#         except ValueError as e:
#             raise ValueError(f"Error deducing provider from model: {e}")

#     # Check if regression or classification
#     is_regression_bool = is_regression(load(data)['y_train'])

#     # Initialize the main controller
#     controller = MainController(
#         joblib_file_path=data,
#         model_provider=model_provider,
#         history_file_path=history_file_path,
#         is_regression_bool=is_regression_bool,
#         extra_info=extra_info,
#         output_models_path=output_models_path
#     )
#     controller.run(iterations=iterations)



# @cli_decorator
# def select_model_cli(data,
                     
#         history_file_path: str = 'model_history.joblib',
#         model: str = 'gpt-4o-mini',
#         iterations: int = 5,
#         extra_info: str = 'Not available',  # Add extra_info parameter
#         output_models_path: str = None  # Add output_models_path parameter
        
#         ):
    
#     """
# - Selects and optimizes a machine learning model for classification or regression using an LLM improver, iterating through models and hyperparameters based on the specified configuration.

# Args:
# - data (dict): A dictionary containing training and test data, with keys such as 'X_train', 'y_train', 'X_test', 'y_test'. These should be NumPy arrays representing the feature and target datasets for model training and evaluation.
# - history_file_path (str, optional): Path to the joblib file where the model history will be stored. The history includes models, their hyperparameters, and performance metrics for each iteration. Default is 'model_history.joblib'.
# - model (str, optional): The name of the LLM model to use for generating suggestions and improvements for models and hyperparameters. Defaults to 'gpt-4o-mini'.
# - iterations (int, optional): The number of iterations to run, where each iteration involves training a model, evaluating its performance, and generating improvements. Default is 5.
# - extra_info (str, optional): Additional context or information to pass to the LLM, such as class imbalance or noisy labels. Default is 'Not available'.
# - output_models_path (str, optional): Directory to save models in each iteration as joblib files. Default is None, which means models won't be saved.

# Returns:
# - None. This function optimizes the model iteratively and stores the history of models and performance metrics in the specified history file. The final model and improvements are made based on the LLM's suggestions.

# Raises:
# - ValueError: If the LLM model specified cannot be initialized.
# - FileNotFoundError: If the specified history file cannot be found or created.
# - RuntimeError: If any issue arises during the model training or optimization process.
# """

#     # Initialize the LLM generator
#     generator = Gpt4AnswerGenerator(api_key, model=model)

#     # Determine if the task is regression or classification
#     is_regression_bool = is_regression(load(data)['y_train'])

#     # Check if it’s a regression task and assign the appropriate LLM improver
#     if is_regression_bool:
#         llm_improver = LLMRegressionImprover(generator)
#     else:
#         llm_improver = LLMImprover(generator)
    
#     # Initialize and run the main controller with the extra_info and output_models_path passed in
#     controller = MainController(
#         joblib_file_path=data, 
#         llm_improver=llm_improver, 
#         history_file_path=history_file_path, 
#         is_regression_bool=is_regression_bool, 
#         extra_info=extra_info,
#         output_models_path=output_models_path  # Pass the output_models_path argument here
#     )
#     controller.run(iterations=iterations)




if __name__ == "__main__":
    select_model_cli()

 