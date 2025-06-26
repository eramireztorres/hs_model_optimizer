from typing import Literal

import sys
import os
sys.path.append(os.path.dirname(__file__))

from cli_decorator import cli_decorator
from main_controller import MainController
from model_api_factory import ModelAPIFactory

#%%

@cli_decorator
def select_model_cli(data,
                     model: str = 'gpt-4o-mini',
                     model_provider: str = None,
                     history_file_path: str = 'model_history.joblib',
                     iterations: int = 10,
                     extra_info: str = 'Not available',
                     output_models_path: str = None,
                     is_regression: Literal[None, "true", "false"] = None,
                     metrics_source: str = 'validation',
                     error_model: str = None,
                     initial_model_path: str = None,
                     ):
    """
    Command-line interface function for selecting and running an optimization model.
    This function loads the input data, initializes the optimization controller, 
    and runs the optimization process for the specified number of iterations.

    Args:
        - data (str): Path to the input dataset. This can be: 1. A `.joblib` file containing pre-split data dictionary with keys 'X_train', 'y_train', 'X_test', and 'y_test'. 2. A `.joblib` file or a folder with `.csv` files containing unsplit data with keys or filenames 'X' and 'y'. The program will perform a validation split if unsplit data is provided. 3. A single `.csv` file containing multiple feature columns and one target column (assumed to be the last column).

        - model (str, optional): The LLM model name to be used for generating suggestions and improvements for the models (e.g., 'gpt-4', 'llama-3.1'). Default is 'gpt-4o-mini'.

        - model_provider (str, optional): The provider of the LLM model (e.g., 'openai','llama', 'google'). If not provided, the provider will be inferred automatically based on the `model` argument. Default is None.

        - history_file_path (str, optional): Path to the `.joblib` file where the model history (including hyperparameters and metrics) will be saved. Default is 'model_history.joblib'.

        - iterations (int, optional): The number of optimization iterations to perform. Each iteration involves training a model, evaluating its performance, and generating improvements. Default is 10.

        - extra_info (str, optional): Additional context or information to provide to the LLM for generating better suggestions. Examples include class imbalance, noisy labels, or outliers in the data. Default is 'Not available'.

        - output_models_path (str, optional): Path to a directory where trained models for each iteration will be saved in `.joblib` files. If not specified, models are not saved to disk. Default is None.

        - metrics_source (str, optional): Source of the metrics to provide to the LLM for evaluation and suggestions. Must be either 'validation' (metrics from a validation split of training data) or 'test' (metrics from test data). Default is 'validation'.

    Raises:
        - ValueError: If `metrics_source` is not 'validation' or 'test'.

    Example:
        ```bash
        hs_optimize -d path/to/data -m gpt-4o-mini --metrics-source validation --iterations 5
        ```

    Usage:
        This function is intended to be used as part of the CLI for running the optimization 
        process. It initializes the `MainController` with the provided arguments and 
        delegates the optimization logic.

    """
    
    error_prompt_path = os.path.join(os.path.dirname(__file__), 'prompts/error_correction_prompt.txt')
    
    if metrics_source not in ['validation', 'test']:
        raise ValueError("metrics_source must be 'validation' or 'test'")
    
    if not model_provider:
        model_provider = ModelAPIFactory.get_provider_from_model(model)

    print(f"Using model: {model} (provider: {model_provider})")
    print(f"Metrics source: {metrics_source}")
    
    if is_regression is not None:
        is_regression = is_regression == 'true'

    controller = MainController(
        joblib_file_path=data,
        model_provider=model_provider,
        model=model,
        history_file_path=history_file_path,
        extra_info=extra_info,
        output_models_path=output_models_path,
        metrics_source=metrics_source,
        is_regression_bool=is_regression,
        error_model=error_model,
        error_prompt_path=error_prompt_path,
        initial_model_path=initial_model_path,
    )
    controller.run(iterations=iterations)

# def select_model_cli(data,
#                      model: str = 'gpt-4o-mini',
#                      model_provider: str = None,
#                      history_file_path: str = 'model_history.joblib',
#                      iterations: int = 10,
#                      extra_info: str = 'Not available',
#                      output_models_path: str = None,
#                      metrics_source: str = 'validation'):
#     """
#     Args:
      
#         data: Path to a `.joblib` file or a directory containing `.csv` files. 
#         The input can include pre-split data ('X_train', 'y_train', 'X_test', 'y_test') or
#         unsplit data ('X', 'y'), in which case a validation split will be created.


#         model (str, optional): The LLM model name (e.g., 'gpt-4', 'llama-3.1').
#         model_provider (str, optional): The LLM provider name ('openai', 'llama', 'google').
#         ...
#         metrics_source (str, optional): Source of the metrics to show to the LLM ('validation' or 'test').
#             Default is 'validation'.
#     """
#     if metrics_source not in ['validation', 'test']:
#         raise ValueError("metrics_source must be 'validation' or 'test'")
    
#     if not model_provider:
#         model_provider = ModelAPIFactory.get_provider_from_model(model)

#     print(f"Using model: {model} (provider: {model_provider})")
#     print(f"Metrics source: {metrics_source}")

#     controller = MainController(
#         joblib_file_path=data,
#         model_provider=model_provider,
#         model=model,
#         history_file_path=history_file_path,
#         extra_info=extra_info,
#         output_models_path=output_models_path,
#         metrics_source=metrics_source
#     )
#     controller.run(iterations=iterations)


# @cli_decorator
# def select_model_cli(data,
#                      model: str = 'gpt-4o-mini',
#                      model_provider: str = None,
#                      history_file_path: str = 'model_history.joblib',
#                      iterations: int = 10,
#                      extra_info: str = 'Not available',
#                      output_models_path: str = None):
#     """
#     Args:
#         model (str, optional): The LLM model name (e.g., 'gpt-4', 'llama-3.1').
#         model_provider (str, optional): The LLM provider name ('openai', 'llama', 'gemini').
#         ...
#     """
#     if not model_provider:
#         model_provider = ModelAPIFactory.get_provider_from_model(model)

#     print(f"Using model: {model} (provider: {model_provider})")

#     controller = MainController(
#         joblib_file_path=data,
#         model_provider=model_provider,
#         model=model,
#         history_file_path=history_file_path,
#         # is_regression_bool=is_regression_bool,
#         extra_info=extra_info,
#         output_models_path=output_models_path
#     )
#     controller.run(iterations=iterations)



if __name__ == "__main__":
    select_model_cli()

 