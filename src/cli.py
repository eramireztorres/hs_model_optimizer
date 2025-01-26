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
                     metrics_source: str = 'validation'):
    """
    Args:
      
        data: Path to a `.joblib` file or a directory containing `.csv` files. 
        The input can include pre-split data ('X_train', 'y_train', 'X_test', 'y_test') or
        unsplit data ('X', 'y'), in which case a validation split will be created.


        model (str, optional): The LLM model name (e.g., 'gpt-4', 'llama-3.1').
        model_provider (str, optional): The LLM provider name ('openai', 'llama', 'google').
        ...
        metrics_source (str, optional): Source of the metrics to show to the LLM ('validation' or 'test').
            Default is 'validation'.
    """
    if metrics_source not in ['validation', 'test']:
        raise ValueError("metrics_source must be 'validation' or 'test'")
    
    if not model_provider:
        model_provider = ModelAPIFactory.get_provider_from_model(model)

    print(f"Using model: {model} (provider: {model_provider})")
    print(f"Metrics source: {metrics_source}")

    controller = MainController(
        joblib_file_path=data,
        model_provider=model_provider,
        model=model,
        history_file_path=history_file_path,
        extra_info=extra_info,
        output_models_path=output_models_path,
        metrics_source=metrics_source
    )
    controller.run(iterations=iterations)


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

 