import sys
import os
sys.path.append(os.path.dirname(__file__))

from cli_decorator import cli_decorator
from main_controller import MainController
from model_api_factory import ModelAPIFactory

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

    controller = MainController(
        joblib_file_path=data,
        model_provider=model_provider,
        model=model,
        history_file_path=history_file_path,
        # is_regression_bool=is_regression_bool,
        extra_info=extra_info,
        output_models_path=output_models_path
    )
    controller.run(iterations=iterations)



if __name__ == "__main__":
    select_model_cli()

 