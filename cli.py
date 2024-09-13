from src.cli_decorator import cli_decorator
from src.main_controller import MainController
from src.gpt import Gpt4AnswerGenerator
from src.llm_improver import LLMImprover

import os

api_key = os.getenv('OPENAI_API_KEY')

@cli_decorator
def select_model_cli(data,
                     
        history_file_path: str = 'model_history.joblib',
        model: str = 'gpt-4o',
        iterations: int = 5
        
        ):
    
    """
- Selects and optimizes a machine learning model for classification using an LLM improver, iterating through models and hyperparameters based on the specified configuration.

Args:
- data (dict): A dictionary containing training and test data, with keys such as 'X_train', 'y_train', 'X_test', 'y_test'. These should be NumPy arrays representing the feature and target datasets for model training and evaluation.
- history_file_path (str, optional): Path to the joblib file where the model history will be stored. The history includes models, their hyperparameters, and performance metrics for each iteration. Default is 'model_history.joblib'.
- model (str, optional): The name of the LLM model to use for generating suggestions and improvements for models and hyperparameters. Defaults to 'gpt-4o'.
- iterations (int, optional): The number of iterations to run, where each iteration involves training a model, evaluating its performance, and generating improvements. Default is 5.

Returns:
- None. This function optimizes the model iteratively and stores the history of models and performance metrics in the specified history file. The final model and improvements are made based on the LLM's suggestions.

Raises:
- ValueError: If the LLM model specified cannot be initialized.
- FileNotFoundError: If the specified history file cannot be found or created.
- RuntimeError: If any issue arises during the model training or optimization process.
"""

    
    generator = Gpt4AnswerGenerator(api_key, model=model)
    llm_improver = LLMImprover(generator, model_history=None)
    controller = MainController(data, llm_improver, history_file_path)
    controller.run(iterations=iterations)


if __name__ == "__main__":
    select_model_cli()

 