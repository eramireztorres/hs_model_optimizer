from typing import Literal

import sys
import os
import atexit

# Suppress noisy multiprocessing ResourceTracker errors at interpreter shutdown.
# Some sklearn/joblib backends may spawn processes that trigger
# ChildProcessError in ResourceTracker.__del__ when exiting.
try:
    from multiprocessing import resource_tracker as _rt
    from multiprocessing.resource_tracker import ResourceTracker as _RTClass

    def _safe_stop_resource_tracker():
        try:
            # Best-effort stop; ignore if it's already gone.
            _rt._resource_tracker._stop()
        except Exception:
            pass

    # Ensure this runs before Python tears down modules
    atexit.register(_safe_stop_resource_tracker)

    # Also harden the ResourceTracker.__del__ against spurious ChildProcessError
    _orig_del = getattr(_RTClass, "__del__", None)
    if _orig_del is not None:
        def _safe_del(self):
            try:
                _orig_del(self)
            except Exception:
                # Ignore shutdown-time tracker errors
                pass
        _RTClass.__del__ = _safe_del
except Exception:
    pass
sys.path.append(os.path.dirname(__file__))

from .utils.cli_decorator import cli_decorator
from .core.main_controller_refactored import MainController
from .llm.model_api_factory import ModelAPIFactory
from .config import OptimizerConfig
from .constants import MetricsSource

# %%


@cli_decorator
def select_model_cli(data,
                     model: str = 'gpt-4.1-mini',
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
    import multiprocessing
    if __name__ == "__main__":
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass  # start method can only be set once

    """
    Command-line interface for running the model optimization process.

    This function initializes and runs the optimization process based on the
    provided command-line arguments. It supports various data input formats and
    allows for detailed configuration of the optimization process.

    Args:
        data (str): Path to the input dataset. It can be one of the following:
            - A `.joblib` file with pre-split data ('X_train', 'y_train', etc.).
            - A folder with `.csv` files ('X.csv', 'y.csv').
            - A single `.csv` file with the target column as the last one.
        model (str, optional): The LLM model to use for generating suggestions.
            Defaults to 'gpt-4.1-mini'.
        model_provider (str, optional): The provider of the LLM model. If not
            provided, it is inferred from the model name. Defaults to None.
        history_file_path (str, optional): Path to the .joblib file for saving
            model history. Defaults to 'model_history.joblib'.
        iterations (int, optional): The number of optimization iterations.
            Defaults to 10.
        extra_info (str, optional): Additional context to provide to the LLM.
            Defaults to 'Not available'.
        output_models_path (str, optional): Directory to save trained models.
            If None, models are not saved. Defaults to None.
        is_regression (Literal[None, "true", "false"], optional): Specifies if
            the task is regression. Defaults to None.
        metrics_source (str, optional): The source for evaluation metrics,
            either 'validation' or 'test'. Defaults to 'validation'.
        error_model (str, optional): The model to use for error correction.
            Defaults to None.
        initial_model_path (str, optional): Path to an initial model to seed
            the optimization. Defaults to None.

    Raises:
        ValueError: If `metrics_source` is not 'validation' or 'test'.

    Example:
        To run the optimization with a specific dataset and model:
        ```bash
        hs_optimize --data path/to/your/data.csv --model gpt-4o-mini --iterations 5
        ```
    """

    # Build configuration object
    if metrics_source not in ['validation', 'test']:
        raise ValueError("metrics_source must be 'validation' or 'test'")

    if not model_provider:
        model_provider = ModelAPIFactory.get_provider_from_model(model)

    if is_regression is not None:
        is_regression = is_regression == 'true'

    error_prompt_path = os.path.join(os.path.dirname(__file__), 'prompts/error_correction_prompt.txt')

    config = OptimizerConfig(
        data_path=data,
        model=model,
        model_provider=model_provider,
        error_model=error_model,
        iterations=iterations,
        is_regression=is_regression,
        metrics_source=MetricsSource(metrics_source),
        extra_info=extra_info,
        history_file_path=history_file_path,
        output_models_path=output_models_path,
        initial_model_path=initial_model_path,
        error_prompt_path=error_prompt_path,
    )

    print(f"Using model: {config.model} (provider: {config.model_provider})")
    print(f"Metrics source: {config.metrics_source.value}")

    controller = MainController(config)
    controller.run()


if __name__ == "__main__":
    select_model_cli()
