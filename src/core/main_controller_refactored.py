"""
Refactored MainController with dependency injection and SOLID principles.
"""
import joblib
import logging
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..config import OptimizerConfig
from ..data.data_loader import DataLoader
from ..data.data_preparation_strategy import DataPreparationStrategyFactory
from ..data.data_splitter import DataSplitter
from ..models.dynamic_model_updater import DynamicModelUpdater, DynamicRegressionModelUpdater
from ..llm.llm_improver import LLMImprover, LLMRegressionImprover
from ..llm.model_api_factory import ModelAPIFactory
from ..llm.llm_code_cleaner import LLMCodeCleaner
from ..llm.error_corrector import ErrorCorrector
from ..utils.error_handler import ErrorHandler
from ..persistence.model_code_repository import ModelCodeRepository
from ..persistence.model_history_manager import ModelHistoryManager
from .iteration_executor import IterationExecutor


def is_regression(y_train):
    """
    Check if the target values suggest a regression problem.

    Args:
        y_train (array-like): The target values from the training set.

    Returns:
        bool: True if the problem is regression, False if it's classification.
    """
    import numpy as np

    if np.issubdtype(y_train.dtype, np.floating):
        if np.all(np.equal(np.mod(y_train, 1), 0)):
            return False  # Classification with integer-like floats

    return np.issubdtype(y_train.dtype, np.floating) or np.issubdtype(y_train.dtype, np.integer) and not np.all(np.equal(np.mod(y_train, 1), 0))


class MainController:
    """
    Refactored main controller using dependency injection and SOLID principles.
    """

    def __init__(self, config: OptimizerConfig,
                 history_manager: ModelHistoryManager = None,
                 code_cleaner: LLMCodeCleaner = None,
                 error_handler: ErrorHandler = None,
                 data_splitter: DataSplitter = None):
        """
        Initialize the MainController with dependency injection.

        Args:
            config (OptimizerConfig): Configuration object.
            history_manager (ModelHistoryManager): Optional history manager.
            code_cleaner (LLMCodeCleaner): Optional code cleaner.
            error_handler (ErrorHandler): Optional error handler.
            data_splitter (DataSplitter): Optional data splitter.
        """
        self.config = config

        # Inject dependencies or create defaults
        self.history_manager = history_manager or ModelHistoryManager(config.history_file_path)
        self.code_cleaner = code_cleaner or LLMCodeCleaner()
        self.error_handler = error_handler or ErrorHandler()
        self.data_splitter = data_splitter or DataSplitter()

        # Load initial model code if provided
        if config.initial_model_path:
            self._load_initial_model()

        # Load data
        self.data = self._load_data()

        # Determine task type
        if self.config.is_regression is None:
            if self.is_pre_split:
                self.config.is_regression = is_regression(self.data['y_train'])
            else:
                self.config.is_regression = is_regression(self.data['y'])

        # Initialize LLM components
        self.llm_improver = self._initialize_llm_improver()
        self.error_corrector = self._initialize_error_corrector()

        # Initialize dynamic updater
        self.dynamic_updater = self._initialize_dynamic_updater()

        # Initialize data preparation strategy
        self.data_prep_strategy = DataPreparationStrategyFactory.create(
            self.config.metrics_source,
            self.data_splitter
        )

        # Initialize iteration executor
        self.iteration_executor = IterationExecutor(
            dynamic_updater=self.dynamic_updater,
            llm_improver=self.llm_improver,
            error_corrector=self.error_corrector,
            code_cleaner=self.code_cleaner,
            error_handler=self.error_handler,
            is_regression=self.config.is_regression,
            max_retries=self.config.max_retries
        )

        # Initialize model code repository
        self.code_repository = ModelCodeRepository(self.dynamic_updater.dynamic_file_path)

    def _load_initial_model(self):
        """Load initial model code if specified."""
        try:
            with open(self.config.initial_model_path, 'r') as f:
                init_code = f.read()
            updater = (DynamicRegressionModelUpdater() if self.config.is_regression
                       else DynamicModelUpdater())
            updater.update_model_code(init_code)
        except Exception as e:
            print(f"Warning: could not load initial model from {self.config.initial_model_path}: {e}")

    def _load_data(self):
        """Load and prepare data."""
        try:
            data = DataLoader.load_data(self.config.data_path)
            logging.info(f"Data loaded successfully from {self.config.data_path}")

            # Store pre-split flag
            self.is_pre_split = data.pop('is_pre_split', True)

            return data
        except Exception as e:
            logging.error(f"Failed to load data from {self.config.data_path}: {e}")
            return None

    def _initialize_llm_improver(self):
        """Initialize the LLM improver."""
        llm_model = ModelAPIFactory.get_model_api(
            provider=self.config.model_provider,
            model=self.config.model
        )
        if self.config.is_regression:
            return LLMRegressionImprover(llm_model)
        return LLMImprover(llm_model)

    def _initialize_error_corrector(self):
        """Initialize the error corrector."""
        if self.config.error_model:
            error_llm = ModelAPIFactory.get_model_api(
                provider=ModelAPIFactory.get_provider_from_model(self.config.error_model),
                model=self.config.error_model
            )
        else:
            error_llm = ModelAPIFactory.get_model_api(
                provider=self.config.model_provider,
                model=self.config.model
            )
        return ErrorCorrector(error_llm, self.config.error_prompt_path)

    def _initialize_dynamic_updater(self):
        """Initialize the dynamic model updater."""
        if self.config.is_regression:
            return DynamicRegressionModelUpdater()
        else:
            return DynamicModelUpdater()

    def run(self):
        """
        Run the training and improvement process for the configured number of iterations.
        """
        original_model_code = self.code_repository.backup()
        if not original_model_code:
            logging.error("Failed to backup the original model. Exiting.")
            return

        last_valid_model_code = original_model_code

        try:
            print(f'ITERATIONS: {self.config.iterations}')

            for iteration in range(self.config.iterations):
                # Prepare data for this iteration
                data_split = self.data_prep_strategy.prepare(self.data, self.is_pre_split)

                # Execute iteration
                result = self.iteration_executor.execute(
                    iteration=iteration,
                    data_split=data_split,
                    last_valid_code=last_valid_model_code,
                    extra_info=self.config.extra_info
                )

                # Save history
                self.history_manager.save_model_history(result.code, result.metrics)
                self.llm_improver.log_model_history(result.code, result.metrics)

                # Save model if configured
                if result.success and self.config.output_models_path:
                    self._save_model(iteration, result.model)

                # Update last valid code if successful
                if result.success:
                    last_valid_model_code = result.code

                # Get improvements for next iteration
                if result.success:
                    improved_code = self.llm_improver.get_model_suggestions(
                        result.code, result.metrics, extra_info=self.config.extra_info
                    )

                    if improved_code:
                        improved_code = self.code_cleaner.clean_code(improved_code)
                        print(f"\n=== IMPROVED MODEL ITERATION {iteration + 1} ===")
                        print(improved_code)
                        self.dynamic_updater.update_model_code(improved_code)
                    else:
                        logging.warning("No improvements suggested by the LLM in this iteration.")
                        print(f"No improvements suggested by the LLM in iteration {iteration}.")
                else:
                    # Try to get improvement even after failure
                    improved_code = self.llm_improver.get_model_suggestions(
                        result.code, result.metrics, extra_info=self.config.extra_info
                    )
                    if improved_code:
                        improved_code = self.code_cleaner.clean_code(improved_code)
                        self.dynamic_updater.update_model_code(improved_code)

                print(f'FINISHED ITERATION {iteration + 1}')

        except ChildProcessError:
            pass  # Ignore multiprocessing errors
        finally:
            if original_model_code:
                self.dynamic_updater.update_model_code(original_model_code)
                print("Original model restored after iterations.")
                logging.info("Original model restored after iterations.")

    def _save_model(self, iteration: int, model):
        """
        Save the trained model to a joblib file after each iteration.

        Args:
            iteration (int): The current iteration number.
            model: The trained model to save.
        """
        if not self.config.output_models_path:
            return

        # Ensure the output directory exists
        if not os.path.exists(self.config.output_models_path):
            os.makedirs(self.config.output_models_path)

        # Save the model
        model_path = os.path.join(self.config.output_models_path, f"model_{iteration + 1}.joblib")

        try:
            joblib.dump(model, model_path)
            logging.info(f"Model for iteration {iteration + 1} saved to {model_path}")
        except Exception as e:
            logging.error(f"Failed to save model for iteration {iteration + 1}: {e}")
