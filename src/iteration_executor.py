"""
Command pattern implementation for executing optimization iterations.
"""
import logging
from typing import Optional, Tuple
from config import ValidationSplit
from llm_code_cleaner import LLMCodeCleaner
from error_handler import ErrorHandler
from model_trainer import ModelTrainer, RegressionModelTrainer
from metrics_calculator import MetricsCalculatorFactory


class IterationResult:
    """Result of an iteration execution."""

    def __init__(self, success: bool, model=None, metrics: dict = None,
                 code: str = None, error: str = None):
        self.success = success
        self.model = model
        self.metrics = metrics or {}
        self.code = code
        self.error = error


class IterationExecutor:
    """Executes a single optimization iteration with retry logic."""

    def __init__(self, dynamic_updater, llm_improver, error_corrector,
                 code_cleaner: LLMCodeCleaner, error_handler: ErrorHandler,
                 is_regression: bool, max_retries: int = 1):
        """
        Initialize the iteration executor.

        Args:
            dynamic_updater: Updater for dynamic model code.
            llm_improver: LLM improver for generating suggestions.
            error_corrector: Error corrector for fixing code errors.
            code_cleaner: Cleaner for LLM-generated code.
            error_handler: Handler for errors.
            is_regression: Whether task is regression.
            max_retries: Maximum retry attempts.
        """
        self.dynamic_updater = dynamic_updater
        self.llm_improver = llm_improver
        self.error_corrector = error_corrector
        self.code_cleaner = code_cleaner
        self.error_handler = error_handler
        self.is_regression = is_regression
        self.max_retries = max_retries
        self.metrics_calculator = MetricsCalculatorFactory.create(is_regression)

    def execute(self, iteration: int, data_split: ValidationSplit,
                last_valid_code: str, extra_info: str) -> IterationResult:
        """
        Execute a single optimization iteration.

        Args:
            iteration (int): Current iteration number.
            data_split (ValidationSplit): Prepared data split.
            last_valid_code (str): Last valid model code.
            extra_info (str): Extra information for LLM.

        Returns:
            IterationResult: Result of the iteration.
        """
        print(f"\n=== Iteration {iteration + 1} ===")

        # Try to load model with retries
        model, error_msg = self._load_model_with_retries(last_valid_code, extra_info)

        if model is None:
            return self._handle_max_retries_exceeded(iteration, error_msg)

        print(f"Model for iteration {iteration + 1}: {model.__class__.__name__}")

        # Train and evaluate model
        try:
            metrics = self._train_and_evaluate(model, data_split, iteration)
        except Exception as e:
            return self._handle_training_error(e, iteration)

        print(f"Metrics for iteration {iteration + 1}: {metrics}")

        # Get current model code
        current_code = self._get_current_code()

        return IterationResult(
            success=True,
            model=model,
            metrics=metrics,
            code=current_code
        )

    def _load_model_with_retries(self, last_valid_code: str,
                                  extra_info: str) -> Tuple[Optional[any], Optional[str]]:
        """
        Attempt to load model with retries and error correction.

        Args:
            last_valid_code (str): Last known valid code.
            extra_info (str): Extra context for LLM.

        Returns:
            Tuple[Optional[model], Optional[str]]: Model and error message.
        """
        retries = 0

        while retries < self.max_retries:
            model, error_msg = self.dynamic_updater.run_dynamic_model()

            if model is not None:
                return model, None

            # Attempt error correction
            improved_code = self._get_error_correction(last_valid_code, error_msg, extra_info)

            if improved_code:
                improved_code = self.code_cleaner.clean_code(improved_code)
                self.dynamic_updater.update_model_code(improved_code)
                model, error_msg = self.dynamic_updater.run_dynamic_model()

                if model is not None:
                    return model, None
            else:
                logging.warning("No new suggestions received from LLM. Skipping retry.")
                print("No new suggestions received from LLM. Skipping retry.")

            retries += 1

        return None, error_msg

    def _get_error_correction(self, last_valid_code: str, error_msg: str,
                              extra_info: str) -> Optional[str]:
        """
        Get error correction from LLM.

        Args:
            last_valid_code (str): Last valid code.
            error_msg (str): Error message.
            extra_info (str): Extra context.

        Returns:
            Optional[str]: Corrected code or None.
        """
        if self.error_corrector:
            current_code = self._get_current_code()
            improved_code = self.error_corrector.get_error_fix(current_code, error_msg)
            print("\n=== CODE after ERROR correction ===")
            print(improved_code)
            return improved_code
        else:
            # Fallback to normal improvement
            return self.llm_improver.get_model_suggestions(
                last_valid_code, {}, extra_info
            )

    def _train_and_evaluate(self, model, data_split: ValidationSplit,
                            iteration: int) -> dict:
        """
        Train and evaluate a model.

        Args:
            model: Model to train.
            data_split (ValidationSplit): Data split.
            iteration (int): Current iteration.

        Returns:
            dict: Evaluation metrics.
        """
        trainer_class = RegressionModelTrainer if self.is_regression else ModelTrainer
        trainer = trainer_class(
            model=model,
            X_train=data_split.X_train,
            y_train=data_split.y_train,
            X_test=data_split.X_val,
            y_test=data_split.y_val,
            metrics_calculator=self.metrics_calculator
        )

        trainer.train_model()
        return trainer.evaluate_model()

    def _handle_max_retries_exceeded(self, iteration: int,
                                      error_msg: str) -> IterationResult:
        """
        Handle case where max retries are exceeded.

        Args:
            iteration (int): Current iteration.
            error_msg (str): Error message.

        Returns:
            IterationResult: Failed iteration result.
        """
        logging.error(
            f"Exceeded maximum retries ({self.max_retries}) for iteration {iteration + 1}. Skipping iteration."
        )
        print(
            f"Exceeded maximum retries ({self.max_retries}) for iteration {iteration + 1}. Skipping iteration."
        )

        current_code = self._get_current_code()
        simplified_error = self.error_handler.simplify_error(Exception(error_msg))
        metrics = {"error": simplified_error}

        return IterationResult(
            success=False,
            metrics=metrics,
            code=current_code,
            error=simplified_error
        )

    def _handle_training_error(self, error: Exception, iteration: int) -> IterationResult:
        """
        Handle training or evaluation errors.

        Args:
            error (Exception): The error that occurred.
            iteration (int): Current iteration.

        Returns:
            IterationResult: Failed iteration result.
        """
        simplified_error = self.error_handler.simplify_error(error)
        logging.error(f"Training or evaluation failed: {simplified_error}")
        print(f"Training or evaluation failed: {simplified_error}")

        metrics = {"error": simplified_error}
        current_code = self._get_current_code()

        return IterationResult(
            success=False,
            metrics=metrics,
            code=current_code,
            error=simplified_error
        )

    def _get_current_code(self) -> str:
        """Get current dynamic model code."""
        try:
            with open(self.dynamic_updater.dynamic_file_path, 'r') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Failed to read dynamic model code: {e}")
            return ""
