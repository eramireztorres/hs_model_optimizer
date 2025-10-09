"""
Unified error handling strategies for the optimizer.
"""
import logging
import traceback
from typing import Optional


class ErrorHandler:
    """Centralized error handling and formatting."""

    @staticmethod
    def simplify_error(error: Exception) -> str:
        """
        Simplify an exception to a single-line string.

        Args:
            error (Exception): The exception to simplify.

        Returns:
            str: Simplified error message (last line).
        """
        msg = str(error)
        return msg.splitlines()[-1] if msg else ''

    @staticmethod
    def format_error_for_llm(error: Exception, code: Optional[str] = None) -> str:
        """
        Format error message for LLM consumption.

        Args:
            error (Exception): The exception that occurred.
            code (Optional[str]): The code that caused the error.

        Returns:
            str: Formatted error message.
        """
        error_type = type(error).__name__
        error_msg = str(error)

        formatted = f"{error_type}: {error_msg}"

        # Add code context if available
        if code:
            formatted += f"\n\nProblematic code:\n{code}"

        return formatted

    @staticmethod
    def log_error(error: Exception, context: str = "", include_traceback: bool = False):
        """
        Log an error with optional context and traceback.

        Args:
            error (Exception): The exception to log.
            context (str): Additional context about where the error occurred.
            include_traceback (bool): Whether to include full traceback.
        """
        error_msg = f"{context}: {str(error)}" if context else str(error)
        logging.error(error_msg)

        if include_traceback:
            logging.error(traceback.format_exc())

    @staticmethod
    def handle_model_loading_error(error: Exception, iteration: int) -> dict:
        """
        Handle errors that occur during model loading.

        Args:
            error (Exception): The loading error.
            iteration (int): Current iteration number.

        Returns:
            dict: Metrics dict with error information.
        """
        simplified = ErrorHandler.simplify_error(error)
        logging.error(f"Model loading failed at iteration {iteration}: {simplified}")
        return {"error": simplified, "iteration": iteration, "stage": "model_loading"}

    @staticmethod
    def handle_training_error(error: Exception, iteration: int) -> dict:
        """
        Handle errors that occur during model training.

        Args:
            error (Exception): The training error.
            iteration (int): Current iteration number.

        Returns:
            dict: Metrics dict with error information.
        """
        simplified = ErrorHandler.simplify_error(error)
        logging.error(f"Training failed at iteration {iteration}: {simplified}")
        return {"error": simplified, "iteration": iteration, "stage": "training"}

    @staticmethod
    def handle_evaluation_error(error: Exception, iteration: int) -> dict:
        """
        Handle errors that occur during model evaluation.

        Args:
            error (Exception): The evaluation error.
            iteration (int): Current iteration number.

        Returns:
            dict: Metrics dict with error information.
        """
        simplified = ErrorHandler.simplify_error(error)
        logging.error(f"Evaluation failed at iteration {iteration}: {simplified}")
        return {"error": simplified, "iteration": iteration, "stage": "evaluation"}
