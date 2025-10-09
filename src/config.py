"""
Configuration dataclasses for the optimizer.
"""
from dataclasses import dataclass, field
from typing import Optional
from constants import (
    MetricsSource, TaskType,
    DEFAULT_ITERATIONS, DEFAULT_HISTORY_FILE, DEFAULT_MODEL,
    DEFAULT_EXTRA_INFO, DEFAULT_METRICS_SOURCE, DEFAULT_MAX_RETRIES
)


@dataclass
class OptimizerConfig:
    """Configuration for the model optimization process."""

    # Required parameters
    data_path: str

    # LLM configuration
    model: str = DEFAULT_MODEL
    model_provider: Optional[str] = None
    error_model: Optional[str] = None

    # Optimization parameters
    iterations: int = DEFAULT_ITERATIONS
    max_retries: int = DEFAULT_MAX_RETRIES

    # Task configuration
    is_regression: Optional[bool] = None
    metrics_source: MetricsSource = DEFAULT_METRICS_SOURCE
    extra_info: str = DEFAULT_EXTRA_INFO

    # File paths
    history_file_path: str = DEFAULT_HISTORY_FILE
    output_models_path: Optional[str] = None
    initial_model_path: Optional[str] = None
    error_prompt_path: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")

        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        # Convert string metrics_source to enum if needed
        if isinstance(self.metrics_source, str):
            self.metrics_source = MetricsSource(self.metrics_source)


@dataclass
class TrainingData:
    """Container for training and test data splits."""
    X_train: any
    y_train: any
    X_test: any
    y_test: any
    is_pre_split: bool = True


@dataclass
class ValidationSplit:
    """Container for a validation data split."""
    X_train: any
    y_train: any
    X_val: any
    y_val: any
