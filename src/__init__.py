"""
HS Model Optimizer - LLM-powered ML model optimization framework.
"""
from .config import OptimizerConfig, TrainingData, ValidationSplit
from .constants import MetricsSource, ModelProvider, TaskType

# Main entry points
from .core import MainController

__version__ = '1.0.0'

__all__ = [
    'OptimizerConfig',
    'TrainingData',
    'ValidationSplit',
    'MetricsSource',
    'ModelProvider',
    'TaskType',
    'MainController',
]
