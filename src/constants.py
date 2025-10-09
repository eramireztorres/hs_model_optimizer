"""
Constants and enumerations for the hs_model_optimizer package.
"""
from enum import Enum


class MetricsSource(str, Enum):
    """Defines where evaluation metrics should be computed."""
    VALIDATION = "validation"
    TEST = "test"


class ModelProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    META = "meta"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    OPENROUTER = "openrouter"


class TaskType(str, Enum):
    """Machine learning task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


# Model provider mappings
MODEL_TO_PROVIDER = {
    'gpt': ModelProvider.OPENAI,
    'gpt-4o': ModelProvider.OPENAI,
    'gpt-4o-mini': ModelProvider.OPENAI,
    'o1-': ModelProvider.OPENAI,
    'o3-': ModelProvider.OPENAI,
    'llama': ModelProvider.META,
    'gemini': ModelProvider.GOOGLE,
    'claude': ModelProvider.ANTHROPIC,
    'deepseek': ModelProvider.DEEPSEEK,
    'cognitivecomputations/': ModelProvider.OPENROUTER,
    'google/': ModelProvider.OPENROUTER,
    'mistralai/': ModelProvider.OPENROUTER,
    'qwen/': ModelProvider.OPENROUTER,
    'meta-llama/': ModelProvider.OPENROUTER,
    'deepseek/': ModelProvider.OPENROUTER,
    'nvidia/': ModelProvider.OPENROUTER,
    'microsoft/': ModelProvider.OPENROUTER,
}

# Default values
DEFAULT_ITERATIONS = 10
DEFAULT_HISTORY_FILE = 'model_history.joblib'
DEFAULT_MODEL = 'gpt-4.1-mini'
DEFAULT_EXTRA_INFO = 'Not available'
DEFAULT_METRICS_SOURCE = MetricsSource.VALIDATION
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_RETRIES = 1
