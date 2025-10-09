"""
Unit tests for constants module.
"""
import pytest
from src.constants import (
    MetricsSource, ModelProvider, TaskType,
    DEFAULT_ITERATIONS, DEFAULT_MODEL, DEFAULT_TEST_SIZE,
    MODEL_TO_PROVIDER
)


class TestEnums:
    """Test enum definitions."""

    def test_metrics_source_enum(self):
        """Test MetricsSource enum values."""
        assert MetricsSource.VALIDATION.value == "validation"
        assert MetricsSource.TEST.value == "test"

    def test_model_provider_enum(self):
        """Test ModelProvider enum values."""
        assert ModelProvider.OPENAI.value == "openai"
        assert ModelProvider.META.value == "meta"
        assert ModelProvider.GOOGLE.value == "google"
        assert ModelProvider.ANTHROPIC.value == "anthropic"

    def test_task_type_enum(self):
        """Test TaskType enum values."""
        assert TaskType.CLASSIFICATION.value == "classification"
        assert TaskType.REGRESSION.value == "regression"


class TestConstants:
    """Test constant values."""

    def test_default_iterations(self):
        """Test default iterations is positive."""
        assert DEFAULT_ITERATIONS > 0
        assert isinstance(DEFAULT_ITERATIONS, int)

    def test_default_model(self):
        """Test default model is specified."""
        assert DEFAULT_MODEL is not None
        assert isinstance(DEFAULT_MODEL, str)
        assert len(DEFAULT_MODEL) > 0

    def test_default_test_size(self):
        """Test default test size is valid."""
        assert 0 < DEFAULT_TEST_SIZE < 1


class TestModelProviderMapping:
    """Test model to provider mapping."""

    def test_model_to_provider_mapping_exists(self):
        """Test that MODEL_TO_PROVIDER dictionary exists."""
        assert MODEL_TO_PROVIDER is not None
        assert isinstance(MODEL_TO_PROVIDER, dict)
        assert len(MODEL_TO_PROVIDER) > 0

    def test_gpt_models_map_to_openai(self):
        """Test GPT models map to OpenAI provider."""
        assert MODEL_TO_PROVIDER.get('gpt') == ModelProvider.OPENAI

    def test_claude_models_map_to_anthropic(self):
        """Test Claude models map to Anthropic provider."""
        assert MODEL_TO_PROVIDER.get('claude') == ModelProvider.ANTHROPIC

    def test_gemini_models_map_to_google(self):
        """Test Gemini models map to Google provider."""
        assert MODEL_TO_PROVIDER.get('gemini') == ModelProvider.GOOGLE
