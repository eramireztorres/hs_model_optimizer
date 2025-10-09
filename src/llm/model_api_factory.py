from .providers.openai_model_api import OpenAIModelAPI
from .providers.llama_model_api import LlamaModelAPI
from .providers.gemini_model_api import GeminiModelAPI
from .providers.anthropic_model_api import AnthropicModelAPI
from ..constants import ModelProvider, MODEL_TO_PROVIDER

class ModelAPIFactory:
    """
    Factory class to instantiate model API clients based on the provider name or model string.
    """

    # Class-level registry mapping provider names to model API classes
    _model_registry = {}

    @classmethod
    def register_model(cls, provider: ModelProvider, model_class):
        """
        Register a new model API class with the factory.

        Args:
            provider (ModelProvider): The provider enum value.
            model_class (class): The model API class to register.
        """
        cls._model_registry[provider.value] = model_class

    @classmethod
    def get_provider_from_model(cls, model_name: str) -> str:
        """
        Deduce the provider from the model string.
        Returns 'meta' if no known substring is found.

        Args:
            model_name (str): The model name/identifier.

        Returns:
            str: The provider name (enum value).
        """
        for key, provider in MODEL_TO_PROVIDER.items():
            if key in model_name:
                return provider.value
        # Fallback to 'meta' for any unknown substring
        return ModelProvider.META.value

    @classmethod
    def get_model_api(cls, provider: str = None, model: str = 'meta-llama/llama-3.1-405b-instruct:free', **kwargs):
        """
        Get an instance of the model API client based on the provider name or model string.

        Args:
            provider (str): The name of the provider (e.g., 'openai', 'meta').
            model (str): The specific model string (e.g., 'gpt-4').
            **kwargs: Additional keyword arguments to pass to the model class constructor.

        Returns:
            An instance of the corresponding model API client.

        Raises:
            ValueError: If neither provider nor model is recognized.
        """
        if not provider and model:
            provider = cls.get_provider_from_model(model)

        provider_key = provider.lower() if provider else None
        if provider_key and provider_key in cls._model_registry:
            model_class = cls._model_registry[provider_key]
            # Pass the model string explicitly to the constructor via kwargs
            return model_class(model=model, **kwargs)
        raise ValueError(f"Unknown provider or model: {provider or model}")


# Register the models with the factory
ModelAPIFactory.register_model(ModelProvider.OPENAI, OpenAIModelAPI)
ModelAPIFactory.register_model(ModelProvider.META, LlamaModelAPI)
ModelAPIFactory.register_model(ModelProvider.GOOGLE, GeminiModelAPI)
ModelAPIFactory.register_model(ModelProvider.ANTHROPIC, AnthropicModelAPI)
ModelAPIFactory.register_model(ModelProvider.DEEPSEEK, LlamaModelAPI)
ModelAPIFactory.register_model(ModelProvider.OPENROUTER, LlamaModelAPI)
