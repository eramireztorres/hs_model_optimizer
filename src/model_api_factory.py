from openai_model_api import OpenAIModelAPI
from llama_model_api import LlamaModelAPI
# from gemini_model_api import GeminiModelAPI

class ModelAPIFactory:
    """
    Factory class to instantiate model API clients based on the provider name or model string.
    """

    # Class-level registry mapping provider names to model API classes
    _model_registry = {}

    # Map models to providers
    _model_to_provider = {
        'gpt': 'openai',
        'gpt-4o': 'openai',
        'gpt-4o-mini': 'openai',
        'llama': 'llama',
        'gemini': 'gemini',  # Uncomment when Gemini is added
    }

    @classmethod
    def register_model(cls, provider_name, model_class):
        """
        Register a new model API class with the factory.

        Args:
            provider_name (str): The name of the provider (e.g., 'openai', 'llama').
            model_class (class): The model API class to register.
        """
        cls._model_registry[provider_name.lower()] = model_class

    @classmethod
    def get_provider_from_model(cls, model_name):
        """
        Deduce the provider from the model string.

        Args:
            model_name (str): The name of the model (e.g., 'gpt-4', 'llama-3.1').

        Returns:
            str: The provider name corresponding to the model.

        Raises:
            ValueError: If the model name is not recognized.
        """
        for key, provider in cls._model_to_provider.items():
            if key in model_name:
                return provider
        raise ValueError(f"Unknown model: {model_name}")

    @classmethod
    def get_model_api(cls, provider='llama', model='meta-llama/llama-3.1-405b-instruct:free', **kwargs):
        """
        Get an instance of the model API client based on the provider name or model string.

        Args:
            provider (str): The name of the provider (e.g., 'openai', 'llama').
            model (str): The specific model string (e.g., 'gpt-4').
            **kwargs: Additional keyword arguments to pass to the model class constructor.

        Returns:
            An instance of the corresponding model API client.

        Raises:
            ValueError: If neither provider nor model is recognized.
        """
        if not provider and model:
            provider = cls.get_provider_from_model(model)           
        if provider and provider.lower() in cls._model_registry:           
            model_class = cls._model_registry[provider.lower()]
            # Pass the model string explicitly to the constructor via kwargs
            return model_class(model=model, **kwargs)
        raise ValueError(f"Unknown provider or model: {provider or model}")


# Register the models with the factory
ModelAPIFactory.register_model('openai', OpenAIModelAPI)
ModelAPIFactory.register_model('llama', LlamaModelAPI)
# ModelAPIFactory.register_model('gemini', GeminiModelAPI)


# from openai_model_api import OpenAIModelAPI
# from llama_model_api import LlamaModelAPI
# # from gemini_model_api import GeminiModelAPI

# class ModelAPIFactory:
#     """
#     Factory class to instantiate model API clients based on the provider name or model string.
#     """

#     # Class-level registry mapping provider names to model API classes
#     _model_registry = {}

#     # Map models to providers
#     _model_to_provider = {
#         'gpt': 'openai',
#         'gpt-4o': 'openai',
#         'gpt-4o-mini': 'openai',
#         'llama': 'llama',
#         'gemini': 'gemini',  # Uncomment when Gemini is added
#     }

#     @classmethod
#     def register_model(cls, provider_name, model_class):
#         """
#         Register a new model API class with the factory.

#         Args:
#             provider_name (str): The name of the provider (e.g., 'openai', 'llama').
#             model_class (class): The model API class to register.
#         """
#         cls._model_registry[provider_name.lower()] = model_class

#     @classmethod
#     def get_provider_from_model(cls, model_name):
#         """
#         Deduce the provider from the model string.

#         Args:
#             model_name (str): The name of the model (e.g., 'gpt-4', 'llama-3.1').

#         Returns:
#             str: The provider name corresponding to the model.

#         Raises:
#             ValueError: If the model name is not recognized.
#         """
#         for key, provider in cls._model_to_provider.items():
#             if key in model_name:
#                 return provider
#         raise ValueError(f"Unknown model: {model_name}")

#     @classmethod
#     def get_model_api(cls, provider=None, model=None, **kwargs):
#         """
#         Get an instance of the model API client based on the provider name or model string.

#         Args:
#             provider (str): The name of the provider (e.g., 'openai', 'llama').
#             model (str): The specific model string (e.g., 'gpt-4').
#             **kwargs: Additional keyword arguments to pass to the model class constructor.

#         Returns:
#             An instance of the corresponding model API client.

#         Raises:
#             ValueError: If neither provider nor model is recognized.
#         """
#         if not provider and model:
#             provider = cls.get_provider_from_model(model)
#         if provider and provider.lower() in cls._model_registry:
#             model_class = cls._model_registry[provider.lower()]
#             return model_class(**kwargs)
#         raise ValueError(f"Unknown provider or model: {provider or model}")

# # Register the models with the factory
# ModelAPIFactory.register_model('openai', OpenAIModelAPI)
# ModelAPIFactory.register_model('llama', LlamaModelAPI)
# # ModelAPIFactory.register_model('gemini', GeminiModelAPI)