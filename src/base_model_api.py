import abc

class BaseModelAPI(abc.ABC):
    """
    Abstract base class for model APIs.
    """

    def __init__(self, api_key=None):
        """
        Initialize the API client with an API key.
        """
        self.api_key = api_key or self.get_api_key_from_env()

    @abc.abstractmethod
    def get_api_key_from_env(self):
        """
        Retrieve the API key from environment variables.
        """
        pass

    @abc.abstractmethod
    def get_response(self, prompt, **kwargs):
        """
        Get a response from the model based on the prompt.
        """
        pass
