import os
import logging
import google.generativeai as genai
from base_model_api import BaseModelAPI

class GeminiModelAPI(BaseModelAPI):
    """
    Gemini model API client.
    """

    def __init__(self, api_key=None, model='models/chat-bison-001'):
        super().__init__(api_key)
        self.model = model
        genai.configure(api_key=self.api_key)
        logging.basicConfig(level=logging.INFO)
        logging.info("Gemini API configured successfully.")

    def get_api_key_from_env(self):
        """
        Retrieve the Gemini API key from environment variables.
        """
        return os.getenv('GEMINI_API_KEY')

    def get_response(self, prompt, **kwargs):
        """
        Get a response from the Gemini model.
        """
        try:
            response = genai.generate_text(
                model=self.model,
                prompt=prompt,
                **kwargs
            )
            return self.format_response(response)
        except Exception as e:
            logging.error(f"Error in generate_text: {e}")
            return None

    @staticmethod
    def format_response(response):
        """
        Format the API response for improved readability.
        """
        if hasattr(response, 'result'):
            return response.result.strip()
        elif isinstance(response, dict):
            return "\n".join([f"{k}: {v}" for k, v in response.items()]).strip()
        else:
            return str(response).strip()
