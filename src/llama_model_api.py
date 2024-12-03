import os
import requests
from base_model_api import BaseModelAPI

class LlamaModelAPI(BaseModelAPI):
    """
    Llama model API client.
    """

    def __init__(self, api_key=None, model='meta-llama/llama-3.1-405b-instruct:free'):
        super().__init__(api_key)
        self.model = model
        self.url = 'https://openrouter.ai/api/v1/chat/completions'

    def get_api_key_from_env(self):
        """
        Retrieve the Llama API key from environment variables.
        """
        return os.getenv('LLAMA_API_KEY') or os.getenv('OPENROUTER_API_KEY')

    def get_response(self, prompt, max_tokens=512, temperature=0.7):
        """
        Get a response from the Llama model.
        """
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            completion = response.json()
            if 'choices' in completion and len(completion['choices']) > 0:
                assistant_response = completion['choices'][0]['message']['content']
                return assistant_response.strip()
            else:
                print(f"Unexpected response structure: {completion}")
                return None

        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response Body: {e.response.text}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
