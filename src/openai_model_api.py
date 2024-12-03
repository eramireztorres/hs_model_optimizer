import os
import openai
from base_model_api import BaseModelAPI
import requests

class OpenAIModelAPI(BaseModelAPI):
    """
    OpenAI GPT-4 API client.
    """

    def __init__(self, api_key=None, model='gpt-4o'):
        super().__init__(api_key)
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
        self.conversation_history = []

    def get_api_key_from_env(self):
        """
        Retrieve the OpenAI API key from environment variables.
        """
        return os.getenv('OPENAI_API_KEY')

    def get_response(self, prompt, max_tokens=1024, temperature=0.5):
        """
        Get a response from the OpenAI model.
        """
        
        # print(f'PROMPT:  \n {prompt}')
        
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

        # Extract the assistant's reply from the response
        assistant_response = response.choices[0].message.content

        # Add assistant's reply to the conversation history
        self.conversation_history.append({"role": "assistant", "content": assistant_response})

        return assistant_response.strip()

# class OpenAIModelAPI(BaseModelAPI):
#     def __init__(self, api_key=None, model='gpt-4', **kwargs):
#         """
#         Initialize the OpenAIModelAPI with a specific model.

#         Args:
#             api_key (str): The API key for OpenAI.
#             model (str): The specific OpenAI model to use (e.g., 'gpt-4', 'gpt-4o').
#         """
#         # self.api_key = api_key
#         super().__init__(api_key)
#         self.model = model

#     def get_response(self, prompt):
#         """
#         Send a prompt to the OpenAI API and get the response.

#         Args:
#             prompt (str): The input prompt.

#         Returns:
#             str: The generated response.
#         """
#         response = requests.post(
#             "https://api.openai.com/v1/completions",
#             headers={"Authorization": f"Bearer {self.api_key}"},
#             json={
#                 "model": self.model,  # Ensure the model parameter is included
#                 "prompt": prompt,
#                 "max_tokens": 150,
#                 "temperature": 0.7,
#             },
#         )
#         if response.status_code != 200:
#             raise ValueError(f"Error code: {response.status_code} - {response.json()}")
#         return response.json().get("choices")[0].get("text").strip()

#     def get_api_key_from_env(self):
#         """
#         Retrieve the OpenAI API key from environment variables.
#         """
#         return os.getenv('OPENAI_API_KEY')