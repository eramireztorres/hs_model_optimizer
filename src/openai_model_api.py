import os
# import openai

import openai
from openai._base_client import SyncHttpxClientWrapper

_old_init = SyncHttpxClientWrapper.__init__

def new_init(self, *args, **kwargs):
    kwargs.pop("proxies", None)
    return _old_init(self, *args, **kwargs)

SyncHttpxClientWrapper.__init__ = new_init



from base_model_api import BaseModelAPI

class OpenAIModelAPI(BaseModelAPI):
    """
    A unified OpenAI API client that works with both legacy GPT models and the new reasoning models (o1/o3).
    """

    def __init__(self, api_key=None, model='gpt-4o'):
        super().__init__(api_key)
        self.api_key = api_key or self.get_api_key_from_env()
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
        self.conversation_history = []

    def get_api_key_from_env(self):
        """Retrieve the OpenAI API key from environment variables."""
        return os.getenv('OPENAI_API_KEY')

    def get_response(self, prompt, max_tokens=2048, temperature=0.5):
        # Append the user prompt to the conversation history.
        self.conversation_history.append({"role": "user", "content": prompt})

        is_o_series = ("o1" in self.model) or ("o3" in self.model) or ("gpt-5-nano" in self.model)

        # Base params common to all attempts
        base_params = {
            "model": self.model,
            "messages": self.conversation_history,
        }

        # Include temperature only when not using o-series models
        if not is_o_series:
            base_params["temperature"] = temperature

        # Decide preferred vs fallback token param based on model hint
        preferred_key = "max_completion_tokens" if is_o_series else "max_tokens"
        fallback_key = "max_tokens" if preferred_key == "max_completion_tokens" else "max_completion_tokens"

        def try_call(params):
            return self.client.chat.completions.create(**params)

        # First attempt: preferred key
        params = dict(base_params)
        params[preferred_key] = max_tokens
        try:
            response = try_call(params)
        except Exception as e:
            msg = str(e)
            # If error suggests the token key is unsupported, swap to the fallback
            unsupported_pref = (preferred_key in msg) or ("unsupported_parameter" in msg and preferred_key.replace("_", "") in msg.replace("_", ""))
            if unsupported_pref:
                params.pop(preferred_key, None)
                params[fallback_key] = max_tokens
                try:
                    response = try_call(params)
                except Exception as e2:
                    # Some models may also reject temperature; try once more without it.
                    msg2 = str(e2)
                    temp_unsupported = "temperature" in msg2 and ("Unsupported" in msg2 or "unsupported" in msg2)
                    if temp_unsupported and "temperature" in params:
                        params.pop("temperature", None)
                        try:
                            response = try_call(params)
                        except Exception as e3:
                            print(f"An error occurred: {e3}")
                            return None
                    else:
                        print(f"An error occurred: {e2}")
                        return None
            else:
                # If failure is for other reasons, surface it
                print(f"An error occurred: {e}")
                return None

        # Extract and store the assistant's reply.
        assistant_response = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        return assistant_response.strip()



# class OpenAIModelAPI(BaseModelAPI):
#     """
#     OpenAI GPT-4 API client.
#     """

#     def __init__(self, api_key=None, model='gpt-4o'):
#         super().__init__(api_key)
#         self.model = model
#         self.client = openai.OpenAI(api_key=self.api_key)
#         self.conversation_history = []

#     def get_api_key_from_env(self):
#         """
#         Retrieve the OpenAI API key from environment variables.
#         """
#         return os.getenv('OPENAI_API_KEY')

#     def get_response(self, prompt, max_tokens=1024, temperature=0.5):
#         """
#         Get a response from the OpenAI model.
#         """
        
#         # print(f'PROMPT:  \n {prompt}')
        
#         # Add user message to conversation history
#         self.conversation_history.append({"role": "user", "content": prompt})

#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=self.conversation_history,
#                 temperature=temperature,
#                 max_tokens=max_tokens
#             )
#         except Exception as e:
#             print(f"An error occurred: {e}")
#             return None

#         # Extract the assistant's reply from the response
#         assistant_response = response.choices[0].message.content

#         # Add assistant's reply to the conversation history
#         self.conversation_history.append({"role": "assistant", "content": assistant_response})

#         return assistant_response.strip()

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
