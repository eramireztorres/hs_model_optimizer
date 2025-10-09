import os
# import openai

import openai
from openai._base_client import SyncHttpxClientWrapper

_old_init = SyncHttpxClientWrapper.__init__

def new_init(self, *args, **kwargs):
    kwargs.pop("proxies", None)
    return _old_init(self, *args, **kwargs)

SyncHttpxClientWrapper.__init__ = new_init



from .base_model_api import BaseModelAPI

class OpenAIModelAPI(BaseModelAPI):
    """
    A unified OpenAI API client that works with both legacy GPT models and the new reasoning models (o1/o3).
    """

    def __init__(self, api_key=None, model='gpt-4.1-mini'):
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

        is_o_series = ("o1" in self.model) or ("o3" in self.model) or ("gpt-5" in self.model)

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



