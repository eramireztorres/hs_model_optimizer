import os
import anthropic
from .base_model_api import BaseModelAPI


class AnthropicModelAPI(BaseModelAPI):
    """
    Anthropic model API client that uses the Messages API and is compatible with BaseModelAPI.
    """

    def __init__(self, api_key=None, model="claude-3-sonnet-20240229"):
        """
        Initialize the Anthropic API client.
        
        Args:
            api_key (str): Your Anthropic API key. If None, it is read from the ANTHROPIC_API_KEY environment variable.
            model (str): The Anthropic model to use (e.g., "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-7-sonnet-20250219").
        """
        super().__init__(api_key)
        self.api_key = api_key or self.get_api_key_from_env()
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.conversation_history = []  # Optionally store conversation history

    def get_api_key_from_env(self):
        """
        Retrieve the Anthropic API key from the environment.
        """
        return os.getenv("ANTHROPIC_API_KEY")

    def get_response(self, prompt, max_tokens=8000, temperature=0.7, stop_sequences=None):
        """
        Get a response from the Anthropic model using the Messages API.
        
        Args:
            prompt (str): The user prompt.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            stop_sequences (list, optional): List of sequences at which to stop generation.
        
        Returns:
            str: The generated assistant response, or None if an error occurs.
        """
        try:
            # Using the Messages API format
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system="",  # Optional system prompt
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stop_sequences=stop_sequences
            )

            # Extract the content from the response
            completion = response.content[0].text

            # Store in conversation history if needed
            self.conversation_history.append({"role": "assistant", "content": completion})

            return completion
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def continue_conversation(self, new_prompt, max_tokens=12000, temperature=0.7, stop_sequences=None):
        """
        Continue a conversation with the model by sending the entire conversation history.
        
        Args:
            new_prompt (str): The new user prompt to add to the conversation.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            stop_sequences (list, optional): List of sequences at which to stop generation.
        
        Returns:
            str: The generated assistant response, or None if an error occurs.
        """
        # Add the new user message to the conversation history
        self.conversation_history.append({"role": "user", "content": new_prompt})

        try:
            # Create messages array from conversation history
            messages = []
            for entry in self.conversation_history:
                messages.append({"role": entry["role"], "content": entry["content"]})

            # Using the Messages API with conversation history
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system="",  # Optional system prompt
                messages=messages,
                stop_sequences=stop_sequences
            )

            # Extract the content from the response
            completion = response.content[0].text

            # Add the assistant's response to the conversation history
            self.conversation_history.append({"role": "assistant", "content": completion})

            return completion
        except Exception as e:
            print(f"An error occurred: {e}")
            # Remove the last user message from history if we couldn't get a response
            if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                self.conversation_history.pop()
            return None

    def reset_conversation(self):
        """
        Reset the conversation history.
        """
        self.conversation_history = []
