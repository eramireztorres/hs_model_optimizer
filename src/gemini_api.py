import logging
import google.generativeai as genai

class GeminiAPI:
    """
    Enhanced Gemini API wrapper for local use in a Python environment.
    """

    def __init__(self, api_key, project_id, default_model='gemini-pro'):
        """
        Initialize the Gemini API wrapper.

        Args:
            api_key (str): Your Google Cloud API key.
            project_id (str): The ID of your Google Cloud project.
            default_model (str): Default model for content generation.
        """

        self.api_key = api_key
        self.project_id = project_id
        self.default_model = default_model

        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        logging.basicConfig(level=logging.INFO)
        logging.info("Gemini API configured successfully.")

    def generate_content(self, text, model_id=None, stream=False):
        """
        Generate text using a specified model.

        Args:
            text (str): The text prompt for the model.
            model_id (str): Model ID to use. If None, uses default model.
            stream (bool): True to stream the response, False to return it as a single object.

        Returns:
            str: The generated text response.
        """
        model_id = model_id if model_id else self.default_model
        try:
            model = genai.GenerativeModel(model_id)
            response = model.generate_content(text, stream=stream)
            return self.format_response(response)
        except Exception as e:
            logging.error(f"Error in generate_content: {e}")
            return None

    def embed_content(self, content, model_id="models/embedding-001", task_type="retrieval_document", title=""):
        """
        Embed text using the specified model.

        Args:
            content (str or list): The text or image content to embed.
            model_id (str): The ID of the model to use for embedding.
            task_type (str): The type of task for which the embedding is used.
            title (str): A title for the embedding.

        Returns:
            dict: The embedding result.
        """
        try:
            result = genai.embed_content(
                model=model_id,
                content=content,
                task_type=task_type,
                title=title
            )
            return result
        except Exception as e:
            logging.error(f"Error in embed_content: {e}")
            return None

    @staticmethod
    def format_response(response):
        """
        Format the API response for improved readability.

        Args:
            response (object): The response object from the API.

        Returns:
            str: Formatted response text.
        """
        formatted_text = ""

        if hasattr(response, 'text'):
            response_text = response.text
        elif isinstance(response, dict):
            response_text = "\n".join([f"{k}: {v}" for k, v in response.items()])
        else:
            response_text = str(response)

        # Remove Markdown bullets and bold formatting
        for line in response_text.split("\n"):
            line = line.replace("* ", "")  # Remove bullet points
            line = line.replace("**", "")  # Remove bold formatting
            formatted_text += line + "\n"

        return formatted_text.strip()

    def get_response(self, prompt):
        response = self.generate_content(prompt)
        return self.format_response(response)

