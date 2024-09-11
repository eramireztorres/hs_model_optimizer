import logging

class LLMImprover:
    def __init__(self, llm_model):
        """
        Initialize the LLMImprover class.

        Args:
            llm_model (object): The instance of the LLM model (e.g., Gemini API or OpenAI API).
        """
        self.llm_model = llm_model

    def propose_improvement(self, models_history, metrics_history):
        """
        Use the LLM to generate improvements for the models and their hyperparameters based on past results.

        Args:
            models_history (list): List of Python code snippets representing the models used so far.
            metrics_history (list): List of dictionaries containing metrics for each model.

        Returns:
            str: The improved model code suggested by the LLM.
        """
        # Prepare the prompt for the LLM by including all model codes and their associated metrics history
        prompt = self._format_prompt(models_history, metrics_history)

        try:
            # Use the LLM to propose an improvement to the models or their hyperparameters
            improved_code = self.llm_model.get_response(prompt)
            logging.info("LLM proposed an improvement to the model code.")
            return improved_code
        except Exception as e:
            logging.error(f"Error in LLMImprover: {e}")
            return None

    def _format_prompt(self, models_history, metrics_history):
        """
        Format the prompt that will be sent to the LLM for proposing improvements.

        Args:
            models_history (list): List of Python code snippets representing the models used so far.
            metrics_history (list): List of metrics for each model.

        Returns:
            str: A formatted prompt for the LLM.
        """
        # Combine all the model codes and their corresponding metrics in a comprehensive prompt
        model_metrics_pairs = [
            f"Model {i+1}:\n{models_history[i]}\nMetrics:\n{metrics_history[i]}\n"
            for i in range(len(models_history))
        ]
        model_metrics_text = "\n===\n".join(model_metrics_pairs)

        prompt_template = """
        You are an expert Python developer. Below is the history of all the machine learning models used so far, along with their performance metrics.

        Models and Metrics History:
        {model_metrics_text}

        Based on this information, suggest an improvement to the model code or the hyperparameters to enhance its performance. Provide only the updated Python code.
        """
        # Format the prompt with the model history and their associated metrics
        formatted_prompt = prompt_template.format(
            model_metrics_text=model_metrics_text
        )
        return formatted_prompt
