import os
import joblib
import logging
import json

class ModelHistoryManager:
    def __init__(self, history_file_path='model_history.joblib'):
        """
        Initialize the ModelHistoryManager.

        Args:
            history_file_path (str): Path to the file where the model history will be saved.
        """
        self.history_file_path = history_file_path
        self.model_history = []
        self.is_text_file = history_file_path.endswith('.txt')

    def save_model_history(self, model_code, metrics):
        """
        Save the current model's code and metrics into the history.

        Args:
            model_code (str): The Python code of the model.
            metrics (dict): The performance metrics of the model.
        """
        history_entry = {
            'model_code': model_code,
            'metrics': metrics
        }
        self.model_history.append(history_entry)

        try:
            if self.is_text_file:
                self._save_as_text()
            else:
                self._save_as_joblib()
        except Exception as e:
            logging.error(f"Failed to save model history: {e}")

    def load_model_history(self):
        """
        Load the model history from the specified file.

        Returns:
            list: The history of models, including their code and performance metrics.
        """
        if not os.path.exists(self.history_file_path):
            logging.warning(f"No existing history found at {self.history_file_path}. Starting with an empty history.")
            self.model_history = []
            return self.model_history

        try:
            if self.is_text_file:
                self._load_from_text()
            else:
                self._load_from_joblib()
        except Exception as e:
            logging.error(f"Failed to load model history: {e}")
            self.model_history = []

        return self.model_history

    def _save_as_joblib(self):
        """Save the history as a joblib file."""
        joblib.dump(self.model_history, self.history_file_path)
        logging.info(f"Model history saved to {self.history_file_path} (joblib format)")

    def _load_from_joblib(self):
        """Load the history from a joblib file."""
        self.model_history = joblib.load(self.history_file_path)
        logging.info(f"Model history loaded from {self.history_file_path} (joblib format)")

    def _save_as_text(self):
        """Save the history as a human-readable text file."""
        with open(self.history_file_path, 'w') as f:
            for idx, entry in enumerate(self.model_history, 1):
                f.write(f"=== Iteration {idx} ===\n")
                f.write("Model Code:\n")
                f.write(f"{entry['model_code']}\n\n")
                f.write("Metrics:\n")
                json.dump(entry['metrics'], f, indent=4)
                f.write("\n\n")
        logging.info(f"Model history saved to {self.history_file_path} (text format)")

    def _load_from_text(self):
        """Load the history from a text file."""
        try:
            with open(self.history_file_path, 'r') as f:
                # For simplicity, assumes the text file is formatted correctly.
                self.model_history = []
                lines = f.readlines()
                current_entry = {}
                for line in lines:
                    if line.startswith("=== Iteration"):
                        if current_entry:
                            self.model_history.append(current_entry)
                            current_entry = {}
                    elif line.startswith("Model Code:"):
                        current_entry['model_code'] = ""
                    elif line.startswith("Metrics:"):
                        metrics_json = line.strip().split("Metrics:")[1]
                        current_entry['metrics'] = json.loads(metrics_json)
                    elif 'model_code' in current_entry and not current_entry['model_code']:
                        current_entry['model_code'] += line.strip()
                if current_entry:
                    self.model_history.append(current_entry)
            logging.info(f"Model history loaded from {self.history_file_path} (text format)")
        except Exception as e:
            raise ValueError(f"Failed to parse text history file: {e}")


# class ModelHistoryManager:
#     def __init__(self, history_file_path='model_history.joblib'):
#         """
#         Initialize the ModelHistoryManager.

#         Args:
#             history_file_path (str): Path to the file where the model history will be saved.
#         """
#         self.history_file_path = history_file_path
#         self.model_history = []

#     def save_model_history(self, model_code, metrics):
#         """
#         Save the current model's code and metrics into the history.

#         Args:
#             model_code (str): The Python code of the model.
#             metrics (dict): The performance metrics of the model.
#         """
#         history_entry = {
#             'model_code': model_code,
#             'metrics': metrics
#         }
#         self.model_history.append(history_entry)

#         try:
#             joblib.dump(self.model_history, self.history_file_path)
#             logging.info(f"Model history saved to {self.history_file_path}")
#         except Exception as e:
#             logging.error(f"Failed to save model history: {e}")

#     def load_model_history(self):
#         """
#         Load the model history from the joblib file.

#         Returns:
#             list: The history of models, including their code and performance metrics.
#         """
#         try:
#             self.model_history = joblib.load(self.history_file_path)
#             logging.info(f"Model history loaded from {self.history_file_path}")
#         except FileNotFoundError:
#             logging.warning(f"No existing history found at {self.history_file_path}. Starting with an empty history.")
#             self.model_history = []
#         except Exception as e:
#             logging.error(f"Failed to load model history: {e}")
#             self.model_history = []

#         return self.model_history
