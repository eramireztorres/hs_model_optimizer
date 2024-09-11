import joblib
import os

class DataHandler:
    def __init__(self):
        """
        Initialize the DataHandler class.
        """
        self.data = None

    def load_data(self, filepath):
        """
        Load training and test data from a joblib file.

        Args:
            filepath (str): The path to the joblib file containing data.

        Returns:
            dict: A dictionary containing 'X_train', 'y_train', 'X_test', 'y_test'.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} does not exist.")
        
        self.data = joblib.load(filepath)
        
        required_keys = ['X_train', 'y_train', 'X_test', 'y_test']
        for key in required_keys:
            if key not in self.data:
                raise KeyError(f"Missing key {key} in the loaded data.")

        return self.data

    def save_model_history(self, models, metrics, filepath):
        """
        Save the history of models and their performance metrics.

        Args:
            models (list): List of trained models.
            metrics (list): List of metrics corresponding to each model.
            filepath (str): The path where the history should be saved.

        Returns:
            None
        """
        history = {
            'models': models,
            'metrics': metrics
        }
        joblib.dump(history, filepath)
        print(f"Model history saved to {filepath}")



