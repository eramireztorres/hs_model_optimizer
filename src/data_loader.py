import pandas as pd
import json
import joblib
from pathlib import Path


class DataLoader:
    @staticmethod
    def load_data(input_path):
        """
        Load data from a directory or file and standardize it into a dictionary format.

        Args:
            input_path (str): Path to the input file or directory.

        Returns:
            dict: A dictionary containing 'X_train', 'y_train', 'X_test', and 'y_test'.

        Raises:
            ValueError: If the input is invalid or unsupported.
        """
        input_path = Path(input_path)

        if input_path.is_file():
            # Handle file input (e.g., .joblib, .csv, .json)
            return DataLoader._load_from_file(input_path)
        elif input_path.is_dir():
            # Handle directory input
            return DataLoader._load_from_directory(input_path)
        else:
            raise ValueError(f"Invalid input path: {input_path}")

    @staticmethod
    def _load_from_file(file_path):
        if file_path.suffix == '.joblib':
            return joblib.load(file_path)
        elif file_path.suffix == '.csv':
            return DataLoader._load_from_csv(file_path)
        elif file_path.suffix == '.json':
            return DataLoader._load_from_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    @staticmethod
    def _load_from_directory(directory_path):
        files = {f.stem: f for f in directory_path.iterdir() if f.suffix == '.csv'}
        required_files = {'X_train', 'y_train', 'X_test', 'y_test'}
        
        if not required_files.issubset(files.keys()):
            raise ValueError(f"Directory must contain files: {', '.join(required_files)}")
        
        return {
            'X_train': pd.read_csv(files['X_train']).to_numpy(),
            'y_train': pd.read_csv(files['y_train']).to_numpy().ravel(),
            'X_test': pd.read_csv(files['X_test']).to_numpy(),
            'y_test': pd.read_csv(files['y_test']).to_numpy().ravel(),
        }

    @staticmethod
    def _load_from_csv(file_path):
        df = pd.read_csv(file_path)
        if {'X_train', 'y_train', 'X_test', 'y_test'}.issubset(df.columns):
            return {
                'X_train': df['X_train'].to_numpy(),
                'y_train': df['y_train'].to_numpy(),
                'X_test': df['X_test'].to_numpy(),
                'y_test': df['y_test'].to_numpy(),
            }
        else:
            raise ValueError("CSV file must contain 'X_train', 'y_train', 'X_test', and 'y_test' columns.")

    @staticmethod
    def _load_from_json(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        if {'X_train', 'y_train', 'X_test', 'y_test'}.issubset(data.keys()):
            return data
        else:
            raise ValueError("JSON file must contain 'X_train', 'y_train', 'X_test', and 'y_test' keys.")
