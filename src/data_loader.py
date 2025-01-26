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

    # @staticmethod
    # def _load_from_file(file_path):
    #     if file_path.suffix == '.joblib':
    #         return joblib.load(file_path)
    #     elif file_path.suffix == '.csv':
    #         return DataLoader._load_from_csv(file_path)
    #     elif file_path.suffix == '.json':
    #         return DataLoader._load_from_json(file_path)
    #     else:
    #         raise ValueError(f"Unsupported file type: {file_path}")

    # @staticmethod
    # def _load_from_directory(directory_path):
    #     files = {f.stem: f for f in directory_path.iterdir() if f.suffix == '.csv'}
    #     required_files = {'X_train', 'y_train', 'X_test', 'y_test'}
        
    #     if not required_files.issubset(files.keys()):
    #         raise ValueError(f"Directory must contain files: {', '.join(required_files)}")
        
    #     return {
    #         'X_train': pd.read_csv(files['X_train']).to_numpy(),
    #         'y_train': pd.read_csv(files['y_train']).to_numpy().ravel(),
    #         'X_test': pd.read_csv(files['X_test']).to_numpy(),
    #         'y_test': pd.read_csv(files['y_test']).to_numpy().ravel(),
    #     }

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


    # Updated _load_from_directory and _load_from_file methods
    @staticmethod
    def _load_from_file(file_path):
        if file_path.suffix == '.joblib':
            data = joblib.load(file_path)
            return DataLoader._handle_data_split(data)
        elif file_path.suffix == '.csv':
            return DataLoader._handle_csv_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")


    @staticmethod
    def _load_from_directory(directory_path):
        """
        Load data from a directory. Supports:
        1. Pre-split files: X_train.csv, y_train.csv, X_test.csv, y_test.csv.
        2. Unsplit files: X.csv, y.csv (splits into train and test internally).
        """
        from sklearn.model_selection import train_test_split
    
        # Collect all CSV files in the directory
        files = {f.stem: f for f in directory_path.iterdir() if f.suffix == '.csv'}
    
        # Case 1: Pre-split data
        required_files_pre_split = {'X_train', 'y_train', 'X_test', 'y_test'}
        if required_files_pre_split.issubset(files.keys()):
            return {
                'X_train': pd.read_csv(files['X_train']).to_numpy(),
                'y_train': pd.read_csv(files['y_train']).to_numpy().ravel(),
                'X_test': pd.read_csv(files['X_test']).to_numpy(),
                'y_test': pd.read_csv(files['y_test']).to_numpy().ravel(),
            }
    
        # Case 2: Unsplit data (X.csv and y.csv)
        required_files_unsplit = {'X', 'y'}
        if required_files_unsplit.issubset(files.keys()):
            X = pd.read_csv(files['X']).to_numpy()
            y = pd.read_csv(files['y']).to_numpy().ravel()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
            }
    
        # If neither case is satisfied, raise an error
        raise ValueError(
            "Directory must contain either:\n"
            "1. Pre-split files: X_train.csv, y_train.csv, X_test.csv, y_test.csv, or\n"
            "2. Unsplit files: X.csv and y.csv."
        )

    
    # @staticmethod
    # def _load_from_directory(directory_path):
    #     files = {f.stem: f for f in directory_path.iterdir() if f.suffix == '.csv'}
    #     if 'X' in files and 'y' in files:
    #         X = pd.read_csv(files['X']).to_numpy()
    #         y = pd.read_csv(files['y']).to_numpy().ravel()
    #         return DataLoader._split_data(X, y)
    #     else:
    #         return super()._load_from_directory(directory_path)
    
    @staticmethod
    def _handle_data_split(data):
        if 'X_train' in data and 'y_train' in data:
            return data
        elif 'X' in data and 'y' in data:
            return DataLoader._split_data(data['X'], data['y'])
        else:
            raise ValueError("Input data must contain either ('X_train', 'y_train', 'X_test', 'y_test') or ('X', 'y').")
    
    @staticmethod
    def _handle_csv_file(file_path):
        """
        Load data from a CSV file. If the columns 'X' and 'y' are present, use them. 
        Otherwise, assume the last column is the target ('y') and all preceding columns are features ('X').
        """
        df = pd.read_csv(file_path)
    
        # Check if 'X' and 'y' are explicitly labeled
        if 'X' in df.columns and 'y' in df.columns:
            X = df.drop('y', axis=1).to_numpy()
            y = df['y'].to_numpy()
        else:
            # Assume the last column is the target and all preceding columns are features
            X = df.iloc[:, :-1].to_numpy()  # All columns except the last
            y = df.iloc[:, -1].to_numpy()   # Last column
    
        return DataLoader._split_data(X, y)

    
    @staticmethod
    def _split_data(X, y, test_size=0.2, random_state=42):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
