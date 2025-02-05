import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


    # @staticmethod
    # def _load_from_directory(directory_path):
    #     """
    #     Load data from a directory. Supports:
    #     1. Pre-split files: X_train.csv, y_train.csv, X_test.csv, y_test.csv.
    #     2. Unsplit files: X.csv, y.csv (splits into train and test internally).
    #     """
    #     from sklearn.model_selection import train_test_split
    
    #     # Collect all CSV files in the directory
    #     files = {f.stem: f for f in directory_path.iterdir() if f.suffix == '.csv'}
    
    #     # Case 1: Pre-split data
    #     required_files_pre_split = {'X_train', 'y_train', 'X_test', 'y_test'}
    #     if required_files_pre_split.issubset(files.keys()):
    #         return {
    #             'X_train': pd.read_csv(files['X_train']).to_numpy(),
    #             'y_train': pd.read_csv(files['y_train']).to_numpy().ravel(),
    #             'X_test': pd.read_csv(files['X_test']).to_numpy(),
    #             'y_test': pd.read_csv(files['y_test']).to_numpy().ravel(),
    #         }
    
    #     # Case 2: Unsplit data (X.csv and y.csv)
    #     required_files_unsplit = {'X', 'y'}
    #     if required_files_unsplit.issubset(files.keys()):
    #         X = pd.read_csv(files['X']).to_numpy()
    #         y = pd.read_csv(files['y']).to_numpy().ravel()
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #         return {
    #             'X_train': X_train,
    #             'y_train': y_train,
    #             'X_test': X_test,
    #             'y_test': y_test,
    #         }
    
    #     # If neither case is satisfied, raise an error
    #     raise ValueError(
    #         "Directory must contain either:\n"
    #         "1. Pre-split files: X_train.csv, y_train.csv, X_test.csv, y_test.csv, or\n"
    #         "2. Unsplit files: X.csv and y.csv."
    #     )


    @staticmethod
    def _load_from_directory(directory_path):
        """
        Load data from a directory, handling cases with pre-split or unsplit CSV files.
        """
        files = {f.stem: f for f in directory_path.iterdir() if f.suffix == '.csv'}
        
        # Case 1: Pre-split data
        if {'X_train', 'y_train', 'X_test', 'y_test'}.issubset(files.keys()):
            X_train = pd.read_csv(files['X_train'])
            y_train = pd.read_csv(files['y_train']).squeeze()
            X_test = pd.read_csv(files['X_test'])
            y_test = pd.read_csv(files['y_test']).squeeze()
            
            # Identify categorical columns in training data
            categorical_cols = X_train.select_dtypes(include=['object']).columns

            if len(categorical_cols) > 0:
                print(f"Encoding categorical columns: {list(categorical_cols)}")
                X_train = DataLoader._encode_categorical(X_train, categorical_cols)
                X_test = DataLoader._encode_categorical(X_test, categorical_cols, fit=False)
                
                
            if y_train.dtype == 'object' or isinstance(y_train.iloc[0], str):
                print("Encoding categorical target labels in y_train.")
                y_train = pd.factorize(y_train)[0]
            
            if y_test.dtype == 'object' or isinstance(y_test.iloc[0], str):
                print("Encoding categorical target labels in y_test.")
                y_test = pd.factorize(y_test)[0]

            
            data = {
                'X_train': X_train.to_numpy(),
                'y_train': y_train.to_numpy(),
                'X_test': X_test.to_numpy(),
                'y_test': y_test.to_numpy()
            }
            data['is_pre_split'] = True
            return data

        # Case 2: Unsplit data (X.csv, y.csv)
        # elif {'X', 'y'}.issubset(files.keys()):
        #     X = pd.read_csv(files['X'])
        #     y = pd.read_csv(files['y']).squeeze()

        #     # Identify categorical columns
        #     categorical_cols = X.select_dtypes(include=['object']).columns

        #     if len(categorical_cols) > 0:
        #         print(f"Encoding categorical columns: {list(categorical_cols)}")
        #         X = DataLoader._encode_categorical(X, categorical_cols)

        #     return DataLoader._split_data(X, y)
        
        elif {'X', 'y'}.issubset(files.keys()):
            X = pd.read_csv(files['X'])
            y = pd.read_csv(files['y']).squeeze()
            
            categorical_cols = X.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                print(f"Encoding categorical columns: {list(categorical_cols)}")
                X = DataLoader._encode_categorical(X, categorical_cols)
            
            # Return unsplit data with flag
            return {'X': X, 'y': y, 'is_pre_split': False}

        
        
        # Fill missing numerical values
        for col in X_train.select_dtypes(include=['number']).columns:
            X_train[col] = X_train[col].fillna(X_train[col].median())
        
        for col in X_test.select_dtypes(include=['number']).columns:
            X_test[col] = X_test[col].fillna(X_test[col].median())


        raise ValueError("Invalid directory structure: Must contain either ('X_train', 'y_train', 'X_test', 'y_test') or ('X', 'y').")


    @staticmethod
    def _encode_categorical(X, categorical_cols, fit=True):
        """
        Encode categorical features using One-Hot Encoding.

        Args:
            X (pd.DataFrame): Feature dataframe.
            categorical_cols (list): List of categorical column names.
            fit (bool): Whether to fit a new encoder or use an existing one.

        Returns:
            pd.DataFrame: Transformed feature dataframe.
        """
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X[categorical_cols] = X[categorical_cols].fillna("Missing")  # Fill NaN values


        if fit:
            transformed = encoder.fit_transform(X[categorical_cols])
        else:
            transformed = encoder.transform(X[categorical_cols])

        # Convert back to DataFrame
        encoded_df = pd.DataFrame(transformed, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)

        # Drop original categorical columns and concatenate new encoded columns
        X = X.drop(columns=categorical_cols)
        X = pd.concat([X, encoded_df], axis=1)

        return X

    
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
            data['is_pre_split'] = True
            return data
        elif 'X' in data and 'y' in data:
            # Return unsplit data along with a flag indicating unsplit data.
            return {'X': data['X'], 'y': data['y'], 'is_pre_split': False}

        else:
            raise ValueError("Input data must contain either ('X_train', 'y_train', 'X_test', 'y_test') or ('X', 'y').")
    
    @staticmethod
    def _handle_csv_file(file_path):
        df = pd.read_csv(file_path)
        
        # Identify target column (last column assumed to be the target)
        target_col = df.columns[-1]
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical target encoding if necessary
        if y.dtype == 'object' or isinstance(y.iloc[0], str):
            print("Encoding categorical target labels.")
            y = pd.factorize(y)[0]  # Encode string labels as integers
        
        # Fill missing numerical values with the median
        for col in X.select_dtypes(include=['number']).columns:
            X[col] = X[col].fillna(X[col].median())
        
        # Identify and encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"Encoding categorical columns: {list(categorical_cols)}")
            X = DataLoader._encode_categorical(X, categorical_cols)
        
        # Standardize numeric values
        X.fillna(0, inplace=True)
        X = X.astype('float32')
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Instead of splitting, return the unsplit data along with a flag.
        # return {'X': X, 'y': y, 'is_pre_split': False}
        return {'X': X if isinstance(X, np.ndarray) else X.to_numpy(), 'y': y if isinstance(y, np.ndarray) else y.to_numpy(), 'is_pre_split': False}
    
    # @staticmethod
    # def _handle_csv_file(file_path):
    #     df = pd.read_csv(file_path)
        
    #     # Identify target column (last column assumed to be the target)
    #     target_col = df.columns[-1]
    #     X = df.drop(columns=[target_col])
    #     y = df[target_col]
        
    #     # Handle categorical target encoding if necessary
    #     if y.dtype == 'object' or isinstance(y.iloc[0], str):
    #         print("Encoding categorical target labels.")
    #         y = pd.factorize(y)[0]  # Encode string labels as integers
        
    #     # Standard preprocessing for features
    #     for col in X.select_dtypes(include=['number']).columns:
    #         X[col] = X[col].fillna(X[col].median())
        
    #     categorical_cols = X.select_dtypes(include=['object']).columns
    #     if len(categorical_cols) > 0:
    #         print(f"Encoding categorical columns: {list(categorical_cols)}")
    #         X = DataLoader._encode_categorical(X, categorical_cols)
        
    #     # Standardize numeric features
    #     scaler = StandardScaler()
    #     X_scaled = scaler.fit_transform(X)
        
    #     return {'X': X_scaled, 'y': y if isinstance(y, np.ndarray) else y.to_numpy(), 'is_pre_split': False}


  
    
    # @staticmethod
    # def _handle_csv_file(file_path):
    #     df = pd.read_csv(file_path)
        
    #     # Identify target column (last column assumed to be the target)
    #     target_col = df.columns[-1]
    #     X = df.drop(columns=[target_col])
    #     y = df[target_col]
        
    #     # Fill missing numerical values with the median
    #     for col in X.select_dtypes(include=['number']).columns:
    #         X[col] = X[col].fillna(X[col].median())
        
    #     # Identify and encode categorical features
    #     categorical_cols = X.select_dtypes(include=['object']).columns
    #     if len(categorical_cols) > 0:
    #         print(f"Encoding categorical columns: {list(categorical_cols)}")
    #         X = DataLoader._encode_categorical(X, categorical_cols)
        
    #     # Standardize numeric values
    #     X.fillna(0, inplace=True)
    #     X = X.astype('float32')
    #     X.replace([np.inf, -np.inf], np.nan, inplace=True)
    #     X.fillna(0, inplace=True)
    #     scaler = StandardScaler()
    #     X_scaled = scaler.fit_transform(X)
    #     X = pd.DataFrame(X_scaled, columns=X.columns)
        
    #     # Instead of splitting, return the unsplit data along with a flag.
    #     # return {'X': X, 'y': y, 'is_pre_split': False}
    #     return {'X': X.to_numpy(), 'y': y.to_numpy(), 'is_pre_split': False}

    # @staticmethod
    # def _handle_csv_file(file_path):
    #     """
    #     Load data from a CSV file. Automatically detects categorical features and encodes them.
    #     """
    #     df = pd.read_csv(file_path)
    
    #     # Identify target column (last column assumed to be the target)
    #     target_col = df.columns[-1]
    #     X = df.drop(columns=[target_col])
    #     y = df[target_col]
    
    #     # Fill missing numerical values with the median
    #     for col in X.select_dtypes(include=['number']).columns:
    #         X[col] = X[col].fillna(X[col].median())
    
    #     # Identify categorical features (non-numeric)
    #     categorical_cols = X.select_dtypes(include=['object']).columns
    
    #     if len(categorical_cols) > 0:
    #         print(f"Encoding categorical columns: {list(categorical_cols)}")
    #         X = DataLoader._encode_categorical(X, categorical_cols)
    
    #     # Fill any remaining NaN values in X after encoding
    #     X.fillna(0, inplace=True)
    
    #     # Convert categorical encodings to float32 (ensures compatibility with models)
    #     X = X.astype('float32')
    
    #     # Replace any infinite values (if present)
    #     X.replace([np.inf, -np.inf], np.nan, inplace=True)
    #     X.fillna(0, inplace=True)
    
    #     # === Apply Standardization Here ===
    #     scaler = StandardScaler()
    #     X_scaled = scaler.fit_transform(X)
    #     X = pd.DataFrame(X_scaled, columns=X.columns)  # Preserve column names
    
    #     return DataLoader._split_data(X, y)

    
    # @staticmethod
    # def _split_data(X, y, test_size=0.2, random_state=42):
    #     from sklearn.model_selection import train_test_split
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    #     return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
    
    # @staticmethod
    # def _split_data(X, y, test_size=0.2, random_state=42):
    #     """
    #     Split data into training and testing sets.
    #     """
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    #     y_train = y_train.fillna(y_train.median())
    #     y_test = y_test.fillna(y_test.median())

        
    #     return {'X_train': X_train.to_numpy(), 'y_train': y_train.to_numpy(),
    #             'X_test': X_test.to_numpy(), 'y_test': y_test.to_numpy()}


    @staticmethod
    def _split_data(X, y, test_size=0.2, random_state=42):
        # Perform the split (train_test_split works with NumPy arrays)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Convert y_train and y_test to pandas Series if they aren't already,
        # so that we can use the fillna method.
        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train)
        if not isinstance(y_test, pd.Series):
            y_test = pd.Series(y_test)
        
        # Now fill missing values using median
        y_train = y_train.fillna(y_train.median())
        y_test = y_test.fillna(y_test.median())
        
        # If your downstream code expects numpy arrays, you can convert back:
        return {
            'X_train': X_train if isinstance(X_train, np.ndarray) else X_train.to_numpy(),
            'y_train': y_train.to_numpy(),
            'X_test': X_test if isinstance(X_test, np.ndarray) else X_test.to_numpy(),
            'y_test': y_test.to_numpy()
        }
