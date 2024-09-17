import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class ModelTrainer:
    def __init__(self, model=None, X_train=None, y_train=None, X_test=None, y_test=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_model(self):
        """
        Train the model on the provided training data.
        """
        if self.model is None:
            raise ValueError("No model provided for training.")
        self.model.fit(self.X_train, self.y_train)


    def evaluate_model(self):
        """
        Evaluate the trained model on test data and return performance metrics.
        Converts any non-JSON serializable types (e.g., NumPy arrays) to native Python types.
        """
        if self.model is None:
            raise ValueError("Model has not been trained.")
        
        # Get predictions
        predictions = self.model.predict(self.X_test)
        
        # Convert metrics to Python-native types
        metrics = {
            "accuracy": float(accuracy_score(self.y_test, predictions)),
            "precision": list(precision_score(self.y_test, predictions, average=None)),
            "recall": list(recall_score(self.y_test, predictions, average=None)),
            "f1_score": list(f1_score(self.y_test, predictions, average=None)),
            "global_metrics": {
                "accuracy": float(accuracy_score(self.y_test, predictions)),
                "precision": float(precision_score(self.y_test, predictions, average='weighted')),
                "recall": float(recall_score(self.y_test, predictions, average='weighted')),
                "f1_score": float(f1_score(self.y_test, predictions, average='weighted'))
            }
        }

        return metrics

    # def evaluate_model(self):
    #     """
    #     Evaluate the trained model on test data and return performance metrics, including per-class metrics.
    #     """
    #     if self.model is None:
    #         raise ValueError("Model has not been trained.")
        
    #     # Get predictions
    #     predictions = self.model.predict(self.X_test)
    
    #     # Calculate metrics per class and globally
    #     metrics = {
    #         "accuracy": accuracy_score(self.y_test, predictions),
    #         "precision_per_class": precision_score(self.y_test, predictions, average=None),
    #         "recall_per_class": recall_score(self.y_test, predictions, average=None),
    #         "f1_score_per_class": f1_score(self.y_test, predictions, average=None),
    #         "overall_precision": precision_score(self.y_test, predictions, average='weighted'),
    #         "overall_recall": recall_score(self.y_test, predictions, average='weighted'),
    #         "overall_f1_score": f1_score(self.y_test, predictions, average='weighted')
    #     }
        
    #     return metrics


    # def evaluate_model(self):
    #     """
    #     Evaluate the trained model on test data and return performance metrics.
    #     """
    #     if self.model is None:
    #         raise ValueError("Model has not been trained.")
    #     predictions = self.model.predict(self.X_test)
    #     metrics = {
    #         "accuracy": accuracy_score(self.y_test, predictions),
    #         "precision": precision_score(self.y_test, predictions, average='weighted'),
    #         "recall": recall_score(self.y_test, predictions, average='weighted'),
    #         "f1_score": f1_score(self.y_test, predictions, average='weighted')
    #     }
    #     return metrics

    def save_model(self, filepath):
        """
        Save the trained model to a joblib file.
        """
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """
        Load a trained model from a joblib file.
        """
        self.model = joblib.load(filepath)
        
        

class RegressionModelTrainer:
    def __init__(self, model=None, X_train=None, y_train=None, X_test=None, y_test=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_model(self):
        """
        Train the regression model on the provided training data.
        """
        if self.model is None:
            raise ValueError("No model provided for training.")
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluate the trained regression model on test data and return regression metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been trained.")
        predictions = self.model.predict(self.X_test)
        metrics = {
            "mean_squared_error": mean_squared_error(self.y_test, predictions),
            "r2_score": r2_score(self.y_test, predictions)
        }
        return metrics

    def save_model(self, filepath):
        """
        Save the trained regression model to a joblib file.
        """
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """
        Load a trained regression model from a joblib file.
        """
        self.model = joblib.load(filepath)

