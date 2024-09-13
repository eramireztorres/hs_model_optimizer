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
        """
        if self.model is None:
            raise ValueError("Model has not been trained.")
        predictions = self.model.predict(self.X_test)
        metrics = {
            "accuracy": accuracy_score(self.y_test, predictions),
            "precision": precision_score(self.y_test, predictions, average='weighted'),
            "recall": recall_score(self.y_test, predictions, average='weighted'),
            "f1_score": f1_score(self.y_test, predictions, average='weighted')
        }
        return metrics

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

