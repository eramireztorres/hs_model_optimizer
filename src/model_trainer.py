from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class ModelTrainer:
    def __init__(self):
        """
        Initialize the ModelTrainer class.
        """
        pass

    def train_model(self, model, X_train, y_train):
        """
        Train the given model on the training data.

        Args:
            model (object): The model to be trained.
            X_train (numpy array): The training features.
            y_train (numpy array): The training labels.

        Returns:
            object: The trained model.
        """
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the given model on the test data and return classification metrics.

        Args:
            model (object): The trained model to be evaluated.
            X_test (numpy array): The test features.
            y_test (numpy array): The test labels.

        Returns:
            dict: A dictionary containing accuracy, precision, recall, and F1 score.
        """
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        return metrics

    def save_model(self, model, filepath):
        """
        Save the trained model to a joblib file.

        Args:
            model (object): The trained model to save.
            filepath (str): The path to save the model.

        Returns:
            None
        """
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")


