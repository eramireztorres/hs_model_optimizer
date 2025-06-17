from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# import xgboost as xgb
import joblib
from inspect import signature

class ModelTrainer:
    def __init__(self, model=None, X_train=None, y_train=None, X_test=None, y_test=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


    def train_model(self):
        """
        Train the model on the provided training data, with optional
        validation data for early stopping if supported, and disable
        any verbose outputs if possible.
        """
        if self.model is None:
            raise ValueError("No model provided for training.")

        # 1. Split out a small validation set
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train, self.y_train,
            test_size=0.2,
            random_state=42
        )
        
        # Ensure CatBoost models do not fail due to mismatched class_weights
        self._ensure_valid_class_weights(self.model, y_train)

        # 2. Try to disable verbosity at the parameter level
        for param, value in [('verbose', False), ('verbosity', -1)]:
            try:
                # will raise if the model doesn't have that param
                self.model.set_params(**{param: value})
            except (ValueError, TypeError):
                pass

        # 3. Build fit parameters dynamically based on fit signature
        fit_params = {}
        sig = signature(self.model.fit).parameters

        # add eval_set if supported
        if 'eval_set' in sig:
            fit_params['eval_set'] = [(X_val, y_val)]

        # turn off any in-fit verbosity if supported
        if 'verbose' in sig:
            fit_params['verbose'] = False

        # if using XGBoost, you might also want to turn off
        # `early_stopping_rounds` logs etc., but verbose=False usually covers it

        # 4. Fit the model
        self.model.fit(X_train, y_train, **fit_params)

    # def train_model(self):
    #     """
    #     Train the model on the provided training data, with optional validation data for early stopping if supported.
    #     """
    #     if self.model is None:
    #         raise ValueError("No model provided for training.")

    #     X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)

    #     if hasattr(self.model, 'fit'):
    #         fit_params = {}
    #         if 'eval_set' in self.model.fit.__code__.co_varnames:
    #             fit_params['eval_set'] = [(X_val, y_val)]

    #         self.model.fit(X_train, y_train, **fit_params)
    #     else:
    #         raise ValueError("The provided model does not support the fit method.") 
    
    # def train_model(self):
    #     """
    #     Train the model on the provided training data, with optional validation data for early stopping if using XGBoost.
    #     """
    #     if self.model is None:
    #         raise ValueError("No model provided for training.")
        
    #     # Optional: Split data for validation
    #     X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)
    
    #     # Check if the model is an XGBoost model
    #     if isinstance(self.model, xgb.XGBClassifier) or isinstance(self.model, xgb.XGBRegressor):
    #         self.model.fit(X_train, y_train, 
    #                        eval_set=[(X_val, y_val)])
    #     else:
    #         # For other scikit-learn models
    #         self.model.fit(X_train, y_train)



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
        

    def _ensure_valid_class_weights(self, model, y):
        """Recursively adjust CatBoost class_weights if mismatched."""
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            CatBoostClassifier = None

        if CatBoostClassifier and isinstance(model, CatBoostClassifier):
            weights = getattr(model, 'class_weights', None)
            if weights is not None:
                n_classes = len(set(y))
                if len(weights) != n_classes:
                    model.set_params(class_weights=None, auto_class_weights='Balanced')

        if hasattr(model, 'estimators') and isinstance(model.estimators, list):
            for _, est in model.estimators:
                self._ensure_valid_class_weights(est, y)
        elif hasattr(model, 'base_estimator'):
            self._ensure_valid_class_weights(model.base_estimator, y)

class RegressionModelTrainer:
    def __init__(self, model=None, X_train=None, y_train=None, X_test=None, y_test=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test



    def train_model(self):
        """
        Train the model on the provided training data, with optional validation data for early stopping if supported.
        """
        if self.model is None:
            raise ValueError("No model provided for training.")

        X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)

        if hasattr(self.model, 'fit'):
            fit_params = {}
            if 'eval_set' in self.model.fit.__code__.co_varnames:
                fit_params['eval_set'] = [(X_val, y_val)]

            self.model.fit(X_train, y_train, **fit_params)
        else:
            raise ValueError("The provided model does not support the fit method.")


    # def train_model(self):
    #     """
    #     Train the model on the provided training data, with optional validation data for early stopping if using XGBoost.
    #     """
    #     if self.model is None:
    #         raise ValueError("No model provided for training.")
        
    #     # Optional: Split data for validation
    #     X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)
    
    #     # Check if the model is an XGBoost model
    #     if isinstance(self.model, xgb.XGBClassifier) or isinstance(self.model, xgb.XGBRegressor):
    #         self.model.fit(X_train, y_train, 
    #                        eval_set=[(X_val, y_val)])
    #     else:
    #         # For other scikit-learn models
    #         self.model.fit(X_train, y_train)

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

