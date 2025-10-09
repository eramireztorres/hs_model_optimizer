import sklearn
from sklearn.model_selection import train_test_split
import joblib
from inspect import signature
from metrics_calculator import MetricsCalculator, ClassificationMetricsCalculator

sklearn.set_config(enable_metadata_routing=True)


class ModelTrainer:
    def __init__(self, model=None, X_train=None, y_train=None, X_test=None, y_test=None,
                 metrics_calculator: MetricsCalculator = None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.metrics_calculator = metrics_calculator or ClassificationMetricsCalculator()


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

        # 2. Recursively disable verbosity on the model and any nested estimators or pipeline steps
        from verbosity_suppressor import VerbositySuppressor

        VerbositySuppressor.suppress(self.model)

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
        """
        if self.model is None:
            raise ValueError("Model has not been trained.")

        predictions = self.model.predict(self.X_test)
        return self.metrics_calculator.calculate(self.y_test, predictions)

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
    def __init__(self, model=None, X_train=None, y_train=None, X_test=None, y_test=None,
                 metrics_calculator: MetricsCalculator = None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Import here to avoid circular dependency
        from metrics_calculator import RegressionMetricsCalculator
        self.metrics_calculator = metrics_calculator or RegressionMetricsCalculator()



    def train_model(self):
        """
        Train the model on the provided training data, with optional validation data for early stopping if supported.
        """
        if self.model is None:
            raise ValueError("No model provided for training.")

        X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)

        if hasattr(self.model, 'fit'):
            fit_params = {}
            # Use signature for robustness
            sig = signature(self.model.fit)
            if 'eval_set' in sig.parameters:
                fit_params['eval_set'] = [(X_val, y_val)]

            try:
                self.model.fit(X_train, y_train, **fit_params)
            except Exception as e:
                if "Metric Huber requires delta as parameter" in str(e) or "unexpected argument(s) {'delta'}" in str(e):
                    from sklearn.ensemble import VotingRegressor
                    if isinstance(self.model, VotingRegressor):
                        # Find the catboost estimator and add delta for it
                        catboost_name = None
                        for name, est in self.model.estimators:
                            if 'CatBoostRegressor' in str(type(est)):
                                catboost_name = name
                                break
                        if catboost_name:
                            fit_params[f'{catboost_name}__delta'] = 1.0
                            self.model.fit(X_train, y_train, **fit_params)
                        else:
                            raise e
                    else:
                        fit_params['delta'] = 1.0
                        self.model.fit(X_train, y_train, **fit_params)
                else:
                    raise e
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
        return self.metrics_calculator.calculate(self.y_test, predictions)

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

