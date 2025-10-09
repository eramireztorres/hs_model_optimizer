"""
Utility to disable verbosity on estimators (sklearn, XGBoost, LightGBM)
and recurse into nested ensembles or pipelines.
"""

from sklearn.base import BaseEstimator
try:
    from lightgbm import LGBMModel
except ImportError:
    LGBMModel = tuple()
try:
    from xgboost import XGBModel
except ImportError:
    XGBModel = tuple()


class VerbositySuppressor:
    @staticmethod
    def suppress(est: BaseEstimator) -> None:
        """
        Disable verbosity on a single estimator and any nested estimators.

        For scikit-learn estimators, sets verbose=False.
        For XGBoost, sets verbosity=0.
        For LightGBM, sets verbosity=-1.
        Recurses into estimators/steps for ensembles and pipelines.
        """
        # scikit-learn verbosity param
        try:
            est.set_params(verbose=False)
        except (ValueError, TypeError):
            pass

        # XGBoost models
        if isinstance(est, XGBModel):
            try:
                est.set_params(verbosity=0)
            except (ValueError, TypeError):
                pass

        # LightGBM models
        if isinstance(est, LGBMModel):
            try:
                est.set_params(verbosity=-1)
            except (ValueError, TypeError):
                pass

        # Recurse into ensemble estimators
        if hasattr(est, 'estimators'):
            for item in est.estimators:
                sub = item[1] if isinstance(item, tuple) else item
                if isinstance(sub, BaseEstimator):
                    VerbositySuppressor.suppress(sub)

        # meta- or base-estimators (StackingClassifier, etc.)
        for attr in ('final_estimator', 'base_estimator', 'estimator'):
            sub = getattr(est, attr, None)
            if isinstance(sub, BaseEstimator):
                VerbositySuppressor.suppress(sub)

        # pipeline steps
        if hasattr(est, 'steps'):
            for _, step in est.steps:
                if isinstance(step, BaseEstimator):
                    VerbositySuppressor.suppress(step)