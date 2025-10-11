import sys, os

# ensure 'src/' is on PYTHONPATH for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from verbosity_suppressor import VerbositySuppressor


def test_single_sklearn_verbose_off():
    from sklearn.ensemble import GradientBoostingClassifier
    gbc = GradientBoostingClassifier(verbose=1)
    VerbositySuppressor.suppress(gbc)
    assert gbc.get_params()['verbose'] is False


def test_xgboost_suppressed():
    xgb = XGBClassifier(verbosity=1)
    VerbositySuppressor.suppress(xgb)
    assert xgb.get_params()['verbosity'] == 0


def test_lightgbm_suppressed():
    lgb = LGBMClassifier(verbosity=1)
    VerbositySuppressor.suppress(lgb)
    assert lgb.get_params()['verbosity'] == -1


def test_stacking_with_boosters():
    base = [('x', XGBClassifier(verbosity=1)), ('l', LGBMClassifier(verbosity=1))]
    stack = StackingClassifier(estimators=base,
                               final_estimator=LGBMClassifier(verbosity=1))
    VerbositySuppressor.suppress(stack)
    for _, est in stack.estimators:
        assert est.get_params()['verbosity'] in (-1, 0)
    assert stack.final_estimator.get_params()['verbosity'] in (-1, 0)


def test_pipeline_with_boosters_and_dt():
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('xgb', XGBClassifier(verbosity=1))
    ])
    VerbositySuppressor.suppress(pipe)
    xgb_step = pipe.named_steps['xgb']
    assert xgb_step.get_params()['verbosity'] == 0
