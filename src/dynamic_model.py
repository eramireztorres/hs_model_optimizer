def load_model():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.05, subsample=0.8)