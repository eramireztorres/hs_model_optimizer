def load_model():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)


