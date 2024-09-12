def load_model():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=400, max_depth=20, min_samples_split=2, min_samples_leaf=1)