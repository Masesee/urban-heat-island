from xgboost import XGBClassifier

def get_model(**kwargs):
    """
    Returns a configured XGBoost model.
    Accepts kwargs to override default parameters.
    """
    # Default parameters for a baseline
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multi:softprob', # Assuming multiclass, will adapt if binary
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Update with any provided kwargs
    params.update(kwargs)
    
    # XGBClassifier handles objective automatically mostly, but good to be explicit if known.
    # If the target is binary, XGBoost switches automatically.
    
    return XGBClassifier(**params)
