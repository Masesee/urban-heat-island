from sklearn.ensemble import RandomForestClassifier

def get_model(**kwargs):
    """
    Returns a configured Random Forest model.
    Accepts kwargs to override default parameters.
    """
    # Default parameters
    params = {
        'n_estimators': 100,
        'random_state': 42,
        'class_weight': 'balanced',
        'n_jobs': -1
    }
    
    # Update with any provided kwargs
    params.update(kwargs)
    
    return RandomForestClassifier(**params)
