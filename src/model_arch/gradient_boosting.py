from sklearn.ensemble import GradientBoostingClassifier

def get_model(**kwargs):
    """
    Returns a configured Gradient Boosting model (sklearn).
    Accepts kwargs to override default parameters.
    """
    # Default parameters for a baseline
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'subsample': 0.8,
        'random_state': 42,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10  # Early stopping
    }
    
    # Update with any provided kwargs
    params.update(kwargs)
    
    return GradientBoostingClassifier(**params)
