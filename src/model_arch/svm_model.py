from sklearn.svm import SVC

def get_model(**kwargs):
    """
    Returns a configured Support Vector Machine (SVM) model.
    Accepts kwargs to override default parameters.
    """
    # Default parameters for a baseline
    # Note: SVMs are sensitive to scaling. Ensure data is scaled before passing (train.py does this).
    params = {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'probability': True, # Required for some metrics and soft voting
        'random_state': 42,
        'class_weight': 'balanced'
    }
    
    # Update with any provided kwargs
    params.update(kwargs)
    
    return SVC(**params)
