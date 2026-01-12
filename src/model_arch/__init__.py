"""
Model Architecture Package.

This package contains the definitions for various model architectures.
New models can be added as separate modules in this directory.
Each module should define a `get_model(**kwargs)` function.
"""

import os
import pkgutil

def list_available_models():
    """
    Returns a list of available model architecture names found in this package.
    """
    package_dir = os.path.dirname(__file__)
    return [
        name
        for _, name, _ in pkgutil.iter_modules([package_dir])
    ]
