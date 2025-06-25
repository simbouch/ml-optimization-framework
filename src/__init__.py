"""
ML Optimization Framework with Optuna

A comprehensive, production-ready machine learning optimization framework
that showcases the full power of Optuna for hyperparameter tuning.

This framework serves as the definitive template for implementing
professional ML optimization pipelines with advanced features including:

- Multi-model optimization (RandomForest, XGBoost, LightGBM)
- Advanced Optuna features (multi-objective, pruning, custom samplers)
- Professional data pipeline with automated preprocessing
- Comprehensive visualization and analysis tools
- Production-ready architecture with proper error handling
- CLI interface and Jupyter notebook integration

Example:
    Basic usage:

    >>> from src.data.data_pipeline import DataPipeline
    >>> from src.models.random_forest_optimizer import RandomForestOptimizer
    >>>
    >>> # Setup data pipeline
    >>> pipeline = DataPipeline(random_state=42)
    >>> pipeline.prepare_data()
    >>> X_train, X_val, y_train, y_val = pipeline.get_train_val_data()
    >>>
    >>> # Run optimization
    >>> optimizer = RandomForestOptimizer(random_state=42)
    >>> study = optimizer.optimize(X_train, X_val, y_train, y_val, n_trials=100)
    >>>
    >>> # Evaluate results
    >>> X_test, y_test = pipeline.get_test_data()
    >>> metrics = optimizer.evaluate(X_test, y_test)
    >>> print(f"Test accuracy: {metrics['accuracy']:.4f}")

Author: ML Optimization Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "ML Optimization Team"
__email__ = "team@mloptimization.com"
__license__ = "MIT"

# Import main classes for easy access
try:
    from .data.data_pipeline import DataPipeline
    from .models.random_forest_optimizer import RandomForestOptimizer
    from .models.xgboost_optimizer import XGBoostOptimizer
    from .models.lightgbm_optimizer import LightGBMOptimizer
    from .optimization.config import OptimizationConfig
    from .optimization.study_manager import StudyManager
    from .visualization.plots import OptimizationPlotter

    # Import advanced features
    from .optimization.advanced_features import (
        MultiObjectiveOptimizer,
        SamplerComparison,
        PrunerComparison
    )

    _IMPORTS_AVAILABLE = True

except ImportError as e:
    # Graceful handling of import errors during setup
    _IMPORTS_AVAILABLE = False
    import warnings
    warnings.warn(f"Some imports failed: {e}. This is normal during installation.")

# Module imports for backward compatibility
from . import data
from . import models
from . import optimization
from . import visualization

# Define public API
if _IMPORTS_AVAILABLE:
    __all__ = [
        # Core classes
        "DataPipeline",
        "RandomForestOptimizer",
        "XGBoostOptimizer",
        "LightGBMOptimizer",
        "OptimizationConfig",
        "StudyManager",
        "OptimizationPlotter",

        # Advanced features
        "MultiObjectiveOptimizer",
        "SamplerComparison",
        "PrunerComparison",

        # Modules
        "data",
        "models",
        "optimization",
        "visualization",

        # Metadata
        "__version__",
        "__author__",
        "__email__",
        "__license__"
    ]
else:
    __all__ = ["data", "models", "optimization", "visualization"]
