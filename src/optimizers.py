#!/usr/bin/env python3
"""
Model optimizers for ML Optimization Framework
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler, GridSampler, QMCSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import numpy as np
# pandas imported when needed
from loguru import logger

from .config import OptimizationConfig, ModelConfig


class ModelOptimizer(ABC):
    """
    Abstract base class for ML model optimizers.
    
    This class provides a common interface for optimizing different ML models
    using Optuna with various samplers, pruners, and optimization strategies.
    """
    
    def __init__(self, config: OptimizationConfig, model_config: ModelConfig):
        """
        Initialize the optimizer with configuration.
        
        Args:
            config: Optimization configuration
            model_config: Model-specific configuration
        """
        self.config = config
        self.model_config = model_config
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        
        # Set up logging (with Windows-compatible settings)
        try:
            logger.add(
                self.config.logs_dir / f"{self.config.study_name}.log",
                format=self.config.log_format,
                level=self.config.log_level,
                rotation="10 MB",
                enqueue=True  # Helps with Windows file locking issues
            )
        except Exception:
            # Fallback to console logging if file logging fails
            pass
    
    def create_study(self) -> optuna.Study:
        """Create and configure Optuna study."""
        sampler = self._create_sampler()
        pruner = self._create_pruner()
        
        study = optuna.create_study(
            study_name=self.config.study_name,
            storage=self.config.storage_url,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=self.config.load_if_exists
        )
        
        logger.info(f"Created study: {self.config.study_name}")
        logger.info(f"Sampler: {self.config.sampler_name}")
        logger.info(f"Pruner: {self.config.pruner_name}")
        
        return study
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create sampler based on configuration."""
        samplers = {
            "TPE": TPESampler,
            "Random": RandomSampler,
            "CmaEs": CmaEsSampler,
            "Grid": GridSampler,
            "QMC": QMCSampler,
        }
        
        sampler_class = samplers[self.config.sampler_name]
        
        # Add seed for reproducibility
        params = {"seed": self.config.random_seed}
        params.update(self.config.sampler_params)
        
        return sampler_class(**params)
    
    def _create_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """Create pruner based on configuration."""
        if self.config.pruner_name == "None":
            return None
        
        pruners = {
            "Median": MedianPruner,
            "SuccessiveHalving": SuccessiveHalvingPruner,
            "Hyperband": HyperbandPruner,
        }
        
        pruner_class = pruners[self.config.pruner_name]
        return pruner_class(**self.config.pruner_params)
    
    @abstractmethod
    def objective(self, trial: optuna.Trial) -> Union[float, List[float]]:
        """
        Objective function to optimize.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Score(s) to optimize
        """
        pass
    
    @abstractmethod
    def suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for the trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        pass
    
    def optimize(self, X: np.ndarray, y: np.ndarray) -> optuna.Study:
        """
        Run optimization process.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Completed Optuna study
        """
        self.X = X
        self.y = y
        
        self.study = self.create_study()
        
        logger.info(f"Starting optimization with {self.config.n_trials} trials")
        
        self.study.optimize(
            self.objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs
        )
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"Optimization completed!")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.study
    
    def get_best_model(self):
        """Get the best model with optimized parameters."""
        if self.best_params is None:
            raise ValueError("No optimization has been performed yet")
        
        return self._create_model(self.best_params)
    
    @abstractmethod
    def _create_model(self, params: Dict[str, Any]):
        """Create model instance with given parameters."""
        pass


class RandomForestOptimizer(ModelOptimizer):
    """Random Forest optimizer implementation."""
    
    def __init__(self, config: OptimizationConfig, task_type: str = "classification"):
        """
        Initialize Random Forest optimizer.
        
        Args:
            config: Optimization configuration
            task_type: "classification" or "regression"
        """
        model_config = ModelConfig(model_type="RandomForest")
        super().__init__(config, model_config)
        self.task_type = task_type
    
    def suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest Random Forest hyperparameters."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 10, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": self.config.random_seed,
        }
    
    def objective(self, trial: optuna.Trial) -> float:
        """Random Forest objective function."""
        params = self.suggest_parameters(trial)
        model = self._create_model(params)
        
        # Use stratified k-fold for classification
        if self.task_type == "classification":
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_seed)
            scoring = "accuracy"
        else:
            cv = self.config.cv_folds
            scoring = "neg_mean_squared_error"
        
        scores = cross_val_score(model, self.X, self.y, cv=cv, scoring=scoring, n_jobs=self.config.n_jobs)
        
        return scores.mean()
    
    def _create_model(self, params: Dict[str, Any]):
        """Create Random Forest model."""
        if self.task_type == "classification":
            return RandomForestClassifier(**params)
        else:
            return RandomForestRegressor(**params)


class GradientBoostingOptimizer(ModelOptimizer):
    """Gradient Boosting optimizer implementation."""

    def __init__(self, config: OptimizationConfig, task_type: str = "classification"):
        """
        Initialize Gradient Boosting optimizer.

        Args:
            config: Optimization configuration
            task_type: "classification" or "regression"
        """
        model_config = ModelConfig(model_type="GradientBoosting")
        super().__init__(config, model_config)
        self.task_type = task_type

    def suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest Gradient Boosting hyperparameters."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": self.config.random_seed,
        }

    def objective(self, trial: optuna.Trial) -> float:
        """Gradient Boosting objective function."""
        params = self.suggest_parameters(trial)
        model = self._create_model(params)

        if self.task_type == "classification":
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_seed)
            scoring = "accuracy"
        else:
            cv = self.config.cv_folds
            scoring = "neg_mean_squared_error"

        scores = cross_val_score(model, self.X, self.y, cv=cv, scoring=scoring, n_jobs=self.config.n_jobs)

        return scores.mean()

    def _create_model(self, params: Dict[str, Any]):
        """Create Gradient Boosting model."""
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        if self.task_type == "classification":
            return GradientBoostingClassifier(**params)
        else:
            return GradientBoostingRegressor(**params)


class SVMOptimizer(ModelOptimizer):
    """SVM optimizer implementation."""

    def __init__(self, config: OptimizationConfig, task_type: str = "classification"):
        """
        Initialize SVM optimizer.

        Args:
            config: Optimization configuration
            task_type: "classification" or "regression"
        """
        model_config = ModelConfig(model_type="SVM")
        super().__init__(config, model_config)
        self.task_type = task_type

    def suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest SVM hyperparameters."""
        return {
            "C": trial.suggest_float("C", 0.1, 100.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
            "random_state": self.config.random_seed,
        }

    def objective(self, trial: optuna.Trial) -> float:
        """SVM objective function."""
        params = self.suggest_parameters(trial)
        model = self._create_model(params)

        if self.task_type == "classification":
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_seed)
            scoring = "accuracy"
        else:
            cv = self.config.cv_folds
            scoring = "neg_mean_squared_error"

        scores = cross_val_score(model, self.X, self.y, cv=cv, scoring=scoring, n_jobs=self.config.n_jobs)

        return scores.mean()

    def _create_model(self, params: Dict[str, Any]):
        """Create SVM model."""
        if self.task_type == "classification":
            return SVC(**params)
        else:
            return SVR(**params)
