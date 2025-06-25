"""
Optimization configuration management for hyperparameter search spaces.

This module provides configuration classes for managing hyperparameter
search spaces and optimization settings across different models.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
import yaml
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler, GridSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner

logger = logging.getLogger(__name__)


@dataclass
class SamplerConfig:
    """Configuration for Optuna samplers."""
    
    name: str = "tpe"
    params: Dict[str, Any] = field(default_factory=dict)
    
    def create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create and return the configured sampler."""
        samplers = {
            "tpe": TPESampler,
            "cmaes": CmaEsSampler,
            "random": RandomSampler,
            "grid": GridSampler
        }
        
        if self.name not in samplers:
            raise ValueError(f"Unknown sampler: {self.name}")
        
        return samplers[self.name](**self.params)


@dataclass
class PrunerConfig:
    """Configuration for Optuna pruners."""
    
    name: str = "median"
    params: Dict[str, Any] = field(default_factory=dict)
    
    def create_pruner(self) -> optuna.pruners.BasePruner:
        """Create and return the configured pruner."""
        pruners = {
            "median": MedianPruner,
            "successive_halving": SuccessiveHalvingPruner
        }
        
        if self.name not in pruners:
            raise ValueError(f"Unknown pruner: {self.name}")
        
        return pruners[self.name](**self.params)


@dataclass
class StudyConfig:
    """Configuration for Optuna studies."""
    
    study_name: str = "optimization_study"
    direction: str = "maximize"
    directions: Optional[List[str]] = None  # For multi-objective
    storage: Optional[str] = None
    load_if_exists: bool = True
    sampler_config: SamplerConfig = field(default_factory=SamplerConfig)
    pruner_config: PrunerConfig = field(default_factory=PrunerConfig)


class OptimizationConfig:
    """
    Comprehensive configuration manager for optimization settings.
    
    Manages hyperparameter search spaces, sampler/pruner configurations,
    and study settings for different models.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize optimization configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.hyperparameter_spaces: Dict[str, Dict[str, Any]] = {}
        self.study_configs: Dict[str, StudyConfig] = {}
        self.default_study_config = StudyConfig()
        
        if config_path:
            self.load_from_file(config_path)
        else:
            self._set_default_configs()
    
    def _set_default_configs(self) -> None:
        """Set default hyperparameter spaces for all models."""
        
        # Random Forest hyperparameters
        self.hyperparameter_spaces["random_forest"] = {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
            "max_depth": {"type": "int", "low": 3, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
            "bootstrap": {"type": "categorical", "choices": [True, False]},
            "class_weight": {"type": "categorical", "choices": [None, "balanced"]}
        }
        
        # XGBoost hyperparameters
        self.hyperparameter_spaces["xgboost"] = {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            "min_child_weight": {"type": "int", "low": 1, "high": 10},
            "gamma": {"type": "float", "low": 0, "high": 5},
            "booster": {"type": "categorical", "choices": ["gbtree", "gblinear"]}
        }
        
        # LightGBM hyperparameters
        self.hyperparameter_spaces["lightgbm"] = {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
            "max_depth": {"type": "int", "low": 3, "high": 15},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 10, "high": 300},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            "min_child_samples": {"type": "int", "low": 5, "high": 100},
            "min_child_weight": {"type": "float", "low": 1e-5, "high": 10.0, "log": True},
            "boosting_type": {"type": "categorical", "choices": ["gbdt", "dart", "goss"]}
        }
        
        logger.info("Default hyperparameter spaces configured")
    
    def get_hyperparameter_space(self, model_name: str) -> Dict[str, Any]:
        """
        Get hyperparameter space for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary defining the hyperparameter search space
        """
        if model_name not in self.hyperparameter_spaces:
            raise ValueError(f"No hyperparameter space defined for model: {model_name}")
        
        return self.hyperparameter_spaces[model_name]
    
    def suggest_hyperparameters(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial using the configured search space.
        
        Args:
            trial: Optuna trial object
            model_name: Name of the model
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        space = self.get_hyperparameter_space(model_name)
        params = {}
        
        for param_name, param_config in space.items():
            param_type = param_config["type"]
            
            if param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", 1)
                )
            elif param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False)
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        
        # Handle conditional parameters for XGBoost
        if model_name == "xgboost" and "booster" in params:
            if params["booster"] == "gblinear":
                # Remove tree-specific parameters for linear booster
                tree_params = ["max_depth", "gamma", "min_child_weight", "subsample", "colsample_bytree"]
                for param in tree_params:
                    if param in params:
                        del params[param]
        
        return params
    
    def add_hyperparameter_space(self, model_name: str, space: Dict[str, Any]) -> None:
        """
        Add or update hyperparameter space for a model.
        
        Args:
            model_name: Name of the model
            space: Hyperparameter space definition
        """
        self.hyperparameter_spaces[model_name] = space
        logger.info(f"Hyperparameter space added for model: {model_name}")
    
    def get_study_config(self, model_name: str) -> StudyConfig:
        """
        Get study configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            StudyConfig object
        """
        return self.study_configs.get(model_name, self.default_study_config)
    
    def set_study_config(self, model_name: str, config: StudyConfig) -> None:
        """
        Set study configuration for a specific model.
        
        Args:
            model_name: Name of the model
            config: StudyConfig object
        """
        self.study_configs[model_name] = config
        logger.info(f"Study configuration set for model: {model_name}")
    
    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        try:
            with open(config_path, 'r') as file:
                config_data = yaml.safe_load(file)
            
            # Load hyperparameter spaces
            if "hyperparameter_spaces" in config_data:
                self.hyperparameter_spaces.update(config_data["hyperparameter_spaces"])
            
            # Load study configurations
            if "study_configs" in config_data:
                for model_name, study_config_data in config_data["study_configs"].items():
                    self.study_configs[model_name] = StudyConfig(**study_config_data)
            
            # Load default study config
            if "default_study_config" in config_data:
                self.default_study_config = StudyConfig(**config_data["default_study_config"])
            
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            raise
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save YAML configuration file
        """
        try:
            config_data = {
                "hyperparameter_spaces": self.hyperparameter_spaces,
                "study_configs": {
                    name: {
                        "study_name": config.study_name,
                        "direction": config.direction,
                        "directions": config.directions,
                        "storage": config.storage,
                        "load_if_exists": config.load_if_exists,
                        "sampler_config": {
                            "name": config.sampler_config.name,
                            "params": config.sampler_config.params
                        },
                        "pruner_config": {
                            "name": config.pruner_config.name,
                            "params": config.pruner_config.params
                        }
                    }
                    for name, config in self.study_configs.items()
                },
                "default_study_config": {
                    "study_name": self.default_study_config.study_name,
                    "direction": self.default_study_config.direction,
                    "directions": self.default_study_config.directions,
                    "storage": self.default_study_config.storage,
                    "load_if_exists": self.default_study_config.load_if_exists,
                    "sampler_config": {
                        "name": self.default_study_config.sampler_config.name,
                        "params": self.default_study_config.sampler_config.params
                    },
                    "pruner_config": {
                        "name": self.default_study_config.pruner_config.name,
                        "params": self.default_study_config.pruner_config.params
                    }
                }
            }
            
            with open(config_path, 'w') as file:
                yaml.dump(config_data, file, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {str(e)}")
            raise
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models with configured hyperparameter spaces.
        
        Returns:
            List of model names
        """
        return list(self.hyperparameter_spaces.keys())
    
    def validate_config(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if configuration is valid
        """
        try:
            # Check if hyperparameter spaces are defined
            if not self.hyperparameter_spaces:
                raise ValueError("No hyperparameter spaces defined")
            
            # Validate each hyperparameter space
            for model_name, space in self.hyperparameter_spaces.items():
                for param_name, param_config in space.items():
                    if "type" not in param_config:
                        raise ValueError(f"Missing 'type' for parameter {param_name} in model {model_name}")
                    
                    param_type = param_config["type"]
                    if param_type in ["int", "float"]:
                        if "low" not in param_config or "high" not in param_config:
                            raise ValueError(f"Missing 'low' or 'high' for parameter {param_name}")
                    elif param_type == "categorical":
                        if "choices" not in param_config:
                            raise ValueError(f"Missing 'choices' for parameter {param_name}")
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False


if __name__ == "__main__":
    # Example usage
    config = OptimizationConfig()
    print("Available models:", config.get_available_models())
    print("Random Forest space:", config.get_hyperparameter_space("random_forest"))
    print("Configuration valid:", config.validate_config())
