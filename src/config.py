#!/usr/bin/env python3
"""
Configuration management for ML Optimization Framework
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import os


@dataclass
class OptimizationConfig:
    """
    Comprehensive configuration class for ML optimization studies.
    
    This class centralizes all configuration parameters for Optuna studies,
    providing a clean interface for different optimization scenarios.
    """
    
    # Study Configuration
    study_name: str = "ml_optimization_study"
    direction: Union[str, List[str]] = "maximize"  # "maximize", "minimize", or list for multi-objective
    storage_url: Optional[str] = None
    load_if_exists: bool = True
    
    # Optimization Parameters
    n_trials: int = 100
    timeout: Optional[float] = None
    n_jobs: int = 1
    
    # Sampler Configuration
    sampler_name: str = "TPE"  # TPE, Random, CmaEs, Grid, QMC
    sampler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Pruner Configuration
    pruner_name: str = "Median"  # Median, SuccessiveHalving, Hyperband, None
    pruner_params: Dict[str, Any] = field(default_factory=dict)
    
    # Data Configuration
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    results_dir: Path = field(default_factory=lambda: Path("./results"))
    studies_dir: Path = field(default_factory=lambda: Path("./studies"))
    logs_dir: Path = field(default_factory=lambda: Path("./logs"))
    
    # ML Model Configuration
    random_seed: int = 42
    cv_folds: int = 5
    test_size: float = 0.2
    validation_size: float = 0.2
    
    # Performance Configuration
    memory_limit_mb: int = 4096
    trial_timeout: float = 3600.0
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    
    def __post_init__(self):
        """Post-initialization to set up directories and validate configuration."""
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.results_dir, self.studies_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set storage URL if not provided
        if self.storage_url is None:
            self.storage_url = f"sqlite:///{self.studies_dir}/{self.study_name}.db"
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        valid_directions = ["maximize", "minimize"]
        if isinstance(self.direction, str):
            if self.direction not in valid_directions:
                raise ValueError(f"Direction must be one of {valid_directions}")
        elif isinstance(self.direction, list):
            for direction in self.direction:
                if direction not in valid_directions:
                    raise ValueError(f"All directions must be one of {valid_directions}")
        
        valid_samplers = ["TPE", "Random", "CmaEs", "Grid", "QMC"]
        if self.sampler_name not in valid_samplers:
            raise ValueError(f"Sampler must be one of {valid_samplers}")
        
        valid_pruners = ["Median", "SuccessiveHalving", "Hyperband", "None"]
        if self.pruner_name not in valid_pruners:
            raise ValueError(f"Pruner must be one of {valid_pruners}")
        
        if self.n_trials <= 0:
            raise ValueError("Number of trials must be positive")
        
        if self.cv_folds <= 1:
            raise ValueError("CV folds must be greater than 1")
        
        if not 0 < self.test_size < 1:
            raise ValueError("Test size must be between 0 and 1")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OptimizationConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_env(cls) -> "OptimizationConfig":
        """Create configuration from environment variables."""
        config_dict = {}
        
        # Map environment variables to config fields
        env_mapping = {
            "STUDY_NAME": "study_name",
            "DIRECTION": "direction",
            "N_TRIALS": ("n_trials", int),
            "SAMPLER_NAME": "sampler_name",
            "PRUNER_NAME": "pruner_name",
            "RANDOM_SEED": ("random_seed", int),
            "CV_FOLDS": ("cv_folds", int),
            "TEST_SIZE": ("test_size", float),
            "LOG_LEVEL": "log_level",
        }
        
        for env_var, field_info in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                if isinstance(field_info, tuple):
                    field_name, field_type = field_info
                    config_dict[field_name] = field_type(env_value)
                else:
                    config_dict[field_info] = env_value
        
        return cls(**config_dict)


@dataclass
class ModelConfig:
    """Configuration for specific ML models."""
    
    model_type: str = "RandomForest"
    model_params: Dict[str, Any] = field(default_factory=dict)
    param_space: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default parameter spaces for different models."""
        if not self.param_space:
            self.param_space = self._get_default_param_space()
    
    def _get_default_param_space(self) -> Dict[str, Any]:
        """Get default parameter space for the model type."""
        spaces = {
            "RandomForest": {
                "n_estimators": ("int", 10, 200),
                "max_depth": ("int", 3, 20),
                "min_samples_split": ("int", 2, 20),
                "min_samples_leaf": ("int", 1, 10),
                "max_features": ("categorical", ["sqrt", "log2", None]),
            },
            "XGBoost": {
                "n_estimators": ("int", 50, 300),
                "max_depth": ("int", 3, 10),
                "learning_rate": ("float", 0.01, 0.3),
                "subsample": ("float", 0.6, 1.0),
                "colsample_bytree": ("float", 0.6, 1.0),
                "reg_alpha": ("float", 0.0, 1.0),
                "reg_lambda": ("float", 0.0, 1.0),
            },
            "LightGBM": {
                "n_estimators": ("int", 50, 300),
                "max_depth": ("int", 3, 15),
                "learning_rate": ("float", 0.01, 0.3),
                "num_leaves": ("int", 10, 300),
                "subsample": ("float", 0.6, 1.0),
                "colsample_bytree": ("float", 0.6, 1.0),
                "reg_alpha": ("float", 0.0, 1.0),
                "reg_lambda": ("float", 0.0, 1.0),
            },
            "SVM": {
                "C": ("float", 0.1, 100.0),
                "gamma": ("categorical", ["scale", "auto"]),
                "kernel": ("categorical", ["rbf", "poly", "sigmoid"]),
            },
        }
        
        return spaces.get(self.model_type, {})
