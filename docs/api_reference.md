# API Reference

## Overview

This document provides comprehensive API documentation for the ML Optimization Framework. The framework is designed with a modular architecture that allows easy extension and customization.

## Core Modules

### 📊 Data Pipeline (`src.data.data_pipeline`)

#### `DataPipeline`

The main class for data loading, preprocessing, and management.

```python
class DataPipeline:
    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.2,
        val_size: float = 0.2
    )
```

**Parameters:**
- `random_state` (int): Random seed for reproducibility
- `test_size` (float): Proportion of data for test set (0.0-1.0)
- `val_size` (float): Proportion of training data for validation set (0.0-1.0)

**Key Methods:**

##### `load_data() -> Tuple[pd.DataFrame, pd.Series]`
Loads the Adult Income dataset from OpenML.

**Returns:**
- `Tuple[pd.DataFrame, pd.Series]`: Features and target data

**Example:**
```python
pipeline = DataPipeline(random_state=42)
X, y = pipeline.load_data()
print(f"Dataset shape: {X.shape}")
```

##### `prepare_data() -> Dict[str, Any]`
Complete data preparation pipeline including loading, analysis, preprocessing, and splitting.

**Returns:**
- `Dict[str, Any]`: Summary of data preparation

**Example:**
```python
pipeline = DataPipeline()
summary = pipeline.prepare_data()
print(f"Prepared {summary['total_samples']} samples")
```

##### `get_train_val_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
Returns training and validation data splits.

**Returns:**
- `Tuple`: (X_train, X_val, y_train, y_val)

##### `get_test_data() -> Tuple[np.ndarray, np.ndarray]`
Returns test data split.

**Returns:**
- `Tuple`: (X_test, y_test)

---

### 🤖 Model Optimizers (`src.models`)

#### `BaseOptimizer` (Abstract Base Class)

Base class for all model optimizers providing common functionality.

```python
class BaseOptimizer(ABC):
    def __init__(
        self,
        random_state: int = 42,
        cv_folds: int = 5,
        scoring_metric: str = "accuracy",
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = True
    )
```

**Abstract Methods:**
- `create_model(trial: Trial) -> BaseEstimator`
- `get_model_name() -> str`

**Key Methods:**

##### `optimize() -> optuna.Study`
Run hyperparameter optimization.

```python
def optimize(
    self,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100,
    study: Optional[optuna.Study] = None,
    callbacks: Optional[List[Callable]] = None
) -> optuna.Study
```

**Parameters:**
- `X_train`, `X_val`: Training and validation features
- `y_train`, `y_val`: Training and validation targets
- `n_trials`: Number of optimization trials
- `study`: Existing study to continue (optional)
- `callbacks`: List of callback functions

**Returns:**
- `optuna.Study`: Completed optimization study

##### `evaluate() -> Dict[str, float]`
Evaluate the best model on test data.

```python
def evaluate(
    self,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]
```

**Returns:**
- `Dict[str, float]`: Dictionary of evaluation metrics

#### `RandomForestOptimizer`

Random Forest specific optimizer with advanced features.

```python
class RandomForestOptimizer(BaseOptimizer):
    def __init__(
        self,
        random_state: int = 42,
        cv_folds: int = 5,
        scoring_metric: str = "accuracy",
        n_jobs: int = -1,
        verbose: bool = True,
        config: Optional[OptimizationConfig] = None
    )
```

**Additional Methods:**

##### `analyze_feature_importance() -> Dict[str, Any]`
Analyze feature importance from the best model.

**Returns:**
- `Dict[str, Any]`: Feature importance analysis

##### `get_model_complexity() -> Dict[str, Any]`
Get model complexity metrics.

**Returns:**
- `Dict[str, Any]`: Complexity metrics (nodes, leaves, depth)

#### `XGBoostOptimizer`

XGBoost specific optimizer with early stopping and pruning integration.

```python
class XGBoostOptimizer(BaseOptimizer):
    def __init__(
        self,
        random_state: int = 42,
        cv_folds: int = 5,
        scoring_metric: str = "accuracy",
        early_stopping_rounds: int = 50,
        verbose: bool = True,
        config: Optional[OptimizationConfig] = None,
        use_gpu: bool = False
    )
```

**Additional Methods:**

##### `optimize_with_early_stopping() -> optuna.Study`
Run optimization with XGBoost early stopping integration.

##### `get_training_history() -> Dict[str, List[float]]`
Get training history from the best model.

#### `LightGBMOptimizer`

LightGBM specific optimizer with boosting-specific parameters.

```python
class LightGBMOptimizer(BaseOptimizer):
    def __init__(
        self,
        random_state: int = 42,
        cv_folds: int = 5,
        scoring_metric: str = "accuracy",
        early_stopping_rounds: int = 50,
        verbose: bool = True,
        config: Optional[OptimizationConfig] = None,
        use_gpu: bool = False
    )
```

**Additional Methods:**

##### `get_leaf_statistics() -> Dict[str, Any]`
Get detailed leaf statistics from the model.

---

### ⚙️ Configuration (`src.optimization.config`)

#### `OptimizationConfig`

Manages hyperparameter search spaces and optimization settings.

```python
class OptimizationConfig:
    def __init__(self, config_path: Optional[str] = None)
```

**Key Methods:**

##### `get_hyperparameter_space(model_name: str) -> Dict[str, Any]`
Get hyperparameter space for a specific model.

##### `suggest_hyperparameters(trial: Trial, model_name: str) -> Dict[str, Any]`
Suggest hyperparameters for a trial using the configured search space.

##### `validate_config() -> bool`
Validate the current configuration.

---

### 📈 Study Management (`src.optimization.study_manager`)

#### `StudyManager`

Comprehensive manager for Optuna studies with persistence and analysis.

```python
class StudyManager:
    def __init__(
        self,
        storage_url: str = "sqlite:///optuna_study.db",
        config: Optional[OptimizationConfig] = None
    )
```

**Key Methods:**

##### `create_study() -> optuna.Study`
Create or load an Optuna study.

```python
def create_study(
    self,
    study_name: str,
    model_name: Optional[str] = None,
    direction: str = "maximize",
    sampler_name: str = "tpe",
    pruner_name: str = "median",
    load_if_exists: bool = True,
    sampler_params: Optional[Dict[str, Any]] = None,
    pruner_params: Optional[Dict[str, Any]] = None
) -> optuna.Study
```

##### `create_multi_objective_study() -> optuna.Study`
Create a multi-objective optimization study.

##### `list_studies() -> List[str]`
List all available studies in storage.

##### `get_study_summary() -> Dict[str, Any]`
Get summary information for a study.

---

### 🎨 Visualization (`src.visualization.plots`)

#### `OptimizationPlotter`

Comprehensive plotter for optimization analysis and visualization.

```python
class OptimizationPlotter:
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        style: str = "seaborn"
    )
```

**Key Methods:**

##### `plot_optimization_history_custom() -> Union[plt.Figure, go.Figure]`
Plot optimization history with custom styling.

```python
def plot_optimization_history_custom(
    self,
    study: optuna.Study,
    target_name: str = "Objective Value",
    show_best: bool = True,
    interactive: bool = True
) -> Union[plt.Figure, go.Figure]
```

##### `plot_parameter_importance_custom() -> Union[plt.Figure, go.Figure]`
Plot parameter importance with custom styling.

##### `plot_multi_objective_pareto() -> Union[plt.Figure, go.Figure]`
Plot Pareto front for multi-objective optimization.

##### `create_optimization_dashboard() -> go.Figure`
Create comprehensive optimization dashboard.

---

### 🔬 Advanced Features (`src.optimization.advanced_features`)

#### `MultiObjectiveOptimizer`

Multi-objective optimization using Optuna.

```python
class MultiObjectiveOptimizer:
    def __init__(
        self,
        objectives: List[str] = ["accuracy", "training_time"],
        directions: List[str] = ["maximize", "minimize"],
        random_state: int = 42
    )
```

**Key Methods:**

##### `optimize() -> optuna.Study`
Run multi-objective optimization.

##### `get_pareto_front() -> pd.DataFrame`
Get Pareto front solutions.

##### `analyze_trade_offs() -> Dict[str, Any]`
Analyze trade-offs between objectives.

#### `SamplerComparison`

Compare different Optuna samplers on the same optimization problem.

```python
class SamplerComparison:
    def __init__(
        self,
        samplers: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    )
```

**Key Methods:**

##### `compare_samplers() -> Dict[str, Dict[str, Any]]`
Compare samplers on the given objective function.

##### `get_comparison_summary() -> pd.DataFrame`
Get summary of sampler comparison results.

---

## Usage Examples

### Basic Optimization

```python
from src.data.data_pipeline import DataPipeline
from src.models.random_forest_optimizer import RandomForestOptimizer

# Setup data
pipeline = DataPipeline(random_state=42)
pipeline.prepare_data()
X_train, X_val, y_train, y_val = pipeline.get_train_val_data()

# Run optimization
optimizer = RandomForestOptimizer(random_state=42)
study = optimizer.optimize(X_train, X_val, y_train, y_val, n_trials=100)

# Evaluate
X_test, y_test = pipeline.get_test_data()
metrics = optimizer.evaluate(X_test, y_test)
print(f"Test accuracy: {metrics['accuracy']:.4f}")
```

### Multi-Objective Optimization

```python
from src.optimization.advanced_features import MultiObjectiveOptimizer

# Setup multi-objective optimization
multi_opt = MultiObjectiveOptimizer(
    objectives=["accuracy", "training_time"],
    directions=["maximize", "minimize"]
)

# Run optimization
study = multi_opt.optimize(optimizer, X_train, X_val, y_train, y_val, n_trials=100)

# Analyze Pareto front
pareto_front = multi_opt.get_pareto_front()
trade_offs = multi_opt.analyze_trade_offs()
```

### Custom Configuration

```python
from src.optimization.config import OptimizationConfig

# Load custom configuration
config = OptimizationConfig("config/custom_hyperparameters.yaml")

# Use with optimizer
optimizer = RandomForestOptimizer(config=config)
```

## Error Handling

The framework includes comprehensive error handling:

- **Data Validation**: Automatic checks for data quality and consistency
- **Configuration Validation**: YAML configuration validation with helpful error messages
- **Study Management**: Graceful handling of study creation and loading errors
- **Optimization Errors**: Robust error handling during optimization with detailed logging

## Logging

The framework uses structured logging:

```python
import logging

# Configure logging level
logging.getLogger('src').setLevel(logging.INFO)

# Custom logger for your code
logger = logging.getLogger(__name__)
logger.info("Starting optimization...")
```

## Type Hints

All public APIs include comprehensive type hints for better IDE support and code clarity:

```python
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import optuna
```
