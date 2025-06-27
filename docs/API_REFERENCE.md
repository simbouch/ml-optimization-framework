# API Reference

Complete API documentation for the ML Optimization Framework.

## 📋 Table of Contents

- [OptimizationConfig](#optimizationconfig)
- [ModelOptimizer](#modeloptimizer)
- [StudyManager](#studymanager)
- [Optimizers](#optimizers)
- [Utilities](#utilities)

## OptimizationConfig

Central configuration class for optimization settings.

### Class Definition

```python
class OptimizationConfig:
    """Configuration class for ML optimization framework."""
```

### Constructor

```python
def __init__(
    self,
    study_name: str = "ml_optimization_study",
    n_trials: int = 100,
    sampler_name: str = "TPE",
    pruner_name: str = "Median",
    directions: Optional[List[str]] = None,
    cv_folds: int = 5,
    test_size: float = 0.2,
    random_seed: int = 42,
    data_dir: Path = Path("data"),
    results_dir: Path = Path("results"),
    studies_dir: Path = Path("studies"),
    logs_dir: Path = Path("logs"),
    log_level: str = "INFO",
    log_format: str = "{time} | {level} | {message}"
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `study_name` | str | "ml_optimization_study" | Name of the optimization study |
| `n_trials` | int | 100 | Number of optimization trials |
| `sampler_name` | str | "TPE" | Sampling algorithm (TPE, Random, CMA-ES, Grid, QMC) |
| `pruner_name` | str | "Median" | Pruning algorithm (Median, SuccessiveHalving, Hyperband) |
| `directions` | List[str] | None | Optimization directions for multi-objective |
| `cv_folds` | int | 5 | Number of cross-validation folds |
| `test_size` | float | 0.2 | Test set size ratio |
| `random_seed` | int | 42 | Random seed for reproducibility |
| `data_dir` | Path | "data" | Directory for data files |
| `results_dir` | Path | "results" | Directory for results |
| `studies_dir` | Path | "studies" | Directory for study databases |
| `logs_dir` | Path | "logs" | Directory for log files |
| `log_level` | str | "INFO" | Logging level |
| `log_format` | str | "{time} \| {level} \| {message}" | Log message format |

### Methods

#### `create_directories()`
Creates all necessary directories for the optimization framework.

```python
def create_directories(self) -> None:
    """Create all necessary directories."""
```

#### `get_sampler()`
Returns the configured Optuna sampler.

```python
def get_sampler(self) -> optuna.samplers.BaseSampler:
    """Get the configured Optuna sampler."""
```

#### `get_pruner()`
Returns the configured Optuna pruner.

```python
def get_pruner(self) -> optuna.pruners.BasePruner:
    """Get the configured Optuna pruner."""
```

### Example Usage

```python
from src.config import OptimizationConfig

# Basic configuration
config = OptimizationConfig(
    study_name="my_optimization",
    n_trials=50,
    sampler_name="CMA-ES"
)

# Multi-objective configuration
config = OptimizationConfig(
    study_name="multi_objective",
    directions=["maximize", "minimize"],
    n_trials=100
)
```

## ModelOptimizer

Abstract base class for ML model optimizers.

### Class Definition

```python
class ModelOptimizer(ABC):
    """Abstract base class for ML model optimizers."""
```

### Constructor

```python
def __init__(
    self,
    config: OptimizationConfig,
    task_type: str = "classification"
):
    """Initialize the optimizer."""
```

### Abstract Methods

#### `define_search_space(trial)`
Define the hyperparameter search space.

```python
@abstractmethod
def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
    """Define the hyperparameter search space."""
```

#### `create_model(params)`
Create a model with given parameters.

```python
@abstractmethod
def create_model(self, params: Dict[str, Any]) -> Any:
    """Create a model with the given parameters."""
```

### Concrete Methods

#### `optimize(X, y)`
Run the optimization process.

```python
def optimize(
    self,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series]
) -> optuna.Study:
    """Run the optimization process."""
```

#### `objective(trial, X, y)`
Objective function for optimization.

```python
def objective(
    self,
    trial: optuna.Trial,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series]
) -> float:
    """Objective function for optimization."""
```

## StudyManager

Manages Optuna studies with persistence and analysis capabilities.

### Class Definition

```python
class StudyManager:
    """Manages Optuna studies with persistence and analysis."""
```

### Constructor

```python
def __init__(self, config: OptimizationConfig):
    """Initialize the StudyManager."""
```

### Methods

#### `create_study()`
Create a new Optuna study.

```python
def create_study(
    self,
    study_name: str,
    direction: str = "maximize",
    directions: Optional[List[str]] = None,
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    pruner: Optional[optuna.pruners.BasePruner] = None
) -> optuna.Study:
    """Create a new Optuna study."""
```

#### `load_study()`
Load an existing study.

```python
def load_study(self, study_name: str) -> optuna.Study:
    """Load an existing study."""
```

#### `get_study_summary()`
Get summary information about a study.

```python
def get_study_summary(self, study_name: str) -> Dict[str, Any]:
    """Get summary information about a study."""
```

#### `export_study_results()`
Export study results to various formats.

```python
def export_study_results(
    self,
    study_name: str,
    format: str = "csv",
    include_params: bool = True,
    include_user_attrs: bool = True
) -> Union[pd.DataFrame, str]:
    """Export study results to various formats."""
```

#### `get_all_studies_summary()`
Get summary of all studies.

```python
def get_all_studies_summary(self) -> List[Dict[str, Any]]:
    """Get summary of all studies."""
```

### Example Usage

```python
from src.study_manager import StudyManager

manager = StudyManager(config)

# Create study
study = manager.create_study("my_study", direction="maximize")

# Get summary
summary = manager.get_study_summary("my_study")

# Export results
df = manager.export_study_results("my_study", format="csv")
```

## Optimizers

Concrete implementations of ModelOptimizer for different ML algorithms.

### RandomForestOptimizer

Optimizes Random Forest hyperparameters.

```python
class RandomForestOptimizer(ModelOptimizer):
    """Random Forest hyperparameter optimizer."""
```

#### Search Space

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `n_estimators` | int | [10, 200] | Number of trees |
| `max_depth` | int | [3, 20] | Maximum tree depth |
| `min_samples_split` | int | [2, 20] | Minimum samples to split |
| `min_samples_leaf` | int | [1, 10] | Minimum samples in leaf |
| `max_features` | categorical | ['sqrt', 'log2', None] | Feature selection strategy |
| `bootstrap` | categorical | [True, False] | Bootstrap sampling |

### XGBoostOptimizer

Optimizes XGBoost hyperparameters.

```python
class XGBoostOptimizer(ModelOptimizer):
    """XGBoost hyperparameter optimizer."""
```

#### Search Space

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `n_estimators` | int | [50, 300] | Number of boosting rounds |
| `max_depth` | int | [3, 10] | Maximum tree depth |
| `learning_rate` | float | [0.01, 0.3] | Learning rate |
| `subsample` | float | [0.6, 1.0] | Subsample ratio |
| `colsample_bytree` | float | [0.6, 1.0] | Feature subsample ratio |
| `reg_alpha` | float | [0, 10] | L1 regularization |
| `reg_lambda` | float | [0, 10] | L2 regularization |

### SVMOptimizer

Optimizes Support Vector Machine hyperparameters.

```python
class SVMOptimizer(ModelOptimizer):
    """SVM hyperparameter optimizer."""
```

#### Search Space

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `C` | float | [0.1, 100] | Regularization parameter |
| `kernel` | categorical | ['linear', 'rbf', 'poly'] | Kernel type |
| `gamma` | categorical | ['scale', 'auto'] + [0.001, 0.01, 0.1, 1] | Kernel coefficient |
| `degree` | int | [2, 5] | Polynomial degree (poly kernel) |

### Example Usage

```python
from src.optimizers import RandomForestOptimizer, XGBoostOptimizer

# Random Forest optimization
rf_optimizer = RandomForestOptimizer(config, task_type="classification")
rf_study = rf_optimizer.optimize(X_train, y_train)

# XGBoost optimization
xgb_optimizer = XGBoostOptimizer(config, task_type="regression")
xgb_study = xgb_optimizer.optimize(X_train, y_train)

# Get best parameters
best_rf_params = rf_study.best_params
best_xgb_params = xgb_study.best_params
```

## Utilities

Helper functions and utilities.

### Data Loading

```python
def load_sample_data(dataset_name: str = "iris") -> Tuple[np.ndarray, np.ndarray]:
    """Load sample datasets for testing."""
```

### Visualization

```python
def plot_optimization_history(study: optuna.Study) -> None:
    """Plot optimization history."""

def plot_parameter_importance(study: optuna.Study) -> None:
    """Plot parameter importance."""
```

### Validation

```python
def validate_config(config: OptimizationConfig) -> bool:
    """Validate configuration settings."""

def validate_data(X: np.ndarray, y: np.ndarray) -> bool:
    """Validate input data."""
```

## Error Handling

### Custom Exceptions

```python
class OptimizationError(Exception):
    """Base exception for optimization errors."""

class ConfigurationError(OptimizationError):
    """Raised when configuration is invalid."""

class DataValidationError(OptimizationError):
    """Raised when data validation fails."""
```

### Example Error Handling

```python
try:
    optimizer = RandomForestOptimizer(config)
    study = optimizer.optimize(X, y)
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
except DataValidationError as e:
    logger.error(f"Data validation error: {e}")
except OptimizationError as e:
    logger.error(f"Optimization error: {e}")
```

## Type Hints

The framework uses comprehensive type hints for better IDE support and code clarity:

```python
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import optuna

# Example type annotations
def optimize_model(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    config: OptimizationConfig
) -> Tuple[optuna.Study, Dict[str, Any]]:
    """Optimize model with type hints."""
```

For more examples and detailed usage, see the [Getting Started Guide](GETTING_STARTED.md) and [Advanced Usage Guide](ADVANCED_USAGE.md).
