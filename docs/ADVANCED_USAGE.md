# Advanced Usage Guide

This guide covers advanced features and techniques for the ML Optimization Framework.

## 🎯 Advanced Optimization Strategies

### Custom Objective Functions

Create custom optimization objectives beyond the built-in optimizers:

```python
import optuna
from src.config import OptimizationConfig
from src.study_manager import StudyManager

def custom_objective(trial):
    # Define hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Your custom model training logic here
    model = create_model(learning_rate, batch_size, dropout_rate)
    accuracy = train_and_evaluate(model)
    
    return accuracy

# Use with StudyManager
config = OptimizationConfig(study_name="custom_optimization")
manager = StudyManager(config)
study = manager.create_study("custom_study", direction="maximize")
study.optimize(custom_objective, n_trials=100)
```

### Multi-Objective Optimization

Optimize multiple conflicting objectives simultaneously:

```python
def multi_objective_function(trial):
    # Model hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    # Multiple objectives
    accuracy = model.score(X_test, y_test)
    model_size = n_estimators * max_depth  # Proxy for model complexity
    
    return accuracy, -model_size  # Maximize accuracy, minimize size

# Create multi-objective study
config = OptimizationConfig(
    study_name="multi_objective",
    directions=["maximize", "minimize"]
)
manager = StudyManager(config)
study = manager.create_study("pareto_optimization", directions=["maximize", "minimize"])
study.optimize(multi_objective_function, n_trials=100)

# Analyze Pareto front
pareto_front = study.best_trials
for trial in pareto_front:
    print(f"Trial {trial.number}: Accuracy={trial.values[0]:.3f}, Size={-trial.values[1]}")
```

### Advanced Sampling Strategies

#### CMA-ES for Continuous Optimization
```python
import optuna.samplers as samplers

config = OptimizationConfig(
    study_name="cmaes_optimization",
    sampler_name="CMA-ES"
)

# Or programmatically
sampler = samplers.CmaEsSampler(seed=42)
study = optuna.create_study(sampler=sampler)
```

#### Grid Search for Exhaustive Search
```python
search_space = {
    'learning_rate': [0.01, 0.1, 0.2],
    'batch_size': [16, 32, 64],
    'dropout': [0.1, 0.3, 0.5]
}

sampler = samplers.GridSampler(search_space)
study = optuna.create_study(sampler=sampler)
```

#### Quasi-Monte Carlo Sampling
```python
sampler = samplers.QMCSampler(seed=42)
study = optuna.create_study(sampler=sampler)
```

### Advanced Pruning Strategies

#### Hyperband Pruning
```python
import optuna.pruners as pruners

pruner = pruners.HyperbandPruner(
    min_resource=1,
    max_resource=100,
    reduction_factor=3
)

study = optuna.create_study(pruner=pruner)
```

#### Successive Halving
```python
pruner = pruners.SuccessiveHalvingPruner(
    min_resource=5,
    reduction_factor=4,
    min_early_stopping_rate=0
)
```

#### Custom Pruning Logic
```python
def objective_with_pruning(trial):
    for epoch in range(100):
        # Training step
        accuracy = train_epoch(model, epoch)
        
        # Report intermediate value
        trial.report(accuracy, epoch)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return accuracy
```

## 🔧 Advanced Configuration

### Database Storage

#### SQLite (Default)
```python
storage = "sqlite:///studies/my_study.db"
study = optuna.create_study(storage=storage)
```

#### PostgreSQL for Production
```python
storage = "postgresql://user:password@localhost:5432/optuna"
study = optuna.create_study(storage=storage)
```

#### MySQL
```python
storage = "mysql://user:password@localhost:3306/optuna"
study = optuna.create_study(storage=storage)
```

### Distributed Optimization

#### Multiple Workers
```python
# Worker 1
study = optuna.create_study(
    study_name="distributed_study",
    storage="postgresql://user:password@localhost:5432/optuna",
    load_if_exists=True
)
study.optimize(objective, n_trials=50)

# Worker 2 (run simultaneously)
study = optuna.create_study(
    study_name="distributed_study",
    storage="postgresql://user:password@localhost:5432/optuna",
    load_if_exists=True
)
study.optimize(objective, n_trials=50)
```

#### Ray Integration
```python
import ray
from ray import tune
from optuna.integration import RayTuneSearchAlgorithm

ray.init()

search_alg = RayTuneSearchAlgorithm(
    optuna_search_space={
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128])
    }
)

tune.run(
    trainable_function,
    search_alg=search_alg,
    num_samples=100
)
```

## 📊 Advanced Analysis

### Custom Visualizations

```python
import optuna.visualization as vis
import plotly.graph_objects as go

# Optimization history
fig = vis.plot_optimization_history(study)
fig.show()

# Parameter importance
fig = vis.plot_param_importances(study)
fig.show()

# Parallel coordinate plot
fig = vis.plot_parallel_coordinate(study)
fig.show()

# Custom visualization
def plot_custom_analysis(study):
    trials = study.trials
    values = [trial.value for trial in trials if trial.value is not None]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=values,
        mode='lines+markers',
        name='Objective Value'
    ))
    fig.update_layout(
        title='Custom Optimization Analysis',
        xaxis_title='Trial',
        yaxis_title='Objective Value'
    )
    return fig
```

### Statistical Analysis

```python
import numpy as np
from scipy import stats

def analyze_study_statistics(study):
    values = [trial.value for trial in study.trials if trial.value is not None]
    
    stats_summary = {
        'mean': np.mean(values),
        'std': np.std(values),
        'median': np.median(values),
        'best': study.best_value,
        'worst': min(values) if study.direction.name == 'MAXIMIZE' else max(values),
        'improvement_rate': calculate_improvement_rate(values),
        'convergence_trial': find_convergence_point(values)
    }
    
    return stats_summary

def calculate_improvement_rate(values):
    """Calculate the rate of improvement over trials."""
    if len(values) < 2:
        return 0
    
    improvements = []
    best_so_far = values[0]
    
    for value in values[1:]:
        if value > best_so_far:  # Assuming maximization
            improvements.append((value - best_so_far) / best_so_far)
            best_so_far = value
        else:
            improvements.append(0)
    
    return np.mean(improvements) if improvements else 0
```

## 🚀 Performance Optimization

### Memory Management

```python
# Limit memory usage
import psutil
import gc

def memory_efficient_objective(trial):
    # Monitor memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Your optimization logic
    result = expensive_computation(trial)
    
    # Clean up
    gc.collect()
    
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_used = final_memory - initial_memory
    
    # Prune if memory usage is too high
    if memory_used > 1000:  # 1GB limit
        raise optuna.TrialPruned()
    
    return result
```

### Parallel Processing

```python
from joblib import Parallel, delayed
import multiprocessing

def parallel_optimization():
    n_jobs = multiprocessing.cpu_count() - 1
    
    def run_study(study_name, n_trials):
        study = optuna.create_study(
            study_name=study_name,
            storage="sqlite:///parallel_studies.db",
            load_if_exists=True
        )
        study.optimize(objective, n_trials=n_trials)
        return study.best_value
    
    # Run multiple studies in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_study)(f"study_{i}", 25) 
        for i in range(n_jobs)
    )
    
    return results
```

### Caching and Memoization

```python
from functools import lru_cache
import hashlib

class CachedOptimizer:
    def __init__(self):
        self.cache = {}
    
    def _hash_params(self, params):
        """Create hash of parameters for caching."""
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def cached_objective(self, trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1),
            'batch_size': trial.suggest_int('batch_size', 16, 128),
        }
        
        param_hash = self._hash_params(params)
        
        if param_hash in self.cache:
            return self.cache[param_hash]
        
        # Expensive computation
        result = expensive_model_training(params)
        
        # Cache result
        self.cache[param_hash] = result
        return result
```

## 🔍 Debugging and Monitoring

### Logging Configuration

```python
import logging
from loguru import logger

# Configure detailed logging
logger.add(
    "optimization_debug.log",
    level="DEBUG",
    format="{time} | {level} | {message}",
    rotation="10 MB"
)

def debug_objective(trial):
    logger.info(f"Starting trial {trial.number}")
    
    try:
        params = trial.suggest_float('param', 0, 1)
        logger.debug(f"Suggested parameters: {params}")
        
        result = compute_result(params)
        logger.info(f"Trial {trial.number} result: {result}")
        
        return result
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise
```

### Real-time Monitoring

```python
import time
from datetime import datetime

class OptimizationMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.trial_times = []
    
    def monitor_callback(self, study, trial):
        elapsed = time.time() - self.start_time
        self.trial_times.append(elapsed)
        
        print(f"Trial {trial.number}: "
              f"Value={trial.value:.4f}, "
              f"Time={elapsed:.2f}s, "
              f"Best={study.best_value:.4f}")
        
        # Estimate completion time
        if len(self.trial_times) > 5:
            avg_time_per_trial = np.mean(np.diff(self.trial_times[-5:]))
            remaining_trials = study.n_trials - trial.number
            eta = remaining_trials * avg_time_per_trial
            print(f"ETA: {eta/60:.1f} minutes")

# Use the monitor
monitor = OptimizationMonitor()
study.optimize(objective, n_trials=100, callbacks=[monitor.monitor_callback])
```

## 🔗 Integration Examples

### MLflow Integration

```python
import mlflow
import mlflow.optuna

def mlflow_objective(trial):
    with mlflow.start_run():
        # Log parameters
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1),
            'batch_size': trial.suggest_int('batch_size', 16, 128)
        }
        mlflow.log_params(params)
        
        # Train model
        model = train_model(params)
        accuracy = evaluate_model(model)
        
        # Log metrics
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_model(model, 'model')
        
        return accuracy

# Auto-log Optuna study
mlflow.optuna.autolog()
study.optimize(mlflow_objective, n_trials=100)
```

### Weights & Biases Integration

```python
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback

wandb.init(project="optuna-optimization")

wandbc = WeightsAndBiasesCallback(metric_name="accuracy")
study.optimize(objective, n_trials=100, callbacks=[wandbc])
```

This advanced guide provides the tools and techniques needed for sophisticated optimization workflows. Experiment with these features to build powerful, efficient optimization pipelines.
