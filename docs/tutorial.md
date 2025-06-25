# 📚 Complete Tutorial: ML Optimization Framework

## Table of Contents

1. [Introduction](#introduction)
2. [Basic Concepts](#basic-concepts)
3. [Your First Optimization](#your-first-optimization)
4. [Understanding the Results](#understanding-the-results)
5. [Advanced Features](#advanced-features)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## 1. Introduction

Welcome to the comprehensive tutorial for the ML Optimization Framework! This guide will take you from zero to expert in using Optuna for hyperparameter optimization.

### What You'll Learn

By the end of this tutorial, you'll be able to:
- ✅ Set up and run basic hyperparameter optimization
- ✅ Understand and interpret optimization results
- ✅ Use advanced features like multi-objective optimization
- ✅ Implement custom optimization strategies
- ✅ Deploy optimization pipelines in production

### Prerequisites

- Basic Python knowledge
- Familiarity with machine learning concepts
- Understanding of hyperparameters and cross-validation

---

## 2. Basic Concepts

### What is Hyperparameter Optimization?

Hyperparameter optimization is the process of finding the best configuration of hyperparameters for a machine learning model. Unlike model parameters (learned during training), hyperparameters are set before training and control the learning process.

### Why Optuna?

Optuna is a state-of-the-art hyperparameter optimization framework that offers:

- **Efficient Search**: Advanced algorithms like TPE (Tree-structured Parzen Estimator)
- **Pruning**: Early stopping of unpromising trials
- **Scalability**: Distributed optimization support
- **Flexibility**: Support for any ML framework

### Key Concepts

#### Trial
A single execution of the objective function with a specific set of hyperparameters.

#### Study
A collection of trials that explores the hyperparameter space.

#### Sampler
The algorithm that suggests hyperparameter values (TPE, Random, CMA-ES, etc.).

#### Pruner
The algorithm that decides whether to stop unpromising trials early.

---

## 3. Your First Optimization

Let's start with a simple example to get you familiar with the framework.

### Step 1: Basic Setup

```python
# Import the framework
from src.data.data_pipeline import DataPipeline
from src.models.random_forest_optimizer import RandomForestOptimizer

# Set up data pipeline
print("🔄 Setting up data pipeline...")
pipeline = DataPipeline(random_state=42)
summary = pipeline.prepare_data()

print(f"✅ Data prepared: {summary['total_samples']} samples")
```

### Step 2: Run Your First Optimization

```python
# Get data splits
X_train, X_val, y_train, y_val = pipeline.get_train_val_data()

# Initialize optimizer
print("🌲 Initializing Random Forest optimizer...")
optimizer = RandomForestOptimizer(
    random_state=42,
    cv_folds=5,
    verbose=True
)

# Run optimization
print("🚀 Starting optimization...")
study = optimizer.optimize(
    X_train, X_val, y_train, y_val,
    n_trials=50  # Start with 50 trials
)

print(f"🎯 Best score: {study.best_value:.4f}")
print(f"⚙️ Best parameters: {study.best_params}")
```

### Step 3: Evaluate Results

```python
# Evaluate on test set
X_test, y_test = pipeline.get_test_data()
test_metrics = optimizer.evaluate(X_test, y_test)

print("📊 Test Results:")
for metric, value in test_metrics.items():
    print(f"   • {metric}: {value:.4f}")
```

**Expected Output:**
```
🎯 Best score: 0.8712
⚙️ Best parameters: {'n_estimators': 200, 'max_depth': 15, ...}
📊 Test Results:
   • accuracy: 0.8685
   • f1_score: 0.8634
   • precision: 0.8701
   • recall: 0.8568
   • roc_auc: 0.9234
```

---

## 4. Understanding the Results

### Interpreting the Output

#### Best Score (Cross-Validation)
- This is the best cross-validation score achieved during optimization
- Higher is better for accuracy-based metrics
- Represents the model's expected performance on unseen data

#### Best Parameters
- The hyperparameter configuration that achieved the best score
- These are the optimal settings for your model
- Can be used to train the final model

#### Test Metrics
- Performance on the held-out test set
- Most important metric for real-world performance
- Should be close to the CV score (if not, investigate overfitting)

### Analyzing Feature Importance

```python
# Get feature importance analysis
importance = optimizer.analyze_feature_importance()

print("🔍 Feature Importance Analysis:")
print(f"   • Mean importance: {importance['mean_importance']:.4f}")
print(f"   • Top 5 features: {importance['top_features_indices'][:5]}")

# Get model complexity
complexity = optimizer.get_model_complexity()
print(f"🏗️ Model Complexity:")
print(f"   • Trees: {complexity['n_estimators']}")
print(f"   • Average depth: {complexity['avg_tree_depth']:.2f}")
```

### Visualizing Results

```python
from src.visualization.plots import OptimizationPlotter

# Create plotter
plotter = OptimizationPlotter()

# Plot optimization history
history_fig = plotter.plot_optimization_history_custom(
    study, 
    interactive=True
)
history_fig.show()

# Plot parameter importance
importance_fig = plotter.plot_parameter_importance_custom(
    study,
    interactive=True
)
importance_fig.show()
```

---

## 5. Advanced Features

### Multi-Model Comparison

Compare multiple algorithms to find the best performer:

```python
from src.models.xgboost_optimizer import XGBoostOptimizer
from src.models.lightgbm_optimizer import LightGBMOptimizer

# Initialize all optimizers
optimizers = {
    'Random Forest': RandomForestOptimizer(random_state=42, verbose=False),
    'XGBoost': XGBoostOptimizer(random_state=42, verbose=False),
    'LightGBM': LightGBMOptimizer(random_state=42, verbose=False)
}

# Run optimization for each model
results = {}
for model_name, optimizer in optimizers.items():
    print(f"🎯 Optimizing {model_name}...")
    
    study = optimizer.optimize(X_train, X_val, y_train, y_val, n_trials=30)
    test_metrics = optimizer.evaluate(X_test, y_test)
    
    results[model_name] = {
        'cv_score': study.best_value,
        'test_accuracy': test_metrics['accuracy'],
        'best_params': study.best_params
    }

# Print comparison
print("\n📈 Model Comparison:")
for model, result in sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True):
    print(f"   {model}: {result['test_accuracy']:.4f}")
```

### Multi-Objective Optimization

Optimize for multiple objectives simultaneously:

```python
from src.optimization.advanced_features import MultiObjectiveOptimizer

# Setup multi-objective optimization
multi_opt = MultiObjectiveOptimizer(
    objectives=["accuracy", "training_time"],
    directions=["maximize", "minimize"],
    random_state=42
)

# Create multi-objective study
study = multi_opt.create_multi_objective_study("accuracy_vs_time")

# Run optimization
print("🎯 Running multi-objective optimization...")
study = multi_opt.optimize(
    optimizer, X_train, X_val, y_train, y_val,
    n_trials=50
)

# Analyze Pareto front
pareto_front = multi_opt.get_pareto_front()
print(f"🏆 Found {len(pareto_front)} Pareto-optimal solutions")

# Analyze trade-offs
trade_offs = multi_opt.analyze_trade_offs()
print("⚖️ Trade-off Analysis:")
print(f"   • Pareto solutions: {trade_offs['n_pareto_solutions']}")
print(f"   • Objective correlations: {trade_offs['objective_correlations']}")
```

### Sampler Comparison

Compare different optimization algorithms:

```python
from src.optimization.advanced_features import SamplerComparison

# Define objective function
def objective(trial):
    model = optimizer.create_model(trial)
    scores = optimizer._cross_validate(model, trial)
    return scores.mean()

# Run sampler comparison
sampler_comp = SamplerComparison(random_state=42)
results = sampler_comp.compare_samplers(
    objective,
    n_trials=25,  # Per sampler
    n_runs=3      # Independent runs
)

# Get summary
summary = sampler_comp.get_comparison_summary()
print("🔬 Sampler Comparison Results:")
print(summary)
```

### Custom Configuration

Create custom hyperparameter spaces:

```python
from src.optimization.config import OptimizationConfig

# Load custom configuration
config = OptimizationConfig("config/hyperparameters.yaml")

# Modify search space
custom_space = {
    "n_estimators": {"type": "int", "low": 100, "high": 1000, "step": 100},
    "max_depth": {"type": "int", "low": 5, "high": 25},
    "min_samples_split": {"type": "int", "low": 2, "high": 50}
}

config.add_hyperparameter_space("custom_rf", custom_space)

# Use custom configuration
optimizer = RandomForestOptimizer(config=config)
```

---

## 6. Best Practices

### Optimization Strategy

#### Start Small, Scale Up
```python
# Phase 1: Quick exploration (20-50 trials)
study = optimizer.optimize(X_train, X_val, y_train, y_val, n_trials=50)

# Phase 2: Focused search (100-200 trials)
study = optimizer.optimize(X_train, X_val, y_train, y_val, n_trials=150, study=study)

# Phase 3: Fine-tuning (50-100 additional trials)
study = optimizer.optimize(X_train, X_val, y_train, y_val, n_trials=100, study=study)
```

#### Use Appropriate Metrics
```python
# For imbalanced datasets
optimizer = RandomForestOptimizer(scoring_metric="f1")

# For ranking problems
optimizer = RandomForestOptimizer(scoring_metric="roc_auc")
```

#### Enable Pruning for Efficiency
```python
from src.optimization.study_manager import StudyManager

# Create study with aggressive pruning
study_manager = StudyManager()
study = study_manager.create_study(
    "efficient_optimization",
    pruner_name="successive_halving"
)
```

### Reproducibility

#### Always Set Random Seeds
```python
# Set seeds everywhere
RANDOM_STATE = 42

pipeline = DataPipeline(random_state=RANDOM_STATE)
optimizer = RandomForestOptimizer(random_state=RANDOM_STATE)
```

#### Save and Version Studies
```python
# Use persistent storage
study_manager = StudyManager(storage_url="sqlite:///my_experiments.db")

# Version your experiments
study = study_manager.create_study(f"experiment_v1_{datetime.now().strftime('%Y%m%d')}")
```

### Performance Optimization

#### Use Parallel Processing
```python
# Enable parallel jobs in Random Forest
optimizer = RandomForestOptimizer(n_jobs=-1)

# Use GPU acceleration for XGBoost/LightGBM
xgb_optimizer = XGBoostOptimizer(use_gpu=True)
```

#### Monitor Resource Usage
```python
import psutil
import time

def monitor_optimization():
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    
    # Run optimization
    study = optimizer.optimize(X_train, X_val, y_train, y_val, n_trials=100)
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used
    
    print(f"⏱️ Time: {end_time - start_time:.2f}s")
    print(f"💾 Memory: {(end_memory - start_memory) / 1024**2:.2f}MB")
    
    return study
```

---

## 7. Troubleshooting

### Common Issues and Solutions

#### Issue: Optimization Not Improving

**Symptoms:**
- Best score plateaus early
- No improvement after many trials

**Solutions:**
```python
# 1. Increase search space
config.add_hyperparameter_space("wider_rf", {
    "n_estimators": {"type": "int", "low": 50, "high": 1000},
    "max_depth": {"type": "int", "low": 3, "high": 50}
})

# 2. Try different sampler
study = study_manager.create_study(
    "exploration_study",
    sampler_name="cmaes"  # Try CMA-ES instead of TPE
)

# 3. Disable pruning temporarily
study = study_manager.create_study(
    "no_pruning_study",
    pruner_name="nop"  # No pruning
)
```

#### Issue: Overfitting to Validation Set

**Symptoms:**
- Large gap between CV and test performance
- Performance degrades with more trials

**Solutions:**
```python
# 1. Use more CV folds
optimizer = RandomForestOptimizer(cv_folds=10)

# 2. Use nested cross-validation
from sklearn.model_selection import cross_val_score

def robust_objective(trial):
    model = optimizer.create_model(trial)
    # Use full cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    return scores.mean()

# 3. Hold out larger test set
pipeline = DataPipeline(test_size=0.3)  # 30% for testing
```

#### Issue: Slow Optimization

**Symptoms:**
- Each trial takes too long
- Optimization doesn't finish in reasonable time

**Solutions:**
```python
# 1. Enable pruning
study = study_manager.create_study(
    "fast_study",
    pruner_name="median",
    pruner_params={"n_startup_trials": 5, "n_warmup_steps": 3}
)

# 2. Reduce CV folds for initial exploration
optimizer = RandomForestOptimizer(cv_folds=3)

# 3. Use subset of data for initial trials
def fast_objective(trial):
    if trial.number < 20:  # First 20 trials
        # Use 50% of data
        subset_size = len(X_train) // 2
        X_subset = X_train[:subset_size]
        y_subset = y_train[:subset_size]
    else:
        # Use full data for later trials
        X_subset = X_train
        y_subset = y_train
    
    model = optimizer.create_model(trial)
    scores = cross_val_score(model, X_subset, y_subset, cv=3)
    return scores.mean()
```

#### Issue: Memory Issues

**Symptoms:**
- Out of memory errors
- System becomes unresponsive

**Solutions:**
```python
# 1. Reduce model complexity
config.add_hyperparameter_space("memory_efficient_rf", {
    "n_estimators": {"type": "int", "low": 10, "high": 100},  # Fewer trees
    "max_depth": {"type": "int", "low": 3, "high": 10}       # Shallower trees
})

# 2. Use data subsampling
pipeline = DataPipeline()
# Use only part of the dataset
X_train_subset = X_train[:5000]  # First 5000 samples

# 3. Enable garbage collection
import gc

def memory_efficient_objective(trial):
    model = optimizer.create_model(trial)
    scores = cross_val_score(model, X_train, y_train, cv=3)
    
    # Clean up
    del model
    gc.collect()
    
    return scores.mean()
```

### Getting Help

#### Enable Debug Logging
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('src')
logger.setLevel(logging.DEBUG)
```

#### Check Study Statistics
```python
# Analyze study progress
study_summary = study_manager.get_study_summary("my_study")
print(f"Completed trials: {study_summary['n_completed_trials']}")
print(f"Pruned trials: {study_summary['n_pruned_trials']}")
print(f"Failed trials: {study_summary['n_failed_trials']}")

# Check for issues
if study_summary['n_failed_trials'] > study_summary['n_completed_trials'] * 0.1:
    print("⚠️ High failure rate - check your objective function")
```

#### Validate Your Setup
```python
# Run framework validation
from tests.test_framework import run_basic_validation

if run_basic_validation():
    print("✅ Framework setup is correct")
else:
    print("❌ Framework setup has issues")
```

---

## Next Steps

Congratulations! You've completed the tutorial. Here are some next steps:

1. **Explore the Jupyter Notebook**: Run `notebooks/ml_optimization_demo.ipynb` for interactive examples
2. **Try the CLI**: Use `scripts/cli_runner.py` for automated optimization
3. **Read the API Reference**: Check `docs/api_reference.md` for detailed API documentation
4. **Join the Community**: Contribute to the project or ask questions

### Advanced Topics to Explore

- **Custom Samplers**: Implement your own optimization algorithms
- **Distributed Optimization**: Scale across multiple machines
- **Neural Architecture Search**: Optimize deep learning architectures
- **AutoML Integration**: Combine with automated feature engineering

Happy optimizing! 🚀
