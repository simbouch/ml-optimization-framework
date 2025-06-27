# 🎯 Complete Optuna Tutorial: From Basics to Advanced

## 📚 Table of Contents

1. [What is Optuna?](#what-is-optuna)
2. [Why Use Optuna?](#why-use-optuna)
3. [Core Concepts](#core-concepts)
4. [Basic Example](#basic-example)
5. [Advanced Features](#advanced-features)
6. [Project Demonstrations](#project-demonstrations)
7. [Hands-on Practice](#hands-on-practice)
8. [Best Practices](#best-practices)

## 🤔 What is Optuna?

**Optuna** is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It's developed by Preferred Networks and is one of the most popular optimization libraries in the Python ecosystem.

### Key Features:
- **Automatic Optimization**: Finds the best hyperparameters without manual tuning
- **Multiple Algorithms**: Supports various optimization algorithms (TPE, Random, CMA-ES, etc.)
- **Pruning**: Stops unpromising trials early to save computation time
- **Multi-objective**: Optimizes multiple objectives simultaneously
- **Distributed**: Supports parallel and distributed optimization
- **Framework Agnostic**: Works with any ML framework (scikit-learn, PyTorch, TensorFlow, etc.)

### Real-World Impact:
- **Saves Time**: Automates the tedious process of manual hyperparameter tuning
- **Improves Performance**: Often finds better hyperparameters than manual search
- **Reduces Costs**: Efficient algorithms reduce computational requirements
- **Scales**: From small experiments to large-scale production systems

## 🎯 Why Use Optuna?

### Traditional Hyperparameter Tuning Problems:
```python
# Manual tuning - time consuming and inefficient
for learning_rate in [0.01, 0.1, 0.2]:
    for n_estimators in [50, 100, 200]:
        for max_depth in [3, 5, 10]:
            model = RandomForestClassifier(
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_depth=max_depth
            )
            # Train and evaluate...
            # This is 3×3×3 = 27 combinations!
```

### Optuna Solution:
```python
# Intelligent optimization - efficient and effective
def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    
    model = RandomForestClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    # Train and evaluate...
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # Intelligent search!
```

## 🧠 Core Concepts

### 1. **Study**
A study is an optimization session. It contains:
- **Objective function**: What you want to optimize
- **Search space**: Range of hyperparameters
- **Optimization history**: All trials and results

### 2. **Trial**
A single evaluation of the objective function with specific hyperparameter values.

### 3. **Objective Function**
The function that Optuna tries to optimize. It:
- Takes a `trial` object as input
- Suggests hyperparameter values
- Returns a score to optimize

### 4. **Samplers**
Algorithms that decide which hyperparameters to try next:
- **TPE (Tree-structured Parzen Estimator)**: Default, very effective
- **Random**: Baseline, uniform random sampling
- **CMA-ES**: Good for continuous parameters
- **Grid**: Exhaustive search over discrete values

### 5. **Pruners**
Algorithms that stop unpromising trials early:
- **Median Pruner**: Stops trials performing worse than median
- **Successive Halving**: Tournament-style elimination
- **Hyperband**: Advanced bandit-based pruning

## 🚀 Basic Example

Let's start with a simple example optimizing a Random Forest classifier:

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Define objective function
def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 1, 20)
    
    # Create and evaluate model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    # Use cross-validation for robust evaluation
    scores = cross_val_score(model, X, y, cv=3)
    return scores.mean()

# Create study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Get results
print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best parameters: {study.best_params}")
```

### Understanding the Output:
```
[I 2025-06-27 12:00:00,000] Trial 0 finished with value: 0.8234 and parameters: {'n_estimators': 45, 'max_depth': 8}
[I 2025-06-27 12:00:01,000] Trial 1 finished with value: 0.8567 and parameters: {'n_estimators': 78, 'max_depth': 12}
...
Best accuracy: 0.8923
Best parameters: {'n_estimators': 89, 'max_depth': 15}
```

## 🔬 Advanced Features

### 1. **Different Parameter Types**

```python
def advanced_objective(trial):
    # Integer parameters
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    
    # Float parameters
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    
    # Categorical parameters
    algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree'])
    
    # Log-scale parameters (for wide ranges)
    C = trial.suggest_float('C', 1e-5, 1e2, log=True)
    
    return score
```

### 2. **Pruning for Efficiency**

```python
def objective_with_pruning(trial):
    model = create_model(trial)
    
    for epoch in range(100):
        # Train one epoch
        train_one_epoch(model)
        
        # Evaluate intermediate performance
        accuracy = evaluate(model)
        
        # Report to Optuna
        trial.report(accuracy, epoch)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return final_accuracy

# Use with pruner
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner()
)
```

### 3. **Multi-objective Optimization**

```python
def multi_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    
    # Objective 1: Maximize accuracy
    accuracy = cross_val_score(model, X, y, cv=3).mean()
    
    # Objective 2: Minimize model complexity
    complexity = n_estimators * max_depth
    
    return accuracy, -complexity  # Minimize complexity (negative)

# Multi-objective study
study = optuna.create_study(directions=['maximize', 'maximize'])
study.optimize(multi_objective, n_trials=100)

# Get Pareto front
pareto_front = study.best_trials
print(f"Found {len(pareto_front)} Pareto optimal solutions")
```

### 4. **Study Persistence**

```python
# Save study to database
storage = optuna.storages.RDBStorage('sqlite:///my_study.db')
study = optuna.create_study(
    study_name='my_optimization',
    storage=storage,
    load_if_exists=True
)

# Continue optimization later
study.optimize(objective, n_trials=50)
```

## 🎪 Project Demonstrations

This project demonstrates all major Optuna features through 6 different studies:

### Study 1: TPE Sampling (RandomForest Classification)
**Purpose**: Show how TPE intelligently explores the parameter space
**Key Learning**: TPE focuses on promising regions after initial exploration

### Study 2: Random Sampling (XGBoost Regression)
**Purpose**: Baseline comparison with random search
**Key Learning**: Random search explores uniformly but less efficiently

### Study 3: Pruning (SVM Classification)
**Purpose**: Demonstrate computational efficiency with early stopping
**Key Learning**: Pruning saves time while maintaining solution quality

### Study 4: Multi-objective (Accuracy vs Complexity)
**Purpose**: Show trade-off optimization
**Key Learning**: No single "best" solution, Pareto front of optimal trade-offs

### Study 5: Simple Baseline (Logistic Regression)
**Purpose**: Provide simple model comparison
**Key Learning**: Sometimes simple models perform surprisingly well

### Study 6: Regression Task (RandomForest Regression)
**Purpose**: Show optimization for different ML tasks
**Key Learning**: Different metrics require different optimization approaches

## 🛠 Hands-on Practice

### Exercise 1: Basic Optimization
**Goal**: Optimize a Random Forest classifier for your colleagues

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score

# Load a real dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def exercise_objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 1, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)

    # Create model with suggested parameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    # Evaluate with cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    return scores.mean()

# Create and run study
study = optuna.create_study(direction='maximize')
study.optimize(exercise_objective, n_trials=50)

print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best parameters: {study.best_params}")

# Test on holdout set
best_model = RandomForestClassifier(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)
test_accuracy = best_model.score(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
```

### Exercise 2: Compare Different Samplers
**Goal**: Understand the difference between TPE and Random sampling

```python
def compare_samplers():
    # Same objective function as above
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 1, 20)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        scores = cross_val_score(model, X_train, y_train, cv=3)
        return scores.mean()

    # TPE Sampler (default)
    study_tpe = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study_tpe.optimize(objective, n_trials=50)

    # Random Sampler
    study_random = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.RandomSampler(seed=42)
    )
    study_random.optimize(objective, n_trials=50)

    print(f"TPE Best: {study_tpe.best_value:.4f}")
    print(f"Random Best: {study_random.best_value:.4f}")
    print(f"TPE is {study_tpe.best_value - study_random.best_value:.4f} better")

compare_samplers()
```

### Exercise 3: Multi-objective Optimization
**Goal**: Balance accuracy and model complexity

```python
def multi_objective_exercise(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 1, 20)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    # Objective 1: Maximize accuracy
    scores = cross_val_score(model, X_train, y_train, cv=3)
    accuracy = scores.mean()

    # Objective 2: Minimize model complexity
    complexity = n_estimators * max_depth

    return accuracy, -complexity  # Minimize complexity (negative)

# Multi-objective study
study_multi = optuna.create_study(directions=['maximize', 'maximize'])
study_multi.optimize(multi_objective_exercise, n_trials=100)

# Analyze Pareto front
pareto_trials = study_multi.best_trials
print(f"Found {len(pareto_trials)} Pareto optimal solutions:")

for i, trial in enumerate(pareto_trials[:5]):  # Show first 5
    accuracy = trial.values[0]
    complexity = -trial.values[1]  # Convert back to positive
    print(f"Solution {i+1}: Accuracy={accuracy:.4f}, Complexity={complexity}")
```

### Exercise 4: Real-World Example for Colleagues
**Complete working example they can run immediately**

```python
"""
Complete Optuna Example: Optimizing XGBoost for Regression
Perfect for sharing with colleagues who want to learn Optuna
"""

import optuna
import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

def colleague_example():
    print("🎯 Optuna Tutorial: XGBoost Regression Optimization")
    print("=" * 60)

    # Load dataset
    print("📊 Loading diabetes dataset...")
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define objective function
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 42
        }

        # Create and evaluate model
        model = xgb.XGBRegressor(**params)

        # Use negative MSE (since Optuna maximizes)
        scores = cross_val_score(model, X_train, y_train, cv=3,
                                scoring='neg_mean_squared_error')
        return scores.mean()

    # Create and run optimization
    print("🔍 Starting optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # Results
    print("\n📈 Optimization Results:")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")

    # Test final model
    print("\n🧪 Testing optimized model...")
    best_model = xgb.XGBRegressor(**study.best_params)
    best_model.fit(X_train, y_train)

    # Compare with default model
    default_model = xgb.XGBRegressor(random_state=42)
    default_model.fit(X_train, y_train)

    optimized_mse = mean_squared_error(y_test, best_model.predict(X_test))
    default_mse = mean_squared_error(y_test, default_model.predict(X_test))

    print(f"Optimized MSE: {optimized_mse:.2f}")
    print(f"Default MSE: {default_mse:.2f}")
    print(f"Improvement: {((default_mse - optimized_mse) / default_mse * 100):.1f}%")

    return study

# Run the example
if __name__ == "__main__":
    study = colleague_example()
```

## 📋 Best Practices

### 1. **Define Good Objective Functions**
```python
def good_objective(trial):
    # Use cross-validation for robust evaluation
    scores = cross_val_score(model, X, y, cv=5)
    
    # Handle exceptions gracefully
    try:
        return scores.mean()
    except Exception as e:
        # Return worst possible score for failed trials
        return 0.0
```

### 2. **Choose Appropriate Search Spaces**
```python
# Good: Reasonable ranges
n_estimators = trial.suggest_int('n_estimators', 10, 200)

# Bad: Too wide range
n_estimators = trial.suggest_int('n_estimators', 1, 10000)

# Good: Log scale for wide ranges
C = trial.suggest_float('C', 1e-5, 1e2, log=True)
```

### 3. **Use Pruning Wisely**
```python
# Good: Pruning with enough startup trials
pruner = optuna.pruners.MedianPruner(n_startup_trials=10)

# Good: Report meaningful intermediate values
trial.report(validation_accuracy, epoch)
```

### 4. **Monitor and Analyze**
```python
# Analyze study results
print(f"Number of finished trials: {len(study.trials)}")
print(f"Best trial: {study.best_trial}")

# Get trials dataframe for analysis
df = study.trials_dataframe()
print(df.head())

# Plot optimization history
optuna.visualization.plot_optimization_history(study)
```

## 🎯 Next Steps

### For Beginners:
1. **Start Simple**: Use the basic example with your own dataset
2. **Experiment**: Try different parameter ranges and types
3. **Visualize**: Use Optuna's built-in plotting functions
4. **Compare**: Run with and without optimization to see the difference

### For Intermediate Users:
1. **Add Pruning**: Implement early stopping for efficiency
2. **Multi-objective**: Explore trade-offs in your models
3. **Custom Samplers**: Try different optimization algorithms
4. **Integration**: Use with your favorite ML framework

### For Advanced Users:
1. **Distributed Optimization**: Scale across multiple machines
2. **Custom Pruners**: Implement domain-specific pruning strategies
3. **Hyperparameter Importance**: Analyze which parameters matter most
4. **Production Integration**: Deploy optimized models in production

## 🎉 Conclusion

Optuna transforms hyperparameter optimization from a tedious manual process into an intelligent, automated system. This project demonstrates:

- **All Major Features**: From basic optimization to advanced multi-objective
- **Real Examples**: Practical ML scenarios with actual models
- **Best Practices**: Professional patterns for production use
- **Educational Value**: Progressive learning from simple to complex

**Ready to optimize? Start with the basic example and work your way up!**

---

*This tutorial is part of the ML Optimization Framework project. For hands-on practice, run the project and explore the interactive dashboard at http://localhost:8080*
