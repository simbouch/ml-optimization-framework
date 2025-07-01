# 🚀 Quick Start: 5 Essential Optuna Projects
*Get hands-on experience with Optuna in just 5 focused projects*

## 🎯 Overview

These 5 projects are designed to give you practical experience with Optuna's most important features. Each project takes 1-2 hours and builds essential skills.

## 📋 Prerequisites

- Basic Python and scikit-learn knowledge
- Completed the [basic tutorial](tutorial.md) sections 1-5
- Working Optuna installation

```bash
# Quick setup verification
python -c "import optuna; print('✅ Ready to start!')"
```

---

## 🥇 Project 1: Your First Optimization (1 hour)
**Goal**: Optimize a model on your own data

### What You'll Learn
- Basic Optuna workflow
- Parameter suggestion methods
- Study creation and management

### Quick Implementation
```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris  # Replace with your data

# Load data (replace with your dataset)
X, y = load_iris(return_X_y=True)

def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    
    # Create and evaluate model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    return cross_val_score(model, X, y, cv=3).mean()

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best score: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### ✅ Success Check
- [ ] Successfully runs without errors
- [ ] Shows improvement over default parameters
- [ ] Understand what each parameter does

### 🎯 Challenge
Try different parameter ranges and see how it affects results!

---

## 🥈 Project 2: Sampler Showdown (1 hour)
**Goal**: Compare different optimization algorithms

### What You'll Learn
- Different sampling strategies
- When to use TPE vs Random vs CMA-ES
- Performance comparison techniques

### Quick Implementation
```python
import optuna
import time

def compare_samplers(objective_func, n_trials=30):
    """Compare different samplers on the same problem"""
    
    samplers = {
        'TPE': optuna.samplers.TPESampler(seed=42),
        'Random': optuna.samplers.RandomSampler(seed=42),
        'CMA-ES': optuna.samplers.CmaEsSampler(seed=42)
    }
    
    results = {}
    
    for name, sampler in samplers.items():
        print(f"Testing {name} sampler...")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )
        
        start_time = time.time()
        study.optimize(objective_func, n_trials=n_trials)
        elapsed_time = time.time() - start_time
        
        results[name] = {
            'best_score': study.best_value,
            'time': elapsed_time,
            'study': study
        }
        
        print(f"  Best score: {study.best_value:.4f}")
        print(f"  Time: {elapsed_time:.2f}s")
    
    return results

# Use your objective function from Project 1
results = compare_samplers(objective)

# Find the winner
best_sampler = max(results.keys(), key=lambda k: results[k]['best_score'])
print(f"\n🏆 Winner: {best_sampler}")
```

### ✅ Success Check
- [ ] All three samplers run successfully
- [ ] Can explain why one sampler performed better
- [ ] Understand trade-offs between speed and quality

### 🎯 Challenge
Try with different numbers of trials (10, 50, 100) and see how results change!

---

## 🥉 Project 3: Speed Optimization with Pruning (1.5 hours)
**Goal**: Make optimization faster with early stopping

### What You'll Learn
- Pruning concepts and implementation
- Intermediate result reporting
- Efficiency vs quality trade-offs

### Quick Implementation
```python
import optuna
from sklearn.neural_network import MLPClassifier
import numpy as np

def pruning_objective(trial):
    """Objective function with pruning support"""
    
    # Suggest parameters
    hidden_size = trial.suggest_int('hidden_size', 50, 200)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
    
    # Create model with incremental training
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_size,),
        learning_rate_init=learning_rate,
        max_iter=1,  # Train one iteration at a time
        warm_start=True,
        random_state=42
    )
    
    # Split data for validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Incremental training with intermediate reporting
    for epoch in range(20):
        model.fit(X_train, y_train)
        
        # Evaluate current performance
        score = model.score(X_val, y_val)
        
        # Report to Optuna
        trial.report(score, epoch)
        
        # Check if should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return score

# Compare with and without pruning
print("Without pruning:")
study_no_pruning = optuna.create_study(direction='maximize')
study_no_pruning.optimize(pruning_objective, n_trials=20, timeout=60)

print("\nWith pruning:")
study_with_pruning = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
)
study_with_pruning.optimize(pruning_objective, n_trials=20, timeout=60)

print(f"\nResults:")
print(f"No pruning: {len(study_no_pruning.trials)} trials, best: {study_no_pruning.best_value:.4f}")
print(f"With pruning: {len(study_with_pruning.trials)} trials, best: {study_with_pruning.best_value:.4f}")
```

### ✅ Success Check
- [ ] Pruning runs faster than no pruning
- [ ] Understand when trials get pruned
- [ ] Can explain the efficiency gains

### 🎯 Challenge
Try different pruning strategies (SuccessiveHalvingPruner, HyperbandPruner)!

---

## 🏅 Project 4: Multi-Objective Trade-offs (1.5 hours)
**Goal**: Optimize multiple conflicting objectives

### What You'll Learn
- Multi-objective optimization concepts
- Pareto front analysis
- Real-world constraint handling

### Quick Implementation
```python
import optuna
import time

def multi_objective_function(trial):
    """Optimize accuracy vs prediction speed"""
    
    # Model parameters
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    
    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Objective 1: Accuracy (maximize)
    accuracy = model.score(X_test, y_test)
    
    # Objective 2: Prediction speed (maximize = minimize time)
    start_time = time.time()
    _ = model.predict(X_test)
    prediction_time = time.time() - start_time
    speed_score = 1.0 / (prediction_time + 1e-6)  # Higher is better
    
    return accuracy, speed_score

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Multi-objective optimization
study = optuna.create_study(directions=['maximize', 'maximize'])
study.optimize(multi_objective_function, n_trials=50)

# Analyze Pareto front
pareto_front = study.best_trials
print(f"Found {len(pareto_front)} Pareto optimal solutions")

print("\nPareto Front Solutions:")
for i, trial in enumerate(pareto_front[:5]):  # Show top 5
    accuracy, speed = trial.values
    print(f"Solution {i+1}: Accuracy={accuracy:.4f}, Speed={speed:.2f}")
    print(f"  Parameters: {trial.params}")
```

### ✅ Success Check
- [ ] Successfully finds multiple Pareto optimal solutions
- [ ] Can explain the trade-off between accuracy and speed
- [ ] Understand how to choose final solution

### 🎯 Challenge
Add a third objective (model complexity) and analyze the 3D trade-offs!

---

## 🎖️ Project 5: Production Pipeline (2 hours)
**Goal**: Build a complete optimized ML pipeline

### What You'll Learn
- End-to-end pipeline optimization
- Study persistence and resumption
- Production deployment considerations

### Quick Implementation
```python
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SelectKBest
from sklearn.ensemble import RandomForestClassifier

def pipeline_objective(trial):
    """Optimize entire ML pipeline"""
    
    # Preprocessing parameters
    k_features = trial.suggest_int('k_features', 5, min(20, X.shape[1]))
    
    # Model parameters
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(k=k_features)),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        ))
    ])
    
    # Evaluate pipeline
    scores = cross_val_score(pipeline, X, y, cv=3)
    return scores.mean()

# Create persistent study
storage = optuna.storages.RDBStorage('sqlite:///my_production_study.db')
study = optuna.create_study(
    study_name='production_pipeline',
    storage=storage,
    direction='maximize',
    load_if_exists=True
)

# Optimize
study.optimize(pipeline_objective, n_trials=100)

# Build final model with best parameters
best_params = study.best_params
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=best_params['k_features'])),
    ('classifier', RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        random_state=42
    ))
])

# Train final model
final_pipeline.fit(X, y)

print(f"Final model accuracy: {study.best_value:.4f}")
print(f"Best parameters: {best_params}")

# Save model for production
import joblib
joblib.dump(final_pipeline, 'optimized_model.pkl')
print("✅ Model saved for production!")
```

### ✅ Success Check
- [ ] Pipeline optimization runs successfully
- [ ] Study persists to database
- [ ] Final model is saved and loadable
- [ ] Can explain the complete workflow

### 🎯 Challenge
Add hyperparameter optimization for the preprocessing steps (scaler types, feature selection methods)!

---

## 🎓 What's Next?

### Immediate Next Steps
1. **Apply to Your Work**: Use these patterns on your real projects
2. **Explore Advanced Features**: Custom samplers, distributed optimization
3. **Share Knowledge**: Teach colleagues what you've learned

### Advanced Learning
- Complete the [full practice projects guide](practice_projects.md)
- Study the [advanced examples](../examples/advanced/)
- Read the [Optuna research paper](https://arxiv.org/abs/1907.10902)

### Get Help
- **Stuck?** Check the [troubleshooting guide](tutorial.md#troubleshooting)
- **Questions?** Visit [Optuna GitHub Discussions](https://github.com/optuna/optuna/discussions)
- **Want More?** Join the ML optimization community!

---

## 📊 Progress Tracker

Track your completion:
- [ ] Project 1: Your First Optimization
- [ ] Project 2: Sampler Showdown
- [ ] Project 3: Speed Optimization with Pruning
- [ ] Project 4: Multi-Objective Trade-offs
- [ ] Project 5: Production Pipeline

**Congratulations! 🎉 You're now ready to optimize any ML project with Optuna!**

---

*Total time investment: 6-8 hours for practical Optuna mastery*
