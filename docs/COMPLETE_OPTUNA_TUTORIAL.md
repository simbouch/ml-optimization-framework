# Complete Optuna Tutorial: From Basics to Advanced ML Optimization

## 📚 Table of Contents

1. [What is Optuna?](#what-is-optuna)
2. [Why Use Optuna for ML?](#why-use-optuna-for-ml)
3. [Optuna Core Concepts](#optuna-core-concepts)
4. [Our Project: Complete ML Optimization Framework](#our-project)
5. [Hands-on Tutorial](#hands-on-tutorial)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)

---

## 🎯 What is Optuna?

**Optuna** is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It's developed by Preferred Networks and has become one of the most popular tools for hyperparameter tuning in the ML community.

### Key Features:
- **Automatic hyperparameter optimization** - No manual tuning required
- **Efficient algorithms** - Uses advanced sampling methods like TPE (Tree-structured Parzen Estimator)
- **Easy parallelization** - Run multiple optimization trials simultaneously
- **Pruning capabilities** - Stop unpromising trials early to save time
- **Multi-objective optimization** - Optimize multiple metrics simultaneously
- **Integration-friendly** - Works with any ML framework (scikit-learn, XGBoost, PyTorch, etc.)

---

## 🤔 Why Use Optuna for ML?

### The Problem with Manual Hyperparameter Tuning:
```python
# Manual approach - time-consuming and inefficient
for learning_rate in [0.01, 0.1, 0.2]:
    for n_estimators in [50, 100, 200]:
        for max_depth in [3, 5, 10]:
            # Train model with these parameters
            # This is 3×3×3 = 27 combinations!
```

### The Optuna Solution:
```python
# Optuna approach - intelligent and efficient
def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 15)

    # Train model and return performance metric
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # Intelligent search!
```

### Benefits:
- ✅ **Faster convergence** - Finds good parameters with fewer trials
- ✅ **Better results** - Often finds better hyperparameters than manual search
- ✅ **Less manual work** - Automates the entire process
- ✅ **Scalable** - Easy to run on multiple machines
- ✅ **Flexible** - Works with any ML algorithm or framework

---

## 🧠 Optuna Core Concepts

### 1. **Study** - The optimization session
```python
study = optuna.create_study(direction='maximize')  # or 'minimize'
```

### 2. **Trial** - A single evaluation of the objective function
```python
def objective(trial):
    # trial.suggest_* methods define the search space
    x = trial.suggest_float('x', -10, 10)
    return x ** 2  # Function to optimize
```

### 3. **Objective Function** - What you want to optimize
```python
def objective(trial):
    # Define hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
    }

    # Train model
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Return metric to optimize
    predictions = model.predict(X_val)
    return accuracy_score(y_val, predictions)
```

### 4. **Samplers** - How Optuna chooses hyperparameters
- **TPESampler** (default) - Tree-structured Parzen Estimator
- **RandomSampler** - Random search
- **CmaEsSampler** - Covariance Matrix Adaptation Evolution Strategy
- **GridSampler** - Grid search

### 5. **Pruners** - Early stopping for unpromising trials
- **MedianPruner** - Stop if performance is below median
- **SuccessiveHalvingPruner** - Successive halving algorithm
- **HyperbandPruner** - Hyperband algorithm

---

## 🚀 Our Project: Complete ML Optimization Framework

This project demonstrates **ALL major Optuna features** through a real-world ML optimization framework. Here's what we've built:

### What Our Framework Includes:

#### 🎛️ **Interactive Optuna Dashboard**
- Real-time visualization of optimization progress
- Parameter importance analysis
- Multi-objective Pareto front visualization
- Study comparison tools
- Trial filtering and analysis

#### 🤖 **Multiple ML Algorithms**
- **Random Forest** - Ensemble method with hyperparameter optimization
- **XGBoost** - Gradient boosting with advanced pruning
- **LightGBM** - Fast gradient boosting optimization

#### 🔬 **All Optuna Features Demonstrated**
- Single-objective optimization
- Multi-objective optimization (Pareto fronts)
- Different samplers (TPE, Random, CMA-ES)
- Different pruners (Median, Successive Halving, Hyperband)
- Custom callbacks and metrics
- Study management and persistence
- Distributed optimization setup

#### 📊 **Real Dataset**
- Adult Income dataset from OpenML
- Complete data preprocessing pipeline
- Professional train/validation/test splits

---

## 🚀 Quick Start

### 1. Setup and Installation

```bash
# Clone the repository
git clone https://github.com/simbouch/ml-optimization-framework.git
cd ml-optimization-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Experience the Complete Demo

```bash
# Option 1: One-click demo (Recommended)
python scripts/deploy_complete_demo.py

# Option 2: Step by step
python scripts/populate_dashboard.py  # Create demo studies
python scripts/start_dashboard.py     # Start dashboard

# Then open http://localhost:8080 in your browser
```

### 3. Explore All Features

```bash
# Run comprehensive feature showcase
python scripts/showcase_all_optuna_features.py

# Validate framework functionality
python scripts/validate_framework.py
```

## 📊 Optuna Features Demonstrated

### 1. Single-Objective Optimization

**What it does:** Optimize a single metric (e.g., accuracy, F1-score)

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 1, 10)
    
    # Create and evaluate model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    scores = cross_val_score(model, X, y, cv=3)
    return scores.mean()

# Create study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")
```

**Dashboard View:** See optimization history, parameter importance, and convergence plots.

### 2. Multi-Objective Optimization

**What it does:** Optimize multiple conflicting objectives simultaneously

```python
def multi_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 1, 15)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    # Objective 1: Accuracy (maximize)
    accuracy = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
    
    # Objective 2: Model complexity (minimize)
    complexity = n_estimators * max_depth
    
    return accuracy, complexity

# Multi-objective study
study = optuna.create_study(directions=['maximize', 'minimize'])
study.optimize(multi_objective, n_trials=50)

# Get Pareto front
pareto_solutions = study.best_trials
print(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
```

**Dashboard View:** Pareto front visualization, trade-off analysis between objectives.

### 3. Different Samplers

**What it does:** Compare different optimization algorithms

```python
samplers = {
    'TPE': optuna.samplers.TPESampler(seed=42),
    'Random': optuna.samplers.RandomSampler(seed=42),
    'CMA-ES': optuna.samplers.CmaEsSampler(seed=42),
}

for name, sampler in samplers.items():
    study = optuna.create_study(
        study_name=f"sampler_{name}",
        direction='maximize',
        sampler=sampler
    )
    study.optimize(objective, n_trials=30)
    print(f"{name}: {study.best_value:.4f}")
```

**Dashboard View:** Compare convergence speed and final performance across samplers.

### 4. Pruning Strategies

**What it does:** Stop unpromising trials early to save computation

```python
def objective_with_pruning(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 1, 10)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    # Report intermediate values
    for step in range(3):
        scores = cross_val_score(model, X, y, cv=2)
        intermediate_value = scores.mean()
        
        trial.report(intermediate_value, step)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Final evaluation
    final_scores = cross_val_score(model, X, y, cv=5)
    return final_scores.mean()

# Study with pruning
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective_with_pruning, n_trials=50)
```

**Dashboard View:** See which trials were pruned and at what stage.

### 5. Callbacks and Custom Metrics

**What it does:** Add custom behavior during optimization

```python
def logging_callback(study, trial):
    if trial.value is not None:
        print(f"Trial {trial.number}: {trial.value:.4f}")

def early_stopping_callback(study, trial):
    if len(study.trials) > 20:
        # Stop if no improvement in last 10 trials
        recent_values = [t.value for t in study.trials[-10:] if t.value is not None]
        if len(recent_values) == 10 and max(recent_values) <= study.best_value:
            study.stop()

def objective_with_attributes(trial):
    # ... optimization code ...
    
    # Set custom attributes
    trial.set_user_attr('model_type', 'RandomForest')
    trial.set_user_attr('cv_std', scores.std())
    
    return accuracy

study.optimize(
    objective_with_attributes, 
    n_trials=100, 
    callbacks=[logging_callback, early_stopping_callback]
)
```

**Dashboard View:** Custom attributes visible in trial details.

### 6. Study Management

**What it does:** Persist studies, share across processes, manage multiple experiments

```python
# Create persistent study
study = optuna.create_study(
    study_name="my_experiment",
    storage="sqlite:///optuna.db",
    direction="maximize",
    load_if_exists=True
)

# Set study attributes
study.set_user_attr('dataset', 'adult_income')
study.set_user_attr('algorithm', 'RandomForest')

# Load existing study
loaded_study = optuna.load_study(
    study_name="my_experiment",
    storage="sqlite:///optuna.db"
)
```

**Dashboard View:** Browse all studies, compare across experiments.

## 🎛️ Dashboard Features

### Main Dashboard Views

1. **Study List**: Overview of all optimization studies
2. **Study Detail**: Detailed view of individual study
3. **Optimization History**: Trial progression over time
4. **Parameter Importance**: Which parameters matter most
5. **Parameter Relationships**: Correlations between parameters
6. **Parallel Coordinate Plot**: Multi-dimensional parameter visualization
7. **Slice Plot**: Parameter vs objective relationships

### Interactive Features

- **Filter trials** by status, parameter values, or objective ranges
- **Compare studies** side by side
- **Export results** to CSV or JSON
- **Real-time updates** as optimization runs
- **Multi-objective visualization** with Pareto fronts

## 🔧 Advanced Features

### 1. Distributed Optimization

```python
# Worker 1
study = optuna.load_study(
    study_name="distributed_experiment",
    storage="postgresql://user:pass@host:5432/optuna"
)
study.optimize(objective, n_trials=50)

# Worker 2 (running simultaneously)
study = optuna.load_study(
    study_name="distributed_experiment", 
    storage="postgresql://user:pass@host:5432/optuna"
)
study.optimize(objective, n_trials=50)
```

### 2. Integration with ML Frameworks

```python
# XGBoost Integration
import optuna_integration.xgboost as optuna_xgb

def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
    }
    
    model = optuna_xgb.train(
        params, dtrain, 
        num_boost_round=100,
        valid_sets=[dvalid],
        callbacks=[optuna_xgb.pruning_callback(trial, 'validation-logloss')]
    )
    
    return model.best_score

# LightGBM Integration  
import optuna_integration.lightgbm as optuna_lgb

model = optuna_lgb.train(
    params, train_set,
    valid_sets=[valid_set],
    callbacks=[optuna_lgb.pruning_callback(trial, 'valid_1-binary_logloss')]
)
```

## 📈 Real-World Examples

### Example 1: Image Classification

```python
# CNN hyperparameter optimization
def objective(trial):
    # Architecture parameters
    n_layers = trial.suggest_int('n_layers', 1, 3)
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    
    for i in range(n_layers):
        filters = trial.suggest_categorical(f'filters_{i}', [32, 64, 128])
        model.add(Conv2D(filters, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(trial.suggest_int('dense_units', 64, 512), activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    # Training parameters
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)
    return max(history.history['val_accuracy'])
```

### Example 2: Time Series Forecasting

```python
def objective(trial):
    # LSTM parameters
    lstm_units = trial.suggest_int('lstm_units', 32, 256)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    
    model = Sequential([
        LSTM(lstm_units, dropout=dropout, return_sequences=True),
        LSTM(lstm_units, dropout=dropout),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, verbose=0)
    
    predictions = model.predict(X_val)
    return mean_squared_error(y_val, predictions)
```

## 🎯 Best Practices

### 1. Parameter Space Design
- Use appropriate parameter types (`suggest_int`, `suggest_float`, `suggest_categorical`)
- Set reasonable bounds based on domain knowledge
- Use log scale for parameters that vary by orders of magnitude

### 2. Objective Function Design
- Keep evaluation fast for quick iteration
- Use cross-validation for robust estimates
- Consider multiple metrics with multi-objective optimization

### 3. Pruning Strategy
- Implement intermediate reporting for long-running trials
- Choose appropriate pruner based on your problem
- Balance exploration vs exploitation

### 4. Study Management
- Use descriptive study names
- Set user attributes for experiment tracking
- Use persistent storage for important experiments

## 🚀 Running the Complete Demo

```bash
# 1. Populate dashboard with all features
python scripts/showcase_all_optuna_features.py

# 2. Start dashboard
python scripts/start_dashboard.py

# 3. Open browser to http://localhost:8080

# 4. Explore all the studies and visualizations!
```

## 📚 Additional Resources

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Optuna Examples](https://github.com/optuna/optuna-examples)
- [Dashboard Documentation](https://optuna-dashboard.readthedocs.io/)
- [Integration Packages](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)

---

**🎉 You now have a complete understanding of Optuna's capabilities! The dashboard provides an interactive way to explore and understand your optimization results.**
