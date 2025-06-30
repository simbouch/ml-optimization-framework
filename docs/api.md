# 🔧 API Reference

## Core Components

### create_unified_demo.py

Main script that creates all 6 optimization studies in a single database.

#### Functions

##### `create_unified_database()`
Creates a unified database with 6 different optimization studies.

**Returns**: 
- `studies_created` (list): Names of created studies
- `storage_url` (str): Database connection string

**Studies Created**:
1. RandomForest_Classification_TPE
2. GradientBoosting_Regression_Random
3. SVM_Classification_Pruning
4. MultiObjective_Accuracy_vs_Complexity
5. LogisticRegression_Comparison
6. RandomForest_Regression

##### `safe_print(text)`
Platform-safe printing function that handles Unicode issues.

**Parameters**:
- `text` (str): Text to print safely

## Docker Configuration

### docker-compose.yml

Single service configuration for the ML optimization framework.

#### Service: ml-optimization

**Configuration**:
- **Build**: Uses local Dockerfile
- **Ports**: 8080:8080 (Optuna Dashboard)
- **Volumes**: 
  - `studies_data:/app/studies` (persistent study storage)
  - `logs_data:/app/logs` (persistent logs)
- **Command**: Creates demos then starts dashboard
- **Health Check**: Verifies dashboard accessibility

### Dockerfile

Multi-stage container build for the optimization framework.

#### Build Process
1. **Base**: Python 3.10-slim
2. **Dependencies**: Install from requirements-minimal.txt
3. **Code**: Copy source and demo scripts
4. **Working Directory**: /app
5. **Command**: Run unified demo + start dashboard

## Database Schema

### SQLite Storage

Studies are stored in SQLite databases using Optuna's built-in storage.

#### Database Location
- **Local**: `studies/unified_demo.db`
- **Docker**: `/app/studies/unified_demo.db`

#### Study Structure
Each study contains:
- **Study metadata**: Name, direction, sampler info
- **Trials**: Parameter values, objective values, state
- **System attributes**: Optuna internal data

## Optimization Studies API

### Study 1: RandomForest Classification

```python
def rf_classification_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    return scores.mean()
```

### Study 2: Gradient Boosting Regression

```python
def gb_regression_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = np.mean((y_test - predictions) ** 2)
    return mse
```

### Study 3: SVM with Pruning

```python
def svm_classification_objective(trial):
    C = trial.suggest_float('C', 1e-3, 1e3, log=True)
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    
    model = SVC(C=C, gamma=gamma, random_state=42)
    
    # Intermediate reporting for pruning
    for step in range(3):
        partial_X = X_train[:len(X_train)//(3-step)]
        partial_y = y_train[:len(y_train)//(3-step)]
        
        model.fit(partial_X, partial_y)
        score = model.score(X_test, y_test)
        
        trial.report(score, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return score
```

### Study 4: Multi-objective

```python
def multi_objective_function(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    # Objective 1: Accuracy (maximize)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    accuracy = scores.mean()
    
    # Objective 2: Model complexity (minimize)
    complexity = n_estimators * max_depth
    
    return accuracy, complexity
```

## Customization Guide

### Adding New Studies

1. **Define Objective Function**:
```python
def my_objective(trial):
    # Define parameters
    param1 = trial.suggest_float('param1', 0.1, 1.0)
    param2 = trial.suggest_int('param2', 1, 100)
    
    # Create and train model
    model = MyModel(param1=param1, param2=param2)
    score = evaluate_model(model)
    
    return score
```

2. **Create Study**:
```python
study = optuna.create_study(
    study_name="my_optimization",
    storage=storage_url,
    direction="maximize",  # or "minimize"
    sampler=optuna.samplers.TPESampler(seed=42),
    load_if_exists=True
)

study.optimize(my_objective, n_trials=50)
```

### Modifying Existing Studies

1. **Change Parameters**:
   - Modify parameter ranges in objective functions
   - Add new parameters with `trial.suggest_*`
   - Remove parameters by commenting out

2. **Change Models**:
   - Replace model classes
   - Modify model parameters
   - Change evaluation metrics

3. **Change Samplers**:
   - TPE: `optuna.samplers.TPESampler()`
   - Random: `optuna.samplers.RandomSampler()`
   - CMA-ES: `optuna.samplers.CmaEsSampler()`

### Environment Variables

Configure behavior through environment variables:

```bash
# Database location
OPTUNA_DB_URL=sqlite:///studies/custom.db

# Dashboard port
DASHBOARD_PORT=8080

# Log level
LOG_LEVEL=INFO
```

## Dependencies

### Core Requirements

```
optuna>=3.4.0          # Optimization framework
scikit-learn>=1.3.0    # Machine learning models
numpy>=1.24.0          # Numerical computing
pandas>=2.0.0          # Data manipulation
plotly>=5.15.0         # Visualization
optuna-dashboard>=0.13.0  # Web dashboard
# Removed XGBoost to avoid large download timeouts
# Can be added back when needed: xgboost>=1.7.0
loguru>=0.7.0          # Logging
pytest>=7.0.0          # Testing
```

### Optional Dependencies

```
jupyter                # Notebook interface
matplotlib            # Additional plotting
seaborn               # Statistical visualization
```

## Error Handling

### Common Issues

1. **Import Errors**: Check package installation
2. **Database Errors**: Verify SQLite permissions
3. **Port Conflicts**: Change dashboard port
4. **Memory Issues**: Reduce trial counts

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

View detailed trial information:
```python
study = optuna.load_study(study_name="study_name", storage=storage_url)
print(study.trials_dataframe())
```

## Performance Optimization

### Trial Parallelization

```python
# Run multiple trials in parallel
study.optimize(objective, n_trials=100, n_jobs=4)
```

### Memory Management

```python
# Limit trial history
study.optimize(objective, n_trials=100, gc_after_trial=True)
```

### Database Optimization

```python
# Use connection pooling
storage = optuna.storages.RDBStorage(
    url="sqlite:///studies/demo.db",
    engine_kwargs={"pool_pre_ping": True}
)
```

## Testing

### Unit Tests

Run framework tests:
```bash
pytest tests/
```

### Integration Tests

Test full workflow:
```bash
python create_unified_demo.py
optuna-dashboard sqlite:///studies/unified_demo.db --port 8080
```

### Performance Tests

Benchmark optimization:
```bash
python -m timeit "import create_unified_demo; create_unified_demo.main()"
```
