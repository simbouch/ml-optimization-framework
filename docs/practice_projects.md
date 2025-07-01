# 🎯 Optuna Practice Projects Guide
*Hands-on Projects for Mastering Hyperparameter Optimization*

## 📚 Overview

This guide provides 6 progressive practice projects designed to help you master Optuna through hands-on experience. Each project builds upon previous knowledge and introduces new concepts.

## 🎓 Learning Progression

```
Beginner → Intermediate → Advanced
   ↓            ↓           ↓
Project 1    Project 3   Project 5
Project 2    Project 4   Project 6
```

## 🚀 Project 1: Personal Dataset Optimization
**Difficulty**: Beginner  
**Time**: 2-3 hours  
**Goal**: Apply Optuna to your own dataset

### 📋 Requirements
- Your own dataset (CSV file with features and target)
- Basic understanding of scikit-learn
- Completed the basic tutorial example

### 🎯 Objectives
1. Load and prepare your own dataset
2. Implement basic hyperparameter optimization
3. Compare optimized vs default model performance
4. Understand parameter importance

### 📝 Step-by-Step Instructions

#### Step 1: Dataset Preparation
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Basic preprocessing
# TODO: Handle missing values, encode categorical variables
X = df.drop('target_column', axis=1)
y = df['target_column']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

#### Step 2: Define Objective Function
```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # Evaluate with cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    return scores.mean()
```

#### Step 3: Run Optimization
```python
# Create study
study = optuna.create_study(
    direction='maximize',
    study_name='my_first_optimization'
)

# Optimize
study.optimize(objective, n_trials=50)

# Results
print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best parameters: {study.best_params}")
```

#### Step 4: Compare Results
```python
# Default model
default_model = RandomForestClassifier(random_state=42)
default_scores = cross_val_score(default_model, X_train, y_train, cv=3)

# Optimized model
optimized_model = RandomForestClassifier(**study.best_params, random_state=42)
optimized_scores = cross_val_score(optimized_model, X_train, y_train, cv=3)

print(f"Default accuracy: {default_scores.mean():.4f} ± {default_scores.std():.4f}")
print(f"Optimized accuracy: {optimized_scores.mean():.4f} ± {optimized_scores.std():.4f}")
print(f"Improvement: {optimized_scores.mean() - default_scores.mean():.4f}")
```

### ✅ Success Criteria
- [ ] Successfully load and preprocess your dataset
- [ ] Implement working objective function
- [ ] Run optimization for 50+ trials
- [ ] Show improvement over default parameters
- [ ] Understand which parameters matter most

### 💡 Extension Ideas
- Try different models (SVM, Gradient Boosting)
- Experiment with different parameter ranges
- Add feature selection to the optimization
- Use different evaluation metrics

---

## 🚀 Project 2: Multi-Model Comparison
**Difficulty**: Beginner-Intermediate  
**Time**: 3-4 hours  
**Goal**: Compare different algorithms with optimization

### 📋 Requirements
- Completed Project 1
- Understanding of different ML algorithms
- Knowledge of model evaluation metrics

### 🎯 Objectives
1. Optimize multiple different algorithms
2. Compare final performance across models
3. Analyze parameter importance for each model
4. Create comprehensive comparison report

### 📝 Implementation Template
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def create_model_optimizers():
    """Define optimization functions for different models"""
    
    def rf_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        }
        model = RandomForestClassifier(**params, random_state=42)
        return cross_val_score(model, X_train, y_train, cv=3).mean()
    
    def gb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
        }
        model = GradientBoostingClassifier(**params, random_state=42)
        return cross_val_score(model, X_train, y_train, cv=3).mean()
    
    def svm_objective(trial):
        params = {
            'C': trial.suggest_float('C', 0.1, 100, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly']),
        }
        model = SVC(**params, random_state=42)
        return cross_val_score(model, X_train, y_train, cv=3).mean()
    
    return {
        'RandomForest': rf_objective,
        'GradientBoosting': gb_objective,
        'SVM': svm_objective
    }

# Run optimization for each model
models = create_model_optimizers()
results = {}

for model_name, objective in models.items():
    print(f"Optimizing {model_name}...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    results[model_name] = {
        'best_score': study.best_value,
        'best_params': study.best_params,
        'study': study
    }
    
    print(f"{model_name} best score: {study.best_value:.4f}")
```

### ✅ Success Criteria
- [ ] Optimize at least 3 different algorithms
- [ ] Compare final performance across all models
- [ ] Identify best performing model for your dataset
- [ ] Analyze parameter importance for each model
- [ ] Create visualization of results

---

## 🚀 Project 3: Advanced Pruning Implementation
**Difficulty**: Intermediate  
**Time**: 4-5 hours  
**Goal**: Implement efficient optimization with pruning

### 📋 Requirements
- Completed Projects 1-2
- Understanding of iterative training processes
- Knowledge of early stopping concepts

### 🎯 Objectives
1. Implement pruning for computationally expensive models
2. Compare efficiency: pruning vs no pruning
3. Understand different pruning strategies
4. Measure time savings while maintaining quality

### 📝 Implementation Guide
```python
import time
from sklearn.neural_network import MLPClassifier

def pruning_objective(trial):
    """Objective function with intermediate reporting and pruning"""
    
    # Suggest hyperparameters
    hidden_layer_sizes = trial.suggest_categorical(
        'hidden_layer_sizes', 
        [(50,), (100,), (50, 50), (100, 50), (100, 100)]
    )
    learning_rate_init = trial.suggest_float('learning_rate_init', 0.001, 0.1, log=True)
    alpha = trial.suggest_float('alpha', 0.0001, 0.01, log=True)
    
    # Create model with warm_start for incremental training
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        max_iter=1,  # Train one iteration at a time
        warm_start=True,
        random_state=42
    )
    
    # Incremental training with intermediate reporting
    for epoch in range(100):
        model.fit(X_train, y_train)
        
        # Evaluate current performance
        score = model.score(X_val, y_val)
        
        # Report intermediate value
        trial.report(score, epoch)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return score

# Compare with and without pruning
def compare_pruning_efficiency():
    """Compare optimization with and without pruning"""
    
    # Without pruning
    start_time = time.time()
    study_no_pruning = optuna.create_study(direction='maximize')
    study_no_pruning.optimize(pruning_objective, n_trials=20)
    time_no_pruning = time.time() - start_time
    
    # With pruning
    start_time = time.time()
    study_with_pruning = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    study_with_pruning.optimize(pruning_objective, n_trials=20)
    time_with_pruning = time.time() - start_time
    
    print(f"Without pruning: {time_no_pruning:.2f}s, Best: {study_no_pruning.best_value:.4f}")
    print(f"With pruning: {time_with_pruning:.2f}s, Best: {study_with_pruning.best_value:.4f}")
    print(f"Time saved: {((time_no_pruning - time_with_pruning) / time_no_pruning * 100):.1f}%")
```

### ✅ Success Criteria
- [ ] Implement working pruning with intermediate reporting
- [ ] Compare efficiency with and without pruning
- [ ] Achieve significant time savings (>30%)
- [ ] Maintain similar optimization quality
- [ ] Test different pruning strategies

---

## 🚀 Project 4: Multi-Objective Real Problem
**Difficulty**: Intermediate-Advanced  
**Time**: 5-6 hours  
**Goal**: Solve real trade-off optimization problem

### 📋 Requirements
- Completed Projects 1-3
- Understanding of business constraints
- Knowledge of Pareto optimization

### 🎯 Objectives
1. Define realistic conflicting objectives
2. Implement multi-objective optimization
3. Analyze Pareto front solutions
4. Choose final solution based on constraints

### 📝 Example Implementation
```python
def multi_objective_real_problem(trial):
    """
    Real-world scenario: Optimize model for production deployment
    Objectives:
    1. Maximize accuracy
    2. Minimize prediction time
    3. Minimize model size
    """
    
    # Model hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 10, 500)
    max_depth = trial.suggest_int('max_depth', 3, 30)
    
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
    
    # Objective 2: Prediction time (minimize - negative for maximization)
    start_time = time.time()
    _ = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Objective 3: Model complexity (minimize - negative for maximization)
    model_complexity = n_estimators * max_depth
    
    return accuracy, -prediction_time, -model_complexity

# Run multi-objective optimization
study = optuna.create_study(directions=['maximize', 'maximize', 'maximize'])
study.optimize(multi_objective_real_problem, n_trials=100)

# Analyze Pareto front
pareto_front = study.best_trials
print(f"Found {len(pareto_front)} Pareto optimal solutions")

# Choose solution based on constraints
def choose_solution(trials, max_time=0.1, max_complexity=1000):
    """Choose solution based on business constraints"""
    
    valid_solutions = []
    for trial in trials:
        accuracy, neg_time, neg_complexity = trial.values
        time_taken = -neg_time
        complexity = -neg_complexity
        
        if time_taken <= max_time and complexity <= max_complexity:
            valid_solutions.append((trial, accuracy))
    
    if valid_solutions:
        best_trial, best_accuracy = max(valid_solutions, key=lambda x: x[1])
        return best_trial
    else:
        return None

chosen_solution = choose_solution(pareto_front)
if chosen_solution:
    print(f"Chosen solution: {chosen_solution.params}")
    print(f"Accuracy: {chosen_solution.values[0]:.4f}")
    print(f"Prediction time: {-chosen_solution.values[1]:.4f}s")
    print(f"Model complexity: {-chosen_solution.values[2]}")
```

### ✅ Success Criteria
- [ ] Define at least 2 conflicting objectives
- [ ] Successfully run multi-objective optimization
- [ ] Analyze and visualize Pareto front
- [ ] Choose final solution based on realistic constraints
- [ ] Understand trade-offs between objectives

---

## 🚀 Project 5: Production Pipeline Integration
**Difficulty**: Advanced  
**Time**: 6-8 hours  
**Goal**: Integrate Optuna into complete ML pipeline

### 📋 Requirements
- Completed Projects 1-4
- Understanding of ML pipelines
- Knowledge of model deployment concepts

### 🎯 Objectives
1. Create end-to-end ML pipeline with optimization
2. Optimize preprocessing + model parameters
3. Implement study persistence and monitoring
4. Deploy optimized model for production

### ✅ Success Criteria
- [ ] Build complete pipeline with preprocessing optimization
- [ ] Implement study persistence and resumption
- [ ] Create model deployment script
- [ ] Add monitoring and logging
- [ ] Document the entire process

---

## 🚀 Project 6: Custom Sampler/Pruner (Advanced)
**Difficulty**: Expert  
**Time**: 8-10 hours  
**Goal**: Implement custom optimization strategy

### 📋 Requirements
- Completed Projects 1-5
- Deep understanding of optimization algorithms
- Advanced Python programming skills

### 🎯 Objectives
1. Study existing sampler/pruner implementations
2. Identify improvement opportunities
3. Implement custom optimization strategy
4. Benchmark against standard approaches

### ✅ Success Criteria
- [ ] Implement working custom sampler or pruner
- [ ] Demonstrate improvement over standard methods
- [ ] Provide comprehensive benchmarking
- [ ] Document when to use your custom approach
- [ ] Consider contributing to Optuna open source

---

## 📊 Progress Tracking

### Project Completion Checklist
- [ ] Project 1: Personal Dataset Optimization
- [ ] Project 2: Multi-Model Comparison  
- [ ] Project 3: Advanced Pruning Implementation
- [ ] Project 4: Multi-Objective Real Problem
- [ ] Project 5: Production Pipeline Integration
- [ ] Project 6: Custom Sampler/Pruner

### Skills Acquired
After completing all projects, you will have:
- ✅ Mastered basic Optuna usage
- ✅ Understood different optimization strategies
- ✅ Implemented efficient optimization with pruning
- ✅ Solved multi-objective optimization problems
- ✅ Built production-ready ML pipelines
- ✅ Created custom optimization algorithms

## 🎓 Next Steps

1. **Apply to Work**: Use Optuna in your daily ML tasks
2. **Teach Others**: Share knowledge with colleagues
3. **Contribute**: Consider contributing to Optuna open source
4. **Research**: Explore cutting-edge optimization techniques
5. **Specialize**: Focus on domain-specific optimization problems

## 📚 Additional Resources

- [Optuna Examples Repository](https://github.com/optuna/optuna-examples)
- [Hyperparameter Optimization Research Papers](https://arxiv.org/search/?query=hyperparameter+optimization)
- [MLOps Best Practices](https://ml-ops.org/)
- [Production ML Systems](https://developers.google.com/machine-learning/guides/rules-of-ml)

---

**Happy Learning! 🎯**
