#!/usr/bin/env python3
"""
Advanced Multi-Objective Optimization Example
Demonstrates complex multi-objective optimization with Pareto front analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import optuna
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path

def multi_objective_ml_optimization():
    """
    Multi-objective optimization example:
    - Maximize accuracy
    - Minimize model complexity
    - Minimize training time
    """
    print("🎯 Multi-Objective ML Optimization")
    print("=" * 50)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=2000, 
        n_features=20, 
        n_informative=15,
        n_redundant=5, 
        n_classes=3, 
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def multi_objective_function(trial):
        """
        Objective function with multiple conflicting goals:
        1. Maximize accuracy
        2. Minimize model complexity (number of parameters)
        3. Minimize overfitting (difference between train and validation accuracy)
        """
        # Suggest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        
        # Create model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        
        # Objective 1: Accuracy (to maximize)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        accuracy = cv_scores.mean()
        
        # Objective 2: Model complexity (to minimize)
        # Approximate number of parameters in the forest
        model_complexity = n_estimators * max_depth * X.shape[1]
        
        # Objective 3: Overfitting measure (to minimize)
        model.fit(X_train, y_train)
        train_accuracy = model.score(X_train, y_train)
        overfitting = train_accuracy - accuracy  # Difference between train and CV accuracy
        
        return accuracy, -model_complexity, -overfitting  # Maximize accuracy, minimize others
    
    # Create multi-objective study
    study = optuna.create_study(
        study_name="multi_objective_ml",
        storage="sqlite:///studies/multi_objective_advanced.db",
        directions=["maximize", "maximize", "maximize"],  # All objectives to maximize (negated minimization)
        load_if_exists=True
    )
    
    print("🚀 Starting multi-objective optimization...")
    study.optimize(multi_objective_function, n_trials=100)
    
    # Analyze results
    print(f"\n📊 Optimization completed with {len(study.trials)} trials")
    print(f"🏆 Found {len(study.best_trials)} Pareto optimal solutions")
    
    # Display Pareto front
    print("\n🎯 Pareto Optimal Solutions:")
    print("-" * 80)
    print(f"{'Trial':<8} {'Accuracy':<10} {'Complexity':<12} {'Overfitting':<12} {'Parameters'}")
    print("-" * 80)
    
    for i, trial in enumerate(study.best_trials[:10]):  # Show top 10
        accuracy = trial.values[0]
        complexity = -trial.values[1]
        overfitting = -trial.values[2]
        params = f"n_est={trial.params['n_estimators']}, depth={trial.params['max_depth']}"
        print(f"{trial.number:<8} {accuracy:<10.4f} {complexity:<12.0f} {overfitting:<12.4f} {params}")
    
    return study

def portfolio_optimization_example():
    """
    Financial portfolio optimization example with multiple objectives
    """
    print("\n🎯 Portfolio Optimization Example")
    print("=" * 50)
    
    # Simulate asset returns (normally this would be real financial data)
    np.random.seed(42)
    n_assets = 10
    n_periods = 252  # Trading days in a year
    
    # Generate correlated asset returns
    correlation_matrix = np.random.rand(n_assets, n_assets)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1)
    
    returns = np.random.multivariate_normal(
        mean=np.random.uniform(0.05, 0.15, n_assets),  # Annual returns 5-15%
        cov=correlation_matrix * 0.01,  # Covariance matrix
        size=n_periods
    )
    
    def portfolio_objective(trial):
        """
        Multi-objective portfolio optimization:
        1. Maximize expected return
        2. Minimize risk (volatility)
        3. Minimize maximum drawdown
        """
        # Suggest portfolio weights (must sum to 1)
        weights = []
        remaining_weight = 1.0
        
        for i in range(n_assets - 1):
            weight = trial.suggest_float(f'weight_{i}', 0, remaining_weight)
            weights.append(weight)
            remaining_weight -= weight
        weights.append(remaining_weight)  # Last weight is determined
        
        weights = np.array(weights)
        
        # Calculate portfolio metrics
        portfolio_returns = np.dot(returns, weights)
        
        # Objective 1: Expected return (to maximize)
        expected_return = np.mean(portfolio_returns)
        
        # Objective 2: Risk/Volatility (to minimize)
        volatility = np.std(portfolio_returns)
        
        # Objective 3: Maximum drawdown (to minimize)
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return expected_return, -volatility, -max_drawdown  # Maximize return, minimize risk and drawdown
    
    # Create study
    study = optuna.create_study(
        study_name="portfolio_optimization",
        storage="sqlite:///studies/portfolio_multi_objective.db",
        directions=["maximize", "maximize", "maximize"],
        load_if_exists=True
    )
    
    print("🚀 Starting portfolio optimization...")
    study.optimize(portfolio_objective, n_trials=200)
    
    print(f"\n📊 Portfolio optimization completed")
    print(f"🏆 Found {len(study.best_trials)} Pareto optimal portfolios")
    
    return study

def hyperparameter_vs_performance_tradeoff():
    """
    Example showing trade-off between model performance and computational cost
    """
    print("\n🎯 Performance vs Computational Cost Trade-off")
    print("=" * 50)
    
    # Load data
    X, y = make_classification(n_samples=5000, n_features=50, n_informative=30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def performance_cost_objective(trial):
        """
        Trade-off between:
        1. Model performance (accuracy)
        2. Training time
        3. Memory usage (model size)
        """
        import time
        
        # Hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 10, 500)
        max_depth = trial.suggest_int('max_depth', 3, 30)
        
        # Measure training time
        start_time = time.time()
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=1  # Single thread for consistent timing
        )
        
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Performance
        accuracy = model.score(X_test, y_test)
        
        # Model size (approximate)
        model_size = n_estimators * max_depth * X.shape[1]  # Proxy for memory usage
        
        return accuracy, -training_time, -model_size  # Maximize accuracy, minimize time and size
    
    # Create study
    study = optuna.create_study(
        study_name="performance_cost_tradeoff",
        storage="sqlite:///studies/performance_cost_tradeoff.db",
        directions=["maximize", "maximize", "maximize"],
        load_if_exists=True
    )
    
    print("🚀 Starting performance vs cost optimization...")
    study.optimize(performance_cost_objective, n_trials=50)
    
    print(f"\n📊 Trade-off analysis completed")
    print(f"🏆 Found {len(study.best_trials)} Pareto optimal solutions")
    
    return study

def main():
    """Run all advanced multi-objective examples"""
    print("🎯 Advanced Multi-Objective Optimization Examples")
    print("=" * 60)
    
    # Ensure directories exist
    Path("studies").mkdir(exist_ok=True)
    
    try:
        # Run examples
        ml_study = multi_objective_ml_optimization()
        portfolio_study = portfolio_optimization_example()
        tradeoff_study = hyperparameter_vs_performance_tradeoff()
        
        print("\n" + "=" * 60)
        print("🎉 All Advanced Examples Completed!")
        print("\n📊 Studies Created:")
        print("  1. multi_objective_advanced.db - ML multi-objective optimization")
        print("  2. portfolio_multi_objective.db - Portfolio optimization")
        print("  3. performance_cost_tradeoff.db - Performance vs cost trade-off")
        
        print("\n📍 View Results:")
        print("  🎨 Streamlit App: http://localhost:8501")
        print("  📊 Optuna Dashboard: http://localhost:8080")
        
        print("\n💡 Key Learnings:")
        print("  - Multi-objective optimization finds trade-off solutions")
        print("  - Pareto front shows optimal compromises")
        print("  - Real-world problems often have conflicting objectives")
        print("  - Optuna handles complex multi-objective scenarios elegantly")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
