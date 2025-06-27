#!/usr/bin/env python3
"""
Quick demo script to create working optimization studies
"""

import optuna
import os
import time
from pathlib import Path

def create_demo_studies():
    """Create several demo optimization studies"""
    
    # Create studies directory
    studies_dir = Path("studies")
    studies_dir.mkdir(exist_ok=True)
    
    print("Creating demo optimization studies...")

    # Demo 1: Simple 2D optimization
    print("\n1. Creating 2D optimization study...")
    storage1 = f"sqlite:///studies/demo_2d.db"
    study1 = optuna.create_study(
        study_name="2d_optimization",
        storage=storage1,
        direction="minimize",
        load_if_exists=True
    )
    
    def objective_2d(trial):
        x = trial.suggest_float('x', -5, 5)
        y = trial.suggest_float('y', -5, 5)
        return x**2 + y**2  # Minimize sum of squares
    
    study1.optimize(objective_2d, n_trials=50)
    print(f"Created study with {len(study1.trials)} trials")
    print(f"   Best value: {study1.best_value:.4f}")
    
    # Demo 2: Machine Learning hyperparameters
    print("\n2. Creating ML hyperparameter study...")
    storage2 = f"sqlite:///studies/demo_ml.db"
    study2 = optuna.create_study(
        study_name="ml_hyperparams",
        storage=storage2,
        direction="maximize",
        load_if_exists=True
    )
    
    def objective_ml(trial):
        # Simulate ML model hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        
        # Simulate accuracy (random but realistic)
        import random
        random.seed(n_estimators + max_depth + int(learning_rate * 100))
        base_accuracy = 0.85
        noise = random.uniform(-0.1, 0.1)
        
        # Better hyperparams generally give better results
        param_bonus = (n_estimators / 100) * 0.05 + (max_depth / 10) * 0.03
        
        return min(0.99, base_accuracy + param_bonus + noise)
    
    study2.optimize(objective_ml, n_trials=30)
    print(f"Created study with {len(study2.trials)} trials")
    print(f"   Best accuracy: {study2.best_value:.4f}")
    
    # Demo 3: Multi-objective optimization
    print("\n3. Creating multi-objective study...")
    storage3 = f"sqlite:///studies/demo_multi.db"
    study3 = optuna.create_study(
        study_name="multi_objective",
        storage=storage3,
        directions=["minimize", "maximize"],  # Two objectives
        load_if_exists=True
    )
    
    def objective_multi(trial):
        x = trial.suggest_float('x', 0, 5)
        y = trial.suggest_float('y', 0, 5)
        
        # Objective 1: minimize distance from origin
        obj1 = x**2 + y**2
        
        # Objective 2: maximize product
        obj2 = x * y
        
        return obj1, obj2
    
    study3.optimize(objective_multi, n_trials=40)
    print(f"Created study with {len(study3.trials)} trials")
    
    print("\nDemo studies created successfully!")
    print("\nNext steps:")
    print("1. Run: python start_simple.py")
    print("2. Open: http://localhost:8501")
    print("3. Click 'Launch Dashboard' in the sidebar")
    print("4. Explore your optimization studies!")

if __name__ == "__main__":
    create_demo_studies()
