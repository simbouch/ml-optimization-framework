#!/usr/bin/env python3
"""
Create Unified Demo Database
Consolidates all optimization examples into a single database for better dashboard viewing
"""

import optuna
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from pathlib import Path
import time

def safe_print(text):
    """Print text safely on all platforms"""
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)

def create_unified_database():
    """Create a single database with multiple studies for better dashboard viewing"""
    
    # Ensure directories exist
    Path("studies").mkdir(exist_ok=True)
    
    # Single database for all studies
    storage_url = "sqlite:///studies/unified_demo.db"
    
    # Generate datasets
    X_class, y_class = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, n_informative=15, random_state=42)
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    studies_created = []
    
    # 1. Random Forest Classification with TPE
    safe_print("1. Creating Random Forest Classification (TPE)...")
    study1 = optuna.create_study(
        study_name="RandomForest_Classification_TPE",
        storage=storage_url,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True
    )
    
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
        
        scores = cross_val_score(model, X_train_c, y_train_c, cv=3, scoring='accuracy')
        return scores.mean()
    
    study1.optimize(rf_classification_objective, n_trials=30)
    studies_created.append("RandomForest_Classification_TPE")
    safe_print(f"   Best accuracy: {study1.best_value:.4f}")
    
    # 2. Gradient Boosting Regression with Random Sampler
    safe_print("2. Creating Gradient Boosting Regression (Random)...")
    study2 = optuna.create_study(
        study_name="GradientBoosting_Regression_Random",
        storage=storage_url,
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=42),
        load_if_exists=True
    )

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

        model.fit(X_train_r, y_train_r)
        predictions = model.predict(X_test_r)
        mse = np.mean((y_test_r - predictions) ** 2)
        return mse

    study2.optimize(gb_regression_objective, n_trials=25)
    studies_created.append("GradientBoosting_Regression_Random")
    safe_print(f"   Best MSE: {study2.best_value:.4f}")
    
    # 3. SVM Classification with Pruning
    safe_print("3. Creating SVM Classification (Pruning)...")
    study3 = optuna.create_study(
        study_name="SVM_Classification_Pruning",
        storage=storage_url,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        load_if_exists=True
    )
    
    def svm_classification_objective(trial):
        C = trial.suggest_float('C', 1e-3, 1e3, log=True)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        
        model = SVC(C=C, gamma=gamma, random_state=42)
        
        # Simulate intermediate reporting for pruning
        for step in range(3):
            partial_X = X_train_c[:len(X_train_c)//(3-step)]
            partial_y = y_train_c[:len(y_train_c)//(3-step)]
            
            model.fit(partial_X, partial_y)
            score = model.score(X_test_c, y_test_c)
            
            trial.report(score, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return score
    
    study3.optimize(svm_classification_objective, n_trials=20)
    studies_created.append("SVM_Classification_Pruning")
    safe_print(f"   Best accuracy: {study3.best_value:.4f}")
    
    # 4. Multi-objective Optimization
    safe_print("4. Creating Multi-objective Optimization...")
    study4 = optuna.create_study(
        study_name="MultiObjective_Accuracy_vs_Complexity",
        storage=storage_url,
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True
    )
    
    def multi_objective_function(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
        # Objective 1: Accuracy (to maximize)
        scores = cross_val_score(model, X_train_c, y_train_c, cv=3, scoring='accuracy')
        accuracy = scores.mean()
        
        # Objective 2: Model complexity (to minimize)
        complexity = n_estimators * max_depth
        
        return accuracy, complexity
    
    study4.optimize(multi_objective_function, n_trials=25)
    studies_created.append("MultiObjective_Accuracy_vs_Complexity")
    safe_print(f"   Found {len(study4.best_trials)} Pareto optimal solutions")
    
    # 5. Logistic Regression Comparison
    safe_print("5. Creating Logistic Regression Comparison...")
    study5 = optuna.create_study(
        study_name="LogisticRegression_Comparison",
        storage=storage_url,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True
    )
    
    def logistic_regression_objective(trial):
        C = trial.suggest_float('C', 1e-4, 1e2, log=True)
        solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        
        model = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=42
        )
        
        scores = cross_val_score(model, X_train_c, y_train_c, cv=3, scoring='accuracy')
        return scores.mean()
    
    study5.optimize(logistic_regression_objective, n_trials=20)
    studies_created.append("LogisticRegression_Comparison")
    safe_print(f"   Best accuracy: {study5.best_value:.4f}")
    
    # 6. Random Forest Regression
    safe_print("6. Creating Random Forest Regression...")
    study6 = optuna.create_study(
        study_name="RandomForest_Regression",
        storage=storage_url,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True
    )
    
    def rf_regression_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        
        model.fit(X_train_r, y_train_r)
        predictions = model.predict(X_test_r)
        mse = np.mean((y_test_r - predictions) ** 2)
        return mse
    
    study6.optimize(rf_regression_objective, n_trials=25)
    studies_created.append("RandomForest_Regression")
    safe_print(f"   Best MSE: {study6.best_value:.4f}")
    
    return studies_created, storage_url

def main():
    """Main function"""
    safe_print("Creating Unified Demo Database for Optuna Dashboard")
    safe_print("=" * 60)
    
    try:
        studies_created, storage_url = create_unified_database()
        
        safe_print("\n" + "=" * 60)
        safe_print("SUCCESS! Unified Demo Database Created")
        safe_print(f"\nDatabase: {storage_url}")
        safe_print(f"Studies created: {len(studies_created)}")
        
        for i, study_name in enumerate(studies_created, 1):
            safe_print(f"  {i}. {study_name}")
        
        safe_print(f"\nTo view all studies in dashboard:")
        safe_print(f"optuna-dashboard {storage_url} --host 0.0.0.0 --port 8080")
        safe_print(f"\nThen open: http://localhost:8080")
        safe_print("=" * 60)
        
        return True
        
    except Exception as e:
        safe_print(f"Error creating unified database: {e}")
        return False

if __name__ == "__main__":
    main()
