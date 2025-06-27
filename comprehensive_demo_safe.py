#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows-Safe Comprehensive Optuna Demo
Simplified version that works reliably on Windows
"""

import optuna
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import time

def safe_print(text):
    """Print text safely on Windows"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback to ASCII-safe version
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)

def create_demo_studies():
    """Create comprehensive demo studies"""
    safe_print("Creating comprehensive Optuna demonstration...")
    safe_print("=" * 50)
    
    # Ensure directories exist
    Path("studies").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                              n_redundant=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    studies_created = []
    
    # 1. TPE Sampler Demo
    safe_print("\n1. Creating TPE Sampler Demo...")
    try:
        study_tpe = optuna.create_study(
            study_name="tpe_demo",
            storage="sqlite:///studies/demo_tpe.db",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            load_if_exists=True
        )
        
        def objective_tpe(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 100)
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
        
        study_tpe.optimize(objective_tpe, n_trials=20)
        studies_created.append("demo_tpe.db")
        safe_print(f"   Best accuracy: {study_tpe.best_value:.4f}")
        
    except Exception as e:
        safe_print(f"   Error creating TPE demo: {e}")
    
    # 2. Random Sampler Demo
    safe_print("\n2. Creating Random Sampler Demo...")
    try:
        study_random = optuna.create_study(
            study_name="random_demo",
            storage="sqlite:///studies/demo_random.db",
            direction="maximize",
            sampler=optuna.samplers.RandomSampler(seed=42),
            load_if_exists=True
        )
        
        study_random.optimize(objective_tpe, n_trials=20)
        studies_created.append("demo_random.db")
        safe_print(f"   Best accuracy: {study_random.best_value:.4f}")
        
    except Exception as e:
        safe_print(f"   Error creating Random demo: {e}")
    
    # 3. CMA-ES Sampler Demo
    safe_print("\n3. Creating CMA-ES Sampler Demo...")
    try:
        study_cmaes = optuna.create_study(
            study_name="cmaes_demo",
            storage="sqlite:///studies/demo_cmaes.db",
            direction="maximize",
            sampler=optuna.samplers.CmaEsSampler(seed=42),
            load_if_exists=True
        )
        
        def objective_cmaes(trial):
            # CMA-ES works better with continuous parameters
            max_depth = trial.suggest_float('max_depth', 3.0, 20.0)
            min_samples_split = trial.suggest_float('min_samples_split', 2.0, 20.0)
            
            model = RandomForestClassifier(
                n_estimators=50,  # Fixed for faster execution
                max_depth=int(max_depth),
                min_samples_split=int(min_samples_split),
                random_state=42
            )
            
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            return scores.mean()
        
        study_cmaes.optimize(objective_cmaes, n_trials=15)
        studies_created.append("demo_cmaes.db")
        safe_print(f"   Best accuracy: {study_cmaes.best_value:.4f}")
        
    except Exception as e:
        safe_print(f"   Error creating CMA-ES demo: {e}")
    
    # 4. Pruning Demo
    safe_print("\n4. Creating Pruning Demo...")
    try:
        study_pruning = optuna.create_study(
            study_name="pruning_demo",
            storage="sqlite:///studies/demo_pruning.db",
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            load_if_exists=True
        )
        
        def objective_pruning(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 100)
            max_depth = trial.suggest_int('max_depth', 3, 20)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            
            # Simulate intermediate reporting for pruning
            for step in range(3):
                model_partial = RandomForestClassifier(
                    n_estimators=max(1, n_estimators // 3 * (step + 1)),
                    max_depth=max_depth,
                    random_state=42
                )
                model_partial.fit(X_train, y_train)
                intermediate_score = model_partial.score(X_test, y_test)
                
                trial.report(intermediate_score, step)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            model.fit(X_train, y_train)
            return model.score(X_test, y_test)
        
        study_pruning.optimize(objective_pruning, n_trials=25)
        studies_created.append("demo_pruning.db")
        safe_print(f"   Best accuracy: {study_pruning.best_value:.4f}")
        
    except Exception as e:
        safe_print(f"   Error creating Pruning demo: {e}")
    
    # 5. Multi-objective Demo
    safe_print("\n5. Creating Multi-objective Demo...")
    try:
        study_multi = optuna.create_study(
            study_name="multi_objective_demo",
            storage="sqlite:///studies/demo_multi_objective.db",
            directions=["maximize", "minimize"],
            load_if_exists=True
        )
        
        def objective_multi(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 100)
            max_depth = trial.suggest_int('max_depth', 3, 20)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            model_complexity = n_estimators * max_depth  # Proxy for model size
            
            return accuracy, model_complexity
        
        study_multi.optimize(objective_multi, n_trials=20)
        studies_created.append("demo_multi_objective.db")
        safe_print(f"   Created {len(study_multi.best_trials)} Pareto optimal solutions")
        
    except Exception as e:
        safe_print(f"   Error creating Multi-objective demo: {e}")
    
    # Summary
    safe_print("\n" + "=" * 50)
    safe_print("COMPREHENSIVE DEMO COMPLETED!")
    safe_print(f"Created {len(studies_created)} study databases:")
    for study in studies_created:
        safe_print(f"  - {study}")
    
    safe_print("\nAccess your results:")
    safe_print("  Streamlit App: http://localhost:8501")
    safe_print("  Optuna Dashboard: http://localhost:8080")
    safe_print("\nRefresh the Optuna dashboard to see all new studies!")
    safe_print("=" * 50)
    
    return len(studies_created)

def main():
    """Main function"""
    try:
        num_studies = create_demo_studies()
        return num_studies
    except Exception as e:
        safe_print(f"Error in main: {e}")
        return 0

if __name__ == "__main__":
    main()
