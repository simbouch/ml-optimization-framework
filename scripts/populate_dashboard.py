#!/usr/bin/env python3
"""
Quick script to populate Optuna dashboard with diverse studies.

This creates multiple studies with different configurations to showcase
all Optuna features in the dashboard.
"""

import os
import sys
import optuna
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def create_database_studies():
    """Create multiple studies with different characteristics."""
    
    # Create studies directory
    os.makedirs("studies", exist_ok=True)
    
    # Database URL
    storage_url = "sqlite:///studies/optuna_dashboard_demo.db"
    
    logger.info("🎯 Creating diverse Optuna studies for dashboard demo...")
    
    # Generate synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    
    # Study 1: Single-objective optimization (maximize accuracy)
    logger.info("📊 Creating Study 1: Single-objective optimization")
    study1 = optuna.create_study(
        study_name="single_objective_accuracy",
        direction="maximize",
        storage=storage_url,
        load_if_exists=True
    )
    
    def objective_accuracy(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 1, 10)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return scores.mean()
    
    study1.optimize(objective_accuracy, n_trials=20)
    logger.info(f"   ✅ Study 1 completed: {len(study1.trials)} trials, best = {study1.best_value:.4f}")
    
    # Study 2: Multi-objective optimization (accuracy vs model complexity)
    logger.info("📊 Creating Study 2: Multi-objective optimization")
    study2 = optuna.create_study(
        study_name="multi_objective_accuracy_complexity",
        directions=["maximize", "minimize"],
        storage=storage_url,
        load_if_exists=True
    )
    
    def objective_multi(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 1, 15)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
        # Accuracy (maximize)
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        accuracy = scores.mean()
        
        # Model complexity (minimize) - approximated by number of parameters
        complexity = n_estimators * max_depth
        
        return accuracy, complexity
    
    study2.optimize(objective_multi, n_trials=25)
    logger.info(f"   ✅ Study 2 completed: {len(study2.trials)} trials, {len(study2.best_trials)} Pareto solutions")
    
    # Study 3: Optimization with pruning
    logger.info("📊 Creating Study 3: Optimization with pruning")
    study3 = optuna.create_study(
        study_name="pruned_optimization",
        direction="maximize",
        storage=storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    
    def objective_pruned(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 1, 10)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
        # Simulate intermediate values for pruning
        for step in range(3):
            scores = cross_val_score(model, X, y, cv=2, scoring='accuracy')
            intermediate_value = scores.mean()
            
            trial.report(intermediate_value, step)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Final evaluation
        final_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return final_scores.mean()
    
    study3.optimize(objective_pruned, n_trials=30)
    logger.info(f"   ✅ Study 3 completed: {len(study3.trials)} trials, pruned = {len([t for t in study3.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    # Study 4: Different sampler comparison
    logger.info("📊 Creating Study 4: TPE Sampler optimization")
    study4 = optuna.create_study(
        study_name="tpe_sampler_optimization",
        direction="maximize",
        storage=storage_url,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study4.optimize(objective_accuracy, n_trials=15)
    logger.info(f"   ✅ Study 4 completed: {len(study4.trials)} trials, best = {study4.best_value:.4f}")
    
    # Study 5: Random sampler for comparison
    logger.info("📊 Creating Study 5: Random Sampler optimization")
    study5 = optuna.create_study(
        study_name="random_sampler_optimization",
        direction="maximize",
        storage=storage_url,
        load_if_exists=True,
        sampler=optuna.samplers.RandomSampler(seed=42)
    )
    
    study5.optimize(objective_accuracy, n_trials=15)
    logger.info(f"   ✅ Study 5 completed: {len(study5.trials)} trials, best = {study5.best_value:.4f}")
    
    # Study 6: Failed trials simulation
    logger.info("📊 Creating Study 6: Optimization with some failed trials")
    study6 = optuna.create_study(
        study_name="optimization_with_failures",
        direction="maximize",
        storage=storage_url,
        load_if_exists=True
    )
    
    def objective_with_failures(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 1, 10)
        
        # Simulate some failures
        if trial.number % 7 == 0:  # Every 7th trial fails
            raise ValueError("Simulated failure")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return scores.mean()
    
    # Run with error handling
    for i in range(20):
        try:
            study6.optimize(objective_with_failures, n_trials=1)
        except:
            pass  # Continue despite failures
    
    logger.info(f"   ✅ Study 6 completed: {len(study6.trials)} trials, failed = {len([t for t in study6.trials if t.state == optuna.trial.TrialState.FAIL])}")
    
    logger.info(f"\n🎉 Dashboard population completed!")
    logger.info(f"📊 Database: {storage_url}")
    logger.info(f"🌐 Start dashboard with: optuna-dashboard {storage_url}")
    
    return storage_url

def main():
    """Main function."""
    logger.info("🚀 Populating Optuna Dashboard with Demo Studies")
    logger.info("=" * 60)
    
    storage_url = create_database_studies()
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 DASHBOARD READY!")
    logger.info("=" * 60)
    logger.info(f"📊 Database: {storage_url}")
    logger.info("🌐 To start the dashboard, run:")
    logger.info(f"   optuna-dashboard {storage_url}")
    logger.info("🔗 Then open: http://localhost:8080")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
