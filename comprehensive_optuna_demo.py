#!/usr/bin/env python3
"""
Comprehensive Optuna Feature Demonstration
This script showcases ALL major Optuna capabilities in a production-ready framework.
"""

import optuna
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
from pathlib import Path
import time
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Import our modular framework
from src.config import OptimizationConfig
from src.optimizers import RandomForestOptimizer, XGBoostOptimizer, SVMOptimizer
from src.study_manager import StudyManager


def setup_logging():
    """Set up comprehensive logging."""
    logger.add(
        "logs/comprehensive_demo.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="INFO",
        rotation="10 MB"
    )


def create_demo_datasets():
    """Create various datasets for demonstration."""
    logger.info("Creating demo datasets...")
    
    datasets = {}
    
    # 1. Binary Classification Dataset
    X_binary, y_binary = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
        n_classes=2, random_state=42
    )
    datasets['binary_classification'] = (X_binary, y_binary)
    
    # 2. Multi-class Classification Dataset
    X_multi, y_multi = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
        n_classes=3, random_state=42
    )
    datasets['multiclass_classification'] = (X_multi, y_multi)
    
    # 3. Regression Dataset
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=20, noise=0.1, random_state=42
    )
    datasets['regression'] = (X_reg, y_reg)
    
    # 4. Real-world datasets
    iris = load_iris()
    datasets['iris'] = (iris.data, iris.target)
    
    wine = load_wine()
    datasets['wine'] = (wine.data, wine.target)
    
    logger.info(f"Created {len(datasets)} datasets")
    return datasets


def demo_single_objective_optimization():
    """Demonstrate single-objective optimization with different samplers."""
    logger.info("=== SINGLE-OBJECTIVE OPTIMIZATION DEMO ===")
    
    # Create dataset
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    samplers = ["TPE", "Random", "CmaEs"]
    
    for sampler_name in samplers:
        logger.info(f"Testing {sampler_name} sampler...")
        
        config = OptimizationConfig(
            study_name=f"single_objective_{sampler_name.lower()}",
            direction="maximize",
            n_trials=30,
            sampler_name=sampler_name,
            pruner_name="Median"
        )
        
        optimizer = RandomForestOptimizer(config, task_type="classification")
        study = optimizer.optimize(X_train, y_train)
        
        # Evaluate best model
        best_model = optimizer.get_best_model()
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"{sampler_name} - Best accuracy: {accuracy:.4f}")
        logger.info(f"{sampler_name} - Best params: {study.best_params}")


def demo_multi_objective_optimization():
    """Demonstrate multi-objective optimization."""
    logger.info("=== MULTI-OBJECTIVE OPTIMIZATION DEMO ===")
    
    def multi_objective_function(trial):
        """Multi-objective function: maximize accuracy, minimize model complexity."""
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        
        # Create and evaluate model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
        X, y = make_classification(n_samples=300, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Objective 1: Maximize accuracy
        obj1 = accuracy
        
        # Objective 2: Minimize complexity (negative of total parameters)
        complexity = n_estimators * max_depth
        obj2 = -complexity / 1000  # Normalize
        
        return obj1, obj2
    
    # Create multi-objective study
    study = optuna.create_study(
        study_name="multi_objective_demo",
        storage="sqlite:///studies/multi_objective_demo.db",
        directions=["maximize", "maximize"],  # Both objectives to maximize
        load_if_exists=True
    )
    
    study.optimize(multi_objective_function, n_trials=50)
    
    logger.info(f"Multi-objective optimization completed with {len(study.trials)} trials")
    logger.info(f"Number of Pareto optimal solutions: {len(study.best_trials)}")
    
    # Log best trials
    for i, trial in enumerate(study.best_trials[:5]):  # Top 5
        logger.info(f"Pareto solution {i+1}: Accuracy={trial.values[0]:.4f}, "
                   f"Complexity={-trial.values[1]*1000:.0f}, Params={trial.params}")


def demo_pruning_strategies():
    """Demonstrate different pruning strategies."""
    logger.info("=== PRUNING STRATEGIES DEMO ===")
    
    def objective_with_intermediate_values(trial):
        """Objective function that reports intermediate values for pruning."""
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        
        X, y = make_classification(n_samples=500, n_features=15, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Simulate iterative training with intermediate reporting
        model = RandomForestClassifier(n_estimators=1, max_depth=max_depth, random_state=42)
        
        for step in range(1, min(n_estimators, 20) + 1):
            model.n_estimators = step
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            intermediate_accuracy = accuracy_score(y_test, y_pred)
            
            # Report intermediate value
            trial.report(intermediate_accuracy, step)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return intermediate_accuracy
    
    pruners = [
        ("Median", optuna.pruners.MedianPruner()),
        ("SuccessiveHalving", optuna.pruners.SuccessiveHalvingPruner()),
        ("Hyperband", optuna.pruners.HyperbandPruner()),
    ]
    
    for pruner_name, pruner in pruners:
        logger.info(f"Testing {pruner_name} pruner...")
        
        study = optuna.create_study(
            study_name=f"pruning_{pruner_name.lower()}",
            storage=f"sqlite:///studies/pruning_{pruner_name.lower()}.db",
            direction="maximize",
            pruner=pruner,
            load_if_exists=True
        )
        
        study.optimize(objective_with_intermediate_values, n_trials=30, timeout=60)
        
        # Count pruned trials
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        logger.info(f"{pruner_name} - Completed: {len(completed_trials)}, "
                   f"Pruned: {len(pruned_trials)}, "
                   f"Best value: {study.best_value:.4f}")


def demo_real_world_ml_scenarios():
    """Demonstrate real-world ML optimization scenarios."""
    logger.info("=== REAL-WORLD ML SCENARIOS DEMO ===")
    
    # Scenario 1: Hyperparameter optimization for different algorithms
    algorithms = [
        ("RandomForest", RandomForestOptimizer),
        ("XGBoost", XGBoostOptimizer),
        ("SVM", SVMOptimizer),
    ]
    
    # Use Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    results = {}
    
    for algo_name, optimizer_class in algorithms:
        logger.info(f"Optimizing {algo_name}...")
        
        config = OptimizationConfig(
            study_name=f"realworld_{algo_name.lower()}",
            direction="maximize",
            n_trials=25,
            sampler_name="TPE",
            pruner_name="Median"
        )
        
        try:
            optimizer = optimizer_class(config, task_type="classification")
            study = optimizer.optimize(X_train, y_train)
            
            # Evaluate on test set
            best_model = optimizer.get_best_model()
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            results[algo_name] = {
                "best_cv_score": study.best_value,
                "test_accuracy": test_accuracy,
                "best_params": study.best_params,
                "n_trials": len(study.trials)
            }
            
            logger.info(f"{algo_name} - CV Score: {study.best_value:.4f}, "
                       f"Test Accuracy: {test_accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error optimizing {algo_name}: {e}")
            results[algo_name] = {"error": str(e)}
    
    # Compare results
    logger.info("=== ALGORITHM COMPARISON ===")
    for algo_name, result in results.items():
        if "error" not in result:
            logger.info(f"{algo_name}: CV={result['best_cv_score']:.4f}, "
                       f"Test={result['test_accuracy']:.4f}")


def demo_study_management():
    """Demonstrate comprehensive study management."""
    logger.info("=== STUDY MANAGEMENT DEMO ===")
    
    config = OptimizationConfig()
    study_manager = StudyManager(config)
    
    # Get all studies summary
    all_studies = study_manager.get_all_studies_summary()
    logger.info(f"Found {len(all_studies)} studies")
    
    for study_summary in all_studies:
        logger.info(f"Study: {study_summary['study_name']}, "
                   f"Trials: {study_summary['n_trials']}, "
                   f"Best: {study_summary['best_value']}")
    
    # Export results if studies exist
    if all_studies:
        study_name = all_studies[0]['study_name']
        try:
            export_path = study_manager.export_study_results(study_name, format="csv")
            logger.info(f"Exported study results to: {export_path}")
        except Exception as e:
            logger.warning(f"Could not export study results: {e}")


def main():
    """Run comprehensive Optuna demonstration."""
    print("🎯 Comprehensive Optuna Feature Demonstration")
    print("=" * 60)
    
    # Setup
    Path("studies").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    setup_logging()
    
    logger.info("Starting comprehensive Optuna demonstration...")
    
    try:
        # Run all demonstrations
        demo_single_objective_optimization()
        demo_multi_objective_optimization()
        demo_pruning_strategies()
        demo_real_world_ml_scenarios()
        demo_study_management()
        
        logger.info("✅ All demonstrations completed successfully!")
        print("\n✅ Comprehensive Optuna demonstration completed!")
        print("📊 Check the Optuna dashboard to visualize results:")
        print("   http://localhost:8080")
        print("📁 Study databases created in: ./studies/")
        print("📋 Logs available in: ./logs/")
        print("📈 Results exported to: ./results/")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()
