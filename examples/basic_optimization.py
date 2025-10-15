#!/usr/bin/env python3
"""
Basic Optimization Example
==========================

This example demonstrates the basic usage of the ML Optimization Framework
for hyperparameter optimization of machine learning models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from src.config import OptimizationConfig
from src.optimizers import RandomForestOptimizer, GradientBoostingOptimizer, SVMOptimizer
from src.study_manager import StudyManager

def basic_classification_example():
    """Basic classification optimization example."""
    print("🎯 Basic Classification Optimization")
    print("=" * 50)
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create configuration
    config = OptimizationConfig(
        study_name="basic_classification",
        n_trials=20,  # Small number for quick demo
        sampler_name="TPE",
        pruner_name="Median",
        cv_folds=3,
        random_seed=42
    )
    
    # Initialize optimizer
    optimizer = RandomForestOptimizer(config, task_type="classification")
    
    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"🔧 Configuration: {config.n_trials} trials, {config.cv_folds}-fold CV")
    print(f"🎲 Sampler: {config.sampler_name}, Pruner: {config.pruner_name}")
    print()
    
    # Run optimization
    print("🚀 Starting optimization...")
    study = optimizer.optimize(X_train, y_train)
    
    # Display results
    print("\n📈 Optimization Results:")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    # Test on holdout set
    best_model = optimizer.get_best_model()
    best_model.fit(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    print(f"\n🎯 Test set accuracy: {test_score:.4f}")
    
    return study

def basic_regression_example():
    """Basic regression optimization example."""
    print("\n🎯 Basic Regression Optimization")
    print("=" * 50)
    
    # Generate sample data
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=0.1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create configuration
    config = OptimizationConfig(
        study_name="basic_regression",
        n_trials=20,
        sampler_name="TPE",
        random_seed=42
    )
    
    # Initialize optimizer
    optimizer = GradientBoostingOptimizer(config, task_type="regression")
    
    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"🔧 Configuration: {config.n_trials} trials")
    print()
    
    # Run optimization
    print("🚀 Starting optimization...")
    study = optimizer.optimize(X_train, y_train)
    
    # Display results
    print("\n📈 Optimization Results:")
    print(f"Best value (R²): {study.best_value:.4f}")
    print(f"Best parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    # Test on holdout set
    best_model = optimizer.get_best_model()
    best_model.fit(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    print(f"\n🎯 Test set R²: {test_score:.4f}")
    
    return study

def compare_optimizers_example():
    """Compare different optimizers on the same dataset."""
    print("\n🎯 Optimizer Comparison Example")
    print("=" * 50)
    
    # Generate sample data
    X, y = make_classification(
        n_samples=800,
        n_features=15,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Configuration
    config = OptimizationConfig(
        n_trials=15,  # Small for quick comparison
        cv_folds=3,
        random_seed=42
    )
    
    # Define optimizers to compare
    optimizers = {
        "Random Forest": RandomForestOptimizer(config, task_type="classification"),
        "Gradient Boosting": GradientBoostingOptimizer(config, task_type="classification"),
        "SVM": SVMOptimizer(config, task_type="classification")
    }
    
    results = {}
    
    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"🔧 Each optimizer: {config.n_trials} trials")
    print()
    
    # Run optimization for each algorithm
    for name, optimizer in optimizers.items():
        print(f"🚀 Optimizing {name}...")
        
        # Update study name for each optimizer
        optimizer.config.study_name = f"comparison_{name.lower().replace(' ', '_')}"
        
        study = optimizer.optimize(X_train, y_train)
        
        # Test on holdout set
        best_model = optimizer.get_best_model()
        best_model.fit(X_train, y_train)
        test_score = best_model.score(X_test, y_test)
        
        results[name] = {
            'cv_score': study.best_value,
            'test_score': test_score,
            'best_params': study.best_params,
            'n_trials': len(study.trials)
        }
        
        print(f"  ✅ CV Score: {study.best_value:.4f}, Test Score: {test_score:.4f}")
    
    # Display comparison
    print("\n📊 Comparison Results:")
    print("-" * 60)
    print(f"{'Algorithm':<15} {'CV Score':<10} {'Test Score':<10} {'Trials':<8}")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"{name:<15} {result['cv_score']:<10.4f} {result['test_score']:<10.4f} {result['n_trials']:<8}")
    
    # Find best performer
    best_algorithm = max(results.keys(), key=lambda k: results[k]['test_score'])
    print(f"\n🏆 Best performer: {best_algorithm} (Test Score: {results[best_algorithm]['test_score']:.4f})")
    
    return results

def study_management_example():
    """Demonstrate study management capabilities."""
    print("\n🎯 Study Management Example")
    print("=" * 50)
    
    # Create configuration
    config = OptimizationConfig(
        study_name="management_demo",
        n_trials=10,
        random_seed=42
    )
    
    # Initialize study manager
    manager = StudyManager(config)
    
    # Generate sample data
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    
    # Run optimization
    print("📝 Creating and running study...")
    optimizer = RandomForestOptimizer(config, task_type="classification")
    study = optimizer.optimize(X, y)
    
    # Get study summary
    print("\n📊 Study Summary:")
    summary = manager.get_study_summary(study.study_name)
    for key, value in summary.items():
        if key != 'best_params':
            print(f"  {key}: {value}")
    
    print("\n🎯 Best Parameters:")
    for param, value in summary['best_params'].items():
        print(f"  {param}: {value}")
    
    # Export results
    print("\n💾 Exporting results...")
    results_df = manager.export_study_results(study.study_name, format="csv")
    print(f"  Exported {len(results_df)} trials")
    print(f"  Columns: {list(results_df.columns)}")
    
    # Show first few trials
    print("\n📋 First 3 trials:")
    print(results_df.head(3)[['number', 'value', 'state']].to_string(index=False))
    
    return manager, results_df

def main():
    """Run all basic examples."""
    print("🎯 ML Optimization Framework - Basic Examples")
    print("=" * 60)
    
    try:
        # Run examples
        classification_study = basic_classification_example()
        regression_study = basic_regression_example()
        comparison_results = compare_optimizers_example()
        manager, results_df = study_management_example()
        
        print("\n✅ All examples completed successfully!")
        print("\n📚 Next Steps:")
        print("  1. Try the advanced examples in examples/advanced_optimization.py")
        print("  2. Check the Optuna dashboard: http://localhost:8080")
        print("  3. Explore the French tutorial materials in tutorial_octobre_2025_french/")
        print("  4. Run the unified demo: python create_unified_demo.py")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
