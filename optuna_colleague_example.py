#!/usr/bin/env python3
"""
🎯 Optuna Tutorial for Colleagues
Complete working example to learn hyperparameter optimization

This script demonstrates:
1. Basic Optuna usage
2. Comparing optimized vs default models
3. Visualizing optimization results
4. Best practices for real projects

Perfect for sharing with team members who want to learn Optuna!
"""

import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import time

def main():
    print("🎯 Optuna Tutorial: Hyperparameter Optimization")
    print("=" * 60)
    print("This example shows how Optuna can improve your ML models!")
    print()
    
    # Load a real dataset
    print("📊 Loading breast cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()
    
    # Step 1: Train default model
    print("🔧 Step 1: Training model with default parameters...")
    default_model = RandomForestClassifier(random_state=42)
    default_model.fit(X_train, y_train)
    default_accuracy = default_model.score(X_test, y_test)
    print(f"Default model accuracy: {default_accuracy:.4f}")
    print()
    
    # Step 2: Define optimization objective
    print("🎯 Step 2: Setting up Optuna optimization...")
    
    def objective(trial):
        # Suggest hyperparameters to optimize
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 1, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        
        # Create model with suggested parameters
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        # Evaluate with cross-validation for robust results
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        return scores.mean()
    
    # Step 3: Run optimization
    print("🔍 Step 3: Running optimization (this may take 1-2 minutes)...")
    start_time = time.time()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    optimization_time = time.time() - start_time
    print(f"Optimization completed in {optimization_time:.1f} seconds")
    print()
    
    # Step 4: Results analysis
    print("📈 Step 4: Analyzing results...")
    print(f"Best accuracy found: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    print()
    
    # Step 5: Train optimized model
    print("🏆 Step 5: Training optimized model...")
    optimized_model = RandomForestClassifier(**study.best_params, random_state=42)
    optimized_model.fit(X_train, y_train)
    optimized_accuracy = optimized_model.score(X_test, y_test)
    
    # Step 6: Compare results
    print("📊 Step 6: Comparing models...")
    print("-" * 40)
    print(f"Default accuracy:   {default_accuracy:.4f}")
    print(f"Optimized accuracy: {optimized_accuracy:.4f}")
    improvement = (optimized_accuracy - default_accuracy) / default_accuracy * 100
    print(f"Improvement:        {improvement:+.2f}%")
    print("-" * 40)
    print()
    
    # Step 7: Detailed analysis
    print("🔍 Step 7: Detailed analysis...")
    
    # Show optimization history
    print("Optimization progress (last 10 trials):")
    trials_df = study.trials_dataframe()
    last_trials = trials_df.tail(10)[['number', 'value', 'params_n_estimators', 'params_max_depth']]
    print(last_trials.to_string(index=False))
    print()
    
    # Parameter importance (if enough trials)
    if len(study.trials) >= 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            print("Parameter importance:")
            for param, imp in importance.items():
                print(f"  {param}: {imp:.3f}")
            print()
        except:
            print("Parameter importance analysis not available")
    
    # Step 8: Practical recommendations
    print("💡 Step 8: Key takeaways for your projects:")
    print("1. Optuna often finds better parameters than defaults")
    print("2. Use cross-validation in objective function for robust results")
    print("3. Start with 50-100 trials for good results")
    print("4. Focus on parameters with high importance")
    print("5. Always test final model on holdout data")
    print()
    
    # Bonus: Quick comparison with different dataset
    print("🎁 Bonus: Quick test on wine dataset...")
    wine_data = load_wine()
    X_wine, y_wine = wine_data.data, wine_data.target
    X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
        X_wine, y_wine, test_size=0.2, random_state=42
    )
    
    # Test both models on wine dataset
    default_wine = RandomForestClassifier(random_state=42)
    default_wine.fit(X_train_wine, y_train_wine)
    default_wine_acc = default_wine.score(X_test_wine, y_test_wine)
    
    optimized_wine = RandomForestClassifier(**study.best_params, random_state=42)
    optimized_wine.fit(X_train_wine, y_train_wine)
    optimized_wine_acc = optimized_wine.score(X_test_wine, y_test_wine)
    
    print(f"Wine dataset - Default: {default_wine_acc:.4f}, Optimized: {optimized_wine_acc:.4f}")
    wine_improvement = (optimized_wine_acc - default_wine_acc) / default_wine_acc * 100
    print(f"Wine dataset improvement: {wine_improvement:+.2f}%")
    print()
    
    print("🎉 Tutorial complete! You now know how to:")
    print("✅ Set up Optuna optimization")
    print("✅ Define objective functions")
    print("✅ Compare optimized vs default models")
    print("✅ Analyze optimization results")
    print("✅ Apply to your own projects")
    print()
    print("🚀 Next steps:")
    print("1. Try this with your own datasets")
    print("2. Experiment with different models (XGBoost, SVM, etc.)")
    print("3. Add pruning for faster optimization")
    print("4. Explore multi-objective optimization")
    print()
    print("📚 Learn more: https://optuna.readthedocs.io/")

def quick_example():
    """
    Super quick example for immediate testing
    """
    print("⚡ Quick Optuna Example (30 seconds)")
    print("-" * 40)
    
    # Simple dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Quick objective
    def quick_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 50)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        return model.score(X_test, y_test)
    
    # Quick optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(quick_objective, n_trials=20)
    
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print("✅ Quick example complete!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_example()
    else:
        main()
