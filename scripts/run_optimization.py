#!/usr/bin/env python3
"""
Simple optimization runner script.

This script provides a quick way to run optimization experiments
with predefined configurations for common use cases.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_pipeline import DataPipeline
from src.models.random_forest_optimizer import RandomForestOptimizer
from src.models.xgboost_optimizer import XGBoostOptimizer
from src.models.lightgbm_optimizer import LightGBMOptimizer
from src.optimization.study_manager import StudyManager
from src.visualization.plots import OptimizationPlotter
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_quick_optimization():
    """Run a quick optimization experiment for demonstration."""
    
    print("🚀 Starting Quick Optimization Demo")
    print("=" * 50)
    
    # Setup data pipeline
    print("📊 Setting up data pipeline...")
    data_pipeline = DataPipeline(random_state=42)
    summary = data_pipeline.prepare_data()
    
    print(f"✅ Data prepared: {summary['total_samples']} samples, {summary['total_features']} features")
    
    # Get data splits
    X_train, X_val, y_train, y_val = data_pipeline.get_train_val_data()
    X_test, y_test = data_pipeline.get_test_data()
    
    # Initialize optimizers
    print("\n🔧 Initializing optimizers...")
    optimizers = {
        'Random Forest': RandomForestOptimizer(random_state=42, verbose=False),
        'XGBoost': XGBoostOptimizer(random_state=42, verbose=False),
        'LightGBM': LightGBMOptimizer(random_state=42, verbose=False)
    }
    
    # Run optimizations
    results = {}
    n_trials = 20  # Quick demo with fewer trials
    
    for model_name, optimizer in optimizers.items():
        print(f"\n🎯 Optimizing {model_name}...")
        start_time = time.time()
        
        study = optimizer.optimize(
            X_train, X_val, y_train, y_val,
            n_trials=n_trials
        )
        
        optimization_time = time.time() - start_time
        test_metrics = optimizer.evaluate(X_test, y_test)
        
        results[model_name] = {
            'best_cv_score': study.best_value,
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1_score'],
            'optimization_time': optimization_time,
            'best_params': study.best_params
        }
        
        print(f"   ✅ CV Score: {study.best_value:.4f}")
        print(f"   ✅ Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   ⏱️ Time: {optimization_time:.2f}s")
    
    # Print summary
    print("\n" + "=" * 50)
    print("📈 OPTIMIZATION RESULTS SUMMARY")
    print("=" * 50)
    
    # Sort by test accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
    
    print(f"{'Model':<15} {'CV Score':<10} {'Test Acc':<10} {'Test F1':<10} {'Time (s)':<10}")
    print("-" * 65)
    
    for model_name, result in sorted_results:
        print(f"{model_name:<15} {result['best_cv_score']:<10.4f} "
              f"{result['test_accuracy']:<10.4f} {result['test_f1']:<10.4f} "
              f"{result['optimization_time']:<10.2f}")
    
    # Best model details
    best_model, best_result = sorted_results[0]
    print(f"\n🏆 Best Model: {best_model}")
    print(f"📊 Test Accuracy: {best_result['test_accuracy']:.4f}")
    print(f"⚙️ Best Parameters:")
    for param, value in best_result['best_params'].items():
        print(f"   • {param}: {value}")
    
    print("\n✅ Quick optimization demo completed!")
    return results


def run_comprehensive_optimization():
    """Run a comprehensive optimization with more trials and analysis."""
    
    print("🚀 Starting Comprehensive Optimization")
    print("=" * 50)
    
    # Setup
    data_pipeline = DataPipeline(random_state=42)
    data_pipeline.prepare_data()
    X_train, X_val, y_train, y_val = data_pipeline.get_train_val_data()
    X_test, y_test = data_pipeline.get_test_data()
    
    # Study manager for persistence
    study_manager = StudyManager()
    
    # Run Random Forest optimization with more trials
    print("🌲 Running comprehensive Random Forest optimization...")
    rf_optimizer = RandomForestOptimizer(random_state=42, verbose=True)
    
    # Create persistent study
    study = study_manager.create_study(
        study_name="comprehensive_rf_optimization",
        model_name="random_forest"
    )
    
    # Run optimization
    start_time = time.time()
    study = rf_optimizer.optimize(
        X_train, X_val, y_train, y_val,
        n_trials=100,
        study=study
    )
    optimization_time = time.time() - start_time
    
    # Evaluate
    test_metrics = rf_optimizer.evaluate(X_test, y_test)
    
    # Analysis
    print("\n📊 Generating analysis...")
    
    # Feature importance
    importance = rf_optimizer.analyze_feature_importance()
    complexity = rf_optimizer.get_model_complexity()
    
    # Results
    print("\n" + "=" * 50)
    print("📈 COMPREHENSIVE RESULTS")
    print("=" * 50)
    
    print(f"🎯 Best CV Score: {study.best_value:.4f}")
    print(f"📊 Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"📊 Test F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"⏱️ Optimization Time: {optimization_time:.2f}s")
    print(f"🔄 Total Trials: {len(study.trials)}")
    
    print(f"\n🏗️ Model Complexity:")
    print(f"   • Trees: {complexity['n_estimators']}")
    print(f"   • Total Nodes: {complexity['total_nodes']}")
    print(f"   • Avg Tree Depth: {complexity['avg_tree_depth']:.2f}")
    
    print(f"\n🔍 Top 5 Important Features:")
    top_features = importance['top_features_indices'][:5]
    for i, feature_idx in enumerate(top_features, 1):
        importance_value = importance['feature_importance_values'][feature_idx]
        print(f"   {i}. Feature {feature_idx}: {importance_value:.4f}")
    
    print(f"\n⚙️ Best Parameters:")
    for param, value in study.best_params.items():
        print(f"   • {param}: {value}")
    
    # Save study data
    study_summary = study_manager.get_study_summary("comprehensive_rf_optimization")
    if study_summary:
        print(f"\n💾 Study saved to database")
        print(f"   • Study name: {study_summary['study_name']}")
        print(f"   • Completed trials: {study_summary['n_completed_trials']}")
        print(f"   • Pruned trials: {study_summary['n_pruned_trials']}")
    
    print("\n✅ Comprehensive optimization completed!")
    return study, rf_optimizer


def main():
    """Main function to run optimization experiments."""
    
    print("ML Optimization Framework - Demo Runner")
    print("=" * 50)
    print("Choose an option:")
    print("1. Quick optimization (20 trials per model)")
    print("2. Comprehensive optimization (100 trials, Random Forest)")
    print("3. Exit")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            results = run_quick_optimization()
        elif choice == "2":
            study, optimizer = run_comprehensive_optimization()
        elif choice == "3":
            print("👋 Goodbye!")
            return
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
            return
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        logger.error(f"Optimization failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
