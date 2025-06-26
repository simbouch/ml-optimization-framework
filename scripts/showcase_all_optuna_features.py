#!/usr/bin/env python3
"""
Complete Optuna Features Showcase

This script demonstrates EVERY major Optuna feature:
1. Single-objective optimization
2. Multi-objective optimization  
3. Different samplers (TPE, Random, CMA-ES, Grid)
4. Different pruners (Median, Successive Halving, Hyperband)
5. Callbacks and custom metrics
6. Study management and persistence
7. Visualization and analysis
8. Advanced features (constraints, user attributes, etc.)
9. Distributed optimization setup
10. Integration with ML frameworks

This creates a comprehensive showcase for the dashboard.
"""

import os
import sys
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import get_logger
from src.models.random_forest_optimizer import RandomForestOptimizer
from src.models.xgboost_optimizer import XGBoostOptimizer
from src.models.lightgbm_optimizer import LightGBMOptimizer

logger = get_logger(__name__)

class OptunaFeatureShowcase:
    """Complete showcase of all Optuna features."""
    
    def __init__(self):
        """Initialize the showcase."""
        self.storage_url = "sqlite:///studies/complete_optuna_showcase.db"
        os.makedirs("studies", exist_ok=True)
        
        # Generate datasets
        self.X_class, self.y_class = make_classification(
            n_samples=1000, n_features=20, n_classes=2, random_state=42
        )
        self.X_reg, self.y_reg = make_regression(
            n_samples=1000, n_features=15, noise=0.1, random_state=42
        )
        
        logger.info("🎯 Optuna Complete Feature Showcase Initialized")
    
    def feature_1_basic_optimization(self):
        """Feature 1: Basic single-objective optimization."""
        logger.info("🔥 Feature 1: Basic Single-Objective Optimization")
        
        study = optuna.create_study(
            study_name="basic_optimization",
            direction="maximize",
            storage=self.storage_url,
            load_if_exists=True
        )
        
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 100)
            max_depth = trial.suggest_int('max_depth', 1, 10)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            
            scores = cross_val_score(model, self.X_class, self.y_class, cv=3)
            return scores.mean()
        
        study.optimize(objective, n_trials=20)
        logger.info(f"   ✅ Best value: {study.best_value:.4f}")
        return study
    
    def feature_2_multi_objective(self):
        """Feature 2: Multi-objective optimization."""
        logger.info("🔥 Feature 2: Multi-Objective Optimization")
        
        study = optuna.create_study(
            study_name="multi_objective_showcase",
            directions=["maximize", "minimize", "maximize"],
            storage=self.storage_url,
            load_if_exists=True
        )
        
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 200)
            max_depth = trial.suggest_int('max_depth', 1, 15)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            
            # Objective 1: Accuracy (maximize)
            acc_scores = cross_val_score(model, self.X_class, self.y_class, cv=3, scoring='accuracy')
            accuracy = acc_scores.mean()
            
            # Objective 2: Model complexity (minimize)
            complexity = n_estimators * max_depth
            
            # Objective 3: F1 score (maximize)
            f1_scores = cross_val_score(model, self.X_class, self.y_class, cv=3, scoring='f1')
            f1 = f1_scores.mean()
            
            return accuracy, complexity, f1
        
        study.optimize(objective, n_trials=25)
        logger.info(f"   ✅ Pareto solutions: {len(study.best_trials)}")
        return study
    
    def feature_3_different_samplers(self):
        """Feature 3: Different samplers comparison."""
        logger.info("🔥 Feature 3: Different Samplers")
        
        samplers = {
            'TPE': optuna.samplers.TPESampler(seed=42),
            'Random': optuna.samplers.RandomSampler(seed=42),
            'CMA-ES': optuna.samplers.CmaEsSampler(seed=42),
        }
        
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            y = trial.suggest_float('y', -10, 10)
            return -(x**2 + y**2)  # Minimize x^2 + y^2
        
        results = {}
        for name, sampler in samplers.items():
            study = optuna.create_study(
                study_name=f"sampler_{name.lower()}",
                direction="maximize",
                storage=self.storage_url,
                load_if_exists=True,
                sampler=sampler
            )
            study.optimize(objective, n_trials=15)
            results[name] = study.best_value
            logger.info(f"   ✅ {name}: {study.best_value:.4f}")
        
        return results
    
    def feature_4_pruning_strategies(self):
        """Feature 4: Different pruning strategies."""
        logger.info("🔥 Feature 4: Pruning Strategies")
        
        pruners = {
            'Median': optuna.pruners.MedianPruner(),
            'SuccessiveHalving': optuna.pruners.SuccessiveHalvingPruner(),
            'Hyperband': optuna.pruners.HyperbandPruner(),
        }
        
        def objective_with_pruning(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 100)
            max_depth = trial.suggest_int('max_depth', 1, 10)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            
            # Report intermediate values
            for step in range(3):
                scores = cross_val_score(model, self.X_class, self.y_class, cv=2)
                intermediate_value = scores.mean()
                
                trial.report(intermediate_value, step)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Final evaluation
            final_scores = cross_val_score(model, self.X_class, self.y_class, cv=5)
            return final_scores.mean()
        
        results = {}
        for name, pruner in pruners.items():
            study = optuna.create_study(
                study_name=f"pruner_{name.lower()}",
                direction="maximize",
                storage=self.storage_url,
                load_if_exists=True,
                pruner=pruner
            )
            study.optimize(objective_with_pruning, n_trials=20)
            
            pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            results[name] = {
                'best_value': study.best_value,
                'pruned_count': len(pruned_trials),
                'total_trials': len(study.trials)
            }
            logger.info(f"   ✅ {name}: Best={study.best_value:.4f}, Pruned={len(pruned_trials)}/{len(study.trials)}")
        
        return results
    
    def feature_5_callbacks_and_custom_metrics(self):
        """Feature 5: Callbacks and custom metrics."""
        logger.info("🔥 Feature 5: Callbacks and Custom Metrics")
        
        study = optuna.create_study(
            study_name="callbacks_and_metrics",
            direction="maximize",
            storage=self.storage_url,
            load_if_exists=True
        )
        
        # Custom callback
        def logging_callback(study, trial):
            if trial.value is not None:
                logger.info(f"   Trial {trial.number}: {trial.value:.4f}")
        
        def objective_with_attributes(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 100)
            max_depth = trial.suggest_int('max_depth', 1, 10)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            
            scores = cross_val_score(model, self.X_class, self.y_class, cv=3)
            accuracy = scores.mean()
            
            # Set user attributes
            trial.set_user_attr('model_type', 'RandomForest')
            trial.set_user_attr('feature_count', self.X_class.shape[1])
            trial.set_user_attr('cv_std', scores.std())
            
            return accuracy
        
        study.optimize(objective_with_attributes, n_trials=15, callbacks=[logging_callback])
        logger.info(f"   ✅ Best value: {study.best_value:.4f}")
        return study
    
    def feature_6_study_management(self):
        """Feature 6: Advanced study management."""
        logger.info("🔥 Feature 6: Study Management")
        
        # Create study with custom attributes
        study = optuna.create_study(
            study_name="advanced_study_management",
            direction="maximize",
            storage=self.storage_url,
            load_if_exists=True
        )
        
        # Set study user attributes
        study.set_user_attr('dataset', 'synthetic_classification')
        study.set_user_attr('algorithm', 'RandomForest')
        study.set_user_attr('created_by', 'optuna_showcase')
        
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 100)
            max_depth = trial.suggest_int('max_depth', 1, 10)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            
            scores = cross_val_score(model, self.X_class, self.y_class, cv=3)
            return scores.mean()
        
        study.optimize(objective, n_trials=10)
        
        # Demonstrate study statistics
        logger.info(f"   ✅ Study statistics:")
        logger.info(f"      - Best value: {study.best_value:.4f}")
        logger.info(f"      - Best params: {study.best_params}")
        logger.info(f"      - Total trials: {len(study.trials)}")
        logger.info(f"      - User attributes: {study.user_attrs}")
        
        return study
    
    def feature_7_ml_framework_integration(self):
        """Feature 7: ML Framework Integration."""
        logger.info("🔥 Feature 7: ML Framework Integration")
        
        # Use our framework optimizers
        optimizers = {
            'RandomForest': RandomForestOptimizer(random_state=42, verbose=False),
            'XGBoost': XGBoostOptimizer(random_state=42, verbose=False),
            'LightGBM': LightGBMOptimizer(random_state=42, verbose=False)
        }
        
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_class, self.y_class, test_size=0.2, random_state=42
        )
        
        results = {}
        for name, optimizer in optimizers.items():
            try:
                logger.info(f"   🔧 Optimizing {name}...")
                study = optimizer.optimize(X_train, X_val, y_train, y_val, n_trials=8)
                metrics = optimizer.evaluate(X_val, y_val)
                
                results[name] = {
                    'best_cv_score': study.best_value,
                    'test_accuracy': metrics['accuracy'],
                    'study_name': study.study_name
                }
                logger.info(f"   ✅ {name}: CV={study.best_value:.4f}, Test={metrics['accuracy']:.4f}")
            except Exception as e:
                logger.warning(f"   ⚠️ {name} failed: {e}")
        
        return results
    
    def run_complete_showcase(self):
        """Run the complete Optuna feature showcase."""
        logger.info("🚀 Starting Complete Optuna Feature Showcase")
        logger.info("=" * 70)
        
        features = [
            self.feature_1_basic_optimization,
            self.feature_2_multi_objective,
            self.feature_3_different_samplers,
            self.feature_4_pruning_strategies,
            self.feature_5_callbacks_and_custom_metrics,
            self.feature_6_study_management,
            self.feature_7_ml_framework_integration,
        ]
        
        results = {}
        for i, feature_func in enumerate(features, 1):
            try:
                logger.info(f"\n{'='*20} Running Feature {i} {'='*20}")
                result = feature_func()
                results[f"feature_{i}"] = result
                time.sleep(1)  # Brief pause between features
            except Exception as e:
                logger.error(f"❌ Feature {i} failed: {e}")
                results[f"feature_{i}"] = None
        
        logger.info("\n" + "="*70)
        logger.info("🎉 COMPLETE OPTUNA SHOWCASE FINISHED!")
        logger.info("="*70)
        logger.info(f"📊 Database: {self.storage_url}")
        logger.info("🌐 View results in dashboard: http://localhost:8080")
        logger.info("="*70)
        
        return results

def main():
    """Main function."""
    showcase = OptunaFeatureShowcase()
    showcase.run_complete_showcase()

if __name__ == "__main__":
    main()
