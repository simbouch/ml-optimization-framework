#!/usr/bin/env python3
"""
Comprehensive Optuna Demo Script

This script demonstrates ALL major Optuna features:
- Single and Multi-objective optimization
- Different samplers (TPE, Random, CMA-ES, Grid)
- Different pruners (Median, Successive Halving, Hyperband)
- Callbacks and custom metrics
- Study management and persistence
- Visualization and analysis
- Distributed optimization
- Advanced features

This runs continuously to populate the Optuna dashboard with rich data.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_pipeline import DataPipeline
from src.models.random_forest_optimizer import RandomForestOptimizer
from src.models.xgboost_optimizer import XGBoostOptimizer
from src.models.lightgbm_optimizer import LightGBMOptimizer
from src.optimization.study_manager import StudyManager
from src.optimization.advanced_features import (
    MultiObjectiveOptimizer,
    SamplerComparison,
    PrunerComparison
)
from src.visualization.plots import OptimizationPlotter
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class ComprehensiveOptunaDemo:
    """Comprehensive demonstration of all Optuna features."""
    
    def __init__(self):
        """Initialize the demo."""
        self.setup_database()
        self.setup_data()
        self.study_manager = StudyManager(storage_url=self.get_storage_url())
        self.plotter = OptimizationPlotter()
        
    def get_storage_url(self) -> str:
        """Get database URL from environment or use SQLite."""
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            logger.info(f"Using PostgreSQL database: {db_url}")
            return db_url
        else:
            logger.info("Using SQLite database")
            return "sqlite:///studies/comprehensive_optuna_demo.db"
    
    def setup_database(self):
        """Setup database connection."""
        # Create studies directory
        os.makedirs("studies", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
    def setup_data(self):
        """Setup datasets for optimization."""
        logger.info("🔄 Setting up datasets...")
        
        # Synthetic dataset for quick demos
        self.X_synthetic, self.y_synthetic = make_classification(
            n_samples=2000, n_features=20, n_classes=2, 
            n_informative=15, n_redundant=5, random_state=42
        )
        
        # Split synthetic data
        X_temp, self.X_test_syn, y_temp, self.y_test_syn = train_test_split(
            self.X_synthetic, self.y_synthetic, test_size=0.2, random_state=42
        )
        self.X_train_syn, self.X_val_syn, self.y_train_syn, self.y_val_syn = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42
        )
        
        logger.info(f"✅ Synthetic dataset ready: {self.X_synthetic.shape}")
        
        # Try to load real dataset
        try:
            self.pipeline = DataPipeline(random_state=42)
            summary = self.pipeline.prepare_data()
            self.X_train_real, self.X_val_real, self.y_train_real, self.y_val_real = self.pipeline.get_train_val_data()
            self.X_test_real, self.y_test_real = self.pipeline.get_test_data()
            logger.info(f"✅ Real dataset ready: {summary['total_samples']} samples")
            self.has_real_data = True
        except Exception as e:
            logger.warning(f"Real dataset not available: {e}")
            self.has_real_data = False
    
    def demo_single_objective_optimization(self):
        """Demonstrate single-objective optimization with different algorithms."""
        logger.info("🎯 Running Single-Objective Optimization Demo...")
        
        optimizers = {
            'RandomForest': RandomForestOptimizer(random_state=42, verbose=False),
            'XGBoost': XGBoostOptimizer(random_state=42, verbose=False),
            'LightGBM': LightGBMOptimizer(random_state=42, verbose=False)
        }
        
        results = {}
        for name, optimizer in optimizers.items():
            logger.info(f"   🌟 Optimizing {name}...")
            try:
                study = optimizer.optimize(
                    self.X_train_syn, self.X_val_syn, 
                    self.y_train_syn, self.y_val_syn, 
                    n_trials=10
                )
                metrics = optimizer.evaluate(self.X_test_syn, self.y_test_syn)
                results[name] = {
                    'best_value': study.best_value,
                    'test_accuracy': metrics['accuracy'],
                    'n_trials': len(study.trials)
                }
                logger.info(f"   ✅ {name}: CV={study.best_value:.4f}, Test={metrics['accuracy']:.4f}")
            except Exception as e:
                logger.error(f"   ❌ {name} failed: {e}")
        
        return results
    
    def demo_multi_objective_optimization(self):
        """Demonstrate multi-objective optimization."""
        logger.info("🎯 Running Multi-Objective Optimization Demo...")
        
        try:
            multi_opt = MultiObjectiveOptimizer(
                objectives=['accuracy', 'f1_score'],
                directions=['maximize', 'maximize'],
                random_state=42
            )
            
            study = multi_opt.create_multi_objective_study(
                study_name="multi_objective_demo",
                storage=self.get_storage_url()
            )
            
            # Run optimization
            rf_optimizer = RandomForestOptimizer(random_state=42, verbose=False)
            
            def multi_objective_function(trial):
                # Get hyperparameters
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                }
                
                # Train model
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import accuracy_score, f1_score
                from sklearn.model_selection import cross_val_score
                
                model = RandomForestClassifier(**params, random_state=42)
                
                # Calculate accuracy
                acc_scores = cross_val_score(model, self.X_train_syn, self.y_train_syn, cv=3, scoring='accuracy')
                accuracy = acc_scores.mean()
                
                # Calculate F1 score
                f1_scores = cross_val_score(model, self.X_train_syn, self.y_train_syn, cv=3, scoring='f1')
                f1 = f1_scores.mean()
                
                return accuracy, f1
            
            study.optimize(multi_objective_function, n_trials=15)
            
            logger.info(f"   ✅ Multi-objective optimization completed: {len(study.trials)} trials")
            logger.info(f"   📊 Pareto front contains {len(study.best_trials)} solutions")
            
            return study
            
        except Exception as e:
            logger.error(f"   ❌ Multi-objective optimization failed: {e}")
            return None
    
    def demo_sampler_comparison(self):
        """Demonstrate different samplers."""
        logger.info("🎯 Running Sampler Comparison Demo...")
        
        try:
            sampler_comp = SamplerComparison(random_state=42)
            
            def objective(trial):
                rf_optimizer = RandomForestOptimizer(random_state=42, verbose=False)
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                }
                
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import cross_val_score
                
                model = RandomForestClassifier(**params, random_state=42)
                scores = cross_val_score(model, self.X_train_syn, self.y_train_syn, cv=3)
                return scores.mean()
            
            results = sampler_comp.compare_samplers(
                objective_function=objective,
                n_trials=8,
                n_runs=2
            )
            
            logger.info("   ✅ Sampler comparison completed:")
            for sampler, result in results.items():
                logger.info(f"   📊 {sampler}: Best={result['best_value']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"   ❌ Sampler comparison failed: {e}")
            return None
    
    def run_continuous_demo(self):
        """Run continuous demo to populate dashboard."""
        logger.info("🚀 Starting Comprehensive Optuna Demo...")
        logger.info("   This will run continuously to populate the Optuna dashboard")
        logger.info("   Access the dashboard at: http://localhost:8080")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                logger.info(f"\n🔄 Demo Iteration {iteration}")
                
                # Run different demos in rotation
                if iteration % 4 == 1:
                    self.demo_single_objective_optimization()
                elif iteration % 4 == 2:
                    self.demo_multi_objective_optimization()
                elif iteration % 4 == 3:
                    self.demo_sampler_comparison()
                else:
                    # Run a quick optimization with real data if available
                    if self.has_real_data:
                        logger.info("🎯 Running Real Data Optimization...")
                        rf_opt = RandomForestOptimizer(random_state=42, verbose=False)
                        study = rf_opt.optimize(
                            self.X_train_real[:1000], self.X_val_real[:200],
                            self.y_train_real[:1000], self.y_val_real[:200],
                            n_trials=5
                        )
                        logger.info(f"   ✅ Real data optimization: {study.best_value:.4f}")
                
                # Wait before next iteration
                logger.info(f"   ⏱️ Waiting 30 seconds before next iteration...")
                time.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("   🛑 Demo stopped by user")
                break
            except Exception as e:
                logger.error(f"   ❌ Demo iteration failed: {e}")
                time.sleep(10)

def main():
    """Main function."""
    logger.info("🎉 Comprehensive Optuna Framework Demo")
    logger.info("=" * 60)
    
    demo = ComprehensiveOptunaDemo()
    demo.run_continuous_demo()

if __name__ == "__main__":
    main()
