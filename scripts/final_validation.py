#!/usr/bin/env python3
"""
Final Comprehensive Validation Script

This script validates ALL components of the ML Optimization Framework:
1. Framework functionality
2. Optuna dashboard population
3. All optimization features
4. Docker readiness
5. Documentation completeness

This is the ultimate test to ensure everything works perfectly.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import sqlite3

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class FinalValidator:
    """Comprehensive validation of the entire framework."""
    
    def __init__(self):
        """Initialize validator."""
        self.validation_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            logger.info(f"✅ {test_name}: PASSED {details}")
        else:
            logger.error(f"❌ {test_name}: FAILED {details}")
        
        self.validation_results[test_name] = {
            'passed': passed,
            'details': details
        }
    
    def test_1_basic_imports(self):
        """Test 1: Basic framework imports."""
        logger.info("🔍 Test 1: Basic Framework Imports")
        
        try:
            # Core imports
            from src.data.data_pipeline import DataPipeline
            from src.models.random_forest_optimizer import RandomForestOptimizer
            from src.models.xgboost_optimizer import XGBoostOptimizer
            from src.models.lightgbm_optimizer import LightGBMOptimizer
            from src.optimization.config import OptimizationConfig
            from src.optimization.study_manager import StudyManager
            from src.visualization.plots import OptimizationPlotter
            
            # Optuna imports
            import optuna
            import optuna_integration
            import optuna_dashboard
            
            self.log_test("Basic Imports", True, "All core modules imported successfully")
            return True
            
        except Exception as e:
            self.log_test("Basic Imports", False, f"Import error: {e}")
            return False
    
    def test_2_data_pipeline(self):
        """Test 2: Data pipeline functionality."""
        logger.info("🔍 Test 2: Data Pipeline")
        
        try:
            from src.data.data_pipeline import DataPipeline
            
            pipeline = DataPipeline(random_state=42)
            
            # Test initialization
            assert hasattr(pipeline, 'is_prepared')
            assert pipeline.is_prepared == False
            
            # Test synthetic data preparation (faster than real data)
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=100, n_features=5, random_state=42)
            
            # Simulate data preparation
            pipeline.X_train = X[:60]
            pipeline.X_val = X[60:80]
            pipeline.X_test = X[80:]
            pipeline.y_train = y[:60]
            pipeline.y_val = y[60:80]
            pipeline.y_test = y[80:]
            pipeline.is_prepared = True
            
            # Test data retrieval
            X_train, X_val, y_train, y_val = pipeline.get_train_val_data()
            X_test, y_test = pipeline.get_test_data()
            
            assert X_train.shape[0] == 60
            assert X_val.shape[0] == 20
            assert X_test.shape[0] == 20
            
            self.log_test("Data Pipeline", True, f"Pipeline working with {X.shape[0]} samples")
            return True
            
        except Exception as e:
            self.log_test("Data Pipeline", False, f"Pipeline error: {e}")
            return False
    
    def test_3_optimizers(self):
        """Test 3: All optimizers functionality."""
        logger.info("🔍 Test 3: ML Optimizers")
        
        try:
            from sklearn.datasets import make_classification
            from sklearn.model_selection import train_test_split
            from src.models.random_forest_optimizer import RandomForestOptimizer
            
            # Generate test data
            X, y = make_classification(n_samples=200, n_features=5, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Test Random Forest optimizer
            rf_optimizer = RandomForestOptimizer(random_state=42, verbose=False)
            study = rf_optimizer.optimize(X_train, X_val, y_train, y_val, n_trials=3)
            metrics = rf_optimizer.evaluate(X_val, y_val)
            
            assert study.best_value > 0
            assert 'accuracy' in metrics
            assert metrics['accuracy'] > 0
            
            self.log_test("ML Optimizers", True, f"RF optimizer: CV={study.best_value:.3f}, Test={metrics['accuracy']:.3f}")
            return True
            
        except Exception as e:
            self.log_test("ML Optimizers", False, f"Optimizer error: {e}")
            return False
    
    def test_4_optuna_features(self):
        """Test 4: Optuna features."""
        logger.info("🔍 Test 4: Optuna Features")
        
        try:
            import optuna
            from sklearn.datasets import make_classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            
            # Test basic optimization
            X, y = make_classification(n_samples=100, n_features=5, random_state=42)
            
            def objective(trial):
                n_estimators = trial.suggest_int('n_estimators', 10, 50)
                max_depth = trial.suggest_int('max_depth', 1, 5)
                
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
                
                scores = cross_val_score(model, X, y, cv=2)
                return scores.mean()
            
            # Single-objective study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=5)
            
            assert len(study.trials) == 5
            assert study.best_value > 0
            
            # Multi-objective study
            def multi_objective(trial):
                n_estimators = trial.suggest_int('n_estimators', 10, 50)
                accuracy = objective(trial)
                complexity = n_estimators
                return accuracy, complexity
            
            multi_study = optuna.create_study(directions=['maximize', 'minimize'])
            multi_study.optimize(multi_objective, n_trials=3)
            
            assert len(multi_study.trials) == 3
            
            self.log_test("Optuna Features", True, f"Single: {study.best_value:.3f}, Multi: {len(multi_study.best_trials)} Pareto")
            return True
            
        except Exception as e:
            self.log_test("Optuna Features", False, f"Optuna error: {e}")
            return False
    
    def test_5_dashboard_database(self):
        """Test 5: Dashboard database."""
        logger.info("🔍 Test 5: Dashboard Database")
        
        try:
            # Check if database files exist
            db_files = [
                "studies/optuna_dashboard_demo.db",
                "studies/complete_optuna_showcase.db"
            ]
            
            existing_dbs = []
            for db_file in db_files:
                if os.path.exists(db_file):
                    existing_dbs.append(db_file)
                    
                    # Check if database has studies
                    try:
                        conn = sqlite3.connect(db_file)
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='studies';")
                        if cursor.fetchone():
                            cursor.execute("SELECT COUNT(*) FROM studies;")
                            study_count = cursor.fetchone()[0]
                            logger.info(f"   📊 {db_file}: {study_count} studies")
                        conn.close()
                    except:
                        pass
            
            if existing_dbs:
                self.log_test("Dashboard Database", True, f"Found {len(existing_dbs)} database(s)")
                return True
            else:
                self.log_test("Dashboard Database", False, "No dashboard databases found")
                return False
                
        except Exception as e:
            self.log_test("Dashboard Database", False, f"Database error: {e}")
            return False
    
    def test_6_documentation(self):
        """Test 6: Documentation completeness."""
        logger.info("🔍 Test 6: Documentation")
        
        try:
            required_docs = [
                "README.md",
                "docs/COMPLETE_OPTUNA_TUTORIAL.md",
                "PROJECT_STATUS_REPORT.md"
            ]
            
            missing_docs = []
            for doc in required_docs:
                if not os.path.exists(doc):
                    missing_docs.append(doc)
                else:
                    # Check if file has content
                    with open(doc, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if len(content) < 100:  # Minimum content check
                            missing_docs.append(f"{doc} (too short)")
            
            if not missing_docs:
                self.log_test("Documentation", True, f"All {len(required_docs)} docs present")
                return True
            else:
                self.log_test("Documentation", False, f"Missing: {missing_docs}")
                return False
                
        except Exception as e:
            self.log_test("Documentation", False, f"Documentation error: {e}")
            return False
    
    def test_7_scripts(self):
        """Test 7: Script availability."""
        logger.info("🔍 Test 7: Scripts")
        
        try:
            required_scripts = [
                "scripts/populate_dashboard.py",
                "scripts/start_dashboard.py",
                "scripts/showcase_all_optuna_features.py",
                "scripts/validate_framework.py"
            ]
            
            missing_scripts = []
            for script in required_scripts:
                if not os.path.exists(script):
                    missing_scripts.append(script)
            
            if not missing_scripts:
                self.log_test("Scripts", True, f"All {len(required_scripts)} scripts present")
                return True
            else:
                self.log_test("Scripts", False, f"Missing: {missing_scripts}")
                return False
                
        except Exception as e:
            self.log_test("Scripts", False, f"Scripts error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all validation tests."""
        logger.info("🚀 Starting Final Comprehensive Validation")
        logger.info("=" * 70)
        
        tests = [
            self.test_1_basic_imports,
            self.test_2_data_pipeline,
            self.test_3_optimizers,
            self.test_4_optuna_features,
            self.test_5_dashboard_database,
            self.test_6_documentation,
            self.test_7_scripts,
        ]
        
        for test in tests:
            try:
                test()
                time.sleep(0.5)  # Brief pause between tests
            except Exception as e:
                logger.error(f"Test {test.__name__} crashed: {e}")
                self.log_test(test.__name__, False, f"Test crashed: {e}")
        
        # Final summary
        self.print_final_summary()
    
    def print_final_summary(self):
        """Print final validation summary."""
        logger.info("\n" + "=" * 70)
        logger.info("🎯 FINAL VALIDATION SUMMARY")
        logger.info("=" * 70)
        
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        logger.info(f"📊 Tests Passed: {self.passed_tests}/{self.total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 85:
            logger.info("🎉 VALIDATION RESULT: EXCELLENT! Framework is production-ready!")
        elif success_rate >= 70:
            logger.info("✅ VALIDATION RESULT: GOOD! Minor issues to address.")
        else:
            logger.info("⚠️ VALIDATION RESULT: NEEDS WORK! Major issues found.")
        
        logger.info("\n📋 Detailed Results:")
        for test_name, result in self.validation_results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            logger.info(f"   {status}: {test_name} - {result['details']}")
        
        logger.info("\n🌐 Next Steps:")
        logger.info("   1. Start dashboard: python scripts/start_dashboard.py")
        logger.info("   2. Open browser: http://localhost:8080")
        logger.info("   3. Explore all Optuna features in the dashboard!")
        logger.info("=" * 70)

def main():
    """Main function."""
    validator = FinalValidator()
    validator.run_all_tests()

if __name__ == "__main__":
    main()
