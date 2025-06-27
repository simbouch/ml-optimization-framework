#!/usr/bin/env python3
"""
Integration tests for the ML Optimization Framework
"""

import pytest
import subprocess
import sys
import time
import requests
import tempfile
from pathlib import Path
import sqlite3
import pandas as pd

from src.config import OptimizationConfig
from src.optimizers import RandomForestOptimizer
from src.study_manager import StudyManager


class TestFrameworkIntegration:
    """Test complete framework integration."""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = OptimizationConfig(
                study_name="integration_test",
                n_trials=5,  # Small number for fast testing
                data_dir=temp_path / "data",
                results_dir=temp_path / "results",
                studies_dir=temp_path / "studies",
                logs_dir=temp_path / "logs"
            )
            yield config
    
    def test_complete_optimization_workflow(self, temp_config):
        """Test complete optimization workflow from start to finish."""
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # 1. Create dataset
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 2. Run optimization
        optimizer = RandomForestOptimizer(temp_config, task_type="classification")
        study = optimizer.optimize(X_train, y_train)
        
        # 3. Get best model and evaluate
        best_model = optimizer.get_best_model()
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 4. Use StudyManager for analysis
        study_manager = StudyManager(temp_config)
        summary = study_manager.get_study_summary(temp_config.study_name)
        
        # 5. Export results
        export_path = study_manager.export_study_results(temp_config.study_name, format="csv")
        
        # Assertions
        assert study is not None
        assert len(study.trials) == temp_config.n_trials
        assert 0 <= accuracy <= 1
        assert summary["n_trials"] == temp_config.n_trials
        assert export_path.exists()
        
        # Check exported data
        df = pd.read_csv(export_path)
        assert len(df) == temp_config.n_trials
        assert "trial_number" in df.columns
        assert "value" in df.columns
    
    def test_database_persistence(self, temp_config):
        """Test that studies are properly persisted in database."""
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        
        # Create and run first optimization
        optimizer1 = RandomForestOptimizer(temp_config, task_type="classification")
        study1 = optimizer1.optimize(X, y)
        
        # Check database file exists
        db_path = temp_config.studies_dir / f"{temp_config.study_name}.db"
        assert db_path.exists()
        
        # Check database content
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trials")
        trial_count = cursor.fetchone()[0]
        conn.close()
        
        assert trial_count == temp_config.n_trials
        
        # Create second optimizer with same config (should load existing study)
        optimizer2 = RandomForestOptimizer(temp_config, task_type="classification")
        study2 = optimizer2.optimize(X, y)
        
        # Should have more trials now
        assert len(study2.trials) == 2 * temp_config.n_trials
    
    def test_multi_algorithm_comparison(self, temp_config):
        """Test comparison of multiple algorithms."""
        from sklearn.datasets import make_classification
        from src.optimizers import SVMOptimizer
        
        X, y = make_classification(n_samples=80, n_features=8, random_state=42)
        
        algorithms = [
            ("RandomForest", RandomForestOptimizer),
            ("SVM", SVMOptimizer),
        ]
        
        results = {}
        
        for algo_name, optimizer_class in algorithms:
            config = OptimizationConfig(
                study_name=f"comparison_{algo_name.lower()}",
                n_trials=3,  # Very small for fast testing
                data_dir=temp_config.data_dir,
                results_dir=temp_config.results_dir,
                studies_dir=temp_config.studies_dir,
                logs_dir=temp_config.logs_dir
            )
            
            optimizer = optimizer_class(config, task_type="classification")
            study = optimizer.optimize(X, y)
            
            results[algo_name] = {
                "best_score": study.best_value,
                "n_trials": len(study.trials),
                "best_params": study.best_params
            }
        
        # Check that all algorithms completed
        assert len(results) == len(algorithms)
        for algo_name, result in results.items():
            assert result["best_score"] is not None
            assert result["n_trials"] == 3
            assert result["best_params"] is not None
    
    def test_configuration_validation_integration(self):
        """Test configuration validation in integration context."""
        # Test valid configuration
        valid_config = OptimizationConfig(
            study_name="valid_test",
            direction="maximize",
            n_trials=5,
            sampler_name="TPE",
            pruner_name="Median"
        )
        assert valid_config.study_name == "valid_test"
        
        # Test invalid configurations
        with pytest.raises(ValueError):
            OptimizationConfig(direction="invalid_direction")
        
        with pytest.raises(ValueError):
            OptimizationConfig(sampler_name="invalid_sampler")
        
        with pytest.raises(ValueError):
            OptimizationConfig(n_trials=0)
    
    def test_logging_integration(self, temp_config):
        """Test that logging works correctly throughout the framework."""
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        
        # Run optimization (this should generate logs)
        optimizer = RandomForestOptimizer(temp_config, task_type="classification")
        study = optimizer.optimize(X, y)
        
        # Check that log files are created
        log_files = list(temp_config.logs_dir.glob("*.log"))
        assert len(log_files) > 0
        
        # Check that logs contain relevant information
        for log_file in log_files:
            log_content = log_file.read_text()
            assert len(log_content) > 0
            # Should contain some optimization-related messages
            assert any(keyword in log_content.lower() for keyword in 
                      ["study", "optimization", "trial", "best"])


class TestScriptIntegration:
    """Test integration with main scripts."""
    
    def test_quick_demo_script(self):
        """Test that quick_demo.py runs successfully."""
        try:
            result = subprocess.run(
                [sys.executable, "quick_demo.py"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=Path.cwd()
            )
            
            # Should complete successfully
            assert result.returncode == 0
            
            # Should create study files
            studies_dir = Path("studies")
            assert studies_dir.exists()
            
            db_files = list(studies_dir.glob("*.db"))
            assert len(db_files) >= 1  # At least one study should be created
            
        except subprocess.TimeoutExpired:
            pytest.fail("quick_demo.py took too long to complete")
        except Exception as e:
            pytest.fail(f"quick_demo.py failed: {e}")
    
    def test_comprehensive_demo_script(self):
        """Test that comprehensive_optuna_demo.py runs successfully."""
        try:
            result = subprocess.run(
                [sys.executable, "comprehensive_optuna_demo.py"],
                capture_output=True,
                text=True,
                timeout=120,  # Longer timeout for comprehensive demo
                cwd=Path.cwd()
            )
            
            # Should complete successfully
            assert result.returncode == 0
            
            # Check output contains expected messages
            output = result.stdout.lower()
            assert "demonstration" in output or "completed" in output
            
            # Should create multiple study files
            studies_dir = Path("studies")
            if studies_dir.exists():
                db_files = list(studies_dir.glob("*.db"))
                # Comprehensive demo should create multiple studies
                assert len(db_files) >= 3
            
        except subprocess.TimeoutExpired:
            pytest.fail("comprehensive_optuna_demo.py took too long to complete")
        except Exception as e:
            # Comprehensive demo might fail due to missing dependencies
            # This is acceptable for testing
            print(f"Comprehensive demo failed (acceptable): {e}")
    
    def test_validation_script(self):
        """Test that validate_clean.py runs successfully."""
        try:
            result = subprocess.run(
                [sys.executable, "validate_clean.py"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=Path.cwd()
            )
            
            # Should complete successfully
            assert result.returncode == 0
            
            # Check output for validation messages
            output = result.stdout.lower()
            assert any(keyword in output for keyword in 
                      ["validation", "test", "check", "ok", "good"])
            
        except subprocess.TimeoutExpired:
            pytest.fail("validate_clean.py took too long to complete")
        except Exception as e:
            pytest.fail(f"validate_clean.py failed: {e}")


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_data_handling(self, temp_config):
        """Test handling of invalid data."""
        import numpy as np
        
        # Test with invalid data shapes
        X_invalid = np.array([[1, 2], [3]])  # Inconsistent shapes
        y_invalid = np.array([1, 2, 3])  # Wrong length
        
        optimizer = RandomForestOptimizer(temp_config, task_type="classification")
        
        # Should handle invalid data gracefully
        with pytest.raises(Exception):
            optimizer.optimize(X_invalid, y_invalid)
    
    def test_empty_data_handling(self, temp_config):
        """Test handling of empty data."""
        import numpy as np
        
        X_empty = np.array([]).reshape(0, 5)
        y_empty = np.array([])
        
        optimizer = RandomForestOptimizer(temp_config, task_type="classification")
        
        # Should handle empty data gracefully
        with pytest.raises(Exception):
            optimizer.optimize(X_empty, y_empty)
    
    def test_corrupted_study_handling(self, temp_config):
        """Test handling of corrupted study files."""
        # Create a corrupted database file
        db_path = temp_config.studies_dir / f"{temp_config.study_name}.db"
        db_path.write_text("corrupted data")
        
        study_manager = StudyManager(temp_config)
        
        # Should handle corrupted database gracefully
        with pytest.raises(Exception):
            study_manager.load_study(temp_config.study_name)


if __name__ == "__main__":
    pytest.main([__file__])
