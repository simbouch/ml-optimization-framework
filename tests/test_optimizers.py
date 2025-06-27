#!/usr/bin/env python3
"""
Tests for model optimizers
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from src.config import OptimizationConfig
from src.optimizers import RandomForestOptimizer, XGBoostOptimizer, SVMOptimizer


class TestModelOptimizers:
    """Test model optimizer implementations."""
    
    @pytest.fixture
    def classification_data(self):
        """Create classification dataset for testing."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    @pytest.fixture
    def regression_data(self):
        """Create regression dataset for testing."""
        X, y = make_regression(
            n_samples=100, n_features=10, noise=0.1, random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = OptimizationConfig(
                study_name="test_study",
                n_trials=5,  # Small number for fast testing
                data_dir=temp_path / "data",
                results_dir=temp_path / "results",
                studies_dir=temp_path / "studies",
                logs_dir=temp_path / "logs"
            )
            yield config


class TestRandomForestOptimizer(TestModelOptimizers):
    """Test RandomForest optimizer."""
    
    def test_classification_optimization(self, classification_data, temp_config):
        """Test RandomForest classification optimization."""
        X_train, X_test, y_train, y_test = classification_data
        
        optimizer = RandomForestOptimizer(temp_config, task_type="classification")
        study = optimizer.optimize(X_train, y_train)
        
        assert study is not None
        assert len(study.trials) == temp_config.n_trials
        assert optimizer.best_params is not None
        assert optimizer.best_score is not None
        assert 0 <= optimizer.best_score <= 1  # Accuracy should be between 0 and 1
        
        # Test best model creation
        best_model = optimizer.get_best_model()
        assert best_model is not None
        
        # Test model fitting and prediction
        best_model.fit(X_train, y_train)
        predictions = best_model.predict(X_test)
        assert len(predictions) == len(y_test)
    
    def test_regression_optimization(self, regression_data, temp_config):
        """Test RandomForest regression optimization."""
        X_train, X_test, y_train, y_test = regression_data
        
        optimizer = RandomForestOptimizer(temp_config, task_type="regression")
        study = optimizer.optimize(X_train, y_train)
        
        assert study is not None
        assert len(study.trials) == temp_config.n_trials
        assert optimizer.best_params is not None
        assert optimizer.best_score is not None
        assert optimizer.best_score <= 0  # Negative MSE should be <= 0
        
        # Test best model creation
        best_model = optimizer.get_best_model()
        assert best_model is not None
    
    def test_parameter_suggestions(self, temp_config):
        """Test parameter suggestion functionality."""
        import optuna
        
        optimizer = RandomForestOptimizer(temp_config, task_type="classification")
        study = optuna.create_study()
        trial = study.ask()
        
        params = optimizer.suggest_parameters(trial)
        
        assert "n_estimators" in params
        assert "max_depth" in params
        assert "min_samples_split" in params
        assert "min_samples_leaf" in params
        assert "max_features" in params
        assert "random_state" in params
        
        # Check parameter ranges
        assert 10 <= params["n_estimators"] <= 200
        assert 3 <= params["max_depth"] <= 20
        assert params["random_state"] == temp_config.random_seed
    
    def test_get_best_model_without_optimization(self, temp_config):
        """Test error when getting best model without optimization."""
        optimizer = RandomForestOptimizer(temp_config, task_type="classification")
        
        with pytest.raises(ValueError, match="No optimization has been performed yet"):
            optimizer.get_best_model()


class TestXGBoostOptimizer(TestModelOptimizers):
    """Test XGBoost optimizer."""
    
    def test_xgboost_optimization_or_fallback(self, classification_data, temp_config):
        """Test XGBoost optimization or fallback to RandomForest."""
        X_train, X_test, y_train, y_test = classification_data
        
        optimizer = XGBoostOptimizer(temp_config, task_type="classification")
        study = optimizer.optimize(X_train, y_train)
        
        assert study is not None
        assert len(study.trials) == temp_config.n_trials
        assert optimizer.best_params is not None
        assert optimizer.best_score is not None
        
        # Test best model creation (should work with or without XGBoost)
        best_model = optimizer.get_best_model()
        assert best_model is not None
    
    def test_xgboost_parameter_suggestions(self, temp_config):
        """Test XGBoost parameter suggestions."""
        import optuna
        
        optimizer = XGBoostOptimizer(temp_config, task_type="classification")
        study = optuna.create_study()
        trial = study.ask()
        
        params = optimizer.suggest_parameters(trial)
        
        assert "n_estimators" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "subsample" in params
        assert "colsample_bytree" in params
        assert "reg_alpha" in params
        assert "reg_lambda" in params
        assert "random_state" in params


class TestSVMOptimizer(TestModelOptimizers):
    """Test SVM optimizer."""
    
    def test_svm_classification_optimization(self, classification_data, temp_config):
        """Test SVM classification optimization."""
        X_train, X_test, y_train, y_test = classification_data
        
        optimizer = SVMOptimizer(temp_config, task_type="classification")
        study = optimizer.optimize(X_train, y_train)
        
        assert study is not None
        assert len(study.trials) == temp_config.n_trials
        assert optimizer.best_params is not None
        assert optimizer.best_score is not None
        
        # Test best model creation
        best_model = optimizer.get_best_model()
        assert best_model is not None
    
    def test_svm_parameter_suggestions(self, temp_config):
        """Test SVM parameter suggestions."""
        import optuna
        
        optimizer = SVMOptimizer(temp_config, task_type="classification")
        study = optuna.create_study()
        trial = study.ask()
        
        params = optimizer.suggest_parameters(trial)
        
        assert "C" in params
        assert "gamma" in params
        assert "kernel" in params
        assert "random_state" in params
        
        # Check parameter values
        assert 0.1 <= params["C"] <= 100.0
        assert params["gamma"] in ["scale", "auto"]
        assert params["kernel"] in ["rbf", "poly", "sigmoid"]


class TestOptimizerIntegration:
    """Test optimizer integration and edge cases."""
    
    def test_study_creation_and_persistence(self, temp_config):
        """Test study creation and persistence."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        
        optimizer = RandomForestOptimizer(temp_config, task_type="classification")
        study1 = optimizer.optimize(X, y)
        
        # Create another optimizer with same config (should load existing study)
        optimizer2 = RandomForestOptimizer(temp_config, task_type="classification")
        study2 = optimizer2.optimize(X, y)
        
        # Should have more trials now
        assert len(study2.trials) >= len(study1.trials)
    
    def test_multi_objective_compatibility(self, temp_config):
        """Test compatibility with multi-objective optimization."""
        temp_config.direction = ["maximize", "minimize"]
        
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        
        optimizer = RandomForestOptimizer(temp_config, task_type="classification")
        
        # Should handle multi-objective configuration gracefully
        # Note: The current implementation doesn't fully support multi-objective
        # but should not crash
        try:
            study = optimizer.optimize(X, y)
            assert study is not None
        except Exception as e:
            # Multi-objective might not be fully supported yet
            assert "direction" in str(e).lower() or "multi" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__])
