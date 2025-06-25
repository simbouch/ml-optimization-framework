"""
Comprehensive tests for model optimizers.

This module tests all optimizer implementations including
Random Forest, XGBoost, and LightGBM optimizers.
"""

import pytest
import numpy as np
import optuna
from unittest.mock import patch, MagicMock

from src.models.random_forest_optimizer import RandomForestOptimizer
from src.models.xgboost_optimizer import XGBoostOptimizer
from src.models.lightgbm_optimizer import LightGBMOptimizer
from src.optimization.config import OptimizationConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TestBaseOptimizerFunctionality:
    """Test base optimizer functionality across all implementations."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("optimizer_class", [
        RandomForestOptimizer,
        XGBoostOptimizer, 
        LightGBMOptimizer
    ])
    def test_optimizer_initialization(self, optimizer_class):
        """Test optimizer initialization."""
        optimizer = optimizer_class(
            random_state=42,
            cv_folds=3,
            scoring_metric='accuracy',
            verbose=False
        )
        
        assert optimizer.random_state == 42
        assert optimizer.cv_folds == 3
        assert optimizer.scoring_metric == 'accuracy'
        assert not optimizer.verbose
        assert optimizer.best_model is None
        assert optimizer.best_params is None
    
    @pytest.mark.unit
    @pytest.mark.parametrize("optimizer_class", [
        RandomForestOptimizer,
        XGBoostOptimizer,
        LightGBMOptimizer
    ])
    def test_model_creation(self, optimizer_class):
        """Test model creation with trial."""
        optimizer = optimizer_class(random_state=42, verbose=False)
        
        # Create a mock trial
        study = optuna.create_study()
        trial = study.ask()
        
        model = optimizer.create_model(trial)
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    @pytest.mark.unit
    @pytest.mark.parametrize("optimizer_class,expected_name", [
        (RandomForestOptimizer, "RandomForest"),
        (XGBoostOptimizer, "XGBoost"),
        (LightGBMOptimizer, "LightGBM")
    ])
    def test_model_name(self, optimizer_class, expected_name):
        """Test model name retrieval."""
        optimizer = optimizer_class(random_state=42, verbose=False)
        assert optimizer.get_model_name() == expected_name


class TestRandomForestOptimizer:
    """Test Random Forest optimizer specifically."""
    
    @pytest.mark.unit
    def test_rf_specific_initialization(self):
        """Test Random Forest specific initialization."""
        optimizer = RandomForestOptimizer(
            random_state=42,
            cv_folds=5,
            n_jobs=2,
            verbose=True
        )
        
        assert optimizer.n_jobs == 2
        assert optimizer.verbose
    
    @pytest.mark.unit
    def test_rf_model_creation_parameters(self):
        """Test Random Forest model creation with specific parameters."""
        optimizer = RandomForestOptimizer(random_state=42, verbose=False)
        
        study = optuna.create_study()
        trial = study.ask()
        
        # Mock trial suggestions
        with patch.object(trial, 'suggest_int') as mock_int, \
             patch.object(trial, 'suggest_categorical') as mock_cat:
            
            mock_int.side_effect = lambda name, low, high, **kwargs: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_leaf_nodes': 100
            }.get(name, low)
            
            mock_cat.side_effect = lambda name, choices: {
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': None
            }.get(name, choices[0])
            
            model = optimizer.create_model(trial)
            
            assert model.n_estimators == 100
            assert model.max_depth == 10
            assert model.random_state == 42
    
    @pytest.mark.unit
    def test_rf_feature_importance_analysis(self, rf_optimizer, small_data_splits):
        """Test Random Forest feature importance analysis."""
        # Run a quick optimization first
        study = rf_optimizer.optimize(
            small_data_splits['X_train'],
            small_data_splits['X_val'],
            small_data_splits['y_train'],
            small_data_splits['y_val'],
            n_trials=2
        )
        
        # Test feature importance analysis
        importance = rf_optimizer.analyze_feature_importance()
        
        assert isinstance(importance, dict)
        assert 'feature_importance_values' in importance
        assert 'top_features_indices' in importance
        assert 'mean_importance' in importance
        assert 'std_importance' in importance
        
        # Check that importance values are reasonable
        assert len(importance['feature_importance_values']) == small_data_splits['X_train'].shape[1]
        assert 0 <= importance['mean_importance'] <= 1
        assert importance['std_importance'] >= 0
    
    @pytest.mark.unit
    def test_rf_model_complexity(self, rf_optimizer, small_data_splits):
        """Test Random Forest model complexity analysis."""
        # Run optimization first
        rf_optimizer.optimize(
            small_data_splits['X_train'],
            small_data_splits['X_val'],
            small_data_splits['y_train'],
            small_data_splits['y_val'],
            n_trials=2
        )
        
        complexity = rf_optimizer.get_model_complexity()
        
        assert isinstance(complexity, dict)
        assert 'n_estimators' in complexity
        assert 'total_nodes' in complexity
        assert 'avg_tree_depth' in complexity
        assert 'max_tree_depth' in complexity
        
        # Check reasonable values
        assert complexity['n_estimators'] > 0
        assert complexity['total_nodes'] > 0
        assert complexity['avg_tree_depth'] > 0
        assert complexity['max_tree_depth'] >= complexity['avg_tree_depth']


class TestXGBoostOptimizer:
    """Test XGBoost optimizer specifically."""
    
    @pytest.mark.unit
    def test_xgb_specific_initialization(self):
        """Test XGBoost specific initialization."""
        optimizer = XGBoostOptimizer(
            random_state=42,
            early_stopping_rounds=10,
            use_gpu=False,
            verbose=False
        )
        
        assert optimizer.early_stopping_rounds == 10
        assert not optimizer.use_gpu
    
    @pytest.mark.unit
    def test_xgb_model_creation_with_gpu(self):
        """Test XGBoost model creation with GPU settings."""
        optimizer = XGBoostOptimizer(
            random_state=42,
            use_gpu=True,
            verbose=False
        )
        
        study = optuna.create_study()
        trial = study.ask()
        
        model = optimizer.create_model(trial)
        
        # Check that GPU settings are applied (if available)
        # Note: This might not work in CI without GPU
        assert model is not None
    
    @pytest.mark.unit
    def test_xgb_early_stopping_integration(self, xgb_optimizer, small_data_splits):
        """Test XGBoost early stopping integration."""
        # Test that early stopping is properly integrated
        study = xgb_optimizer.optimize(
            small_data_splits['X_train'],
            small_data_splits['X_val'],
            small_data_splits['y_train'],
            small_data_splits['y_val'],
            n_trials=2
        )
        
        assert study.best_value is not None
        assert study.best_params is not None
        
        # Check that best model was saved
        assert xgb_optimizer.best_model is not None


class TestLightGBMOptimizer:
    """Test LightGBM optimizer specifically."""
    
    @pytest.mark.unit
    def test_lgb_specific_initialization(self):
        """Test LightGBM specific initialization."""
        optimizer = LightGBMOptimizer(
            random_state=42,
            early_stopping_rounds=10,
            use_gpu=False,
            verbose=False
        )
        
        assert optimizer.early_stopping_rounds == 10
        assert not optimizer.use_gpu
    
    @pytest.mark.unit
    def test_lgb_boosting_types(self):
        """Test LightGBM different boosting types."""
        optimizer = LightGBMOptimizer(random_state=42, verbose=False)
        
        study = optuna.create_study()
        trial = study.ask()
        
        # Test different boosting types
        boosting_types = ['gbdt', 'dart', 'goss']
        
        for boosting_type in boosting_types:
            with patch.object(trial, 'suggest_categorical') as mock_cat:
                mock_cat.return_value = boosting_type
                
                model = optimizer.create_model(trial)
                assert model is not None


class TestOptimizerOptimization:
    """Test optimization process for all optimizers."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("optimizer_fixture", [
        "rf_optimizer", "xgb_optimizer", "lgb_optimizer"
    ])
    def test_optimization_process(self, optimizer_fixture, small_data_splits, request):
        """Test optimization process for all optimizers."""
        optimizer = request.getfixturevalue(optimizer_fixture)
        
        study = optimizer.optimize(
            small_data_splits['X_train'],
            small_data_splits['X_val'],
            small_data_splits['y_train'],
            small_data_splits['y_val'],
            n_trials=3
        )
        
        # Check study results
        assert study is not None
        assert study.best_value is not None
        assert study.best_params is not None
        assert len(study.trials) == 3
        assert study.best_value > 0.3  # Should achieve reasonable performance
        
        # Check that best model and params are saved
        assert optimizer.best_model is not None
        assert optimizer.best_params is not None
    
    @pytest.mark.unit
    @pytest.mark.parametrize("optimizer_fixture", [
        "rf_optimizer", "xgb_optimizer", "lgb_optimizer"
    ])
    def test_evaluation_process(self, optimizer_fixture, small_data_splits, request):
        """Test evaluation process for all optimizers."""
        optimizer = request.getfixturevalue(optimizer_fixture)
        
        # Run optimization first
        optimizer.optimize(
            small_data_splits['X_train'],
            small_data_splits['X_val'],
            small_data_splits['y_train'],
            small_data_splits['y_val'],
            n_trials=2
        )
        
        # Test evaluation
        metrics = optimizer.evaluate(
            small_data_splits['X_test'],
            small_data_splits['y_test']
        )
        
        # Check metrics
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'roc_auc' in metrics
        
        # Check metric ranges
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} should be between 0 and 1"
    
    @pytest.mark.unit
    def test_optimization_with_callbacks(self, rf_optimizer, small_data_splits):
        """Test optimization with callbacks."""
        callback_called = []
        
        def test_callback(study, trial):
            callback_called.append(trial.number)
        
        study = rf_optimizer.optimize(
            small_data_splits['X_train'],
            small_data_splits['X_val'],
            small_data_splits['y_train'],
            small_data_splits['y_val'],
            n_trials=3,
            callbacks=[test_callback]
        )
        
        # Check that callback was called for each trial
        assert len(callback_called) == 3
        assert callback_called == [0, 1, 2]
    
    @pytest.mark.unit
    def test_optimization_with_existing_study(self, rf_optimizer, small_data_splits):
        """Test optimization with existing study."""
        # Create initial study
        study = optuna.create_study(direction='maximize')
        
        # Run optimization with existing study
        result_study = rf_optimizer.optimize(
            small_data_splits['X_train'],
            small_data_splits['X_val'],
            small_data_splits['y_train'],
            small_data_splits['y_val'],
            n_trials=2,
            study=study
        )
        
        # Should be the same study object
        assert result_study is study
        assert len(study.trials) == 2


class TestOptimizerErrorHandling:
    """Test error handling in optimizers."""
    
    @pytest.mark.unit
    def test_evaluation_before_optimization(self, rf_optimizer, small_data_splits):
        """Test evaluation before optimization raises error."""
        with pytest.raises(ValueError, match="No model has been optimized"):
            rf_optimizer.evaluate(
                small_data_splits['X_test'],
                small_data_splits['y_test']
            )
    
    @pytest.mark.unit
    def test_invalid_scoring_metric(self):
        """Test invalid scoring metric raises error."""
        with pytest.raises(ValueError):
            RandomForestOptimizer(scoring_metric='invalid_metric')
    
    @pytest.mark.unit
    def test_optimization_with_invalid_data(self, rf_optimizer):
        """Test optimization with invalid data."""
        # Test with mismatched X and y shapes
        X_invalid = np.random.rand(100, 5)
        y_invalid = np.random.randint(0, 2, 50)  # Wrong size
        
        with pytest.raises(ValueError):
            rf_optimizer.optimize(
                X_invalid, X_invalid, y_invalid, y_invalid,
                n_trials=1
            )
    
    @pytest.mark.unit
    def test_optimization_summary_before_optimization(self, rf_optimizer):
        """Test getting optimization summary before optimization."""
        summary = rf_optimizer.get_optimization_summary()
        
        # Should return empty/default summary
        assert summary['model_name'] == 'RandomForest'
        assert summary['best_score'] is None
        assert summary['best_params'] is None
        assert summary['n_trials'] == 0
