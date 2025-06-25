"""
Comprehensive tests for the ML optimization framework.

This module provides comprehensive pytest-based tests to validate
all framework functionality and ensure reproducibility.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import time
from typing import Dict, Any

from src.data.data_pipeline import DataPipeline
from src.models.random_forest_optimizer import RandomForestOptimizer
from src.models.xgboost_optimizer import XGBoostOptimizer
from src.models.lightgbm_optimizer import LightGBMOptimizer
from src.optimization.config import OptimizationConfig
from src.optimization.study_manager import StudyManager
from src.visualization.plots import OptimizationPlotter
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TestDataPipeline:
    """Test data pipeline functionality."""

    @pytest.mark.unit
    def test_data_loading(self, data_pipeline):
        """Test data loading functionality."""
        X, y = data_pipeline.load_data()

        # Check data shapes
        assert X.shape[0] > 1000  # Should have substantial data
        assert X.shape[1] > 5     # Should have multiple features
        assert len(y) == X.shape[0]  # Target should match features

        # Check target values
        unique_targets = np.unique(y)
        assert len(unique_targets) == 2  # Binary classification

    @pytest.mark.unit
    def test_data_analysis(self, data_pipeline):
        """Test data analysis functionality."""
        data_pipeline.load_data()
        analysis = data_pipeline.analyze_data()

        # Check analysis structure
        required_keys = [
            'shape', 'missing_values', 'data_types',
            'categorical_features', 'numerical_features',
            'target_distribution', 'duplicate_rows'
        ]

        for key in required_keys:
            assert key in analysis

        # Check feature categorization
        assert isinstance(analysis['categorical_features'], list)
        assert isinstance(analysis['numerical_features'], list)
        assert len(analysis['categorical_features']) > 0
        assert len(analysis['numerical_features']) > 0

    @pytest.mark.unit
    def test_data_preparation(self, data_pipeline):
        """Test complete data preparation."""
        summary = data_pipeline.prepare_data()

        # Check summary structure
        required_keys = [
            'total_samples', 'total_features', 'train_samples',
            'val_samples', 'test_samples', 'preprocessing_complete'
        ]

        for key in required_keys:
            assert key in summary

        # Check data splits
        X_train, X_val, y_train, y_val = data_pipeline.get_train_val_data()
        X_test, y_test = data_pipeline.get_test_data()

        # Verify shapes
        assert X_train.shape[0] == len(y_train)
        assert X_val.shape[0] == len(y_val)
        assert X_test.shape[0] == len(y_test)

        # Verify feature consistency
        assert X_train.shape[1] == X_val.shape[1]
        assert X_train.shape[1] == X_test.shape[1]

    @pytest.mark.integration
    def test_data_pipeline_reproducibility(self):
        """Test that data pipeline produces reproducible results."""
        pipeline1 = DataPipeline(random_state=42)
        pipeline2 = DataPipeline(random_state=42)

        summary1 = pipeline1.prepare_data()
        summary2 = pipeline2.prepare_data()

        # Should have same summary
        assert summary1['total_samples'] == summary2['total_samples']
        assert summary1['total_features'] == summary2['total_features']

        # Should have same splits
        X_train1, _, y_train1, _ = pipeline1.get_train_val_data()
        X_train2, _, y_train2, _ = pipeline2.get_train_val_data()

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(y_train1, y_train2)


class TestOptimizers:
    """Test model optimizers."""

    @pytest.mark.unit
    def test_random_forest_optimizer(self, small_data_splits, test_utils):
        """Test Random Forest optimizer."""
        optimizer = RandomForestOptimizer(
            random_state=42,
            cv_folds=3,
            verbose=False
        )

        # Test optimization
        study = optimizer.optimize(
            small_data_splits['X_train'],
            small_data_splits['X_val'],
            small_data_splits['y_train'],
            small_data_splits['y_val'],
            n_trials=3
        )

        # Check results
        test_utils.assert_valid_study(study)
        assert study.best_value > 0.5  # Should be reasonable accuracy

        # Test evaluation
        metrics = optimizer.evaluate(
            small_data_splits['X_test'],
            small_data_splits['y_test']
        )
        test_utils.assert_valid_metrics(metrics)

    @pytest.mark.unit
    def test_xgboost_optimizer(self, small_data_splits, test_utils):
        """Test XGBoost optimizer."""
        optimizer = XGBoostOptimizer(
            random_state=42,
            cv_folds=3,
            verbose=False
        )

        # Test optimization
        study = optimizer.optimize(
            small_data_splits['X_train'],
            small_data_splits['X_val'],
            small_data_splits['y_train'],
            small_data_splits['y_val'],
            n_trials=3
        )

        # Check results
        test_utils.assert_valid_study(study)
        assert study.best_value > 0.5

        # Test evaluation
        metrics = optimizer.evaluate(
            small_data_splits['X_test'],
            small_data_splits['y_test']
        )
        test_utils.assert_valid_metrics(metrics)

    @pytest.mark.unit
    def test_lightgbm_optimizer(self, small_data_splits, test_utils):
        """Test LightGBM optimizer."""
        optimizer = LightGBMOptimizer(
            random_state=42,
            cv_folds=3,
            verbose=False
        )

        # Test optimization
        study = optimizer.optimize(
            small_data_splits['X_train'],
            small_data_splits['X_val'],
            small_data_splits['y_train'],
            small_data_splits['y_val'],
            n_trials=3
        )

        # Check results
        test_utils.assert_valid_study(study)
        assert study.best_value > 0.5

        # Test evaluation
        metrics = optimizer.evaluate(
            small_data_splits['X_test'],
            small_data_splits['y_test']
        )
        test_utils.assert_valid_metrics(metrics)

    @pytest.mark.unit
    @pytest.mark.parametrize("optimizer_name", ["random_forest", "xgboost", "lightgbm"])
    def test_all_optimizers(self, optimizer_name, all_optimizers, small_data_splits, test_utils):
        """Test all optimizers with parametrized test."""
        optimizer = all_optimizers[optimizer_name]
        test_utils.assert_valid_optimizer(optimizer)

        # Test model creation
        import optuna
        study = optuna.create_study()
        trial = study.ask()

        model = optimizer.create_model(trial)
        assert model is not None

        # Test model name
        model_name = optimizer.get_model_name()
        assert isinstance(model_name, str)
        assert len(model_name) > 0


class TestConfiguration:
    """Test optimization configuration."""

    @pytest.mark.unit
    def test_default_config(self, optimization_config):
        """Test default configuration."""
        # Check available models
        models = optimization_config.get_available_models()
        expected_models = ['random_forest', 'xgboost', 'lightgbm']

        for model in expected_models:
            assert model in models

        # Check hyperparameter spaces
        for model in expected_models:
            space = optimization_config.get_hyperparameter_space(model)
            assert isinstance(space, dict)
            assert len(space) > 0

    @pytest.mark.unit
    def test_config_validation(self, optimization_config):
        """Test configuration validation."""
        # Should validate successfully
        assert optimization_config.validate_config()

    @pytest.mark.unit
    def test_hyperparameter_suggestion(self, optimization_config):
        """Test hyperparameter suggestion."""
        import optuna

        study = optuna.create_study()
        trial = study.ask()

        # Test suggestion for each model
        for model_name in optimization_config.get_available_models():
            params = optimization_config.suggest_hyperparameters(trial, model_name)
            assert isinstance(params, dict)
            assert len(params) > 0


class TestStudyManager:
    """Test study manager functionality."""

    @pytest.mark.unit
    def test_study_creation(self, study_manager):
        """Test study creation."""
        study = study_manager.create_study(
            study_name="test_study",
            direction="maximize"
        )

        assert study is not None
        assert study.study_name == "test_study"

    @pytest.mark.unit
    def test_study_listing(self, study_manager):
        """Test study listing."""
        # Create a study first
        study_manager.create_study("test_list_study")

        # List studies
        studies = study_manager.list_studies()
        assert isinstance(studies, list)


class TestReproducibility:
    """Test reproducibility of results."""

    @pytest.mark.integration
    def test_reproducible_optimization(self):
        """Test that optimization results are reproducible."""
        # Setup
        pipeline1 = DataPipeline(random_state=42)
        pipeline1.prepare_data()
        X_train1, X_val1, y_train1, y_val1 = pipeline1.get_train_val_data()

        pipeline2 = DataPipeline(random_state=42)
        pipeline2.prepare_data()
        X_train2, _, y_train2, _ = pipeline2.get_train_val_data()

        # Check data consistency
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(y_train1, y_train2)

        # Test optimizer reproducibility
        optimizer1 = RandomForestOptimizer(random_state=42, verbose=False)
        optimizer2 = RandomForestOptimizer(random_state=42, verbose=False)

        # Use small subset for faster testing
        X_small = X_train1[:100]
        y_small = y_train1[:100]
        X_val_small = X_val1[:50]
        y_val_small = y_val1[:50]

        study1 = optimizer1.optimize(X_small, X_val_small, y_small, y_val_small, n_trials=2)
        study2 = optimizer2.optimize(X_small, X_val_small, y_small, y_val_small, n_trials=2)

        # Results should be identical with same random seed
        assert study1.best_value == study2.best_value


# Smoke tests for quick validation
@pytest.mark.smoke
def test_basic_framework_validation():
    """Run basic validation of the framework."""
    logger.info("🧪 Running basic framework validation...")

    # Test data pipeline
    logger.info("   📊 Testing data pipeline...")
    pipeline = DataPipeline(random_state=42)
    summary = pipeline.prepare_data()
    assert summary['total_samples'] > 0
    logger.info(f"   ✅ Data pipeline: {summary['total_samples']} samples loaded")

    # Test optimizer
    logger.info("   🌲 Testing Random Forest optimizer...")
    X_train, X_val, y_train, y_val = pipeline.get_train_val_data()

    # Use small subset for quick validation
    X_train_small = X_train[:100]
    y_train_small = y_train[:100]
    X_val_small = X_val[:50]
    y_val_small = y_val[:50]

    optimizer = RandomForestOptimizer(random_state=42, verbose=False)
    study = optimizer.optimize(
        X_train_small, X_val_small, y_train_small, y_val_small,
        n_trials=2
    )

    assert study.best_value > 0.5
    logger.info(f"   ✅ Optimizer: Best score {study.best_value:.4f}")

    # Test configuration
    logger.info("   ⚙️ Testing configuration...")
    config = OptimizationConfig()
    models = config.get_available_models()
    assert len(models) >= 3
    logger.info(f"   ✅ Configuration: {len(models)} models available")

    logger.info("🎉 Basic validation completed successfully!")


@pytest.mark.integration
@pytest.mark.slow
def test_full_optimization_pipeline():
    """Test complete optimization pipeline."""
    logger.info("🚀 Testing full optimization pipeline...")

    # Setup
    pipeline = DataPipeline(random_state=42)
    pipeline.prepare_data()
    X_train, X_val, y_train, y_val = pipeline.get_train_val_data()
    X_test, y_test = pipeline.get_test_data()

    # Use subset for faster testing
    X_train = X_train[:500]
    y_train = y_train[:500]
    X_val = X_val[:100]
    y_val = y_val[:100]
    X_test = X_test[:100]
    y_test = y_test[:100]

    # Test Random Forest
    rf_optimizer = RandomForestOptimizer(random_state=42, verbose=False)
    rf_study = rf_optimizer.optimize(X_train, X_val, y_train, y_val, n_trials=5)
    rf_metrics = rf_optimizer.evaluate(X_test, y_test)

    assert rf_study.best_value > 0.5
    assert rf_metrics['accuracy'] > 0.5

    logger.info(f"✅ Random Forest: CV={rf_study.best_value:.3f}, Test={rf_metrics['accuracy']:.3f}")

    # Test XGBoost
    xgb_optimizer = XGBoostOptimizer(random_state=42, verbose=False)
    xgb_study = xgb_optimizer.optimize(X_train, X_val, y_train, y_val, n_trials=5)
    xgb_metrics = xgb_optimizer.evaluate(X_test, y_test)

    assert xgb_study.best_value > 0.5
    assert xgb_metrics['accuracy'] > 0.5

    logger.info(f"✅ XGBoost: CV={xgb_study.best_value:.3f}, Test={xgb_metrics['accuracy']:.3f}")

    logger.info("🎉 Full pipeline test completed successfully!")
