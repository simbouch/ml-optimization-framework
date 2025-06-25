"""
Pytest configuration and fixtures.

This module provides shared fixtures and configuration for all tests
in the ML optimization framework.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from typing import Tuple, Dict, Any
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_testing_logging
from src.data.data_pipeline import DataPipeline
from src.models.random_forest_optimizer import RandomForestOptimizer
from src.models.xgboost_optimizer import XGBoostOptimizer
from src.models.lightgbm_optimizer import LightGBMOptimizer
from src.optimization.config import OptimizationConfig
from src.optimization.study_manager import StudyManager

# Setup testing logging
setup_testing_logging()


@pytest.fixture(scope="session")
def sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample dataset for testing.
    
    Returns:
        Tuple of (X, y) arrays
    """
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with some signal
    weights = np.random.randn(n_features)
    y_continuous = X @ weights + np.random.randn(n_samples) * 0.1
    y = (y_continuous > np.median(y_continuous)).astype(int)
    
    return X, y


@pytest.fixture(scope="session")
def small_sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create small sample dataset for fast testing.
    
    Returns:
        Tuple of (X, y) arrays
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    weights = np.random.randn(n_features)
    y_continuous = X @ weights + np.random.randn(n_samples) * 0.1
    y = (y_continuous > np.median(y_continuous)).astype(int)
    
    return X, y


@pytest.fixture
def data_splits(sample_data) -> Dict[str, np.ndarray]:
    """
    Create train/validation/test splits.
    
    Args:
        sample_data: Sample dataset fixture
        
    Returns:
        Dictionary with data splits
    """
    X, y = sample_data
    
    # Simple split
    n_train = int(0.6 * len(X))
    n_val = int(0.2 * len(X))
    
    return {
        'X_train': X[:n_train],
        'X_val': X[n_train:n_train+n_val],
        'X_test': X[n_train+n_val:],
        'y_train': y[:n_train],
        'y_val': y[n_train:n_train+n_val],
        'y_test': y[n_train+n_val:]
    }


@pytest.fixture
def small_data_splits(small_sample_data) -> Dict[str, np.ndarray]:
    """
    Create small train/validation/test splits for fast testing.
    
    Args:
        small_sample_data: Small sample dataset fixture
        
    Returns:
        Dictionary with small data splits
    """
    X, y = small_sample_data
    
    # Simple split
    n_train = int(0.6 * len(X))
    n_val = int(0.2 * len(X))
    
    return {
        'X_train': X[:n_train],
        'X_val': X[n_train:n_train+n_val],
        'X_test': X[n_train+n_val:],
        'y_train': y[:n_train],
        'y_val': y[n_train:n_train+n_val],
        'y_test': y[n_train+n_val:]
    }


@pytest.fixture
def temp_dir():
    """
    Create temporary directory for testing.
    
    Yields:
        Path to temporary directory
    """
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def optimization_config():
    """
    Create optimization configuration for testing.
    
    Returns:
        OptimizationConfig instance
    """
    return OptimizationConfig()


@pytest.fixture
def study_manager(temp_dir):
    """
    Create study manager with temporary storage.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        StudyManager instance
    """
    storage_url = f"sqlite:///{temp_dir}/test_study.db"
    return StudyManager(storage_url=storage_url)


@pytest.fixture
def rf_optimizer():
    """
    Create Random Forest optimizer for testing.
    
    Returns:
        RandomForestOptimizer instance
    """
    return RandomForestOptimizer(
        random_state=42,
        cv_folds=3,  # Reduced for faster testing
        verbose=False
    )


@pytest.fixture
def xgb_optimizer():
    """
    Create XGBoost optimizer for testing.
    
    Returns:
        XGBoostOptimizer instance
    """
    return XGBoostOptimizer(
        random_state=42,
        cv_folds=3,  # Reduced for faster testing
        early_stopping_rounds=5,
        verbose=False
    )


@pytest.fixture
def lgb_optimizer():
    """
    Create LightGBM optimizer for testing.
    
    Returns:
        LightGBMOptimizer instance
    """
    return LightGBMOptimizer(
        random_state=42,
        cv_folds=3,  # Reduced for faster testing
        early_stopping_rounds=5,
        verbose=False
    )


@pytest.fixture
def all_optimizers(rf_optimizer, xgb_optimizer, lgb_optimizer):
    """
    Create dictionary of all optimizers for testing.
    
    Returns:
        Dictionary of optimizer instances
    """
    return {
        'random_forest': rf_optimizer,
        'xgboost': xgb_optimizer,
        'lightgbm': lgb_optimizer
    }


@pytest.fixture
def data_pipeline():
    """
    Create data pipeline for testing.
    
    Returns:
        DataPipeline instance
    """
    return DataPipeline(random_state=42, test_size=0.2, val_size=0.2)


# Pytest markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "smoke: Smoke tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "network: Tests requiring network")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(marker.name in ['integration', 'smoke', 'slow'] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to tests with 'slow' in name
        if 'slow' in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to tests with 'integration' in name
        if 'integration' in item.name:
            item.add_marker(pytest.mark.integration)


# Test utilities
class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def assert_valid_study(study):
        """Assert that a study is valid."""
        assert study is not None
        assert hasattr(study, 'best_value')
        assert hasattr(study, 'best_params')
        assert hasattr(study, 'trials')
    
    @staticmethod
    def assert_valid_metrics(metrics):
        """Assert that metrics dictionary is valid."""
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        
        if 'f1_score' in metrics:
            assert 0 <= metrics['f1_score'] <= 1
        if 'precision' in metrics:
            assert 0 <= metrics['precision'] <= 1
        if 'recall' in metrics:
            assert 0 <= metrics['recall'] <= 1
        if 'roc_auc' in metrics:
            assert 0 <= metrics['roc_auc'] <= 1
    
    @staticmethod
    def assert_valid_optimizer(optimizer):
        """Assert that an optimizer is valid."""
        assert optimizer is not None
        assert hasattr(optimizer, 'optimize')
        assert hasattr(optimizer, 'evaluate')
        assert hasattr(optimizer, 'create_model')
        assert hasattr(optimizer, 'get_model_name')


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils()
