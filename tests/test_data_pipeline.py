"""
Comprehensive tests for the data pipeline module.

This module tests all aspects of data loading, preprocessing,
and pipeline functionality.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.data_pipeline import DataPipeline
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TestDataPipelineCore:
    """Test core data pipeline functionality."""
    
    @pytest.mark.unit
    def test_initialization(self):
        """Test data pipeline initialization."""
        pipeline = DataPipeline(random_state=42, test_size=0.2, val_size=0.2)
        
        assert pipeline.random_state == 42
        assert pipeline.test_size == 0.2
        assert pipeline.val_size == 0.2
        assert not pipeline.is_prepared
    
    @pytest.mark.unit
    def test_initialization_with_defaults(self):
        """Test data pipeline initialization with default values."""
        pipeline = DataPipeline()
        
        assert pipeline.random_state == 42
        assert pipeline.test_size == 0.2
        assert pipeline.val_size == 0.2
    
    @pytest.mark.unit
    def test_invalid_initialization(self):
        """Test data pipeline initialization with invalid parameters."""
        with pytest.raises(ValueError):
            DataPipeline(test_size=1.5)  # Invalid test size
        
        with pytest.raises(ValueError):
            DataPipeline(val_size=-0.1)  # Invalid validation size
        
        with pytest.raises(ValueError):
            DataPipeline(test_size=0.5, val_size=0.6)  # Sum > 1


class TestDataLoading:
    """Test data loading functionality."""
    
    @pytest.mark.network
    def test_data_loading_success(self):
        """Test successful data loading from OpenML."""
        pipeline = DataPipeline(random_state=42)
        X, y = pipeline.load_data()
        
        # Check data properties
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape[0] > 1000  # Should have substantial data
        assert X.shape[1] > 5     # Should have multiple features
        assert len(y) == X.shape[0]  # Target should match features
        
        # Check target values
        unique_targets = y.unique()
        assert len(unique_targets) == 2  # Binary classification
        assert set(unique_targets) == {0, 1}  # Should be 0 and 1
    
    @pytest.mark.unit
    def test_data_loading_failure_handling(self):
        """Test data loading failure handling."""
        pipeline = DataPipeline(random_state=42)
        
        # Mock openml.datasets.get_dataset to raise an exception
        with patch('openml.datasets.get_dataset') as mock_get_dataset:
            mock_get_dataset.side_effect = Exception("Network error")
            
            with pytest.raises(Exception):
                pipeline.load_data()
    
    @pytest.mark.unit
    def test_data_loading_caching(self):
        """Test that data loading is cached."""
        pipeline = DataPipeline(random_state=42)
        
        # Mock the actual loading
        with patch.object(pipeline, '_load_adult_income_data') as mock_load:
            mock_load.return_value = (
                pd.DataFrame({'feature': [1, 2, 3]}),
                pd.Series([0, 1, 0])
            )
            
            # First call should load data
            X1, y1 = pipeline.load_data()
            assert mock_load.call_count == 1
            
            # Second call should use cached data
            X2, y2 = pipeline.load_data()
            assert mock_load.call_count == 1  # Should not be called again
            
            # Data should be identical
            pd.testing.assert_frame_equal(X1, X2)
            pd.testing.assert_series_equal(y1, y2)


class TestDataAnalysis:
    """Test data analysis functionality."""
    
    @pytest.mark.unit
    def test_data_analysis_with_loaded_data(self, data_pipeline):
        """Test data analysis with loaded data."""
        # Load data first
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
        
        # Check data types
        assert isinstance(analysis['shape'], tuple)
        assert isinstance(analysis['missing_values'], dict)
        assert isinstance(analysis['categorical_features'], list)
        assert isinstance(analysis['numerical_features'], list)
        assert isinstance(analysis['target_distribution'], dict)
        assert isinstance(analysis['duplicate_rows'], int)
        
        # Check feature categorization
        assert len(analysis['categorical_features']) > 0
        assert len(analysis['numerical_features']) > 0
        
        # Check target distribution
        assert len(analysis['target_distribution']) == 2
        assert 0 in analysis['target_distribution']
        assert 1 in analysis['target_distribution']
    
    @pytest.mark.unit
    def test_data_analysis_without_loaded_data(self):
        """Test data analysis without loaded data."""
        pipeline = DataPipeline(random_state=42)
        
        with pytest.raises(ValueError, match="Data not loaded"):
            pipeline.analyze_data()


class TestDataPreprocessing:
    """Test data preprocessing functionality."""
    
    @pytest.mark.unit
    def test_data_preparation_complete(self, data_pipeline):
        """Test complete data preparation."""
        summary = data_pipeline.prepare_data()
        
        # Check summary structure
        required_keys = [
            'total_samples', 'total_features', 'train_samples',
            'val_samples', 'test_samples', 'preprocessing_complete'
        ]
        
        for key in required_keys:
            assert key in summary
        
        # Check that pipeline is marked as prepared
        assert data_pipeline.is_prepared
        assert summary['preprocessing_complete']
        
        # Check sample counts
        total = summary['total_samples']
        train = summary['train_samples']
        val = summary['val_samples']
        test = summary['test_samples']
        
        assert train + val + test == total
        assert train > 0
        assert val > 0
        assert test > 0
    
    @pytest.mark.unit
    def test_data_splits_consistency(self, data_pipeline):
        """Test that data splits are consistent."""
        data_pipeline.prepare_data()
        
        # Get data splits
        X_train, X_val, y_train, y_val = data_pipeline.get_train_val_data()
        X_test, y_test = data_pipeline.get_test_data()
        
        # Check shapes
        assert X_train.shape[0] == len(y_train)
        assert X_val.shape[0] == len(y_val)
        assert X_test.shape[0] == len(y_test)
        
        # Check feature consistency
        assert X_train.shape[1] == X_val.shape[1]
        assert X_train.shape[1] == X_test.shape[1]
        
        # Check no data leakage (no overlapping indices)
        train_indices = set(range(len(y_train)))
        val_indices = set(range(len(y_train), len(y_train) + len(y_val)))
        test_indices = set(range(len(y_train) + len(y_val), len(y_train) + len(y_val) + len(y_test)))
        
        assert len(train_indices & val_indices) == 0
        assert len(train_indices & test_indices) == 0
        assert len(val_indices & test_indices) == 0
    
    @pytest.mark.unit
    def test_get_data_before_preparation(self):
        """Test getting data before preparation raises error."""
        pipeline = DataPipeline(random_state=42)
        
        with pytest.raises(ValueError, match="Data not prepared"):
            pipeline.get_train_val_data()
        
        with pytest.raises(ValueError, match="Data not prepared"):
            pipeline.get_test_data()


class TestDataPipelineReproducibility:
    """Test reproducibility of data pipeline."""
    
    @pytest.mark.integration
    def test_reproducible_data_preparation(self):
        """Test that data preparation is reproducible."""
        # Create two pipelines with same random state
        pipeline1 = DataPipeline(random_state=42)
        pipeline2 = DataPipeline(random_state=42)
        
        # Prepare data
        summary1 = pipeline1.prepare_data()
        summary2 = pipeline2.prepare_data()
        
        # Summaries should be identical
        assert summary1 == summary2
        
        # Data splits should be identical
        X_train1, X_val1, y_train1, y_val1 = pipeline1.get_train_val_data()
        X_train2, X_val2, y_train2, y_val2 = pipeline2.get_train_val_data()
        
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_val1, X_val2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_val1, y_val2)
        
        X_test1, y_test1 = pipeline1.get_test_data()
        X_test2, y_test2 = pipeline2.get_test_data()
        
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_test1, y_test2)
    
    @pytest.mark.integration
    def test_different_random_states_produce_different_splits(self):
        """Test that different random states produce different splits."""
        # Create two pipelines with different random states
        pipeline1 = DataPipeline(random_state=42)
        pipeline2 = DataPipeline(random_state=123)
        
        # Prepare data
        pipeline1.prepare_data()
        pipeline2.prepare_data()
        
        # Get training data
        X_train1, _, y_train1, _ = pipeline1.get_train_val_data()
        X_train2, _, y_train2, _ = pipeline2.get_train_val_data()
        
        # Training sets should be different (with high probability)
        # We check that at least some samples are different
        different_samples = np.any(X_train1 != X_train2, axis=1)
        assert np.sum(different_samples) > 0


class TestDataPipelineEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.unit
    def test_very_small_dataset_splits(self):
        """Test behavior with very small dataset."""
        # Create a very small mock dataset
        small_X = pd.DataFrame({'feature': range(10)})
        small_y = pd.Series([0, 1] * 5)
        
        pipeline = DataPipeline(random_state=42, test_size=0.2, val_size=0.2)
        
        # Mock the data loading to return small dataset
        with patch.object(pipeline, '_load_adult_income_data') as mock_load:
            mock_load.return_value = (small_X, small_y)
            
            # Should still work but with very small splits
            summary = pipeline.prepare_data()
            
            assert summary['total_samples'] == 10
            assert summary['train_samples'] >= 1
            assert summary['val_samples'] >= 1
            assert summary['test_samples'] >= 1
    
    @pytest.mark.unit
    def test_extreme_split_ratios(self):
        """Test extreme but valid split ratios."""
        # Test with very small test set
        pipeline = DataPipeline(random_state=42, test_size=0.01, val_size=0.01)
        
        # Mock data loading
        mock_X = pd.DataFrame({'feature': range(1000)})
        mock_y = pd.Series([0, 1] * 500)
        
        with patch.object(pipeline, '_load_adult_income_data') as mock_load:
            mock_load.return_value = (mock_X, mock_y)
            
            summary = pipeline.prepare_data()
            
            # Should have very small test and validation sets
            assert summary['test_samples'] < 50
            assert summary['val_samples'] < 50
            assert summary['train_samples'] > 900
