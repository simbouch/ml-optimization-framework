3#!/usr/bin/env python3
"""
Tests for configuration management
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.config import OptimizationConfig, ModelConfig


class TestOptimizationConfig:
    """Test OptimizationConfig class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = OptimizationConfig()
        
        assert config.study_name == "ml_optimization_study"
        assert config.direction == "maximize"
        assert config.n_trials == 100
        assert config.sampler_name == "TPE"
        assert config.pruner_name == "Median"
        assert config.random_seed == 42
        assert config.cv_folds == 5
        assert config.test_size == 0.2
    
    def test_custom_config(self):
        """Test custom configuration creation."""
        config = OptimizationConfig(
            study_name="test_study",
            direction="minimize",
            n_trials=50,
            sampler_name="Random",
            pruner_name="SuccessiveHalving"
        )
        
        assert config.study_name == "test_study"
        assert config.direction == "minimize"
        assert config.n_trials == 50
        assert config.sampler_name == "Random"
        assert config.pruner_name == "SuccessiveHalving"
    
    def test_multi_objective_config(self):
        """Test multi-objective configuration."""
        config = OptimizationConfig(
            direction=["maximize", "minimize"]
        )
        
        assert config.direction == ["maximize", "minimize"]
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid direction
        with pytest.raises(ValueError, match="Direction must be one of"):
            OptimizationConfig(direction="invalid")
        
        # Test invalid sampler
        with pytest.raises(ValueError, match="Sampler must be one of"):
            OptimizationConfig(sampler_name="invalid")
        
        # Test invalid pruner
        with pytest.raises(ValueError, match="Pruner must be one of"):
            OptimizationConfig(pruner_name="invalid")
        
        # Test invalid n_trials
        with pytest.raises(ValueError, match="Number of trials must be positive"):
            OptimizationConfig(n_trials=0)
        
        # Test invalid cv_folds
        with pytest.raises(ValueError, match="CV folds must be greater than 1"):
            OptimizationConfig(cv_folds=1)
        
        # Test invalid test_size
        with pytest.raises(ValueError, match="Test size must be between 0 and 1"):
            OptimizationConfig(test_size=1.5)
    
    def test_directory_creation(self):
        """Test that directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            config = OptimizationConfig(
                data_dir=temp_path / "data",
                results_dir=temp_path / "results",
                studies_dir=temp_path / "studies",
                logs_dir=temp_path / "logs"
            )
            
            assert config.data_dir.exists()
            assert config.results_dir.exists()
            assert config.studies_dir.exists()
            assert config.logs_dir.exists()
    
    def test_storage_url_generation(self):
        """Test storage URL generation."""
        config = OptimizationConfig(study_name="test_study")
        expected_url = f"sqlite:///{config.studies_dir}/test_study.db"
        assert config.storage_url == expected_url
    
    def test_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "study_name": "dict_study",
            "direction": "minimize",
            "n_trials": 25,
            "sampler_name": "Random"
        }
        
        config = OptimizationConfig.from_dict(config_dict)
        
        assert config.study_name == "dict_study"
        assert config.direction == "minimize"
        assert config.n_trials == 25
        assert config.sampler_name == "Random"
    
    def test_to_dict(self):
        """Test configuration conversion to dictionary."""
        config = OptimizationConfig(study_name="test_study")
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["study_name"] == "test_study"
        assert "direction" in config_dict
        assert "n_trials" in config_dict
    
    def test_from_env(self):
        """Test configuration creation from environment variables."""
        # Set environment variables
        os.environ["STUDY_NAME"] = "env_study"
        os.environ["N_TRIALS"] = "75"
        os.environ["SAMPLER_NAME"] = "CmaEs"
        
        try:
            config = OptimizationConfig.from_env()
            
            assert config.study_name == "env_study"
            assert config.n_trials == 75
            assert config.sampler_name == "CmaEs"
        finally:
            # Clean up environment variables
            for var in ["STUDY_NAME", "N_TRIALS", "SAMPLER_NAME"]:
                if var in os.environ:
                    del os.environ[var]


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_default_model_config(self):
        """Test default model configuration."""
        config = ModelConfig()
        
        assert config.model_type == "RandomForest"
        assert isinstance(config.model_params, dict)
        assert isinstance(config.param_space, dict)
    
    def test_custom_model_config(self):
        """Test custom model configuration."""
        config = ModelConfig(
            model_type="XGBoost",
            model_params={"n_jobs": -1}
        )
        
        assert config.model_type == "XGBoost"
        assert config.model_params == {"n_jobs": -1}
    
    def test_parameter_space_generation(self):
        """Test parameter space generation for different models."""
        # Test RandomForest
        rf_config = ModelConfig(model_type="RandomForest")
        assert "n_estimators" in rf_config.param_space
        assert "max_depth" in rf_config.param_space
        
        # Test XGBoost
        xgb_config = ModelConfig(model_type="XGBoost")
        assert "learning_rate" in xgb_config.param_space
        assert "subsample" in xgb_config.param_space
        
        # Test SVM
        svm_config = ModelConfig(model_type="SVM")
        assert "C" in svm_config.param_space
        assert "kernel" in svm_config.param_space
        
        # Test unknown model type
        unknown_config = ModelConfig(model_type="Unknown")
        assert unknown_config.param_space == {}


if __name__ == "__main__":
    pytest.main([__file__])
