#!/usr/bin/env python3
"""
Tests for study manager
"""

import pytest
import tempfile
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification

from src.config import OptimizationConfig
from src.study_manager import StudyManager


class TestStudyManager:
    """Test StudyManager functionality."""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = OptimizationConfig(
                study_name="test_study",
                n_trials=10,
                data_dir=temp_path / "data",
                results_dir=temp_path / "results",
                studies_dir=temp_path / "studies",
                logs_dir=temp_path / "logs"
            )
            yield config
    
    @pytest.fixture
    def study_manager(self, temp_config):
        """Create StudyManager instance."""
        return StudyManager(temp_config)
    
    @pytest.fixture
    def sample_study(self, study_manager):
        """Create a sample study with trials."""
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            y = trial.suggest_float('y', -10, 10)
            return x**2 + y**2
        
        study = study_manager.create_study(
            study_name="sample_study",
            direction="minimize",
            sampler_name="Random",
            pruner_name="None"
        )
        
        study.optimize(objective, n_trials=5)
        return study
    
    def test_create_study(self, study_manager):
        """Test study creation."""
        study = study_manager.create_study(
            study_name="test_create_study",
            direction="maximize",
            sampler_name="TPE",
            pruner_name="Median"
        )
        
        assert study is not None
        assert study.study_name == "test_create_study"
        assert "test_create_study" in study_manager.studies
        
        # Check sampler and pruner types
        assert "TPE" in str(type(study.sampler))
        assert study.pruner is not None
    
    def test_create_multi_objective_study(self, study_manager):
        """Test multi-objective study creation."""
        study = study_manager.create_study(
            study_name="multi_obj_study",
            direction=["maximize", "minimize"],
            sampler_name="Random",
            pruner_name="None"
        )
        
        assert study is not None
        # Check that directions are set correctly (Optuna returns enum objects)
        assert len(study.directions) == 2
        # Check the enum values directly
        assert study.directions[0].name == "MAXIMIZE"
        assert study.directions[1].name == "MINIMIZE"
    
    def test_load_study(self, study_manager, sample_study):
        """Test study loading."""
        study_name = sample_study.study_name
        
        # Load the study
        loaded_study = study_manager.load_study(study_name)
        
        assert loaded_study is not None
        assert loaded_study.study_name == study_name
        assert len(loaded_study.trials) == len(sample_study.trials)
    
    def test_load_nonexistent_study(self, study_manager):
        """Test loading non-existent study."""
        with pytest.raises(Exception):
            study_manager.load_study("nonexistent_study")
    
    def test_get_study_summary(self, study_manager, sample_study):
        """Test study summary generation."""
        study_name = sample_study.study_name
        summary = study_manager.get_study_summary(study_name)
        
        assert isinstance(summary, dict)
        assert summary["study_name"] == study_name
        assert summary["n_trials"] == len(sample_study.trials)
        assert "direction" in summary
        assert "best_value" in summary
        assert "best_params" in summary
        assert "sampler" in summary
        assert "pruner" in summary
        assert "state_counts" in summary
        assert "created_at" in summary
        assert "last_updated" in summary
    
    def test_get_all_studies_summary(self, study_manager, sample_study):
        """Test getting all studies summary."""
        # Create another study
        study_manager.create_study("another_study", direction="maximize")
        
        summaries = study_manager.get_all_studies_summary()
        
        assert isinstance(summaries, list)
        assert len(summaries) >= 1  # At least the sample study
        
        # Check that summaries contain expected fields
        for summary in summaries:
            assert "study_name" in summary
            assert "n_trials" in summary
    
    def test_compare_studies(self, study_manager, sample_study):
        """Test study comparison."""
        # Create another study for comparison
        def objective2(trial):
            x = trial.suggest_float('x', 0, 5)
            return x**3
        
        study2 = study_manager.create_study("comparison_study", direction="minimize")
        study2.optimize(objective2, n_trials=3)
        
        comparison = study_manager.compare_studies([
            sample_study.study_name,
            "comparison_study"
        ])
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert "study_name" in comparison.columns
        assert "n_trials" in comparison.columns
    
    def test_export_study_results_csv(self, study_manager, sample_study, temp_config):
        """Test exporting study results to CSV."""
        study_name = sample_study.study_name
        
        file_path = study_manager.export_study_results(study_name, format="csv")
        
        assert file_path.exists()
        assert file_path.suffix == ".csv"
        assert file_path.parent == temp_config.results_dir
        
        # Check file content
        df = pd.read_csv(file_path)
        assert len(df) == len(sample_study.trials)
        assert "trial_number" in df.columns
        assert "value" in df.columns
        assert "x" in df.columns  # Parameter from objective function
        assert "y" in df.columns  # Parameter from objective function
    
    def test_export_study_results_json(self, study_manager, sample_study, temp_config):
        """Test exporting study results to JSON."""
        study_name = sample_study.study_name
        
        file_path = study_manager.export_study_results(study_name, format="json")
        
        assert file_path.exists()
        assert file_path.suffix == ".json"
        
        # Check file content
        df = pd.read_json(file_path)
        assert len(df) == len(sample_study.trials)
    
    def test_export_study_results_excel(self, study_manager, sample_study, temp_config):
        """Test exporting study results to Excel."""
        study_name = sample_study.study_name
        
        try:
            file_path = study_manager.export_study_results(study_name, format="excel")
            
            assert file_path.exists()
            assert file_path.suffix == ".xlsx"
            
            # Check file content
            df = pd.read_excel(file_path)
            assert len(df) == len(sample_study.trials)
        except ImportError:
            # openpyxl might not be installed
            pytest.skip("openpyxl not available for Excel export")
    
    def test_export_unsupported_format(self, study_manager, sample_study):
        """Test exporting with unsupported format."""
        study_name = sample_study.study_name
        
        with pytest.raises(ValueError, match="Unsupported format"):
            study_manager.export_study_results(study_name, format="unsupported")
    
    def test_sampler_creation(self, study_manager):
        """Test different sampler creation."""
        samplers = ["TPE", "Random", "CmaEs", "QMC"]
        
        for sampler_name in samplers:
            try:
                sampler = study_manager._create_sampler(sampler_name)
                assert sampler is not None
            except Exception as e:
                # Some samplers might have additional requirements
                assert "sampler" in str(e).lower() or "import" in str(e).lower()
    
    def test_pruner_creation(self, study_manager):
        """Test different pruner creation."""
        pruners = ["Median", "SuccessiveHalving", "Hyperband", "None"]
        
        for pruner_name in pruners:
            pruner = study_manager._create_pruner(pruner_name)
            
            if pruner_name == "None":
                assert pruner is None
            else:
                assert pruner is not None
    
    def test_trial_state_counts(self, study_manager, sample_study):
        """Test trial state counting."""
        state_counts = study_manager._get_trial_state_counts(sample_study)
        
        assert isinstance(state_counts, dict)
        assert "COMPLETE" in state_counts
        assert state_counts["COMPLETE"] == len(sample_study.trials)
    
    def test_study_timing_methods(self, study_manager, sample_study):
        """Test study timing methods."""
        creation_time = study_manager._get_study_creation_time(sample_study)
        last_update_time = study_manager._get_study_last_update_time(sample_study)
        
        if sample_study.trials:
            assert creation_time is not None
            assert last_update_time is not None
            assert isinstance(creation_time, str)
            assert isinstance(last_update_time, str)
        else:
            assert creation_time is None
            assert last_update_time is None


class TestStudyManagerIntegration:
    """Test StudyManager integration scenarios."""

    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = OptimizationConfig(
                study_name="integration_test",
                n_trials=5,
                data_dir=temp_path / "data",
                results_dir=temp_path / "results",
                studies_dir=temp_path / "studies",
                logs_dir=temp_path / "logs"
            )
            yield config

    def test_real_ml_optimization_workflow(self, temp_config):
        """Test complete ML optimization workflow."""
        from src.optimizers import RandomForestOptimizer
        
        # Create data
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        # Create optimizer and run optimization
        optimizer = RandomForestOptimizer(temp_config, task_type="classification")
        study = optimizer.optimize(X, y)
        
        # Use StudyManager to analyze results
        study_manager = StudyManager(temp_config)
        summary = study_manager.get_study_summary(temp_config.study_name)
        
        assert summary["n_trials"] == temp_config.n_trials
        assert summary["best_value"] is not None
        assert summary["best_params"] is not None
        
        # Export results
        export_path = study_manager.export_study_results(
            temp_config.study_name, 
            format="csv"
        )
        assert export_path.exists()


if __name__ == "__main__":
    pytest.main([__file__])
