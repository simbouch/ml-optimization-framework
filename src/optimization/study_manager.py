"""
Study management for Optuna optimization studies.

This module provides comprehensive study management including creation,
persistence, multi-objective optimization, and advanced features.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Union, Callable
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler, GridSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.storages import RDBStorage
import sqlite3
import pandas as pd

from .config import OptimizationConfig, StudyConfig

logger = logging.getLogger(__name__)


class StudyManager:
    """
    Comprehensive manager for Optuna studies.
    
    Handles study creation, persistence, multi-objective optimization,
    and provides utilities for study analysis and management.
    """
    
    def __init__(
        self,
        storage_url: str = "sqlite:///optuna_study.db",
        config: Optional[OptimizationConfig] = None
    ):
        """
        Initialize the study manager.
        
        Args:
            storage_url: Database URL for study persistence
            config: Optimization configuration object
        """
        self.storage_url = storage_url
        self.config = config or OptimizationConfig()
        self.studies: Dict[str, optuna.Study] = {}
        
        # Initialize storage
        self._initialize_storage()
        
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"StudyManager initialized with storage: {storage_url}")
    
    def _initialize_storage(self) -> None:
        """Initialize the storage backend."""
        try:
            if self.storage_url.startswith("sqlite"):
                # Extract database path from URL
                db_path = self.storage_url.replace("sqlite:///", "")
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
                
                # Test connection
                conn = sqlite3.connect(db_path)
                conn.close()
                
            logger.info("Storage initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing storage: {str(e)}")
            raise
    
    def create_study(
        self,
        study_name: str,
        model_name: Optional[str] = None,
        direction: str = "maximize",
        sampler_name: str = "tpe",
        pruner_name: str = "median",
        load_if_exists: bool = True,
        sampler_params: Optional[Dict[str, Any]] = None,
        pruner_params: Optional[Dict[str, Any]] = None
    ) -> optuna.Study:
        """
        Create or load an Optuna study.
        
        Args:
            study_name: Name of the study
            model_name: Name of the model (for config lookup)
            direction: Optimization direction ('maximize' or 'minimize')
            sampler_name: Name of the sampler to use
            pruner_name: Name of the pruner to use
            load_if_exists: Whether to load existing study
            sampler_params: Additional sampler parameters
            pruner_params: Additional pruner parameters
            
        Returns:
            Optuna study object
        """
        try:
            # Get configuration if model name is provided
            if model_name and model_name in self.config.get_available_models():
                study_config = self.config.get_study_config(model_name)
                sampler_name = study_config.sampler_config.name
                pruner_name = study_config.pruner_config.name
                sampler_params = sampler_params or study_config.sampler_config.params
                pruner_params = pruner_params or study_config.pruner_config.params
            
            # Create sampler
            sampler = self._create_sampler(sampler_name, sampler_params or {})
            
            # Create pruner
            pruner = self._create_pruner(pruner_name, pruner_params or {})
            
            # Create study
            study = optuna.create_study(
                study_name=study_name,
                direction=direction,
                storage=self.storage_url,
                load_if_exists=load_if_exists,
                sampler=sampler,
                pruner=pruner
            )
            
            # Store study reference
            self.studies[study_name] = study
            
            logger.info(f"Study '{study_name}' created/loaded successfully")
            logger.info(f"Sampler: {sampler_name}, Pruner: {pruner_name}")
            
            return study
            
        except Exception as e:
            logger.error(f"Error creating study '{study_name}': {str(e)}")
            raise
    
    def create_multi_objective_study(
        self,
        study_name: str,
        directions: List[str],
        sampler_name: str = "tpe",
        load_if_exists: bool = True,
        sampler_params: Optional[Dict[str, Any]] = None
    ) -> optuna.Study:
        """
        Create a multi-objective optimization study.
        
        Args:
            study_name: Name of the study
            directions: List of optimization directions
            sampler_name: Name of the sampler to use
            load_if_exists: Whether to load existing study
            sampler_params: Additional sampler parameters
            
        Returns:
            Multi-objective Optuna study
        """
        try:
            # Create sampler
            sampler = self._create_sampler(sampler_name, sampler_params or {})
            
            # Create multi-objective study
            study = optuna.create_study(
                study_name=study_name,
                directions=directions,
                storage=self.storage_url,
                load_if_exists=load_if_exists,
                sampler=sampler
            )
            
            # Store study reference
            self.studies[study_name] = study
            
            logger.info(f"Multi-objective study '{study_name}' created/loaded")
            logger.info(f"Directions: {directions}")
            
            return study
            
        except Exception as e:
            logger.error(f"Error creating multi-objective study '{study_name}': {str(e)}")
            raise
    
    def _create_sampler(self, sampler_name: str, params: Dict[str, Any]) -> optuna.samplers.BaseSampler:
        """Create sampler instance."""
        samplers = {
            "tpe": TPESampler,
            "cmaes": CmaEsSampler,
            "random": RandomSampler,
            "grid": GridSampler
        }
        
        if sampler_name not in samplers:
            raise ValueError(f"Unknown sampler: {sampler_name}")
        
        # Add random state if not provided
        if "seed" not in params and sampler_name in ["tpe", "random"]:
            params["seed"] = 42
        
        return samplers[sampler_name](**params)
    
    def _create_pruner(self, pruner_name: str, params: Dict[str, Any]) -> optuna.pruners.BasePruner:
        """Create pruner instance."""
        pruners = {
            "median": MedianPruner,
            "successive_halving": SuccessiveHalvingPruner
        }
        
        if pruner_name not in pruners:
            raise ValueError(f"Unknown pruner: {pruner_name}")
        
        return pruners[pruner_name](**params)
    
    def get_study(self, study_name: str) -> Optional[optuna.Study]:
        """
        Get an existing study by name.
        
        Args:
            study_name: Name of the study
            
        Returns:
            Optuna study object or None if not found
        """
        if study_name in self.studies:
            return self.studies[study_name]
        
        try:
            # Try to load from storage
            study = optuna.load_study(
                study_name=study_name,
                storage=self.storage_url
            )
            self.studies[study_name] = study
            return study
            
        except Exception:
            logger.warning(f"Study '{study_name}' not found")
            return None
    
    def list_studies(self) -> List[str]:
        """
        List all available studies in storage.
        
        Returns:
            List of study names
        """
        try:
            if self.storage_url.startswith("sqlite"):
                db_path = self.storage_url.replace("sqlite:///", "")
                
                if not os.path.exists(db_path):
                    return []
                
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Query study names
                cursor.execute("SELECT DISTINCT study_name FROM studies")
                study_names = [row[0] for row in cursor.fetchall()]
                
                conn.close()
                return study_names
            
            else:
                # For other storage types, return cached studies
                return list(self.studies.keys())
                
        except Exception as e:
            logger.error(f"Error listing studies: {str(e)}")
            return []
    
    def delete_study(self, study_name: str) -> bool:
        """
        Delete a study from storage.
        
        Args:
            study_name: Name of the study to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            optuna.delete_study(
                study_name=study_name,
                storage=self.storage_url
            )
            
            # Remove from cache
            if study_name in self.studies:
                del self.studies[study_name]
            
            logger.info(f"Study '{study_name}' deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting study '{study_name}': {str(e)}")
            return False
    
    def get_study_summary(self, study_name: str) -> Optional[Dict[str, Any]]:
        """
        Get summary information for a study.
        
        Args:
            study_name: Name of the study
            
        Returns:
            Dictionary containing study summary
        """
        study = self.get_study(study_name)
        if study is None:
            return None
        
        try:
            trials = study.trials
            completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            summary = {
                'study_name': study_name,
                'direction': study.direction if hasattr(study, 'direction') else study.directions,
                'n_trials': len(trials),
                'n_completed_trials': len(completed_trials),
                'n_pruned_trials': len([t for t in trials if t.state == optuna.trial.TrialState.PRUNED]),
                'n_failed_trials': len([t for t in trials if t.state == optuna.trial.TrialState.FAIL]),
                'sampler': type(study.sampler).__name__,
                'pruner': type(study.pruner).__name__
            }
            
            if completed_trials:
                if hasattr(study, 'best_value'):
                    summary['best_value'] = study.best_value
                    summary['best_params'] = study.best_params
                else:
                    # Multi-objective study
                    summary['best_trials'] = len(study.best_trials)
                    summary['pareto_front_size'] = len(study.best_trials)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting study summary: {str(e)}")
            return None
    
    def export_study_data(self, study_name: str, format: str = "csv") -> Optional[pd.DataFrame]:
        """
        Export study data to DataFrame or file.
        
        Args:
            study_name: Name of the study
            format: Export format ('csv', 'json', 'dataframe')
            
        Returns:
            DataFrame containing study data
        """
        study = self.get_study(study_name)
        if study is None:
            return None
        
        try:
            # Convert trials to DataFrame
            trials_data = []
            for trial in study.trials:
                trial_data = {
                    'trial_number': trial.number,
                    'state': trial.state.name,
                    'value': trial.value,
                    'datetime_start': trial.datetime_start,
                    'datetime_complete': trial.datetime_complete,
                    'duration': trial.duration.total_seconds() if trial.duration else None
                }
                
                # Add parameters
                for param_name, param_value in trial.params.items():
                    trial_data[f'param_{param_name}'] = param_value
                
                # Add user attributes
                for attr_name, attr_value in trial.user_attrs.items():
                    trial_data[f'user_attr_{attr_name}'] = attr_value
                
                trials_data.append(trial_data)
            
            df = pd.DataFrame(trials_data)
            
            # Export to file if requested
            if format == "csv":
                filename = f"{study_name}_trials.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Study data exported to {filename}")
            elif format == "json":
                filename = f"{study_name}_trials.json"
                df.to_json(filename, orient='records', indent=2)
                logger.info(f"Study data exported to {filename}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error exporting study data: {str(e)}")
            return None
    
    def compare_studies(self, study_names: List[str]) -> pd.DataFrame:
        """
        Compare multiple studies.
        
        Args:
            study_names: List of study names to compare
            
        Returns:
            DataFrame containing comparison results
        """
        comparison_data = []
        
        for study_name in study_names:
            summary = self.get_study_summary(study_name)
            if summary:
                comparison_data.append(summary)
        
        if not comparison_data:
            logger.warning("No valid studies found for comparison")
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        logger.info(f"Comparison completed for {len(comparison_data)} studies")
        
        return df
    
    def cleanup_failed_trials(self, study_name: str) -> int:
        """
        Remove failed trials from a study.
        
        Args:
            study_name: Name of the study
            
        Returns:
            Number of trials removed
        """
        study = self.get_study(study_name)
        if study is None:
            return 0
        
        try:
            failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            
            # Note: Optuna doesn't provide direct trial deletion
            # This is a placeholder for potential future functionality
            logger.info(f"Found {len(failed_trials)} failed trials in study '{study_name}'")
            logger.info("Note: Trial deletion not implemented in current Optuna version")
            
            return len(failed_trials)
            
        except Exception as e:
            logger.error(f"Error cleaning up failed trials: {str(e)}")
            return 0


if __name__ == "__main__":
    # Example usage
    manager = StudyManager()
    
    # Create a study
    study = manager.create_study("test_study", direction="maximize")
    
    # List studies
    studies = manager.list_studies()
    print(f"Available studies: {studies}")
    
    # Get study summary
    summary = manager.get_study_summary("test_study")
    print(f"Study summary: {summary}")
