#!/usr/bin/env python3
"""
Study management for ML Optimization Framework
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour,
    plot_pareto_front
)
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
from loguru import logger
import json
from datetime import datetime

from .config import OptimizationConfig


class StudyManager:
    """
    Comprehensive study management for Optuna optimization studies.
    
    This class provides functionality to create, manage, analyze, and visualize
    optimization studies with support for multiple objectives and advanced features.
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize the study manager.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.studies: Dict[str, optuna.Study] = {}
        
        # Set up logging
        logger.add(
            self.config.logs_dir / "study_manager.log",
            format=self.config.log_format,
            level=self.config.log_level,
            rotation="10 MB"
        )
    
    def create_study(
        self,
        study_name: str,
        direction: Union[str, List[str]] = "maximize",
        sampler_name: str = "TPE",
        pruner_name: str = "Median",
        storage_url: Optional[str] = None
    ) -> optuna.Study:
        """
        Create a new optimization study.
        
        Args:
            study_name: Name of the study
            direction: Optimization direction(s)
            sampler_name: Sampler to use
            pruner_name: Pruner to use
            storage_url: Storage URL (optional)
            
        Returns:
            Created Optuna study
        """
        if storage_url is None:
            storage_url = f"sqlite:///{self.config.studies_dir}/{study_name}.db"
        
        # Create sampler
        sampler = self._create_sampler(sampler_name)
        
        # Create pruner
        pruner = self._create_pruner(pruner_name)
        
        # Handle multi-objective optimization
        if isinstance(direction, list):
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                directions=direction,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True
            )
        else:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction=direction,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True
            )
        
        self.studies[study_name] = study
        
        logger.info(f"Created study: {study_name}")
        logger.info(f"Direction: {direction}")
        logger.info(f"Sampler: {sampler_name}")
        logger.info(f"Pruner: {pruner_name}")
        
        return study
    
    def load_study(self, study_name: str, storage_url: Optional[str] = None) -> optuna.Study:
        """
        Load an existing study.
        
        Args:
            study_name: Name of the study to load
            storage_url: Storage URL (optional)
            
        Returns:
            Loaded Optuna study
        """
        if storage_url is None:
            storage_url = f"sqlite:///{self.config.studies_dir}/{study_name}.db"
        
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage_url
            )
            self.studies[study_name] = study
            logger.info(f"Loaded study: {study_name} with {len(study.trials)} trials")
            return study
        except Exception as e:
            logger.error(f"Failed to load study {study_name}: {e}")
            raise
    
    def get_study_summary(self, study_name: str) -> Dict[str, Any]:
        """
        Get comprehensive summary of a study.
        
        Args:
            study_name: Name of the study
            
        Returns:
            Dictionary containing study summary
        """
        if study_name not in self.studies:
            self.load_study(study_name)
        
        study = self.studies[study_name]
        
        summary = {
            "study_name": study_name,
            "n_trials": len(study.trials),
            "direction": study.direction if hasattr(study, 'direction') else study.directions,
            "best_value": study.best_value if hasattr(study, 'best_value') else None,
            "best_params": study.best_params if hasattr(study, 'best_params') else None,
            "best_trial": study.best_trial.number if hasattr(study, 'best_trial') else None,
            "sampler": type(study.sampler).__name__,
            "pruner": type(study.pruner).__name__ if study.pruner else "None",
            "state_counts": self._get_trial_state_counts(study),
            "created_at": self._get_study_creation_time(study),
            "last_updated": self._get_study_last_update_time(study),
        }
        
        return summary
    
    def get_all_studies_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all available studies.
        
        Returns:
            List of study summaries
        """
        summaries = []
        
        # Find all database files in studies directory
        db_files = list(self.config.studies_dir.glob("*.db"))
        
        for db_file in db_files:
            study_name = db_file.stem
            try:
                summary = self.get_study_summary(study_name)
                summaries.append(summary)
            except Exception as e:
                logger.warning(f"Could not load study {study_name}: {e}")
        
        return summaries
    
    def compare_studies(self, study_names: List[str]) -> pd.DataFrame:
        """
        Compare multiple studies.
        
        Args:
            study_names: List of study names to compare
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for study_name in study_names:
            summary = self.get_study_summary(study_name)
            comparison_data.append(summary)
        
        return pd.DataFrame(comparison_data)
    
    def export_study_results(self, study_name: str, format: str = "csv") -> Path:
        """
        Export study results to file.
        
        Args:
            study_name: Name of the study
            format: Export format ("csv", "json", "excel")
            
        Returns:
            Path to exported file
        """
        if study_name not in self.studies:
            self.load_study(study_name)
        
        study = self.studies[study_name]
        
        # Convert trials to DataFrame
        trials_data = []
        for trial in study.trials:
            trial_data = {
                "trial_number": trial.number,
                "value": trial.value,
                "state": trial.state.name,
                "datetime_start": trial.datetime_start,
                "datetime_complete": trial.datetime_complete,
                "duration": trial.duration.total_seconds() if trial.duration else None,
            }
            trial_data.update(trial.params)
            trials_data.append(trial_data)
        
        df = pd.DataFrame(trials_data)
        
        # Export based on format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "csv":
            file_path = self.config.results_dir / f"{study_name}_results_{timestamp}.csv"
            df.to_csv(file_path, index=False)
        elif format == "json":
            file_path = self.config.results_dir / f"{study_name}_results_{timestamp}.json"
            df.to_json(file_path, orient="records", date_format="iso")
        elif format == "excel":
            file_path = self.config.results_dir / f"{study_name}_results_{timestamp}.xlsx"
            df.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported study results to: {file_path}")
        return file_path
    
    def _create_sampler(self, sampler_name: str) -> optuna.samplers.BaseSampler:
        """Create sampler based on name."""
        from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler, GridSampler, QMCSampler
        
        samplers = {
            "TPE": TPESampler,
            "Random": RandomSampler,
            "CmaEs": CmaEsSampler,
            "Grid": GridSampler,
            "QMC": QMCSampler,
        }
        
        sampler_class = samplers[sampler_name]
        return sampler_class(seed=self.config.random_seed)
    
    def _create_pruner(self, pruner_name: str) -> Optional[optuna.pruners.BasePruner]:
        """Create pruner based on name."""
        if pruner_name == "None":
            return None
        
        from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
        
        pruners = {
            "Median": MedianPruner,
            "SuccessiveHalving": SuccessiveHalvingPruner,
            "Hyperband": HyperbandPruner,
        }
        
        pruner_class = pruners[pruner_name]
        return pruner_class()
    
    def _get_trial_state_counts(self, study: optuna.Study) -> Dict[str, int]:
        """Get counts of trials by state."""
        from collections import Counter
        states = [trial.state.name for trial in study.trials]
        return dict(Counter(states))
    
    def _get_study_creation_time(self, study: optuna.Study) -> Optional[str]:
        """Get study creation time."""
        if study.trials:
            first_trial = min(study.trials, key=lambda t: t.datetime_start)
            return first_trial.datetime_start.isoformat()
        return None
    
    def _get_study_last_update_time(self, study: optuna.Study) -> Optional[str]:
        """Get study last update time."""
        if study.trials:
            last_trial = max(study.trials, key=lambda t: t.datetime_start)
            return last_trial.datetime_start.isoformat()
        return None
