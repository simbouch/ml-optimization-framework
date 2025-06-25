"""
Custom callbacks for Optuna optimization studies.

This module provides various callback functions for monitoring,
logging, and controlling optimization studies.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Callable
import optuna
from optuna.trial import TrialState
import json
import os

logger = logging.getLogger(__name__)


class OptimizationCallback:
    """
    Base class for optimization callbacks.
    
    Provides common functionality for monitoring and controlling
    optimization studies during execution.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the callback.
        
        Args:
            verbose: Whether to print verbose output
        """
        self.verbose = verbose
        self.start_time = None
        self.trial_times = []
    
    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """
        Callback function called after each trial.
        
        Args:
            study: Optuna study object
            trial: Completed trial object
        """
        if self.start_time is None:
            self.start_time = time.time()
        
        self._log_trial_result(study, trial)
        self._update_statistics(study, trial)
    
    def _log_trial_result(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Log trial results."""
        if not self.verbose:
            return
        
        if trial.state == TrialState.COMPLETE:
            logger.info(f"Trial {trial.number} completed: {trial.value:.4f}")
        elif trial.state == TrialState.PRUNED:
            logger.info(f"Trial {trial.number} pruned")
        elif trial.state == TrialState.FAIL:
            logger.warning(f"Trial {trial.number} failed")
    
    def _update_statistics(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Update internal statistics."""
        if trial.duration:
            self.trial_times.append(trial.duration.total_seconds())


class BestTrialLogger(OptimizationCallback):
    """
    Callback that logs information about the best trial found so far.
    """
    
    def __init__(self, verbose: bool = True, log_frequency: int = 10):
        """
        Initialize the best trial logger.
        
        Args:
            verbose: Whether to print verbose output
            log_frequency: How often to log best trial info (every N trials)
        """
        super().__init__(verbose)
        self.log_frequency = log_frequency
        self.last_best_value = None
    
    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Log best trial information."""
        super().__call__(study, trial)
        
        if trial.number % self.log_frequency == 0 or trial.state == TrialState.COMPLETE:
            if hasattr(study, 'best_value') and study.best_value != self.last_best_value:
                self.last_best_value = study.best_value
                
                if self.verbose:
                    logger.info(f"New best value: {study.best_value:.4f}")
                    logger.info(f"Best parameters: {study.best_params}")


class ProgressLogger(OptimizationCallback):
    """
    Callback that logs optimization progress and statistics.
    """
    
    def __init__(self, verbose: bool = True, log_frequency: int = 25):
        """
        Initialize the progress logger.
        
        Args:
            verbose: Whether to print verbose output
            log_frequency: How often to log progress (every N trials)
        """
        super().__init__(verbose)
        self.log_frequency = log_frequency
    
    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Log optimization progress."""
        super().__call__(study, trial)
        
        if trial.number % self.log_frequency == 0:
            self._log_progress(study, trial)
    
    def _log_progress(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Log detailed progress information."""
        if not self.verbose:
            return
        
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
        failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        avg_trial_time = sum(self.trial_times) / len(self.trial_times) if self.trial_times else 0
        
        logger.info(f"=== Progress Report (Trial {trial.number}) ===")
        logger.info(f"Completed trials: {len(completed_trials)}")
        logger.info(f"Pruned trials: {len(pruned_trials)}")
        logger.info(f"Failed trials: {len(failed_trials)}")
        logger.info(f"Elapsed time: {elapsed_time:.2f}s")
        logger.info(f"Average trial time: {avg_trial_time:.2f}s")
        
        if hasattr(study, 'best_value'):
            logger.info(f"Current best value: {study.best_value:.4f}")


class EarlyStoppingCallback(OptimizationCallback):
    """
    Callback that implements early stopping based on various criteria.
    """
    
    def __init__(
        self,
        patience: int = 50,
        min_improvement: float = 1e-4,
        verbose: bool = True
    ):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of trials without improvement before stopping
            min_improvement: Minimum improvement to reset patience counter
            verbose: Whether to print verbose output
        """
        super().__init__(verbose)
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_value = None
        self.trials_without_improvement = 0
    
    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Check early stopping criteria."""
        super().__call__(study, trial)
        
        if trial.state != TrialState.COMPLETE:
            return
        
        if not hasattr(study, 'best_value'):
            return
        
        current_best = study.best_value
        
        if self.best_value is None:
            self.best_value = current_best
            self.trials_without_improvement = 0
        else:
            improvement = current_best - self.best_value
            
            if improvement > self.min_improvement:
                self.best_value = current_best
                self.trials_without_improvement = 0
                if self.verbose:
                    logger.info(f"Improvement found: {improvement:.6f}")
            else:
                self.trials_without_improvement += 1
        
        if self.trials_without_improvement >= self.patience:
            if self.verbose:
                logger.info(f"Early stopping triggered after {self.patience} trials without improvement")
            study.stop()


class FileLogger(OptimizationCallback):
    """
    Callback that logs trial results to a file.
    """
    
    def __init__(
        self,
        filepath: str,
        format: str = "json",
        verbose: bool = True
    ):
        """
        Initialize file logger.
        
        Args:
            filepath: Path to log file
            format: Log format ('json' or 'csv')
            verbose: Whether to print verbose output
        """
        super().__init__(verbose)
        self.filepath = filepath
        self.format = format
        self.trial_data = []
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    
    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Log trial to file."""
        super().__call__(study, trial)
        
        trial_info = {
            'trial_number': trial.number,
            'state': trial.state.name,
            'value': trial.value,
            'params': trial.params,
            'user_attrs': trial.user_attrs,
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            'duration_seconds': trial.duration.total_seconds() if trial.duration else None
        }
        
        self.trial_data.append(trial_info)
        self._write_to_file()
    
    def _write_to_file(self) -> None:
        """Write trial data to file."""
        try:
            if self.format == "json":
                with open(self.filepath, 'w') as f:
                    json.dump(self.trial_data, f, indent=2)
            elif self.format == "csv":
                import pandas as pd
                df = pd.json_normalize(self.trial_data)
                df.to_csv(self.filepath, index=False)
            
        except Exception as e:
            logger.error(f"Error writing to file {self.filepath}: {str(e)}")


class MetricsTracker(OptimizationCallback):
    """
    Callback that tracks additional metrics during optimization.
    """
    
    def __init__(self, metrics: List[str], verbose: bool = True):
        """
        Initialize metrics tracker.
        
        Args:
            metrics: List of metric names to track
            verbose: Whether to print verbose output
        """
        super().__init__(verbose)
        self.metrics = metrics
        self.metric_history = {metric: [] for metric in metrics}
    
    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Track metrics from trial."""
        super().__call__(study, trial)
        
        for metric in self.metrics:
            if metric in trial.user_attrs:
                self.metric_history[metric].append(trial.user_attrs[metric])
            else:
                self.metric_history[metric].append(None)
    
    def get_metric_history(self, metric: str) -> List[Any]:
        """Get history for a specific metric."""
        return self.metric_history.get(metric, [])
    
    def get_all_metrics(self) -> Dict[str, List[Any]]:
        """Get all tracked metrics."""
        return self.metric_history.copy()


def create_callback_list(
    log_best: bool = True,
    log_progress: bool = True,
    early_stopping: bool = False,
    file_logging: bool = False,
    metrics_tracking: bool = False,
    **kwargs
) -> List[Callable]:
    """
    Create a list of callbacks based on configuration.
    
    Args:
        log_best: Whether to include best trial logging
        log_progress: Whether to include progress logging
        early_stopping: Whether to include early stopping
        file_logging: Whether to include file logging
        metrics_tracking: Whether to include metrics tracking
        **kwargs: Additional parameters for callbacks
        
    Returns:
        List of callback functions
    """
    callbacks = []
    
    if log_best:
        callbacks.append(BestTrialLogger(
            verbose=kwargs.get('verbose', True),
            log_frequency=kwargs.get('best_log_frequency', 10)
        ))
    
    if log_progress:
        callbacks.append(ProgressLogger(
            verbose=kwargs.get('verbose', True),
            log_frequency=kwargs.get('progress_log_frequency', 25)
        ))
    
    if early_stopping:
        callbacks.append(EarlyStoppingCallback(
            patience=kwargs.get('patience', 50),
            min_improvement=kwargs.get('min_improvement', 1e-4),
            verbose=kwargs.get('verbose', True)
        ))
    
    if file_logging:
        callbacks.append(FileLogger(
            filepath=kwargs.get('log_filepath', 'optimization_log.json'),
            format=kwargs.get('log_format', 'json'),
            verbose=kwargs.get('verbose', True)
        ))
    
    if metrics_tracking:
        callbacks.append(MetricsTracker(
            metrics=kwargs.get('tracked_metrics', ['accuracy', 'f1_score']),
            verbose=kwargs.get('verbose', True)
        ))
    
    return callbacks


if __name__ == "__main__":
    # Example usage
    callbacks = create_callback_list(
        log_best=True,
        log_progress=True,
        early_stopping=True,
        verbose=True
    )
    
    print(f"Created {len(callbacks)} callbacks")
    for i, callback in enumerate(callbacks):
        print(f"  {i+1}. {type(callback).__name__}")
