"""
Advanced Optuna features implementation.

This module provides implementations of advanced Optuna features including
multi-objective optimization, sampler comparison, custom search spaces,
and advanced pruning strategies.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler, GridSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner

from .config import OptimizationConfig
from .study_manager import StudyManager
from ..models.base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization using Optuna.
    
    Optimizes multiple objectives simultaneously and provides
    Pareto front analysis and trade-off visualization.
    """
    
    def __init__(
        self,
        objectives: List[str] = ["accuracy", "training_time"],
        directions: List[str] = ["maximize", "minimize"],
        random_state: int = 42
    ):
        """
        Initialize multi-objective optimizer.
        
        Args:
            objectives: List of objective names to optimize
            directions: List of optimization directions for each objective
            random_state: Random seed for reproducibility
        """
        self.objectives = objectives
        self.directions = directions
        self.random_state = random_state
        
        if len(objectives) != len(directions):
            raise ValueError("Number of objectives must match number of directions")
        
        self.study: Optional[optuna.Study] = None
        self.optimization_history: List[Dict[str, Any]] = []
        
        logger.info(f"Multi-objective optimizer initialized: {objectives}")
    
    def create_multi_objective_study(
        self,
        study_name: str = "multi_objective_study",
        storage: Optional[str] = None,
        sampler_name: str = "tpe"
    ) -> optuna.Study:
        """
        Create multi-objective study.
        
        Args:
            study_name: Name of the study
            storage: Storage URL for persistence
            sampler_name: Name of the sampler to use
            
        Returns:
            Multi-objective Optuna study
        """
        # Create sampler
        samplers = {
            "tpe": TPESampler(seed=self.random_state),
            "random": RandomSampler(seed=self.random_state),
            "cmaes": CmaEsSampler(seed=self.random_state)
        }
        
        sampler = samplers.get(sampler_name, TPESampler(seed=self.random_state))
        
        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            directions=self.directions,
            storage=storage,
            sampler=sampler,
            load_if_exists=True
        )
        
        logger.info(f"Multi-objective study created: {study_name}")
        return self.study
    
    def multi_objective_function(
        self,
        trial: Trial,
        model_optimizer: BaseOptimizer,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray
    ) -> List[float]:
        """
        Multi-objective function for optimization.
        
        Args:
            trial: Optuna trial object
            model_optimizer: Model optimizer instance
            X_train: Training features
            X_val: Validation features
            y_train: Training targets
            y_val: Validation targets
            
        Returns:
            List of objective values
        """
        start_time = time.time()
        
        # Create and train model
        model = model_optimizer.create_model(trial)
        model.fit(X_train, y_train)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate objectives
        objectives_values = []
        
        for objective in self.objectives:
            if objective == "accuracy":
                value = accuracy_score(y_val, y_pred)
            elif objective == "f1_score":
                value = f1_score(y_val, y_pred, average='weighted')
            elif objective == "roc_auc" and y_pred_proba is not None:
                value = roc_auc_score(y_val, y_pred_proba)
            elif objective == "training_time":
                value = training_time
            elif objective == "model_complexity":
                # Simple complexity measure based on model type
                if hasattr(model, 'n_estimators'):
                    value = model.n_estimators
                elif hasattr(model, 'C'):
                    value = 1.0 / model.C  # Inverse regularization
                else:
                    value = 1.0  # Default complexity
            else:
                raise ValueError(f"Unknown objective: {objective}")
            
            objectives_values.append(value)
        
        # Store trial information
        trial_info = {
            'trial_number': trial.number,
            'params': trial.params.copy(),
            'objectives': dict(zip(self.objectives, objectives_values)),
            'training_time': training_time
        }
        self.optimization_history.append(trial_info)
        
        return objectives_values
    
    def optimize(
        self,
        model_optimizer: BaseOptimizer,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 100
    ) -> optuna.Study:
        """
        Run multi-objective optimization.
        
        Args:
            model_optimizer: Model optimizer instance
            X_train: Training features
            X_val: Validation features
            y_train: Training targets
            y_val: Validation targets
            n_trials: Number of optimization trials
            
        Returns:
            Completed multi-objective study
        """
        if self.study is None:
            self.create_multi_objective_study()
        
        logger.info(f"Starting multi-objective optimization for {n_trials} trials")
        
        # Define objective function
        def objective(trial):
            return self.multi_objective_function(
                trial, model_optimizer, X_train, X_val, y_train, y_val
            )
        
        # Run optimization
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Multi-objective optimization completed!")
        logger.info(f"Number of Pareto optimal solutions: {len(self.study.best_trials)}")
        
        return self.study
    
    def get_pareto_front(self) -> pd.DataFrame:
        """
        Get Pareto front solutions.
        
        Returns:
            DataFrame containing Pareto optimal solutions
        """
        if self.study is None:
            raise ValueError("No study found. Run optimization first.")
        
        pareto_trials = self.study.best_trials
        
        pareto_data = []
        for trial in pareto_trials:
            trial_data = {
                'trial_number': trial.number,
                **trial.params,
                **dict(zip(self.objectives, trial.values))
            }
            pareto_data.append(trial_data)
        
        return pd.DataFrame(pareto_data)
    
    def analyze_trade_offs(self) -> Dict[str, Any]:
        """
        Analyze trade-offs between objectives.
        
        Returns:
            Dictionary containing trade-off analysis
        """
        pareto_df = self.get_pareto_front()
        
        if len(pareto_df) < 2:
            return {"error": "Not enough Pareto solutions for trade-off analysis"}
        
        # Calculate correlations between objectives
        objective_correlations = pareto_df[self.objectives].corr()
        
        # Find extreme solutions
        extreme_solutions = {}
        for objective in self.objectives:
            if self.directions[self.objectives.index(objective)] == "maximize":
                best_idx = pareto_df[objective].idxmax()
            else:
                best_idx = pareto_df[objective].idxmin()
            
            extreme_solutions[f"best_{objective}"] = pareto_df.loc[best_idx].to_dict()
        
        analysis = {
            'n_pareto_solutions': len(pareto_df),
            'objective_correlations': objective_correlations.to_dict(),
            'extreme_solutions': extreme_solutions,
            'objective_ranges': {
                obj: {
                    'min': pareto_df[obj].min(),
                    'max': pareto_df[obj].max(),
                    'mean': pareto_df[obj].mean(),
                    'std': pareto_df[obj].std()
                }
                for obj in self.objectives
            }
        }
        
        logger.info("Trade-off analysis completed")
        return analysis


class SamplerComparison:
    """
    Compare different Optuna samplers on the same optimization problem.
    """
    
    def __init__(
        self,
        samplers: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        """
        Initialize sampler comparison.
        
        Args:
            samplers: Dictionary of sampler configurations
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        
        if samplers is None:
            self.samplers = {
                "TPE": TPESampler(seed=random_state),
                "Random": RandomSampler(seed=random_state),
                "CMA-ES": CmaEsSampler(seed=random_state)
            }
        else:
            self.samplers = samplers
        
        self.comparison_results: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Sampler comparison initialized with: {list(self.samplers.keys())}")
    
    def compare_samplers(
        self,
        objective_function: Callable,
        n_trials: int = 100,
        n_runs: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare samplers on the given objective function.
        
        Args:
            objective_function: Function to optimize
            n_trials: Number of trials per run
            n_runs: Number of independent runs per sampler
            
        Returns:
            Dictionary containing comparison results
        """
        logger.info(f"Starting sampler comparison: {n_trials} trials, {n_runs} runs each")
        
        for sampler_name, sampler in self.samplers.items():
            logger.info(f"Testing sampler: {sampler_name}")
            
            run_results = []
            
            for run in range(n_runs):
                # Create study with current sampler
                study = optuna.create_study(
                    direction="maximize",
                    sampler=sampler,
                    study_name=f"{sampler_name}_run_{run}"
                )
                
                # Run optimization
                study.optimize(objective_function, n_trials=n_trials, show_progress_bar=False)
                
                # Store results
                run_results.append({
                    'best_value': study.best_value,
                    'best_params': study.best_params,
                    'n_trials': len(study.trials),
                    'convergence_history': [trial.value for trial in study.trials if trial.value is not None]
                })
            
            # Aggregate results
            best_values = [result['best_value'] for result in run_results]
            convergence_histories = [result['convergence_history'] for result in run_results]
            
            self.comparison_results[sampler_name] = {
                'mean_best_value': np.mean(best_values),
                'std_best_value': np.std(best_values),
                'min_best_value': np.min(best_values),
                'max_best_value': np.max(best_values),
                'convergence_histories': convergence_histories,
                'run_results': run_results
            }
            
            logger.info(f"{sampler_name}: {np.mean(best_values):.4f} ± {np.std(best_values):.4f}")
        
        return self.comparison_results
    
    def get_comparison_summary(self) -> pd.DataFrame:
        """
        Get summary of sampler comparison results.
        
        Returns:
            DataFrame containing comparison summary
        """
        if not self.comparison_results:
            raise ValueError("No comparison results found. Run compare_samplers first.")
        
        summary_data = []
        for sampler_name, results in self.comparison_results.items():
            summary_data.append({
                'sampler': sampler_name,
                'mean_best_value': results['mean_best_value'],
                'std_best_value': results['std_best_value'],
                'min_best_value': results['min_best_value'],
                'max_best_value': results['max_best_value']
            })
        
        return pd.DataFrame(summary_data).sort_values('mean_best_value', ascending=False)


class PrunerComparison:
    """
    Compare different Optuna pruners for early stopping.
    """
    
    def __init__(
        self,
        pruners: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize pruner comparison.
        
        Args:
            pruners: Dictionary of pruner configurations
        """
        if pruners is None:
            self.pruners = {
                "Median": MedianPruner(n_startup_trials=5, n_warmup_steps=5),
                "SuccessiveHalving": SuccessiveHalvingPruner(),
                "Hyperband": HyperbandPruner(),
                "NoPruning": optuna.pruners.NopPruner()
            }
        else:
            self.pruners = pruners
        
        self.comparison_results: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Pruner comparison initialized with: {list(self.pruners.keys())}")
    
    def compare_pruners(
        self,
        objective_function: Callable,
        n_trials: int = 100,
        n_runs: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare pruners on the given objective function.
        
        Args:
            objective_function: Function to optimize (should support pruning)
            n_trials: Number of trials per run
            n_runs: Number of independent runs per pruner
            
        Returns:
            Dictionary containing comparison results
        """
        logger.info(f"Starting pruner comparison: {n_trials} trials, {n_runs} runs each")
        
        for pruner_name, pruner in self.pruners.items():
            logger.info(f"Testing pruner: {pruner_name}")
            
            run_results = []
            
            for run in range(n_runs):
                # Create study with current pruner
                study = optuna.create_study(
                    direction="maximize",
                    pruner=pruner,
                    study_name=f"{pruner_name}_run_{run}"
                )
                
                # Run optimization
                study.optimize(objective_function, n_trials=n_trials, show_progress_bar=False)
                
                # Analyze pruning statistics
                completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
                
                run_results.append({
                    'best_value': study.best_value,
                    'n_completed': len(completed_trials),
                    'n_pruned': len(pruned_trials),
                    'pruning_rate': len(pruned_trials) / len(study.trials),
                    'total_trials': len(study.trials)
                })
            
            # Aggregate results
            best_values = [result['best_value'] for result in run_results]
            pruning_rates = [result['pruning_rate'] for result in run_results]
            
            self.comparison_results[pruner_name] = {
                'mean_best_value': np.mean(best_values),
                'std_best_value': np.std(best_values),
                'mean_pruning_rate': np.mean(pruning_rates),
                'std_pruning_rate': np.std(pruning_rates),
                'run_results': run_results
            }
            
            logger.info(f"{pruner_name}: {np.mean(best_values):.4f} ± {np.std(best_values):.4f} "
                       f"(pruning: {np.mean(pruning_rates):.2%})")
        
        return self.comparison_results
    
    def get_comparison_summary(self) -> pd.DataFrame:
        """
        Get summary of pruner comparison results.
        
        Returns:
            DataFrame containing comparison summary
        """
        if not self.comparison_results:
            raise ValueError("No comparison results found. Run compare_pruners first.")
        
        summary_data = []
        for pruner_name, results in self.comparison_results.items():
            summary_data.append({
                'pruner': pruner_name,
                'mean_best_value': results['mean_best_value'],
                'std_best_value': results['std_best_value'],
                'mean_pruning_rate': results['mean_pruning_rate'],
                'std_pruning_rate': results['std_pruning_rate']
            })
        
        return pd.DataFrame(summary_data).sort_values('mean_best_value', ascending=False)


class CustomSearchSpace:
    """
    Custom search space implementations for advanced hyperparameter optimization.
    """
    
    @staticmethod
    def conditional_search_space(trial: Trial, model_type: str) -> Dict[str, Any]:
        """
        Create conditional search space based on model type.
        
        Args:
            trial: Optuna trial object
            model_type: Type of model ('random_forest', 'xgboost', 'lightgbm')
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        if model_type == "random_forest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            
            # Conditional: if bootstrap is True, add oob_score option
            bootstrap = trial.suggest_categorical('bootstrap', [True, False])
            params['bootstrap'] = bootstrap
            
            if bootstrap:
                params['oob_score'] = trial.suggest_categorical('oob_score', [True, False])
            else:
                params['max_samples'] = trial.suggest_float('max_samples', 0.5, 1.0)
        
        elif model_type == "xgboost":
            booster = trial.suggest_categorical('booster', ['gbtree', 'gblinear'])
            params = {
                'booster': booster,
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            }
            
            # Conditional parameters based on booster type
            if booster == 'gbtree':
                params.update({
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                })
        
        elif model_type == "lightgbm":
            boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])
            params = {
                'boosting_type': boosting_type,
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            }
            
            # Conditional parameters based on boosting type
            if boosting_type == 'dart':
                params.update({
                    'drop_rate': trial.suggest_float('drop_rate', 0.1, 0.5),
                    'skip_drop': trial.suggest_float('skip_drop', 0.1, 0.9)
                })
            elif boosting_type == 'goss':
                params.update({
                    'top_rate': trial.suggest_float('top_rate', 0.1, 0.5),
                    'other_rate': trial.suggest_float('other_rate', 0.05, 0.2)
                })
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return params
    
    @staticmethod
    def hierarchical_search_space(trial: Trial) -> Dict[str, Any]:
        """
        Create hierarchical search space with dependencies.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        # First level: Choose preprocessing strategy
        preprocessing = trial.suggest_categorical('preprocessing', ['standard', 'robust', 'minmax'])
        
        params = {'preprocessing': preprocessing}
        
        # Second level: Choose model type
        model_type = trial.suggest_categorical('model_type', ['rf', 'xgb', 'lgb'])
        params['model_type'] = model_type
        
        # Third level: Model-specific parameters
        if model_type == 'rf':
            params.update({
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15)
            })
        elif model_type == 'xgb':
            params.update({
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
            })
        elif model_type == 'lgb':
            params.update({
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'num_leaves': trial.suggest_int('num_leaves', 10, 200)
            })
        
        return params


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from ..models.random_forest_optimizer import RandomForestOptimizer
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Multi-objective optimization example
    multi_opt = MultiObjectiveOptimizer(
        objectives=["accuracy", "training_time"],
        directions=["maximize", "minimize"]
    )
    
    rf_optimizer = RandomForestOptimizer()
    study = multi_opt.optimize(rf_optimizer, X_train, X_val, y_train, y_val, n_trials=20)
    
    # Analyze results
    pareto_front = multi_opt.get_pareto_front()
    trade_offs = multi_opt.analyze_trade_offs()
    
    print(f"Pareto front size: {len(pareto_front)}")
    print(f"Trade-off analysis: {trade_offs}")
    
    # Sampler comparison example
    def simple_objective(trial):
        x = trial.suggest_float('x', -10, 10)
        y = trial.suggest_float('y', -10, 10)
        return -(x**2 + y**2)  # Minimize x^2 + y^2
    
    sampler_comp = SamplerComparison()
    sampler_results = sampler_comp.compare_samplers(simple_objective, n_trials=50, n_runs=2)
    summary = sampler_comp.get_comparison_summary()
    
    print("Sampler comparison results:")
    print(summary)
