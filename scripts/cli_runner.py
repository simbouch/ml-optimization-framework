#!/usr/bin/env python3
"""
Command-line interface for ML optimization framework.

This script provides a comprehensive CLI for running hyperparameter optimization
with different models, configurations, and analysis options.
"""

import argparse
import logging
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_pipeline import DataPipeline
from src.models.random_forest_optimizer import RandomForestOptimizer
from src.models.xgboost_optimizer import XGBoostOptimizer
from src.models.lightgbm_optimizer import LightGBMOptimizer
from src.optimization.config import OptimizationConfig
from src.optimization.study_manager import StudyManager
from src.optimization.callbacks import create_callback_list
from src.optimization.advanced_features import MultiObjectiveOptimizer, SamplerComparison
from src.visualization.plots import OptimizationPlotter
import pandas as pd
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizationCLI:
    """
    Command-line interface for ML optimization framework.
    
    Provides comprehensive CLI functionality for running optimizations,
    comparing models, and generating analysis reports.
    """
    
    def __init__(self):
        """Initialize the CLI."""
        self.data_pipeline = None
        self.optimizers = {}
        self.study_manager = None
        self.config = None
        self.plotter = OptimizationPlotter()
        
    def setup_data_pipeline(self, args: argparse.Namespace) -> None:
        """Setup data pipeline with CLI arguments."""
        logger.info("Setting up data pipeline...")
        
        self.data_pipeline = DataPipeline(
            random_state=args.random_state,
            test_size=args.test_size,
            val_size=args.val_size
        )
        
        # Prepare data
        summary = self.data_pipeline.prepare_data()
        logger.info(f"Data preparation completed: {summary}")
    
    def setup_configuration(self, args: argparse.Namespace) -> None:
        """Setup optimization configuration."""
        logger.info("Setting up optimization configuration...")
        
        config_path = args.config_path if args.config_path else None
        self.config = OptimizationConfig(config_path)
        
        # Setup study manager
        storage_url = args.storage_url or "sqlite:///optuna_study.db"
        self.study_manager = StudyManager(storage_url, self.config)
        
        logger.info("Configuration setup completed")
    
    def setup_optimizers(self, args: argparse.Namespace) -> None:
        """Setup model optimizers."""
        logger.info("Setting up model optimizers...")
        
        optimizer_configs = {
            'random_forest': {
                'class': RandomForestOptimizer,
                'kwargs': {
                    'random_state': args.random_state,
                    'cv_folds': args.cv_folds,
                    'scoring_metric': args.scoring_metric,
                    'verbose': args.verbose,
                    'config': self.config
                }
            },
            'xgboost': {
                'class': XGBoostOptimizer,
                'kwargs': {
                    'random_state': args.random_state,
                    'cv_folds': args.cv_folds,
                    'scoring_metric': args.scoring_metric,
                    'early_stopping_rounds': args.early_stopping_rounds,
                    'verbose': args.verbose,
                    'config': self.config,
                    'use_gpu': args.use_gpu
                }
            },
            'lightgbm': {
                'class': LightGBMOptimizer,
                'kwargs': {
                    'random_state': args.random_state,
                    'cv_folds': args.cv_folds,
                    'scoring_metric': args.scoring_metric,
                    'early_stopping_rounds': args.early_stopping_rounds,
                    'verbose': args.verbose,
                    'config': self.config,
                    'use_gpu': args.use_gpu
                }
            }
        }
        
        # Initialize requested optimizers
        models = args.model if isinstance(args.model, list) else [args.model]
        if 'all' in models:
            models = list(optimizer_configs.keys())
        
        for model_name in models:
            if model_name in optimizer_configs:
                config = optimizer_configs[model_name]
                self.optimizers[model_name] = config['class'](**config['kwargs'])
                logger.info(f"Initialized {model_name} optimizer")
        
        logger.info(f"Setup completed for {len(self.optimizers)} optimizers")
    
    def run_single_optimization(
        self,
        model_name: str,
        args: argparse.Namespace
    ) -> Dict[str, Any]:
        """Run optimization for a single model."""
        logger.info(f"Starting optimization for {model_name}")
        
        optimizer = self.optimizers[model_name]
        
        # Get data
        X_train, X_val, y_train, y_val = self.data_pipeline.get_train_val_data()
        
        # Create callbacks
        callbacks = create_callback_list(
            log_best=True,
            log_progress=True,
            early_stopping=args.early_stopping,
            file_logging=args.save_logs,
            verbose=args.verbose,
            patience=args.patience,
            log_filepath=f"{model_name}_optimization.json"
        )
        
        # Create study
        study = self.study_manager.create_study(
            study_name=f"{model_name}_{int(time.time())}",
            model_name=model_name,
            load_if_exists=False
        )
        
        # Run optimization
        start_time = time.time()
        study = optimizer.optimize(
            X_train, X_val, y_train, y_val,
            n_trials=args.n_trials,
            study=study,
            callbacks=callbacks
        )
        optimization_time = time.time() - start_time
        
        # Evaluate on test set
        X_test, y_test = self.data_pipeline.get_test_data()
        test_metrics = optimizer.evaluate(X_test, y_test)
        
        # Get optimization summary
        summary = optimizer.get_optimization_summary()
        summary.update({
            'optimization_time': optimization_time,
            'test_metrics': test_metrics
        })
        
        logger.info(f"Optimization completed for {model_name}")
        logger.info(f"Best score: {summary['best_score']:.4f}")
        logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        
        return {
            'model_name': model_name,
            'optimizer': optimizer,
            'study': study,
            'summary': summary
        }
    
    def run_multi_objective_optimization(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Run multi-objective optimization."""
        logger.info("Starting multi-objective optimization")
        
        # Setup multi-objective optimizer
        objectives = args.objectives.split(',') if args.objectives else ["accuracy", "training_time"]
        directions = args.directions.split(',') if args.directions else ["maximize", "minimize"]
        
        multi_opt = MultiObjectiveOptimizer(
            objectives=objectives,
            directions=directions,
            random_state=args.random_state
        )
        
        # Use first available optimizer
        model_name = list(self.optimizers.keys())[0]
        optimizer = self.optimizers[model_name]
        
        # Get data
        X_train, X_val, y_train, y_val = self.data_pipeline.get_train_val_data()
        
        # Run optimization
        study = multi_opt.optimize(
            optimizer, X_train, X_val, y_train, y_val,
            n_trials=args.n_trials
        )
        
        # Analyze results
        pareto_front = multi_opt.get_pareto_front()
        trade_offs = multi_opt.analyze_trade_offs()
        
        logger.info(f"Multi-objective optimization completed")
        logger.info(f"Pareto front size: {len(pareto_front)}")
        
        return {
            'study': study,
            'pareto_front': pareto_front,
            'trade_offs': trade_offs,
            'multi_optimizer': multi_opt
        }
    
    def run_sampler_comparison(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Run sampler comparison."""
        logger.info("Starting sampler comparison")
        
        # Use first available optimizer
        model_name = list(self.optimizers.keys())[0]
        optimizer = self.optimizers[model_name]
        
        # Get data
        X_train, X_val, y_train, y_val = self.data_pipeline.get_train_val_data()
        
        # Define objective function
        def objective(trial):
            model = optimizer.create_model(trial)
            scores = optimizer._cross_validate(model, trial)
            return scores.mean()
        
        # Run comparison
        sampler_comp = SamplerComparison(random_state=args.random_state)
        results = sampler_comp.compare_samplers(
            objective,
            n_trials=args.n_trials // 4,  # Fewer trials per sampler
            n_runs=3
        )
        
        summary = sampler_comp.get_comparison_summary()
        
        logger.info("Sampler comparison completed")
        logger.info(f"Best sampler: {summary.iloc[0]['sampler']}")
        
        return {
            'results': results,
            'summary': summary,
            'comparison': sampler_comp
        }
    
    def generate_visualizations(
        self,
        results: List[Dict[str, Any]],
        args: argparse.Namespace
    ) -> None:
        """Generate optimization visualizations."""
        if not args.generate_plots:
            return
        
        logger.info("Generating visualizations...")
        
        for result in results:
            model_name = result['model_name']
            study = result['study']
            
            # Create plots directory
            plots_dir = Path("plots")
            plots_dir.mkdir(exist_ok=True)
            
            try:
                # Optimization history
                fig = self.plotter.plot_optimization_history_custom(
                    study, interactive=args.interactive_plots
                )
                if args.interactive_plots:
                    fig.write_html(plots_dir / f"{model_name}_history.html")
                else:
                    fig.savefig(plots_dir / f"{model_name}_history.png", dpi=300, bbox_inches='tight')
                
                # Parameter importance
                fig = self.plotter.plot_parameter_importance_custom(
                    study, interactive=args.interactive_plots
                )
                if args.interactive_plots:
                    fig.write_html(plots_dir / f"{model_name}_importance.html")
                else:
                    fig.savefig(plots_dir / f"{model_name}_importance.png", dpi=300, bbox_inches='tight')
                
                logger.info(f"Plots saved for {model_name}")
                
            except Exception as e:
                logger.warning(f"Error generating plots for {model_name}: {str(e)}")
        
        logger.info("Visualization generation completed")
    
    def save_results(
        self,
        results: List[Dict[str, Any]],
        args: argparse.Namespace
    ) -> None:
        """Save optimization results."""
        if not args.save_results:
            return
        
        logger.info("Saving results...")
        
        # Create results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Save individual results
        for result in results:
            model_name = result['model_name']
            summary = result['summary']
            
            # Save summary
            with open(results_dir / f"{model_name}_summary.json", 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Save study data
            study_df = result['study'].trials_dataframe()
            study_df.to_csv(results_dir / f"{model_name}_trials.csv", index=False)
        
        # Save comparison results
        if len(results) > 1:
            comparison_data = []
            for result in results:
                summary = result['summary']
                comparison_data.append({
                    'model': summary['model_name'],
                    'best_score': summary['best_score'],
                    'n_trials': summary['n_trials'],
                    'optimization_time': summary['optimization_time'],
                    'test_accuracy': summary['test_metrics']['accuracy'],
                    'test_f1': summary['test_metrics']['f1_score']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(results_dir / "model_comparison.csv", index=False)
        
        logger.info("Results saved successfully")
    
    def run(self, args: argparse.Namespace) -> None:
        """Run the optimization based on CLI arguments."""
        logger.info("Starting ML optimization framework")
        
        # Setup components
        self.setup_data_pipeline(args)
        self.setup_configuration(args)
        self.setup_optimizers(args)
        
        results = []
        
        # Run optimizations based on mode
        if args.mode == 'single':
            for model_name in self.optimizers.keys():
                result = self.run_single_optimization(model_name, args)
                results.append(result)
        
        elif args.mode == 'multi_objective':
            result = self.run_multi_objective_optimization(args)
            # Handle multi-objective results differently
            logger.info("Multi-objective optimization completed")
            return
        
        elif args.mode == 'sampler_comparison':
            result = self.run_sampler_comparison(args)
            # Handle sampler comparison results differently
            logger.info("Sampler comparison completed")
            return
        
        # Generate outputs
        self.generate_visualizations(results, args)
        self.save_results(results, args)
        
        # Print summary
        if results:
            logger.info("\n" + "="*50)
            logger.info("OPTIMIZATION SUMMARY")
            logger.info("="*50)
            
            for result in results:
                summary = result['summary']
                logger.info(f"\nModel: {summary['model_name']}")
                logger.info(f"Best CV Score: {summary['best_score']:.4f}")
                logger.info(f"Test Accuracy: {summary['test_metrics']['accuracy']:.4f}")
                logger.info(f"Optimization Time: {summary['optimization_time']:.2f}s")
                logger.info(f"Trials: {summary['n_trials']}")
        
        logger.info("\nOptimization framework completed successfully!")


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="ML Optimization Framework with Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model optimization
  python cli_runner.py --model random_forest --n_trials 100
  
  # Compare multiple models
  python cli_runner.py --model all --n_trials 50 --compare
  
  # Multi-objective optimization
  python cli_runner.py --mode multi_objective --model xgboost --n_trials 100
  
  # Sampler comparison
  python cli_runner.py --mode sampler_comparison --model lightgbm --n_trials 200
        """
    )
    
    # Main arguments
    parser.add_argument(
        '--model', 
        choices=['random_forest', 'xgboost', 'lightgbm', 'all'],
        default='random_forest',
        help='Model(s) to optimize'
    )
    
    parser.add_argument(
        '--mode',
        choices=['single', 'multi_objective', 'sampler_comparison'],
        default='single',
        help='Optimization mode'
    )
    
    parser.add_argument(
        '--n_trials',
        type=int,
        default=100,
        help='Number of optimization trials'
    )
    
    # Data arguments
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Test set size ratio'
    )
    
    parser.add_argument(
        '--val_size',
        type=float,
        default=0.2,
        help='Validation set size ratio'
    )
    
    # Optimization arguments
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    
    parser.add_argument(
        '--scoring_metric',
        choices=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
        default='accuracy',
        help='Scoring metric for optimization'
    )
    
    parser.add_argument(
        '--early_stopping_rounds',
        type=int,
        default=50,
        help='Early stopping rounds for tree models'
    )
    
    parser.add_argument(
        '--early_stopping',
        action='store_true',
        help='Enable early stopping in optimization'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Patience for early stopping'
    )
    
    # Multi-objective arguments
    parser.add_argument(
        '--objectives',
        type=str,
        default='accuracy,training_time',
        help='Comma-separated list of objectives for multi-objective optimization'
    )
    
    parser.add_argument(
        '--directions',
        type=str,
        default='maximize,minimize',
        help='Comma-separated list of directions for multi-objective optimization'
    )
    
    # Configuration arguments
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--storage_url',
        type=str,
        help='Optuna storage URL'
    )
    
    # Output arguments
    parser.add_argument(
        '--save_results',
        action='store_true',
        help='Save optimization results'
    )
    
    parser.add_argument(
        '--save_logs',
        action='store_true',
        help='Save optimization logs'
    )
    
    parser.add_argument(
        '--generate_plots',
        action='store_true',
        help='Generate optimization plots'
    )
    
    parser.add_argument(
        '--interactive_plots',
        action='store_true',
        help='Generate interactive plots'
    )
    
    # System arguments
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Use GPU acceleration for tree models'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run optimization
    cli = OptimizationCLI()
    try:
        cli.run(args)
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
