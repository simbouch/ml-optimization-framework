#!/usr/bin/env python3
"""
Custom Optimizer Template
Use this template to create your own optimizers
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.optimizers import ModelOptimizer

class MyCustomOptimizer(ModelOptimizer):
    """
    Template for creating custom optimizers
    Replace this with your specific optimization logic
    """
    
    def define_search_space(self, trial):
        """
        Define the hyperparameter search space
        
        Args:
            trial: Optuna trial object
            
        Returns:
            dict: Dictionary of hyperparameters
        """
        return {
            'param1': trial.suggest_float('param1', 0.1, 1.0),
            'param2': trial.suggest_int('param2', 1, 100),
            'param3': trial.suggest_categorical('param3', ['option1', 'option2', 'option3'])
        }
    
    def _create_model(self, params):
        """
        Create a model with the given parameters
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            model: Trained model object
        """
        # Replace with your model creation logic
        from sklearn.ensemble import RandomForestClassifier
        
        return RandomForestClassifier(
            n_estimators=params['param2'],
            max_depth=int(params['param1'] * 20),
            random_state=42
        )
    
    def objective(self, trial, X, y):
        """
        Optional: Override the objective function for custom evaluation
        
        Args:
            trial: Optuna trial object
            X: Features
            y: Target
            
        Returns:
            float: Objective value to optimize
        """
        # Use default implementation or customize
        return super().objective(trial, X, y)

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from src.config import OptimizationConfig
    
    print("🔧 Custom Optimizer Template Example")
    print("=" * 50)
    
    # Generate sample data
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    
    # Create configuration
    config = OptimizationConfig(
        study_name="custom_template_test",
        n_trials=10,
        cv_folds=3
    )
    
    # Create and run optimizer
    optimizer = MyCustomOptimizer(config, task_type="classification")
    study = optimizer.optimize(X, y)
    
    print(f"\n✅ Template test completed!")
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    print(f"\n💡 Next steps:")
    print(f"1. Modify the define_search_space() method")
    print(f"2. Implement your model in _create_model()")
    print(f"3. Optionally override objective() for custom evaluation")
    print(f"4. Test with your own data")
