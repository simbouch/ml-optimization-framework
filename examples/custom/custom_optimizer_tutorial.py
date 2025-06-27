#!/usr/bin/env python3
"""
Custom Optimizer Tutorial
Learn how to build your own optimizers using the framework
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import optuna
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
from pathlib import Path

# Import framework components
from src.config import OptimizationConfig
from src.optimizers import ModelOptimizer

class NeuralNetworkOptimizer(ModelOptimizer):
    """
    Custom Neural Network Optimizer
    Demonstrates how to create a custom optimizer for neural networks
    """
    
    def define_search_space(self, trial):
        """Define the hyperparameter search space for neural networks"""
        return {
            'hidden_layer_sizes': trial.suggest_categorical(
                'hidden_layer_sizes', 
                [(50,), (100,), (50, 50), (100, 50), (100, 100), (200, 100, 50)]
            ),
            'activation': trial.suggest_categorical(
                'activation', 
                ['relu', 'tanh', 'logistic']
            ),
            'solver': trial.suggest_categorical(
                'solver', 
                ['adam', 'lbfgs', 'sgd']
            ),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'learning_rate': trial.suggest_categorical(
                'learning_rate', 
                ['constant', 'invscaling', 'adaptive']
            ),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),
            'validation_fraction': trial.suggest_float('validation_fraction', 0.1, 0.3)
        }
    
    def _create_model(self, params):
        """Create a neural network model with given parameters"""
        if self.task_type == "classification":
            return MLPClassifier(
                hidden_layer_sizes=params['hidden_layer_sizes'],
                activation=params['activation'],
                solver=params['solver'],
                alpha=params['alpha'],
                learning_rate=params['learning_rate'],
                learning_rate_init=params['learning_rate_init'],
                max_iter=params['max_iter'],
                early_stopping=params['early_stopping'],
                validation_fraction=params['validation_fraction'],
                random_state=42
            )
        else:  # regression
            return MLPRegressor(
                hidden_layer_sizes=params['hidden_layer_sizes'],
                activation=params['activation'],
                solver=params['solver'],
                alpha=params['alpha'],
                learning_rate=params['learning_rate'],
                learning_rate_init=params['learning_rate_init'],
                max_iter=params['max_iter'],
                early_stopping=params['early_stopping'],
                validation_fraction=params['validation_fraction'],
                random_state=42
            )

class EnsembleOptimizer(ModelOptimizer):
    """
    Custom Ensemble Optimizer
    Optimizes a combination of different algorithms
    """
    
    def define_search_space(self, trial):
        """Define search space for ensemble methods"""
        return {
            'base_model': trial.suggest_categorical(
                'base_model', 
                ['logistic_regression', 'svm', 'neural_network']
            ),
            'ensemble_method': trial.suggest_categorical(
                'ensemble_method',
                ['voting', 'stacking', 'bagging']
            ),
            # Model-specific parameters
            'lr_C': trial.suggest_float('lr_C', 1e-3, 1e3, log=True),
            'svm_C': trial.suggest_float('svm_C', 1e-3, 1e3, log=True),
            'svm_gamma': trial.suggest_categorical('svm_gamma', ['scale', 'auto']),
            'nn_hidden_size': trial.suggest_int('nn_hidden_size', 50, 200),
            'nn_alpha': trial.suggest_float('nn_alpha', 1e-5, 1e-1, log=True)
        }
    
    def _create_model(self, params):
        """Create an ensemble model"""
        from sklearn.ensemble import VotingClassifier, BaggingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        
        # Create base models
        lr = LogisticRegression(C=params['lr_C'], random_state=42, max_iter=1000)
        svm = SVC(C=params['svm_C'], gamma=params['svm_gamma'], random_state=42, probability=True)
        nn = MLPClassifier(
            hidden_layer_sizes=(params['nn_hidden_size'],),
            alpha=params['nn_alpha'],
            random_state=42,
            max_iter=500
        )
        
        if params['ensemble_method'] == 'voting':
            return VotingClassifier(
                estimators=[('lr', lr), ('svm', svm), ('nn', nn)],
                voting='soft'
            )
        elif params['ensemble_method'] == 'bagging':
            # Use the selected base model for bagging
            if params['base_model'] == 'logistic_regression':
                base = lr
            elif params['base_model'] == 'svm':
                base = svm
            else:
                base = nn
            
            return BaggingClassifier(
                base_estimator=base,
                n_estimators=10,
                random_state=42
            )
        else:  # stacking
            from sklearn.ensemble import StackingClassifier
            return StackingClassifier(
                estimators=[('lr', lr), ('svm', svm), ('nn', nn)],
                final_estimator=LogisticRegression(random_state=42),
                cv=3
            )

class TimeSeriesOptimizer(ModelOptimizer):
    """
    Custom Time Series Optimizer
    Specialized for time series forecasting problems
    """
    
    def define_search_space(self, trial):
        """Define search space for time series models"""
        return {
            'model_type': trial.suggest_categorical(
                'model_type',
                ['linear', 'polynomial', 'neural_network']
            ),
            'window_size': trial.suggest_int('window_size', 5, 50),
            'polynomial_degree': trial.suggest_int('polynomial_degree', 1, 5),
            'regularization': trial.suggest_float('regularization', 1e-5, 1e-1, log=True),
            'nn_layers': trial.suggest_int('nn_layers', 1, 3),
            'nn_units': trial.suggest_int('nn_units', 32, 128)
        }
    
    def _create_time_series_features(self, X, window_size):
        """Create time series features with sliding window"""
        features = []
        targets = []
        
        for i in range(window_size, len(X)):
            features.append(X[i-window_size:i])
            targets.append(X[i])
        
        return np.array(features), np.array(targets)
    
    def _create_model(self, params):
        """Create a time series model"""
        if params['model_type'] == 'linear':
            return LinearRegression()
        elif params['model_type'] == 'polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import Pipeline
            from sklearn.linear_model import Ridge
            
            return Pipeline([
                ('poly', PolynomialFeatures(degree=params['polynomial_degree'])),
                ('ridge', Ridge(alpha=params['regularization']))
            ])
        else:  # neural_network
            layers = [params['nn_units']] * params['nn_layers']
            return MLPRegressor(
                hidden_layer_sizes=tuple(layers),
                alpha=params['regularization'],
                random_state=42,
                max_iter=500
            )
    
    def objective(self, trial, X, y):
        """Custom objective function for time series"""
        params = self.define_search_space(trial)
        
        # Create time series features
        X_ts, y_ts = self._create_time_series_features(X.flatten(), params['window_size'])
        
        if len(X_ts) < 10:  # Not enough data
            return 0.0
        
        # Split data (time series split)
        split_point = int(0.8 * len(X_ts))
        X_train, X_test = X_ts[:split_point], X_ts[split_point:]
        y_train, y_test = y_ts[:split_point], y_ts[split_point:]
        
        # Create and train model
        model = self._create_model(params)
        
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Use negative MSE as score (to maximize)
            mse = mean_squared_error(y_test, predictions)
            return -mse
        except Exception:
            return -1e6  # Large penalty for failed models

def demonstrate_custom_optimizers():
    """Demonstrate all custom optimizers"""
    print("🎯 Custom Optimizer Demonstrations")
    print("=" * 60)
    
    # Ensure directories exist
    Path("studies").mkdir(exist_ok=True)
    
    # Generate datasets
    X_class, y_class = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, random_state=42)
    
    # Configuration
    config = OptimizationConfig(
        n_trials=30,
        cv_folds=3,
        random_seed=42
    )
    
    results = {}
    
    # 1. Neural Network Optimizer
    print("\n1️⃣ Testing Neural Network Optimizer...")
    config.study_name = "custom_neural_network"
    nn_optimizer = NeuralNetworkOptimizer(config, task_type="classification")
    nn_study = nn_optimizer.optimize(X_class, y_class)
    results['Neural Network'] = nn_study.best_value
    print(f"   Best accuracy: {nn_study.best_value:.4f}")
    
    # 2. Ensemble Optimizer
    print("\n2️⃣ Testing Ensemble Optimizer...")
    config.study_name = "custom_ensemble"
    ensemble_optimizer = EnsembleOptimizer(config, task_type="classification")
    ensemble_study = ensemble_optimizer.optimize(X_class, y_class)
    results['Ensemble'] = ensemble_study.best_value
    print(f"   Best accuracy: {ensemble_study.best_value:.4f}")
    
    # 3. Time Series Optimizer
    print("\n3️⃣ Testing Time Series Optimizer...")
    config.study_name = "custom_time_series"
    ts_optimizer = TimeSeriesOptimizer(config, task_type="regression")
    ts_study = ts_optimizer.optimize(X_reg[:, 0:1], y_reg)  # Use first feature as time series
    results['Time Series'] = -ts_study.best_value  # Convert back from negative MSE
    print(f"   Best MSE: {-ts_study.best_value:.4f}")
    
    return results

def create_custom_optimizer_template():
    """Create a template for building custom optimizers"""
    template_code = '''
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
'''
    
    print("\n📝 Custom Optimizer Template:")
    print("=" * 50)
    print(template_code)
    
    # Save template to file
    with open("examples/custom/custom_optimizer_template.py", "w") as f:
        f.write("#!/usr/bin/env python3\n")
        f.write('"""\nCustom Optimizer Template\nUse this template to create your own optimizers\n"""\n\n')
        f.write("import sys\nimport os\n")
        f.write("sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))\n\n")
        f.write("from src.optimizers import ModelOptimizer\n\n")
        f.write(template_code)
    
    print("✅ Template saved to examples/custom/custom_optimizer_template.py")

def main():
    """Run custom optimizer tutorial"""
    print("🎯 Custom Optimizer Tutorial")
    print("=" * 60)
    
    try:
        # Demonstrate custom optimizers
        results = demonstrate_custom_optimizers()
        
        # Create template
        create_custom_optimizer_template()
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 Custom Optimizer Tutorial Complete!")
        
        print("\n📊 Results Summary:")
        for optimizer, score in results.items():
            print(f"  {optimizer}: {score:.4f}")
        
        print("\n📁 Files Created:")
        print("  - examples/custom/custom_optimizer_template.py")
        print("  - studies/custom_neural_network.db")
        print("  - studies/custom_ensemble.db") 
        print("  - studies/custom_time_series.db")
        
        print("\n💡 Key Learnings:")
        print("  - Inherit from ModelOptimizer base class")
        print("  - Implement define_search_space() method")
        print("  - Implement _create_model() method")
        print("  - Optionally override objective() for custom evaluation")
        print("  - Use trial.suggest_* methods for hyperparameters")
        
        print("\n📍 Next Steps:")
        print("  - Modify the template for your specific use case")
        print("  - Add custom evaluation metrics")
        print("  - Implement domain-specific constraints")
        print("  - Integrate with your existing ML pipeline")
        
    except Exception as e:
        print(f"\n❌ Error in tutorial: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
