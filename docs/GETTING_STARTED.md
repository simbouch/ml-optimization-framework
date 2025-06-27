# Getting Started with ML Optimization Framework

Welcome to the ML Optimization Framework! This guide will help you get up and running quickly with Optuna-powered machine learning optimization.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/simbouch/ml-optimization-framework.git
cd ml-optimization-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Your First Optimization

```bash
# Quick demo with all Optuna features
python quick_demo.py

# Comprehensive demo with real ML scenarios
python comprehensive_optuna_demo.py

# Interactive Streamlit app
streamlit run simple_app.py
```

### 3. Docker Deployment

```bash
# Build and start services
docker-compose up -d

# Access applications
# Streamlit App: http://localhost:8501
# Optuna Dashboard: http://localhost:8080
```

## 📚 Core Concepts

### OptimizationConfig
Central configuration class that manages all optimization settings:

```python
from src.config import OptimizationConfig

config = OptimizationConfig(
    study_name="my_optimization",
    n_trials=100,
    sampler_name="TPE",
    pruner_name="Median"
)
```

### ModelOptimizer
Base class for implementing ML model optimizations:

```python
from src.optimizers import RandomForestOptimizer

optimizer = RandomForestOptimizer(config, task_type="classification")
study = optimizer.optimize(X_train, y_train)
```

### StudyManager
Manages Optuna studies with persistence and analysis:

```python
from src.study_manager import StudyManager

manager = StudyManager(config)
study = manager.create_study("my_study", direction="maximize")
results_df = manager.export_study_results(study.study_name)
```

## 🎯 Use Cases

### 1. Hyperparameter Optimization
Optimize ML model hyperparameters with various algorithms:

```python
# Random Forest optimization
rf_optimizer = RandomForestOptimizer(config, task_type="classification")
study = rf_optimizer.optimize(X, y)

# XGBoost optimization  
xgb_optimizer = XGBoostOptimizer(config, task_type="regression")
study = xgb_optimizer.optimize(X, y)
```

### 2. Multi-Objective Optimization
Optimize multiple objectives simultaneously:

```python
config = OptimizationConfig(
    study_name="multi_objective",
    directions=["maximize", "minimize"]  # accuracy vs model size
)
```

### 3. Advanced Sampling Strategies
Choose from multiple sampling algorithms:

- **TPE (Tree-structured Parzen Estimator)**: Default, works well for most cases
- **CMA-ES**: Good for continuous optimization
- **Random**: Baseline comparison
- **Grid**: Exhaustive search for small spaces
- **QMC**: Quasi-Monte Carlo for better coverage

### 4. Pruning Strategies
Stop unpromising trials early:

- **Median**: Stop if below median performance
- **SuccessiveHalving**: Resource allocation strategy
- **Hyperband**: Advanced early stopping

## 🔧 Configuration Options

### Environment Variables
Create a `.env` file (copy from `.env.example`):

```bash
# Service Ports
STREAMLIT_PORT=8501
OPTUNA_DASHBOARD_PORT=8080

# Study Configuration
STUDY_NAME=ml_optimization_study
N_TRIALS=100
SAMPLER_NAME=TPE
PRUNER_NAME=Median

# Performance Settings
CV_FOLDS=5
TEST_SIZE=0.2
MEMORY_LIMIT_MB=4096
```

### Programmatic Configuration
```python
config = OptimizationConfig(
    study_name="custom_study",
    n_trials=50,
    sampler_name="CMA-ES",
    pruner_name="SuccessiveHalving",
    cv_folds=3,
    test_size=0.3,
    random_seed=42
)
```

## 📊 Monitoring & Analysis

### Optuna Dashboard
Access the web-based dashboard at `http://localhost:8080` to:
- View optimization history
- Analyze parameter importance
- Compare different studies
- Export results

### Streamlit Interface
Use the interactive app at `http://localhost:8501` to:
- Run optimizations with custom parameters
- Visualize results in real-time
- Download optimization reports
- Manage multiple studies

### Programmatic Analysis
```python
# Get study summary
summary = manager.get_study_summary("my_study")

# Export results
df = manager.export_study_results("my_study", format="csv")

# Get best parameters
best_params = study.best_params
best_value = study.best_value
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated
2. **Port Conflicts**: Change ports in `.env` file
3. **Memory Issues**: Reduce `n_trials` or adjust `MEMORY_LIMIT_MB`
4. **Database Locks**: Stop all processes before restarting

### Performance Tips

1. **Start Small**: Begin with 10-20 trials, then scale up
2. **Use Pruning**: Enable early stopping for faster results
3. **Parallel Execution**: Run multiple studies simultaneously
4. **Resource Monitoring**: Watch memory and CPU usage

## 📖 Next Steps

- Read the [Advanced Usage Guide](ADVANCED_USAGE.md)
- Explore [API Documentation](API_REFERENCE.md)
- Check out [Examples](../examples/)
- Join our [Community](https://github.com/simbouch/ml-optimization-framework/discussions)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
