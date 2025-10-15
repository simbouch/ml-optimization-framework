# ML Optimization Framework with Optuna

A professional machine learning hyperparameter optimization framework built with Optuna, demonstrating optimization techniques including multi-objective optimization, pruning strategies, and sampler comparison.

## Overview

This framework provides a complete solution for hyperparameter optimization in machine learning projects:

- 6 optimization studies covering different ML algorithms
- Interactive Optuna dashboard for real-time visualization
- Production-ready modular architecture
- Comprehensive test suite
- French tutorial materials for teaching

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for local development)

### Launch the Project

```bash
# Start the optimization framework
docker-compose up -d --build

# Check status
docker-compose ps

# Access the Optuna Dashboard
# http://localhost:8080
```

## Optimization Studies

The framework includes 6 pre-configured optimization studies:

1. **Random Forest Classifier** - Iris dataset, 100 trials, TPE sampler
2. **Gradient Boosting Regressor** - California Housing dataset, 100 trials
3. **SVM with Pruning** - Digits dataset, 50 trials with MedianPruner
4. **Neural Network (MLP)** - Wine dataset, 75 trials
5. **Multi-Objective Optimization** - Breast Cancer dataset, accuracy vs model size
6. **Sampler Comparison** - Comparing TPE, Random, Grid, and CMA-ES samplers

## Project Structure

```
optimization_with_optuna/
├── src/                          # Source code
│   ├── config.py                 # Configuration management
│   ├── optimizers.py             # ML model optimizers
│   └── study_manager.py          # Optuna study management
├── examples/                     # Example scripts
│   ├── basic_optimization.py
│   ├── advanced/
│   └── custom/
├── tests/                        # Test suite
├── studies/                      # SQLite database
├── tutorial_octobre_2025_french/ # French tutorial materials
├── docker-compose.yml
├── Dockerfile
└── requirements-minimal.txt
```

## French Tutorial Materials

Complete tutorial materials in French are available in `tutorial_octobre_2025_french/`:

- **PRESENTATION_OPTUNA.md** - Introduction to Optuna framework
- **PROJET_PRATIQUE.md** - Hands-on project (house price prediction)
- **projet_prix_maisons.py** - Executable Python project
- **EXERCICES_PRATIQUES.md** - Progressive exercises
- **GUIDE_ENSEIGNANT.md** - Teaching guide
- **GUIDE_GRAPHIQUES.md** - Dashboard visualization guide

These materials are designed for teaching Optuna to colleagues and students.

## Development

### Local Setup

```bash
# Install dependencies
pip install -r requirements-minimal.txt

# Run tests
pytest tests/

# Create optimization studies
python create_unified_demo.py
```

### Docker Commands

```bash
# Build and start
docker-compose up -d --build

# View logs
docker logs ml-optimization-framework

# Stop services
docker-compose down

# Restart
docker-compose restart
```

## Dashboard Features

The Optuna dashboard provides visualization tools:

- Optimization History
- Parameter Importance
- Parallel Coordinate Plot
- Contour Plot
- Slice Plot
- EDF Plot
- Timeline
- Pareto Front (multi-objective)

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_optimizers.py

# Run with coverage
pytest --cov=src tests/
```

## Technologies

- **Optuna** - Hyperparameter optimization framework
- **Scikit-learn** - Machine learning algorithms
- **Docker** - Containerization
- **SQLite** - Study persistence
- **Pytest** - Testing framework

## License

MIT License - See LICENSE file for details

## Resources

- Optuna Documentation: https://optuna.readthedocs.io
- GitHub Repository: https://github.com/optuna/optuna
- French Tutorial: `tutorial_octobre_2025_french/`

