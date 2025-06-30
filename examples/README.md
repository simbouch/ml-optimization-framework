# Examples Directory

This directory contains comprehensive examples demonstrating various aspects of the ML Optimization Framework.

## 📁 Directory Structure

```
examples/
├── basic_optimization.py          # Basic usage examples
├── advanced/                      # Advanced optimization techniques
│   ├── multi_objective_optimization.py
│   └── distributed_optimization.py
├── custom/                        # Custom optimizer tutorials
│   ├── custom_optimizer_tutorial.py
│   └── custom_optimizer_template.py
└── README.md                      # This file
```

## 🚀 Getting Started

### Basic Examples
Start with the basic optimization example to understand the framework fundamentals:

```bash
python examples/basic_optimization.py
```

This demonstrates:
- Simple classification and regression optimization
- Algorithm comparison (Random Forest, Gradient Boosting, SVM)
- Study management and result analysis

### Advanced Examples
Explore advanced optimization techniques:

```bash
# Multi-objective optimization
python examples/advanced/multi_objective_optimization.py

# Distributed optimization
python examples/advanced/distributed_optimization.py
```

### Custom Optimizers
Learn to build your own optimizers:

```bash
# Complete tutorial
python examples/custom/custom_optimizer_tutorial.py

# Use the template for your own optimizer
cp examples/custom/custom_optimizer_template.py my_optimizer.py
```

## 📊 Example Categories

### 1. Basic Optimization (`basic_optimization.py`)
- **Classification Example**: Random Forest hyperparameter tuning
- **Regression Example**: Gradient Boosting optimization for regression
- **Algorithm Comparison**: Compare multiple ML algorithms
- **Study Management**: Create, load, and analyze studies

**Key Features Demonstrated:**
- OptimizationConfig usage
- ModelOptimizer implementations
- StudyManager functionality
- Cross-validation and evaluation

### 2. Advanced Examples (`advanced/`)

#### Multi-Objective Optimization
- **ML Multi-Objective**: Optimize accuracy vs model complexity
- **Portfolio Optimization**: Financial portfolio with multiple objectives
- **Performance vs Cost**: Trade-off between accuracy and computational cost

**Key Features:**
- Pareto front analysis
- Multiple conflicting objectives
- Real-world optimization scenarios

#### Distributed Optimization
- **Multi-Process Optimization**: Parallel optimization across CPU cores
- **Database Backends**: SQLite, PostgreSQL, MySQL, Redis
- **Load Balancing**: Different strategies for worker coordination

**Key Features:**
- Concurrent optimization
- Shared study storage
- Performance comparison with sequential optimization

### 3. Custom Optimizers (`custom/`)

#### Custom Optimizer Tutorial
- **Neural Network Optimizer**: Custom MLP optimization
- **Ensemble Optimizer**: Voting, bagging, and stacking ensembles
- **Time Series Optimizer**: Specialized for temporal data

**Key Features:**
- Inherit from ModelOptimizer base class
- Custom search space definition
- Specialized evaluation metrics

#### Custom Optimizer Template
- **Ready-to-use template** for building new optimizers
- **Step-by-step guide** for implementation
- **Best practices** and common patterns

## 🎯 Usage Patterns

### Running Individual Examples
```bash
# Basic examples
python examples/basic_optimization.py

# Advanced examples
python examples/advanced/multi_objective_optimization.py
python examples/advanced/distributed_optimization.py

# Custom optimizer examples
python examples/custom/custom_optimizer_tutorial.py
```

### Viewing Results
After running examples, view results in:
- **Optuna Dashboard**: http://localhost:8080

### Study Databases
Examples create study databases in the `studies/` directory:
- `unified_demo.db` (main demonstration database)
- Additional databases created by individual examples

## 💡 Learning Path

### Beginner
1. Start with `basic_optimization.py`
2. Understand the framework architecture
3. Explore different ML algorithms

### Intermediate
1. Try `multi_objective_optimization.py`
2. Learn about Pareto optimization
3. Understand trade-offs in ML

### Advanced
1. Explore `distributed_optimization.py`
2. Build custom optimizers with the tutorial
3. Implement domain-specific optimizations

## 🔧 Customization

### Modifying Examples
All examples are designed to be easily modified:

1. **Change datasets**: Replace with your own data
2. **Adjust parameters**: Modify search spaces and trial counts
3. **Add metrics**: Include custom evaluation criteria
4. **Extend algorithms**: Add new ML models

### Creating New Examples
Use this template structure:

```python
#!/usr/bin/env python3
"""
Your Example Title
Description of what this example demonstrates
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Your imports here
from src.config import OptimizationConfig
from src.optimizers import ModelOptimizer

def your_example_function():
    """Your example implementation"""
    pass

def main():
    """Main function"""
    print("🎯 Your Example Title")
    print("=" * 50)
    
    try:
        your_example_function()
        print("✅ Example completed successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
```

## 📚 Additional Resources

- **[Framework Documentation](../docs/)**: Complete documentation
- **[Setup Guide](../docs/setup.md)**: Setup instructions
- **[Usage Guide](../docs/usage.md)**: Dashboard usage
- **[Tutorial](../docs/tutorial.md)**: Complete Optuna tutorial
- **[API Reference](../docs/api.md)**: Detailed API documentation

## 🤝 Contributing

To contribute new examples:

1. Follow the existing code structure
2. Include comprehensive documentation
3. Add error handling and logging
4. Test with different datasets
5. Update this README with your example

## 📄 License

All examples are provided under the same license as the main project (MIT License).
