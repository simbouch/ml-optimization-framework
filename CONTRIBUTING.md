# Contributing to ML Optimization Framework

Thank you for your interest in contributing to the ML Optimization Framework! This document provides guidelines and information for contributors.

## 🎯 How to Contribute

We welcome contributions in many forms:
- 🐛 **Bug reports and fixes**
- ✨ **New features and enhancements**
- 📚 **Documentation improvements**
- 🧪 **Tests and test coverage**
- 🎨 **Code quality improvements**
- 💡 **Ideas and suggestions**

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/ml-optimization-framework.git
cd ml-optimization-framework

# Add the original repository as upstream
git remote add upstream https://github.com/original-username/ml-optimization-framework.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Setup

```bash
# Run tests to ensure everything works
python -m pytest tests/

# Run basic validation
python tests/test_framework.py

# Check code quality
black --check src/
flake8 src/
mypy src/
```

## 📋 Development Workflow

### 1. Create a Branch

```bash
# Create a new branch for your feature/fix
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 2. Make Changes

- Write clean, readable code following our style guidelines
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add multi-objective optimization visualization

- Add Pareto front plotting functionality
- Include trade-off analysis methods
- Update documentation with examples
- Add comprehensive tests

Closes #123"
```

### 4. Push and Create Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Create a pull request on GitHub
```

## 📝 Code Style Guidelines

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Use Black for formatting (line length: 88)
black src/ tests/

# Use type hints for all public functions
def optimize_model(
    self,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 100
) -> optuna.Study:
    """
    Optimize model hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_trials: Number of optimization trials
        
    Returns:
        Completed Optuna study
    """
    pass

# Use descriptive variable names
hyperparameter_space = config.get_hyperparameter_space("random_forest")
optimization_study = create_study(direction="maximize")

# Add docstrings to all classes and public methods
class ModelOptimizer:
    """
    Base class for model optimization.
    
    This class provides common functionality for hyperparameter
    optimization across different machine learning models.
    """
    pass
```

### Documentation Style

```python
# Use Google-style docstrings
def suggest_hyperparameters(
    self,
    trial: optuna.Trial,
    model_name: str
) -> Dict[str, Any]:
    """
    Suggest hyperparameters for a trial.
    
    Args:
        trial: Optuna trial object for parameter suggestion
        model_name: Name of the model to optimize
        
    Returns:
        Dictionary containing suggested hyperparameters
        
    Raises:
        ValueError: If model_name is not supported
        
    Example:
        >>> config = OptimizationConfig()
        >>> trial = study.ask()
        >>> params = config.suggest_hyperparameters(trial, "random_forest")
        >>> print(params["n_estimators"])
        200
    """
    pass
```

## 🧪 Testing Guidelines

### Writing Tests

```python
import pytest
import numpy as np
from src.models.random_forest_optimizer import RandomForestOptimizer

class TestRandomForestOptimizer:
    """Test suite for Random Forest optimizer."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.optimizer = RandomForestOptimizer(random_state=42)
        # Create small test dataset
        self.X_train = np.random.rand(100, 5)
        self.y_train = np.random.randint(0, 2, 100)
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization with default parameters."""
        assert self.optimizer.random_state == 42
        assert self.optimizer.cv_folds == 5
        assert self.optimizer.scoring_metric == "accuracy"
    
    def test_optimization_runs_successfully(self):
        """Test that optimization completes without errors."""
        study = self.optimizer.optimize(
            self.X_train, self.X_train, self.y_train, self.y_train,
            n_trials=3
        )
        
        assert study.best_value is not None
        assert study.best_params is not None
        assert len(study.trials) == 3
    
    @pytest.mark.parametrize("n_trials", [1, 5, 10])
    def test_optimization_with_different_trial_counts(self, n_trials):
        """Test optimization with different numbers of trials."""
        study = self.optimizer.optimize(
            self.X_train, self.X_train, self.y_train, self.y_train,
            n_trials=n_trials
        )
        assert len(study.trials) == n_trials
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run specific test method
pytest tests/test_models.py::TestRandomForestOptimizer::test_optimization_runs_successfully

# Run tests with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

## 📚 Documentation Guidelines

### Code Documentation

- Add docstrings to all public classes, methods, and functions
- Use type hints for all function parameters and return values
- Include examples in docstrings where helpful
- Document complex algorithms and design decisions

### README and Guides

- Keep README.md up to date with new features
- Add examples for new functionality
- Update installation instructions if dependencies change
- Include troubleshooting information for common issues

### API Documentation

- Document all public APIs in `docs/api_reference.md`
- Include parameter descriptions and examples
- Document error conditions and exceptions
- Keep documentation in sync with code changes

## 🐛 Bug Reports

When reporting bugs, please include:

### Bug Report Template

```markdown
## Bug Description
A clear and concise description of the bug.

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
A clear description of what you expected to happen.

## Actual Behavior
A clear description of what actually happened.

## Environment
- OS: [e.g. Ubuntu 20.04, Windows 10, macOS 12.0]
- Python version: [e.g. 3.9.7]
- Framework version: [e.g. 1.0.0]
- Dependencies: [paste output of `pip freeze`]

## Additional Context
Add any other context about the problem here.

## Possible Solution
If you have ideas about what might be causing the issue.
```

## ✨ Feature Requests

### Feature Request Template

```markdown
## Feature Description
A clear and concise description of the feature you'd like to see.

## Use Case
Describe the problem this feature would solve or the workflow it would improve.

## Proposed Solution
Describe how you envision this feature working.

## Alternatives Considered
Describe any alternative solutions or features you've considered.

## Additional Context
Add any other context, mockups, or examples about the feature request.

## Implementation Notes
If you have ideas about how this could be implemented.
```

## 🔄 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (for significant changes)
- [ ] Pre-commit hooks pass

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced

## Related Issues
Closes #(issue number)
```

### Review Process

1. **Automated Checks**: All CI checks must pass
2. **Code Review**: At least one maintainer review required
3. **Testing**: Comprehensive test coverage for new features
4. **Documentation**: Updated documentation for user-facing changes

## 🏗️ Development Setup Details

### Project Structure

```
ml-optimization-framework/
├── src/                    # Main source code
│   ├── data/              # Data pipeline modules
│   ├── models/            # Model optimizer implementations
│   ├── optimization/      # Optuna configuration and management
│   └── visualization/     # Plotting and analysis tools
├── tests/                 # Test suite
├── docs/                  # Documentation
├── config/                # Configuration files
├── scripts/               # CLI and utility scripts
├── notebooks/             # Jupyter notebooks
└── requirements.txt       # Dependencies
```

### Code Quality Tools

```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/

# Security scanning
bandit -r src/

# Import sorting
isort src/ tests/

# Documentation generation
sphinx-build -b html docs/ docs/_build/
```

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Pull Requests**: Code contributions and reviews
- **Email**: team@mloptimization.com for private matters

## 🎖️ Recognition

Contributors will be recognized in:
- README.md contributors section
- CHANGELOG.md for significant contributions
- GitHub contributors page
- Release notes for major contributions

## 📞 Getting Help

If you need help with contributing:

1. **Check Documentation**: README.md, docs/, and this guide
2. **Search Issues**: Look for similar questions or problems
3. **Ask Questions**: Create a GitHub Discussion
4. **Join Community**: Participate in discussions and reviews

Thank you for contributing to the ML Optimization Framework! 🚀
