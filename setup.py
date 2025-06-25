"""
Setup script for ML Optimization Framework with Optuna.

This package provides a comprehensive, production-ready framework for
hyperparameter optimization using Optuna with advanced features.
"""

from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
def get_version():
    with open(os.path.join("src", "__init__.py"), "r") as f:
        content = f.read()
        match = re.search(r'__version__ = ["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)
    raise RuntimeError("Unable to find version string.")

# Read long description from README
def get_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# Read requirements
def get_requirements():
    with open("requirements.txt", "r") as f:
        lines = f.readlines()
    
    requirements = []
    for line in lines:
        line = line.strip()
        # Skip comments and empty lines
        if line and not line.startswith("#"):
            requirements.append(line)
    
    return requirements

# Package metadata
PACKAGE_NAME = "ml-optimization-framework"
DESCRIPTION = "Professional ML Optimization Framework with Optuna"
AUTHOR = "ML Optimization Team"
AUTHOR_EMAIL = "team@mloptimization.com"
URL = "https://github.com/your-username/ml-optimization-framework"
LICENSE = "MIT"

# Classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: System :: Distributed Computing",
]

# Keywords for discoverability
KEYWORDS = [
    "machine learning",
    "hyperparameter optimization",
    "optuna",
    "automl",
    "model selection",
    "cross-validation",
    "bayesian optimization",
    "random forest",
    "xgboost",
    "lightgbm",
    "data science",
    "mlops"
]

# Entry points for CLI
ENTRY_POINTS = {
    "console_scripts": [
        "ml-optimize=scripts.cli_runner:main",
        "ml-optimize-run=scripts.run_optimization:main",
    ],
}

# Extra requirements for optional features
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "pre-commit>=2.20.0",
    ],
    "docs": [
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "myst-parser>=0.18.0",
        "sphinx-autodoc-typehints>=1.19.0",
    ],
    "jupyter": [
        "jupyter>=1.0.0",
        "notebook>=6.5.0",
        "ipywidgets>=8.0.0",
        "nbconvert>=7.0.0",
    ],
    "database": [
        "psycopg2-binary>=2.9.0",
        "sqlalchemy>=1.4.0",
        "alembic>=1.8.0",
    ],
    "distributed": [
        "redis>=4.0.0",
        "celery>=5.2.0",
        "dask[complete]>=2022.8.0",
    ],
    "monitoring": [
        "mlflow>=2.5.0",
        "wandb>=0.13.0",
        "tensorboard>=2.10.0",
    ],
    "deep-learning": [
        "torch>=1.12.0",
        "tensorflow>=2.9.0",
        "keras-tuner>=1.1.0",
    ],
    "all": [
        # Include all optional dependencies
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "pre-commit>=2.20.0",
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "jupyter>=1.0.0",
        "notebook>=6.5.0",
        "psycopg2-binary>=2.9.0",
        "redis>=4.0.0",
        "mlflow>=2.5.0",
        "torch>=1.12.0",
        "tensorflow>=2.9.0",
    ]
}

# Package data to include
PACKAGE_DATA = {
    "": [
        "*.yaml",
        "*.yml",
        "*.json",
        "*.md",
        "*.txt",
        "*.cfg",
        "*.ini",
    ],
}

# Data files to include
DATA_FILES = [
    ("config", ["config/hyperparameters.yaml", "config/optimization_config.yaml"]),
    ("docs", [
        "docs/api_reference.md",
        "docs/tutorial.md",
        "docs/dashboard_guide.md",
        "docs/optimization_report.md"
    ]),
]

setup(
    name=PACKAGE_NAME,
    version=get_version(),
    description=DESCRIPTION,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    
    # Package discovery
    packages=find_packages(include=["src", "src.*"]),
    package_dir={"": "."},
    
    # Dependencies
    install_requires=get_requirements(),
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.8",
    
    # Package metadata
    classifiers=CLASSIFIERS,
    keywords=", ".join(KEYWORDS),
    
    # Entry points
    entry_points=ENTRY_POINTS,
    
    # Package data
    package_data=PACKAGE_DATA,
    data_files=DATA_FILES,
    include_package_data=True,
    
    # Project URLs
    project_urls={
        "Documentation": f"{URL}/docs",
        "Source": URL,
        "Tracker": f"{URL}/issues",
        "Changelog": f"{URL}/blob/main/CHANGELOG.md",
    },
    
    # Additional metadata
    zip_safe=False,
    platforms=["any"],
    
    # Test suite
    test_suite="tests",
    tests_require=EXTRAS_REQUIRE["dev"],
)

# Post-installation message
print("""
🚀 ML Optimization Framework installed successfully!

Quick Start:
  1. Run basic optimization:
     ml-optimize --model random_forest --n_trials 50

  2. Launch Optuna Dashboard:
     optuna-dashboard sqlite:///optuna_study.db

  3. Try the interactive notebook:
     jupyter notebook notebooks/ml_optimization_demo.ipynb

Documentation: https://github.com/your-username/ml-optimization-framework/docs
Support: https://github.com/your-username/ml-optimization-framework/issues

Happy optimizing! 🎯
""")
