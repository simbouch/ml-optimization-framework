#!/usr/bin/env python3
"""
Final Project Validation Script

This script validates that all components of the ML Optimization Framework
are working correctly before deployment.
"""

import sys
import os
import subprocess
import sqlite3
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_validation() -> bool:
    """Run comprehensive validation of the project."""
    
    print("🎯 ML Optimization Framework - Final Validation")
    print("=" * 60)
    
    validations = [
        ("📦 Dependencies", validate_dependencies),
        ("🏗️ Project Structure", validate_project_structure),
        ("🧪 Core Framework", validate_core_framework),
        ("📊 Data Pipeline", validate_data_pipeline),
        ("🤖 ML Models", validate_ml_models),
        ("🎛️ Dashboard Components", validate_dashboard),
        ("🐳 Docker Configuration", validate_docker),
        ("📚 Documentation", validate_documentation),
        ("🔧 Scripts", validate_scripts),
    ]
    
    passed = 0
    total = len(validations)
    
    for name, validation_func in validations:
        print(f"\n{name}:")
        try:
            if validation_func():
                print(f"  ✅ {name} - PASSED")
                passed += 1
            else:
                print(f"  ❌ {name} - FAILED")
        except Exception as e:
            print(f"  ❌ {name} - ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Validation Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 ALL VALIDATIONS PASSED! Project is ready for deployment.")
        return True
    else:
        print("⚠️  Some validations failed. Please check the issues above.")
        return False

def validate_dependencies() -> bool:
    """Validate that all required dependencies are available."""
    
    required_packages = [
        'optuna', 'optuna_dashboard', 'streamlit', 'pandas', 'numpy',
        'sklearn', 'xgboost', 'lightgbm', 'matplotlib', 'plotly',
        'seaborn', 'loguru', 'yaml', 'openml'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"    Missing packages: {', '.join(missing)}")
        return False
    
    print("    All required packages available")
    return True

def validate_project_structure() -> bool:
    """Validate project directory structure."""
    
    required_dirs = [
        'src', 'src/data', 'src/models', 'src/optimization', 'src/utils',
        'tests', 'scripts', 'docs', 'config'
    ]
    
    required_files = [
        'README.md', 'requirements.txt', 'Dockerfile', 'docker-compose.yml',
        '.env.example', 'streamlit_app.py', 'setup.py'
    ]
    
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_dirs or missing_files:
        if missing_dirs:
            print(f"    Missing directories: {', '.join(missing_dirs)}")
        if missing_files:
            print(f"    Missing files: {', '.join(missing_files)}")
        return False
    
    print("    Project structure is complete")
    return True

def validate_core_framework() -> bool:
    """Validate core framework components."""
    
    try:
        from src.data.data_pipeline import DataPipeline
        from src.models.random_forest_optimizer import RandomForestOptimizer
        from src.optimization.study_manager import StudyManager
        from src.optimization.config import OptimizationConfig
        
        # Test basic instantiation
        config = OptimizationConfig()
        data_pipeline = DataPipeline()
        study_manager = StudyManager(config=config)
        
        print("    Core framework components loaded successfully")
        return True
        
    except Exception as e:
        print(f"    Error loading core components: {e}")
        return False

def validate_data_pipeline() -> bool:
    """Validate data pipeline functionality."""
    
    try:
        from src.data.data_pipeline import DataPipeline
        
        pipeline = DataPipeline()
        pipeline.prepare_data()  # This loads and splits the data

        X_train, X_val, y_train, y_val = pipeline.get_train_val_data()
        X_test, y_test = pipeline.get_test_data()

        if X_train is None or len(X_train) == 0:
            print("    Data pipeline returned empty data")
            return False

        print(f"    Data pipeline working - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return True
        
    except Exception as e:
        print(f"    Data pipeline error: {e}")
        return False

def validate_ml_models() -> bool:
    """Validate ML model optimizers."""
    
    try:
        from src.models.random_forest_optimizer import RandomForestOptimizer
        from src.models.xgboost_optimizer import XGBoostOptimizer
        from src.models.lightgbm_optimizer import LightGBMOptimizer
        from src.optimization.config import OptimizationConfig
        
        config = OptimizationConfig()
        
        # Test model instantiation
        rf_optimizer = RandomForestOptimizer(config)
        xgb_optimizer = XGBoostOptimizer(config)
        lgb_optimizer = LightGBMOptimizer(config)
        
        print("    All ML model optimizers loaded successfully")
        return True
        
    except Exception as e:
        print(f"    ML models error: {e}")
        return False

def validate_dashboard() -> bool:
    """Validate dashboard components."""
    
    # Check if Streamlit app exists and is valid
    if not os.path.exists('streamlit_app.py'):
        print("    Streamlit app not found")
        return False
    
    # Check if dashboard scripts exist
    dashboard_scripts = [
        'scripts/start_dashboard.py',
        'scripts/populate_dashboard.py',
        'scripts/start_streamlit.py'
    ]
    
    missing_scripts = [s for s in dashboard_scripts if not os.path.exists(s)]
    if missing_scripts:
        print(f"    Missing dashboard scripts: {', '.join(missing_scripts)}")
        return False
    
    print("    Dashboard components available")
    return True

def validate_docker() -> bool:
    """Validate Docker configuration."""
    
    if not os.path.exists('Dockerfile'):
        print("    Dockerfile not found")
        return False
    
    if not os.path.exists('docker-compose.yml'):
        print("    docker-compose.yml not found")
        return False
    
    # Check if .env files exist
    if not os.path.exists('.env.example'):
        print("    .env.example not found")
        return False
    
    print("    Docker configuration files present")
    return True

def validate_documentation() -> bool:
    """Validate documentation files."""
    
    doc_files = [
        'README.md',
        'docs/COMPLETE_OPTUNA_TUTORIAL.md',
        'GITHUB_RELEASE_INFO.md'
    ]
    
    missing_docs = [d for d in doc_files if not os.path.exists(d)]
    if missing_docs:
        print(f"    Missing documentation: {', '.join(missing_docs)}")
        return False
    
    print("    Documentation files present")
    return True

def validate_scripts() -> bool:
    """Validate utility scripts."""
    
    required_scripts = [
        'scripts/deploy_complete_demo.py',
        'scripts/populate_dashboard.py',
        'scripts/start_dashboard.py',
        'scripts/start_streamlit.py',
        'scripts/showcase_all_optuna_features.py',
        'scripts/validate_framework.py'
    ]
    
    missing_scripts = [s for s in required_scripts if not os.path.exists(s)]
    if missing_scripts:
        print(f"    Missing scripts: {', '.join(missing_scripts)}")
        return False
    
    print("    All utility scripts present")
    return True

def main():
    """Main validation function."""
    
    success = run_validation()
    
    if success:
        print("\n🎉 PROJECT VALIDATION COMPLETE!")
        print("🚀 Ready for deployment and GitHub push!")
        print("\n📋 Next Steps:")
        print("  1. 🎛️ Test dashboard: python scripts/deploy_complete_demo.py")
        print("  2. 🎯 Test Streamlit: streamlit run streamlit_app.py")
        print("  3. 🐳 Test Docker: docker-compose up")
        print("  4. 📤 Push to GitHub with the info in GITHUB_RELEASE_INFO.md")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED!")
        print("Please fix the issues above before deployment.")
        sys.exit(1)

if __name__ == "__main__":
    main()
