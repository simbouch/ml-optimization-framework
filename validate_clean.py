#!/usr/bin/env python3
"""
Final validation script for clean ML optimization framework
"""

import sys
import subprocess
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("Testing package imports...")
    
    packages = [
        'streamlit',
        'optuna', 
        'pandas',
        'plotly',
        'numpy',
        'sklearn',
        'requests'
    ]
    
    all_good = True
    for package in packages:
        try:
            __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [FAIL] {package} - MISSING")
            all_good = False
    
    return all_good

def test_file_syntax():
    """Test Python file syntax"""
    print("\nTesting file syntax...")
    
    files = [
        'simple_app.py',
        'start_simple.py', 
        'quick_demo.py',
        'docker-start.py'
    ]
    
    all_good = True
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, file_path, 'exec')
            print(f"  ✓ {file_path}")
        except (SyntaxError, FileNotFoundError) as e:
            print(f"  ✗ {file_path} - {e}")
            all_good = False
    
    return all_good

def test_demo_creation():
    """Test demo study creation"""
    print("\nTesting demo creation...")
    
    try:
        result = subprocess.run([
            sys.executable, 'quick_demo.py'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("  ✓ Demo studies created")
            
            # Check database files
            studies_dir = Path("studies")
            expected_files = ["demo_2d.db", "demo_ml.db", "demo_multi.db"]
            
            for db_file in expected_files:
                if (studies_dir / db_file).exists():
                    print(f"    ✓ {db_file}")
                else:
                    print(f"    ✗ {db_file} - Missing")
                    return False
            
            return True
        else:
            print(f"  ✗ Demo creation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ✗ Demo creation error: {e}")
        return False

def test_optuna_functionality():
    """Test basic Optuna functionality"""
    print("\nTesting Optuna functionality...")
    
    try:
        import optuna
        
        # Create test study
        study = optuna.create_study(direction="minimize")
        
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            return x ** 2
        
        study.optimize(objective, n_trials=5)
        
        print(f"  ✓ Optuna study with {len(study.trials)} trials")
        print(f"  ✓ Best value: {study.best_value:.4f}")
        return True
        
    except Exception as e:
        print(f"  ✗ Optuna test failed: {e}")
        return False

def test_docker_files():
    """Test Docker configuration files"""
    print("\nTesting Docker files...")
    
    docker_files = [
        'Dockerfile',
        'docker-compose.yml'
    ]
    
    all_good = True
    for file_path in docker_files:
        if Path(file_path).exists():
            print(f"  [OK] {file_path}")
        else:
            print(f"  [FAIL] {file_path} - Missing")
            all_good = False
    
    return all_good

def main():
    """Run all validation tests"""
    print("ML Optimization Framework - Clean Validation")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("File Syntax", test_file_syntax),
        ("Demo Creation", test_demo_creation),
        ("Optuna Functionality", test_optuna_functionality),
        ("Docker Files", test_docker_files),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  ⚠ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"Validation Results: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Your clean framework is ready!")
        print("\nNext steps:")
        print("  1. Run: python start_simple.py")
        print("  2. Open: http://localhost:8501")
        print("  3. Click 'Launch Dashboard' in sidebar")
        print("  4. Explore your optimization studies!")
        print("\nDocker option:")
        print("  docker-compose -f docker-compose.simple.yml up")
        return True
    else:
        print(f"\n❌ {total - passed} tests failed")
        print("Please check the issues above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
