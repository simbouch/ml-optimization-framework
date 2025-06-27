#!/usr/bin/env python3
"""
Production Docker startup script for ML Optimization Framework
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def create_demo_data():
    """Create demo data if it doesn't exist"""
    studies_dir = Path("studies")
    studies_dir.mkdir(exist_ok=True)

    # Check if any database files exist
    existing_files = list(studies_dir.glob("*.db"))

    if len(existing_files) == 0:
        print("No study databases found. Creating comprehensive demo studies...")
        try:
            # Try to run comprehensive demo first
            result = subprocess.run([sys.executable, "comprehensive_demo_safe.py"],
                                  capture_output=True, timeout=180)
            if result.returncode == 0:
                print("Comprehensive demo studies created successfully!")
                return
            else:
                print("Comprehensive demo failed, trying quick demo...")
                result = subprocess.run([sys.executable, "quick_demo.py"],
                                      capture_output=True, timeout=60)
                if result.returncode == 0:
                    print("Quick demo studies created successfully!")
                    return
                else:
                    print("Quick demo failed, creating minimal fallback...")
                    create_minimal_study()
        except Exception as e:
            print(f"Could not run demos: {e}")
            print("Creating minimal fallback study...")
            create_minimal_study()
    else:
        print(f"Found {len(existing_files)} existing study database(s)")

def create_minimal_study():
    """Create a minimal study as fallback"""
    try:
        import optuna
        print("Creating minimal demo study...")
        study = optuna.create_study(storage="sqlite:///studies/minimal_demo.db")

        def simple_objective(trial):
            x = trial.suggest_float("x", -10, 10)
            y = trial.suggest_float("y", -10, 10)
            return x**2 + y**2

        study.optimize(simple_objective, n_trials=10)
        print("Minimal demo study created successfully!")
    except Exception as e:
        print(f"Failed to create minimal study: {e}")

def start_streamlit():
    """Start Streamlit app"""
    print("Starting Streamlit application...")
    print("Access at: http://localhost:8501")

    try:
        subprocess.run([
            "streamlit", "run", "simple_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--server.enableCORS", "false"
        ])
    except KeyboardInterrupt:
        print("Shutting down...")

def start_optuna_dashboard():
    """Start Optuna dashboard"""
    print("Starting Optuna dashboard...")
    print("Access at: http://localhost:8080")

    # Wait for studies to be ready
    time.sleep(2)

    # Find available database files
    studies_dir = Path("studies")
    db_files = list(studies_dir.glob("*.db"))

    if not db_files:
        print("Warning: No database files found, creating minimal study...")
        # Create a minimal study as fallback
        import optuna
        study = optuna.create_study(storage="sqlite:///studies/fallback.db")
        study.optimize(lambda trial: trial.suggest_float("x", -10, 10) ** 2, n_trials=5)
        db_files = [Path("studies/fallback.db")]

    # Use the first database file (optuna-dashboard takes one storage URL)
    db_url = f"sqlite:///{db_files[0]}"

    try:
        cmd = [
            "optuna-dashboard", db_url,
            "--host", "0.0.0.0",
            "--port", "8080"
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("Shutting down...")

def main():
    """Main startup function"""
    print("ML Optimization Framework - Docker Startup")
    print("=" * 50)

    # Create demo data
    create_demo_data()

    # Determine which service to start
    service = os.environ.get("SERVICE_TYPE", "streamlit")
    if len(sys.argv) > 1:
        service = sys.argv[1]

    if service == "dashboard":
        start_optuna_dashboard()
    else:
        start_streamlit()

if __name__ == "__main__":
    main()
