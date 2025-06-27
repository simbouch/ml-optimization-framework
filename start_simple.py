#!/usr/bin/env python3
"""
Enhanced start script for ML Optimization Framework
Launches both Streamlit app and Optuna dashboard
"""

import subprocess
import sys
import time
import os
import threading
import webbrowser
from pathlib import Path

def check_requirements():
    """Check if minimal requirements are installed"""
    required_packages = [
        'streamlit',
        'optuna',
        'pandas',
        'plotly'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")
    
    return missing

def install_requirements():
    """Install minimal requirements"""
    print("Installing minimal requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements-minimal.txt"
        ])
        return True
    except subprocess.CalledProcessError:
        return False

def create_directories():
    """Create necessary directories"""
    dirs = ['studies', 'logs']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created {dir_name}/ directory")

def find_study_databases():
    """Find available study databases"""
    studies_dir = Path("studies")
    if not studies_dir.exists():
        return []

    db_files = list(studies_dir.glob("*.db"))
    return [f"sqlite:///{f}" for f in db_files]

def start_optuna_dashboard():
    """Start Optuna dashboard in background"""
    print("🔧 Starting Optuna dashboard...")

    # Find study databases
    db_urls = find_study_databases()

    if not db_urls:
        print("⚠️  No study databases found. Creating demo study...")
        # Create a quick demo study
        try:
            subprocess.run([sys.executable, "quick_demo.py"],
                         capture_output=True, timeout=30)
            db_urls = find_study_databases()
        except:
            print("❌ Could not create demo studies")
            return None

    if db_urls:
        print(f"📊 Found {len(db_urls)} study database(s)")
        cmd = ["optuna-dashboard"] + db_urls + [
            "--host", "0.0.0.0",
            "--port", "8080"
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(2)  # Give it time to start
            print("✅ Optuna dashboard started at http://localhost:8080")
            return process
        except Exception as e:
            print(f"❌ Failed to start Optuna dashboard: {e}")
            return None
    else:
        print("❌ No study databases available")
        return None

def start_streamlit():
    """Start the Streamlit app"""
    print("🚀 Starting Streamlit app...")
    print("📍 Streamlit URL: http://localhost:8501")
    print("📍 Optuna Dashboard URL: http://localhost:8080")
    print("🛑 Press Ctrl+C to stop both services")

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "simple_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Stopping applications...")

def main():
    print("🎯 ML Optimization Framework - Enhanced Setup")
    print("=" * 60)

    # Check requirements
    print("\n1️⃣ Checking requirements...")
    missing = check_requirements()

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Installing missing packages...")
        if not install_requirements():
            print("❌ Failed to install requirements")
            return
        print("✅ Requirements installed")
    else:
        print("✅ All requirements satisfied")

    # Create directories
    print("\n2️⃣ Setting up directories...")
    create_directories()

    # Start Optuna dashboard first
    print("\n3️⃣ Starting Optuna dashboard...")
    dashboard_process = start_optuna_dashboard()

    # Start Streamlit app
    print("\n4️⃣ Starting Streamlit application...")
    try:
        start_streamlit()
    finally:
        # Clean up dashboard process
        if dashboard_process:
            print("\n🧹 Cleaning up Optuna dashboard...")
            dashboard_process.terminate()
            try:
                dashboard_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dashboard_process.kill()

if __name__ == "__main__":
    main()
