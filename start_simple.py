#!/usr/bin/env python3
"""
Simple start script for ML Optimization Framework
"""

import subprocess
import sys
import time
import os
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

def start_streamlit():
    """Start the Streamlit app"""
    print("\n🚀 Starting Streamlit app...")
    print("📍 URL: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "simple_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Stopping application...")

def main():
    print("🎯 ML Optimization Framework - Simple Setup")
    print("=" * 50)
    
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
    
    # Start application
    print("\n3️⃣ Starting application...")
    start_streamlit()

if __name__ == "__main__":
    main()
