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
    demo_files = ["demo_2d.db", "demo_ml.db", "demo_multi.db"]
    
    # Check if demo data exists
    existing_files = [f for f in demo_files if (studies_dir / f).exists()]
    
    if len(existing_files) < len(demo_files):
        print("Creating demo optimization studies...")
        try:
            subprocess.run([sys.executable, "quick_demo.py"], check=True)
            print("Demo studies created successfully!")
        except subprocess.CalledProcessError:
            print("Warning: Could not create demo studies")

def start_streamlit():
    """Start Streamlit app"""
    print("Starting Streamlit application...")
    print("Access at: http://localhost:8501")
    
    try:
        subprocess.run([
            "streamlit", "run", "simple_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("Shutting down...")

def main():
    """Main startup function"""
    print("ML Optimization Framework - Docker Startup")
    print("=" * 50)
    
    # Create demo data
    create_demo_data()
    
    # Start Streamlit
    start_streamlit()

if __name__ == "__main__":
    main()
