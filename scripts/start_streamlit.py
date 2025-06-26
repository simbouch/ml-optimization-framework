#!/usr/bin/env python3
"""
Start Streamlit Dashboard for ML Optimization Framework

This script starts the Streamlit interface that provides:
- Easy access to Optuna dashboard
- Project documentation
- Optimization tools
- Analytics and monitoring
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Start the Streamlit dashboard."""
    
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Change to project directory
    os.chdir(project_root)
    
    print("🎯 Starting ML Optimization Framework - Streamlit Interface")
    print("=" * 60)
    print("🌐 Dashboard will be available at: http://localhost:8501")
    print("📊 Features:")
    print("  - Interactive project interface")
    print("  - Optuna dashboard launcher")
    print("  - Documentation viewer")
    print("  - Optimization tools")
    print("  - Analytics and monitoring")
    print("=" * 60)
    print()
    
    try:
        # Start Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.address", "0.0.0.0",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light"
        ]

        print("🚀 Starting Streamlit...")
        print("🌐 Access the dashboard at: http://localhost:8501")
        print("📊 From there you can:")
        print("  - Launch Optuna dashboard")
        print("  - View live analytics")
        print("  - Run optimization demos")
        print("  - Monitor progress")
        print()
        subprocess.run(cmd)

    except KeyboardInterrupt:
        print("\n⏹️  Streamlit dashboard stopped by user")
    except FileNotFoundError:
        print("❌ Error: Streamlit not found. Please install it:")
        print("   pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
