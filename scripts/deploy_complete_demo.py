#!/usr/bin/env python3
"""
Complete Demo Deployment Script

This script sets up and runs the complete Optuna ML Optimization Framework demo:
1. Validates the framework
2. Populates the dashboard with comprehensive studies
3. Starts the Optuna dashboard
4. Provides instructions for exploring all features

This is the one-click solution to see everything working.
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def run_command(command, description, timeout=60):
    """Run a command and return success status."""
    logger.info(f"🔄 {description}...")
    try:
        if sys.platform == "win32":
            # Windows
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=Path(__file__).parent.parent
            )
        else:
            # Unix/Linux
            result = subprocess.run(
                command.split(), 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=Path(__file__).parent.parent
            )
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully")
            return True
        else:
            logger.error(f"❌ {description} failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} timed out after {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"❌ {description} error: {e}")
        return False

def check_dependencies():
    """Check if all dependencies are installed."""
    logger.info("🔍 Checking dependencies...")
    
    try:
        import optuna
        import optuna_integration
        import optuna_dashboard
        import sklearn
        import numpy
        import pandas
        logger.info("✅ All dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("💡 Run: pip install -r requirements.txt")
        return False

def populate_dashboard():
    """Populate the dashboard with demo studies."""
    logger.info("📊 Populating dashboard with demo studies...")
    
    try:
        # Import and run the population script
        sys.path.insert(0, str(Path(__file__).parent))
        from populate_dashboard import create_database_studies
        
        storage_url = create_database_studies()
        logger.info(f"✅ Dashboard populated: {storage_url}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to populate dashboard: {e}")
        return False

def start_dashboard_server():
    """Start the Optuna dashboard server."""
    logger.info("🌐 Starting Optuna dashboard server...")
    
    try:
        import optuna_dashboard
        
        # Database path
        db_path = "sqlite:///studies/optuna_dashboard_demo.db"
        
        logger.info("🚀 Dashboard starting...")
        logger.info("🌐 URL: http://localhost:8080")
        logger.info("🛑 Press Ctrl+C to stop")
        
        # Start dashboard (this will block)
        optuna_dashboard.run_server(
            storage=db_path,
            host="0.0.0.0",
            port=8080
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Dashboard stopped by user")
        return True
    except Exception as e:
        logger.error(f"❌ Dashboard failed to start: {e}")
        return False

def main():
    """Main deployment function."""
    logger.info("🚀 Complete Optuna ML Optimization Framework Demo")
    logger.info("=" * 70)
    logger.info("This will set up and run the complete demo with dashboard!")
    logger.info("=" * 70)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependencies check failed. Please install requirements first.")
        return False
    
    # Step 2: Create directories
    logger.info("📁 Creating necessary directories...")
    os.makedirs("studies", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logger.info("✅ Directories created")
    
    # Step 3: Populate dashboard
    if not populate_dashboard():
        logger.error("❌ Dashboard population failed")
        return False
    
    # Step 4: Show what's available
    logger.info("\n" + "=" * 70)
    logger.info("🎯 DEMO READY! Here's what you can explore:")
    logger.info("=" * 70)
    logger.info("📊 Dashboard Studies Created:")
    logger.info("   1. Single-objective optimization (20 trials)")
    logger.info("   2. Multi-objective optimization (25 trials, Pareto front)")
    logger.info("   3. Optimization with pruning (30 trials, 19 pruned)")
    logger.info("   4. TPE Sampler comparison (15 trials)")
    logger.info("   5. Random Sampler comparison (15 trials)")
    logger.info("   6. Optimization with failures (20 trials, 3 failed)")
    
    logger.info("\n🎛️ Dashboard Features to Explore:")
    logger.info("   ✅ Optimization history and convergence plots")
    logger.info("   ✅ Parameter importance analysis")
    logger.info("   ✅ Multi-objective Pareto front visualization")
    logger.info("   ✅ Trial filtering and comparison tools")
    logger.info("   ✅ Parameter relationship plots")
    logger.info("   ✅ Study management and statistics")
    
    logger.info("\n🚀 Additional Scripts Available:")
    logger.info("   📈 python scripts/showcase_all_optuna_features.py")
    logger.info("   🔍 python scripts/final_validation.py")
    logger.info("   🎯 python scripts/validate_framework.py")
    
    # Step 5: Start dashboard
    logger.info("\n" + "=" * 70)
    logger.info("🌐 Starting Optuna Dashboard...")
    logger.info("=" * 70)
    
    # Give user a moment to read
    time.sleep(3)
    
    # Try to open browser automatically
    try:
        logger.info("🌐 Opening browser automatically...")
        webbrowser.open("http://localhost:8080")
    except:
        logger.info("💡 Please open http://localhost:8080 in your browser")
    
    # Start the dashboard server (this will block)
    start_dashboard_server()

if __name__ == "__main__":
    main()
