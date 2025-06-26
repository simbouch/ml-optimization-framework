#!/usr/bin/env python3
"""
Start Optuna Dashboard with proper configuration.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def start_dashboard():
    """Start the Optuna dashboard."""
    
    # Database path
    db_path = "sqlite:///studies/optuna_dashboard_demo.db"
    
    logger.info("🚀 Starting Optuna Dashboard")
    logger.info("=" * 50)
    logger.info(f"📊 Database: {db_path}")
    logger.info("🌐 Host: 0.0.0.0")
    logger.info("🔗 Port: 8080")
    logger.info("=" * 50)
    
    try:
        import optuna_dashboard
        logger.info("✅ optuna-dashboard imported successfully")
        
        logger.info("🔄 Starting dashboard server...")
        logger.info("🌐 Dashboard will be available at: http://localhost:8080")
        logger.info("🛑 Press Ctrl+C to stop the server")
        
        # Start the dashboard
        optuna_dashboard.run_server(
            storage=db_path,
            host="0.0.0.0",
            port=8080
        )
        
    except ImportError as e:
        logger.error(f"❌ Failed to import optuna-dashboard: {e}")
        logger.info("💡 Try installing with: pip install optuna-dashboard")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to start dashboard: {e}")
        return False

def main():
    """Main function."""
    start_dashboard()

if __name__ == "__main__":
    main()
