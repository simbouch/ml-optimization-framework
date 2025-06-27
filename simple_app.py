#!/usr/bin/env python3
"""
Simple Working Streamlit App for ML Optimization Framework
"""

import streamlit as st
import subprocess
import sys
import time
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="ML Optimization Framework",
    page_icon="🎯",
    layout="wide"
)

# Title
st.title("🎯 ML Optimization Framework")
st.markdown("**Simple, Working Dashboard for Optuna Optimization**")

# Sidebar
st.sidebar.title("🔧 Controls")

# Check if Optuna dashboard is running
def check_dashboard_status():
    try:
        import requests
        response = requests.get("http://localhost:8080", timeout=2)
        return response.status_code == 200
    except:
        return False

# Launch Optuna dashboard
def launch_dashboard():
    try:
        # Create a simple study database if it doesn't exist
        db_path = "studies/simple_demo.db"
        os.makedirs("studies", exist_ok=True)
        
        # Launch dashboard
        cmd = f"optuna-dashboard sqlite:///{db_path} --host 0.0.0.0 --port 8080"
        subprocess.Popen(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        st.error(f"Error launching dashboard: {e}")
        return False

# Create demo study
def create_demo_study():
    try:
        import optuna
        
        # Create study database
        db_path = "studies/simple_demo.db"
        os.makedirs("studies", exist_ok=True)
        
        storage = f"sqlite:///{db_path}"
        study = optuna.create_study(
            study_name="simple_demo",
            storage=storage,
            direction="maximize",
            load_if_exists=True
        )
        
        # Simple objective function
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            y = trial.suggest_float('y', -10, 10)
            return -(x**2 + y**2)  # Maximize negative of sum of squares
        
        # Run optimization
        study.optimize(objective, n_trials=20)
        
        return True, len(study.trials)
    except Exception as e:
        return False, str(e)

# Dashboard status
dashboard_running = check_dashboard_status()

if dashboard_running:
    st.sidebar.success("✅ Dashboard Running")
    st.sidebar.markdown("[Open Dashboard](http://localhost:8080)")
else:
    st.sidebar.warning("❌ Dashboard Not Running")
    if st.sidebar.button("🚀 Launch Dashboard"):
        with st.spinner("Starting Optuna Dashboard..."):
            if launch_dashboard():
                time.sleep(3)
                st.sidebar.success("Dashboard launched! Refresh page.")
                st.experimental_rerun()

# Demo controls
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Demo Options")

if st.sidebar.button("Create Simple Demo Study"):
    with st.spinner("Creating simple demo optimization study..."):
        success, result = create_demo_study()
        if success:
            st.sidebar.success(f"✅ Created study with {result} trials")
        else:
            st.sidebar.error(f"❌ Error: {result}")

if st.sidebar.button("🚀 Create Comprehensive Demo"):
    with st.spinner("Creating comprehensive Optuna demonstration... This may take 2-3 minutes."):
        try:
            # Try the safe version first
            result = subprocess.run([sys.executable, "comprehensive_demo_safe.py"],
                                  capture_output=True, timeout=300, text=True)
            if result.returncode == 0:
                st.sidebar.success("✅ Comprehensive demo completed! Multiple studies created.")
                st.sidebar.info("🔄 Refresh the Optuna dashboard to see all new studies.")
                st.sidebar.info("📊 Studies created: TPE, Random, CMA-ES, Pruning, Multi-objective")
            else:
                st.sidebar.error(f"❌ Demo failed: {result.stderr}")
                # Fallback to original version
                st.sidebar.info("🔄 Trying alternative demo...")
                result2 = subprocess.run([sys.executable, "comprehensive_optuna_demo.py"],
                                       capture_output=True, timeout=300, text=True)
                if result2.returncode == 0:
                    st.sidebar.success("✅ Alternative demo completed!")
                else:
                    st.sidebar.error("❌ Both demo versions failed. Try running manually.")
        except subprocess.TimeoutExpired:
            st.sidebar.warning("⏱️ Demo is taking longer than expected but may still be running.")
        except Exception as e:
            st.sidebar.error(f"❌ Error running comprehensive demo: {e}")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Quick Start")
    st.markdown("""
    1. **Launch Dashboard**: Click the button in the sidebar
    2. **Create Demo**: Choose between simple or comprehensive demo
    3. **View Results**: Open the dashboard to see visualizations
    4. **Explore**: Try different optimization parameters

    **Demo Options:**
    - 🔹 **Simple Demo**: Quick 5-trial optimization study
    - 🚀 **Comprehensive Demo**: Full Optuna feature showcase with multiple studies
    """)
    
    # Show available studies
    st.subheader("📊 Available Studies")
    studies_dir = Path("studies")
    if studies_dir.exists():
        db_files = list(studies_dir.glob("*.db"))
        if db_files:
            for db_file in db_files:
                st.write(f"📁 {db_file.name}")
        else:
            st.write("No studies found. Create a demo study!")
    else:
        st.write("No studies directory found.")

with col2:
    st.subheader("🎯 Framework Features")
    st.markdown("""
    - **Optuna Integration**: Advanced hyperparameter optimization
    - **Multiple Algorithms**: Random Forest, XGBoost, LightGBM
    - **Real-time Dashboard**: Interactive visualization
    - **Study Management**: Create, view, and compare studies
    - **Easy Deployment**: Docker and local setup
    """)
    
    # System info
    st.subheader("💻 System Status")
    try:
        import optuna
        st.write(f"✅ Optuna: {optuna.__version__}")
    except:
        st.write("❌ Optuna not available")
    
    try:
        import sklearn
        st.write(f"✅ Scikit-learn: {sklearn.__version__}")
    except:
        st.write("❌ Scikit-learn not available")

# Footer
st.markdown("---")
st.markdown("**ML Optimization Framework** - Simple, Fast, Effective")

# Auto-refresh option
if st.sidebar.checkbox("Auto-refresh (30s)"):
    time.sleep(30)
    st.rerun()
