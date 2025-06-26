#!/usr/bin/env python3
"""
Streamlit Dashboard for ML Optimization Framework

This Streamlit app provides an easy interface to:
1. Launch the Optuna dashboard
2. Run optimization demos
3. View project documentation
4. Monitor optimization progress
"""

import streamlit as st
import subprocess
import time
import os
import sys
import sqlite3
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="ML Optimization Framework",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 ML Optimization Framework</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Complete Optuna Demonstration with Interactive Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🚀 Quick Start", "📊 Dashboard", "🔧 Tools", "📚 Documentation", "📈 Analytics"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🚀 Quick Start":
        show_quick_start()
    elif page == "📊 Dashboard":
        show_dashboard_page()
    elif page == "🔧 Tools":
        show_tools_page()
    elif page == "📚 Documentation":
        show_documentation()
    elif page == "📈 Analytics":
        show_analytics()

def show_home_page():
    """Show the home page."""
    
    st.markdown("## 🌟 What is this project?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
        <h3>🎯 Complete Optuna Showcase</h3>
        <p>This project demonstrates <strong>ALL major Optuna features</strong> through a real-world ML optimization framework:</p>
        <ul>
        <li>✓ Single & Multi-objective optimization</li>
        <li>✓ All samplers (TPE, Random, CMA-ES)</li>
        <li>✓ All pruners (Median, Successive Halving, Hyperband)</li>
        <li>✓ Interactive dashboard with rich visualizations</li>
        <li>✓ Production-ready code quality</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
        <h3>🤖 What is Optuna?</h3>
        <p><strong>Optuna</strong> is an automatic hyperparameter optimization framework for machine learning.</p>
        <ul>
        <li>* <strong>Intelligent Search</strong> - Uses advanced algorithms</li>
        <li>* <strong>Faster Results</strong> - Better hyperparameters with fewer trials</li>
        <li>* <strong>Easy Parallelization</strong> - Run multiple trials simultaneously</li>
        <li>* <strong>Smart Pruning</strong> - Stop unpromising trials early</li>
        <li>* <strong>Multi-objective</strong> - Optimize multiple metrics</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## 🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎛️ Launch Dashboard", type="primary", use_container_width=True):
            st.info("Dashboard launching instructions in the Dashboard tab!")
    
    with col2:
        if st.button("📊 Populate Demo Data", use_container_width=True):
            with st.spinner("Creating demo studies..."):
                try:
                    result = subprocess.run([
                        sys.executable, "scripts/populate_dashboard.py"
                    ], capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        st.success("✅ Demo data created successfully!")
                    else:
                        st.error(f"❌ Error: {result.stderr}")
                except subprocess.TimeoutExpired:
                    st.error("❌ Operation timed out")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col3:
        if st.button("🔍 Validate Framework", use_container_width=True):
            with st.spinner("Validating framework..."):
                try:
                    result = subprocess.run([
                        sys.executable, "scripts/validate_framework.py"
                    ], capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        st.success("✅ Framework validation passed!")
                        st.text(result.stdout[-500:])  # Show last 500 chars
                    else:
                        st.error(f"❌ Validation failed: {result.stderr}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

def show_quick_start():
    """Show quick start guide."""
    
    st.markdown("## 🚀 Quick Start Guide")
    
    st.markdown("""
    <div class="info-box">
    <h3>📋 Prerequisites</h3>
    <p>Make sure you have Python 3.10+ installed and all dependencies from requirements.txt</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Step 1: Populate Dashboard with Demo Data")
    
    if st.button("🎯 Create Demo Studies", type="primary"):
        with st.spinner("Creating comprehensive demo studies..."):
            try:
                result = subprocess.run([
                    sys.executable, "scripts/populate_dashboard.py"
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    st.success("✅ Demo studies created successfully!")
                    st.code(result.stdout, language="text")
                else:
                    st.error(f"❌ Error creating studies: {result.stderr}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.markdown("### Step 2: Start the Optuna Dashboard")
    
    st.markdown("""
    <div class="success-box">
    <h4>🌐 Dashboard Instructions</h4>
    <p>To start the Optuna dashboard, run this command in your terminal:</p>
    <code>python scripts/start_dashboard.py</code>
    <p>Then open <strong>http://localhost:8080</strong> in your browser</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Step 3: Explore All Features")
    
    if st.button("🔥 Run Complete Feature Showcase"):
        st.info("This will run in the background. Check the terminal for progress.")
        try:
            subprocess.Popen([sys.executable, "scripts/showcase_all_optuna_features.py"])
            st.success("✅ Feature showcase started! Check your terminal for progress.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def show_dashboard_page():
    """Show dashboard information."""
    
    st.markdown("## 📊 Optuna Dashboard")
    
    st.markdown("""
    <div class="info-box">
    <h3>🎛️ Interactive Dashboard Features</h3>
    <p>The Optuna dashboard provides comprehensive visualization and analysis tools:</p>
    <ul>
    <li>📈 <strong>Optimization History</strong> - See how trials improve over time</li>
    <li>🎯 <strong>Parameter Importance</strong> - Understand which hyperparameters matter most</li>
    <li>🔄 <strong>Pareto Front</strong> - Multi-objective trade-off analysis</li>
    <li>🔍 <strong>Trial Filtering</strong> - Filter by status, parameters, or objectives</li>
    <li>📊 <strong>Parameter Relationships</strong> - Correlation and interaction plots</li>
    <li>📋 <strong>Study Comparison</strong> - Compare different optimization strategies</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Launch Dashboard")
        
        if st.button("🎛️ Start Optuna Dashboard", type="primary"):
            st.markdown("""
            <div class="success-box">
            <h4>🌐 Dashboard Starting...</h4>
            <p>Run this command in your terminal:</p>
            <code>python scripts/start_dashboard.py</code>
            <br><br>
            <p>Then open: <a href="http://localhost:8080" target="_blank">http://localhost:8080</a></p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Check Database")
        
        if st.button("🔍 Check Studies Database"):
            check_database_status()

def check_database_status():
    """Check the status of the studies database."""
    
    db_files = [
        "studies/optuna_dashboard_demo.db",
        "studies/complete_optuna_showcase.db",
        "studies/optuna_studies.db"
    ]
    
    found_studies = False
    
    for db_file in db_files:
        if os.path.exists(db_file):
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                
                # Check if studies table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='studies';")
                if cursor.fetchone():
                    cursor.execute("SELECT COUNT(*) FROM studies;")
                    study_count = cursor.fetchone()[0]
                    
                    if study_count > 0:
                        st.success(f"✅ Found {study_count} studies in {db_file}")
                        found_studies = True
                        
                        # Get study details
                        cursor.execute("SELECT study_name, direction FROM studies LIMIT 10;")
                        studies = cursor.fetchall()
                        
                        if studies:
                            df = pd.DataFrame(studies, columns=["Study Name", "Direction"])
                            st.dataframe(df, use_container_width=True)
                
                conn.close()
                
            except Exception as e:
                st.warning(f"⚠️ Could not read {db_file}: {str(e)}")
    
    if not found_studies:
        st.warning("⚠️ No studies found. Run 'Populate Demo Data' first!")

def show_tools_page():
    """Show tools and utilities."""
    
    st.markdown("## 🔧 Tools & Utilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Optimization Tools")
        
        if st.button("🔥 Run Feature Showcase", use_container_width=True):
            st.info("Starting comprehensive feature showcase...")
            try:
                subprocess.Popen([sys.executable, "scripts/showcase_all_optuna_features.py"])
                st.success("✅ Showcase started! Check terminal for progress.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        if st.button("🔍 Validate Framework", use_container_width=True):
            with st.spinner("Validating framework..."):
                try:
                    result = subprocess.run([
                        sys.executable, "scripts/validate_framework.py"
                    ], capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        st.success("✅ Framework validation passed!")
                    else:
                        st.error("❌ Validation failed!")
                        st.code(result.stderr)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.markdown("### 📊 Data Tools")
        
        if st.button("📈 Populate Dashboard", use_container_width=True):
            with st.spinner("Creating demo studies..."):
                try:
                    result = subprocess.run([
                        sys.executable, "scripts/populate_dashboard.py"
                    ], capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        st.success("✅ Demo data created!")
                    else:
                        st.error("❌ Failed to create demo data!")
                        st.code(result.stderr)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        if st.button("🧹 Clean Studies", use_container_width=True):
            if st.checkbox("⚠️ Confirm deletion of all studies"):
                try:
                    for db_file in ["studies/optuna_dashboard_demo.db", "studies/complete_optuna_showcase.db"]:
                        if os.path.exists(db_file):
                            os.remove(db_file)
                    st.success("✅ Studies cleaned!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

def show_documentation():
    """Show documentation."""
    
    st.markdown("## 📚 Documentation")
    
    st.markdown("""
    <div class="info-box">
    <h3>📖 Available Documentation</h3>
    <p>Comprehensive guides and tutorials for the ML Optimization Framework:</p>
    </div>
    """, unsafe_allow_html=True)
    
    docs = [
        ("📋 README.md", "Main project documentation", "README.md"),
        ("🎯 Complete Tutorial", "Comprehensive Optuna tutorial", "docs/COMPLETE_OPTUNA_TUTORIAL.md"),
        ("🔧 API Reference", "Code documentation", "src/"),
        ("🐳 Docker Guide", "Containerization setup", "docker-compose.yml"),
    ]
    
    for title, description, path in docs:
        with st.expander(f"{title} - {description}"):
            if os.path.exists(path):
                if path.endswith('.md'):
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        st.markdown(content[:2000] + "..." if len(content) > 2000 else content)
                else:
                    st.info(f"📁 Directory: {path}")
            else:
                st.warning(f"⚠️ File not found: {path}")

def show_analytics():
    """Show analytics and metrics."""
    
    st.markdown("## 📈 Analytics")
    
    st.info("📊 Analytics features coming soon! This will show optimization metrics and performance analysis.")
    
    # Placeholder for future analytics
    if os.path.exists("studies/optuna_dashboard_demo.db"):
        st.markdown("### 📊 Study Statistics")
        
        try:
            conn = sqlite3.connect("studies/optuna_dashboard_demo.db")
            
            # Simple analytics
            query = """
            SELECT study_name, COUNT(*) as trial_count 
            FROM trials t 
            JOIN studies s ON t.study_id = s.study_id 
            GROUP BY study_name
            """
            
            df = pd.read_sql_query(query, conn)
            
            if not df.empty:
                fig = px.bar(df, x='study_name', y='trial_count', 
                           title='Trials per Study')
                st.plotly_chart(fig, use_container_width=True)
            
            conn.close()
            
        except Exception as e:
            st.warning(f"⚠️ Could not load analytics: {str(e)}")

if __name__ == "__main__":
    main()
