#!/usr/bin/env python3
"""
Enhanced Streamlit Dashboard for ML Optimization Framework

This Streamlit app provides a comprehensive interface to:
1. Launch and monitor the Optuna dashboard
2. Run optimization demos with real-time progress
3. Visualize optimization results and analytics
4. Manage studies and experiments
5. View project documentation
"""

import streamlit as st
import subprocess
import time
import os
import sys
import sqlite3
import json
import threading
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def check_optuna_dashboard_status():
    """Check if Optuna dashboard is running."""
    try:
        response = requests.get("http://localhost:8080/api/studies", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_study_data():
    """Get study data from database."""
    db_files = [
        "studies/optuna_dashboard_demo.db",
        "studies/complete_optuna_showcase.db",
        "studies/comprehensive_optuna_demo.db"
    ]

    all_studies = []

    for db_file in db_files:
        if os.path.exists(db_file):
            try:
                conn = sqlite3.connect(db_file)

                # Get studies
                studies_df = pd.read_sql_query("""
                    SELECT study_id, study_name, direction,
                           datetime(date_created) as created_at
                    FROM studies
                """, conn)

                # Get trials for each study
                for _, study in studies_df.iterrows():
                    trials_df = pd.read_sql_query(f"""
                        SELECT trial_id, number, value, state,
                               datetime(datetime_start) as start_time,
                               datetime(datetime_complete) as end_time
                        FROM trials
                        WHERE study_id = {study['study_id']}
                        ORDER BY number
                    """, conn)

                    if not trials_df.empty:
                        study_info = {
                            'study_name': study['study_name'],
                            'direction': study['direction'],
                            'created_at': study['created_at'],
                            'total_trials': len(trials_df),
                            'completed_trials': len(trials_df[trials_df['state'] == 'COMPLETE']),
                            'best_value': trials_df[trials_df['state'] == 'COMPLETE']['value'].max() if study['direction'] == 'MAXIMIZE' else trials_df[trials_df['state'] == 'COMPLETE']['value'].min(),
                            'trials': trials_df,
                            'db_file': db_file
                        }
                        all_studies.append(study_info)

                conn.close()

            except Exception as e:
                st.error(f"Error reading {db_file}: {e}")

    return all_studies

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

    # Check dashboard status
    dashboard_status = check_optuna_dashboard_status()

    # Status indicator
    if dashboard_status:
        st.success("🟢 Optuna Dashboard is running at http://localhost:8080")
    else:
        st.warning("🟡 Optuna Dashboard is not running")

    # Sidebar
    st.sidebar.title("🎛️ Navigation")

    # Dashboard quick access
    if dashboard_status:
        if st.sidebar.button("🌐 Open Optuna Dashboard", type="primary"):
            st.markdown("""
            <script>
            window.open('http://localhost:8080', '_blank');
            </script>
            """, unsafe_allow_html=True)

    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🚀 Quick Start", "📊 Dashboard Control", "📈 Live Analytics", "🔧 Tools", "📚 Documentation"]
    )

    if page == "🏠 Home":
        show_home_page()
    elif page == "🚀 Quick Start":
        show_quick_start()
    elif page == "📊 Dashboard Control":
        show_dashboard_control()
    elif page == "📈 Live Analytics":
        show_live_analytics()
    elif page == "🔧 Tools":
        show_tools_page()
    elif page == "📚 Documentation":
        show_documentation()

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

def show_dashboard_control():
    """Show dashboard control and management."""

    st.markdown("## 📊 Dashboard Control Center")

    # Dashboard status
    dashboard_status = check_optuna_dashboard_status()

    col1, col2, col3 = st.columns(3)

    with col1:
        if dashboard_status:
            st.success("🟢 Dashboard Running")
            if st.button("🌐 Open Dashboard", type="primary"):
                st.markdown("""
                <script>
                window.open('http://localhost:8080', '_blank');
                </script>
                """, unsafe_allow_html=True)
        else:
            st.error("🔴 Dashboard Stopped")
            if st.button("🚀 Start Dashboard", type="primary"):
                with st.spinner("Starting Optuna dashboard..."):
                    try:
                        subprocess.Popen([
                            sys.executable, "scripts/start_dashboard.py"
                        ])
                        st.success("✅ Dashboard starting! Check http://localhost:8080 in a few seconds.")
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error starting dashboard: {e}")

    with col2:
        if st.button("📊 Populate Demo Data"):
            with st.spinner("Creating demo studies..."):
                try:
                    result = subprocess.run([
                        sys.executable, "scripts/populate_dashboard.py"
                    ], capture_output=True, text=True, timeout=120)

                    if result.returncode == 0:
                        st.success("✅ Demo data created!")
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {result.stderr}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    with col3:
        if st.button("🔥 Run Feature Showcase"):
            st.info("Starting comprehensive feature showcase...")
            try:
                subprocess.Popen([sys.executable, "scripts/showcase_all_optuna_features.py"])
                st.success("✅ Showcase started! Check terminal for progress.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # Study management
    st.markdown("### 📋 Study Management")

    studies = get_study_data()

    if studies:
        study_summary = []
        for study in studies:
            study_summary.append({
                'Study Name': study['study_name'],
                'Direction': study['direction'],
                'Total Trials': study['total_trials'],
                'Completed': study['completed_trials'],
                'Best Value': f"{study['best_value']:.4f}" if study['best_value'] is not None else "N/A",
                'Created': study['created_at']
            })

        df = pd.DataFrame(study_summary)
        st.dataframe(df, use_container_width=True)

        # Study selector for detailed view
        selected_study = st.selectbox(
            "Select study for detailed view:",
            options=[s['study_name'] for s in studies]
        )

        if selected_study:
            study_data = next(s for s in studies if s['study_name'] == selected_study)
            show_study_details(study_data)
    else:
        st.warning("⚠️ No studies found. Create some demo data first!")

def show_study_details(study_data):
    """Show detailed information about a specific study."""

    st.markdown(f"#### 📊 Study: {study_data['study_name']}")

    trials_df = study_data['trials']

    if not trials_df.empty:
        # Optimization progress chart
        fig = px.line(
            trials_df[trials_df['state'] == 'COMPLETE'],
            x='number',
            y='value',
            title=f"Optimization Progress - {study_data['study_name']}",
            labels={'number': 'Trial Number', 'value': 'Objective Value'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Trial statistics
        col1, col2, col3, col4 = st.columns(4)

        completed_trials = trials_df[trials_df['state'] == 'COMPLETE']

        with col1:
            st.metric("Total Trials", len(trials_df))

        with col2:
            st.metric("Completed", len(completed_trials))

        with col3:
            if not completed_trials.empty:
                best_value = completed_trials['value'].max() if study_data['direction'] == 'MAXIMIZE' else completed_trials['value'].min()
                st.metric("Best Value", f"{best_value:.4f}")

        with col4:
            if not completed_trials.empty:
                avg_value = completed_trials['value'].mean()
                st.metric("Average Value", f"{avg_value:.4f}")

        # Recent trials
        st.markdown("##### 📋 Recent Trials")
        recent_trials = trials_df.tail(10)[['number', 'value', 'state', 'start_time']]
        st.dataframe(recent_trials, use_container_width=True)

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

def show_live_analytics():
    """Show live analytics and comprehensive metrics."""

    st.markdown("## 📈 Live Analytics & Insights")

    # Auto-refresh option
    auto_refresh = st.checkbox("🔄 Auto-refresh (every 30 seconds)")

    if auto_refresh:
        # Auto-refresh every 30 seconds
        time.sleep(30)
        st.rerun()

    studies = get_study_data()

    if not studies:
        st.warning("⚠️ No studies found. Create some demo data first!")
        return

    # Overall statistics
    st.markdown("### 📊 Overall Statistics")

    total_studies = len(studies)
    total_trials = sum(s['total_trials'] for s in studies)
    total_completed = sum(s['completed_trials'] for s in studies)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Studies", total_studies)

    with col2:
        st.metric("Total Trials", total_trials)

    with col3:
        st.metric("Completed Trials", total_completed)

    with col4:
        completion_rate = (total_completed / total_trials * 100) if total_trials > 0 else 0
        st.metric("Completion Rate", f"{completion_rate:.1f}%")

    # Studies comparison
    st.markdown("### 🔄 Studies Comparison")

    # Prepare data for comparison
    comparison_data = []
    for study in studies:
        comparison_data.append({
            'Study': study['study_name'],
            'Total Trials': study['total_trials'],
            'Completed Trials': study['completed_trials'],
            'Best Value': study['best_value'] if study['best_value'] is not None else 0,
            'Direction': study['direction']
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Trials comparison chart
    fig1 = px.bar(
        comparison_df,
        x='Study',
        y=['Total Trials', 'Completed Trials'],
        title="Trials per Study",
        barmode='group'
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Best values comparison
    fig2 = px.bar(
        comparison_df,
        x='Study',
        y='Best Value',
        color='Direction',
        title="Best Values Comparison"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Detailed study analysis
    st.markdown("### 🔍 Detailed Study Analysis")

    selected_study = st.selectbox(
        "Select study for detailed analysis:",
        options=[s['study_name'] for s in studies],
        key="analytics_study_selector"
    )

    if selected_study:
        study_data = next(s for s in studies if s['study_name'] == selected_study)
        show_detailed_analytics(study_data)

def show_detailed_analytics(study_data):
    """Show detailed analytics for a specific study."""

    st.markdown(f"#### 🔬 Detailed Analysis: {study_data['study_name']}")

    trials_df = study_data['trials']
    completed_trials = trials_df[trials_df['state'] == 'COMPLETE']

    if completed_trials.empty:
        st.warning("No completed trials found for this study.")
        return

    # Performance over time
    col1, col2 = st.columns(2)

    with col1:
        # Optimization progress
        fig1 = px.line(
            completed_trials,
            x='number',
            y='value',
            title="Optimization Progress",
            labels={'number': 'Trial Number', 'value': 'Objective Value'}
        )

        # Add best value line
        if study_data['direction'] == 'MAXIMIZE':
            best_so_far = completed_trials['value'].cummax()
        else:
            best_so_far = completed_trials['value'].cummin()

        fig1.add_scatter(
            x=completed_trials['number'],
            y=best_so_far,
            mode='lines',
            name='Best So Far',
            line=dict(color='red', dash='dash')
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Value distribution
        fig2 = px.histogram(
            completed_trials,
            x='value',
            title="Value Distribution",
            nbins=20
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Performance metrics
    st.markdown("##### 📊 Performance Metrics")

    values = completed_trials['value']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean", f"{values.mean():.4f}")

    with col2:
        st.metric("Std Dev", f"{values.std():.4f}")

    with col3:
        st.metric("Min", f"{values.min():.4f}")

    with col4:
        st.metric("Max", f"{values.max():.4f}")

    # Trial timeline
    if 'start_time' in completed_trials.columns:
        st.markdown("##### ⏱️ Trial Timeline")

        # Convert to datetime if needed
        completed_trials['start_time'] = pd.to_datetime(completed_trials['start_time'])

        fig3 = px.scatter(
            completed_trials,
            x='start_time',
            y='value',
            color='value',
            title="Trials Timeline",
            labels={'start_time': 'Start Time', 'value': 'Objective Value'}
        )
        st.plotly_chart(fig3, use_container_width=True)

if __name__ == "__main__":
    main()
