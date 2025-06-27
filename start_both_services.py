#!/usr/bin/env python3
"""
Simple and Reliable Service Starter
Starts both Streamlit and Optuna Dashboard
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def main():
    print("🎯 ML Optimization Framework - Service Starter")
    print("=" * 55)
    
    # Ensure directories exist
    Path("studies").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Check for studies
    studies_dir = Path("studies")
    db_files = list(studies_dir.glob("*.db"))
    
    if not db_files:
        print("📊 No study databases found. Creating demo studies...")
        try:
            subprocess.run([sys.executable, "quick_demo.py"], timeout=60)
            db_files = list(studies_dir.glob("*.db"))
            print(f"✅ Created {len(db_files)} demo studies")
        except Exception as e:
            print(f"⚠️  Could not create demo studies: {e}")
            # Create minimal study
            try:
                import optuna
                study = optuna.create_study(storage="sqlite:///studies/minimal.db")
                study.optimize(lambda trial: trial.suggest_float("x", -1, 1) ** 2, n_trials=3)
                db_files = [Path("studies/minimal.db")]
                print("✅ Created minimal study")
            except Exception as e2:
                print(f"❌ Could not create minimal study: {e2}")
                return
    
    if db_files:
        print(f"📊 Found {len(db_files)} study database(s)")
        
        # Start Optuna Dashboard
        print(f"\n🔧 Starting Optuna Dashboard...")
        db_path = db_files[0]
        storage_url = f"sqlite:///{db_path}"
        
        print(f"📍 Dashboard will be at: http://localhost:8080")
        print(f"📊 Using database: {db_path}")
        
        optuna_cmd = f'start "Optuna Dashboard" optuna-dashboard {storage_url} --port 8080 --host 0.0.0.0'
        subprocess.Popen(optuna_cmd, shell=True)
        print("✅ Optuna Dashboard started in new window")
        
        # Give dashboard time to start
        time.sleep(3)
    
    # Start Streamlit
    print(f"\n🚀 Starting Streamlit App...")
    print(f"📍 Streamlit will be at: http://localhost:8501")
    
    streamlit_cmd = f'start "Streamlit App" streamlit run simple_app.py --server.port 8501 --server.address 0.0.0.0'
    subprocess.Popen(streamlit_cmd, shell=True)
    print("✅ Streamlit App started in new window")
    
    # Give Streamlit time to start
    time.sleep(5)
    
    # Open browsers
    print(f"\n🌐 Opening browsers...")
    try:
        webbrowser.open("http://localhost:8501")
        time.sleep(1)
        webbrowser.open("http://localhost:8080")
        print("✅ Browsers opened")
    except Exception as e:
        print(f"⚠️  Could not open browsers: {e}")
    
    # Display status
    print("\n" + "=" * 55)
    print("🎉 Services Started Successfully!")
    print("\n📍 Access URLs:")
    print("  🎨 Streamlit App: http://localhost:8501")
    print("  📊 Optuna Dashboard: http://localhost:8080")
    
    print("\n💡 Usage Tips:")
    print("  - Both services are running in separate windows")
    print("  - Use the Streamlit app for interactive optimization")
    print("  - Use the Optuna dashboard to analyze results")
    print("  - Close the terminal windows to stop services")
    
    print("\n🔧 Available Commands:")
    print("  - Run comprehensive demo: python comprehensive_optuna_demo.py")
    print("  - Run basic examples: python examples/basic_optimization.py")
    print("  - Validate setup: python validate_clean.py")
    
    print("\n🐳 Docker Alternative:")
    print("  - Build and run: docker-compose up -d")
    print("  - Stop services: docker-compose down")
    
    print("=" * 55)
    
    input("\n⏸️  Press Enter to exit (services will continue running)...")

if __name__ == "__main__":
    main()
