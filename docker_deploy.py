#!/usr/bin/env python3
"""
Simple Docker Deployment Script
Builds and deploys the ML Optimization Framework with Docker Compose
"""

import subprocess
import time
import webbrowser
import sys

def run_command(cmd, timeout=300):
    """Run a command and return success status"""
    try:
        print(f"🚀 Running: {cmd}")
        result = subprocess.run(cmd, shell=True, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"⏱️ Command timed out: {cmd}")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def main():
    """Main deployment function"""
    print("🐳 ML Optimization Framework - Docker Deployment")
    print("=" * 60)
    
    # Step 1: Clean up existing containers
    print("\n1️⃣ Cleaning up existing containers...")
    run_command("docker-compose down -v")
    
    # Step 2: Build and start services
    print("\n2️⃣ Building and starting services...")
    print("   This may take a few minutes...")
    
    success = run_command("docker-compose up -d --build")
    
    if not success:
        print("❌ Failed to start services")
        return False
    
    print("✅ Services started successfully!")
    
    # Step 3: Wait for services to be ready
    print("\n3️⃣ Waiting for services to initialize...")
    print("   Please wait 30-60 seconds for full startup...")
    
    for i in range(6):
        print(f"   ⏳ {(i+1)*10} seconds...")
        time.sleep(10)
    
    # Step 4: Show status
    print("\n4️⃣ Checking service status...")
    run_command("docker-compose ps")
    
    # Step 5: Open browsers
    print("\n5️⃣ Opening dashboards in browser...")
    try:
        webbrowser.open("http://localhost:8501")
        time.sleep(2)
        webbrowser.open("http://localhost:8080")
        print("✅ Browsers opened")
    except Exception as e:
        print(f"⚠️ Could not open browsers: {e}")
    
    # Step 6: Display final information
    print("\n" + "=" * 60)
    print("🎉 Docker Deployment Complete!")
    print("\n📍 Access URLs:")
    print("  🎨 Streamlit App: http://localhost:8501")
    print("  📊 Optuna Dashboard: http://localhost:8080")
    
    print("\n💡 Usage Tips:")
    print("  - Wait 1-2 minutes for full initialization")
    print("  - Use Streamlit app to create comprehensive demos")
    print("  - Check Optuna dashboard for study visualizations")
    print("  - Run 'docker-compose logs' to see service logs")
    print("  - Run 'docker-compose down' to stop services")
    
    print("\n🔧 Management Commands:")
    print("  - View logs: docker-compose logs")
    print("  - Stop services: docker-compose down")
    print("  - Restart: docker-compose restart")
    print("  - Update: docker-compose up -d --build")
    
    print("=" * 60)
    
    # Keep script running to show logs
    try:
        choice = input("\n📋 Show live logs? (y/N): ").lower().strip()
        if choice == 'y':
            print("\n📋 Showing live logs (Ctrl+C to exit)...")
            subprocess.run("docker-compose logs -f", shell=True)
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Deployment failed!")
        sys.exit(1)
    else:
        print("\n✅ Deployment successful!")
        sys.exit(0)
