#!/usr/bin/env python3
"""
Docker Testing Script for ML Optimization Framework
Tests Docker Compose deployment and service functionality
"""

import subprocess
import time
import requests
import sys
from pathlib import Path

def run_command(cmd, timeout=60):
    """Run a command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, 
                              text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def test_docker_compose():
    """Test Docker Compose deployment"""
    print("🐳 Testing Docker Compose Deployment")
    print("=" * 50)
    
    # Step 1: Clean up any existing containers
    print("1. Cleaning up existing containers...")
    run_command("docker-compose down -v")
    time.sleep(2)
    
    # Step 2: Build and start services
    print("2. Building and starting services...")
    success, stdout, stderr = run_command("docker-compose up -d --build", timeout=300)
    
    if not success:
        print(f"❌ Failed to start services: {stderr}")
        return False
    
    print("✅ Services started successfully")
    
    # Step 3: Wait for services to be ready
    print("3. Waiting for services to be ready...")
    time.sleep(30)  # Give services time to start
    
    # Step 4: Check service status
    print("4. Checking service status...")
    success, stdout, stderr = run_command("docker-compose ps")
    print(stdout)
    
    # Step 5: Test Streamlit accessibility
    print("5. Testing Streamlit accessibility...")
    streamlit_working = False
    for attempt in range(5):
        try:
            response = requests.get("http://localhost:8501", timeout=10)
            if response.status_code == 200:
                print("✅ Streamlit is accessible")
                streamlit_working = True
                break
        except requests.RequestException:
            pass
        
        print(f"   Attempt {attempt + 1}/5 failed, retrying...")
        time.sleep(10)
    
    if not streamlit_working:
        print("❌ Streamlit is not accessible")
    
    # Step 6: Test Optuna Dashboard accessibility
    print("6. Testing Optuna Dashboard accessibility...")
    optuna_working = False
    for attempt in range(5):
        try:
            response = requests.get("http://localhost:8080", timeout=10)
            if response.status_code == 200:
                print("✅ Optuna Dashboard is accessible")
                optuna_working = True
                break
        except requests.RequestException:
            pass
        
        print(f"   Attempt {attempt + 1}/5 failed, retrying...")
        time.sleep(10)
    
    if not optuna_working:
        print("❌ Optuna Dashboard is not accessible")
    
    # Step 7: Check logs if services failed
    if not streamlit_working or not optuna_working:
        print("7. Checking service logs...")
        print("\n--- Streamlit Logs ---")
        success, stdout, stderr = run_command("docker-compose logs streamlit-app")
        print(stdout)
        
        print("\n--- Optuna Dashboard Logs ---")
        success, stdout, stderr = run_command("docker-compose logs optuna-dashboard")
        print(stdout)
    
    # Step 8: Test study creation inside container
    print("8. Testing study creation inside container...")
    success, stdout, stderr = run_command(
        "docker-compose exec -T streamlit-app python comprehensive_demo_safe.py",
        timeout=180
    )
    
    if success:
        print("✅ Comprehensive demo created successfully in container")
    else:
        print(f"❌ Failed to create demo in container: {stderr}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🐳 Docker Test Results:")
    print(f"  Streamlit App: {'✅ Working' if streamlit_working else '❌ Failed'}")
    print(f"  Optuna Dashboard: {'✅ Working' if optuna_working else '❌ Failed'}")
    print(f"  Demo Creation: {'✅ Working' if success else '❌ Failed'}")
    
    overall_success = streamlit_working and optuna_working
    
    if overall_success:
        print("\n🎉 Docker deployment is working correctly!")
        print("📍 Access URLs:")
        print("  🎨 Streamlit: http://localhost:8501")
        print("  📊 Optuna Dashboard: http://localhost:8080")
    else:
        print("\n⚠️  Docker deployment has issues. Check logs above.")
    
    print("=" * 50)
    return overall_success

def cleanup_docker():
    """Clean up Docker resources"""
    print("\n🧹 Cleaning up Docker resources...")
    run_command("docker-compose down -v")
    print("✅ Cleanup complete")

def main():
    """Main test function"""
    print("🎯 ML Optimization Framework - Docker Testing")
    print("=" * 60)
    
    try:
        # Test Docker Compose
        success = test_docker_compose()
        
        # Ask user if they want to keep services running
        if success:
            keep_running = input("\n🤔 Keep services running? (y/N): ").lower().strip()
            if keep_running != 'y':
                cleanup_docker()
        else:
            cleanup_docker()
        
        return success
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        cleanup_docker()
        return False
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        cleanup_docker()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
