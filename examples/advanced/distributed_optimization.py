#!/usr/bin/env python3
"""
Distributed Optimization Example
Demonstrates how to run optimization across multiple processes/machines
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import optuna
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import multiprocessing
import time
from pathlib import Path
import concurrent.futures

def single_worker_optimization(worker_id, n_trials, study_name, storage_url):
    """
    Single worker optimization function
    This function will be run by each worker process
    """
    print(f"🔧 Worker {worker_id} starting with {n_trials} trials")
    
    # Generate dataset (each worker uses same data for consistency)
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    def objective(trial):
        """Objective function for optimization"""
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=1  # Single job per model to avoid conflicts
        )
        
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return scores.mean()
    
    # Create or load study (shared across workers)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    print(f"✅ Worker {worker_id} completed. Best value: {study.best_value:.4f}")
    return study.best_value

def distributed_optimization_demo():
    """
    Demonstrate distributed optimization using multiple processes
    """
    print("🎯 Distributed Optimization Demo")
    print("=" * 50)
    
    # Configuration
    n_workers = min(4, multiprocessing.cpu_count())  # Use up to 4 workers
    trials_per_worker = 25
    total_trials = n_workers * trials_per_worker
    study_name = "distributed_demo"
    storage_url = "sqlite:///studies/distributed_optimization.db"
    
    print(f"🔧 Configuration:")
    print(f"  Workers: {n_workers}")
    print(f"  Trials per worker: {trials_per_worker}")
    print(f"  Total trials: {total_trials}")
    print(f"  Storage: {storage_url}")
    
    # Ensure studies directory exists
    Path("studies").mkdir(exist_ok=True)
    
    # Create initial study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True
    )
    
    print(f"\n🚀 Starting distributed optimization...")
    start_time = time.time()
    
    # Run workers in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit jobs to workers
        futures = []
        for worker_id in range(n_workers):
            future = executor.submit(
                single_worker_optimization,
                worker_id,
                trials_per_worker,
                study_name,
                storage_url
            )
            futures.append(future)
        
        # Wait for all workers to complete
        results = []
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                print(f"📊 Worker completed. Best so far: {max(results):.4f}")
            except Exception as e:
                print(f"❌ Worker {i} failed: {e}")
    
    end_time = time.time()
    
    # Load final study results
    final_study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True
    )
    
    # Display results
    print(f"\n📊 Distributed Optimization Results:")
    print(f"  Total time: {end_time - start_time:.2f} seconds")
    print(f"  Total trials: {len(final_study.trials)}")
    print(f"  Best value: {final_study.best_value:.4f}")
    print(f"  Best parameters: {final_study.best_params}")
    
    # Compare with sequential optimization
    print(f"\n🔄 Running sequential optimization for comparison...")
    sequential_start = time.time()
    
    sequential_study = optuna.create_study(
        study_name="sequential_demo",
        storage="sqlite:///studies/sequential_optimization.db",
        direction="maximize",
        load_if_exists=True
    )
    
    # Generate same dataset
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return scores.mean()
    
    sequential_study.optimize(objective, n_trials=total_trials)
    sequential_end = time.time()
    
    # Comparison
    print(f"\n📈 Performance Comparison:")
    print(f"  Distributed time: {end_time - start_time:.2f}s")
    print(f"  Sequential time: {sequential_end - sequential_start:.2f}s")
    print(f"  Speedup: {(sequential_end - sequential_start) / (end_time - start_time):.2f}x")
    print(f"  Distributed best: {final_study.best_value:.4f}")
    print(f"  Sequential best: {sequential_study.best_value:.4f}")
    
    return final_study, sequential_study

def database_backend_demo():
    """
    Demonstrate different database backends for distributed optimization
    """
    print("\n🎯 Database Backend Demo")
    print("=" * 50)
    
    print("📚 Available Storage Backends:")
    print("  1. SQLite (default) - Good for single machine")
    print("  2. PostgreSQL - Best for distributed systems")
    print("  3. MySQL - Alternative for distributed systems")
    print("  4. Redis - In-memory, fast but not persistent")
    
    # SQLite example (already demonstrated above)
    sqlite_url = "sqlite:///studies/sqlite_backend.db"
    print(f"\n✅ SQLite URL: {sqlite_url}")
    
    # PostgreSQL example (commented out as it requires setup)
    postgresql_url = "postgresql://username:password@localhost:5432/optuna"
    print(f"📝 PostgreSQL URL: {postgresql_url}")
    print("   (Requires PostgreSQL server setup)")
    
    # MySQL example (commented out as it requires setup)
    mysql_url = "mysql://username:password@localhost:3306/optuna"
    print(f"📝 MySQL URL: {mysql_url}")
    print("   (Requires MySQL server setup)")
    
    # Redis example (commented out as it requires setup)
    redis_url = "redis://localhost:6379/0"
    print(f"📝 Redis URL: {redis_url}")
    print("   (Requires Redis server setup)")
    
    print(f"\n💡 Tips for Production:")
    print("  - Use PostgreSQL/MySQL for multi-machine setups")
    print("  - SQLite is fine for single-machine multi-process")
    print("  - Redis is fastest but data is not persistent")
    print("  - Always use connection pooling for databases")

def load_balancing_demo():
    """
    Demonstrate load balancing strategies for distributed optimization
    """
    print("\n🎯 Load Balancing Strategies")
    print("=" * 50)
    
    strategies = {
        "Equal Distribution": "Each worker gets same number of trials",
        "Dynamic Allocation": "Workers request trials as they complete",
        "Priority-based": "Important trials get more resources",
        "Adaptive": "Allocation based on worker performance"
    }
    
    for strategy, description in strategies.items():
        print(f"📋 {strategy}: {description}")
    
    print(f"\n🔧 Implementation Example (Dynamic Allocation):")
    print("""
    # Worker requests trials dynamically
    while not optimization_complete:
        trial = study.ask()  # Request next trial
        result = evaluate_trial(trial)
        study.tell(trial, result)  # Report result
    """)

def main():
    """Run distributed optimization examples"""
    print("🎯 Distributed Optimization Examples")
    print("=" * 60)
    
    try:
        # Run distributed optimization demo
        distributed_study, sequential_study = distributed_optimization_demo()
        
        # Show database backend options
        database_backend_demo()
        
        # Show load balancing strategies
        load_balancing_demo()
        
        print("\n" + "=" * 60)
        print("🎉 Distributed Optimization Examples Complete!")
        
        print("\n📁 Studies Created:")
        print("  - studies/distributed_optimization.db")
        print("  - studies/sequential_optimization.db")
        
        print("\n💡 Key Learnings:")
        print("  - Distributed optimization can significantly speed up search")
        print("  - Shared storage enables coordination between workers")
        print("  - Different backends suit different deployment scenarios")
        print("  - Load balancing strategies affect efficiency")
        
        print("\n📍 Production Tips:")
        print("  - Use PostgreSQL/MySQL for multi-machine setups")
        print("  - Monitor worker performance and adjust allocation")
        print("  - Handle worker failures gracefully")
        print("  - Use connection pooling for database efficiency")
        
    except Exception as e:
        print(f"\n❌ Error in distributed optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
