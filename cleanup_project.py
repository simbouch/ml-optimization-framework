#!/usr/bin/env python3
"""
Project cleanup script for ML Optimization Framework
Removes unnecessary files and optimizes the project structure
"""

import os
import shutil
from pathlib import Path

def cleanup_pycache():
    """Remove all __pycache__ directories."""
    print("🧹 Cleaning up __pycache__ directories...")
    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            pycache_path = Path(root) / "__pycache__"
            print(f"  Removing: {pycache_path}")
            shutil.rmtree(pycache_path)
            dirs.remove("__pycache__")

def cleanup_logs():
    """Clean up old log files but keep the directory."""
    print("🧹 Cleaning up old log files...")
    logs_dir = Path("logs")
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log"):
            if log_file.stat().st_size > 10 * 1024 * 1024:  # > 10MB
                print(f"  Removing large log: {log_file}")
                log_file.unlink()

def cleanup_temp_files():
    """Remove temporary files."""
    print("🧹 Cleaning up temporary files...")
    temp_patterns = [
        "*.tmp",
        "*.temp",
        ".DS_Store",
        "Thumbs.db",
        "*.pyc",
        "*.pyo"
    ]
    
    for pattern in temp_patterns:
        for file_path in Path(".").rglob(pattern):
            if file_path.is_file():
                print(f"  Removing: {file_path}")
                file_path.unlink()

def optimize_database_files():
    """Optimize SQLite database files."""
    print("🔧 Optimizing database files...")
    studies_dir = Path("studies")
    if studies_dir.exists():
        for db_file in studies_dir.glob("*.db"):
            try:
                import sqlite3
                conn = sqlite3.connect(db_file)
                conn.execute("VACUUM")
                conn.close()
                print(f"  Optimized: {db_file}")
            except Exception as e:
                print(f"  Warning: Could not optimize {db_file}: {e}")

def create_gitignore():
    """Create or update .gitignore file."""
    print("📝 Creating/updating .gitignore...")
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/*.log
*.log

# Temporary files
*.tmp
*.temp

# Database files (keep structure, ignore large files)
studies/*.db-journal
studies/*.db-wal

# Results (optional - uncomment if you don't want to track results)
# results/

# Docker
.dockerignore

# Environment variables
.env
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("  ✅ .gitignore updated")

def display_project_stats():
    """Display project statistics."""
    print("\n📊 Project Statistics:")
    
    # Count files by type
    file_counts = {}
    total_size = 0
    
    for file_path in Path(".").rglob("*"):
        if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
            suffix = file_path.suffix or "no_extension"
            file_counts[suffix] = file_counts.get(suffix, 0) + 1
            total_size += file_path.stat().st_size
    
    print(f"  Total files: {sum(file_counts.values())}")
    print(f"  Total size: {total_size / 1024 / 1024:.2f} MB")
    print("  File types:")
    for suffix, count in sorted(file_counts.items()):
        print(f"    {suffix}: {count}")

def main():
    """Main cleanup function."""
    print("🎯 ML Optimization Framework - Project Cleanup")
    print("=" * 50)
    
    # Perform cleanup operations
    cleanup_pycache()
    cleanup_logs()
    cleanup_temp_files()
    optimize_database_files()
    create_gitignore()
    
    # Display final stats
    display_project_stats()
    
    print("\n✅ Project cleanup completed!")
    print("\nNext steps:")
    print("  1. Review the cleaned project structure")
    print("  2. Run tests to ensure everything still works")
    print("  3. Commit the cleaned code")

if __name__ == "__main__":
    main()
