# 🔧 Setup Guide

## Prerequisites

- **Docker Desktop** installed and running
- **Docker Compose** (usually included with Docker Desktop)
- **2-3 minutes** for initial setup

## Quick Setup

### 1. Clone Repository
```bash
git clone https://github.com/simbouch/ml-optimization-framework.git
cd ml-optimization-framework
```

### 2. Start Framework
```bash
docker-compose up -d --build
```

### 3. Wait for Completion
The framework will:
- Build the Docker container (2-3 minutes first time)
- Create 6 optimization studies automatically
- Start the Optuna dashboard

### 4. Access Dashboard
Open http://localhost:8080 in your browser

## What Happens During Setup

### Container Build Process
1. **Base Image**: Python 3.10 slim
2. **Dependencies**: Installs Optuna, scikit-learn, XGBoost, etc.
3. **Framework Code**: Copies source code and demo scripts
4. **Study Creation**: Runs unified demo to create 6 studies
5. **Dashboard Start**: Launches Optuna dashboard on port 8080

### Studies Created
1. **RandomForest_Classification_TPE** (30 trials)
2. **XGBoost_Regression_Random** (25 trials)
3. **SVM_Classification_Pruning** (20 trials)
4. **MultiObjective_Accuracy_vs_Complexity** (25 trials)
5. **LogisticRegression_Comparison** (20 trials)
6. **RandomForest_Regression** (25 trials)

## Verification

### Check Container Status
```bash
docker-compose ps
```
Should show container as "Up" and "healthy"

### Check Logs
```bash
docker-compose logs
```
Should show successful study creation and dashboard startup

### Test Dashboard
1. Open http://localhost:8080
2. You should see 6 studies in the study list
3. Click any study to explore optimization results

## Troubleshooting

### Container Won't Start
```bash
# Check Docker is running
docker --version
docker-compose --version

# Check port availability
netstat -an | grep 8080

# Restart Docker Desktop if needed
```

### Dashboard Not Loading
```bash
# Wait longer (demos take 2-3 minutes)
docker-compose logs -f

# Check if studies were created
docker-compose exec ml-optimization ls -la studies/
```

### Build Errors
```bash
# Clean rebuild
docker-compose down -v
docker-compose up -d --build --no-cache
```

## Manual Setup (Alternative)

If Docker isn't available, you can run locally:

### 1. Install Python Dependencies
```bash
pip install -r requirements-minimal.txt
```

### 2. Create Studies
```bash
python create_unified_demo.py
```

### 3. Start Dashboard
```bash
optuna-dashboard sqlite:///studies/unified_demo.db --host 0.0.0.0 --port 8080
```

## Next Steps

Once setup is complete:
1. **Explore Studies**: Click through different optimization studies
2. **Analyze Results**: Use parameter importance and optimization plots
3. **Read Documentation**: Check other docs for detailed usage
4. **Customize**: Modify `create_unified_demo.py` for your own studies
