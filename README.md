# 🎯 ML Optimization Framework with Optuna

**Educational project demonstrating Optuna's hyperparameter optimization capabilities through interactive examples and comprehensive tutorials**

## 🎓 **Educational Purpose**

This project is designed to **teach Optuna** - the automatic hyperparameter optimization framework. Perfect for:
- **Learning Optuna**: From basics to advanced features
- **Team Training**: Share with colleagues to learn optimization
- **Demonstrations**: Show Optuna's capabilities in action
- **Practice**: Hands-on exercises and real examples

### 🎯 **What Your Colleagues Will Learn**
- **Core Concepts**: Studies, trials, samplers, pruners
- **Practical Skills**: Writing objective functions, parameter tuning
- **Advanced Features**: Multi-objective optimization, pruning, visualization
- **Best Practices**: Cross-validation, parameter importance, production tips
- **Real Examples**: Complete working code they can modify and use

## 🚀 Quick Start

### **Option 1: Interactive Tutorial Launcher (Recommended)**
```powershell
.\start-tutorial.ps1
```
Choose from:
- 🚀 Interactive Dashboard with 6 studies
- 📖 Complete colleague tutorial example
- ⚡ Quick 30-second demo
- 📋 Documentation browser

### **Option 2: Direct Dashboard Access**
```bash
docker-compose up -d --build
```
**Then open:** http://localhost:8080

### **Option 3: Colleague Tutorial Example**
```bash
python optuna_colleague_example.py
```
Perfect standalone example to share with team members!

## 📊 What You'll See

**6 Different Optimization Studies:**
1. **RandomForest Classification (TPE)** - Smart hyperparameter optimization
2. **XGBoost Regression (Random)** - Random sampling comparison  
3. **SVM Classification (Pruning)** - Early stopping demonstration
4. **Multi-objective Optimization** - Accuracy vs Complexity trade-offs
5. **Logistic Regression** - Simple model baseline
6. **RandomForest Regression** - Regression task optimization

**Interactive Dashboard Features:**
- 📈 Optimization history plots
- 🎯 Parameter importance analysis
- 📊 Parallel coordinate plots
- 🔍 Trial details and comparisons
- 📋 Pareto front visualization (multi-objective)

## 🌟 About Optuna

**Optuna** is an automatic hyperparameter optimization framework for machine learning. This project demonstrates:

- **TPE Sampling**: Tree-structured Parzen Estimator for intelligent optimization
- **Pruning**: Early stopping of unpromising trials
- **Multi-objective**: Pareto frontier optimization
- **Visualization**: Rich interactive dashboards

## 🔧 Technical Details

**Optimization Techniques:**
- TPE (Tree-structured Parzen Estimator) sampling
- Random sampling for comparison
- Median pruning for early stopping
- Multi-objective optimization with Pareto fronts

**Machine Learning Models:**
- Random Forest (Classification & Regression)
- XGBoost (Regression)
- Support Vector Machine (Classification)
- Logistic Regression (Classification)

**Infrastructure:**
- Docker containerization
- SQLite database storage
- Optuna dashboard visualization
- Automated demo execution

## 📋 Requirements

- Docker and Docker Compose
- 2-3 minutes for initial setup

## 🎯 Usage

### Start the Framework
```bash
docker-compose up -d --build
```

### Access Dashboard
Open http://localhost:8080 in your browser

### Stop the Framework
```bash
docker-compose down
```

## 📚 Documentation & Learning

### 🎯 **Start Here: Complete Optuna Tutorial**
- **[📖 Optuna Tutorial](docs/tutorial.md)** - **Complete guide from basics to advanced**
  - What is Optuna and why use it?
  - Core concepts and terminology
  - Step-by-step examples
  - Hands-on exercises for colleagues
  - Real-world practice examples

### 📋 **Project Documentation**
- **[Setup Guide](docs/setup.md)** - Installation and configuration
- **[User Guide](docs/usage.md)** - How to use the dashboard
- **[Study Details](docs/studies.md)** - Explanation of each optimization study
- **[API Reference](docs/api.md)** - Technical implementation details

## 🏗️ Project Structure

```
ml-optimization-framework/
├── create_unified_demo.py    # Creates 6 optimization studies
├── docker-compose.yml        # Docker deployment configuration
├── Dockerfile               # Container definition
├── requirements-minimal.txt  # Python dependencies
├── src/                     # Framework source code
├── examples/                # Example optimization scripts
├── docs/                    # Detailed documentation
└── studies/                 # Generated study databases
```

## 🎓 Learning Objectives

This framework demonstrates:
- **TPE vs Random Sampling**: Compare intelligent vs random optimization
- **Pruning Benefits**: Early stopping for computational efficiency  
- **Multi-objective Trade-offs**: Accuracy vs model complexity
- **Real ML Scenarios**: Practical hyperparameter optimization
- **Optuna Best Practices**: Professional optimization workflows

## 🔧 Troubleshooting

**Dashboard not loading?**
- Wait 2-3 minutes for demos to complete
- Check container status: `docker-compose ps`
- View logs: `docker-compose logs`

**Port already in use?**
- Stop existing containers: `docker-compose down`
- Check what's using port 8080: `netstat -an | grep 8080`

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

---

**🎯 Ready to explore Optuna optimization? Start with `docker-compose up -d --build`**
