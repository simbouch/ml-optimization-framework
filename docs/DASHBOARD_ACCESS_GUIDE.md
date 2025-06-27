# Dashboard Access Guide

This guide provides comprehensive instructions for accessing and using both the Streamlit app and Optuna dashboard.

## 🚀 Quick Access Methods

### Method 1: Instant Launch (Recommended)
```bash
python start_both_services.py
```
This script will:
- ✅ Create demo studies if none exist
- ✅ Start Optuna Dashboard at http://localhost:8080
- ✅ Start Streamlit App at http://localhost:8501
- ✅ Open both URLs in your browser automatically

### Method 2: Docker Compose
```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs

# Stop services
docker-compose down
```

### Method 3: Manual Launch
```bash
# Terminal 1: Start Optuna Dashboard
optuna-dashboard sqlite:///studies/demo_ml.db --port 8080 --host 0.0.0.0

# Terminal 2: Start Streamlit App
streamlit run simple_app.py --server.port 8501 --server.address 0.0.0.0
```

## 📊 Dashboard Features

### Streamlit App (http://localhost:8501)

**Main Features:**
- 🎯 **Interactive Optimization**: Run optimizations with custom parameters
- 📊 **Study Management**: Create, load, and analyze studies
- 📈 **Real-time Visualization**: Live plots and progress tracking
- 💾 **Export Capabilities**: Download results in various formats
- 🔧 **System Status**: Monitor framework health and dependencies

**Key Sections:**
1. **Quick Start**: Basic optimization examples
2. **Study Browser**: View and manage existing studies
3. **Parameter Configuration**: Customize optimization settings
4. **Results Analysis**: Visualize and export results
5. **System Information**: Check framework status

### Optuna Dashboard (http://localhost:8080)

**Main Features:**
- 📈 **Optimization History**: Track progress over time
- 🎯 **Parameter Importance**: Identify key hyperparameters
- 📊 **Parallel Coordinates**: Multi-dimensional visualization
- 🔍 **Trial Details**: Individual trial analysis
- 📋 **Study Comparison**: Compare multiple optimization runs

**Key Views:**
1. **Study List**: Overview of all studies
2. **Optimization History**: Progress visualization
3. **Parameter Importance**: Feature analysis
4. **Parallel Coordinate Plot**: Multi-parameter view
5. **Trial Table**: Detailed trial information

## 🔧 Troubleshooting

### Common Issues

#### 1. "Port already in use" Error
```bash
# Check what's using the port
netstat -ano | findstr :8501
netstat -ano | findstr :8080

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

#### 2. "No study databases found"
```bash
# Create demo studies
python comprehensive_optuna_demo.py

# Or create minimal study
python -c "
import optuna
study = optuna.create_study(storage='sqlite:///studies/demo.db')
study.optimize(lambda trial: trial.suggest_float('x', -1, 1)**2, n_trials=5)
print('Demo study created!')
"
```

#### 3. "Module not found" Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### 4. Dashboard Shows Empty/No Data
- Ensure study databases exist in the `studies/` directory
- Check that the database files are not corrupted
- Verify the correct storage URL is being used

#### 5. Streamlit App Won't Start
```bash
# Check Streamlit installation
streamlit --version

# Try running with verbose output
streamlit run simple_app.py --server.port 8501 --logger.level debug
```

### Performance Tips

1. **For Large Studies**: Use database optimization
   ```bash
   # Optimize SQLite databases
   sqlite3 studies/your_study.db "VACUUM;"
   ```

2. **For Multiple Studies**: Use PostgreSQL for better performance
   ```bash
   # Example PostgreSQL connection
   optuna-dashboard postgresql://user:password@localhost:5432/optuna
   ```

3. **Memory Management**: Monitor resource usage
   - Close unused browser tabs
   - Restart services periodically for long-running optimizations

## 🎯 Usage Scenarios

### Scenario 1: Quick Demo
```bash
# 1. Start services
python start_both_services.py

# 2. Open Streamlit (http://localhost:8501)
# 3. Click "Create Demo Study" in sidebar
# 4. View results in Optuna Dashboard (http://localhost:8080)
```

### Scenario 2: Custom Optimization
```bash
# 1. Start services
python start_both_services.py

# 2. Run your optimization
python examples/basic_optimization.py

# 3. View results in both dashboards
```

### Scenario 3: Production Deployment
```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your settings

# 2. Deploy with Docker
docker-compose up -d

# 3. Access via configured ports
```

## 📱 Mobile Access

Both dashboards are responsive and work on mobile devices:
- **Streamlit**: Optimized mobile interface
- **Optuna Dashboard**: Touch-friendly navigation

Access from mobile using your computer's IP address:
- `http://YOUR_IP:8501` (Streamlit)
- `http://YOUR_IP:8080` (Optuna Dashboard)

## 🔒 Security Considerations

### Local Development
- Services bind to `0.0.0.0` for Docker compatibility
- Use `127.0.0.1` for localhost-only access

### Production Deployment
- Configure firewall rules
- Use reverse proxy (nginx/Apache)
- Enable HTTPS
- Set up authentication if needed

## 📚 Additional Resources

- **Framework Documentation**: See `docs/` directory
- **API Reference**: `docs/API_REFERENCE.md`
- **Advanced Usage**: `docs/ADVANCED_USAGE.md`
- **Comprehensive Tutorial**: `docs/COMPREHENSIVE_TUTORIAL.md`
- **Examples**: `examples/` directory

## 🆘 Getting Help

If you encounter issues:
1. Check this troubleshooting guide
2. Review the logs in the `logs/` directory
3. Run the validation script: `python validate_clean.py`
4. Check the GitHub repository for updates
5. Create an issue with detailed error information

---

**🎉 Happy Optimizing!** Both dashboards provide powerful tools for hyperparameter optimization and analysis.
