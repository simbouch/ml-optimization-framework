# 📊 Usage Guide

## Dashboard Overview

The Optuna Dashboard provides a comprehensive interface for exploring optimization studies. After starting the framework, access it at http://localhost:8080.

## Main Interface

### Study List
- **Left Panel**: Shows all 6 available studies
- **Search**: Filter studies by name
- **Sort**: Order by creation date or name
- **Study Info**: Shows direction (minimize/maximize) and trial count

### Study Selection
Click any study name to explore its optimization results in detail.

## Study Analysis Features

### 1. Optimization History
**What it shows**: Progress of optimization over trials
**How to use**: 
- X-axis: Trial number
- Y-axis: Objective value (accuracy, MSE, etc.)
- Look for convergence patterns
- Identify best performing trials

### 2. Parameter Importance
**What it shows**: Which hyperparameters have the most impact
**How to use**:
- Bars show relative importance (0-1 scale)
- Focus optimization on high-importance parameters
- Ignore parameters with very low importance

### 3. Parallel Coordinate Plot
**What it shows**: Relationship between parameters and objective
**How to use**:
- Each line represents one trial
- Color indicates objective value
- Identify parameter combinations that work well

### 4. Slice Plot
**What it shows**: How objective changes with individual parameters
**How to use**:
- Select parameter from dropdown
- See optimal ranges for each parameter
- Understand parameter sensitivity

### 5. Contour Plot
**What it shows**: 2D visualization of parameter interactions
**How to use**:
- Select two parameters
- See how they interact to affect objective
- Find optimal parameter combinations

## Study-Specific Guides

### 1. RandomForest Classification (TPE)
**Purpose**: Demonstrate TPE sampler effectiveness
**Key insights**:
- Watch how TPE focuses on promising regions
- Compare early vs late trials
- Note parameter importance ranking

**Parameters to watch**:
- `n_estimators`: Number of trees
- `max_depth`: Tree depth
- `min_samples_split`: Minimum samples for splitting

### 2. Gradient Boosting Regression (Random)
**Purpose**: Show random sampling approach
**Key insights**:
- Random exploration across parameter space
- Less focused than TPE
- Good baseline comparison

**Parameters to watch**:
- `learning_rate`: Step size
- `n_estimators`: Number of boosting rounds
- `max_depth`: Tree complexity

### 3. SVM Classification (Pruning)
**Purpose**: Demonstrate early stopping
**Key insights**:
- Some trials marked as "PRUNED"
- Computational efficiency gains
- Still finds good solutions

**Parameters to watch**:
- `C`: Regularization strength
- `gamma`: Kernel coefficient

### 4. Multi-objective Optimization
**Purpose**: Show trade-off optimization
**Key insights**:
- **Pareto Front**: Non-dominated solutions
- **Trade-offs**: Accuracy vs complexity
- **Multiple objectives**: No single "best" solution

**Special features**:
- Pareto front visualization
- Dominated vs non-dominated trials
- Trade-off analysis

### 5. Logistic Regression
**Purpose**: Simple model baseline
**Key insights**:
- Faster optimization
- Fewer parameters
- Good comparison point

### 6. RandomForest Regression
**Purpose**: Regression task demonstration
**Key insights**:
- Different metric (MSE vs accuracy)
- Similar patterns to classification
- Parameter importance differences

## Advanced Features

### Trial Details
- Click any trial number to see full details
- View all parameter values
- See intermediate values (for pruning studies)
- Check trial duration and status

### Export Data
- Download trial data as CSV
- Export plots as images
- Save study database for sharing

### Compare Studies
- Open multiple browser tabs
- Compare different samplers
- Analyze optimization strategies

## Best Practices

### 1. Start with Overview
- Look at optimization history first
- Check if optimization converged
- Identify best trial range

### 2. Analyze Parameters
- Check parameter importance
- Focus on high-impact parameters
- Understand parameter ranges

### 3. Look for Patterns
- Use parallel coordinates for patterns
- Check contour plots for interactions
- Identify optimal regions

### 4. Compare Approaches
- Compare TPE vs Random sampling
- Analyze pruning effectiveness
- Study multi-objective trade-offs

## Common Questions

**Q: Why do some trials show "PRUNED"?**
A: These trials were stopped early because they showed poor intermediate results. This saves computation time.

**Q: What's the difference between TPE and Random?**
A: TPE learns from previous trials to suggest better parameters. Random sampling explores uniformly.

**Q: How do I interpret parameter importance?**
A: Higher values mean the parameter has more impact on the objective. Focus optimization efforts on these parameters.

**Q: What is a Pareto front?**
A: In multi-objective optimization, it's the set of solutions where improving one objective requires worsening another.

**Q: Can I add my own studies?**
A: Yes! Modify `create_unified_demo.py` to add your own optimization problems.

## Next Steps

1. **Explore Each Study**: Spend time with each optimization study
2. **Compare Techniques**: Understand when to use different samplers
3. **Analyze Trade-offs**: Study the multi-objective optimization
4. **Customize Framework**: Add your own optimization problems
5. **Read API Docs**: Learn to build custom optimizers
