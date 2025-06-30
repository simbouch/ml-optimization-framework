# 📊 Study Details

## Overview

The framework creates 6 different optimization studies to demonstrate various Optuna capabilities. Each study uses different algorithms, samplers, and techniques.

## Study 1: RandomForest Classification (TPE)

### Purpose
Demonstrate the effectiveness of TPE (Tree-structured Parzen Estimator) sampling for hyperparameter optimization.

### Configuration
- **Model**: Random Forest Classifier
- **Dataset**: Synthetic classification (1000 samples, 20 features)
- **Sampler**: TPE Sampler
- **Objective**: Maximize accuracy
- **Trials**: 30

### Parameters Optimized
- `n_estimators` (10-200): Number of trees in the forest
- `max_depth` (3-20): Maximum depth of trees
- `min_samples_split` (2-20): Minimum samples required to split

### Key Learning Points
- **TPE Intelligence**: Watch how TPE focuses on promising parameter regions
- **Convergence**: Notice how optimization improves over trials
- **Parameter Importance**: See which parameters matter most

### Expected Results
- Best accuracy: ~0.85-0.95
- TPE should outperform random sampling
- `n_estimators` and `max_depth` typically most important

## Study 2: Gradient Boosting Regression (Random)

### Purpose
Compare random sampling approach and demonstrate gradient boosting optimization.

### Configuration
- **Model**: Gradient Boosting Regressor
- **Dataset**: Synthetic regression (1000 samples, 20 features)
- **Sampler**: Random Sampler
- **Objective**: Minimize MSE (Mean Squared Error)
- **Trials**: 25

### Parameters Optimized
- `n_estimators` (50-200): Number of boosting rounds
- `max_depth` (3-10): Maximum tree depth
- `learning_rate` (0.01-0.3): Step size shrinkage
- `subsample` (0.6-1.0): Subsample ratio of training instances

### Key Learning Points
- **Random Exploration**: Uniform exploration across parameter space
- **Gradient Boosting Behavior**: How boosting parameters interact
- **Regression Metrics**: Different from classification (MSE vs accuracy)

### Expected Results
- Best MSE: Varies with random seed
- More scattered optimization than TPE
- `learning_rate` and `n_estimators` often important

## Study 3: SVM Classification (Pruning)

### Purpose
Demonstrate pruning capabilities for computational efficiency.

### Configuration
- **Model**: Support Vector Machine
- **Dataset**: Synthetic classification (1000 samples, 20 features)
- **Sampler**: TPE Sampler
- **Pruner**: Median Pruner (stops unpromising trials early)
- **Objective**: Maximize accuracy
- **Trials**: 20

### Parameters Optimized
- `C` (1e-3 to 1e3, log scale): Regularization parameter
- `gamma` (categorical): Kernel coefficient ('scale' or 'auto')

### Key Learning Points
- **Pruning Benefits**: Some trials stop early to save computation
- **Efficiency**: Faster optimization through early stopping
- **SVM Behavior**: How regularization affects performance

### Expected Results
- Some trials marked as "PRUNED"
- Faster overall optimization
- Still achieves good accuracy despite early stopping

## Study 4: Multi-objective Optimization

### Purpose
Demonstrate multi-objective optimization with trade-off analysis.

### Configuration
- **Model**: Random Forest Classifier
- **Dataset**: Synthetic classification (1000 samples, 20 features)
- **Sampler**: TPE Multi-objective Sampler
- **Objectives**: 
  - Maximize accuracy
  - Minimize model complexity (n_estimators × max_depth)
- **Trials**: 25

### Parameters Optimized
- `n_estimators` (10-200): Number of trees
- `max_depth` (3-20): Maximum tree depth

### Key Learning Points
- **Pareto Front**: Set of optimal trade-off solutions
- **Trade-offs**: High accuracy vs low complexity
- **Multi-objective Thinking**: No single "best" solution

### Expected Results
- Multiple non-dominated solutions
- Clear trade-off between accuracy and complexity
- Pareto front visualization available

## Study 5: Logistic Regression Comparison

### Purpose
Provide a simple baseline model for comparison.

### Configuration
- **Model**: Logistic Regression
- **Dataset**: Synthetic classification (1000 samples, 20 features)
- **Sampler**: TPE Sampler
- **Objective**: Maximize accuracy
- **Trials**: 20

### Parameters Optimized
- `C` (1e-4 to 1e2, log scale): Inverse regularization strength
- `solver` (categorical): Algorithm ('liblinear', 'lbfgs')
- `max_iter` (100-1000): Maximum iterations

### Key Learning Points
- **Simple Models**: Sometimes perform surprisingly well
- **Baseline Comparison**: Compare with complex models
- **Fast Optimization**: Fewer parameters, faster trials

### Expected Results
- Good baseline accuracy
- Fast optimization convergence
- Simpler parameter relationships

## Study 6: RandomForest Regression

### Purpose
Demonstrate regression optimization and compare with classification.

### Configuration
- **Model**: Random Forest Regressor
- **Dataset**: Synthetic regression (1000 samples, 20 features)
- **Sampler**: TPE Sampler
- **Objective**: Minimize MSE
- **Trials**: 25

### Parameters Optimized
- `n_estimators` (10-200): Number of trees
- `max_depth` (3-20): Maximum tree depth
- `min_samples_split` (2-20): Minimum samples for splitting

### Key Learning Points
- **Regression vs Classification**: Different metrics and behaviors
- **Parameter Patterns**: Similar to classification but different optimal values
- **MSE Interpretation**: Lower is better (opposite of accuracy)

### Expected Results
- Best MSE varies with dataset
- Similar parameter importance to classification
- Different optimal parameter values

## Comparative Analysis

### Sampler Comparison
- **TPE Studies** (1, 3, 4, 5, 6): Show intelligent parameter selection
- **Random Study** (2): Shows uniform exploration
- **Multi-objective** (4): Shows trade-off optimization

### Model Comparison
- **Tree-based** (1, 2, 6): Random Forest and XGBoost
- **Linear** (5): Logistic Regression
- **Kernel** (3): Support Vector Machine

### Technique Comparison
- **Standard Optimization** (1, 2, 5, 6): Single objective
- **Pruning** (3): Early stopping for efficiency
- **Multi-objective** (4): Trade-off optimization

## Using the Studies

### For Learning
1. **Start with Study 5** (Logistic Regression): Simplest
2. **Compare Studies 1 & 2**: TPE vs Random sampling
3. **Explore Study 3**: Understand pruning benefits
4. **Analyze Study 4**: Multi-objective trade-offs

### For Research
- Use as baselines for your own optimization problems
- Compare different sampling strategies
- Study parameter importance patterns
- Analyze convergence behaviors

### For Development
- Modify parameters and ranges
- Add new models and datasets
- Experiment with different samplers
- Create custom objective functions

## Next Steps

1. **Explore Each Study**: Use the dashboard to analyze results
2. **Compare Techniques**: Understand when to use each approach
3. **Modify Studies**: Change parameters or add new ones
4. **Create Custom Studies**: Build your own optimization problems
