# 👨‍🏫 Optuna Teaching Guide for Instructors
*Complete Guide for Teaching Hyperparameter Optimization with Optuna*

## 🎯 Overview

This guide is designed for instructors, team leads, and mentors who want to effectively teach Optuna and hyperparameter optimization to their colleagues, students, or team members.

## 📋 Pre-Class Preparation

### 🛠 Technical Setup (30 minutes)

#### 1. Verify Project Setup
```bash
# Clone and test the project
git clone <repository-url>
cd optimization_with_optuna

# Test Docker deployment
docker-compose up -d --build

# Verify dashboard at http://localhost:8080
curl -f http://localhost:8080 || echo "Setup issue detected"

# Clean up
docker-compose down
```

#### 2. Prepare Teaching Environment
- [ ] Ensure stable internet connection
- [ ] Test screen sharing/projection setup
- [ ] Prepare backup datasets (in case of download issues)
- [ ] Have troubleshooting commands ready

#### 3. Review Key Concepts
- [ ] Understand the 6 demonstration studies
- [ ] Review common student questions (see FAQ section)
- [ ] Prepare real-world examples from your domain
- [ ] Test all code examples in the tutorial

### 📚 Content Preparation

#### Essential Topics to Cover
1. **Why Hyperparameter Optimization Matters** (15 min)
2. **Optuna Core Concepts** (20 min)
3. **Hands-on Basic Example** (30 min)
4. **Advanced Features Overview** (25 min)
5. **Q&A and Troubleshooting** (15 min)

#### Backup Materials
- Alternative datasets for different domains
- Simplified examples for struggling students
- Advanced challenges for quick learners
- Real-world case studies from your organization

## 🎓 Suggested Teaching Schedules

### 📅 Option 1: Single Workshop (2 hours)

#### **Session Structure**
```
0:00-0:15  Introduction & Motivation
0:15-0:35  Core Concepts & Theory
0:35-1:05  Hands-on Basic Example
1:05-1:15  Break
1:15-1:40  Advanced Features Demo
1:40-1:55  Practice Exercise
1:55-2:00  Wrap-up & Next Steps
```

#### **Detailed Timeline**

**Introduction & Motivation (15 min)**
- Problem: Manual hyperparameter tuning is inefficient
- Solution: Automated optimization with Optuna
- Show before/after results from the dashboard

**Core Concepts (20 min)**
- Studies, trials, objective functions
- Samplers (TPE vs Random)
- Parameter types (int, float, categorical)
- Live demo of basic optimization

**Hands-on Basic Example (30 min)**
- Students follow along with provided notebook
- Implement Random Forest optimization
- Discuss results and parameter importance

**Advanced Features Demo (25 min)**
- Pruning for efficiency
- Multi-objective optimization
- Study persistence and analysis

**Practice Exercise (15 min)**
- Students modify the basic example
- Try different parameter ranges
- Compare results with neighbors

### 📅 Option 2: Three-Session Course (6 hours total)

#### **Session 1: Foundations (2 hours)**
- Deep dive into hyperparameter optimization theory
- Comprehensive hands-on with basic examples
- Setup personal optimization projects

#### **Session 2: Advanced Features (2 hours)**
- Pruning and efficiency optimization
- Multi-objective optimization
- Custom samplers and advanced techniques

#### **Session 3: Real-World Applications (2 hours)**
- Production pipeline integration
- Case studies and best practices
- Student project presentations

### 📅 Option 3: Self-Paced Learning Track (3 weeks)

#### **Week 1: Basics**
- Complete tutorial sections 1-6
- Implement Practice Project 1
- Weekly check-in meeting (30 min)

#### **Week 2: Intermediate**
- Complete tutorial sections 7-9
- Implement Practice Projects 2-3
- Peer review session (45 min)

#### **Week 3: Advanced**
- Complete tutorial sections 10-13
- Implement Practice Projects 4-5
- Final presentation (60 min)

## 🎯 Interactive Exercises for Class

### 🔬 Exercise 1: Parameter Range Impact
**Time**: 15 minutes  
**Goal**: Understand how parameter ranges affect optimization

```python
# Give students this template
def range_experiment(trial):
    # Students experiment with different ranges
    n_estimators = trial.suggest_int('n_estimators', ?, ?)  # Fill in
    max_depth = trial.suggest_int('max_depth', ?, ?)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        random_state=42
    )
    return cross_val_score(model, X, y, cv=3).mean()

# Discussion questions:
# 1. What happens with very narrow ranges (e.g., 10-15)?
# 2. What happens with very wide ranges (e.g., 1-1000)?
# 3. How does the number of trials affect results?
```

**Teaching Tips**:
- Have students work in pairs
- Ask them to predict results before running
- Discuss why extreme ranges can be problematic

### 🏆 Exercise 2: Sampler Comparison Challenge
**Time**: 20 minutes  
**Goal**: Compare different optimization algorithms

```python
# Challenge: Which sampler works best?
samplers = {
    'TPE': optuna.samplers.TPESampler(seed=42),
    'Random': optuna.samplers.RandomSampler(seed=42),
    'CMA-ES': optuna.samplers.CmaEsSampler(seed=42)
}

# Students implement comparison and discuss results
```

**Teaching Tips**:
- Make it competitive between groups
- Have each group present their findings
- Discuss when each sampler might be preferred

### 🎪 Exercise 3: Real-World Scenario
**Time**: 25 minutes  
**Goal**: Apply optimization to realistic constraints

```python
"""
Scenario: You're deploying a model to production
Constraints:
- Prediction time must be < 100ms
- Model accuracy should be maximized
- Model size should be reasonable

Design a multi-objective optimization for this scenario.
"""
```

**Teaching Tips**:
- Use scenarios relevant to your organization
- Encourage creative constraint definitions
- Discuss trade-off decisions

## ❓ Common Student Questions & Answers

### **Q: "How many trials should I run?"**
**A**: Start with 50-100 for learning. In practice:
- Simple problems: 50-200 trials
- Complex problems: 200-1000+ trials
- Use optimization history plots to see diminishing returns

**Teaching Tip**: Show optimization history plots to illustrate convergence

### **Q: "Why is TPE better than random search?"**
**A**: TPE learns from previous trials and focuses on promising regions.

**Teaching Tip**: Show side-by-side optimization plots comparing TPE vs Random

### **Q: "When should I use pruning?"**
**A**: When training is expensive and you can evaluate intermediate performance.

**Teaching Tip**: Demonstrate time savings with a slow model (neural network)

### **Q: "How do I choose the final solution in multi-objective optimization?"**
**A**: Consider business constraints, risk tolerance, and future requirements.

**Teaching Tip**: Use real business scenarios to make this concrete

### **Q: "What if my optimization gets stuck in local optima?"**
**A**: Try different samplers, increase trials, or adjust parameter ranges.

**Teaching Tip**: Show examples of poor parameter ranges leading to suboptimal results

## 🔧 Troubleshooting Guide

### Common Technical Issues

#### **Docker Won't Start**
```bash
# Quick fixes
docker-compose down
docker system prune -f
docker-compose up -d --build

# Check port conflicts
netstat -an | grep 8080
```

#### **Dashboard Shows No Data**
```bash
# Verify database exists
ls -la studies/
sqlite3 studies/unified_demo.db ".tables"

# Restart services
docker-compose restart
```

#### **Slow Optimization**
```python
# Reduce dataset size for demos
X_small, _, y_small, _ = train_test_split(X, y, train_size=0.1)

# Use fewer CV folds
scores = cross_val_score(model, X, y, cv=3)  # Instead of cv=5

# Reduce trial count
study.optimize(objective, n_trials=20)  # Instead of 100
```

#### **Import Errors**
```bash
# Verify environment
python -c "import optuna, pandas, numpy, sklearn; print('All imports OK')"

# Reinstall if needed
pip install -r requirements-minimal.txt
```

### Pedagogical Challenges

#### **Students with Different Skill Levels**
- **Beginners**: Provide simplified examples, focus on concepts
- **Advanced**: Give additional challenges, encourage exploration
- **Mixed Groups**: Use pair programming with mixed skill pairs

#### **Time Management**
- **Running Behind**: Skip advanced features, focus on core concepts
- **Ahead of Schedule**: Add bonus exercises or deeper discussions
- **Technical Delays**: Have backup slides ready, use pre-run examples

#### **Engagement Issues**
- **Low Participation**: Use smaller groups, ask direct questions
- **Confusion**: Stop and review concepts, use analogies
- **Boredom**: Add competitive elements, real-world examples

## 📊 Assessment Ideas

### 🎯 Beginner Assessment (30 minutes)
1. **Basic Implementation** (15 min)
   - Implement optimization for provided dataset
   - Explain parameter choices

2. **Concept Questions** (10 min)
   - Difference between TPE and random sampling
   - When to use different parameter types

3. **Results Interpretation** (5 min)
   - Analyze optimization history plot
   - Identify best parameters

### 🎯 Intermediate Assessment (45 minutes)
1. **Multi-Objective Problem** (25 min)
   - Design optimization with trade-offs
   - Analyze Pareto front

2. **Efficiency Optimization** (15 min)
   - Add pruning to reduce computation
   - Compare efficiency gains

3. **Troubleshooting** (5 min)
   - Debug provided broken code
   - Explain the fix

### 🎯 Advanced Assessment (60 minutes)
1. **Production Pipeline** (35 min)
   - Design end-to-end optimization
   - Include preprocessing and model selection

2. **Custom Implementation** (20 min)
   - Implement custom objective function
   - Handle domain-specific constraints

3. **Presentation** (5 min)
   - Present optimization strategy
   - Justify design decisions

## 🎓 Certification and Follow-up

### Learning Outcomes
After completing the course, students should be able to:
- [ ] Implement basic hyperparameter optimization with Optuna
- [ ] Choose appropriate samplers and pruners for different scenarios
- [ ] Design multi-objective optimization for real-world trade-offs
- [ ] Integrate optimization into production ML pipelines
- [ ] Troubleshoot common optimization issues

### Continuing Education
- **Advanced Topics**: Custom samplers, distributed optimization
- **Domain Applications**: NLP, computer vision, time series
- **Research**: Latest optimization algorithms and techniques
- **Community**: Encourage participation in Optuna community

### Resources for Further Learning
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Practice Projects Guide](practice_projects.md)
- [Advanced Examples](../examples/)
- [Research Papers](https://scholar.google.com/scholar?q=optuna+hyperparameter+optimization)

## 📝 Teaching Checklist

### Before Class
- [ ] Test all technical setup
- [ ] Review student backgrounds and adjust content
- [ ] Prepare backup materials and examples
- [ ] Set up breakout rooms (if virtual)

### During Class
- [ ] Start with motivation and real examples
- [ ] Encourage questions and interaction
- [ ] Use live coding when possible
- [ ] Check understanding frequently
- [ ] Provide individual help as needed

### After Class
- [ ] Share resources and next steps
- [ ] Collect feedback for improvement
- [ ] Follow up with struggling students
- [ ] Plan advanced sessions if interest exists

---

**Happy Teaching! 🎓**

*Remember: The best way to learn Optuna is through hands-on practice. Encourage experimentation and don't be afraid to let students explore beyond the planned curriculum.*
