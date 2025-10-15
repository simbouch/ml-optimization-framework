# Optuna : L'outil qui va changer votre façon de faire du ML

Salut ! Alors, Optuna... c'est quoi exactement ? Je vais vous expliquer ça simplement.

## En gros, c'est quoi Optuna ?

Imaginez que vous avez un modèle Random Forest et que vous devez choisir :
- Combien d'arbres ? (n_estimators)
- Quelle profondeur ? (max_depth)
- Etc.

Normalement, vous testez à la main : 50 arbres, puis 100, puis 200... C'est long et chiant.

**Optuna fait ça automatiquement.** Vous lui dites "trouve-moi les meilleurs paramètres" et il le fait. Point.

## Le problème qu'on a tous

Quand je fais du ML, j'ai toujours ce problème :

```python
model = RandomForestClassifier(
    n_estimators=???,      # 10 ? 50 ? 100 ? 500 ?
    max_depth=???,         # 5 ? 10 ? 20 ? illimitée ?
    min_samples_split=???, # 2 ? 5 ? 10 ? 20 ?
    # ... et plein d'autres paramètres
)
```

Vous voyez le problème ? Il y a des MILLIARDS de combinaisons possibles. Même en testant une combinaison par seconde, il faudrait des années pour tout essayer.

## Ce qu'on faisait avant (et pourquoi c'est nul)

### Méthode 1 : Les valeurs par défaut
```python
model = RandomForestClassifier()  # On croise les doigts
```
**Problème :** Ça marche rarement bien sur vos données.

### Méthode 2 : Essai-erreur à la main
```python
# On teste à la main comme des sauvages
model1 = RandomForestClassifier(n_estimators=50)
model2 = RandomForestClassifier(n_estimators=100)
# ... 3 heures plus tard, on a testé 5 combinaisons
```
**Problème :** C'est long, pas systématique, et on rate sûrement le meilleur.

### Méthode 3 : Grid Search
```python
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [5, 10, 20, None]
}
# Teste TOUTES les combinaisons : 4 × 4 = 16 essais
```
**Problème :** Ça explose exponentiellement. Avec 5 paramètres, vous avez déjà des milliers de combinaisons.

## Pourquoi Optuna c'est génial

### 1. C'est BEAUCOUP plus rapide

**Avant (Grid Search) :**
- Je teste 1000 combinaisons
- Ça prend des jours
- Résultats moyens

**Avec Optuna :**
- Il teste intelligemment 50-100 combinaisons
- Ça prend quelques heures
- Résultats excellents
- **10 à 100 fois plus rapide !**

### 2. Il apprend de ses erreurs

Contrairement au Random Search qui teste au hasard, Optuna est intelligent :
- Il regarde les résultats précédents
- Il comprend quels paramètres marchent bien
- Il concentre ses efforts sur les zones prometteuses

C'est comme avoir un assistant qui apprend de vos expériences.

### 3. C'est super simple à utiliser

Regardez, voici tout le code dont vous avez besoin :

```python
import optuna

def objective(trial):
    # Optuna suggère des paramètres
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32)

    # Vous testez votre modèle
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    score = cross_val_score(model, X, y, cv=3).mean()

    return score

# Vous lancez l'optimisation
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Vous récupérez les meilleurs paramètres
print(f"Meilleurs paramètres : {study.best_params}")
```

C'est tout ! Optuna fait le reste.

## Les algorithmes derrière Optuna (pour les curieux)

### TPE (Tree-structured Parzen Estimator)

**Principe :** TPE modélise la distribution des hyperparamètres en fonction des performances passées.

**Comment ça marche :**
1. **Divise les trials** en deux groupes : bons résultats (top 20%) et mauvais résultats (80%)
2. **Modélise deux distributions** :
   - P(hyperparamètres | bon résultat)
   - P(hyperparamètres | mauvais résultat)
3. **Choisit les paramètres** qui maximisent le ratio P(bon)/P(mauvais)

**Avantage :** Plus il y a d'essais, plus TPE devient intelligent.

### Bayesian Optimization

**Principe :** Optuna utilise l'optimisation bayésienne pour équilibrer exploration et exploitation.

- **Exploration** : Tester des zones inconnues de l'espace des paramètres
- **Exploitation** : Se concentrer sur les zones prometteuses déjà découvertes

**Acquisition Function :** Fonction mathématique qui décide où chercher ensuite.

### Multi-objective Optimization

**Principe :** Optimiser plusieurs objectifs simultanément (ex: précision ET vitesse).

**Front de Pareto :** Ensemble des solutions où on ne peut améliorer un objectif sans dégrader l'autre.

```python
def objective(trial):
    model = create_model(trial)

    accuracy = evaluate_accuracy(model)
    inference_time = measure_speed(model)

    # Retourner les deux objectifs
    return accuracy, inference_time

# Créer une étude multi-objectifs
study = optuna.create_study(directions=['maximize', 'minimize'])
```

### 4. Le dashboard est magnifique

Optuna vous donne un dashboard web super clean où vous pouvez :
- Voir comment l'optimisation progresse
- Comprendre quels paramètres sont les plus importants
- Analyser les relations entre paramètres
- Comparer différentes expériences

C'est vraiment bien fait, vous allez voir.

### 5. Plein de fonctionnalités cool

- **Pruning** : Il arrête les essais pourris avant la fin (gain de temps énorme)
- **Multi-objectifs** : Vous pouvez optimiser précision ET vitesse en même temps
- **Parallélisation** : Il peut lancer plusieurs essais en parallèle
- **Sauvegarde auto** : Tout est sauvé, vous pouvez reprendre plus tard

## Les concepts de base (important à comprendre)

### Study (Étude)
C'est votre expérience d'optimisation complète. Vous créez une study pour chaque problème.

```python
study = optuna.create_study(direction='maximize')  # On veut maximiser le score
```

### Trial (Essai)
Chaque fois qu'Optuna teste une combinaison de paramètres, c'est un trial.

```python
def objective(trial):
    # Optuna va appeler cette fonction plein de fois
    # Chaque appel = 1 trial avec des paramètres différents
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2
```

### **3. Objective Function (Fonction Objectif)**

**Définition technique :** La fonction objectif est une fonction mathématique qui prend en entrée un ensemble d'hyperparamètres et retourne une métrique de performance à optimiser.

**En pratique :** C'est la fonction qu'Optuna va essayer d'optimiser. Vous lui dites "voici comment évaluer une combinaison de paramètres".

```python
def objective(trial):
    # 1. Optuna suggère des paramètres
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
        'max_depth': trial.suggest_int('max_depth', 2, 32)
    }

    # 2. Vous entraînez votre modèle avec ces paramètres
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # 3. Vous évaluez et retournez le score
    score = model.score(X_test, y_test)
    return score  # Optuna va essayer de maximiser ça
```

**Points importants :**
- La fonction doit être **déterministe** (même entrée = même sortie)
- Elle peut retourner une ou plusieurs valeurs (multi-objectifs)
- Plus la fonction est rapide, plus l'optimisation est efficace

### **4. Sampler (Échantillonneur)**

**Définition technique :** Un sampler est un algorithme qui détermine comment explorer l'espace des hyperparamètres pour trouver l'optimum global de manière efficace.

**En pratique :** C'est la stratégie qu'Optuna utilise pour choisir intelligemment les paramètres à tester.

```python
# TPE (Tree-structured Parzen Estimator) - Le plus intelligent
study = optuna.create_study(sampler=optuna.samplers.TPESampler())

# Random Sampler - Choix aléatoire (baseline)
study = optuna.create_study(sampler=optuna.samplers.RandomSampler())

# Grid Sampler - Teste toutes les combinaisons
study = optuna.create_study(sampler=optuna.samplers.GridSampler(...))

# CMA-ES - Optimisation évolutionnaire
study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())
```

**Comparaison des samplers :**
- **TPE** : Apprend des essais précédents, très efficace (recommandé)
- **Random** : Baseline simple, bon pour débuter
- **Grid** : Exhaustif mais lent, bon pour peu de paramètres
- **CMA-ES** : Excellent pour espaces continus, bon pour deep learning

### **5. Pruner (Élagueur)**

**Définition technique :** Un pruner analyse les résultats intermédiaires d'un trial en cours et décide s'il faut l'arrêter prématurément basé sur des critères statistiques.

**En pratique :** Il arrête les essais qui vont mal avant la fin. Ça économise énormément de temps (jusqu'à 70% !).

```python
# MedianPruner - Arrête si en dessous de la médiane
study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,    # Attendre 5 trials avant de commencer
        n_warmup_steps=10      # Attendre 10 étapes avant de pruner
    )
)

def objective(trial):
    model = create_model(trial)

    for epoch in range(100):
        train_loss = train_one_epoch(model)
        val_loss = validate(model)

        # Rapporter le score intermédiaire à Optuna
        trial.report(val_loss, epoch)

        # Le pruner décide s'il faut arrêter
        if trial.should_prune():
            raise optuna.TrialPruned()  # Arrêt précoce

    return final_val_loss
```

**Types de pruners :**
- **MedianPruner** : Arrête si en dessous de la médiane (recommandé)
- **PercentilePruner** : Arrête si en dessous d'un percentile
- **SuccessiveHalvingPruner** : Élimine progressivement les mauvais trials
- **HyperbandPruner** : Version avancée de Successive Halving

---

## ⚙️ Comment Fonctionne Optuna ?

### **Processus d'Optimisation**

```
1. Initialisation
   ↓
2. Suggérer des paramètres (Sampler)
   ↓
3. Évaluer la fonction objectif
   ↓
4. Enregistrer le résultat
   ↓
5. Apprendre des résultats précédents
   ↓
6. Répéter 2-5 jusqu'à n_trials
   ↓
7. Retourner les meilleurs paramètres
```

### **Algorithme TPE (Simplifié)**

```
Pour chaque nouveau trial :
1. Diviser les trials précédents en 2 groupes :
   - Bons résultats (top 20%)
   - Mauvais résultats (bottom 80%)

2. Modéliser la distribution des paramètres :
   - P(params | bon résultat)
   - P(params | mauvais résultat)

3. Choisir les paramètres qui maximisent :
   P(bon | params) / P(mauvais | params)

4. Tester ces paramètres

5. Mettre à jour les modèles
```

**Résultat :** Concentration progressive sur les zones prometteuses

---

## 🎯 Fonctionnalités Principales

### **1. Types de Paramètres Supportés**

```python
def objective(trial):
    # Entier
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    
    # Flottant
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    
    # Catégoriel
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    
    # Discret
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    
    return score
```

### **2. Optimisation Multi-objectifs**

```python
def objective(trial):
    params = {...}
    model = train_model(params)
    
    accuracy = evaluate_accuracy(model)
    complexity = count_parameters(model)
    
    # Retourner plusieurs objectifs
    return accuracy, complexity

# Créer une étude multi-objectifs
study = optuna.create_study(
    directions=['maximize', 'minimize']  # accuracy ↑, complexity ↓
)
```

### **3. Pruning (Arrêt Précoce)**

```python
def objective(trial):
    model = create_model(trial)
    
    for epoch in range(100):
        train_loss = train_one_epoch(model)
        val_loss = validate(model)
        
        # Rapporter le score intermédiaire
        trial.report(val_loss, epoch)
        
        # Arrêter si non prometteur
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return final_val_loss
```

**Avantage :** Économise jusqu'à 50-70% du temps de calcul

### **4. Callbacks et Monitoring**

```python
def callback(study, trial):
    if trial.value < best_threshold:
        print(f"Nouveau meilleur score : {trial.value}")

study.optimize(objective, n_trials=100, callbacks=[callback])
```

---

## 💻 Exemple Simple

### **Problème : Optimiser un Random Forest**

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

# Charger les données
X, y = load_iris(return_X_y=True)

# Définir la fonction objectif
def objective(trial):
    # Suggérer des hyperparamètres
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
        'max_depth': trial.suggest_int('max_depth', 2, 32),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
    }
    
    # Créer et évaluer le modèle
    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    
    return score

# Créer et lancer l'étude
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Afficher les résultats
print("Meilleurs paramètres :", study.best_params)
print("Meilleur score :", study.best_value)
```

**Résultat :**
```
Meilleurs paramètres : {
    'n_estimators': 156,
    'max_depth': 8,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}
Meilleur score : 0.973
```

---

## 🌍 Cas d'Usage Réels

### **1. Classification d'Images**
- Optimiser les hyperparamètres de CNN
- Trouver le meilleur learning rate
- Optimiser l'architecture du réseau

### **2. Traitement du Langage Naturel**
- Optimiser les modèles de transformers
- Trouver la meilleure taille de vocabulaire
- Optimiser les paramètres d'embedding

### **3. Séries Temporelles**
- Optimiser les modèles LSTM/GRU
- Trouver la meilleure fenêtre temporelle
- Optimiser les paramètres de prédiction

### **4. Systèmes de Recommandation**
- Optimiser les algorithmes de filtrage collaboratif
- Trouver les meilleurs paramètres de factorisation matricielle
- Optimiser les modèles hybrides

### **5. Détection d'Anomalies**
- Optimiser les seuils de détection
- Trouver les meilleurs paramètres d'isolation forest
- Optimiser les autoencodeurs

---

## ⭐ Avantages d'Optuna

### **Comparé à Grid Search**
- ✅ **10-100x plus rapide**
- ✅ Trouve de meilleurs résultats
- ✅ Pas besoin de définir une grille

### **Comparé à Random Search**
- ✅ **Apprend des essais précédents**
- ✅ Converge plus rapidement
- ✅ Résultats plus stables

### **Comparé à Hyperopt**
- ✅ API plus simple et intuitive
- ✅ Dashboard interactif intégré
- ✅ Meilleure documentation
- ✅ Développement plus actif

### **Comparé à Ray Tune**
- ✅ Plus léger et facile à installer
- ✅ Courbe d'apprentissage plus douce
- ✅ Parfait pour projets moyens

---

## 🚀 Démarrer avec Optuna

### **Installation**

```bash
pip install optuna
```

### **Premier Script**

```python
import optuna

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)

print(f"Meilleur x : {study.best_params['x']}")
print(f"Meilleure valeur : {study.best_value}")
```

### **Lancer le Dashboard**

```bash
optuna-dashboard sqlite:///optuna_study.db
```

Ouvrir : http://localhost:8080

---

## 📚 Ressources et Documentation

### **Documentation Officielle**
- Site web : https://optuna.org
- Documentation : https://optuna.readthedocs.io
- GitHub : https://github.com/optuna/optuna

### **Tutoriels**
- Tutorial officiel : https://optuna.readthedocs.io/en/stable/tutorial/
- Exemples : https://github.com/optuna/optuna-examples

### **Communauté**
- Discord : https://discord.gg/optuna
- Stack Overflow : Tag `optuna`

---

## 🎯 Conclusion

**Optuna est l'outil idéal pour :**
- ✅ Optimiser automatiquement vos modèles ML
- ✅ Gagner du temps (10-100x plus rapide que Grid Search)
- ✅ Obtenir de meilleurs résultats
- ✅ Visualiser et comprendre l'optimisation
- ✅ Intégrer facilement dans vos projets

**Commencez dès maintenant avec ce projet !**

---

**Prochaine étape : Faire les exercices pratiques !**

