# Exercice Pratique : Créer Votre Premier Projet Optuna

Cet exercice vous guide pas à pas pour créer un projet d'optimisation d'hyperparamètres avec Optuna.

## Objectif

Créer un projet complet d'optimisation pour un modèle de classification, en utilisant Optuna pour trouver les meilleurs hyperparamètres.

## Durée Estimée

2-3 heures

## Prérequis

- Python 3.9+ installé
- Connaissances de base en machine learning
- Compréhension des concepts Optuna (voir PRESENTATION_OPTUNA.md)

## Étape 1 : Préparation de l'Environnement

### 1.1 Créer un dossier pour le projet

```powershell
# Créer un nouveau dossier
mkdir mon_projet_optuna
cd mon_projet_optuna
```

### 1.2 Installer les dépendances

```powershell
# Installer les bibliothèques nécessaires
pip install optuna scikit-learn pandas numpy matplotlib
```

## Étape 2 : Préparer les Données

### 2.1 Créer le fichier `prepare_data.py`

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data():
    """Charger et préparer les données"""
    # Charger le dataset Breast Cancer
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print(f"Données d'entraînement : {X_train.shape}")
    print(f"Données de test : {X_test.shape}")
```

### 2.2 Tester le chargement des données

```powershell
python prepare_data.py
```

**Résultat attendu :**
```
Données d'entraînement : (455, 30)
Données de test : (114, 30)
```

## Étape 3 : Créer la Fonction Objectif

### 3.1 Créer le fichier `optimize.py`

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from prepare_data import load_data

def objective(trial):
    """
    Fonction objectif pour Optuna
    
    Cette fonction :
    1. Suggère des hyperparamètres
    2. Crée un modèle avec ces hyperparamètres
    3. Évalue le modèle
    4. Retourne le score à optimiser
    """
    # Charger les données
    X_train, X_test, y_train, y_test = load_data()
    
    # Suggérer des hyperparamètres
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
    # Créer le modèle avec les hyperparamètres suggérés
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    
    # Évaluer le modèle avec validation croisée
    scores = cross_val_score(
        model, X_train, y_train, 
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    # Retourner le score moyen
    return scores.mean()

if __name__ == "__main__":
    # Créer une étude Optuna
    study = optuna.create_study(
        study_name='breast_cancer_optimization',
        direction='maximize',  # Maximiser la précision
        storage='sqlite:///optuna_study.db',
        load_if_exists=True
    )
    
    # Lancer l'optimisation
    study.optimize(objective, n_trials=50)
    
    # Afficher les résultats
    print("\nMeilleurs hyperparamètres :")
    print(study.best_params)
    print(f"\nMeilleur score : {study.best_value:.4f}")
```

### 3.2 Lancer l'optimisation

```powershell
python optimize.py
```

**Ce qui se passe :**
- Optuna teste 50 combinaisons d'hyperparamètres
- Chaque combinaison est évaluée avec validation croisée
- Les meilleurs paramètres sont sauvegardés

## Étape 4 : Visualiser les Résultats

### 4.1 Créer le fichier `visualize.py`

```python
import optuna
import matplotlib.pyplot as plt

# Charger l'étude
study = optuna.load_study(
    study_name='breast_cancer_optimization',
    storage='sqlite:///optuna_study.db'
)

# 1. Historique d'optimisation
fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title("Historique d'Optimisation")
plt.tight_layout()
plt.savefig('optimization_history.png')
plt.close()

# 2. Importance des paramètres
fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
plt.title("Importance des Paramètres")
plt.tight_layout()
plt.savefig('param_importances.png')
plt.close()

# 3. Graphique de contour
fig3 = optuna.visualization.matplotlib.plot_contour(study)
plt.tight_layout()
plt.savefig('contour_plot.png')
plt.close()

print("Visualisations sauvegardées :")
print("- optimization_history.png")
print("- param_importances.png")
print("- contour_plot.png")
```

### 4.2 Générer les visualisations

```powershell
python visualize.py
```

## Étape 5 : Utiliser le Dashboard Optuna

### 5.1 Lancer le dashboard

```powershell
optuna-dashboard sqlite:///optuna_study.db
```

### 5.2 Accéder au dashboard

Ouvrez votre navigateur : http://localhost:8080

### 5.3 Explorer les visualisations

Dans le dashboard, vous pouvez :
- Voir l'historique des trials
- Analyser l'importance des paramètres
- Explorer les relations entre paramètres
- Comparer les différents trials

## Étape 6 : Entraîner le Modèle Final

### 6.1 Créer le fichier `train_final.py`

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import optuna
from prepare_data import load_data

# Charger les données
X_train, X_test, y_train, y_test = load_data()

# Charger l'étude
study = optuna.load_study(
    study_name='breast_cancer_optimization',
    storage='sqlite:///optuna_study.db'
)

# Récupérer les meilleurs hyperparamètres
best_params = study.best_params

# Créer le modèle final
model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42,
    n_jobs=-1
)

# Entraîner le modèle
model.fit(X_train, y_train)

# Évaluer sur l'ensemble de test
y_pred = model.predict(X_test)

print("Rapport de Classification :")
print(classification_report(y_test, y_pred))

print("\nMatrice de Confusion :")
print(confusion_matrix(y_test, y_pred))
```

### 6.2 Entraîner et évaluer

```powershell
python train_final.py
```

## Étape 7 : Défis Supplémentaires

Une fois l'exercice de base terminé, essayez ces défis :

### Défi 1 : Ajouter le Pruning
Modifiez `optimize.py` pour ajouter le pruning et arrêter les trials non prometteurs.

### Défi 2 : Comparer Plusieurs Algorithmes
Créez une fonction objectif qui teste Random Forest, Gradient Boosting et SVM.

### Défi 3 : Optimisation Multi-Objectifs
Optimisez à la fois la précision et le temps d'entraînement.

### Défi 4 : Utiliser Vos Propres Données
Remplacez le dataset Breast Cancer par vos propres données.

## Résumé de l'Exercice

Vous avez appris à :

1. Préparer des données pour l'optimisation
2. Créer une fonction objectif Optuna
3. Lancer une étude d'optimisation
4. Visualiser les résultats
5. Utiliser le dashboard Optuna
6. Entraîner un modèle final avec les meilleurs paramètres

## Structure Finale du Projet

```
mon_projet_optuna/
├── prepare_data.py           # Chargement des données
├── optimize.py               # Optimisation Optuna
├── visualize.py              # Visualisations
├── train_final.py            # Modèle final
├── optuna_study.db           # Base de données Optuna
├── optimization_history.png  # Graphique 1
├── param_importances.png     # Graphique 2
└── contour_plot.png          # Graphique 3
```

## Prochaines Étapes

1. Expérimentez avec différents algorithmes
2. Testez différents samplers (TPE, Random, CMA-ES)
3. Ajoutez le pruning pour accélérer l'optimisation
4. Créez des optimisations multi-objectifs
5. Appliquez Optuna à vos propres projets ML

## Ressources

- Documentation Optuna : https://optuna.readthedocs.io
- Exemples Optuna : https://github.com/optuna/optuna-examples
- Scikit-learn : https://scikit-learn.org

