# Le Projet que j'ai fait pour vous

Alors, qu'est-ce que j'ai préparé exactement ?

## En gros, c'est quoi

J'ai créé un projet complet avec Optuna qui montre 6 exemples différents d'optimisation. L'idée c'est que vous puissiez voir concrètement comment ça marche sur de vrais problèmes ML.

## Pourquoi j'ai fait ça

Franchement, quand j'ai découvert Optuna, j'ai galéré à comprendre comment bien l'utiliser. La doc officielle est bien mais c'est toujours mieux d'avoir des exemples concrets.

Donc j'ai fait ce projet pour :
- Vous montrer Optuna en action sur de vrais cas
- Avoir des exemples que vous pouvez réutiliser dans vos projets
- Vous éviter de galérer comme moi au début
- Vous donner de bonnes pratiques que j'ai apprises

## Comment c'est organisé

J'ai structuré le projet comme ça :

```
optimization_with_optuna/
├── src/                          # Le code principal
│   ├── config.py                 # Configuration
│   ├── optimizers.py             # Les optimiseurs ML
│   └── study_manager.py          # Gestion des études Optuna
├── examples/                     # Exemples d'utilisation
├── tests/                        # Tests (oui, j'ai fait des tests !)
├── studies/                      # Base de données SQLite
├── tutorial_octobre_2025_french/ # Cette doc que vous lisez
├── docker-compose.yml            # Pour lancer avec Docker
└── requirements-minimal.txt      # Les dépendances Python
```

## Les parties importantes

### Le code source (dossier src/)

**config.py** : Toute la configuration des optimisations. J'ai mis les paramètres par défaut pour différents modèles.

**optimizers.py** : Les classes pour optimiser Random Forest, Gradient Boosting, SVM, etc. C'est là que la magie opère.

**study_manager.py** : Pour gérer les études Optuna - créer, sauvegarder, comparer les résultats.

## Architecture technique du framework

### Design Patterns utilisés

**1. Strategy Pattern (Optimizers)**
Chaque algorithme ML a son propre optimiseur qui implémente la même interface :

```python
class BaseOptimizer:
    def suggest_params(self, trial): pass
    def create_model(self, params): pass
    def evaluate_model(self, model, X, y): pass

class RandomForestOptimizer(BaseOptimizer):
    # Implémentation spécifique pour Random Forest

class SVMOptimizer(BaseOptimizer):
    # Implémentation spécifique pour SVM
```

**2. Factory Pattern (Study Creation)**
Le StudyManager crée les études selon le type demandé :

```python
class StudyManager:
    def create_study(self, study_type, config):
        if study_type == "classification":
            return self._create_classification_study(config)
        elif study_type == "regression":
            return self._create_regression_study(config)
        # etc.
```

**3. Configuration Pattern**
Toute la configuration est centralisée et typée :

```python
@dataclass
class OptimizationConfig:
    n_trials: int = 100
    sampler_type: str = "TPE"
    pruner_type: Optional[str] = None
    direction: str = "maximize"
    timeout: Optional[int] = None
```

### Technologies et dépendances

**Core ML Stack :**
- **Optuna 3.4+** : Framework d'optimisation
- **Scikit-learn 1.3+** : Algorithmes ML
- **Pandas 2.0+** : Manipulation de données
- **NumPy 1.24+** : Calculs numériques

**Visualisation et Dashboard :**
- **Optuna-Dashboard** : Interface web interactive
- **Plotly** : Graphiques interactifs
- **Matplotlib** : Visualisations statiques

**Infrastructure :**
- **SQLite** : Base de données pour les études
- **Docker** : Containerisation
- **Python 3.9+** : Runtime

### Stockage et persistance

**Base de données SQLite :**
```
studies/unified_demo.db
├── studies (table)          # Métadonnées des études
├── trials (table)           # Résultats de chaque trial
├── trial_params (table)     # Paramètres testés
├── trial_values (table)     # Scores obtenus
└── trial_intermediate_values # Scores intermédiaires (pruning)
```

**Avantages de SQLite :**
- Pas de serveur à gérer
- Fichier unique portable
- Compatible avec Optuna Dashboard
- Performances suffisantes pour nos besoins

## Les 6 études d'optimisation que j'ai préparées

J'ai conçu 6 études différentes pour vous démontrer les capacités d'Optuna sur des cas concrets :

### 1. Random Forest Classifier - Étude de Base
**Dataset :** Iris (150 échantillons, 4 features, 3 classes)
**Algorithme :** Random Forest avec optimisation TPE
**Paramètres optimisés :**
- `n_estimators` : [10, 200] - Nombre d'arbres
- `max_depth` : [2, 32] - Profondeur maximale
- `min_samples_split` : [2, 20] - Échantillons min pour diviser
- `min_samples_leaf` : [1, 10] - Échantillons min par feuille

**Objectif :** Maximiser l'accuracy (classification)
**Trials :** 100 essais
**Intérêt pédagogique :** Introduction aux concepts de base d'Optuna

### 2. Gradient Boosting Regressor - Optimisation de Régression
**Dataset :** California Housing (20,640 échantillons, 8 features)
**Algorithme :** Gradient Boosting avec TPE
**Paramètres optimisés :**
- `n_estimators` : [50, 300] - Nombre d'estimateurs
- `learning_rate` : [0.01, 0.3] - Taux d'apprentissage
- `max_depth` : [3, 10] - Profondeur des arbres
- `subsample` : [0.8, 1.0] - Fraction d'échantillons

**Objectif :** Minimiser le Mean Squared Error
**Trials :** 100 essais
**Intérêt pédagogique :** Optimisation pour la régression, gestion des hyperparamètres continus

### 3. SVM avec Pruning - Optimisation avec Arrêt Précoce
**Dataset :** Digits (1,797 échantillons, 64 features, 10 classes)
**Algorithme :** Support Vector Machine avec MedianPruner
**Paramètres optimisés :**
- `C` : [0.1, 100] - Paramètre de régularisation
- `gamma` : [0.001, 1] - Coefficient du kernel RBF
- `kernel` : ['rbf', 'poly', 'sigmoid'] - Type de kernel

**Objectif :** Maximiser l'accuracy avec pruning
**Trials :** 50 essais (avec arrêt précoce)
**Intérêt pédagogique :** Démonstration du pruning pour économiser du temps

### 4. Multi-Layer Perceptron - Réseaux de Neurones
**Dataset :** Wine (178 échantillons, 13 features, 3 classes)
**Algorithme :** MLP (Multi-Layer Perceptron)
**Paramètres optimisés :**
- `hidden_layer_sizes` : Architecture du réseau
- `learning_rate_init` : [0.0001, 0.1] - Taux d'apprentissage initial
- `alpha` : [0.0001, 0.01] - Régularisation L2
- `activation` : ['relu', 'tanh', 'logistic'] - Fonction d'activation

**Objectif :** Maximiser l'accuracy
**Trials :** 75 essais
**Intérêt pédagogique :** Optimisation de réseaux de neurones, paramètres complexes

### 5. Optimisation Multi-Objectifs - Front de Pareto
**Dataset :** Breast Cancer (569 échantillons, 30 features, 2 classes)
**Algorithme :** Random Forest avec optimisation multi-objectifs
**Objectifs simultanés :**
- **Maximiser** : Accuracy (performance)
- **Minimiser** : Nombre de features utilisées (complexité)

**Paramètres optimisés :**
- `n_estimators` : [10, 200]
- `max_depth` : [2, 20]
- `max_features` : [0.1, 1.0] - Fraction de features

**Trials :** 100 essais
**Intérêt pédagogique :** Optimisation multi-objectifs, front de Pareto, trade-offs

### 6. Comparaison de Samplers - Étude Comparative
**Dataset :** Iris (même que l'étude 1)
**Algorithme :** Random Forest (paramètres identiques)
**Samplers comparés :**
- **TPE** : Tree-structured Parzen Estimator
- **Random** : Échantillonnage aléatoire
- **Grid** : Recherche exhaustive
- **CMA-ES** : Covariance Matrix Adaptation

**Objectif :** Maximiser l'accuracy
**Trials :** 50 par sampler (200 total)
**Intérêt pédagogique :** Comparaison empirique des stratégies d'optimisation

## Métriques et évaluation

### Métriques utilisées par étude

**Classification (Études 1, 3, 4, 5) :**
- **Accuracy** : Pourcentage de prédictions correctes
- **Cross-validation 5-fold** : Validation croisée pour robustesse
- **Stratified split** : Préservation des proportions de classes

**Régression (Étude 2) :**
- **Mean Squared Error (MSE)** : Erreur quadratique moyenne
- **Cross-validation 5-fold** : Validation croisée
- **Train/test split 80/20** : Division standard

**Multi-objectifs (Étude 5) :**
- **Accuracy** : Performance du modèle
- **Feature count** : Nombre de features utilisées (complexité)
- **Pareto efficiency** : Solutions non-dominées

### Validation et robustesse

**Stratégies de validation :**
```python
# Classification avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Cross-validation pour robustesse
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
return scores.mean()
```

**Gestion de l'aléatoire :**
- **Random state fixe** : Reproductibilité des résultats
- **Seed global** : Cohérence entre les études
- **Multiple runs** : Validation de la stabilité

### Critères d'arrêt

**Convergence automatique :**
- **Plateau detection** : Arrêt si pas d'amélioration sur 20 trials
- **Timeout** : Limite de temps par étude
- **Target value** : Arrêt si objectif atteint

**Pruning criteria :**
- **MedianPruner** : Arrêt si en dessous de la médiane
- **Warmup period** : 10 étapes avant pruning
- **Startup trials** : 5 trials avant activation

## Résultats attendus et insights

### Performance typique par étude

**Étude 1 (Random Forest - Iris) :**
- **Baseline (défaut)** : ~95% accuracy
- **Après optimisation** : ~97-98% accuracy
- **Gain** : +2-3% avec paramètres optimaux
- **Convergence** : ~30-50 trials

**Étude 2 (Gradient Boosting - Housing) :**
- **Baseline (défaut)** : MSE ~0.5
- **Après optimisation** : MSE ~0.3-0.4
- **Gain** : 20-40% réduction d'erreur
- **Convergence** : ~50-80 trials

**Étude 3 (SVM - Digits) :**
- **Baseline (défaut)** : ~95% accuracy
- **Après optimisation** : ~98-99% accuracy
- **Gain** : +3-4% avec pruning efficace
- **Temps économisé** : 60-70% grâce au pruning

### Insights pédagogiques

**Ce que vous allez apprendre :**

1. **Impact des hyperparamètres** : Voir concrètement l'effet de chaque paramètre
2. **Stratégies d'optimisation** : Comparer TPE vs Random vs Grid
3. **Trade-offs** : Précision vs complexité vs temps de calcul
4. **Pruning efficace** : Économiser du temps sans perdre en qualité
5. **Multi-objectifs** : Gérer des objectifs contradictoires

**Questions que ça va répondre :**
- Combien de trials faut-il vraiment ?
- TPE est-il vraiment meilleur que Random ?
- Le pruning fait-il perdre en qualité ?
- Comment gérer précision vs complexité ?

## Utilisation pédagogique recommandée

### Parcours d'apprentissage suggéré

**Phase 1 : Découverte (30 min)**
1. Lancer le projet et explorer le dashboard
2. Observer l'étude 1 (Random Forest - Iris)
3. Comprendre les graphiques de base

**Phase 2 : Analyse (45 min)**
1. Comparer les études 1 et 6 (même dataset, différents samplers)
2. Analyser l'impact du pruning (étude 3)
3. Explorer l'optimisation multi-objectifs (étude 5)

**Phase 3 : Pratique (60 min)**
1. Modifier les paramètres d'une étude existante
2. Créer une nouvelle étude sur vos données
3. Expérimenter avec différents samplers

**Phase 4 : Approfondissement (45 min)**
1. Analyser les résultats en détail
2. Comprendre les trade-offs
3. Planifier l'intégration dans vos projets

### Points clés à retenir

**Pour vos futurs projets :**
- Commencez toujours par TPE (le plus efficace)
- Utilisez le pruning pour les modèles lents
- 50-100 trials suffisent généralement
- Validez avec cross-validation
- Documentez vos espaces de recherche

**Erreurs à éviter :**
- Espaces de paramètres trop larges
- Pas assez de trials pour converger
- Oublier la validation croisée
- Ignorer le temps de calcul
- Ne pas fixer les random seeds

## Technologies Utilisées

### Framework d'Optimisation
- **Optuna** : Framework d'optimisation automatique d'hyperparamètres
- **Optuna Dashboard** : Interface web pour visualiser les optimisations

### Machine Learning
- **Scikit-learn** : Bibliothèque ML pour les algorithmes
- **NumPy** : Calculs numériques
- **Pandas** : Manipulation de données

### Infrastructure
- **Docker** : Conteneurisation de l'application
- **Docker Compose** : Orchestration des services
- **SQLite** : Stockage des études et résultats

### Tests
- **Pytest** : Framework de tests
- **Coverage** : Couverture de code

## Fonctionnalités Principales

### 1. Optimisation Automatique
Le projet crée automatiquement 6 études d'optimisation au démarrage, démontrant différentes techniques et algorithmes.

### 2. Dashboard Interactif
Un dashboard web accessible sur http://localhost:8080 permet de :
- Visualiser l'historique des optimisations
- Analyser l'importance des paramètres
- Explorer les relations entre paramètres
- Comparer les trials

### 3. Persistance des Données
Toutes les études sont sauvegardées dans une base de données SQLite, permettant :
- Reprise des optimisations
- Analyse historique
- Partage des résultats

### 4. Déploiement Simplifié
Grâce à Docker, le projet se lance en une seule commande :
```bash
docker-compose up -d --build
```

## Cas d'Usage

### Pour l'Apprentissage
- Comprendre comment fonctionne l'optimisation d'hyperparamètres
- Voir des exemples concrets d'utilisation d'Optuna
- Apprendre à interpréter les visualisations

### Pour l'Enseignement
- Démontrer Optuna à des collègues
- Servir de base pour des exercices pratiques
- Illustrer les concepts d'optimisation

### Pour la Production
- Template pour vos propres projets d'optimisation
- Architecture modulaire réutilisable
- Bonnes pratiques implémentées

## Avantages du Projet

### Complet
- 6 études différentes couvrant divers cas d'usage
- Documentation complète en français
- Exemples de code commentés

### Prêt à l'Emploi
- Configuration Docker incluse
- Dépendances gérées
- Démarrage en une commande

### Éducatif
- Conçu pour l'apprentissage
- Documentation pédagogique
- Exercices pratiques

### Professionnel
- Code testé
- Architecture modulaire
- Bonnes pratiques ML

## Prochaines Étapes

Après avoir compris ce projet, vous pouvez :

1. Explorer le dashboard pour voir les résultats
2. Consulter le code source dans `src/`
3. Réaliser l'exercice pratique
4. Adapter le projet à vos propres besoins

## Ressources

- Code source : `src/` folder
- Exemples : `examples/` folder
- Tests : `tests/` folder
- Documentation Optuna : https://optuna.readthedocs.io

