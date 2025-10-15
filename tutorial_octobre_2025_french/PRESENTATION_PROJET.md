# Présentation du Projet d'Optimisation ML

Ce document présente le projet d'optimisation d'hyperparamètres avec Optuna.

## Vue d'Ensemble

Ce projet est un framework complet d'optimisation d'hyperparamètres pour le machine learning, construit avec Optuna. Il démontre différentes techniques d'optimisation à travers 6 études pré-configurées.

## Objectif du Projet

Le projet a été conçu pour :

- Démontrer les capacités d'Optuna
- Fournir des exemples concrets d'optimisation
- Servir de base d'apprentissage pour les équipes
- Montrer les meilleures pratiques d'optimisation

## Architecture du Projet

### Structure des Dossiers

```
optimization_with_optuna/
├── src/                          # Code source
│   ├── config.py                 # Configuration
│   ├── optimizers.py             # Optimiseurs ML
│   └── study_manager.py          # Gestion des études
├── examples/                     # Exemples
├── tests/                        # Tests
├── studies/                      # Base de données SQLite
├── tutorial_octobre_2025_french/ # Documentation française
├── docker-compose.yml            # Configuration Docker
├── Dockerfile                    # Image Docker
└── requirements-minimal.txt      # Dépendances Python
```

### Composants Principaux

#### 1. Configuration (src/config.py)
Gère la configuration des optimisations :
- Paramètres des modèles
- Configuration des études
- Paramètres du dashboard

#### 2. Optimiseurs (src/optimizers.py)
Implémente les optimiseurs pour différents algorithmes ML :
- Random Forest
- Gradient Boosting
- SVM
- Réseaux de neurones (MLP)

#### 3. Gestionnaire d'Études (src/study_manager.py)
Gère le cycle de vie des études Optuna :
- Création des études
- Exécution des optimisations
- Sauvegarde des résultats

## Les 6 Études d'Optimisation

### Étude 1 : Random Forest Classifier
- **Dataset** : Iris (classification de fleurs)
- **Algorithme** : Random Forest
- **Sampler** : TPE (Tree-structured Parzen Estimator)
- **Trials** : 100
- **Objectif** : Maximiser la précision de classification

### Étude 2 : Gradient Boosting Regressor
- **Dataset** : California Housing (prix des maisons)
- **Algorithme** : Gradient Boosting
- **Sampler** : TPE
- **Trials** : 100
- **Objectif** : Minimiser l'erreur de prédiction

### Étude 3 : SVM avec Pruning
- **Dataset** : Digits (reconnaissance de chiffres)
- **Algorithme** : Support Vector Machine
- **Sampler** : TPE avec MedianPruner
- **Trials** : 50
- **Objectif** : Maximiser la précision avec arrêt précoce

### Étude 4 : Réseau de Neurones (MLP)
- **Dataset** : Wine (classification de vins)
- **Algorithme** : Multi-layer Perceptron
- **Sampler** : TPE
- **Trials** : 75
- **Objectif** : Maximiser la précision

### Étude 5 : Optimisation Multi-Objectifs
- **Dataset** : Breast Cancer (détection de cancer)
- **Algorithme** : Random Forest
- **Sampler** : TPE
- **Trials** : 100
- **Objectifs** : Maximiser la précision ET minimiser la taille du modèle
- **Résultat** : Front de Pareto

### Étude 6 : Comparaison de Samplers
- **Dataset** : Iris
- **Algorithme** : Random Forest
- **Samplers** : TPE, Random, Grid, CMA-ES
- **Trials** : 50 par sampler
- **Objectif** : Comparer les stratégies d'optimisation

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

