# Commandes Essentielles

Guide des commandes pour utiliser le projet d'optimisation Optuna.

## Démarrage du Projet

### Lancer le projet

```powershell
# Construire et démarrer les conteneurs Docker
docker-compose up -d --build
```

### Vérifier le statut

```powershell
# Vérifier que le conteneur est en cours d'exécution
docker-compose ps
```

### Accéder au dashboard

Ouvrez votre navigateur et allez à :
```
http://localhost:8080
```

## Gestion du Projet

### Arrêter le projet

```powershell
# Arrêter les conteneurs
docker-compose down
```

### Redémarrer le projet

```powershell
# Redémarrer les conteneurs
docker-compose restart
```

### Voir les logs

```powershell
# Voir les logs du conteneur
docker-compose logs

# Suivre les logs en temps réel
docker-compose logs -f
```

## Dépannage

### Le dashboard ne se charge pas

1. Attendez 2-3 minutes après le démarrage
2. Vérifiez le statut : `docker-compose ps`
3. Consultez les logs : `docker-compose logs`

### Port 8080 déjà utilisé

```powershell
# Arrêter les conteneurs existants
docker-compose down

# Vérifier les processus utilisant le port 8080
netstat -ano | findstr :8080
```

### Reconstruire complètement

```powershell
# Arrêter et supprimer les conteneurs
docker-compose down

# Reconstruire depuis zéro
docker-compose up -d --build --force-recreate
```

## Commandes Avancées

### Exécuter des commandes dans le conteneur

```powershell
# Accéder au shell du conteneur
docker exec -it ml-optimization-framework sh

# Exécuter une commande Python
docker exec ml-optimization-framework python create_unified_demo.py
```

### Voir les études créées

```powershell
# Lister les fichiers dans le dossier studies
docker exec ml-optimization-framework ls -la studies/
```

## Développement Local (sans Docker)

### Installation des dépendances

```powershell
# Installer les dépendances Python
pip install -r requirements-minimal.txt
```

### Créer les études d'optimisation

```powershell
# Exécuter le script de création des études
python create_unified_demo.py
```

### Lancer le dashboard manuellement

```powershell
# Démarrer le dashboard Optuna
optuna-dashboard sqlite:///studies/unified_demo.db --host 0.0.0.0 --port 8080
```

## Résumé des Commandes Principales

| Action | Commande |
|--------|----------|
| Démarrer | `docker-compose up -d --build` |
| Arrêter | `docker-compose down` |
| Statut | `docker-compose ps` |
| Logs | `docker-compose logs` |
| Redémarrer | `docker-compose restart` |
| Dashboard | http://localhost:8080 |

