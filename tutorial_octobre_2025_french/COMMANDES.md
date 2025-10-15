# Les Commandes dont vous avez besoin

Bon, voici toutes les commandes pour faire tourner le projet. C'est du Docker, donc c'est assez simple.

## Pour démarrer

### Lancer tout le bordel

```powershell
docker-compose up -d --build
```

Cette commande :
- Construit l'image Docker
- Lance le conteneur
- Crée les 6 études d'optimisation
- Démarre le dashboard

**Attendez 2-3 minutes** que tout se lance.

### Vérifier que ça tourne

```powershell
docker-compose ps
```

Vous devriez voir quelque chose comme "healthy" dans le statut.

### Accéder au dashboard

Ouvrez votre navigateur : **http://localhost:8080**

## Pour gérer le projet

### Arrêter le projet

```powershell
docker-compose down
```

### Redémarrer

```powershell
docker-compose restart
```

### Voir ce qui se passe (logs)

```powershell
# Voir les logs
docker-compose logs

# Suivre en temps réel
docker-compose logs -f
```

## Si ça marche pas

### Le dashboard ne s'affiche pas

Pas de panique, c'est normal :
1. **Attendez 2-3 minutes** après le démarrage (il faut créer les études)
2. Vérifiez que ça tourne : `docker-compose ps`
3. Regardez les logs : `docker-compose logs`

### "Port 8080 déjà utilisé"

Quelqu'un d'autre utilise le port :

```powershell
# Arrêtez tout
docker-compose down

# Voyez qui utilise le port 8080
netstat -ano | findstr :8080
```

### Tout reconstruire (solution de bourrin)

Si vraiment ça marche pas :

```powershell
# Tout arrêter et supprimer
docker-compose down

# Reconstruire depuis zéro
docker-compose up -d --build --force-recreate
```

## Commandes pour les curieux

### Rentrer dans le conteneur

```powershell
# Accéder au shell du conteneur
docker exec -it ml-optimization-framework sh
```

### Voir les fichiers créés

```powershell
# Lister les études dans le conteneur
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

