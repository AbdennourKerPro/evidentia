# Milestone 03 — Environnement Compose et disponibilité de Qdrant

## Objectif

Exécuter l'API Evidentia et Qdrant comme deux services Docker coordonnés, puis
vérifier depuis l'API que la base vectorielle est réellement accessible. Cette
étape ne construit pas encore un RAG : elle installe et teste son infrastructure
minimale.

## Architecture obtenue

```text
Navigateur Windows
    │
    ├── http://localhost:8000 ──> API FastAPI (service `api`)
    │                                  │
    │                                  └── http://qdrant:6333
    │                                           réseau Docker privé
    │
    └── http://localhost:6333 ──> Qdrant (service `qdrant`)
                                         │
                                         └── volume `qdrant_storage`
```

Le nom `qdrant` est une résolution DNS fournie automatiquement par Docker
Compose sur le réseau privé du projet. Il n'est donc pas utilisé depuis le
navigateur Windows : il est réservé aux communications entre conteneurs.

## Fichier `compose.yaml`

Le fichier `compose.yaml` décrit l'environnement local complet.

### Service `api`

- construit l'image avec le `Dockerfile` du projet ;
- publie le port `8000` vers Windows ;
- reçoit `QDRANT_URL=http://qdrant:6333` via son environnement ;
- dépend du démarrage du service Qdrant.

`depends_on` fournit un ordre de démarrage, mais ne vérifie pas que Qdrant est
déjà prêt à répondre. Cette différence motive le point de contrôle `/ready`.

### Service `qdrant`

- utilise l'image officielle `qdrant/qdrant:v1.18.2` ;
- publie `127.0.0.1:6333:6333`, donc uniquement sur la machine locale ;
- conserve ses données dans le volume Docker nommé `qdrant_storage`.

Le volume contient les collections, vecteurs, index et métadonnées Qdrant. Il
survit à `docker compose down` et à la recréation du conteneur. Il est supprimé
uniquement si l'on demande explicitement la suppression des volumes, par
exemple avec `docker compose down -v`.

Un volume Docker nommé est préféré à un montage de dossier Windows pour Qdrant,
car les systèmes de fichiers partagés avec WSL peuvent être incompatibles avec
son stockage interne.

## Client Qdrant dans l'API

### Dépendance

`requirements.txt` contient :

```text
qdrant-client==1.18.0
```

Cette bibliothèque Python officielle parle à Qdrant par HTTP. La version est
épinglée afin d'éviter qu'une reconstruction ultérieure installe une version
différente de façon silencieuse.

### Configuration : `app/settings.py`

La fonction `get_qdrant_url()` lit la variable d'environnement `QDRANT_URL`.
La valeur par défaut `http://qdrant:6333` fonctionne dans Docker Compose. Cette
séparation évite de disperser des adresses réseau dans les routes HTTP et
permettra plus tard d'utiliser une instance Qdrant distante sans modifier le
code métier.

### Passerelle : `app/qdrant_gateway.py`

La fonction `qdrant_is_available()` :

1. crée un client Qdrant avec un délai de deux secondes ;
2. exécute `get_collections()`, une requête légère qui prouve que le service
   répond ;
3. ferme le client dans tous les cas ;
4. renvoie `True` ou `False` plutôt que de laisser un détail technique remonter
   au navigateur.

Ce module constitue une frontière claire : le reste de l'API ne dépend pas
directement de la bibliothèque Qdrant. Les futures opérations d'indexation et
de recherche seront ajoutées ici.

## Endpoints de contrôle

| Route | Signification | Réponse de succès |
|---|---|---|
| `GET /health` | Le processus FastAPI est vivant. | `{"status":"ok"}` |
| `GET /ready` | FastAPI peut joindre Qdrant. | `{"status":"ready","qdrant":"ok"}` |

Si Qdrant est indisponible, `/ready` retourne HTTP 503. Cela signale que
l'instance ne doit pas encore recevoir de requête RAG, tandis que `/health`
continue de distinguer une panne de dépendance d'une panne de l'API elle-même.

## Commande d'application

```powershell
docker compose up --build -d
```

- `up` crée ou met à jour les services définis par Compose ;
- `--build` reconstruit l'image de l'API car les dépendances et le code ont
  changé ;
- `-d` rend la main au terminal tout en laissant les deux services actifs.

La vérification fonctionnelle a réussi avec `http://localhost:8000/ready`.

## État du projet à la fin du jalon

- API FastAPI conteneurisée et accessible ;
- Qdrant conteneurisé avec stockage persistant ;
- réseau privé entre les deux services ;
- configuration externe par variable d'environnement ;
- test de disponibilité de la dépendance vectorielle ;
- aucun document, embedding ni collection Qdrant n'a encore été créé.

## Suite

Le prochain jalon introduira les embeddings et un premier corpus contrôlé. Il
permettra de créer une collection Qdrant, d'y stocker des chunks accompagnés de
leurs métadonnées, puis d'effectuer la première recherche sémantique.
