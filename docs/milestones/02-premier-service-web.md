# Milestone 02 — Première image applicative et premier service web

## Objectif

Transformer le squelette Python d'Evidentia en un service HTTP réellement
exécutable dans Docker. Cette étape valide la chaîne complète : code source,
construction d'image, création d'un conteneur et accès depuis le navigateur.

## Les fichiers de l'application

| Fichier | Rôle |
|---|---|
| `app/main.py` | Crée l'application FastAPI et les routes HTTP. |
| `app/__init__.py` | Indique que `app` est un module Python importable. |
| `requirements.txt` | Liste les dépendances Python : FastAPI et Uvicorn. |
| `Dockerfile` | Décrit comment construire l'image applicative. |
| `.dockerignore` | Exclut du contexte de construction les fichiers inutiles ou sensibles. |

L'application expose volontairement deux routes très simples :

- `GET /` : confirme que le service Evidentia est démarré ;
- `GET /health` : point de contrôle minimal, réutilisable plus tard par Docker
  Compose, un orchestrateur ou un outil de supervision.

FastAPI génère également `GET /docs`, une interface interactive permettant de
consulter et d'appeler les routes sans écrire de client HTTP.

## Construction de l'image

```powershell
docker build --tag evidentia-app:dev .
```

`docker build` lit le `Dockerfile` dans le dossier courant et produit une image
locale. Le tag `evidentia-app:dev` est un nom humain destiné au développement :
il pourra être remplacé ou complété par des versions plus précises plus tard.

Les instructions essentielles du `Dockerfile` sont :

1. `FROM python:3.12-slim` : point de départ avec Python et un système Linux
   réduit.
2. `WORKDIR /app` : dossier de travail interne au conteneur.
3. `COPY requirements.txt .` puis `RUN pip install ...` : installation des
   dépendances dans une couche Docker distincte, réutilisable si seul le code
   change.
4. `COPY app ./app` : copie du code du projet dans l'image.
5. `CMD ["uvicorn", ...]` : processus principal lancé à chaque démarrage du
   conteneur.

`EXPOSE 8000` documente le port utilisé par le programme ; il ne rend pas ce
port accessible depuis Windows à lui seul.

## Lancement du conteneur applicatif

```powershell
docker run -d --name evidentia-app -p 8000:8000 evidentia-app:dev
```

Cette commande a créé et démarré un conteneur nommé `evidentia-app`, à partir
de l'image construite. L'absence d'erreur confirme que Docker a pu démarrer le
processus Uvicorn.

| Élément | Valeur |
|---|---|
| Image | `evidentia-app:dev` |
| Conteneur | `evidentia-app` |
| Processus principal | `uvicorn app.main:app --host 0.0.0.0 --port 8000` |
| Mode | Détaché (`-d`) |
| Port hôte | `8000` sur Windows |
| Port conteneur | `8000` dans le réseau Docker |
| Volume | Aucun à ce stade |

L'option `-p 8000:8000` suit la forme `port_hôte:port_conteneur` : une requête
vers `http://localhost:8000` sur Windows est transmise au port 8000 du
conteneur. Le processus écoute sur `0.0.0.0` afin d'accepter cette connexion
venant de l'extérieur du conteneur.

## Vérification fonctionnelle

Les URLs à consulter sont :

- `http://localhost:8000/health` : la réponse attendue contient
  `{"status":"ok"}` ;
- `http://localhost:8000/docs` : documentation Swagger UI générée par FastAPI.

Si le navigateur ne peut pas atteindre le service, les premiers diagnostics
seront :

```powershell
docker ps
docker logs evidentia-app
```

`docker ps` vérifie que le conteneur est encore `Up`. `docker logs` affiche les
journaux du processus Uvicorn sans entrer dans le conteneur.

## Ce que cette étape prouve

Le projet possède désormais une frontière claire entre :

- le code Python versionné dans le dossier du projet ;
- une image reproductible qui contient Python, les dépendances et ce code ;
- un conteneur exécutant une instance isolée du service ;
- un port explicitement publié pour rendre l'API accessible.

Cette même API accueillera progressivement l'ingestion de documents, le RAG,
les modèles LLM/VLM, puis le graphe d'agents LangGraph. Les composants lourds
ne seront pas ajoutés au conteneur applicatif avant d'avoir une utilité et une
interface de test concrètes.
