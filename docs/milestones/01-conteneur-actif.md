# Milestone 01 — Conteneur actif et exécution en arrière-plan

## Objectif

Créer un conteneur qui reste actif afin d'observer un service en cours
d'exécution, puis d'expérimenter `docker ps`, `docker exec`, `docker stop`,
`docker start` et la persistance de la couche d'écriture.

## Commande exécutée

```powershell
docker run -d --name evidentia-sandbox alpine:3.20 sleep 3600
```

### Décomposition

- `docker run` : créer puis démarrer un nouveau conteneur ;
- `-d` : mode détaché, le conteneur continue en arrière-plan et le terminal
  récupère immédiatement la main ;
- `--name evidentia-sandbox` : nom explicite du conteneur ;
- `alpine:3.20` : image Linux Alpine avec un tag de version explicite ;
- `sleep 3600` : processus principal qui attend 3600 secondes, soit une heure.

Le processus principal est volontairement long pour que le conteneur reste dans
l'état `running`. Tant que ce processus existe, le conteneur est actif.

## Déroulement observé

### Recherche et téléchargement de l'image

```text
Unable to find image 'alpine:3.20' locally
3.20: Pulling from library/alpine
25f1d6b1951a: Pull complete
3a030ca3f633: Download complete
d4050d56ebf2: Download complete
```

L'image Alpine n'était pas encore dans le cache local. Docker l'a téléchargée
depuis le dépôt officiel `library/alpine`, en récupérant ses différentes
couches.

### Vérification et lancement

```text
Digest: sha256:d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc
Status: Downloaded newer image for alpine:3.20
```

Le digest identifie précisément le contenu récupéré. Le tag `3.20` est plus
reproductible que `latest`, car il désigne une version majeure et mineure
explicite.

### Identifiant retourné

```text
a2099628fa5ea989be933eecc1ddee92f6b069b96552e59921a9c27e90118e2b
```

Comme le mode détaché a été utilisé, Docker n'affiche pas la sortie du
processus `sleep`. Il retourne immédiatement l'identifiant complet du nouveau
conteneur.

## État attendu

| Élément | État |
|---|---|
| Image `alpine:3.20` | Téléchargée localement |
| Conteneur `evidentia-sandbox` | En cours d'exécution |
| Processus principal | `sleep 3600` |
| Mode | Détaché (`-d`) |
| Port publié | Aucun |
| Volume | Aucun |
| Suppression automatique | Désactivée |

## Prochaine action

Utiliser `docker ps` pour vérifier que le conteneur actif apparaît, contrairement
au conteneur `hello-world` arrêté qui ne s'affichait qu'avec `docker ps -a`.

