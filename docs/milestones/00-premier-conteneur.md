# Milestone 00 — Premier conteneur Docker

## Objectif

Vérifier que Docker Desktop fonctionne et observer le premier cycle de vie :

```text
image Docker → création du conteneur → exécution → arrêt du processus
```

Ce jalon n'utilise encore aucun fichier du projet Evidentia. Il valide uniquement
le moteur Docker et permet de distinguer une image d'un conteneur.

## Commande exécutée

```powershell
docker run --name evidentia-hello hello-world
```

### Décomposition

- `docker` : client en ligne de commande qui envoie une requête au moteur Docker ;
- `run` : demande de créer puis démarrer un conteneur ;
- `--name evidentia-hello` : nom explicite donné au conteneur ;
- `hello-world` : image utilisée comme modèle ;
- l'absence de `--rm` est volontaire : le conteneur arrêté sera conservé afin de
  pouvoir l'inspecter.

## Déroulement observé

### 1. Recherche locale de l'image

```text
Unable to find image 'hello-world:latest' locally
```

Docker ne trouvait pas l'image dans son cache local.

### 2. Téléchargement depuis Docker Hub

```text
latest: Pulling from library/hello-world
4f55086f7dd0: Pull complete
d5e71e642bf5: Download complete
```

Docker a téléchargé l'image officielle `library/hello-world` depuis Docker Hub.
Une image est constituée de couches. Ces couches sont conservées localement et
peuvent être réutilisées par de futurs conteneurs.

### 3. Vérification de l'identité de l'image

```text
Digest: sha256:7f4da0fc94bcece205a8c0b6f4d11c8196924654ffe5c4d1aa439b7f632048b2
```

Le digest identifie précisément le contenu téléchargé. Le tag `latest` est un
alias pratique pour une démonstration, mais nous utiliserons des versions
explicitement fixées dans le projet afin d'améliorer la reproductibilité.

### 4. Création et démarrage du conteneur

```text
Status: Downloaded newer image for hello-world:latest
```

Docker a créé le conteneur `evidentia-hello` à partir de l'image téléchargée,
puis a lancé son processus principal.

### 5. Exécution du processus

```text
Hello from Docker!
```

Le programme contenu dans l'image a affiché son message puis s'est terminé
normalement. Comme le processus principal est terminé, le conteneur est
maintenant arrêté, mais il n'est pas supprimé.

## État attendu après la commande

| Élément | État |
|---|---|
| Image `hello-world` | Présente localement |
| Conteneur `evidentia-hello` | Présent mais arrêté |
| Processus du conteneur | Terminé avec succès |
| Mémoire du processus | Libérée |
| Couche d'écriture du conteneur | Conservée tant que le conteneur existe |
| Volume Docker | Aucun utilisé dans ce jalon |
| Fichiers du projet | Aucun modifié par Docker |

## Concepts validés

1. Une image est un modèle immuable.
2. Un conteneur est une instance exécutable d'une image.
3. `docker run` peut télécharger une image absente, créer un conteneur et le
   démarrer.
4. Un processus peut se terminer sans que le conteneur soit automatiquement
   supprimé.
5. L'arrêt ou la fin du processus ne supprime pas l'image.
6. L'option `--rm` aurait supprimé automatiquement le conteneur à sa fin ; elle
   n'a volontairement pas été utilisée ici.

## Prochaine vérification

Inspecter séparément :

- les conteneurs, y compris ceux qui sont arrêtés ;
- l'image locale `hello-world` ;
- le statut et le code de sortie du conteneur.

La suppression du conteneur ne sera effectuée qu'après avoir observé ces états.

## Inspection du conteneur arrêté

### Commande exécutée

```powershell
docker ps -a --filter "name=evidentia-hello"
```

### Résultat observé

```text
CONTAINER ID   IMAGE         COMMAND    CREATED          STATUS                      PORTS     NAMES
d04dea6c9787   hello-world   "/hello"   47 minutes ago   Exited (0) 47 minutes ago             evidentia-hello
```

### Interprétation des colonnes

- `CONTAINER ID` : identifiant court du conteneur ;
- `IMAGE` : image à partir de laquelle le conteneur a été créé ;
- `COMMAND` : processus principal lancé dans le conteneur ;
- `CREATED` : moment de création du conteneur ;
- `STATUS` : état actuel et code de sortie ;
- `PORTS` : aucun port publié, car `hello-world` n'est pas un serveur ;
- `NAMES` : nom lisible choisi avec `--name`.

### Signification de `Exited (0)`

`Exited` signifie que le processus principal n'est plus actif. Le conteneur
reste néanmoins présent, car l'option `--rm` n'a pas été utilisée.

Le code `0` signifie que le processus s'est terminé normalement. Un code
différent de `0` signalerait généralement une erreur du programme, qu'il
faudrait ensuite analyser dans les logs ou avec une inspection détaillée.

Cette sortie confirme donc la différence entre :

```text
conteneur existant + processus arrêté
```

et :

```text
conteneur supprimé
```

La commande `docker ps` seule n'aurait pas affiché cette ligne, car elle ne
montre que les conteneurs en cours d'exécution. L'option `-a` inclut les
conteneurs arrêtés.

## Inspection de l'image locale

### Commande exécutée

```powershell
docker image ls hello-world
```

### Résultat observé

```text
IMAGE                ID             DISK USAGE   CONTENT SIZE   EXTRA
hello-world:latest   7f4da0fc94bc       25.9kB         9.49kB    U
```

### Interprétation

- `hello-world:latest` : nom du dépôt et tag de l'image ;
- `7f4da0fc94bc` : identifiant court de l'image stockée localement ;
- `DISK USAGE` : espace actuellement utilisé localement par l'image et ses
  métadonnées ;
- `CONTENT SIZE` : taille du contenu de l'image telle que rapportée par Docker ;
- `U` dans `EXTRA` : l'image est actuellement marquée comme utilisée.

Le marquage `U` est présent parce que le conteneur `evidentia-hello` référence
encore cette image. Le conteneur peut être arrêté tout en continuant à utiliser
la référence vers son image de base.

Cette observation permet de distinguer clairement :

```text
image locale conservée
conteneur arrêté conservé
processus terminé
```

La suppression du conteneur libérerait sa couche d'écriture, mais ne supprimerait
pas automatiquement cette image. L'image resterait disponible pour créer un
nouveau conteneur.

## Inspection détaillée du conteneur

### Commande exécutée

```powershell
docker container inspect evidentia-hello
```

Cette commande renvoie la configuration complète du conteneur au format JSON.
Le JSON est organisé en grandes familles : identité, exécution, stockage,
configuration de l'hôte, configuration du programme et réseau.

### Identité

```json
"Id": "d04dea6c...",
"Name": "/evidentia-hello",
"Image": "sha256:7f4da0..."
```

- `Id` identifie le conteneur lui-même ;
- `Name` est le nom lisible choisi au moment de `docker run` ;
- `Image` est l'identifiant exact de l'image utilisée.

Le conteneur et l'image ont donc deux identifiants différents : le conteneur
référence l'image, mais n'est pas l'image.

### Processus exécuté

```json
"Path": "/hello",
"Args": [],
"Config": {
  "Cmd": ["/hello"],
  "Image": "hello-world",
  "WorkingDir": "/"
}
```

Le processus principal était `/hello`, sans argument, avec `/` comme répertoire
de travail. Lorsqu'il s'est terminé, Docker a considéré que le conteneur avait
terminé son exécution.

### État d'exécution

```json
"State": {
  "Status": "exited",
  "Running": false,
  "Paused": false,
  "Restarting": false,
  "OOMKilled": false,
  "Pid": 0,
  "ExitCode": 0
}
```

- `Status: exited` : le conteneur est arrêté ;
- `Running: false` : aucun processus du conteneur ne tourne ;
- `Paused: false` : il n'est pas suspendu ;
- `Restarting: false` : Docker n'essaie pas de le redémarrer ;
- `OOMKilled: false` : il n'a pas été arrêté pour dépassement mémoire ;
- `Pid: 0` : il n'a plus de processus actif ;
- `ExitCode: 0` : le programme s'est terminé normalement.

Les champs `StartedAt` et `FinishedAt` indiquent la durée réelle d'exécution,
très courte pour `hello-world`.

### Stockage

```json
"Driver": "overlayfs",
"Mounts": [],
"HostConfig": {
  "Binds": null,
  "ReadonlyRootfs": false
}
```

- `overlayfs` est le système de couches utilisé pour les conteneurs Linux ;
- `Mounts: []` confirme qu'aucun volume n'est attaché ;
- `Binds: null` confirme qu'aucun dossier Windows n'est monté ;
- `ReadonlyRootfs: false` indique que la couche d'écriture du conteneur n'est
  pas configurée en lecture seule.

Toute modification écrite dans le système de fichiers interne de ce conteneur
aurait donc été conservée après `docker stop`, mais aurait disparu avec
`docker rm`, car aucun volume ne la protégerait.

### Configuration de l'hôte

```json
"NetworkMode": "bridge",
"PortBindings": {},
"RestartPolicy": {"Name": "no"},
"AutoRemove": false,
"Privileged": false,
"Memory": 0
```

- `bridge` est le réseau Docker par défaut ;
- `PortBindings: {}` signifie qu'aucun port n'est publié vers Windows ;
- `RestartPolicy: no` signifie qu'il ne redémarre pas automatiquement ;
- `AutoRemove: false` confirme que `--rm` n'a pas été utilisé ;
- `Privileged: false` indique que le conteneur n'a pas de privilèges étendus ;
- `Memory: 0` signifie qu'aucune limite mémoire explicite n'a été configurée,
  et non que le conteneur possède zéro mémoire.

### Réseau

Le conteneur est rattaché au réseau `bridge`, mais les champs d'adresse IP sont
vides car le processus est arrêté. Aucun service n'écoutait sur un port.

### Fichiers internes Docker Desktop

Les chemins `ResolvConfPath`, `HostnamePath`, `HostsPath` et `LogPath` pointent
vers les fichiers internes du moteur Linux utilisé par Docker Desktop. Ils ne
doivent pas être modifiés directement depuis Windows. Nous utiliserons les
commandes Docker, les volumes et les fichiers du projet pour agir proprement.

## Redémarrage du même conteneur

### Commande exécutée

```powershell
docker start -a evidentia-hello
```

- `docker start` redémarre un conteneur déjà créé ;
- `-a` rattache la sortie standard du conteneur au terminal ;
- aucun nouveau conteneur n'est créé ;
- aucune nouvelle image n'est téléchargée.

Le programme a de nouveau affiché `Hello from Docker!`, puis s'est terminé avec
le code `0`. L'identifiant et le nom du conteneur restent les mêmes.

### Attention au texte affiché

Le message affiché par `hello-world` contient les phrases :

```text
The Docker client contacted the Docker daemon.
The Docker daemon pulled the "hello-world" image from the Docker Hub.
The Docker daemon created a new container from that image.
```

Ces phrases font partie du texte produit par le programme `/hello`. Elles
décrivent le scénario général de première exécution ; elles ne constituent pas
le journal détaillé de cette seconde commande. Lors de `docker start`, l'image et
le conteneur existaient déjà et ont été réutilisés.

Cette différence est importante : il faut distinguer la sortie du programme
contenu dans un conteneur des messages produits par le client Docker lui-même.

## Suppression du conteneur

### Commande exécutée

```powershell
docker rm evidentia-hello
```

### Résultat

```text
evidentia-hello
```

Docker a confirmé la suppression du conteneur portant ce nom.

### Conséquences

- le conteneur `evidentia-hello` n'existe plus ;
- sa couche d'écriture est supprimée ;
- son état, sa configuration et ses logs associés au conteneur sont supprimés ;
- aucun volume n'a été supprimé, car le conteneur n'en possédait aucun ;
- l'image `hello-world` n'est pas supprimée par `docker rm`.

Ce jalon démontre donc le cycle complet :

```text
docker run       → image disponible + conteneur créé + programme exécuté
docker start     → même conteneur réutilisé
docker rm        → conteneur et couche d'écriture supprimés
image            → conservée indépendamment du conteneur
```

La vérification finale de l'absence du conteneur et de la présence de l'image
reste à effectuer.

## Vérification après suppression

### Commande exécutée

```powershell
docker ps -a --filter "name=evidentia-hello"
```

### Résultat observé

```text
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
```

Seul l'en-tête est affiché. Aucune ligne ne correspond au conteneur
`evidentia-hello`, ce qui confirme qu'il a bien été supprimé.

Cette sortie est différente de la précédente : auparavant une ligne décrivait
le conteneur avec le statut `Exited (0)` ; maintenant il n'existe plus dans la
liste des conteneurs, y compris avec l'option `-a`.

## Vérification finale de l'image après suppression du conteneur

### Commande exécutée

```powershell
docker image ls hello-world
```

### Résultat observé

```text
IMAGE                ID             DISK USAGE   CONTENT SIZE   EXTRA
hello-world:latest   7f4da0fc94bc       25.9kB         9.49kB
```

L'image est toujours disponible localement. Le marqueur `U` a disparu, car
aucun conteneur ne la référence désormais.

Cette observation confirme définitivement que `docker rm` a supprimé le
conteneur, mais pas son image.

## Conclusion du milestone

Le premier cycle Docker est validé :

```text
image téléchargée
→ conteneur créé
→ processus exécuté
→ conteneur arrêté
→ même conteneur redémarré
→ conteneur supprimé
→ image conservée
```

Les notions d'image, de conteneur, d'arrêt, de redémarrage, de suppression,
de code de sortie et de couche d'écriture ont été observées directement avec
les commandes Docker.

### Prochain milestone

Créer un conteneur Alpine volontairement maintenu en fonctionnement. Il
permettra d'observer :

- `docker ps` pendant l'exécution ;
- `docker exec` pour entrer dans un conteneur actif ;
- création d'un fichier dans la couche d'écriture ;
- `docker stop` ;
- `docker start` et vérification de la conservation du fichier ;
- suppression de la couche avec `docker rm`.
