# Milestone 23 — Logo vectoriel sans nom de marque

## Choix et dessin

La proposition 4 est redessinée en SVG : deux pages opposées et une courbe
centrale rappelant la queue d'un point d'interrogation, sans point détaché.
Il s'agit d'une reconstruction géométrique du concept validé, pas d'une image
PNG intégrée dans un fichier SVG, ni d'une copie pixel à pixel du rendu généré.

`app/static/logo.svg` contient seulement un titre et deux tracés remplis en gris
très clair. Le premier dessine la page gauche. Le second réunit la page droite
et la courbe centrale. Les commandes `M`, `L`, `H`, `V`, `C` et `Z` déplacent le
point de dessin, tracent des segments ou des courbes de Bézier et ferment les
formes. Le `viewBox="0 0 100 100"` définit un repère indépendant des pixels.
Le fond est transparent : aucun rectangle de fond n'est nécessaire.

## Intégration

- `app/static/index.html` remplace le nom en haut à gauche par un lien d'accueil
  contenant le SVG. Un élément `img` peut afficher du SVG : le dessin reste
  vectoriel et n'utilise aucune des images générées.
- Le lien dispose d'un libellé accessible ; l'image a un texte alternatif vide
  pour éviter une lecture redondante. Le titre de l'onglet et le nom de l'auteur
  des réponses sont désormais neutres, sans ancienne marque.
- Le même SVG sert d'icône d'onglet, sans dépendance ou téléchargement extérieur.
- `app/static/styles.css` réserve 40 pixels au lien, affiche le logo à 36 pixels
  et à 30 pixels dans le panneau réduit. Dans ce dernier, logo et commande
  d'ouverture sont empilés pour rester utilisables dans le rail de 56 pixels.
- Les versions des URL du CSS et du logo évitent de réutiliser des ressources
  anciennes du cache après rechargement.

Le JavaScript, l'historique local, les sources, la pipeline et les appels LLM
ne sont pas modifiés. Le lien d'accueil recharge la page ; l'historique demeure
dans le stockage local du navigateur, comme avant.

## Vérification

Le test ajouté dans `tests/test_api_llm.py` vérifie la présence du logo et du
libellé accessible, l'absence de l'ancien nom dans le HTML du chat, le service
du fichier avec le type MIME SVG et sa structure XML. Il interdit notamment une
image raster embarquée, du script et les formes isolées de type point carré.

Le déploiement utilise le Dockerfile incrémental et recrée seulement l'API,
sans toucher aux volumes, aux modèles ou à Qdrant. Une vérification visuelle
complète les tests automatisés ; aucune évaluation RAG payante n'est nécessaire
pour ce changement exclusivement graphique.

Résultat : les 18 tests UI et les 158 tests Python passent. Dans le navigateur,
le fichier SVG est chargé correctement, avec une taille de 36 pixels dans le
panneau ouvert et de 30 pixels dans le rail réduit. Les deux états ont été
inspectés visuellement, sans chevauchement des commandes. Le panneau a été
laissé ouvert et la conversation existante préservée. L'API a été redéployée.
