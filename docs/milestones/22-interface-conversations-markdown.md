# Milestone 22 — Interface épurée, Markdown et conversations locales

Date : 30 septembre 2026. Cette étape ne modifie ni les prompts, ni la recherche,
ni les contrats de validation du RAG déployé. Le profil reste `candidate_recovery`.

## 1. Disposition et parcours utilisateur

La vue principale contient la conversation et la saisie. L'en-tête technique a
disparu : pas de sélecteur de pipeline ni de lien d'évaluation dans cette vue.
La palette utilise un fond sombre et des nuances de gris/blanc ; aucun téléchargement
de polices ou de bibliothèque externe n'est nécessaire.

Le panneau à gauche affiche le bouton de nouvelle conversation en haut, puis les
conversations précédentes. Il peut être réduit en une barre de 56 pixels, avec
le bouton de nouvelle conversation et l'engrenage toujours accessibles. Sur petit
écran il est réduit par défaut, puis s'ouvre au-dessus du contenu ; choisir une
conversation le referme. Le bouton reste accessible au clavier et expose son
état par `aria-expanded`.

L'engrenage en bas à gauche ouvre une fenêtre de réglages : sélection d'articles,
choix LangGraph/Baseline, option d'affichage des détails techniques, état du modèle,
lien vers l'évaluation du RAG et information sur l'historique local. Le dialogue
natif `<dialog>` gère le focus et la fermeture par Échap ; une croix et le clic
hors de la fenêtre permettent aussi de le fermer.

Les citations restent sous la réponse. Un clic ouvre le passage correspondant,
avec sa provenance et son texte complet, y compris les chunks ajoutés par la
récupération de couverture. Les traces LangGraph et le comptage de tokens restent
disponibles mais sont masqués par défaut, avec une option dans les réglages.

## 2. Pourquoi les étoiles et tirets apparaissaient

L'ancien code faisait `answer.textContent = payload.answer`. Cela protégeait contre
l'injection HTML, mais affichait littéralement le Markdown produit par le modèle.
Le nouveau rendu construit des éléments DOM : `<strong>` pour le gras, `<ul>` et
`<li>` pour les listes, `<ol>` pour les étapes numérotées, etc. Le CSS leur donne
les espacements et marqueurs attendus.

Le support est un sous-ensemble explicite de Markdown, pas une implémentation
exhaustive de CommonMark : paragraphes, gras, italique avec étoiles, titres,
listes simples et imbriquées, citations, code inline/blocs, tableaux simples et
liens HTTP(S). Les références `[S1]` restent intactes. Les passages sources ne
sont pas interprétés en Markdown : leur texte brut reste visible.

Le HTML brut n'est jamais interprété. Tous les contenus sont insérés par nœuds
texte ou `textContent`, sans `innerHTML`. Un `<script>` ou un `<img onerror=...>`
reste du texte. Les URL `javascript:` et `data:` ne deviennent pas des liens.
Les liens HTTP(S) s'ouvrent avec `noopener noreferrer`. Cela limite le rendu
aux éléments autorisés et ne donne pas au texte du modèle le droit de modifier
la page. Les formules LaTeX restent du texte : cette étape n'ajoute pas MathJax.

## 3. Les scripts et leurs responsabilités

### `app/static/index.html`

Définit trois zones : panneau de conversations, zone de dialogue/saisie et dialogue
de réglages. Les templates de messages restent réutilisables. Le script principal
est chargé comme module ES pour séparer les responsabilités. Les URL des assets
portent un suffixe de version pour éviter de mélanger une ancienne feuille de
style et un nouveau script après déploiement.

### `app/static/styles.css`

Définit les couleurs, la largeur du panneau, sa réduction, le comportement mobile,
les bulles des questions, le rendu typographique Markdown, les citations, les
preuves, le dialogue de réglages et la saisie fixe. Les animations sont désactivées
si le navigateur indique une préférence de mouvement réduit. Les états masqués
utilisent une règle `[hidden]` explicite pour éviter qu'un style de grille/flex
ne les rende visibles par erreur.

### `app/static/markdown.js`

`inlineTokens` reconnaît le gras, l'italique, le code et les liens, en conservant
les autres caractères. `markdownBlocks` reconnaît les blocs et délègue les listes
à `readList`. Les fonctions de rendu transforment ces structures en éléments DOM
autorisés. `renderMarkdown(target, text)` remplace le contenu de la réponse. La
récursion est bornée pour ne pas traiter un nombre illimité d'imbrications.

Ce module ne dépend ni de l'API, ni du stockage. Sa lecture des liens et tableaux
est volontairement simple : les constructions Markdown très complexes peuvent
rester du texte ou ne pas reproduire exactement un moteur CommonMark complet.

### `app/static/conversations.js`

`createConversationStore(storage, onError)` gère l'historique sous la clé
`evidentia.conversations.v1`. `active`, `list`, `create`, `activate` et `append`
permettent de créer, sélectionner et compléter les conversations. Le titre est
extrait de la première question, sans appel LLM supplémentaire.

Chaque réponse conserve le payload API, avec les citations et le texte entier des
chunks, afin de restaurer les preuves après rechargement. L'enveloppe versionnée
contient la conversation active. Les formats inconnus ou illisibles ne sont pas
écrasés ; un avertissement est affiché et la nouvelle session peut fonctionner en
mémoire. Une erreur de quota ne supprime aucune ancienne conversation.

Le stockage est local au navigateur et à l'origine `localhost:8000`. Il n'est ni
chiffré par l'application, ni partagé entre appareils, ni sauvegardé côté serveur.
Effacer les données du site peut effacer l'historique. Si le quota est atteint,
les nouveaux messages restent en mémoire dans l'onglet et ne sont pas garantis
après fermeture ; l'application le signale. Aucune clé API n'y est enregistrée.

### `app/static/app.js`

Relie les contrôles à l'historique, aux réglages et aux appels API. Charge les
articles par `/arxiv/documents` et l'état du modèle par `/llm/status`. Enregistre
les préférences sous `evidentia.preferences.v1` : articles, pipeline, diagnostics
et état du panneau. Les articles sauvegardés sont intersectés avec ceux du corpus
réel pour ne pas envoyer d'identifiants devenus indisponibles.

`sendQuestion` enregistre la question, affiche l'attente, appelle `/agentic/ask`
ou `/ask`, enregistre la réponse complète puis l'affiche avec Markdown et sources.
Les changements de conversation sont bloqués pendant une génération pour ne pas
attribuer une réponse à une autre conversation. Le message d'erreur est aussi
conservé si l'appel échoue. Les fonctions de citations et de provenance restent
basées sur le payload API, non sur du HTML inventé par le modèle.

Important : cet historique est une **mémoire d'affichage**, pas une nouvelle
mémoire conversationnelle du LLM. Chaque appel continue à envoyer la question
courante et les articles sélectionnés, pas tous les échanges précédents.
Interpréter « et celui-ci ? » à partir d'un ancien message nécessiterait une
évolution séparée de l'API et du graphe, non incluse dans cette refonte visuelle.

## 4. Vérification

`tests/ui.test.mjs` utilise Node, sans dépendance de navigateur ou API distante.
Les 18 tests vérifient le Markdown, ses listes, tableaux, code, références et
liens ; ils contrôlent aussi l'absence d'interprétation de HTML, la création et
restauration des conversations, les preuves conservées, les erreurs de quota,
les formats de stockage inconnus et le fonctionnement sans stockage disponible.
Le petit DOM simulé sert uniquement à vérifier les éléments construits.

Un test HTTP supplémentaire dans `tests/test_api_llm.py` vérifie la présence du
panneau et du dialogue, l'absence de l'ancien en-tête, la localisation des réglages
et la distribution des deux nouveaux modules JavaScript avec leur bon type MIME.
La suite Python complète contient maintenant 157 tests réussis.

Commandes depuis le dossier de l'application :

```powershell
node --test tests/ui.test.mjs
node --check app/static/app.js
docker compose run --rm --no-deps api python -m unittest discover -s tests -q
```

Le déploiement utilise l'image incrémentale, sans changement des dépendances ni
réindexation de Qdrant. Seul le service API est recréé. Les tests de rendu ne
remplacent pas une vérification réelle du navigateur, effectuée après déploiement.

### Vérification dans le navigateur après déploiement

Une question réelle comparant les prompts de CLIP et de SAM a été soumise via
l'interface. La réponse affiche sept éléments en gras et sept éléments de liste,
sans marqueurs `**` visibles. Les citations restent sous la réponse : l'ouverture
de S6 permet de lire le chunk SAM complet et sa provenance.

La vérification confirme également :

- L'ancien en-tête a disparu ; les articles, la pipeline et le lien de revue du
  benchmark sont accessibles dans les réglages.
- Une nouvelle conversation conserve la précédente dans l'historique. Revenir à
  cette conversation, puis recharger la page, restaure sa réponse et ses sources.
- Le panneau réduit conserve les commandes de déploiement du panneau, de nouvelle
  conversation et de réglages.
- L'option de détails techniques affiche ou masque la trace et les statistiques
  LLM ; ces détails sont masqués par défaut.

À la fin du test, les cinq articles sont sélectionnés, le panneau est ouvert et
les détails techniques sont masqués. Aucun historique existant n'a été supprimé.
Le test interactif utilise le LLM configuré ; les tests automatisés UI et Python
utilisent des doubles de test, sans appels LLM réels. La campagne des trente cas
n'a pas été relancée, puisque la pipeline backend n'a pas été modifiée.

La vérification visuelle a porté sur la fenêtre de bureau disponible. Les règles
CSS pour petits écrans sont présentes, mais n'ont pas fait l'objet d'une
vérification interactive à une résolution mobile.
