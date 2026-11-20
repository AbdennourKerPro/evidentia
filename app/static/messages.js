/** User-facing wording: French versions of server messages, and links to the original PDF.
 * The server keeps stable English strings (they appear in evaluation reports); the interface
 * translates the ones it knows and leaves any other message untouched.
 */

/** Server errors identified by their stable `code`. */
const ERROR_BY_CODE = {
  server_busy: "Le serveur est occupé : réessayez dans un instant.",
  internal_error: "La question n’a pas pu être traitée.",
  missing_api_key: "La clé d’API du modèle n’est pas configurée sur le serveur.",
  invalid_api_key: "La clé d’API du modèle a été refusée.",
  model_access_denied: "Ce projet d’API n’a pas accès au modèle demandé.",
  model_unavailable: "Le modèle demandé n’est pas disponible pour ce projet.",
  upstream_rate_limit: "Quota ou limite de débit du modèle atteint. Réessayez plus tard.",
  llm_timeout: "Le modèle a mis trop de temps à répondre. Réessayez.",
  llm_connection_error: "Connexion au modèle impossible.",
  upstream_api_error: "La requête au modèle a échoué.",
  incomplete_llm_response: "Le modèle a renvoyé une réponse incomplète. Réessayez.",
  failed_llm_response: "Le modèle n’a pas pu produire de réponse. Réessayez.",
  empty_llm_response: "Le modèle n’a renvoyé aucun texte. Réessayez.",
  llm_refusal: "Le modèle a refusé de répondre à cette question.",
};

/** Exact English sentences, for messages that have no code (abstention reasons, HTTP details). */
const EXACT = new Map([
  ["No evidence was found in the selected arXiv articles.", "Aucun passage pertinent n’a été trouvé dans les articles sélectionnés."],
  ["The language model judged the retrieved evidence insufficient.", "Les passages retrouvés ont été jugés insuffisants pour répondre."],
  ["No paper is attached to this conversation yet.", "Aucun article n’est encore rattaché à cette conversation."],
  ["The agent searched arXiv but found no suitable paper.", "L’agent a cherché sur arXiv mais n’a trouvé aucun article adapté."],
  ["The agent did not select any paper for this question.", "L’agent n’a sélectionné aucun article pour cette question."],
  ["The server is busy: please retry in a moment.", ERROR_BY_CODE.server_busy],
  ["The question could not be processed.", ERROR_BY_CODE.internal_error],
  ["The conversation is full.", "La conversation contient déjà le nombre maximal d’articles."],
  ["Paper not attached.", "Cet article n’est pas rattaché à la conversation."],
  ["The conversation id must be a UUID.", "L’identifiant de conversation est invalide."],
  ["Not found", "Introuvable."],
  ["Qdrant is unavailable.", "La base de recherche est indisponible."],
  ["arXiv returned an unreadable response.", "arXiv a renvoyé une réponse illisible."],
  ["The downloaded file is not a PDF.", "Le fichier téléchargé n’est pas un PDF."],
]);

/** English sentences with a variable tail: [prefix, French prefix]. */
const PREFIXES = [
  ["The generated answer failed the response contract: ", "La réponse générée ne respecte pas le contrat de réponse : "],
  ["The agent could not attach any paper ", "L’agent n’a pu rattacher aucun article "],
  ["arXiv is unreachable", "arXiv est injoignable"],
  ["Not a valid modern arXiv identifier", "Identifiant arXiv invalide"],
  ["arXiv has no paper with identifier ", "arXiv ne contient aucun article avec l’identifiant "],
  ["arXiv answered HTTP ", "arXiv a répondu avec le code HTTP "],
  ["PDF download failed", "Le téléchargement du PDF a échoué"],
  ["The PDF is larger than ", "Le PDF dépasse la taille maximale de "],
];

/** French wording for a server message; unknown messages are returned unchanged. */
export function localizeMessage(text, code = null) {
  if (code && ERROR_BY_CODE[code]) return ERROR_BY_CODE[code];
  if (typeof text !== "string") return text;
  if (EXACT.has(text)) return EXACT.get(text);
  for (const [english, french] of PREFIXES) {
    if (text.startsWith(english)) return french + text.slice(english.length);
  }
  return text;
}

/** Abstention label shown under an answer. */
export function describeAbstention(reason) {
  return reason ? `Abstention — ${localizeMessage(reason)}` : "Abstention";
}

/** Fixed corpus: source id → arXiv id and version (data/sources/arxiv_papers.json). */
const FIXED_PAPERS = {
  "clip-2021": "2103.00020v1",
  "dinov2-2023": "2304.07193v1",
  "sam-2023": "2304.02643v1",
  "blip2-2023": "2301.12597v1",
  "llava-2023": "2304.08485v1",
};

/** arXiv identifier with version (e.g. "2312.00752v2") for an indexed document, or null. */
export function arxivIdentifier(documentId) {
  if (FIXED_PAPERS[documentId]) return FIXED_PAPERS[documentId];
  const dynamic = /^arxiv-(\d{4}\.\d{4,5}v\d+)$/.exec(documentId || "");
  return dynamic ? dynamic[1] : null;
}

/** Link to the page of the original PDF that holds a passage, or null if the paper is unknown. */
export function arxivPdfUrl(documentId, page) {
  const identifier = arxivIdentifier(documentId);
  if (!identifier) return null;
  const anchor = Number.isInteger(page) && page > 0 ? `#page=${page}` : "";
  return `https://arxiv.org/pdf/${identifier}${anchor}`;
}
