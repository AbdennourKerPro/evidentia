const form = document.querySelector("#question-form");
const questionInput = document.querySelector("#question");
const submitButton = document.querySelector("#submit-button");
const formStatus = document.querySelector("#form-status");
const welcome = document.querySelector("#welcome");
const welcomeScope = document.querySelector("#welcome-scope");
const messageList = document.querySelector("#message-list");
const sourceList = document.querySelector("#source-list");
const sourceSummary = document.querySelector("#source-summary");
const sourceError = document.querySelector("#source-error");
const documentCount = document.querySelector("#document-count");
const topbarScope = document.querySelector("#topbar-scope");
const selectAllButton = document.querySelector("#select-all-button");
const clearSelectionButton = document.querySelector("#clear-selection-button");
const newChatButton = document.querySelector("#new-chat-button");
const corpusToggle = document.querySelector("#corpus-toggle");
const corpusSidebar = document.querySelector("#corpus-sidebar");
const userTemplate = document.querySelector("#user-message-template");
const assistantTemplate = document.querySelector("#assistant-message-template");

let documents = [];
let selectedDocumentIds = new Set();
let requestInProgress = false;

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendQuestion(questionInput.value.trim());
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

questionInput.addEventListener("input", resizeComposer);

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    const scope = button.dataset.scope ? button.dataset.scope.split(",") : null;
    if (scope) {
      selectedDocumentIds = new Set(scope);
      renderSourceList();
    }
    sendQuestion(button.dataset.question);
  });
});

selectAllButton.addEventListener("click", () => {
  selectedDocumentIds = new Set(documents.map((document) => document.document_id));
  renderSourceList();
});

clearSelectionButton.addEventListener("click", () => {
  selectedDocumentIds.clear();
  renderSourceList();
});

newChatButton.addEventListener("click", () => {
  messageList.replaceChildren();
  welcome.hidden = false;
  questionInput.value = "";
  resizeComposer();
  setStatus("Sélectionnez au moins un article · Entrée pour envoyer");
  questionInput.focus();
});

corpusToggle.addEventListener("click", () => {
  const sidebarIsOpen = corpusSidebar.classList.toggle("is-open");
  corpusToggle.setAttribute("aria-expanded", String(sidebarIsOpen));
});

async function loadCorpus() {
  try {
    const response = await fetch("/arxiv/documents");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Le corpus est indisponible.");
    }

    documents = payload.documents;
    selectedDocumentIds = new Set(documents.map((document) => document.document_id));
    renderSourceList();
  } catch (error) {
    sourceError.hidden = false;
    sourceError.textContent = `Impossible de charger les articles : ${error.message}`;
    sourceSummary.textContent = "Aucun article disponible.";
    documentCount.textContent = "0";
    updateScopeText();
  }
}

function renderSourceList() {
  sourceList.replaceChildren(
    ...documents.map((document) => makeSourceOption(document))
  );
  documentCount.textContent = String(documents.length);
  updateScopeText();
  updateComposerAvailability();
}

function makeSourceOption(indexedDocument) {
  const option = document.createElement("label");
  option.className = "source-option";

  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = selectedDocumentIds.has(indexedDocument.document_id);
  checkbox.addEventListener("change", () => {
    if (checkbox.checked) {
      selectedDocumentIds.add(indexedDocument.document_id);
    } else {
      selectedDocumentIds.delete(indexedDocument.document_id);
    }
    updateScopeText();
    updateComposerAvailability();
  });

  const text = document.createElement("span");
  const title = document.createElement("span");
  title.className = "source-title";
  title.textContent = indexedDocument.title;
  const meta = document.createElement("span");
  meta.className = "source-meta";
  meta.textContent = `${indexedDocument.indexed_chunks} chunks · ${indexedDocument.document_id}`;
  text.append(title, meta);
  option.append(checkbox, text);
  return option;
}

function updateScopeText() {
  const selectedDocuments = getSelectedDocuments();
  const selectedChunkCount = selectedDocuments.reduce(
    (total, document) => total + document.indexed_chunks,
    0
  );

  if (selectedDocuments.length === 0) {
    sourceSummary.textContent = "Aucun article sélectionné.";
    topbarScope.textContent = "Aucune source sélectionnée";
    welcomeScope.textContent = "Sélectionnez au moins un article dans la bibliothèque pour commencer.";
    return;
  }

  const selectionText = `${selectedDocuments.length} article${selectedDocuments.length > 1 ? "s" : ""} sélectionné${selectedDocuments.length > 1 ? "s" : ""}`;
  sourceSummary.textContent = `${selectionText} · ${selectedChunkCount} chunks disponibles`;
  topbarScope.textContent = selectionText;
  welcomeScope.textContent = `Les questions interrogeront ${selectionText.toLowerCase()} (${selectedChunkCount} chunks). Sélectionnez plusieurs articles pour les comparer.`;
}

function updateComposerAvailability() {
  const canAsk = selectedDocumentIds.size > 0 && !requestInProgress;
  submitButton.disabled = !canAsk;
  questionInput.disabled = requestInProgress || selectedDocumentIds.size === 0;
}

async function sendQuestion(question) {
  if (requestInProgress || question.length < 3 || selectedDocumentIds.size === 0) {
    return;
  }

  const selectedDocuments = getSelectedDocuments();
  welcome.hidden = true;
  appendUserMessage(question, describeScope(selectedDocuments));
  const assistantMessage = appendLoadingMessage();
  questionInput.value = "";
  resizeComposer();
  setLoading(true);
  scrollToMessage(assistantMessage);

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        limit: 3,
        document_ids: [...selectedDocumentIds],
      }),
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "La requête n’a pas pu aboutir.");
    }

    renderAssistantMessage(assistantMessage, payload);
    setStatus("Réponse locale · les citations ouvrent les preuves récupérées");
  } catch (error) {
    renderError(assistantMessage, error.message);
    setStatus(`Erreur : ${error.message}`, true);
  } finally {
    setLoading(false);
    questionInput.focus();
  }
}

function getSelectedDocuments() {
  return documents.filter((document) => selectedDocumentIds.has(document.document_id));
}

function describeScope(selectedDocuments) {
  if (selectedDocuments.length === 1) {
    return `Corpus : ${selectedDocuments[0].title}`;
  }
  return `Corpus : ${selectedDocuments.length} articles sélectionnés`;
}

function appendUserMessage(question, scope) {
  const fragment = userTemplate.content.cloneNode(true);
  const message = fragment.querySelector(".message");
  fragment.querySelector(".user-content").textContent = question;
  fragment.querySelector(".message-scope").textContent = scope;
  messageList.append(fragment);
  return messageList.lastElementChild;
}

function appendLoadingMessage() {
  const fragment = assistantTemplate.content.cloneNode(true);
  const content = fragment.querySelector(".answer-content");
  const typing = document.createElement("span");
  typing.className = "typing";
  typing.setAttribute("aria-label", "Génération en cours");
  typing.append(document.createElement("span"), document.createElement("span"), document.createElement("span"));
  content.append(typing);
  fragment.querySelector(".answer-state").hidden = true;
  fragment.querySelector(".sources-block").hidden = true;
  messageList.append(fragment);
  return messageList.lastElementChild;
}

function renderAssistantMessage(message, payload) {
  const answer = message.querySelector(".answer-content");
  const state = message.querySelector(".answer-state");
  const sourcesBlock = message.querySelector(".sources-block");
  const citations = message.querySelector(".citation-list");
  const disclosure = message.querySelector(".evidence-disclosure");
  const summary = message.querySelector(".sources-summary");
  const evidenceList = message.querySelector(".evidence-list");

  answer.textContent = payload.answer;
  state.hidden = false;
  state.textContent = payload.abstained
    ? `Abstention — ${payload.reason}`
    : "Réponse fondée sur les passages récupérés";
  state.classList.toggle("is-abstained", payload.abstained);
  sourcesBlock.hidden = false;

  const citedReferences = new Set(payload.citations.map((citation) => citation.reference));
  const evidenceByReference = new Map(
    payload.evidence.map((evidence) => [evidence.reference, evidence])
  );
  const sourceCount = new Set(payload.evidence.map((evidence) => evidence.document_id)).size;

  if (payload.citations.length === 0) {
    const emptyCitation = document.createElement("p");
    emptyCitation.className = "empty-citation";
    emptyCitation.textContent = "Aucune citation validée";
    citations.append(emptyCitation);
  } else {
    payload.citations.forEach((citation) => {
      citations.append(makeCitationButton(citation, disclosure, evidenceByReference));
    });
  }

  summary.textContent = `Voir ${payload.evidence.length} chunks provenant de ${sourceCount} article${sourceCount > 1 ? "s" : ""}`;
  evidenceList.append(
    ...payload.evidence.map((evidence) => makeEvidencePanel(evidence, citedReferences))
  );
}

function makeCitationButton(citation, disclosure, evidenceByReference) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "citation-button";

  const reference = document.createElement("span");
  reference.className = "citation-reference";
  reference.textContent = `[${citation.reference}] `;
  button.append(reference, `${shortTitle(citation.title)} · p. ${citation.page}`);

  button.addEventListener("click", () => {
    disclosure.open = true;
    const evidence = evidenceByReference.get(citation.reference);
    const target = disclosure.querySelector(`[data-reference="${citation.reference}"]`);
    if (!evidence || !target) {
      return;
    }
    target.open = true;
    target.classList.add("is-targeted");
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    window.setTimeout(() => target.classList.remove("is-targeted"), 1600);
  });

  return button;
}

function makeEvidencePanel(evidence, citedReferences) {
  const panel = document.createElement("details");
  panel.className = "evidence-item";
  panel.dataset.reference = evidence.reference;

  const summary = document.createElement("summary");
  const reference = document.createElement("span");
  reference.className = "reference";
  reference.textContent = `[${evidence.reference}]`;

  const main = document.createElement("span");
  main.className = "summary-main";
  const title = document.createElement("span");
  title.className = "summary-title";
  title.textContent = evidence.section;
  const meta = document.createElement("span");
  meta.className = "summary-meta";
  meta.textContent = `${shortTitle(evidence.title)} · page ${evidence.page} · similarité ${evidence.score.toFixed(3)}`;
  main.append(title, meta);
  summary.append(reference, main);

  if (citedReferences.has(evidence.reference)) {
    const badge = document.createElement("span");
    badge.className = "cited-badge";
    badge.textContent = "CITÉ";
    summary.append(badge);
  }

  const body = document.createElement("div");
  body.className = "chunk-body";
  const provenance = document.createElement("p");
  provenance.className = "chunk-provenance";
  provenance.textContent = `${evidence.title} · ${evidence.document_id} · page ${evidence.page}`;
  const text = document.createElement("pre");
  text.className = "chunk-text";
  text.textContent = evidence.text;
  body.append(provenance, text);

  panel.append(summary, body);
  return panel;
}

function shortTitle(title) {
  return title.length > 42 ? `${title.slice(0, 39)}…` : title;
}

function renderError(message, errorMessage) {
  message.querySelector(".answer-content").textContent = "Je n’ai pas pu produire de réponse.";
  const state = message.querySelector(".answer-state");
  state.hidden = false;
  state.className = "answer-state is-error";
  state.textContent = errorMessage;
  message.querySelector(".sources-block").hidden = true;
}

function setLoading(isLoading) {
  requestInProgress = isLoading;
  updateComposerAvailability();
  if (isLoading) {
    setStatus("Recherche des preuves et génération locale en cours…");
  }
}

function setStatus(message, isError = false) {
  formStatus.textContent = message;
  formStatus.classList.toggle("error", isError);
}

function resizeComposer() {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 180)}px`;
}

function scrollToMessage(message) {
  message.scrollIntoView({ behavior: "smooth", block: "start" });
}

updateComposerAvailability();
loadCorpus();
