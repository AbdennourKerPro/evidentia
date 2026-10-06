import { renderMarkdown } from "./markdown.js?v=conversations-20260930";
import { createConversationStore } from "./conversations.js?v=conversations-20260930";

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
const selectAllButton = document.querySelector("#select-all-button");
const clearSelectionButton = document.querySelector("#clear-selection-button");
const newChatButton = document.querySelector("#new-chat-button");
const userTemplate = document.querySelector("#user-message-template");
const assistantTemplate = document.querySelector("#assistant-message-template");
const pipelineMode = document.querySelector("#pipeline-mode");
const llmStatus = document.querySelector("#llm-status");
const shell = document.querySelector("#app-shell");
const sidebarToggle = document.querySelector("#sidebar-toggle");
const conversationList = document.querySelector("#conversation-list");
const settingsDialog = document.querySelector("#settings-dialog");
const settingsButton = document.querySelector("#settings-button");
const diagnosticsToggle = document.querySelector("#show-diagnostics");

const PREFERENCES_KEY = "evidentia.preferences.v1";
let storage;
try { storage = window.localStorage; } catch { storage = null; }
function showStorageNotice(message) {
  const notice = document.querySelector("#storage-notice");
  notice.textContent = message;
  notice.hidden = false;
  formStatus.textContent = message;
}
function readPreferences() {
  try { return JSON.parse(storage?.getItem(PREFERENCES_KEY) || "{}") || {}; }
  catch { return {}; }
}
const preferences = readPreferences();
const history = createConversationStore(storage, showStorageNotice);
let documents = [];
let selectedDocumentIds = new Set();
let requestInProgress = false;

function persistPreferences() {
  try {
    storage?.setItem(PREFERENCES_KEY, JSON.stringify({
      documentIds: [...selectedDocumentIds],
      pipeline: pipelineMode.value,
      diagnostics: diagnosticsToggle.checked,
      collapsed: shell.classList.contains("sidebar-collapsed"),
    }));
  } catch { showStorageNotice("Les réglages ne peuvent pas être enregistrés dans ce navigateur."); }
}

pipelineMode.value = preferences.pipeline === "baseline" ? "baseline" : "agentic";
diagnosticsToggle.checked = preferences.diagnostics === true;
setSidebarCollapsed(typeof preferences.collapsed === "boolean"
  ? preferences.collapsed : window.matchMedia("(max-width: 760px)").matches);

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
    if (requestInProgress) return;
    if (button.dataset.scope) {
      const available = new Set(documents.map(document => document.document_id));
      selectedDocumentIds = new Set(button.dataset.scope.split(",").filter(id => available.has(id)));
      renderSourceList();
      persistPreferences();
    }
    sendQuestion(button.dataset.question);
  });
});
selectAllButton.addEventListener("click", () => {
  selectedDocumentIds = new Set(documents.map(document => document.document_id));
  renderSourceList(); persistPreferences();
});
clearSelectionButton.addEventListener("click", () => {
  selectedDocumentIds.clear();
  renderSourceList(); persistPreferences();
});
newChatButton.addEventListener("click", () => {
  if (requestInProgress) return;
  history.create();
  renderConversation();
  questionInput.value = ""; resizeComposer();
  setStatus("Vérifiez les informations dans les sources.");
  questionInput.focus();
});
sidebarToggle.addEventListener("click", () => {
  setSidebarCollapsed(!shell.classList.contains("sidebar-collapsed"));
  persistPreferences();
});
settingsButton.addEventListener("click", () => settingsDialog.showModal());
settingsDialog.addEventListener("click", (event) => {
  const bounds = settingsDialog.getBoundingClientRect();
  if (event.target === settingsDialog &&
      (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) {
    settingsDialog.close();
  }
});
pipelineMode.addEventListener("change", persistPreferences);
diagnosticsToggle.addEventListener("change", () => {
  applyDiagnosticsVisibility(); persistPreferences();
});

function setSidebarCollapsed(collapsed) {
  shell.classList.toggle("sidebar-collapsed", collapsed);
  sidebarToggle.setAttribute("aria-expanded", String(!collapsed));
  const label = collapsed ? "Ouvrir le panneau" : "Réduire le panneau";
  sidebarToggle.setAttribute("aria-label", label);
  sidebarToggle.title = label;
}

function renderConversationList() {
  const chats = history.list();
  document.querySelector("#history-empty").hidden = chats.some(chat => chat.messages.length);
  conversationList.replaceChildren(...chats.filter(chat => chat.messages.length).map(chat => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "conversation-item";
    button.textContent = chat.title;
    button.title = chat.title;
    button.disabled = requestInProgress;
    if (chat.id === history.active().id) button.setAttribute("aria-current", "page");
    button.addEventListener("click", () => {
      if (requestInProgress) return;
      history.activate(chat.id);
      renderConversation();
      if (window.matchMedia("(max-width: 760px)").matches) setSidebarCollapsed(true);
    });
    return button;
  }));
}

function renderConversation() {
  messageList.replaceChildren();
  const messages = history.active().messages;
  welcome.hidden = messages.length > 0;
  for (const entry of messages) {
    if (entry.role === "user") appendUserMessage(entry.text, entry.scope || "");
    else {
      const message = appendLoadingMessage();
      if (entry.role === "assistant") renderAssistantMessage(message, entry.payload);
      else renderError(message, entry.text);
    }
  }
  renderConversationList();
  if (messageList.lastElementChild) scrollToMessage(messageList.lastElementChild);
}

function applyDiagnosticsVisibility() {
  for (const node of messageList.querySelectorAll(".execution-disclosure, .llm-usage")) {
    node.hidden = !diagnosticsToggle.checked || node.dataset.available !== "true";
  }
  for (const state of messageList.querySelectorAll(".answer-state:not(.llm-usage)")) {
    if (!state.classList.contains("is-abstained") && !state.classList.contains("is-error")) {
      state.hidden = !diagnosticsToggle.checked;
    }
  }
}

async function loadCorpus() {
  try {
    const response = await fetch("/arxiv/documents");
    const payload = await response.json();
    if (!response.ok || !Array.isArray(payload.documents)) throw new Error(payload.detail || "Le corpus est indisponible.");
    documents = payload.documents;
    const available = new Set(documents.map(document => document.document_id));
    selectedDocumentIds = Array.isArray(preferences.documentIds)
      ? new Set(preferences.documentIds.filter(id => available.has(id))) : available;
    renderSourceList();
  } catch (error) {
    sourceError.hidden = false;
    sourceError.textContent = `Impossible de charger les articles : ${error.message}`;
    documentCount.textContent = "0";
    updateScopeText(); updateComposerAvailability();
  }
}

function renderSourceList() {
  sourceList.replaceChildren(...documents.map(document => makeSourceOption(document)));
  documentCount.textContent = String(documents.length);
  updateScopeText(); updateComposerAvailability();
}

function updateScopeText() {
  const selected = getSelectedDocuments();
  const chunks = selected.reduce((total, document) => total + document.indexed_chunks, 0);
  sourceSummary.textContent = selected.length
    ? `${selected.length} article(s) sélectionné(s) · ${chunks} passages disponibles`
    : "Aucun article sélectionné.";
  welcomeScope.textContent = selected.length
    ? "Posez une question sur vos articles scientifiques."
    : "Choisissez au moins un article dans les réglages pour commencer.";
  if (!selected.length) setStatus("Choisissez vos sources via l’engrenage en bas à gauche.");
}

function updateComposerAvailability() {
  const canAsk = selectedDocumentIds.size > 0 && !requestInProgress;
  submitButton.disabled = !canAsk;
  questionInput.disabled = !canAsk;
  newChatButton.disabled = requestInProgress;
  pipelineMode.disabled = requestInProgress;
  selectAllButton.disabled = requestInProgress;
  clearSelectionButton.disabled = requestInProgress;
  sourceList.querySelectorAll("input").forEach(input => { input.disabled = requestInProgress; });
  document.querySelectorAll("[data-question]").forEach(button => { button.disabled = !canAsk; });
}

async function sendQuestion(question) {
  if (requestInProgress || question.length < 3 || selectedDocumentIds.size === 0) return;
  const scope = describeScope(getSelectedDocuments());
  const documentIds = [...selectedDocumentIds];
  history.append({ role: "user", text: question, scope });
  welcome.hidden = true;
  appendUserMessage(question, scope);
  const assistantMessage = appendLoadingMessage();
  questionInput.value = ""; resizeComposer();
  setLoading(true);
  scrollToMessage(assistantMessage);
  try {
    const endpoint = pipelineMode.value === "agentic" ? "/agentic/ask" : "/ask";
    const response = await fetch(endpoint, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, limit: 5, document_ids: documentIds }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "La réponse n’a pas pu être produite.");
    history.append({ role: "assistant", payload });
    renderAssistantMessage(assistantMessage, payload);
    setStatus("Vérifiez les informations dans les sources.");
  } catch (error) {
    history.append({ role: "error", text: error.message });
    renderError(assistantMessage, error.message);
    setStatus(error.message, true);
  } finally {
    setLoading(false);
    if (!questionInput.disabled) questionInput.focus();
  }
}

function setLoading(isLoading) {
  requestInProgress = isLoading;
  updateComposerAvailability();
  renderConversationList();
  if (isLoading) setStatus("Recherche dans les articles…");
}

async function loadLlmStatus() {
  try {
    const response = await fetch("/llm/status");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Configuration LLM indisponible.");
    }
    llmStatus.textContent = payload.configured
      ? `${payload.model_id} · API configurée`
      : "Clé API à configurer";
    llmStatus.title = payload.configured
      ? "Configuration détectée. L'accès au modèle sera vérifié au premier appel."
      : payload.configuration_error;
  } catch (error) {
    llmStatus.textContent = "Configuration API à vérifier";
    llmStatus.title = error.message;
  }
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
    persistPreferences();
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
  const executionDisclosure = message.querySelector(".execution-disclosure");

  renderMarkdown(answer, payload.answer);
  state.hidden = !payload.abstained && !diagnosticsToggle.checked;
  state.textContent = payload.abstained
    ? `Abstention — ${payload.reason}`
    : "Réponse fondée sur les passages récupérés";
  state.classList.toggle("is-abstained", payload.abstained);
  if (payload.llm) {
    const usage = message.querySelector(".llm-usage");
    const tokens = payload.llm.total_tokens === null
      ? "usage incomplet"
      : `${payload.llm.total_tokens} tokens`;
    usage.dataset.available = "true";
    usage.hidden = !diagnosticsToggle.checked;
    usage.textContent = `${payload.llm.model_id} · ${payload.llm.call_count} appel(s) LLM · ${tokens}`;
  }
  sourcesBlock.hidden = payload.evidence.length === 0 && payload.citations.length === 0;

  if (payload.execution) {
    renderExecutionTrace(executionDisclosure, payload.execution);
    executionDisclosure.dataset.available = "true";
    executionDisclosure.hidden = !diagnosticsToggle.checked;
  }

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

function renderExecutionTrace(disclosure, execution) {
  disclosure.hidden = false;
  const summary = disclosure.querySelector(".execution-summary");
  const steps = disclosure.querySelector(".execution-steps");
  const strategyLabel = execution.strategy === "per_source"
    ? "recherche parallèle par article"
    : "recherche globale";
  const languageLabels = { en: "anglais", fr: "français", zh: "chinois" };
  const languageLabel = languageLabels[execution.expected_language]
    || execution.expected_language;
  const contractLabel = execution.response_contract_valid
    ? "contrat validé"
    : execution.response_publishable
      ? "complétude factuelle partielle"
      : "contrat rejeté";
  const correctionLabel = execution.correction_attempted
    ? "1 correction"
    : "sans correction";
  const plannedFactCount = execution.fact_plan?.length || 0;
  const factLabel = execution.fact_planning_fallback
    ? "plan factuel indisponible"
    : `${execution.covered_planned_facts}/${plannedFactCount} faits validés`;

  summary.textContent = `${strategyLabel} · ${languageLabel} · ${contractLabel} · ${factLabel} · ${correctionLabel} · ${execution.candidate_chunks} candidats · ${execution.selected_chunks} chunks transmis au LLM`;
  const factItems = (execution.fact_plan || []).map((fact) => {
    const item = document.createElement("li");
    const node = document.createElement("code");
    node.textContent = fact.id;
    const detail = document.createElement("span");
    detail.textContent = `${fact.claim} · ancrage « ${fact.anchor} » · ${fact.references.join(", ")}`;
    item.append(node, detail);
    return item;
  });
  steps.replaceChildren(
    ...execution.steps.map((step) => {
      const item = document.createElement("li");
      const node = document.createElement("code");
      node.textContent = step.node;
      const detail = document.createElement("span");
      detail.textContent = `${step.detail} · ${step.duration_ms} ms`;
      item.append(node, detail);
      return item;
    }),
    ...factItems
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
    const target = [...disclosure.querySelectorAll("[data-reference]")].find(panel => panel.dataset.reference === citation.reference);
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
  meta.textContent = `${shortTitle(evidence.title)} · page ${evidence.page} · score de classement ${evidence.score.toFixed(3)}`;
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

renderConversation();
updateComposerAvailability();
loadCorpus();
loadLlmStatus();
