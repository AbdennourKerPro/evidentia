const state = {
  cases: [],
  filteredCases: [],
  selectedId: null,
  dirty: false,
  saving: false,
};

const elements = {
  sidebar: document.querySelector(".queue-sidebar"),
  caseList: document.querySelector("#case-list"),
  queueEmpty: document.querySelector("#queue-empty"),
  search: document.querySelector("#case-search"),
  statusFilter: document.querySelector("#status-filter"),
  categoryFilter: document.querySelector("#category-filter"),
  verifiedCount: document.querySelector("#verified-count"),
  totalCount: document.querySelector("#total-count"),
  remainingCount: document.querySelector("#remaining-count"),
  progressPercent: document.querySelector("#progress-percent"),
  progressTrack: document.querySelector(".progress-track"),
  progressBar: document.querySelector("#progress-bar"),
  loadingState: document.querySelector("#loading-state"),
  errorState: document.querySelector("#error-state"),
  errorMessage: document.querySelector("#error-message"),
  retryButton: document.querySelector("#retry-button"),
  editor: document.querySelector("#case-editor"),
  reviewActions: document.querySelector("#review-actions"),
  form: document.querySelector("#review-form"),
  caseId: document.querySelector("#case-id"),
  caseStatus: document.querySelector("#case-status"),
  metadata: document.querySelector("#case-metadata"),
  casePosition: document.querySelector("#case-position"),
  saveState: document.querySelector("#save-state"),
  question: document.querySelector("#question-field"),
  answer: document.querySelector("#reference-answer-field"),
  terms: document.querySelector("#terms-field"),
  notes: document.querySelector("#notes-field"),
  answerBlock: document.querySelector("#answer-field-block"),
  termsBlock: document.querySelector("#terms-field-block"),
  abstentionPanel: document.querySelector("#abstention-panel"),
  expectedDocuments: document.querySelector("#expected-documents"),
  evidenceList: document.querySelector("#evidence-list"),
  noEvidence: document.querySelector("#no-evidence"),
  previousButton: document.querySelector("#previous-case"),
  nextButton: document.querySelector("#next-case"),
  saveDraftButton: document.querySelector("#save-draft-button"),
  verifyButton: document.querySelector("#verify-button"),
  mobileQueueToggle: document.querySelector("#mobile-queue-toggle"),
  caseTemplate: document.querySelector("#case-list-item-template"),
  evidenceTemplate: document.querySelector("#evidence-template"),
};

const CATEGORY_LABELS = {
  factoid: "Fait",
  method: "Méthode",
  comparison: "Comparaison",
  abstention: "Abstention",
};

elements.retryButton.addEventListener("click", loadCases);
elements.search.addEventListener("input", () => applyFilters());
elements.statusFilter.addEventListener("change", () => applyFilters());
elements.categoryFilter.addEventListener("change", () => applyFilters());
elements.previousButton.addEventListener("click", () => moveSelection(-1));
elements.nextButton.addEventListener("click", () => moveSelection(1));
elements.saveDraftButton.addEventListener("click", () => saveCase("needs_human_review"));
elements.verifyButton.addEventListener("click", () => saveCase("verified"));
elements.mobileQueueToggle.addEventListener("click", () => {
  elements.sidebar.classList.toggle("is-open");
});

elements.form.addEventListener("input", () => {
  state.dirty = true;
  setSaveState("Modifications non enregistrées");
});

window.addEventListener("beforeunload", (event) => {
  if (!state.dirty) return;
  event.preventDefault();
});

document.addEventListener("keydown", (event) => {
  if (!event.altKey) return;
  if (event.key === "ArrowLeft") moveSelection(-1);
  if (event.key === "ArrowRight") moveSelection(1);
});

async function loadCases() {
  elements.loadingState.hidden = false;
  elements.errorState.hidden = true;
  elements.editor.hidden = true;
  elements.reviewActions.hidden = true;

  try {
    const response = await fetch("/evaluation/cases");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Le serveur a refusé la requête.");
    }

    state.cases = payload.cases;
    renderProgress();
    applyFilters({ preserveSelection: false });
    elements.loadingState.hidden = true;
  } catch (error) {
    elements.loadingState.hidden = true;
    elements.errorState.hidden = false;
    elements.errorMessage.textContent = error.message;
  }
}

function applyFilters(options = { preserveSelection: true }) {
  const query = elements.search.value.trim().toLocaleLowerCase("fr");
  const status = elements.statusFilter.value;
  const category = elements.categoryFilter.value;

  state.filteredCases = state.cases.filter(({ case: evaluationCase }) => {
    const matchesQuery =
      !query ||
      evaluationCase.id.toLocaleLowerCase("fr").includes(query) ||
      evaluationCase.question.toLocaleLowerCase("fr").includes(query);
    const matchesStatus = status === "all" || evaluationCase.review_status === status;
    const matchesCategory = category === "all" || evaluationCase.category === category;
    return matchesQuery && matchesStatus && matchesCategory;
  });

  renderQueue();
  if (state.filteredCases.length === 0) {
    state.selectedId = null;
    elements.editor.hidden = true;
    elements.reviewActions.hidden = true;
    updateNavigation();
    return;
  }

  const selectionStillVisible = state.filteredCases.some(
    ({ case: evaluationCase }) => evaluationCase.id === state.selectedId
  );
  if (!options.preserveSelection || !selectionStillVisible) {
    const nextToReview = state.filteredCases.find(
      ({ case: evaluationCase }) =>
        evaluationCase.review_status === "needs_human_review"
    );
    selectCase((nextToReview || state.filteredCases[0]).case.id, { force: true });
  } else {
    renderQueue();
    updateNavigation();
  }
}

function renderProgress() {
  const total = state.cases.length;
  const verified = state.cases.filter(
    ({ case: evaluationCase }) => evaluationCase.review_status === "verified"
  ).length;
  const remaining = total - verified;
  const percent = total === 0 ? 0 : Math.round((verified / total) * 100);

  elements.verifiedCount.textContent = verified;
  elements.totalCount.textContent = total;
  elements.remainingCount.textContent = `${remaining} cas à vérifier`;
  elements.progressPercent.textContent = `${percent} %`;
  elements.progressBar.style.width = `${percent}%`;
  elements.progressTrack.setAttribute("aria-valuenow", String(percent));
}

function renderQueue() {
  const items = state.filteredCases.map((reviewCase) => {
    const fragment = elements.caseTemplate.content.cloneNode(true);
    const button = fragment.querySelector(".case-list-item");
    const evaluationCase = reviewCase.case;
    button.dataset.caseId = evaluationCase.id;
    button.classList.toggle("is-active", evaluationCase.id === state.selectedId);
    button.classList.toggle("is-verified", evaluationCase.review_status === "verified");
    fragment.querySelector(".list-meta").textContent =
      `${CATEGORY_LABELS[evaluationCase.category]} · ${evaluationCase.language.toUpperCase()}`;
    fragment.querySelector(".list-question").textContent = evaluationCase.question;
    fragment.querySelector(".list-id").textContent = evaluationCase.id;
    button.addEventListener("click", () => selectCase(evaluationCase.id));
    return fragment;
  });

  elements.caseList.replaceChildren(...items);
  elements.queueEmpty.hidden = items.length !== 0;
}

function selectCase(caseId, { force = false } = {}) {
  if (!force && caseId !== state.selectedId && !canDiscardChanges()) return;
  const reviewCase = state.cases.find(({ case: item }) => item.id === caseId);
  if (!reviewCase) return;

  state.selectedId = caseId;
  state.dirty = false;
  renderQueue();
  renderCase(reviewCase);
  updateNavigation();
  elements.sidebar.classList.remove("is-open");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderCase(reviewCase) {
  const evaluationCase = reviewCase.case;
  const isAbstention = evaluationCase.should_abstain;

  elements.editor.hidden = false;
  elements.reviewActions.hidden = false;
  elements.caseId.textContent = evaluationCase.id;
  elements.caseStatus.textContent =
    evaluationCase.review_status === "verified" ? "Vérifié" : "À vérifier";
  elements.caseStatus.classList.toggle(
    "is-verified",
    evaluationCase.review_status === "verified"
  );
  elements.question.value = evaluationCase.question;
  elements.answer.value = evaluationCase.reference_answer;
  elements.terms.value = formatTermGroups(evaluationCase.required_answer_terms);
  elements.notes.value = evaluationCase.notes;
  elements.answerBlock.hidden = isAbstention;
  elements.termsBlock.hidden = isAbstention;
  elements.abstentionPanel.hidden = !isAbstention;
  elements.verifyButton.firstChild.textContent =
    evaluationCase.review_status === "verified" ? "Confirmer la validation " : "Valider ce cas ";

  elements.metadata.replaceChildren(
    makeMetadataChip(CATEGORY_LABELS[evaluationCase.category]),
    makeMetadataChip(evaluationCase.language === "fr" ? "Français" : "Anglais"),
    makeMetadataChip(
      evaluationCase.scope_document_ids === null
        ? "Corpus complet"
        : `${evaluationCase.scope_document_ids.length} source(s) ciblée(s)`
    ),
    makeMetadataChip(isAbstention ? "Abstention attendue" : "Réponse attendue")
  );

  const documentChips = evaluationCase.expected_document_ids.map((documentId) => {
    const chip = document.createElement("span");
    chip.className = "document-chip";
    chip.textContent = documentId;
    return chip;
  });
  if (documentChips.length === 0) {
    const chip = document.createElement("span");
    chip.className = "document-chip";
    chip.textContent = "aucun — abstention";
    documentChips.push(chip);
  }
  elements.expectedDocuments.replaceChildren(...documentChips);

  renderEvidence(reviewCase.evidence_excerpts);
  setSaveState(evaluationCase.review_status === "verified" ? "Cas vérifié" : "Prêt pour la revue", true);
  updatePosition();
}

function renderEvidence(excerpts) {
  const cards = excerpts.map((excerpt, index) => {
    const fragment = elements.evidenceTemplate.content.cloneNode(true);
    fragment.querySelector(".evidence-index").textContent = `P${index + 1}`;
    fragment.querySelector(".evidence-document").textContent = excerpt.document_id;
    fragment.querySelector(".evidence-location").textContent =
      `${excerpt.section} · page ${excerpt.page} · chunk ${excerpt.chunk_id}`;
    fragment.querySelector(".evidence-text").textContent = excerpt.text;

    const phrases = excerpt.match_phrases.map((phrase) => {
      const chip = document.createElement("span");
      chip.className = "target-phrase";
      chip.textContent = phrase;
      return chip;
    });
    fragment.querySelector(".target-phrase-list").append(...phrases);
    return fragment;
  });

  elements.evidenceList.replaceChildren(...cards);
  elements.noEvidence.hidden = cards.length !== 0;
}

async function saveCase(reviewStatus) {
  if (state.saving || !state.selectedId) return;
  const current = state.cases.find(({ case: item }) => item.id === state.selectedId);
  if (!current) return;

  const payload = {
    question: elements.question.value.trim(),
    reference_answer: current.case.should_abstain ? "" : elements.answer.value.trim(),
    required_answer_terms: current.case.should_abstain ? [] : parseTermGroups(elements.terms.value),
    notes: elements.notes.value.trim(),
    review_status: reviewStatus,
  };

  const validationError = validatePayload(payload, current.case.should_abstain);
  if (validationError) {
    setSaveState(validationError);
    return;
  }

  setSaving(true);
  setSaveState("Enregistrement…");
  try {
    const response = await fetch(`/evaluation/cases/${encodeURIComponent(state.selectedId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const savedCase = await response.json();
    if (!response.ok) {
      throw new Error(formatApiError(savedCase.detail));
    }

    const caseIndex = state.cases.findIndex(({ case: item }) => item.id === state.selectedId);
    state.cases[caseIndex] = savedCase;
    state.dirty = false;
    renderProgress();
    const savedLabel = reviewStatus === "verified" ? "Validation enregistrée" : "Brouillon enregistré";
    applyFilters();
    if (state.selectedId === savedCase.case.id) {
      // Refresh the badge and action label when the saved case remains visible.
      renderCase(savedCase);
      setSaveState(savedLabel, true);
    }
  } catch (error) {
    setSaveState(`Erreur : ${error.message}`);
  } finally {
    setSaving(false);
  }
}

function validatePayload(payload, shouldAbstain) {
  if (payload.question.length < 3) return "La question est trop courte.";
  if (!shouldAbstain && !payload.reference_answer) return "La réponse de référence est obligatoire.";
  if (!shouldAbstain && payload.required_answer_terms.length === 0) {
    return "Ajoutez au moins un groupe de termes attendus.";
  }
  return null;
}

function parseTermGroups(value) {
  return value
    .split("\n")
    .map((line) => line.split("|").map((term) => term.trim()).filter(Boolean))
    .filter((group) => group.length > 0);
}

function formatTermGroups(groups) {
  return groups.map((group) => group.join(" | ")).join("\n");
}

function formatApiError(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => item.msg).join(" · ");
  return "La modification a été refusée.";
}

function moveSelection(offset) {
  if (state.filteredCases.length === 0) return;
  const currentIndex = state.filteredCases.findIndex(
    ({ case: evaluationCase }) => evaluationCase.id === state.selectedId
  );
  const nextIndex = Math.min(
    Math.max(currentIndex + offset, 0),
    state.filteredCases.length - 1
  );
  if (nextIndex === currentIndex) return;
  selectCase(state.filteredCases[nextIndex].case.id);
}

function updateNavigation() {
  const currentIndex = state.filteredCases.findIndex(
    ({ case: evaluationCase }) => evaluationCase.id === state.selectedId
  );
  elements.previousButton.disabled = currentIndex <= 0;
  elements.nextButton.disabled =
    currentIndex < 0 || currentIndex >= state.filteredCases.length - 1;
  updatePosition();
}

function updatePosition() {
  const currentIndex = state.filteredCases.findIndex(
    ({ case: evaluationCase }) => evaluationCase.id === state.selectedId
  );
  elements.casePosition.textContent = currentIndex < 0
    ? `${state.filteredCases.length} cas affichés`
    : `Cas ${currentIndex + 1} sur ${state.filteredCases.length}`;
}

function canDiscardChanges() {
  return !state.dirty || window.confirm("Abandonner les modifications non enregistrées ?");
}

function setSaving(saving) {
  state.saving = saving;
  elements.saveDraftButton.disabled = saving;
  elements.verifyButton.disabled = saving;
}

function setSaveState(message, saved = false) {
  elements.saveState.textContent = message;
  elements.saveState.classList.toggle("is-saved", saved);
}

function makeMetadataChip(text) {
  const chip = document.createElement("span");
  chip.className = "metadata-chip";
  chip.textContent = text;
  return chip;
}

loadCases();
