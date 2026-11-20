/** Why an answer looks the way it does: what the source-selection agent and the pipeline decided.
 * Pure functions (no DOM, no imports) used by the execution details and by tester feedback.
 */

const STRATEGY_LABELS = { per_source: "recherche parallèle par article", global: "recherche globale" };

function toolCallLabel(call) {
  const args = call.arguments || {};
  const detail = call.name === "search_arxiv" ? `« ${args.query ?? ""} »`
    : call.name === "lookup_arxiv_ids" ? (args.arxiv_ids || []).join(", ")
    : call.name === "add_paper_to_conversation" ? (args.arxiv_id ?? "")
    : call.name === "finish_source_selection" ? "" : "";
  return `${call.name}${detail ? ` ${detail}` : ""} ${call.ok ? "✓" : `✗ ${call.error ?? ""}`.trim()}`;
}

/** Readable lines for the "Détails de l'exécution" panel, in the order the decisions were taken. */
export function diagnosisLines(execution) {
  if (!execution) return [];
  const lines = [];
  if (execution.rewritten_question) lines.push(`Question reformulée : « ${execution.rewritten_question} »`);
  const discovery = execution.discovery;
  if (discovery) {
    if (discovery.name_hints?.length) lines.push(`Noms de la question sans article correspondant (indice automatique) : ${discovery.name_hints.join(", ")}`);
    for (const call of discovery.tool_calls || []) lines.push(`Agent · ${toolCallLabel(call)}`);
    if (discovery.guard_triggered) lines.push("Garde-fou : l’agent a déclaré un sujet sans article sans avoir cherché ; il a été renvoyé chercher une fois.");
    if (discovery.uncovered_subjects?.length) lines.push(`Sujets sans article attaché : ${discovery.uncovered_subjects.join(", ")}`);
    if (discovery.rationale) lines.push(`Justification de l’agent : ${discovery.rationale}`);
    if (discovery.added_document_ids?.length) lines.push(`Articles ajoutés : ${discovery.added_document_ids.join(", ")}`);
    if (discovery.focus_document_ids?.length) lines.push(`Articles ciblés par l’agent : ${discovery.focus_document_ids.join(", ")}`);
    lines.push(`Agent : ${discovery.model_calls} appel(s), arrêt « ${discovery.stopped_reason} »`);
  }
  const targets = execution.detected_document_ids || [];
  lines.push(`Stratégie : ${STRATEGY_LABELS[execution.strategy] || execution.strategy}${targets.length ? ` — articles visés : ${targets.join(", ")}` : ""}`);
  return lines;
}

/** Compact diagnosis stored with tester feedback. `gaps` is { covered, missing } from the coverage check. */
export function feedbackContext(payload, gaps = { covered: [], missing: [] }) {
  const execution = payload.execution || {};
  const discovery = execution.discovery;
  const unique = (values) => [...new Set(values)];
  return {
    abstained: !!payload.abstained,
    reason: payload.reason || null,
    strategy: execution.strategy || null,
    retrieval_profile: execution.retrieval_profile || null,
    expected_language: execution.expected_language || null,
    rewritten_question: execution.rewritten_question || null,
    detected_documents: execution.detected_document_ids || [],
    evidence_documents: unique((payload.evidence || []).map(item => item.document_id)),
    cited_documents: unique((payload.citations || []).map(item => item.document_id)),
    correction_attempted: !!execution.correction_attempted,
    contract_issues: execution.contract_issues || [],
    coverage: { covered: gaps.covered, missing: gaps.missing },
    discovery: discovery ? {
      stopped_reason: discovery.stopped_reason,
      model_calls: discovery.model_calls,
      rationale: discovery.rationale || null,
      tool_calls: (discovery.tool_calls || []).map(call => ({
        name: call.name, ok: call.ok, error: call.error || null,
        argument: call.arguments?.query ?? call.arguments?.arxiv_id ?? call.arguments?.arxiv_ids ?? null,
      })),
      added: discovery.added_document_ids || [],
      focus: discovery.focus_document_ids || [],
      uncovered_subjects: discovery.uncovered_subjects || [],
      name_hints: discovery.name_hints || [],
      guard_triggered: !!discovery.guard_triggered,
      conversation_documents: discovery.conversation_document_ids || [],
    } : null,
  };
}
