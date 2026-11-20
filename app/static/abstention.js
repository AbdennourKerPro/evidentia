/** What an abstention or a partial answer can tell the user, from data the server already returns.
 * Pure functions (no DOM) so they can be tested with Node.
 *
 * Coverage comes from an LLM judge: it is a hint, not a guarantee. The wording stays cautious
 * ("ne semblent pas couvrir") and aspects are shown as the server wrote them.
 */

/** Question aspects the final coverage check found (`covered`) or did not find (`missing`) in the passages. */
export function coverageGaps(execution) {
  const checks = (execution && execution.coverage_checks) || [];
  const last = checks[checks.length - 1];
  if (!last || last.sufficient === null || last.sufficient === undefined) return { covered: [], missing: [] };
  const covered = [], missing = [];
  for (const requirement of last.requirements || []) {
    if (!requirement || !requirement.aspect) continue;
    (requirement.reference && requirement.quote ? covered : missing).push(requirement.aspect);
  }
  return { covered, missing };
}

/** A published answer that the coverage check says leaves part of the question open. */
export function isPartialAnswer(payload) {
  return !!payload && !payload.abstained && coverageGaps(payload.execution).missing.length > 0;
}

/**
 * One-click follow-ups after an abstention. `context`:
 *   agentMode, inviteMode, discoveryAuto, allArticlesSelected (booleans)
 * Always starts with the actions that change the least.
 */
export function abstentionActions(payload, context) {
  const actions = [{ id: "rephrase", label: "Reformuler la question" }];
  if (payload && payload.evidence && payload.evidence.length > 0) {
    actions.push({ id: "show-evidence", label: "Voir les passages les plus proches" });
  }
  if (!context.agentMode && !context.allArticlesSelected) {
    actions.push({ id: "all-articles", label: "Réessayer avec tous les articles" });
  }
  if (!context.inviteMode) {
    if (!context.agentMode) actions.push({ id: "agent-mode", label: "Chercher sur arXiv (mode agent)" });
    else if (!context.discoveryAuto) actions.push({ id: "enable-discovery", label: "Autoriser l’agent à ajouter des articles" });
  }
  return actions;
}

/** Plain-language summary line for the gap list. */
export function gapSummary(gaps, abstained) {
  if (gaps.missing.length === 0) return "";
  if (!abstained) return "Réponse partielle : les passages retrouvés ne semblent pas couvrir tous les points de la question.";
  return gaps.covered.length
    ? "Des preuves ont été trouvées pour une partie de la question seulement."
    : "Aucun point de la question n’est couvert par les passages retrouvés.";
}
