/** Progress of one answer: only the current task is visible; an arrow unfolds the whole path.
 * DOM construction only (`document.createElement`), so it can be tested with a minimal fake DOM.
 */

/**
 * Builds <details><summary>current task</summary><ol>all tasks</ol></details> inside `container`
 * (replacing its content) and returns { add(text), lines() }.
 */
export function createProgress(container, firstLine = "Envoi de la question…") {
  const details = document.createElement("details");
  details.className = "progress";
  const summary = document.createElement("summary");
  summary.className = "progress-summary";
  const pulse = document.createElement("span");
  pulse.className = "progress-pulse";
  const current = document.createElement("span");
  current.className = "progress-current";
  summary.append(pulse, current);
  const list = document.createElement("ol");
  list.className = "progress-lines";
  details.append(summary, list);
  container.replaceChildren(details);

  const lines = [];
  let last = null;
  const add = (text) => {
    if (!text || text === lines[lines.length - 1]) return false; // parallel nodes repeat the same step
    if (last) last.className = "is-done";
    last = document.createElement("li");
    last.className = "is-current";
    last.textContent = text;
    list.append(last);
    lines.push(text);
    current.textContent = text;
    return true;
  };
  add(firstLine);
  return { add, lines: () => [...lines] };
}
