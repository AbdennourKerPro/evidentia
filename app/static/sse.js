/** Server-Sent Events over fetch (EventSource cannot POST), plus progress wording. */

function parseBlock(block) {
  let event = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (line.startsWith(":")) continue;
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  if (!data.length) return null;
  try { return { event, data: JSON.parse(data.join("\n")) }; } catch { return null; }
}

/** Feed arbitrary text chunks; get back the complete events they finish. */
export function createSseParser() {
  let buffer = "";
  return {
    feed(chunk) {
      buffer += chunk.replace(/\r\n/g, "\n");
      const events = [];
      let end;
      while ((end = buffer.indexOf("\n\n")) >= 0) {
        const parsed = parseBlock(buffer.slice(0, end));
        buffer = buffer.slice(end + 2);
        if (parsed) events.push(parsed);
      }
      return events;
    },
  };
}

/** Short French status line for a progress event, or null for silent events. */
export function describeProgress(event, data) {
  switch (event) {
    case "discovery.started": return "Sélection des sources…";
    case "discovery.tool_call": {
      const args = data.arguments || {};
      if (data.name === "search_arxiv") return `Recherche arXiv : « ${args.query ?? ""} »`;
      if (data.name === "lookup_arxiv_ids") return "Vérification des identifiants arXiv…";
      if (data.name === "add_paper_to_conversation") return `Ajout de l’article ${args.arxiv_id ?? ""}…`;
      if (data.name === "finish_source_selection") return "Sources sélectionnées.";
      return null;
    }
    case "ingestion.progress":
      if (data.stage === "downloading") return "Téléchargement du PDF…";
      if (data.stage === "converting") return "Conversion du PDF (peut prendre une minute)…";
      if (data.stage === "indexing") return `Indexation (${data.chunks ?? "?"} passages)…`;
      return null;
    case "queue.waiting": return "Serveur occupé : votre question est en file d’attente…";
    case "question.rewritten": return `Question reformulée : « ${data.question} »`;
    case "paper.added": return `Article ajouté : ${data.title}`;
    case "answer.started": return "Rédaction de la réponse…";
    default: return null;
  }
}

/** POST a JSON body and call onEvent for each SSE event until the stream ends. */
export async function postEventStream(url, body, onEvent, fetchImpl = fetch) {
  const response = await fetchImpl(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = "";
    try { detail = (await response.json()).detail; } catch { /* not JSON */ }
    throw new Error(typeof detail === "string" && detail ? detail : "La requête a échoué.");
  }
  const parser = createSseParser();
  const decoder = new TextDecoder();
  const reader = response.body.getReader();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (const item of parser.feed(decoder.decode(value, { stream: true }))) onEvent(item.event, item.data);
  }
}
