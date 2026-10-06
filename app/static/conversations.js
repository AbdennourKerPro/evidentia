/** Local display history, not LLM memory. Requests still contain one question.
 * Full API payloads retain citations and chunk text when reopening a chat.
 * Quota/corruption never silently deletes or overwrites existing history.
 */
export const HISTORY_KEY = "evidentia.conversations.v1";

function validMessage(message) {
  if (message?.role === "user" || message?.role === "error") return typeof message.text === "string";
  return message?.role === "assistant" && typeof message.payload?.answer === "string"
    && Array.isArray(message.payload.citations) && Array.isArray(message.payload.evidence);
}

export function createConversationStore(storage, onError = () => {}) {
  let conversations = [], activeId = null, canPersist = true;
  try {
    if (!storage) throw new Error("Storage unavailable");
    const raw = storage.getItem(HISTORY_KEY);
    if (raw) {
      const saved = JSON.parse(raw);
      if (saved.version !== 1 || !Array.isArray(saved.conversations) || !saved.conversations.every(chat =>
        typeof chat.id === "string" && typeof chat.title === "string" && typeof chat.updatedAt === "number"
        && Array.isArray(chat.messages) && chat.messages.every(validMessage)
      )) throw new Error("Invalid history");
      conversations = saved.conversations;
      activeId = conversations.some(chat => chat.id === saved.activeId) ? saved.activeId : conversations[0]?.id;
    }
  } catch {
    canPersist = false;
    onError("Historique local indisponible ou illisible. Les nouveaux échanges restent dans cet onglet ; l’ancien historique n’est pas écrasé.");
  }
  const save = () => {
    if (!canPersist) return;
    try { storage.setItem(HISTORY_KEY, JSON.stringify({ version: 1, activeId, conversations })); }
    catch { onError("Stockage local indisponible ou plein. Les nouveaux échanges restent dans cet onglet ; aucune conversation n’a été supprimée."); }
  };
  const active = () => conversations.find(chat => chat.id === activeId);
  const create = () => {
    if (active()?.messages.length === 0) return active();
    const chat = { id: globalThis.crypto.randomUUID(), title: "Nouvelle conversation", updatedAt: Date.now(), messages: [] };
    conversations.unshift(chat); activeId = chat.id; save(); return chat;
  };
  if (!active()) create();
  return {
    active,
    list: () => [...conversations].sort((a, b) => b.updatedAt - a.updatedAt),
    create,
    activate(id) { if (conversations.some(chat => chat.id === id)) { activeId = id; save(); } return active(); },
    append(message) {
      if (!validMessage(message)) throw new Error("Invalid conversation message");
      const chat = active();
      if (message.role === "user" && !chat.messages.some(item => item.role === "user")) chat.title = message.text.slice(0, 80);
      chat.messages.push(message); chat.updatedAt = Date.now(); save();
    },
  };
}
