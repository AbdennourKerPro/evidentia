// No browser dependencies or API calls: test the pure UI modules with Node.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { webcrypto } from "node:crypto";
if (!globalThis.crypto) globalThis.crypto = webcrypto;

async function loadModule(name) {
  const source = await readFile(new URL(`../app/static/${name}.js`, import.meta.url), "utf8");
  return import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
}
const { markdownBlocks, inlineTokens, renderMarkdown } = await loadModule("markdown");
const { createConversationStore, HISTORY_KEY } = await loadModule("conversations");

class Element {
  constructor(tag, text = "") { this.tag = tag; this.text = text; this.children = []; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; this.text = ""; }
  set textContent(text) { this.text = text; this.children = []; }
  get textContent() { return this.text + this.children.map(node => node.textContent).join(""); }
}
const findTags = (node, tag) => [node, ...node.children.flatMap(child => findTags(child, tag))].filter(child => child.tag === tag);
function render(text) {
  globalThis.document = { createElement: tag => new Element(tag), createTextNode: text => new Element("#text", text) };
  const root = new Element("div"); renderMarkdown(root, text); return root;
}
function memoryStorage(initial = null) {
  const data = new Map(initial ? [[HISTORY_KEY, initial]] : []);
  return { getItem: key => data.get(key) ?? null, setItem: (key, value) => data.set(key, value) };
}

test("bold and bullet points become semantic HTML elements", () => {
  const root = render("- **CLIP:** classification [S1]\n- **SAM:** segmentation [S2]");
  assert.equal(findTags(root, "ul").length, 1);
  assert.equal(findTags(root, "li").length, 2);
  assert.equal(findTags(root, "strong").length, 2);
  assert.ok(!root.textContent.includes("**"));
  assert.ok(root.textContent.includes("[S1]"));
});
test("ordered lists preserve the starting number", () => {
  const root = render("3. first\n4. second");
  assert.equal(findTags(root, "ol")[0].start, 3);
  assert.equal(findTags(root, "li").length, 2);
});
test("nested list and multiline item preserve their content", () => {
  const blocks = markdownBlocks("- Parent\n  continuation\n  - Child\n- Next");
  assert.equal(blocks[0].items.length, 2);
  assert.equal(blocks[0].items[0].children[0].items[0].text, "Child");
  assert.match(blocks[0].items[0].text, /continuation/);
});
test("paragraphs, headings and quotes remain separate blocks", () => {
  assert.deepEqual(markdownBlocks("## Title\n\nParagraph\n\n> Quote").map(block => block.type), ["h2", "p", "blockquote"]);
});
test("fenced code stays literal, including markup", () => {
  const root = render("```python\n**not bold** <script>\n```");
  assert.equal(findTags(root, "strong").length, 0);
  assert.equal(findTags(root, "code")[0].textContent, "**not bold** <script>");
});
test("tables create header and body cells", () => {
  const root = render("| Model | Role |\n| --- | --- |\n| **A** | Bridge |");
  assert.equal(findTags(root, "table").length, 1);
  assert.equal(findTags(root, "th").length, 2);
  assert.equal(findTags(root, "td").length, 2);
});
test("raw HTML and unsafe links never become executable elements", () => {
  const root = render('<img src=x onerror=alert(1)> [click](javascript:alert) [data](data:text/html,hi)');
  assert.equal(findTags(root, "img").length, 0);
  assert.equal(findTags(root, "a").length, 0);
  assert.ok(root.textContent.includes("<img"));
});
test("HTTP links have isolated external targets", () => {
  const root = render("[paper](https://arxiv.org/abs/1234)");
  const link = findTags(root, "a")[0];
  assert.equal(link.href, "https://arxiv.org/abs/1234");
  assert.equal(link.rel, "noopener noreferrer");
});
test("inline code and emphasis can be contained in bold", () => {
  const root = render("**Use `code`** and *emphasis*");
  assert.equal(findTags(root, "strong").length, 1);
  assert.equal(findTags(root, "code").length, 1);
  assert.equal(findTags(root, "em").length, 1);
});
test("unmatched Markdown and LaTeX remain readable text", () => {
  assert.ok(inlineTokens("Unclosed **bold").some(token => token.type === "text"));
  assert.ok(render("\\(x + y\\)").textContent.includes("\\(x + y\\)"));
});

const user = text => ({ role: "user", text, scope: "Corpus" });
const assistant = { role: "assistant", payload: { answer: "**Answer** [S1]", citations: [{ reference: "S1" }], evidence: [{ reference: "S1", text: "Full chunk" }] } };
test("new conversation gets its title from the first question", () => {
  const store = createConversationStore(memoryStorage());
  store.append(user("First question")); store.append(user("Second question"));
  assert.equal(store.active().title, "First question");
});
test("empty new conversations are reused, not duplicated", () => {
  const store = createConversationStore(memoryStorage());
  store.create(); store.create(); assert.equal(store.list().length, 1);
});
test("switching conversations restores isolated messages", () => {
  const store = createConversationStore(memoryStorage());
  const firstId = store.active().id; store.append(user("First")); store.append(assistant);
  store.create(); store.append(user("Second"));
  assert.equal(store.list().length, 2);
  store.activate(firstId); assert.equal(store.active().messages.length, 2);
  assert.equal(store.active().messages[1].payload.evidence[0].text, "Full chunk");
});
test("reloading preserves answers, citation data, full chunks and active chat", () => {
  const storage = memoryStorage(), store = createConversationStore(storage);
  store.append(user("Question")); store.append(assistant);
  const loaded = createConversationStore(storage);
  assert.equal(loaded.active().id, store.active().id);
  assert.equal(loaded.active().messages[1].payload.answer, assistant.payload.answer);
  assert.equal(loaded.active().messages[1].payload.evidence[0].text, "Full chunk");
});
test("storage quota failures keep in-memory history and report the problem", () => {
  const warnings = [], storage = memoryStorage(), store = createConversationStore(storage, text => warnings.push(text));
  store.append(user("Saved")); const saved = storage.getItem(HISTORY_KEY);
  storage.setItem = () => { throw new Error("QuotaExceeded"); };
  store.append(assistant);
  assert.equal(store.active().messages.length, 2);
  assert.equal(storage.getItem(HISTORY_KEY), saved);
  assert.ok(warnings.length > 0);
});
test("corrupt or unknown-version history is never overwritten", () => {
  for (const raw of ["not JSON", JSON.stringify({ version: 99, conversations: [] })]) {
    const storage = memoryStorage(raw), store = createConversationStore(storage);
    store.append(user("New in-memory question"));
    assert.equal(storage.getItem(HISTORY_KEY), raw);
  }
});
test("unavailable storage still permits an in-memory conversation", () => {
  const warnings = [], store = createConversationStore(null, text => warnings.push(text));
  store.append(user("Question")); assert.equal(store.active().messages.length, 1);
  assert.equal(warnings.length, 1);
});
test("invalid messages cannot corrupt the history", () => {
  const store = createConversationStore(memoryStorage());
  assert.throws(() => store.append({ role: "assistant", payload: {} }), /Invalid/);
  assert.equal(store.active().messages.length, 0);
});
