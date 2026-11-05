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

const { createSseParser, describeProgress, postEventStream } = await loadModule("sse");

test("SSE parser reassembles events split across chunks and ignores keep-alives", () => {
  const parser = createSseParser();
  assert.deepEqual(parser.feed('event: paper.added\ndata: {"title": "Ma'), []);
  assert.deepEqual(parser.feed('mba"}\n\n: ping\n\nevent: answer\r\ndata: {"a":1}\r\n\r\n'), [
    { event: "paper.added", data: { title: "Mamba" } },
    { event: "answer", data: { a: 1 } },
  ]);
  assert.deepEqual(parser.feed("event: broken\ndata: {nope\n\n"), []);
});

test("progress lines describe the agent's tool calls and ingestion stages", () => {
  assert.equal(describeProgress("discovery.tool_call", { name: "search_arxiv", arguments: { query: "mamba" } }), "Recherche arXiv : « mamba »");
  assert.match(describeProgress("ingestion.progress", { stage: "indexing", chunks: 12 }), /12 passages/);
  assert.match(describeProgress("paper.added", { title: "Mamba" }), /Mamba/);
  assert.equal(describeProgress("question.rewritten", { question: "Explain SimCLR's loss" }), "Question reformulée : « Explain SimCLR's loss »");
  assert.equal(describeProgress("discovery.tool_result", { ok: true }), null);
  assert.equal(describeProgress("done", {}), null);
});

test("postEventStream delivers events in order and rejects HTTP errors with the server detail", async () => {
  const encoder = new TextEncoder();
  const chunks = [encoder.encode("event: discovery.started\ndata: {}\n\nevent: ans"), encoder.encode('wer\ndata: {"answer":"x"}\n\n')];
  const stream = { ok: true, body: { getReader: () => { let i = 0; return { read: async () => i < chunks.length ? { value: chunks[i++], done: false } : { done: true } }; } } };
  const seen = [];
  await postEventStream("/x", { q: 1 }, (event, data) => seen.push([event, data]), async () => stream);
  assert.deepEqual(seen, [["discovery.started", {}], ["answer", { answer: "x" }]]);
  await assert.rejects(
    postEventStream("/x", {}, () => {}, async () => ({ ok: false, json: async () => ({ detail: "UUID requis" }) })),
    /UUID requis/,
  );
});

test("removing a conversation activates another one or creates a fresh chat", () => {
  const storage = memoryStorage();
  const store = createConversationStore(storage);
  const first = store.active();
  store.append({ role: "user", text: "first question" });
  const second = store.create();
  store.append({ role: "user", text: "second question" });
  assert.equal(store.list().length, 2);
  const after = store.remove(second.id);
  assert.equal(after.id, first.id);
  assert.equal(store.list().length, 1);
  const fresh = store.remove(first.id);
  assert.equal(fresh.messages.length, 0);
  assert.notEqual(fresh.id, first.id);
  assert.equal(JSON.parse(storage.getItem(HISTORY_KEY)).conversations.length, 1);
  assert.equal(store.remove("unknown").id, fresh.id);
});

const { setMathRenderer } = await loadModule("markdown");

test("math is tokenised before Markdown: backslashes, asterisks and underscores survive", () => {
  const math = (text) => inlineTokens(text).filter(token => token.type === "math");
  assert.deepEqual(math("loss \\(2(N-1)\\) negatives"), [{ type: "math", text: "2(N-1)", display: false }]);
  assert.deepEqual(math("$\\ell_{i,j} = a*b$"), [{ type: "math", text: "\\ell_{i,j} = a*b", display: false }]);
  assert.deepEqual(math("see \\[ \\sum_i x_i \\] here"), [{ type: "math", text: "\\sum_i x_i", display: true }]);
  assert.deepEqual(math("$$ a_1 + b_2 $$"), [{ type: "math", text: "a_1 + b_2", display: true }]);
  // `\[` used to lose its backslash; an escaped bracket outside math still works.
  assert.equal(inlineTokens("\\[S1\\]").map(token => token.text ?? "").join(""), "[S1]");
});

test("currency and unmatched dollars are not mistaken for math", () => {
  for (const text of ["It costs $5 and $6 in total", "Price: $ 5 or $", "only one $ sign", "\\(unclosed"]) {
    assert.equal(inlineTokens(text).some(token => token.type === "math"), false, text);
  }
});

test("math is typeset by the registered renderer, with TeX source as the fallback", () => {
  globalThis.document = { createElement: tag => new Element(tag), createTextNode: text => new Element("#text", text) };
  const calls = [];
  setMathRenderer((element, tex, display) => { calls.push([tex, display]); element.textContent = `<<${tex}>>`; });
  const root = new Element("div");
  renderMarkdown(root, "Loss: \\(x^2\\) and\n\n\\[y=1\\]");
  assert.deepEqual(calls, [["x^2", false], ["y=1", true]]);
  assert.ok(root.textContent.includes("<<x^2>>"));
  setMathRenderer(() => { throw new Error("bad tex"); });
  const failing = new Element("div");
  renderMarkdown(failing, "Loss: \\(\\badmacro\\)");
  assert.ok(failing.textContent.includes("\\(\\badmacro\\)"));
  setMathRenderer(null);
  const plain = new Element("div");
  renderMarkdown(plain, "$a_b$");
  assert.ok(plain.textContent.includes("\\(a_b\\)"));
});

test("the vendored KaTeX typesets the NT-Xent loss as MathML", async () => {
  const katex = (await import("../app/static/vendor/katex.mjs")).default;
  const tex = "\\ell_{i,j} = -\\log \\frac{\\exp(\\mathrm{sim}(z_i, z_j)/\\tau)}{\\sum_{k=1}^{2N} \\mathbb{1}_{[k \\neq i]} \\exp(\\mathrm{sim}(z_i, z_k)/\\tau)}";
  const html = katex.renderToString(tex, { displayMode: true, output: "mathml", throwOnError: true, strict: "ignore", trust: false });
  assert.match(html, /<math/);
  assert.match(html, /<mfrac/);
  assert.throws(() => katex.renderToString("\\badmacro{", { throwOnError: true }));
  // Untrusted answers cannot inject links or HTML through TeX.
  const hostile = katex.renderToString("\\href{javascript:alert(1)}{x}<img src=x onerror=alert(1)>", { output: "mathml", throwOnError: false, trust: false });
  assert.ok(!/<img/i.test(hostile) && !/javascript:/i.test(hostile.replace(/<annotation[\s\S]*?<\/annotation>/g, "")));
});

const { inviteFromHash, describeQuota, canAsk, loadAccess } = await loadModule("access");

test("the invitation code is read from the URL fragment only", () => {
  assert.equal(inviteFromHash("#invite=abcDEF123_-xyz"), "abcDEF123_-xyz");
  assert.equal(inviteFromHash("#a=1&invite=abcDEF123_-xyz&b=2"), "abcDEF123_-xyz");
  for (const bad of ["", "#invite=short", "#invite=has space inside", "?invite=abcDEF123_-xyz", "#xinvite=abcDEF123_-xyz", null]) {
    assert.equal(inviteFromHash(bad), null, String(bad));
  }
});

test("quota wording and the ask permission follow the invitation state", () => {
  assert.equal(describeQuota({ remaining: 14, quota: 20 }), "Questions restantes : 14 / 20");
  assert.match(describeQuota({ remaining: 0, quota: 20 }), /Quota épuisé/);
  assert.equal(describeQuota(undefined), "");
  assert.equal(canAsk({ mode: "open" }), true);
  assert.equal(canAsk({ mode: "invite", authenticated: false }), false);
  assert.equal(canAsk({ mode: "invite", authenticated: true, invite: { remaining: 0 } }), false);
  assert.equal(canAsk({ mode: "invite", authenticated: true, invite: { remaining: 3 } }), true);
});

test("loadAccess redeems the code once, clears the fragment and reports a refusal", async () => {
  const calls = [];
  const okFetch = async (url, options) => {
    calls.push([url, options?.method || "GET", options?.body]);
    return { ok: true, json: async () => ({ mode: "invite", authenticated: true, invite: { remaining: 20, quota: 20 } }) };
  };
  let cleared = 0;
  const access = await loadAccess(okFetch, "#invite=abcDEF123_-xyz", () => { cleared++; });
  assert.deepEqual(calls.map(call => call.slice(0, 2)), [["/access/redeem", "POST"], ["/access/me", "GET"]]);
  assert.equal(JSON.parse(calls[0][2]).code, "abcDEF123_-xyz");
  assert.equal(cleared, 1);
  assert.equal(access.authenticated, true);

  const refusing = async (url) => url === "/access/redeem"
    ? { ok: false, json: async () => ({ detail: "Invitation invalide, expirée ou révoquée." }) }
    : { ok: true, json: async () => ({ mode: "invite", authenticated: false }) };
  const refused = await loadAccess(refusing, "#invite=abcDEF123_-xyz", () => {});
  assert.equal(refused.authenticated, false);
  assert.match(refused.error, /invalide/);

  calls.length = 0;
  await loadAccess(okFetch, "", () => { cleared++; });
  assert.deepEqual(calls.map(call => call[0]), ["/access/me"]);
  assert.equal(cleared, 1);  // no code in the link: nothing to clear
  const offline = await loadAccess(async () => { throw new Error("down"); }, "", () => {});
  assert.equal(offline.mode, "open");
});
