/** Small, safe Markdown subset for model answers. No raw HTML is interpreted.
 * Supports paragraphs, headings, lists, quotes, code, tables and HTTP(S) links.
 * Math ($..$, $$..$$, \(..\), \[..\]) is parsed first so that its backslashes,
 * asterisks and underscores are never read as Markdown. It is typeset by an optional
 * renderer (KaTeX, registered by app.js); without one the TeX source stays visible.
 * It deliberately does not claim full CommonMark support.
 */
let mathRenderer = null;
/** renderer(element, tex, display) typesets into element and may throw on bad TeX. */
export function setMathRenderer(renderer) { mathRenderer = renderer; }

const MATH_DELIMITERS = [
  { open: "$$", close: "$$", display: true },
  { open: "\\[", close: "\\]", display: true },
  { open: "\\(", close: "\\)", display: false },
  { open: "$", close: "$", display: false },
];

/** Math starting exactly at text[i], or null. A lone $ must look like math, not money. */
function mathAt(text, i) {
  for (const { open, close, display } of MATH_DELIMITERS) {
    if (!text.startsWith(open, i)) continue;
    const from = i + open.length;
    let end = text.indexOf(close, from);
    if (open === "$") {
      while (end > 0 && text[end - 1] === "\\") end = text.indexOf(close, end + 1);
      if (end < 0 || /\s/.test(text[from] || " ") || /\s/.test(text[end - 1]) || /\d/.test(text[end + 1] || "")) continue;
    }
    if (end < 0 || end === from || end - from > 2000) continue;
    // `\[S1\]` is an escaped citation, not a formula.
    if (/^S\d+(\s*,\s*S\d+)*$/.test(text.slice(from, end).trim())) continue;
    return { tex: text.slice(from, end).trim(), display, length: end + close.length - i };
  }
  return null;
}
export function inlineTokens(text, depth = 0) {
  if (depth > 12) return [{ type: "text", text }];
  const tokens = [];
  let plain = "";
  const flush = () => { if (plain) tokens.push({ type: "text", text: plain }); plain = ""; };
  for (let i = 0; i < text.length;) {
    if (text[i] === "$" || (text[i] === "\\" && /[(\[]/.test(text[i + 1] || ""))) {
      const math = mathAt(text, i);
      if (math) { flush(); tokens.push({ type: "math", text: math.tex, display: math.display }); i += math.length; continue; }
    }
    if (text[i] === "\\" && /[\\*_`\[\]]/.test(text[i + 1] || "")) {
      plain += text[i + 1]; i += 2; continue;
    }
    const marker = text.startsWith("**", i) ? "**" : text.startsWith("__", i) ? "__" : ["`", "*"].includes(text[i]) ? text[i] : null;
    if (marker) {
      const end = text.indexOf(marker, i + marker.length);
      if (end > i + marker.length) {
        flush();
        const content = text.slice(i + marker.length, end);
        tokens.push(marker === "`" ? { type: "code", text: content } : {
          type: marker.length === 2 ? "strong" : "em", children: inlineTokens(content, depth + 1),
        });
        i = end + marker.length; continue;
      }
    }
    if (text[i] === "[") {
      const link = text.slice(i).match(/^\[([^\]\n]+)\]\(([^\s)]+)\)/);
      if (link) {
        flush();
        // An unsafe URL stays literal text. Never assign javascript:/data: URLs.
        if (/^https?:\/\//i.test(link[2])) {
          tokens.push({ type: "link", href: link[2], children: inlineTokens(link[1], depth + 1) });
        } else tokens.push({ type: "text", text: link[0] });
        i += link[0].length; continue;
      }
    }
    plain += text[i++];
  }
  flush();
  return tokens;
}

function listMatch(line) {
  const match = line.match(/^(\s*)([-+*]|\d+[.)])\s+(.+)$/);
  return match && { indent: match[1].replace(/\t/g, "    ").length, ordered: /^\d/.test(match[2]), start: parseInt(match[2], 10) || 1, text: match[3] };
}

function readList(lines, start, depth) {
  const first = listMatch(lines[start]);
  const block = { type: first.ordered ? "ol" : "ul", start: first.start, items: [] };
  let i = start;
  while (i < lines.length) {
    const item = listMatch(lines[i]);
    if (!item || item.indent !== first.indent || item.ordered !== first.ordered) break;
    const entry = { text: item.text, children: [] };
    block.items.push(entry); i++;
    while (i < lines.length) {
      if (!lines[i].trim()) {
        if (listMatch(lines[i + 1] || "")) { i++; continue; }
        break;
      }
      const nested = listMatch(lines[i]);
      if (nested && nested.indent > first.indent && depth < 12) {
        const parsed = readList(lines, i, depth + 1);
        entry.children.push(parsed.block); i = parsed.end; continue;
      }
      if (!nested && lines[i].match(/^\s*/)[0].length > first.indent) {
        entry.text += "\n" + lines[i].trim(); i++; continue;
      }
      break;
    }
  }
  return { block, end: i };
}

const cells = (line) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(cell => cell.trim());
const tableRule = (line) => line.includes("|") && cells(line).every(cell => /^:?-{3,}:?$/.test(cell));
const startsBlock = (line) => /^(?:\s*```|#{1,6}\s|>\s?)/.test(line) || !!listMatch(line);

export function markdownBlocks(text, depth = 0) {
  if (depth > 12) return [{ type: "p", text }];
  const lines = String(text).replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  for (let i = 0; i < lines.length;) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (/^\s*```/.test(line)) {
      const code = []; i++;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) code.push(lines[i++]);
      if (i < lines.length) i++;
      blocks.push({ type: "pre", text: code.join("\n") }); continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) { blocks.push({ type: `h${heading[1].length}`, text: heading[2] }); i++; continue; }
    if (listMatch(line)) {
      const parsed = readList(lines, i, depth);
      blocks.push(parsed.block); i = parsed.end; continue;
    }
    if (/^>\s?/.test(line)) {
      const quoted = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) quoted.push(lines[i++].replace(/^>\s?/, ""));
      blocks.push({ type: "blockquote", children: markdownBlocks(quoted.join("\n"), depth + 1) }); continue;
    }
    if (line.includes("|") && tableRule(lines[i + 1] || "")) {
      const header = cells(line), rows = []; i += 2;
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) rows.push(cells(lines[i++]));
      blocks.push({ type: "table", header, rows }); continue;
    }
    const paragraph = [line]; i++;
    while (i < lines.length && lines[i].trim() && !startsBlock(lines[i]) && !tableRule(lines[i + 1] || "")) paragraph.push(lines[i++]);
    blocks.push({ type: "p", text: paragraph.join("\n") });
  }
  return blocks;
}

function mathNode(token) {
  const node = document.createElement("span");
  node.className = token.display ? "math math-display" : "math";
  try {
    if (!mathRenderer) throw new Error("no renderer");
    mathRenderer(node, token.text, token.display);
  } catch {
    // Unsupported TeX or no renderer: show the source rather than hide or break it.
    node.replaceChildren();
    node.className += " math-source";
    node.textContent = token.display ? `\\[${token.text}\\]` : `\\(${token.text}\\)`;
  }
  return node;
}

function appendInline(parent, text) {
  const append = (target, tokens) => {
    for (const token of tokens) {
      if (token.type === "text") { target.append(document.createTextNode(token.text)); continue; }
      if (token.type === "math") { target.append(mathNode(token)); continue; }
      const node = document.createElement(token.type === "link" ? "a" : token.type);
      if (token.type === "link") { node.href = token.href; node.target = "_blank"; node.rel = "noopener noreferrer"; }
      if (token.children) append(node, token.children); else node.textContent = token.text;
      target.append(node);
    }
  };
  append(parent, inlineTokens(text));
}

function blockNode(block) {
  const node = document.createElement(block.type === "table" ? "div" : block.type);
  if (block.type === "pre") {
    const code = document.createElement("code"); code.textContent = block.text; node.append(code);
  } else if (block.items) {
    if (block.type === "ol") node.start = block.start;
    for (const item of block.items) {
      const li = document.createElement("li"); appendInline(li, item.text);
      li.append(...item.children.map(blockNode)); node.append(li);
    }
  } else if (block.type === "table") {
    node.className = "table-scroll";
    const table = document.createElement("table"), head = document.createElement("thead"), body = document.createElement("tbody");
    const row = (values, tag) => {
      const tr = document.createElement("tr");
      for (const value of values) { const td = document.createElement(tag); appendInline(td, value); tr.append(td); }
      return tr;
    };
    head.append(row(block.header, "th")); body.append(...block.rows.map(values => row(values, "td")));
    table.append(head, body); node.append(table);
  } else if (block.children) node.append(...block.children.map(blockNode));
  else appendInline(node, block.text);
  return node;
}

export function renderMarkdown(target, text) {
  target.replaceChildren(...markdownBlocks(text).map(blockNode));
}
