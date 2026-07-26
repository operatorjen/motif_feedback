const INLINE_TOKEN = /(\\[\\`*_[\]{}()#+.!~-]|`[^`\n]+`|\\\([^\n]+?\\\)|\$(?!\$)[^$\n]+?\$|!\[[^\]\n]*\]\([^\s)]+(?:\s+["'][^"']*["'])?\)|\[[^\]\n]+\]\([^\s)]+(?:\s+["'][^"']*["'])?\)|\*\*[^*\n]+\*\*|(?<![A-Za-z0-9])__(?=\S)[^_\n]+?(?<=\S)__(?![A-Za-z0-9])|~~[^~\n]+~~|\*[^*\n]+\*|(?<![A-Za-z0-9])_(?=\S)[^_\n]+?(?<=\S)_(?![A-Za-z0-9])|\\?(?:Longleftrightarrow|Longrightarrow|Longleftarrow|longleftrightarrow|longrightarrow|longleftarrow|longmapsto)\b|<https?:\/\/[^>\s]+>)/i;
const MIN_DISPLAY_MATH_OPERATORS = 2;
const SINGLE_OPERATOR_MATH_MAX_CHARS = 300;

const FLOW_SYMBOLS = {
  longrightarrow: "⟶",
  longleftarrow: "⟵",
  longleftrightarrow: "⟷",
  longmapsto: "⟼",
  Longrightarrow: "⟹",
  Longleftarrow: "⟸",
  Longleftrightarrow: "⟺",
};
const FLOW_OPERATOR_PATTERN = /\\?(?:Longleftrightarrow|Longrightarrow|Longleftarrow|longleftrightarrow|longrightarrow|longleftarrow|longmapsto)\b/g;

const TEX_SYMBOLS = {
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", varepsilon: "ε",
  zeta: "ζ", eta: "η", theta: "θ", vartheta: "ϑ", iota: "ι", kappa: "κ",
  lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", omicron: "ο", pi: "π", varpi: "ϖ",
  rho: "ρ", varrho: "ϱ", sigma: "σ", varsigma: "ς", tau: "τ", upsilon: "υ",
  phi: "φ", varphi: "ϕ", chi: "χ", psi: "ψ", omega: "ω",
  Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Xi: "Ξ", Pi: "Π", Sigma: "Σ",
  Upsilon: "Υ", Phi: "Φ", Psi: "Ψ", Omega: "Ω",
  infty: "∞", partial: "∂", nabla: "∇", forall: "∀", exists: "∃", neg: "¬",
  pm: "±", mp: "∓", times: "×", div: "÷", cdot: "·", circ: "∘", bullet: "•",
  sum: "∑", prod: "∏", int: "∫", oint: "∮", sqrt: "√", propto: "∝",
  approx: "≈", sim: "∼", simeq: "≃", cong: "≅", equiv: "≡", ne: "≠", neq: "≠",
  le: "≤", leq: "≤", ge: "≥", geq: "≥", ll: "≪", gg: "≫",
  in: "∈", notin: "∉", ni: "∋", subset: "⊂", supset: "⊃", subseteq: "⊆",
  supseteq: "⊇", cup: "∪", cap: "∩", emptyset: "∅", setminus: "∖",
  to: "→", rightarrow: "→", leftarrow: "←", leftrightarrow: "↔", mapsto: "↦",
  Rightarrow: "⇒", Leftarrow: "⇐", Leftrightarrow: "⇔", uparrow: "↑", downarrow: "↓",
  longrightarrow: "⟶", longleftarrow: "⟵", longleftrightarrow: "⟷", longmapsto: "⟼",
  Longrightarrow: "⟹", Longleftarrow: "⟸", Longleftrightarrow: "⟺",
  land: "∧", wedge: "∧", lor: "∨", vee: "∨", top: "⊤", bot: "⊥",
  ldots: "…", cdots: "⋯", vdots: "⋮", ddots: "⋱", angle: "∠", degree: "°",
};

function readTexGroup(source, start) {
  if (source[start] !== "{") return { value: source[start] || "", end: start + 1 };
  let depth = 1;
  let index = start + 1;
  while (index < source.length && depth) {
    if (source[index] === "{") depth += 1;
    if (source[index] === "}") depth -= 1;
    index += 1;
  }
  return { value: source.slice(start + 1, depth ? source.length : index - 1), end: index };
}

function appendTex(parent, source) {
  let index = 0;
  while (index < source.length) {
    const character = source[index];
    if (character === "\\") {
      const command = source.slice(index + 1).match(/^[A-Za-z]+|^./)?.[0] || "";
      index += command.length + 1;
      if (["left", "right", "limits", "nolimits"].includes(command)) continue;
      if ([",", ";", ":", "!", "quad", "qquad", " "].includes(command)) {
        parent.append(document.createTextNode(command === "!" ? "" : " "));
        continue;
      }
      if (command === "frac") {
        const numerator = readTexGroup(source, index);
        const denominator = readTexGroup(source, numerator.end);
        const fraction = document.createElement("span");
        fraction.className = "math-fraction";
        const top = document.createElement("span");
        const bottom = document.createElement("span");
        appendTex(top, numerator.value);
        appendTex(bottom, denominator.value);
        fraction.append(top, bottom);
        parent.append(fraction);
        index = denominator.end;
        continue;
      }
      if (command === "sqrt") {
        const radicand = readTexGroup(source, index);
        const root = document.createElement("span");
        root.className = "math-root";
        root.append(document.createTextNode("√"));
        const body = document.createElement("span");
        appendTex(body, radicand.value);
        root.append(body);
        parent.append(root);
        index = radicand.end;
        continue;
      }
      if (["text", "mathrm", "mathbf", "mathit", "operatorname"].includes(command)) {
        const group = readTexGroup(source, index);
        const styled = document.createElement("span");
        styled.className = `math-${command}`;
        if (command === "text" || command === "operatorname") styled.textContent = group.value;
        else appendTex(styled, group.value);
        parent.append(styled);
        index = group.end;
        continue;
      }
      parent.append(document.createTextNode(TEX_SYMBOLS[command] || command));
      continue;
    }
    if (character === "^" || character === "_") {
      const group = readTexGroup(source, index + 1);
      const script = document.createElement(character === "^" ? "sup" : "sub");
      appendTex(script, group.value);
      parent.append(script);
      index = group.end;
      continue;
    }
    if (character === "{") {
      const group = readTexGroup(source, index);
      appendTex(parent, group.value);
      index = group.end;
      continue;
    }
    if (character !== "}") parent.append(document.createTextNode(character));
    index += 1;
  }
}

function mathNode(source, display = false) {
  const node = document.createElement(display ? "div" : "span");
  node.className = display ? "math-display" : "math-inline";
  node.setAttribute("role", "math");
  node.setAttribute("aria-label", source.trim());
  appendTex(node, source.trim());
  return node;
}

function normalizedFlowExpression(source) {
  return source.replace(FLOW_OPERATOR_PATTERN, (token) => (
    FLOW_SYMBOLS[token.replace(/^\\/, "")] || token
  ));
}

function isBareFlowExpression(source) {
  if (/`|\$(?!\$)|\\\(|\\\[/.test(source)) return false;
  const operators = source.match(FLOW_OPERATOR_PATTERN) || [];
  if (operators.length >= MIN_DISPLAY_MATH_OPERATORS) return true;
  return (
    operators.length === 1
    && source.trim().length <= SINGLE_OPERATOR_MATH_MAX_CHARS
    && !/[.!?](?:\s|$)/.test(source)
  );
}

function flowNode(source) {
  const node = document.createElement("div");
  node.className = "math-display math-flow";
  node.setAttribute("role", "math");
  node.setAttribute("aria-label", source.trim());
  node.textContent = normalizedFlowExpression(source.trim());
  return node;
}

function appendTextWithBreaks(parent, text) {
  const parts = text.split("\n");
  parts.forEach((part, index) => {
    if (index) parent.append(document.createElement("br"));
    parent.append(document.createTextNode(part));
  });
}

function safeHref(rawHref) {
  const href = rawHref.trim();
  if (!href || /[\u0000-\u001f\u007f]/.test(href)) return null;
  if (href.startsWith("#") || href.startsWith("/") || href.startsWith("./") || href.startsWith("../")) {
    return href;
  }
  if (!/^[a-z][a-z\d+.-]*:/i.test(href)) return href;
  return /^(https?:|mailto:)/i.test(href) ? href : null;
}

function parseLinkToken(token) {
  const image = token.startsWith("![");
  const match = token.match(/^!?\[([^\]]*)\]\(([^\s)]+)(?:\s+["']([^"']*)["'])?\)$/);
  if (!match) return null;
  return { image, label: match[1], href: match[2], title: match[3] || "" };
}

function appendInline(parent, source, options = {}) {
  let remaining = String(source ?? "");
  while (remaining) {
    const match = remaining.match(INLINE_TOKEN);
    if (!match || match.index === undefined) {
      appendTextWithBreaks(parent, remaining);
      break;
    }
    appendTextWithBreaks(parent, remaining.slice(0, match.index));
    const token = match[0];
    let node = null;

    const flowSymbol = FLOW_SYMBOLS[token.replace(/^\\/, "")];
    if (flowSymbol) {
      node = document.createTextNode(flowSymbol);
    } else if (token.startsWith("\\") && token.length === 2) {
      node = document.createTextNode(token.slice(1));
    } else if (token.startsWith("`")) {
      node = document.createElement("code");
      node.textContent = token.slice(1, -1);
    } else if (token.startsWith("$")) {
      node = mathNode(token.slice(1, -1));
    } else if (token.startsWith("\\(")) {
      node = mathNode(token.slice(2, -2));
    } else if (token.startsWith("**") || token.startsWith("__")) {
      node = document.createElement("strong");
      appendInline(node, token.slice(2, -2), options);
    } else if (token.startsWith("~~")) {
      node = document.createElement("del");
      appendInline(node, token.slice(2, -2), options);
    } else if (token.startsWith("*") || token.startsWith("_")) {
      node = document.createElement("em");
      appendInline(node, token.slice(1, -1), options);
    } else if (token.startsWith("<")) {
      const href = safeHref(token.slice(1, -1));
      if (href) {
        node = document.createElement("a");
        node.href = href;
        node.textContent = href;
        node.target = "_blank";
        node.rel = "noopener noreferrer";
      }
    } else {
      const link = parseLinkToken(token);
      const href = link && safeHref(link.href);
      if (link && href && !link.image) {
        node = document.createElement("a");
        node.href = href;
        node.title = link.title;
        appendInline(node, link.label, options);
        if (/^https?:/i.test(href)) {
          node.target = "_blank";
          node.rel = "noopener noreferrer";
        } else if (!href.startsWith("#") && options.onProjectFile) {
          node.addEventListener("click", (event) => {
            event.preventDefault();
            options.onProjectFile(href.replace(/^\.\//, ""));
          });
        }
      } else if (link && href && link.image) {
        // Do not fetch arbitrary model-supplied images. Keep them as safe, clickable links.
        node = document.createElement("a");
        node.href = href;
        node.textContent = link.label || "image";
        node.title = link.title;
        if (/^https?:/i.test(href)) {
          node.target = "_blank";
          node.rel = "noopener noreferrer";
        }
      }
    }

    if (node) parent.append(node);
    else parent.append(document.createTextNode(token));
    remaining = remaining.slice(match.index + token.length);
  }
}

function isTableDivider(line) {
  return /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
}

function tableCells(line) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
}

function appendTable(parent, lines, start, options) {
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headingRow = document.createElement("tr");
  for (const cell of tableCells(lines[start])) {
    const th = document.createElement("th");
    appendInline(th, cell, options);
    headingRow.append(th);
  }
  thead.append(headingRow);
  table.append(thead);

  const tbody = document.createElement("tbody");
  let index = start + 2;
  while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
    const row = document.createElement("tr");
    for (const cell of tableCells(lines[index])) {
      const td = document.createElement("td");
      appendInline(td, cell, options);
      row.append(td);
    }
    tbody.append(row);
    index += 1;
  }
  table.append(tbody);
  parent.append(table);
  return index;
}

function startsBlock(lines, index) {
  const line = lines[index] || "";
  return /^\s*```/.test(line)
    || /^\s*(?:\$\$|\\\[)/.test(line)
    || isBareFlowExpression(line)
    || /^\s{0,3}#{1,6}\s+/.test(line)
    || /^\s{0,3}>\s?/.test(line)
    || /^\s{0,3}(?:[-+*]|\d+[.)])\s+/.test(line)
    || /^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)
    || (line.includes("|") && isTableDivider(lines[index + 1] || ""));
}

function appendBlocks(parent, source, options) {
  const lines = String(source ?? "").replace(/\r\n?/g, "\n").split("\n");
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }

    const fence = line.match(/^\s*```\s*([\w+-]*)\s*$/);
    if (fence) {
      const codeLines = [];
      index += 1;
      while (index < lines.length && !/^\s*```\s*$/.test(lines[index])) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      if (fence[1]) code.className = `language-${fence[1].toLowerCase()}`;
      code.textContent = codeLines.join("\n");
      pre.append(code);
      parent.append(pre);
      continue;
    }

    const singleLineMath = line.match(/^\s*\$\$\s*(.*?)\s*\$\$\s*$/)
      || line.match(/^\s*\\\[\s*(.*?)\s*\\\]\s*$/);
    if (singleLineMath) {
      parent.append(mathNode(singleLineMath[1], true));
      index += 1;
      continue;
    }

    if (/^\s*(?:\$\$|\\\[)\s*$/.test(line)) {
      const dollarFence = /^\s*\$\$/.test(line);
      const closing = dollarFence ? /^\s*\$\$\s*$/ : /^\s*\\\]\s*$/;
      const mathLines = [];
      index += 1;
      while (index < lines.length && !closing.test(lines[index])) {
        mathLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      parent.append(mathNode(mathLines.join(" "), true));
      continue;
    }

    if (isBareFlowExpression(line)) {
      parent.append(flowNode(line));
      index += 1;
      continue;
    }

    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      const node = document.createElement(`h${heading[1].length}`);
      appendInline(node, heading[2], options);
      parent.append(node);
      index += 1;
      continue;
    }

    if (/^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      parent.append(document.createElement("hr"));
      index += 1;
      continue;
    }

    if (/^\s{0,3}>\s?/.test(line)) {
      const quoteLines = [];
      while (index < lines.length && /^\s{0,3}>\s?/.test(lines[index])) {
        quoteLines.push(lines[index].replace(/^\s{0,3}>\s?/, ""));
        index += 1;
      }
      const quote = document.createElement("blockquote");
      appendBlocks(quote, quoteLines.join("\n"), options);
      parent.append(quote);
      continue;
    }

    const listItem = line.match(/^\s{0,3}([-+*]|\d+[.)])\s+(.+)$/);
    if (listItem) {
      const ordered = /^\d/.test(listItem[1]);
      const list = document.createElement(ordered ? "ol" : "ul");
      while (index < lines.length) {
        const itemMatch = lines[index].match(/^\s{0,3}([-+*]|\d+[.)])\s+(.+)$/);
        if (!itemMatch || /^\d/.test(itemMatch[1]) !== ordered) break;
        const item = document.createElement("li");
        const task = itemMatch[2].match(/^\[([ xX])\]\s+(.+)$/);
        if (task) {
          const checkbox = document.createElement("input");
          checkbox.type = "checkbox";
          checkbox.checked = task[1].toLowerCase() === "x";
          checkbox.disabled = true;
          item.className = "task-list-item";
          item.append(checkbox);
          appendInline(item, task[2], options);
        } else {
          appendInline(item, itemMatch[2], options);
        }
        list.append(item);
        index += 1;
      }
      parent.append(list);
      continue;
    }

    if (line.includes("|") && isTableDivider(lines[index + 1] || "")) {
      index = appendTable(parent, lines, index, options);
      continue;
    }

    const paragraph = [];
    while (index < lines.length && lines[index].trim() && (!paragraph.length || !startsBlock(lines, index))) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    const node = document.createElement("p");
    appendInline(node, paragraph.join("\n").replace(/(?<!  )\n/g, " ").replace(/ {2}\n/g, "\n"), options);
    parent.append(node);
  }
}

export function renderMarkdown(source, options = {}) {
  const fragment = document.createDocumentFragment();
  appendBlocks(fragment, source, options);
  return fragment;
}
