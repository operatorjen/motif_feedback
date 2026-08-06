const LANGUAGE_BY_SUFFIX = {
  ".py": "Python",
  ".js": "JavaScript",
  ".jsx": "JSX",
  ".ts": "TypeScript",
  ".tsx": "TSX",
  ".json": "JSON",
  ".yaml": "YAML",
  ".yml": "YAML",
  ".html": "HTML",
  ".css": "CSS",
  ".csv": "CSV",
  ".sh": "Shell",
  ".toml": "TOML",
  ".ini": "INI",
  ".xml": "XML",
  ".sql": "SQL",
  ".c": "C",
  ".h": "C Header",
  ".cpp": "C++",
  ".hpp": "C++ Header",
  ".java": "Java",
  ".go": "Go",
  ".rs": "Rust",
};

const KEYWORDS = new Set([
  "and", "as", "async", "await", "break", "case", "catch", "class", "const",
  "continue", "def", "default", "del", "do", "elif", "else", "except", "export",
  "extends", "false", "finally", "for", "from", "function", "if", "import", "in",
  "is", "let", "new", "none", "not", "null", "of", "or", "pass", "raise", "return",
  "static", "super", "switch", "this", "throw", "true", "try", "typeof", "undefined",
  "var", "while", "with", "yield",
]);

function suffixFor(path) {
  const dot = path.lastIndexOf(".");
  return dot >= 0 ? path.slice(dot).toLowerCase() : "";
}

function token(className, text) {
  const span = document.createElement("span");
  span.className = className;
  span.textContent = text;
  return span;
}

function highlightLine(line, suffix, state) {
  const fragment = document.createDocumentFragment();
  const hashComments = [".py", ".yaml", ".yml", ".sh", ".toml", ".ini"].includes(suffix);
  let index = 0;

  while (index < line.length) {
    if (state.blockComment) {
      const end = line.indexOf("*/", index);
      if (end < 0) {
        fragment.append(token("syntax-comment", line.slice(index)));
        return fragment;
      }
      fragment.append(token("syntax-comment", line.slice(index, end + 2)));
      state.blockComment = false;
      index = end + 2;
      continue;
    }

    if (line.startsWith("/*", index)) {
      const end = line.indexOf("*/", index + 2);
      if (end < 0) {
        state.blockComment = true;
        fragment.append(token("syntax-comment", line.slice(index)));
        return fragment;
      }
      fragment.append(token("syntax-comment", line.slice(index, end + 2)));
      index = end + 2;
      continue;
    }
    if (line.startsWith("//", index) || (hashComments && line[index] === "#")) {
      fragment.append(token("syntax-comment", line.slice(index)));
      return fragment;
    }

    const character = line[index];
    if (character === "\"" || character === "'" || character === "`") {
      const quote = character;
      let end = index + 1;
      while (end < line.length) {
        if (line[end] === "\\") {
          end += 2;
          continue;
        }
        end += 1;
        if (line[end - 1] === quote) break;
      }
      fragment.append(token("syntax-string", line.slice(index, end)));
      index = end;
      continue;
    }

    const number = line.slice(index).match(/^(?:0x[\da-f]+|\d+(?:\.\d+)?)/i);
    if (number) {
      fragment.append(token("syntax-number", number[0]));
      index += number[0].length;
      continue;
    }

    const word = line.slice(index).match(/^[A-Za-z_$][\w$-]*/);
    if (word) {
      const normalized = word[0].toLowerCase();
      fragment.append(token(KEYWORDS.has(normalized) ? "syntax-keyword" : "syntax-name", word[0]));
      index += word[0].length;
      continue;
    }

    if (/[{}[\]().,:;<>+=!*&|?-]/.test(character)) {
      fragment.append(token("syntax-punctuation", character));
    } else {
      fragment.append(document.createTextNode(character));
    }
    index += 1;
  }
  return fragment;
}

export function renderCodeViewer(content, path) {
  const suffix = suffixFor(path);
  const viewer = document.createElement("div");
  viewer.className = "code-viewer";
  const toolbar = document.createElement("div");
  toolbar.className = "code-viewer-toolbar";
  const filename = document.createElement("strong");
  filename.textContent = path;
  const language = document.createElement("span");
  language.textContent = LANGUAGE_BY_SUFFIX[suffix] || "CODE";
  toolbar.append(filename, language);

  const code = document.createElement("div");
  code.className = "code-viewer-lines";
  const state = { blockComment: false };
  const lines = content.split("\n");
  lines.forEach((line, lineIndex) => {
    const row = document.createElement("div");
    row.className = "code-line";
    const number = document.createElement("span");
    number.className = "code-line-number";
    number.textContent = String(lineIndex + 1);
    const source = document.createElement("code");
    source.append(highlightLine(line, suffix, state));
    row.append(number, source);
    code.append(row);
  });
  viewer.append(toolbar, code);
  return viewer;
}
