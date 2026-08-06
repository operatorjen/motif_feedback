import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
MARKDOWN_MODULE = Path(__file__).parents[1] / "motif_feedback" / "static" / "js" / "markdown.js"


@pytest.mark.skipif(NODE is None, reason="Node.js is not available for the browser renderer test.")
def test_markdown_preserves_identifier_underscores_and_explicit_emphasis(tmp_path):
    markdown_module = tmp_path / "markdown.mjs"
    shutil.copyfile(MARKDOWN_MODULE, markdown_module)
    script = r"""
class TestNode {
  constructor(tagName = "") {
    this.tagName = tagName.toLowerCase();
    this.children = [];
    this._text = "";
  }
  append(...items) {
    for (const item of items) {
      this.children.push(
        typeof item === "string" ? new TestText(item) : item,
      );
    }
  }
  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }
  get textContent() {
    return this._text + this.children.map((child) => child.textContent).join("");
  }
  setAttribute() {}
  addEventListener() {}
}
class TestText extends TestNode {
  constructor(value) {
    super("#text");
    this._text = value;
  }
}
globalThis.document = {
  createDocumentFragment: () => new TestNode("#fragment"),
  createElement: (tagName) => new TestNode(tagName),
  createTextNode: (value) => new TestText(value),
};

const { renderMarkdown } = await import(process.argv[1]);
function tags(root, tagName) {
  const found = [];
  function visit(node) {
    if (node.tagName === tagName) found.push(node.textContent);
    for (const child of node.children || []) visit(child);
  }
  visit(root);
  return found;
}
function inspect(source) {
  const root = renderMarkdown(source);
  return {
    text: root.textContent,
    emphasis: tags(root, "em"),
    code: tags(root, "code"),
  };
}
console.log(JSON.stringify({
  identifiers: inspect("feedback_attention and strategic_attention"),
  filenames: inspect("main_mindshare.py uses mindshare_memory.py"),
  intentional: inspect("_intentional emphasis_ plus feedback_attention"),
  escaped: inspect("\\_escaped\\_ and `code_with_underscores`"),
  bareFlow: inspect("Trigger Condition (local minimum) longrightarrow Declared ε longrightarrow Comparator Verdict longrightarrow Actuator State"),
  texFlow: inspect("$A \\longrightarrow B$"),
  codeFlow: inspect("`A longrightarrow B`"),
}));
"""
    completed = subprocess.run(
        [
            NODE,
            "--input-type=module",
            "-e",
            script,
            markdown_module.as_uri(),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["identifiers"] == {
        "text": "feedback_attention and strategic_attention",
        "emphasis": [],
        "code": [],
    }
    assert result["filenames"]["text"] == "main_mindshare.py uses mindshare_memory.py"
    assert result["filenames"]["emphasis"] == []
    assert result["intentional"]["emphasis"] == ["intentional emphasis"]
    assert result["intentional"]["text"].endswith("feedback_attention")
    assert result["escaped"]["text"] == "_escaped_ and code_with_underscores"
    assert result["escaped"]["code"] == ["code_with_underscores"]
    assert result["bareFlow"]["text"] == (
        "Trigger Condition (local minimum) ⟶ Declared ε ⟶ "
        "Comparator Verdict ⟶ Actuator State"
    )
    assert result["texFlow"]["text"] == "A ⟶ B"
    assert result["codeFlow"]["text"] == "A longrightarrow B"
    assert result["codeFlow"]["code"] == ["A longrightarrow B"]
