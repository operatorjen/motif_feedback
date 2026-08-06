import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
ROOT = Path(__file__).parents[1]


def run_module_test(tmp_path: Path, source_name: str, script: str) -> dict:
    module = tmp_path / f"{Path(source_name).stem}.mjs"
    shutil.copyfile(ROOT / "motif_feedback" / "static" / "js" / source_name, module)
    completed = subprocess.run(
        [NODE, "--input-type=module", "-e", script, module.as_uri()],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.skipif(NODE is None, reason="Node.js is not available.")
def test_message_reconciliation_reuses_only_unchanged_nodes(tmp_path):
    result = run_module_test(
        tmp_path,
        "message_reconciler.js",
        r"""
const { decorateMessageNode, reconcileMessageNodes } = await import(process.argv[1]);
const render = (message) => ({ dataset: {}, rendered: message.content });
const first = { id: "one", role: "user", content: "first" };
const second = { id: "two", role: "agent", content: "second" };
const firstNode = decorateMessageNode(render(first), first);
const secondNode = decorateMessageNode(render(second), second);
const container = {
  children: [firstNode, secondNode],
  replaceChildren(...nodes) { this.children = nodes; },
};
const outcome = reconcileMessageNodes(
  container,
  [second, { ...first, content: "changed" }],
  render,
);
console.log(JSON.stringify({
  outcome,
  reusedSecond: container.children[0] === secondNode,
  replacedFirst: container.children[1] !== firstNode,
  order: container.children.map((node) => node.dataset.messageId),
}));
""",
    )
    assert result == {
        "outcome": {"created": 1, "reused": 1},
        "reusedSecond": True,
        "replacedFirst": True,
        "order": ["two", "one"],
    }


@pytest.mark.skipif(NODE is None, reason="Node.js is not available.")
def test_turn_refresh_targets_only_panels_changed_by_events(tmp_path):
    result = run_module_test(
        tmp_path,
        "turn_refresh.js",
        r"""
const {
  createTurnRefreshState,
  observeTurnRefreshEvent,
  observeTurnRefreshResult,
  turnRefreshTargets,
} = await import(process.argv[1]);
const plain = createTurnRefreshState();
const changed = createTurnRefreshState();
observeTurnRefreshEvent(changed, {
  type: "tool",
  tool: "write_project_file",
  result: { ok: true },
});
observeTurnRefreshEvent(changed, {
  type: "tool",
  tool: "record_motif_observations",
  result: { ok: false },
});
observeTurnRefreshResult(changed, {
  messages: [{ metadata: { tool_events: [{
    tool: "propose_persona_update",
    result: { ok: true },
  }] } }],
  web_sources: [{ id: "source" }],
});
console.log(JSON.stringify({
  plain: turnRefreshTargets(plain),
  changed: turnRefreshTargets(changed),
}));
""",
    )
    assert result["plain"] == ["messages", "memory", "recovery"]
    assert result["changed"] == [
        "messages",
        "memory",
        "recovery",
        "files",
        "sources",
        "proposals",
    ]
