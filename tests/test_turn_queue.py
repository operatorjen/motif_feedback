import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
TURN_QUEUE_MODULE = (
    Path(__file__).parents[1] / "app" / "static" / "js" / "turn_queue.js"
)


@pytest.mark.skipif(NODE is None, reason="Node.js is not available for the turn queue test.")
def test_turn_queue_is_bounded_removable_and_fifo(tmp_path):
    queue_module = tmp_path / "turn_queue.mjs"
    shutil.copyfile(TURN_QUEUE_MODULE, queue_module)
    script = r"""
const { TurnQueue } = await import(process.argv[1]);
const queue = new TurnQueue(2);
const participants = ["agent_a"];
const first = queue.enqueue({
  message: "first",
  participants,
  projectId: "one",
});
participants.push("agent_b");
const second = queue.enqueue({
  message: "second",
  participants: ["agent_c"],
  projectId: "two",
});
const snapshot = queue.snapshot();
snapshot[0].participants.push("agent_c");
const removed = queue.remove(second.id);
const shifted = queue.shift();
let overflow = "";
const bounded = new TurnQueue(1);
bounded.enqueue({ message: "kept", participants: [], projectId: "one" });
try {
  bounded.enqueue({ message: "rejected", participants: [], projectId: "one" });
} catch (error) {
  overflow = error.message;
}
console.log(JSON.stringify({
  first,
  removed,
  shifted,
  remaining: queue.length,
  overflow,
}));
"""
    completed = subprocess.run(
        [
            NODE,
            "--input-type=module",
            "-e",
            script,
            queue_module.as_uri(),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["removed"] is True
    assert result["shifted"]["message"] == "first"
    assert result["shifted"]["participants"] == ["agent_a"]
    assert result["remaining"] == 0
    assert result["overflow"] == "The turn queue is limited to 1 prompts."


def test_chat_composer_exposes_the_turn_queue_without_disabling_typing():
    root = Path(__file__).parents[1]
    html = (root / "app" / "static" / "index.html").read_text(encoding="utf-8")
    source = (root / "app" / "static" / "js" / "app.js").read_text(encoding="utf-8")

    assert 'id="prompt-queue"' in html
    assert "messageInput.disabled = !setupComplete" in source
    assert 'busy ? "QUEUE NEXT ↵" : "SEND ↵"' in source
    assert "async function drainPromptQueue()" in source
    assert "turn_id: turn.turnId" in source
    assert "window.crypto?.randomUUID?.()" in source
