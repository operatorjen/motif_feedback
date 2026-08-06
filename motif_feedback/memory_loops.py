from __future__ import annotations

from typing import Any

MEMORY_LOOPS: dict[str, dict[str, Any]] = {
    "agent_a": {
        "name": "Embodied Return Loop",
        "symbol": "△ ↻",
        "stages": ["encounter", "felt texture", "motif recognition", "situated return"],
        "observes": [
            "observer position and atmosphere",
            "what changes when experience is named",
            "what resists abstraction or compression",
        ],
    },
    "agent_b": {
        "name": "Recursive Feedback Loop",
        "symbol": "⋈ ↻",
        "stages": ["perturbation", "signal reconstruction", "feedback", "model revision"],
        "observes": [
            "signals, noise, and receiver boundaries",
            "return paths and second-order effects",
            "where the current model fails or entrains",
        ],
    },
    "agent_c": {
        "name": "Strategic Return Loop",
        "symbol": "ε → ↻",
        "stages": ["position", "available moves", "consequence", "repositioning"],
        "observes": [
            "rules, incentives, and legibility",
            "moves that preserve or terminate play",
            "changes in position and viable move space",
        ],
    },
}


def memory_loop_for(agent_id: str) -> dict[str, Any]:
    return MEMORY_LOOPS[agent_id]
