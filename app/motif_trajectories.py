from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence

from .constants import (
    MOTIF_PATTERN_MIN_DISTINCT_TURNS,
    MOTIF_TRAJECTORY_WINDOW,
)


def motif_sequence_summary(
    sequence: Sequence[Hashable],
    *,
    turn_ids: Sequence[str | None] | None = None,
    labels: dict[Hashable, str] | None = None,
    window: int = MOTIF_TRAJECTORY_WINDOW,
    pattern_min_distinct_turns: int = MOTIF_PATTERN_MIN_DISTINCT_TURNS,
) -> dict:
    """Describe recurrence, transitions, and repeated motif sequences."""
    values = tuple(sequence[-window:])
    if turn_ids is None:
        turns = tuple(f"position:{index}" for index in range(len(values)))
    else:
        if len(turn_ids) != len(sequence):
            raise ValueError("turn_ids must align with the motif sequence.")
        turns = tuple(turn_ids[-window:])
    label_by_value = labels or {}
    seen: set[Hashable] = set()
    repeated_positions = 0
    for value in values:
        if value in seen:
            repeated_positions += 1
        seen.add(value)
    transitions = list(zip(values, values[1:], strict=False))
    unique_transitions = len(set(transitions))
    length = len(values)
    transition_count = len(transitions)
    patterns = _frequent_patterns(
        values,
        turns,
        label_by_value,
        minimum_distinct_turns=pattern_min_distinct_turns,
    )
    return_patterns = _return_patterns(
        values,
        turns,
        label_by_value,
        minimum_distinct_turns=pattern_min_distinct_turns,
    )
    return {
        "sample_size": length,
        "recurrence_rate": round(repeated_positions / length, 3) if length else 0.0,
        "transition_diversity": (
            round(unique_transitions / transition_count, 3)
            if transition_count
            else 0.0
        ),
        "pattern_min_distinct_turns": pattern_min_distinct_turns,
        "frequent_patterns": patterns,
        "return_patterns": return_patterns,
    }


def _frequent_patterns(
    values: Sequence[Hashable],
    turn_ids: Sequence[str | None],
    labels: dict[Hashable, str],
    *,
    minimum_distinct_turns: int,
) -> list[dict]:
    records = []
    for size in (2, 3):
        occurrences: dict[tuple[Hashable, ...], dict] = defaultdict(
            lambda: {"occurrence_count": 0, "turn_ids": set()}
        )
        for start in range(0, len(values) - size + 1):
            pattern = tuple(values[start : start + size])
            anchor = turn_ids[start + size - 1] or f"position:{start + size - 1}"
            occurrences[pattern]["occurrence_count"] += 1
            occurrences[pattern]["turn_ids"].add(anchor)
        for pattern, stats in occurrences.items():
            distinct_turn_count = len(stats["turn_ids"])
            if distinct_turn_count < minimum_distinct_turns:
                continue
            records.append(
                {
                    "motif_ids": list(pattern),
                    "labels": [labels.get(value, str(value)) for value in pattern],
                    "length": size,
                    "occurrence_count": stats["occurrence_count"],
                    "distinct_turn_count": distinct_turn_count,
                }
            )
    records.sort(
        key=lambda item: (
            -item["distinct_turn_count"],
            -item["occurrence_count"],
            -item["length"],
            item["labels"],
        )
    )
    return records[:8]


def _return_patterns(
    values: Sequence[Hashable],
    turn_ids: Sequence[str | None],
    labels: dict[Hashable, str],
    *,
    minimum_distinct_turns: int,
) -> list[dict]:
    occurrences: dict[tuple[Hashable, ...], dict] = defaultdict(
        lambda: {"occurrence_count": 0, "turn_ids": set()}
    )
    for size in range(3, min(6, len(values)) + 1):
        for start in range(0, len(values) - size + 1):
            pattern = tuple(values[start : start + size])
            if pattern[0] != pattern[-1]:
                continue
            anchor = turn_ids[start + size - 1] or f"position:{start + size - 1}"
            occurrences[pattern]["occurrence_count"] += 1
            occurrences[pattern]["turn_ids"].add(anchor)
    records = []
    for pattern, stats in occurrences.items():
        distinct_turn_count = len(stats["turn_ids"])
        if distinct_turn_count < minimum_distinct_turns:
            continue
        records.append(
            {
                "motif_ids": list(pattern),
                "labels": [labels.get(value, str(value)) for value in pattern],
                "length": len(pattern),
                "occurrence_count": stats["occurrence_count"],
                "distinct_turn_count": distinct_turn_count,
            }
        )
    records.sort(
        key=lambda item: (
            -item["distinct_turn_count"],
            -item["occurrence_count"],
            item["length"],
            item["labels"],
        )
    )
    return records[:6]
