from __future__ import annotations

import re

MEMORY_RELEVANCE_STOPWORDS = {
    "about",
    "after",
    "again",
    "could",
    "from",
    "have",
    "into",
    "more",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}


class MemoryContext:
    @classmethod
    def _select_memory_history(
        cls,
        events: list[dict],
        *,
        query: str,
        limit: int,
        recent: list[dict] | None = None,
        agent_id: str | None = None,
    ) -> list[dict]:
        if limit <= 0:
            return []
        covered_user_messages: set[str] = set()
        if recent is not None and agent_id is not None:
            for message in recent:
                if message.get("role") != "agent" or message.get("agent_id") != agent_id:
                    continue
                metadata = message.get("metadata")
                if isinstance(metadata, dict) and metadata.get("user_message_id"):
                    covered_user_messages.add(str(metadata["user_message_id"]))
        candidates: list[dict] = []
        for event in events:
            candidate = dict(event)
            if str(event.get("user_message_id", "")) in covered_user_messages:
                candidate["_transcript_covered"] = True
            candidates.append(candidate)
        query_terms = cls._memory_terms(query)

        def rank(event: dict) -> tuple[int, int]:
            event_text = " ".join(
                str(event.get(key, ""))
                for key in (
                    "trigger_text",
                    "return_text",
                    "trigger_summary",
                    "return_summary",
                    "source_project_name",
                )
            )
            overlap = len(query_terms & cls._memory_terms(event_text))
            return overlap, int(event.get("sequence") or 0)

        return sorted(candidates, key=rank, reverse=True)[:limit]

    @staticmethod
    def _consolidate_memory_turns(events: list[dict]) -> list[dict]:
        """Group response beats for prompt context while preserving the raw ledger."""
        grouped: dict[tuple[str, str], list[dict]] = {}
        order: list[tuple[str, str]] = []
        for event in events:
            project_id = str(event.get("project_id") or event.get("source_project_id") or "")
            user_message_id = str(event.get("user_message_id") or event.get("id") or "")
            key = project_id, user_message_id
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(event)

        consolidated: list[dict] = []
        for key in order:
            beats = sorted(
                grouped[key],
                key=lambda event: int(event.get("sequence") or 0),
            )
            combined = dict(beats[-1])
            combined["actions"] = MemoryContext._merge_memory_actions(beats)
            combined["outcome"] = "+".join(
                dict.fromkeys(str(beat.get("outcome", "")) for beat in beats)
            ).strip("+")
            combined["trigger_text"] = next(
                (str(beat.get("trigger_text", "")) for beat in beats if beat.get("trigger_text")),
                "",
            )
            combined["return_text"] = "\n\n".join(
                str(beat.get("return_text", "")).strip()
                for beat in beats
                if str(beat.get("return_text", "")).strip()
            )
            combined["trigger_summary"] = next(
                (
                    str(beat.get("trigger_summary", ""))
                    for beat in beats
                    if beat.get("trigger_summary")
                ),
                "",
            )
            combined["return_summary"] = " ".join(
                str(beat.get("return_summary", "")).strip()
                for beat in beats
                if str(beat.get("return_summary", "")).strip()
            )
            combined["_evidence_event_ids"] = [str(beat["id"]) for beat in beats if beat.get("id")]
            combined["_source_memory_event_ids"] = [
                str(beat["source_memory_event_id"])
                for beat in beats
                if beat.get("source_memory_event_id")
            ]
            consolidated.append(combined)
        return consolidated

    @staticmethod
    def _merge_memory_actions(events: list[dict]) -> list[dict]:
        actions: list[dict] = []
        seen: set[str] = set()
        for event in events:
            for action in event.get("actions", []):
                if not isinstance(action, dict):
                    continue
                key = repr(sorted(action.items()))
                if key in seen:
                    continue
                seen.add(key)
                actions.append(action)
        return actions

    @staticmethod
    def _memory_terms(text: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[a-z0-9_]{4,}", str(text).lower())
            if term not in MEMORY_RELEVANCE_STOPWORDS
        }
