from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .models import AGENT_IDS, ResearchMode

SEARCH_PATTERNS = (
    r"\bsearch(?: online| the web)?\b",
    r"\blook (?:it|this|that|them) up\b",
    r"\bfind (?:me )?(?:sources?|results?|information|details|evidence|articles?|papers?|documentation)\b",
    r"\b(latest|current|currently|today|tonight|this week|recent|newest|up[- ]to[- ]date)\b",
    r"\b(news|price|prices|schedule|weather|forecast|score|standings|release notes|version)\b",
    r"\b(verify|fact[- ]check|confirm|cite|citation|source|sourced|research)\b",
    r"\bwho (?:is|are) (?:the )?(?:president|prime minister|ceo|cto|mayor|governor|director|chair|head|leader)\b",
    r"\bwho won\b",
    r"\bwhat happened (?:today|yesterday|this week|recently)?\b",
    r"\bwhen (?:is|does|will) .{0,80}(?:open|close|start|end|release|launch|play|air|begin)\b",
    r"\bwhere (?:can|should) i (?:buy|find|watch|stream|stay|eat|go|visit)\b",
    r"\b(?:recommend|recommendation|recommendations|best place|best product|best option)\b",
    r"https?://",
)

ALL_RESEARCH_PATTERNS = (
    r"\bdeep research\b",
    r"\bcomprehensive research\b",
    r"\bcompare (?:multiple )?sources\b",
    r"\ball three\b",
    r"\beach of you\b",
    r"\bfrom each agent\b",
    r"\bmultiple perspectives\b",
    r"\blegal\b",
    r"\bmedical\b",
    r"\bfinancial advice\b",
)

PHENOMENOLOGY_TERMS = {
    "phenomenology",
    "embodied",
    "embodiment",
    "interoception",
    "perception",
    "experience",
    "lived",
    "umwelt",
    "consciousness",
    "affect",
    "body",
    "motif",
    "taste",
    "aesthetic",
    "culture",
    "art",
    "social",
    "first-person",
}

CYBERNETICS_TERMS = {
    "cybernetics",
    "feedback",
    "signal",
    "noise",
    "black box",
    "entrainment",
    "algorithm",
    "bot",
    "metrics",
    "data",
    "experiment",
    "api",
    "code",
    "docker",
    "python",
    "javascript",
    "software",
    "protocol",
    "library",
    "package",
    "server",
    "database",
    "model",
    "spec",
    "documentation",
    "github",
    "implementation",
    "security",
}

GAME_THEORY_TERMS = {
    "game theory",
    "game",
    "rules",
    "player",
    "position",
    "strategy",
    "incentive",
    "finite",
    "infinite play",
    "local minimum",
    "vector of flight",
    "negotiation",
    "competition",
    "law",
    "policy",
    "governance",
    "institution",
    "market",
    "company",
    "politics",
    "election",
    "regulation",
    "economy",
    "organization",
}


@dataclass(frozen=True)
class SearchDecision:
    needs_search: bool
    scope: str
    lead_agent: str | None
    reason: str

    def model_dump(self) -> dict:
        return asdict(self)


class SearchRouter:
    def decide(
        self,
        message: str,
        mode: ResearchMode,
        participants: list[str],
    ) -> SearchDecision:
        selected = [agent_id for agent_id in AGENT_IDS if agent_id in participants]
        if not selected:
            selected = list(AGENT_IDS)

        if mode == "off":
            return SearchDecision(False, "none", None, "Research mode is disabled.")

        detected = any(re.search(pattern, message, flags=re.IGNORECASE) for pattern in SEARCH_PATTERNS)
        if mode in {"lead", "all"}:
            detected = True
        if not detected:
            return SearchDecision(False, "none", None, "No online-information signal was detected.")

        all_requested = mode == "all" or any(
            re.search(pattern, message, flags=re.IGNORECASE) for pattern in ALL_RESEARCH_PATTERNS
        )
        if all_requested:
            return SearchDecision(
                True,
                "all",
                None,
                "The request calls for deep, comparative, multi-agent, or high-stakes research.",
            )

        lead = self._select_lead(message, selected)
        return SearchDecision(
            True,
            "lead",
            lead,
            "One motif-aligned research lead can ground the room without duplicating searches.",
        )

    @staticmethod
    def _select_lead(message: str, participants: list[str]) -> str:
        lowered = message.lower()
        scores = {agent_id: 0 for agent_id in participants}

        for term in PHENOMENOLOGY_TERMS:
            if term in lowered and "agent_a" in scores:
                scores["agent_a"] += 2
        for term in CYBERNETICS_TERMS:
            if term in lowered and "agent_b" in scores:
                scores["agent_b"] += 2
        for term in GAME_THEORY_TERMS:
            if term in lowered and "agent_c" in scores:
                scores["agent_c"] += 2

        explicit_names = {
            "agent_a": ("phenomenologist", "phenomenology", "agent a"),
            "agent_b": ("cyberneticist", "cybernetics", "agent b"),
            "agent_c": ("game theorist", "game theory", "agent c"),
        }
        for agent_id, names in explicit_names.items():
            if agent_id in scores and any(name in lowered for name in names):
                scores[agent_id] += 10

        order = [agent_id for agent_id in AGENT_IDS if agent_id in participants]
        return max(order, key=lambda agent_id: (scores[agent_id], -order.index(agent_id)))
