const TOOL_TARGETS = Object.freeze({
  write_project_file: "files",
  propose_persona_update: "proposals",
  record_motif_observations: "motifs",
});

export function createTurnRefreshState() {
  return {
    files: false,
    motifs: false,
    proposals: false,
    sources: false,
  };
}

export function observeTurnRefreshEvent(state, event) {
  if (!event || typeof event !== "object") return state;
  if (event.type === "source_fetch_complete") state.sources = true;
  if (event.type !== "tool" || event.result?.ok === false) return state;
  const target = TOOL_TARGETS[event.tool];
  if (target) state[target] = true;
  return state;
}

export function observeTurnRefreshResult(state, result) {
  if (!result || typeof result !== "object") return state;
  if (result.web_sources?.length) state.sources = true;
  for (const message of result.messages || []) {
    for (const event of message.metadata?.tool_events || []) {
      observeTurnRefreshEvent(state, { type: "tool", ...event });
    }
  }
  return state;
}

export function turnRefreshTargets(state) {
  const targets = ["messages", "memory", "recovery"];
  for (const optional of ["motifs", "files", "sources", "proposals"]) {
    if (state[optional]) targets.push(optional);
  }
  return targets;
}
