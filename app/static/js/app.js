import { api, setSessionToken, streamApi } from "./api.js";
import { renderCodeViewer } from "./code_viewer.js";
import { appendRoleSignal, createDemoController } from "./demo_controller.js";
import { renderMarkdown } from "./markdown.js";
import { decorateMessageNode, reconcileMessageNodes } from "./message_reconciler.js";
import { TurnQueue } from "./turn_queue.js";
import {
  createTurnRefreshState,
  observeTurnRefreshEvent,
  observeTurnRefreshResult,
  turnRefreshTargets,
} from "./turn_refresh.js";

const UI_DEFAULTS = Object.freeze({
  toastDurationMs: 4_200,
  liveOutputFollowThresholdPx: 40,
  livePromptTailChars: 256,
  liveRunPollMs: 180,
  liveRunInitialPollMs: 80,
  canceledRunRefreshDelayMs: 500,
  conversationFollowThresholdPx: 72,
  agentFileMaxBytes: 15_000,
  agentTurnBeats: 3,
  promptQueueMaxItems: 20,
  proposalListMaxItems: 20,
  narrowViewportQuery: "(max-width: 720px)",
});

const state = {
  session: null,
  projects: [],
  agents: [],
  currentProject: "general",
  currentPersona: "agent_a",
  busy: false,
  activePrompt: null,
  promptQueue: new TurnQueue(UI_DEFAULTS.promptQueueMaxItems),
  progressNode: null,
  leftCollapsed: false,
  rightCollapsed: false,
  activeDemo: null,
  activeRunController: null,
  activeRunId: null,
  activeRunCursor: 0,
  activeRunPollTimer: null,
  activeRunInputClosed: false,
  activeRunPromptTail: "",
  renderedProject: null,
  motifData: null,
  selectedMotif: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const elements = {
  appGrid: $("#app-grid"),
  toggleLeftPanel: $("#toggle-left-panel"),
  toggleRightPanel: $("#toggle-right-panel"),
  status: $("#system-status"),
  projectSelect: $("#project-select"),
  newProjectName: $("#new-project-name"),
  createProject: $("#create-project"),
  deleteProject: $("#delete-project"),
  projectTitle: $("#project-title"),
  downloadLog: $("#download-log"),
  researchMode: $("#research-mode"),
  researchIndicator: $("#research-indicator"),
  setupWarning: $("#setup-warning"),
  turnRecovery: $("#turn-recovery"),
  messages: $("#messages"),
  composer: $("#composer"),
  messageInput: $("#message-input"),
  promptQueue: $("#prompt-queue"),
  sendButton: $("#send-button"),
  personaSelect: $("#persona-select"),
  personaSummary: $("#persona-summary"),
  memoryLoopSummary: $("#memory-loop-summary"),
  memoryLoopEvents: $("#memory-loop-events"),
  globalMemorySummary: $("#global-memory-summary"),
  globalMemoryEvents: $("#global-memory-events"),
  motifAgentFilter: $("#motif-agent-filter"),
  motifStatusFilter: $("#motif-status-filter"),
  motifCheckpoints: $("#motif-checkpoints"),
  motifPatterns: $("#motif-patterns"),
  motifTrajectories: $("#motif-trajectories"),
  motifTags: $("#motif-tags"),
  motifDetail: $("#motif-detail"),
  personaEditor: $("#persona-editor"),
  reloadPersona: $("#reload-persona"),
  savePersona: $("#save-persona"),
  proposalCount: $("#proposal-count"),
  proposalList: $("#proposal-list"),
  sharedContextEditor: $("#shared-context-editor"),
  reloadSharedContext: $("#reload-shared-context"),
  saveSharedContext: $("#save-shared-context"),
  keyStatus: $("#key-status"),
  setupForm: $("#setup-form"),
  providerA: $("#provider-agent-a"),
  providerB: $("#provider-agent-b"),
  providerC: $("#provider-agent-c"),
  modelA: $("#model-agent-a"),
  modelB: $("#model-agent-b"),
  modelC: $("#model-agent-c"),
  modelOptionsA: $("#model-options-agent-a"),
  modelOptionsB: $("#model-options-agent-b"),
  modelOptionsC: $("#model-options-agent-c"),
  defaultResearchMode: $("#default-research-mode"),
  temperature: $("#temperature"),
  maxTokens: $("#max-tokens"),
  providerCatalogEditor: $("#provider-catalog-editor"),
  reloadProviderCatalog: $("#reload-provider-catalog"),
  saveProviderCatalog: $("#save-provider-catalog"),
  uploadForm: $("#upload-form"),
  fileInput: $("#file-input"),
  fileList: $("#file-list"),
  filePreview: $("#file-preview"),
  demoOverlay: $("#demo-overlay"),
  demoTitle: $("#demo-title"),
  demoRefresh: $("#demo-refresh"),
  demoCollapse: $("#demo-collapse"),
  demoClose: $("#demo-close"),
  demoFrame: $("#demo-frame"),
  demoOutput: $("#demo-output"),
  demoRunControls: $("#demo-run-controls"),
  demoArguments: $("#demo-arguments"),
  demoStdin: $("#demo-stdin"),
  demoStart: $("#demo-start"),
  demoCancel: $("#demo-cancel"),
  demoSendInput: $("#demo-send-input"),
  demoSendEof: $("#demo-send-eof"),
  demoRoleSignals: $("#demo-role-signals"),
  newReturns: $("#new-returns"),
  sourceList: $("#source-list"),
  sourcePreview: $("#source-preview"),
  toast: $("#toast"),
};

const LAYOUT_STORAGE_KEY = "local-motif-feedback-layout";

const agentMeta = {
  agent_a: { name: "The Phenomenologist", className: "agent-a" },
  agent_b: { name: "The Cyberneticist", className: "agent-b" },
  agent_c: { name: "The Game Theorist", className: "agent-c" },
};

function showToast(message, isError = false) {
  elements.toast.textContent = message;
  elements.toast.classList.remove("hidden");
  elements.toast.style.boxShadow = isError ? "5px 5px 0 var(--danger)" : "5px 5px 0 var(--hot)";
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(
    () => elements.toast.classList.add("hidden"),
    UI_DEFAULTS.toastDurationMs,
  );
}

function setStatus(text, mode = "") {
  elements.status.textContent = text;
  elements.status.className = `system-status ${mode}`.trim();
}

function applyPanelLayout() {
  elements.appGrid.classList.toggle("left-collapsed", state.leftCollapsed);
  elements.appGrid.classList.toggle("right-collapsed", state.rightCollapsed);
  elements.toggleLeftPanel.setAttribute("aria-expanded", String(!state.leftCollapsed));
  elements.toggleRightPanel.setAttribute("aria-expanded", String(!state.rightCollapsed));
  elements.toggleLeftPanel.textContent = state.leftCollapsed ? "PROJECTS ▶" : "PROJECTS ◀";
  elements.toggleRightPanel.textContent = state.rightCollapsed ? "INSPECTOR ◀" : "INSPECTOR ▶";
  elements.toggleLeftPanel.title = state.leftCollapsed ? "Show project controls" : "Hide project controls";
  elements.toggleRightPanel.title = state.rightCollapsed ? "Show inspector" : "Hide inspector";
}

function loadPanelLayout() {
  try {
    const saved = JSON.parse(localStorage.getItem(LAYOUT_STORAGE_KEY) || "{}");
    state.leftCollapsed = saved.leftCollapsed === true;
    state.rightCollapsed = saved.rightCollapsed === true;
  } catch {
    state.leftCollapsed = false;
    state.rightCollapsed = false;
  }
  applyPanelLayout();
}

function savePanelLayout() {
  try {
    localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify({
      leftCollapsed: state.leftCollapsed,
      rightCollapsed: state.rightCollapsed,
    }));
  } catch { }
}

window.addEventListener("motif:session-reconnected", () => {
  setStatus("READY", "ok");
  showToast("Reconnected to the local server.");
});

function setBusy(busy) {
  state.busy = busy;
  const setupComplete = state.session?.setup_complete !== false;
  elements.sendButton.disabled = !setupComplete;
  elements.messageInput.disabled = !setupComplete;
  elements.deleteProject.disabled = (
    busy
    || state.promptQueue.length > 0
    || !state.currentProject
  );
  elements.turnRecovery.querySelectorAll("button").forEach((button) => {
    button.disabled = busy;
  });
  elements.sendButton.textContent = busy ? "QUEUE ↵" : "SEND ↵";
}

function currentParticipants() {
  return $$('.agent-toggle input[type="checkbox"]:checked').map((input) => input.value);
}

function renderPromptQueue() {
  const queued = state.promptQueue.snapshot();
  const active = state.activePrompt;
  elements.promptQueue.replaceChildren();
  elements.promptQueue.classList.toggle("hidden", !active && !queued.length);
  if (!active && !queued.length) return;

  const head = document.createElement("div");
  head.className = "prompt-queue-head";
  const activity = document.createElement("span");
  activity.textContent = active ? "ROOM TURN RUNNING" : "TURN QUEUE";
  const count = document.createElement("span");
  count.textContent = `${queued.length} WAITING`;
  head.append(activity, count);

  const list = document.createElement("div");
  list.className = "prompt-queue-list";
  for (const turn of queued) {
    const item = document.createElement("div");
    item.className = "prompt-queue-item";
    const copy = document.createElement("div");
    copy.className = "prompt-queue-copy";
    const message = document.createElement("strong");
    message.textContent = turn.message;
    message.title = turn.message;
    const project = document.createElement("small");
    project.textContent = `${turn.projectName} · ${turn.participants.length} AGENT${turn.participants.length === 1 ? "" : "S"} · ${turn.researchMode}`;
    copy.append(message, project);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "prompt-queue-remove";
    remove.dataset.queuedTurnId = turn.id;
    remove.textContent = "REMOVE";
    remove.setAttribute("aria-label", `Remove queued prompt: ${turn.message}`);
    item.append(copy, remove);
    list.append(item);
  }
  elements.promptQueue.append(head, list);
}

function userDisplayName() {
  return state.session?.user_display_name || "User";
}

const {
  cancelLiveRun,
  closeLiveRunInput,
  openHtmlDemo,
  preparePythonDemo,
  refreshActiveDemo,
  refreshHtmlDemoInPlace,
  runPythonDemo,
  sendLiveRunInput,
  showRoleSignals,
} = createDemoController({
  state,
  elements,
  defaults: UI_DEFAULTS,
  userDisplayName,
  showToast,
  loadMessages,
});

function setupIncomplete(session) {
  return !session.setup_complete || !session.key_configured;
}

function renderProjects() {
  elements.projectSelect.replaceChildren();
  for (const project of state.projects) {
    const option = document.createElement("option");
    option.value = project.id;
    option.textContent = project.name.toUpperCase();
    option.selected = project.id === state.currentProject;
    elements.projectSelect.append(option);
  }
  const current = state.projects.find((project) => project.id === state.currentProject);
  elements.projectTitle.textContent = (current?.name || state.currentProject).toUpperCase();
  elements.deleteProject.disabled = (
    state.busy
    || state.promptQueue.length > 0
    || !current
  );
  elements.downloadLog.disabled = !current;
}

function renderAgentOptions() {
  elements.personaSelect.replaceChildren();
  for (const agent of state.agents) {
    agentMeta[agent.agent_id].name = agent.display_name;
    const option = document.createElement("option");
    option.value = agent.agent_id;
    option.textContent = `${agent.display_name.toUpperCase()} / V${agent.version}`;
    option.selected = agent.agent_id === state.currentPersona;
    elements.personaSelect.append(option);
  }
}

function motifMatchesFilters(motif) {
  const agent = elements.motifAgentFilter.value;
  const status = elements.motifStatusFilter.value;
  const agentMatches = agent === "all" || motif.observer_agent_id === agent;
  const statusMatches = status === "all"
    || (status === "current"
      ? ["candidate", "supported", "active"].includes(motif.status)
      : motif.status === status);
  return agentMatches && statusMatches;
}

function motifAgentVisible(agentId) {
  const selected = elements.motifAgentFilter.value;
  return selected === "all" || selected === agentId;
}

function renderMotifTrajectories() {
  const fragment = document.createDocumentFragment();
  for (const agentId of Object.keys(agentMeta)) {
    if (!motifAgentVisible(agentId)) continue;
    const measurements = state.motifData?.trajectories?.[agentId] || {};
    const observed = measurements.observed || {};
    const established = measurements.established || {};
    const usesEstablished = Number(established.sample_size || 0) > 0;
    const summary = usesEstablished ? established : observed;
    const card = document.createElement("article");
    card.className = `motif-trajectory-card ${agentMeta[agentId].className}`;
    const name = document.createElement("strong");
    name.textContent = agentMeta[agentId].name.toUpperCase();
    const value = document.createElement("span");
    value.textContent = `${usesEstablished ? "ESTABLISHED" : "OBSERVED"} · ${summary.sample_size || 0} PRIMARY OBSERVATIONS`;
    const note = document.createElement("small");
    const recurrence = `${Math.round((summary.recurrence_rate || 0) * 100)}% RETURNS`;
    const transitions = `${Math.round((summary.transition_diversity || 0) * 100)}% TRANSITION VARIETY`;
    note.textContent = `${recurrence} · ${transitions}`;
    card.append(name, value, note);
    fragment.append(card);
  }
  elements.motifTrajectories.replaceChildren(fragment);
}

function patternPreferenceCopy(checkpoint) {
  const name = agentMeta[checkpoint.observer_agent_id]?.name || checkpoint.observer_agent_id;
  if (checkpoint.preference === "follow") {
    return `You asked ${name} to keep following and deepen this pattern when relevant.`;
  }
  if (checkpoint.preference === "test") {
    return `You asked ${name} to test this pattern's limits when relevant.`;
  }
  if (checkpoint.preference === "paused") {
    return `Paused. This checkpoint is not shown to ${name}.`;
  }
  return `Available to ${name} as quiet reflection context; no response is required.`;
}

function renderMotifCheckpoints() {
  const checkpoints = (state.motifData?.checkpoints || [])
    .filter((checkpoint) => motifAgentVisible(checkpoint.observer_agent_id));
  if (!checkpoints.length) {
    const empty = document.createElement("p");
    empty.className = "microcopy";
    empty.textContent = "No established sequence has crossed the checkpoint threshold yet.";
    elements.motifCheckpoints.replaceChildren(empty);
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const checkpoint of checkpoints) {
    const card = document.createElement("article");
    card.className = `motif-checkpoint ${agentMeta[checkpoint.observer_agent_id]?.className || ""} preference-${checkpoint.preference}`;
    const head = document.createElement("div");
    head.className = "motif-checkpoint-head";
    const kind = document.createElement("strong");
    kind.textContent = checkpoint.kind === "return_path" ? "RETURN PATH" : "REPEATING SEQUENCE";
    const observer = document.createElement("small");
    observer.textContent = (agentMeta[checkpoint.observer_agent_id]?.name || checkpoint.observer_agent_id).toUpperCase();
    head.append(kind, observer);
    const sequence = document.createElement("p");
    sequence.className = "motif-checkpoint-sequence";
    sequence.textContent = checkpoint.labels.join(" → ");
    const support = document.createElement("small");
    support.className = "motif-checkpoint-support";
    support.textContent = `${checkpoint.distinct_turn_count} DISTINCT TURNS · ${checkpoint.occurrence_count} OCCURRENCES`;
    const explanation = document.createElement("p");
    explanation.className = "motif-checkpoint-explanation";
    explanation.textContent = patternPreferenceCopy(checkpoint);
    const actions = document.createElement("div");
    actions.className = "motif-checkpoint-actions";
    const choices = [
      ["notice", "JUST NOTICE", "Let the agent recognize it only when naturally relevant."],
      ["follow", "FOLLOW", "Invite the agent to stay with and deepen the pattern."],
      ["test", "TEST", "Invite the agent to look for a boundary or counterexample."],
      ["paused", "PAUSE", "Stop supplying this checkpoint to the agent."],
    ];
    for (const [preference, label, title] of choices) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.title = title;
      button.setAttribute("aria-label", `${label}: ${title}`);
      button.disabled = checkpoint.preference === preference;
      button.addEventListener("click", () => updatePatternPreference(checkpoint, preference));
      actions.append(button);
    }
    card.append(head, sequence, support, explanation, actions);
    fragment.append(card);
  }
  elements.motifCheckpoints.replaceChildren(fragment);
}

function renderMotifPatterns() {
  const fragment = document.createDocumentFragment();
  for (const agentId of Object.keys(agentMeta)) {
    if (!motifAgentVisible(agentId)) continue;
    const measurements = state.motifData?.trajectories?.[agentId] || {};
    const established = measurements.established || {};
    const observed = measurements.observed || {};
    const establishedPatterns = [
      ...(established.return_patterns || []),
      ...(established.frequent_patterns || []),
    ];
    const observedPatterns = [
      ...(observed.return_patterns || []),
      ...(observed.frequent_patterns || []),
    ];
    const usesEstablished = establishedPatterns.length > 0;
    const patterns = (usesEstablished ? establishedPatterns : observedPatterns).slice(0, 6);
    const card = document.createElement("article");
    card.className = `motif-pattern-card ${agentMeta[agentId].className}`;
    const head = document.createElement("div");
    head.className = "motif-pattern-head";
    const name = document.createElement("strong");
    name.textContent = agentMeta[agentId].name.toUpperCase();
    const basis = document.createElement("small");
    basis.textContent = usesEstablished ? "ESTABLISHED THREADS" : "OBSERVED DIAGNOSTIC";
    head.append(name, basis);
    card.append(head);
    if (!patterns.length) {
      const empty = document.createElement("p");
      empty.textContent = `No sequence has recurred across ${observed.pattern_min_distinct_turns || 3} distinct turns.`;
      card.append(empty);
    } else {
      const seen = new Set();
      for (const pattern of patterns) {
        const key = pattern.motif_ids.join("→");
        if (seen.has(key)) continue;
        seen.add(key);
        const row = document.createElement("div");
        row.className = "motif-pattern";
        const sequence = document.createElement("span");
        sequence.textContent = pattern.labels.join(" → ");
        const support = document.createElement("small");
        const isReturn = pattern.motif_ids[0] === pattern.motif_ids.at(-1)
          && pattern.motif_ids.length > 2;
        support.textContent = `${pattern.distinct_turn_count} TURNS · ${pattern.occurrence_count} OCCURRENCES${isReturn ? " · RETURN" : ""}`;
        row.append(sequence, support);
        card.append(row);
      }
    }
    fragment.append(card);
  }
  elements.motifPatterns.replaceChildren(fragment);
}

function renderMotifTags() {
  const motifs = (state.motifData?.motifs || []).filter(motifMatchesFilters);
  if (!motifs.length) {
    const empty = document.createElement("p");
    empty.className = "microcopy";
    empty.textContent = state.motifData?.motifs?.length
      ? "No motifs match these filters."
      : "No motif observations yet. They will appear after agents notice a meaningful pattern.";
    elements.motifTags.replaceChildren(empty);
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const motif of motifs) {
    const tag = document.createElement("button");
    tag.type = "button";
    tag.className = `motif-tag ${agentMeta[motif.observer_agent_id]?.className || ""} status-${motif.status}`;
    if (motif.id === state.selectedMotif) tag.classList.add("selected");
    const label = document.createElement("span");
    label.textContent = motif.label;
    const count = document.createElement("small");
    count.textContent = `${motif.distinct_turn_count} turn${motif.distinct_turn_count === 1 ? "" : "s"} · ${motif.support_count} obs · ${motif.status}`;
    tag.title = motif.description;
    tag.append(label, count);
    tag.addEventListener("click", () => openMotif(motif.id));
    fragment.append(tag);
  }
  elements.motifTags.replaceChildren(fragment);
}

function renderMotifDetail(motif) {
  if (!motif) {
    elements.motifDetail.textContent = "Select a motif tag.";
    return;
  }
  const heading = document.createElement("div");
  heading.className = `motif-detail-head ${agentMeta[motif.observer_agent_id]?.className || ""}`;
  const title = document.createElement("strong");
  title.textContent = motif.label;
  const meta = document.createElement("small");
  meta.textContent = `${agentMeta[motif.observer_agent_id]?.name || motif.observer_agent_id} · ${motif.status} · ${motif.distinct_turn_count} distinct turns · ${motif.support_count} observations · ${Math.round(motif.confidence * 100)}% mean confidence`;
  heading.append(title, meta);
  const description = document.createElement("p");
  description.textContent = motif.description;
  const aliases = document.createElement("div");
  aliases.className = "motif-aliases";
  aliases.textContent = `ALIASES: ${(motif.aliases || []).join(" · ") || motif.label}`;

  const actions = document.createElement("div");
  actions.className = "motif-actions";
  for (const status of ["active", "dormant", "rejected"]) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = status.toUpperCase();
    button.className = status === "rejected" ? "danger" : "";
    button.disabled = motif.status === status;
    button.addEventListener("click", () => updateMotifStatus(motif.id, status));
    actions.append(button);
  }

  const historyLabel = document.createElement("div");
  historyLabel.className = "motif-history-label";
  historyLabel.textContent = "APPEND-ONLY OBSERVATION HISTORY";
  const history = document.createElement("div");
  history.className = "motif-history";
  for (const event of motif.events || []) {
    const row = document.createElement("article");
    const head = document.createElement("strong");
    const relation = event.relation ? ` · ${event.relation}` : "";
    head.textContent = `${event.event_type.replaceAll("_", " ")}${relation}${event.primary ? " · primary" : ""}`;
    const copy = document.createElement("p");
    copy.textContent = event.description;
    const time = document.createElement("small");
    time.textContent = new Date(event.created_at).toLocaleString();
    row.append(head, copy);
    if (event.evidence?.length) {
      const evidence = document.createElement("div");
      evidence.className = "motif-evidence";
      for (const message of event.evidence) {
        const excerpt = document.createElement("blockquote");
        const speaker = message.role === "user"
          ? userDisplayName()
          : agentMeta[message.agent_id]?.name || message.agent_id || message.role;
        excerpt.textContent = `${String(speaker).toUpperCase()}: ${message.excerpt}`;
        evidence.append(excerpt);
      }
      row.append(evidence);
    }
    row.append(time);
    history.append(row);
  }

  const relationLabel = document.createElement("div");
  relationLabel.className = "motif-history-label";
  relationLabel.textContent = "PROVISIONAL RELATIONS · NO AUTOMATIC MERGING";
  const relations = document.createElement("div");
  relations.className = "motif-relations";
  if (!(motif.relations || []).length) {
    relations.textContent = "No cross-motif relations recorded.";
  } else {
    for (const relation of motif.relations) {
      const row = document.createElement("article");
      const otherIsSource = relation.target_motif_id === motif.id;
      const otherLabel = otherIsSource ? relation.source_label : relation.target_label;
      const otherAgent = otherIsSource
        ? relation.source_observer_agent_id
        : relation.target_observer_agent_id;
      const head = document.createElement("strong");
      head.textContent = `${relation.relation.replaceAll("_", " ")} ↔ ${otherLabel}`;
      const copy = document.createElement("p");
      copy.textContent = relation.description;
      const observer = document.createElement("small");
      observer.textContent = `ASSERTED BY ${agentMeta[relation.observer_agent_id]?.name || relation.observer_agent_id} · OTHER MOTIF OWNED BY ${agentMeta[otherAgent]?.name || otherAgent} · ${Math.round(relation.confidence * 100)}%`;
      row.append(head, copy, observer);
      relations.append(row);
    }
  }
  elements.motifDetail.replaceChildren(
    heading,
    description,
    aliases,
    actions,
    relationLabel,
    relations,
    historyLabel,
    history,
  );
}

async function openMotif(motifId) {
  state.selectedMotif = motifId;
  renderMotifTags();
  elements.motifDetail.textContent = "Loading motif history...";
  try {
    const motif = await api(
      `/api/motifs/${encodeURIComponent(state.currentProject)}/${encodeURIComponent(motifId)}`,
    );
    if (state.selectedMotif === motifId) renderMotifDetail(motif);
  } catch (error) {
    elements.motifDetail.textContent = "Could not load this motif.";
    showToast(error.message, true);
  }
}

async function updateMotifStatus(motifId, status) {
  try {
    const motif = await api(
      `/api/motifs/${encodeURIComponent(state.currentProject)}/${encodeURIComponent(motifId)}/status`,
      { method: "PUT", body: JSON.stringify({ status }) },
    );
    await loadMotifs();
    state.selectedMotif = motifId;
    renderMotifTags();
    renderMotifDetail(motif);
    showToast(`Motif marked ${status}.`);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function updatePatternPreference(checkpoint, preference) {
  try {
    await api(
      `/api/motif-patterns/${encodeURIComponent(state.currentProject)}/${encodeURIComponent(checkpoint.id)}/preference`,
      {
        method: "PUT",
        body: JSON.stringify({
          observer_agent_id: checkpoint.observer_agent_id,
          preference,
        }),
      },
    );
    await loadMotifs();
    const messages = {
      notice: "The agent may quietly notice this pattern when relevant.",
      follow: "The agent will keep this pattern in view when relevant.",
      test: "The agent will test this pattern's limits when relevant.",
      paused: "This pattern checkpoint is paused.",
    };
    showToast(messages[preference]);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function loadMotifs() {
  state.motifData = await api(`/api/motifs/${encodeURIComponent(state.currentProject)}`);
  if (
    state.selectedMotif
    && !(state.motifData.motifs || []).some((motif) => motif.id === state.selectedMotif)
  ) {
    state.selectedMotif = null;
    renderMotifDetail(null);
  }
  renderMotifTrajectories();
  renderMotifCheckpoints();
  renderMotifPatterns();
  renderMotifTags();
}

function getSources(annotations = [], snapshots = []) {
  const sources = [];
  const seen = new Set();
  for (const annotation of annotations) {
    const citation = annotation?.url_citation;
    const url = citation?.url;
    if (!url || seen.has(url) || !/^https?:\/\//i.test(url)) continue;
    seen.add(url);
    sources.push({ url, title: citation.title || url });
  }
  for (const snapshot of snapshots) {
    const url = snapshot?.final_url || snapshot?.requested_url;
    if (!url || seen.has(url) || !/^https?:\/\//i.test(url)) continue;
    seen.add(url);
    sources.push({ url, title: snapshot.title || url });
  }
  return sources;
}

async function openProjectFile(path) {
  activateTab("files");
  elements.filePreview.textContent = `Loading ${path}...`;
  try {
    if (/\.(?:svg|png|jpe?g|gif|webp)$/i.test(path)) {
      const figure = document.createElement("figure");
      figure.className = "graphic-preview";
      const image = document.createElement("img");
      image.alt = `Preview of ${path}`;
      image.src = `/api/files/${encodeURIComponent(state.currentProject)}/preview?path=${encodeURIComponent(path)}`;
      const caption = document.createElement("figcaption");
      caption.textContent = path;
      figure.append(image, caption);
      elements.filePreview.replaceChildren(figure);
      elements.filePreview.scrollTop = 0;
      return;
    }
    const data = await api(
      `/api/files/${encodeURIComponent(state.currentProject)}/read?path=${encodeURIComponent(path)}`,
    );
    const content = `${data.content}${data.truncated ? "\n\n> **Preview truncated.**" : ""}`;
    elements.filePreview.replaceChildren();
    if (/\.(?:md|markdown)$/i.test(path)) {
      elements.filePreview.append(renderMarkdown(content, { onProjectFile: openProjectFile }));
    } else if (/\.(?:json|ya?ml|csv|html|css|jsx?|tsx?|py|sh|toml|ini|xml|sql|c|h|cpp|hpp|java|go|rs)$/i.test(path)) {
      elements.filePreview.append(renderCodeViewer(content, path));
    } else {
      const plain = document.createElement("pre");
      plain.className = "plain-file-preview";
      plain.textContent = content;
      elements.filePreview.append(plain);
    }
    elements.filePreview.scrollTop = 0;
  } catch (error) {
    elements.filePreview.textContent = `Could not open ${path}.`;
    showToast(error.message, true);
  }
}

function fileToolActivity(toolEvents = []) {
  const activities = [];
  const seen = new Set();
  for (const event of toolEvents) {
    const result = event.result || {};
    const path = result.path || event.arguments?.path;
    if (!path || !["read_project_file", "write_project_file"].includes(event.tool)) continue;
    const key = `${event.tool}:${path}`;
    if (seen.has(key)) continue;
    seen.add(key);
    let action = "READ";
    if (event.tool === "write_project_file") action = result.overwritten ? "UPDATED" : "CREATED";
    activities.push({ action, path, ok: result.ok !== false, error: result.error || "" });
  }
  return activities;
}

function renderToolActivity(toolEvents = []) {
  const activities = fileToolActivity(toolEvents);
  const failures = [];
  const laterSuccessfulTools = new Set();
  for (let index = toolEvents.length - 1; index >= 0; index -= 1) {
    const event = toolEvents[index];
    if (event.result?.ok !== false) {
      laterSuccessfulTools.add(event.tool);
      continue;
    }
    const retryable = event.result?.retryable
      || /(?:unterminated string|invalid json|json.*(?:decode|delimiter))/i.test(event.result?.error || "");
    if (!retryable || !laterSuccessfulTools.has(event.tool)) failures.push(event);
  }
  failures.reverse();
  if (!activities.length && !failures.length) return null;

  const box = document.createElement("div");
  box.className = "tool-activity";
  const label = document.createElement("div");
  label.className = "tool-activity-label";
  label.textContent = activities.some((item) => ["CREATED", "UPDATED"].includes(item.action))
    ? "PROJECT FILE ACTIVITY"
    : "PROJECT FILES EXAMINED";
  box.append(label);

  for (const activity of activities) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `tool-file-link${activity.ok ? "" : " error"}`;
    button.textContent = `${activity.action} ↗ ${activity.path}`;
    button.addEventListener("click", () => openProjectFile(activity.path));
    box.append(button);
  }
  for (const failure of failures) {
    const error = document.createElement("div");
    error.className = "tool-event-error";
    const sizeLimited = failure.result?.reason === "agent_file_size_limit";
    const retryable = failure.result?.retryable
      || /(?:unterminated string|invalid json|json.*(?:decode|delimiter))/i.test(failure.result?.error || "");
    error.textContent = sizeLimited
      ? "The file exceeded its agent-owned size limit. The agent was asked to consolidate it; no oversized version was saved."
      : retryable
      ? `${failure.tool || "Tool"} could not prepare a valid request; no file was changed.`
      : `${failure.tool || "Tool"} failed: ${failure.result.error || "Unknown error"}`;
    box.append(error);
  }
  return box;
}

function renderMessage(message) {
  const article = document.createElement("article");
  const isUser = message.role === "user";
  const isRunner = message.role === "runner";
  const meta = agentMeta[message.agent_id] || { name: "Agent", className: "" };
  article.className = `message ${isUser ? "user" : isRunner ? "runner" : meta.className}`;

  const head = document.createElement("div");
  head.className = "message-head";
  const speaker = document.createElement("span");
  speaker.textContent = isUser
    ? userDisplayName().toUpperCase()
    : isRunner
      ? "ISOLATED RUNNER"
      : meta.name.toUpperCase();
  const time = document.createElement("span");
  const parsedDate = new Date(message.created_at);
  time.textContent = Number.isNaN(parsedDate.getTime()) ? "" : parsedDate.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  head.append(speaker, time);

  const body = document.createElement("div");
  body.className = "message-body markdown-body";
  body.append(renderMarkdown(message.content, { onProjectFile: openProjectFile }));
  article.append(head, body);

  const toolEvents = message.metadata?.tool_events || [];
  const toolActivity = renderToolActivity(toolEvents);
  if (toolActivity) article.append(toolActivity);

  const roleSignals = message.metadata?.role_signals || [];
  if (roleSignals.length) {
    const signalBox = document.createElement("div");
    signalBox.className = "demo-role-signals";
    for (const signal of roleSignals) {
      appendRoleSignal(signalBox, signal, "message-role-signal");
    }
    article.append(signalBox);
  }

  const sources = getSources(message.annotations, message.metadata?.web_sources || []);
  if (sources.length) {
    const sourceBox = document.createElement("div");
    sourceBox.className = "sources";
    for (const source of sources) {
      const link = document.createElement("a");
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = `↗ ${source.title}`;
      sourceBox.append(link);
    }
    article.append(sourceBox);
  }

  const sourceFailures = message.metadata?.web_source_failures || [];
  if (sourceFailures.length) {
    const failureBox = document.createElement("div");
    failureBox.className = "source-failures";
    const fallbackAgent = message.metadata?.search_fallback_agent;
    for (const failure of sourceFailures) {
      const row = document.createElement("div");
      const isSearchFailure = failure.retrieval_method === "agent_search";
      const attempts = failure.attempt_count
        ? ` AFTER ${failure.attempt_count} DIRECT ATTEMPT${failure.attempt_count === 1 ? "" : "S"}`
        : "";
      const fallback = fallbackAgent && !isSearchFailure
        ? ` · SEARCH FALLBACK: ${agentMeta[fallbackAgent]?.name?.toUpperCase() || fallbackAgent.toUpperCase()}`
        : "";
      const label = isSearchFailure
        ? "SEARCH FALLBACK FOUND NO CITED EVIDENCE"
        : "DIRECT RETRIEVAL FAILED";
      row.textContent = `${label}${attempts}: ${failure.url || "URL"} — ${failure.detail || "Could not load the page."}${fallback}`;
      failureBox.append(row);
    }
    article.append(failureBox);
  }

  const provenance = message.metadata?.research_provenance;
  if (provenance?.method === "agent_search") {
    const provenanceBox = document.createElement("div");
    provenanceBox.className = "source-provenance";
    const citationCount = provenance.citations?.length || 0;
    const provider = [provenance.provider, provenance.model].filter(Boolean).join(" / ");
    const via = provider ? ` · VIA ${provider}` : "";
    provenanceBox.textContent = `PROVENANCE: AGENT SEARCH AFTER RECOVERABLE DIRECT-READ FAILURE${via} · ${citationCount} CITATION${citationCount === 1 ? "" : "S"}`;
    article.append(provenanceBox);
  }

  return decorateMessageNode(article, message);
}

function beginTurnProgress() {
  state.progressNode?.remove();
  const panel = document.createElement("section");
  panel.className = "turn-progress";
  panel.setAttribute("aria-live", "polite");
  const heading = document.createElement("div");
  heading.className = "turn-progress-heading";
  heading.textContent = "ROOM PROGRESS";
  const events = document.createElement("div");
  events.className = "turn-progress-events";
  panel.append(heading, events);
  elements.messages.append(panel);
  state.progressNode = panel;
  return panel;
}

function messagesNearBottom() {
  const remaining = elements.messages.scrollHeight
    - elements.messages.scrollTop
    - elements.messages.clientHeight;
  return remaining < UI_DEFAULTS.conversationFollowThresholdPx;
}

function captureMessagesViewport() {
  return {
    follow: messagesNearBottom(),
    scrollTop: elements.messages.scrollTop,
  };
}

function restoreMessagesViewport(viewport, hasNewContent = true) {
  if (viewport.follow) {
    elements.messages.scrollTop = elements.messages.scrollHeight;
    elements.newReturns.classList.add("hidden");
    return;
  }
  elements.messages.scrollTop = viewport.scrollTop;
  if (hasNewContent) elements.newReturns.classList.remove("hidden");
}

function progressText(event) {
  const name = event.display_name || agentMeta[event.agent_id]?.name || "Agent";
  if (event.type === "source_fetch_start") return `Loading the supplied page: ${event.url}`;
  if (event.type === "source_fetch_complete") return `Loaded and stored: ${event.title || event.final_url}`;
  if (event.type === "source_fetch_cached") return `Reusing the recent project snapshot: ${event.title || event.final_url}`;
  if (event.type === "source_fetch_error") return `Could not load ${event.url}: ${event.detail}`;
  if (event.type === "source_search_fallback") {
    const count = event.urls?.length || 1;
    return `${name} will search for ${count === 1 ? "the blocked source" : `${count} blocked sources`}...`;
  }
  if (event.type === "source_search_no_evidence") {
    return `Search fallback found no cited evidence for ${event.url}.`;
  }
  if (event.type === "source_no_evidence") return event.detail;
  if (event.type === "agent_start") return `${name} is reading the room...`;
  if (event.type === "agent_followup_start") {
    const maxBeats = state.session?.max_agent_turn_beats
      || UI_DEFAULTS.agentTurnBeats;
    return `${name} is adding response beat ${event.turn_beat || 2} of ${maxBeats}...`;
  }
  if (event.type === "model_request") {
    const route = event.provider && event.model
      ? ` via ${event.provider.toUpperCase()} / ${event.model}`
      : "";
    return event.round > 1
      ? `${name} is continuing the response${route}...`
      : `${name} is thinking${route}...`;
  }
  if (event.type === "synthesizing") return `${name} is forming an answer from the project files...`;
  if (event.type === "participation_retry") return `${name} returned silence; requesting a visible turn...`;
  if (event.type === "agent_complete") {
    const maxBeats = state.session?.max_agent_turn_beats
      || UI_DEFAULTS.agentTurnBeats;
    return event.turn_beat > 1
      ? `${name} added response beat ${event.turn_beat} of ${maxBeats}.`
      : `${name} responded.`;
  }
  if (event.type === "agent_pass") return `${name} had nothing additive to add.`;
  if (event.type === "agent_empty") return `${name} used tools but returned no written answer.`;
  if (event.type === "agent_timeout") return `${name} timed out. Passing the turn to the next agent...`;
  if (event.type === "agent_no_response") return `${name} returned no response after two retries. The failed turn was recorded.`;
  if (event.type === "agent_provider_error") return `${name}'s provider failed. Recording the turn and continuing...`;
  if (event.type === "tool") {
    if (event.tool === "record_motif_observations") return "";
    const path = event.result?.path || event.arguments?.path;
    if (event.result?.reason === "agent_file_size_limit") {
      return `${name} reached the file limit and is reframing the file into a tighter version...`;
    }
    if (event.result?.retryable) return `${name} is repairing an invalid ${event.tool || "tool"} request...`;
    if (event.tool === "list_project_files") return `${name} listed the project files.`;
    if (event.tool === "list_project_sources") return `${name} listed the stored web sources.`;
    if (event.tool === "read_project_source") return `${name} read a stored web source.`;
    if (event.tool === "read_project_file") return `${name} read ${path || "a project file"}.`;
    if (event.tool === "search_project_files") return `${name} searched the project files.`;
    if (event.tool === "write_project_file") {
      return `${name} ${event.result?.overwritten ? "updated" : "created"} ${path || "a project file"}.`;
    }
    if (event.tool === "propose_persona_update") return `${name} recorded a persona update.`;
    return `${name} used ${event.tool || "a tool"}.`;
  }
  return "";
}

function updateTurnProgress(event) {
  if (["turn_start", "turn_complete", "result"].includes(event.type)) return;
  const viewport = captureMessagesViewport();
  const panel = state.progressNode || beginTurnProgress();
  if (
    event.type === "tool"
    && event.tool === "write_project_file"
    && event.result?.ok !== false
  ) {
    const changedPath = event.result?.path || event.arguments?.path;
    if (changedPath) void refreshHtmlDemoInPlace(changedPath);
  }
  if (event.type === "agent_complete" && event.message) {
    panel.before(renderMessage(event.message));
  }
  const text = progressText(event);
  if (!text) {
    restoreMessagesViewport(viewport, event.type === "agent_complete");
    return;
  }
  const row = document.createElement("div");
  row.className = `turn-progress-event ${["agent_empty", "agent_timeout", "agent_no_response", "agent_provider_error", "source_fetch_error", "source_search_no_evidence", "source_no_evidence"].includes(event.type) ? "error" : ""}`;
  row.textContent = text;
  panel.querySelector(".turn-progress-events").append(row);
  restoreMessagesViewport(viewport);
}

function renderAgentFailures(failures = []) {
  const viewport = captureMessagesViewport();
  for (const failure of failures) {
    const panel = document.createElement("section");
    panel.className = "agent-failure";
    const heading = document.createElement("strong");
    const noResponse = failure.kind === "no_response";
    const providerError = failure.kind === "provider_error";
    heading.textContent = `${failure.display_name || "An agent"} ${providerError ? "had a provider error" : noResponse ? "returned no response" : "timed out"}`;
    const detail = document.createElement("div");
    detail.textContent = providerError
      ? "The failed provider turn was stored in its memory-loop data, and the room continued to the next selected agent."
      : noResponse
        ? "Two participation retries produced neither speech nor a successful action. The failed turn was stored in its memory-loop data."
        : "The room continued to the next selected agent, and the timeout was stored in its memory-loop data.";
    const model = document.createElement("small");
    model.textContent = failure.model ? `MODEL: ${failure.model}` : "";
    panel.append(heading, detail, model);
    elements.messages.append(panel);
  }
  if (failures.length) restoreMessagesViewport(viewport);
}

function renderRequestError(message) {
  const viewport = captureMessagesViewport();
  const panel = document.createElement("section");
  panel.className = "request-error";
  const heading = document.createElement("strong");
  heading.textContent = "THE REQUEST STOPPED";
  const detail = document.createElement("div");
  detail.textContent = message;
  panel.append(heading, detail);
  elements.messages.append(panel);
  restoreMessagesViewport(viewport);
}

function renderMessages(messages) {
  const projectChanged = state.renderedProject !== state.currentProject;
  const viewport = projectChanged
    ? { follow: true, scrollTop: 0 }
    : captureMessagesViewport();
  if (!messages.length) {
    const empty = document.createElement("div");
    empty.className = "empty-room";
    const title = document.createElement("h2");
    title.textContent = "THE ROOM IS QUIET.";
    const copy = document.createElement("p");
    copy.textContent = "Speak to everyone, address one agent by name, ask for current research, or ask them to create a file in this project.";
    empty.append(title, copy);
    elements.messages.replaceChildren(empty);
    state.renderedProject = state.currentProject;
    elements.newReturns.classList.add("hidden");
    return;
  }
  reconcileMessageNodes(elements.messages, messages, renderMessage, {
    reuseExisting: !projectChanged,
  });
  state.renderedProject = state.currentProject;
  restoreMessagesViewport(viewport, !projectChanged);
}

async function loadMessages() {
  const messages = await api(`/api/projects/${encodeURIComponent(state.currentProject)}/messages`);
  renderMessages(messages);
}

async function loadFiles() {
  const files = await api(`/api/files/${encodeURIComponent(state.currentProject)}`);
  if (!files.length) {
    const empty = document.createElement("p");
    empty.className = "microcopy";
    empty.textContent = "No project files yet.";
    elements.fileList.replaceChildren(empty);
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const file of files) {
    const row = document.createElement("div");
    row.className = "file-row";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "file-open";
    const name = document.createElement("span");
    name.textContent = file.path;
    const kind = document.createElement("b");
    kind.className = `file-kind file-kind-${file.kind || "text"}`;
    kind.textContent = String(file.kind || "text").toUpperCase();
    const size = document.createElement("small");
    const owner = file.owner_type === "agent"
      ? agentMeta[file.owner_id]?.name || file.owner_id
      : userDisplayName();
    const agentLimit = state.session?.agent_file_max_bytes
      || UI_DEFAULTS.agentFileMaxBytes;
    size.textContent = file.owner_type === "agent"
      ? `${file.size_bytes} / ${agentLimit} B · ${owner}${file.shared_agent_edit ? " · SHARED EDITING" : " · PRIVATE"}`
      : `${file.size_bytes} B · ${owner}`;
    const fileIdentity = document.createElement("span");
    fileIdentity.className = "file-identity";
    fileIdentity.append(name, kind);
    button.append(fileIdentity, size);
    button.addEventListener("click", async () => {
      try {
        await openProjectFile(file.path);
      } catch (error) {
        showToast(error.message, true);
      }
    });
    const download = document.createElement("a");
    download.className = "file-download";
    download.textContent = "DOWNLOAD";
    download.href = `/api/files/${encodeURIComponent(state.currentProject)}/download?path=${encodeURIComponent(file.path)}`;
    download.setAttribute("download", file.path.split("/").at(-1) || "project-file");
    download.setAttribute("aria-label", `Download ${file.path}`);
    const actions = document.createElement("div");
    actions.className = "file-actions";
    if (file.owner_type === "agent") {
      const sharing = document.createElement("button");
      sharing.type = "button";
      sharing.className = `file-sharing${file.shared_agent_edit ? " active" : ""}`;
      sharing.textContent = file.shared_agent_edit ? "SHARED: ON" : "SHARED: OFF";
      sharing.setAttribute(
        "aria-label",
        `${file.shared_agent_edit ? "Disable" : "Allow"} shared agent editing for ${file.path}`,
      );
      sharing.addEventListener("click", async () => {
        try {
          await api(`/api/files/${encodeURIComponent(state.currentProject)}/sharing`, {
            method: "PUT",
            body: JSON.stringify({
              path: file.path,
              shared_agent_edit: !Boolean(file.shared_agent_edit),
            }),
          });
          await loadFiles();
          showToast(
            file.shared_agent_edit
              ? `${file.path} is private to its creator again.`
              : `${file.path} can now be edited by the other agents.`,
          );
        } catch (error) {
          showToast(error.message, true);
        }
      });
      actions.append(sharing);
    }
    if (/\.py$/i.test(file.path)) {
      const run = document.createElement("button");
      run.type = "button";
      run.className = "file-run";
      run.textContent = "RUN";
      run.setAttribute("aria-label", `Run ${file.path} in isolated container`);
      run.addEventListener("click", () => preparePythonDemo(file.path));
      actions.append(run);
    } else if (/\.html$/i.test(file.path)) {
      const demo = document.createElement("button");
      demo.type = "button";
      demo.className = "file-run";
      demo.textContent = "DEMO";
      demo.setAttribute("aria-label", `Open sandboxed demo for ${file.path}`);
      demo.addEventListener("click", () => openHtmlDemo(file.path));
      actions.append(demo);
    }
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "file-remove";
    remove.textContent = "REMOVE";
    remove.setAttribute("aria-label", `Remove ${file.path}`);
    remove.addEventListener("click", async () => {
      if (!window.confirm(`Remove ${file.path} from this project? This cannot be undone.`)) return;
      try {
        await api(
          `/api/files/${encodeURIComponent(state.currentProject)}?path=${encodeURIComponent(file.path)}`,
          { method: "DELETE" },
        );
        elements.filePreview.textContent = "Select a file.";
        await loadFiles();
        showToast(`${file.path} removed.`);
      } catch (error) {
        showToast(error.message, true);
      }
    });
    actions.append(download, remove);
    row.append(button, actions);
    fragment.append(row);
  }
  elements.fileList.replaceChildren(fragment);
}

async function openWebSource(sourceId) {
  activateTab("sources");
  elements.sourcePreview.textContent = "Loading extracted page text...";
  try {
    const source = await api(
      `/api/web-sources/${encodeURIComponent(state.currentProject)}/${encodeURIComponent(sourceId)}`,
    );
    elements.sourcePreview.replaceChildren();
    const title = document.createElement("strong");
    title.textContent = source.title || source.final_url;
    const link = document.createElement("a");
    link.href = source.final_url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = `OPEN ORIGINAL ↗ ${source.final_url}`;
    const details = document.createElement("small");
    details.textContent = `${source.content_type} · ${source.byte_count} BYTES · ${source.char_count} CHARACTERS · ${(source.retrieval_method || "direct_http").toUpperCase()} · ${source.retrieval_attempts || 1} ATTEMPT${source.retrieval_attempts === 1 ? "" : "S"}${source.truncated ? " · TRUNCATED" : ""}`;
    const content = document.createElement("pre");
    content.textContent = source.content_text || "No extracted text.";
    elements.sourcePreview.append(title, link, details, content);
    elements.sourcePreview.scrollTop = 0;
  } catch (error) {
    elements.sourcePreview.textContent = "Could not open this source snapshot.";
    showToast(error.message, true);
  }
}

async function loadSources() {
  const sources = await api(`/api/web-sources/${encodeURIComponent(state.currentProject)}`);
  if (!sources.length) {
    const empty = document.createElement("p");
    empty.className = "microcopy";
    empty.textContent = "No supplied web pages stored in this project yet.";
    elements.sourceList.replaceChildren(empty);
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const source of sources) {
    const row = document.createElement("div");
    row.className = "source-row";
    const open = document.createElement("button");
    open.type = "button";
    open.className = "source-open";
    const title = document.createElement("span");
    title.textContent = source.title || source.final_url;
    const url = document.createElement("small");
    url.textContent = `${source.final_url} · ${source.char_count} CHARS · ${(source.retrieval_method || "direct_http").toUpperCase()}${source.truncated ? " · TRUNCATED" : ""}`;
    open.append(title, url);
    open.addEventListener("click", () => openWebSource(source.id));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "source-remove";
    remove.textContent = "REMOVE";
    remove.setAttribute("aria-label", `Remove source ${source.title || source.final_url}`);
    remove.addEventListener("click", async () => {
      if (!window.confirm(`Remove the stored snapshot for ${source.title || source.final_url}?`)) return;
      try {
        await api(
          `/api/web-sources/${encodeURIComponent(state.currentProject)}/${encodeURIComponent(source.id)}`,
          { method: "DELETE" },
        );
        elements.sourcePreview.textContent = "Select a source.";
        await loadSources();
        showToast("Source snapshot removed.");
      } catch (error) {
        showToast(error.message, true);
      }
    });
    row.append(open, remove);
    fragment.append(row);
  }
  elements.sourceList.replaceChildren(fragment);
}

async function refreshChangedTurnPanels(refreshState) {
  const loaders = {
    messages: loadMessages,
    memory: loadMemoryLoop,
    recovery: loadTurnRecovery,
    motifs: loadMotifs,
    files: loadFiles,
    sources: loadSources,
    proposals: loadProposals,
  };
  await Promise.all(turnRefreshTargets(refreshState).map((target) => loaders[target]()));
}

function formatTurnUsage(turn) {
  const usage = turn.provider_usage || {};
  const tokenCount = usage.total_tokens;
  const duration = Number.isFinite(turn.duration_ms)
    ? `${(turn.duration_ms / 1000).toFixed(1)}S`
    : "—";
  return `${duration} · ${turn.provider_requests || 0} PROVIDER REQUESTS`
    + `${Number.isFinite(tokenCount) ? ` · ${tokenCount} TOKENS` : ""}`;
}

async function resumeTurn(turn) {
  if (state.busy) {
    showToast("The current room turn must finish before resuming another.", true);
    return;
  }
  setBusy(true);
  beginTurnProgress();
  const refreshState = createTurnRefreshState();
  try {
    const result = await streamApi(
      `/api/chat-turns/${encodeURIComponent(state.currentProject)}/${encodeURIComponent(turn.id)}/resume/stream`,
      { method: "POST" },
      (event) => {
        observeTurnRefreshEvent(refreshState, event);
        updateTurnProgress(event);
      },
    );
    if (!result) throw new Error("The server closed before returning a resumed result.");
    observeTurnRefreshResult(refreshState, result);
    updateResearchIndicator(result);
    await refreshChangedTurnPanels(refreshState);
    renderAgentFailures(result.agent_failures || []);
    showToast("Turn resumed from its stored progress.");
  } catch (error) {
    showToast(error.message, true);
    await Promise.all([loadMessages(), loadMotifs(), loadTurnRecovery()]);
  } finally {
    state.progressNode?.remove();
    state.progressNode = null;
    setBusy(false);
  }
}

async function acceptPartialTurn(turn) {
  if (state.busy) {
    showToast("The current room turn must finish before accepting another.", true);
    return;
  }
  try {
    await api(
      `/api/chat-turns/${encodeURIComponent(state.currentProject)}/${encodeURIComponent(turn.id)}/accept`,
      { method: "POST" },
    );
    await loadTurnRecovery();
    showToast("Partial turn accepted.");
  } catch (error) {
    showToast(error.message, true);
  }
}

async function loadTurnRecovery() {
  const turns = await api(
    `/api/chat-turns/${encodeURIComponent(state.currentProject)}?resumable_only=true`,
  );
  if (!turns.length) {
    elements.turnRecovery.replaceChildren();
    elements.turnRecovery.classList.add("hidden");
    return;
  }

  const heading = document.createElement("div");
  heading.className = "turn-recovery-head";
  const title = document.createElement("strong");
  title.textContent = "ROOM TURN NEEDS ATTENTION";
  const count = document.createElement("span");
  count.textContent = `${turns.length} RECOVERABLE`;
  heading.append(title, count);

  const list = document.createElement("div");
  list.className = "turn-recovery-list";
  const fragment = document.createDocumentFragment();
  for (const turn of turns) {
    const row = document.createElement("article");
    row.className = "turn-recovery-row";
    const head = document.createElement("div");
    head.className = "turn-recovery-row-head";
    const status = document.createElement("strong");
    status.textContent = (turn.resolution || turn.status).replaceAll("_", " ").toUpperCase();
    const date = document.createElement("small");
    date.textContent = new Date(turn.started_at).toLocaleString();
    head.append(status, date);
    const message = document.createElement("p");
    message.textContent = turn.message || "Older turn without a stored request.";
    row.append(head, message);
    if (turn.execution_stage && turn.status !== "completed") {
      const stage = document.createElement("small");
      const agentName = agentMeta[turn.execution_stage.agent_id]?.name
        || turn.execution_stage.agent_id;
      const operationLabels = {
        provider_completion: "MODEL RESPONSE",
        message_committed: "MESSAGE SAVE",
        memory_committed: "MEMORY SAVE",
        beat_finished: "RESPONSE BEAT",
        agent_finished: "AGENT TURN",
      };
      const operation = turn.execution_stage.operation.startsWith("tool:")
        ? "PROJECT ACTION"
        : operationLabels[turn.execution_stage.operation]
          || turn.execution_stage.operation.replaceAll("_", " ").toUpperCase();
      stage.textContent = `${agentName} · BEAT ${turn.execution_stage.turn_beat} · ${operation} · ${turn.execution_stage.status}`;
      row.append(stage);
    }
    if (turn.failure_detail) {
      const failure = document.createElement("small");
      failure.textContent = turn.failure_detail;
      row.append(failure);
    }
    const details = document.createElement("details");
    const detailsLabel = document.createElement("summary");
    detailsLabel.textContent = "TECHNICAL DETAILS";
    const metrics = document.createElement("small");
    metrics.textContent = formatTurnUsage(turn);
    details.append(detailsLabel, metrics);
    row.append(details);

    const actions = document.createElement("div");
    actions.className = "turn-recovery-actions";
    const resume = document.createElement("button");
    resume.type = "button";
    resume.textContent = "RESUME";
    resume.disabled = state.busy;
    resume.addEventListener("click", () => resumeTurn(turn));
    const accept = document.createElement("button");
    accept.type = "button";
    accept.className = "secondary";
    accept.textContent = "ACCEPT PARTIAL";
    accept.disabled = state.busy;
    accept.addEventListener("click", () => acceptPartialTurn(turn));
    actions.append(resume, accept);
    row.append(actions);
    fragment.append(row);
  }
  list.append(fragment);
  elements.turnRecovery.replaceChildren(heading, list);
  elements.turnRecovery.classList.remove("hidden");
}

async function loadPersona() {
  const data = await api(`/api/personas/${state.currentPersona}`);
  elements.personaEditor.value = data.yaml_text;
  const summary = state.agents.find((item) => item.agent_id === state.currentPersona);
  elements.personaSummary.textContent = summary
    ? `${summary.core_motif_symbol || ""} ${summary.core_motif_name || "CORE MOTIF"} — ${summary.core_motif_statement || ""}

${summary.archetype} — ${summary.systems_orientation}. ${summary.summary}`
    : "";
}

async function loadMemoryLoop() {
  const data = await api(
    `/api/memory-loops/${encodeURIComponent(state.currentProject)}/${encodeURIComponent(state.currentPersona)}`,
  );
  const definition = data.definition || {};
  const stats = data.stats || {};
  elements.memoryLoopSummary.replaceChildren();
  const title = document.createElement("strong");
  title.textContent = `${definition.symbol || "↻"} ${definition.name || "Memory Loop"}`;
  const stages = document.createElement("div");
  stages.className = "memory-loop-stages";
  stages.textContent = (definition.stages || []).join(" → ");
  const metrics = document.createElement("div");
  metrics.className = "memory-loop-metrics";
  metrics.textContent = `${stats.event_count || 0} RETURNS · ${stats.action_count || 0} WITH ACTIONS · ${stats.failure_count || 0} FAILED`;
  const observes = document.createElement("ul");
  for (const observation of definition.observes || []) {
    const item = document.createElement("li");
    item.textContent = observation;
    observes.append(item);
  }
  elements.memoryLoopSummary.append(title, stages, metrics, observes);

  if (!data.events?.length) {
    const empty = document.createElement("p");
    empty.className = "microcopy";
    empty.textContent = "No returns stored for this agent in this project yet.";
    elements.memoryLoopEvents.replaceChildren(empty);
  } else {
    const fragment = document.createDocumentFragment();
    for (const event of data.events) {
      const entry = document.createElement("details");
      entry.className = "memory-loop-event";
      const heading = document.createElement("summary");
      heading.textContent = `CYCLE ${event.sequence} · ${String(event.outcome || "return").replaceAll("_", " ").toUpperCase()}`;
      const trigger = document.createElement("p");
      const triggerLabel = document.createElement("strong");
      triggerLabel.textContent = "TRIGGER: ";
      trigger.append(triggerLabel, document.createTextNode(event.trigger_text || ""));
      const returned = document.createElement("p");
      const returnLabel = document.createElement("strong");
      returnLabel.textContent = "RETURN: ";
      returned.append(returnLabel, document.createTextNode(event.return_text || ""));
      entry.append(heading, trigger, returned);
      if (event.actions?.length) {
        const actions = document.createElement("p");
        actions.textContent = `ACTIONS: ${event.actions.map((action) => `${action.tool}${action.path ? ` (${action.path})` : ""}`).join(" · ")}`;
        entry.append(actions);
      }
      fragment.append(entry);
    }
    elements.memoryLoopEvents.replaceChildren(fragment);
  }

  const globalStats = data.global_stats || {};
  elements.globalMemorySummary.textContent = `${globalStats.event_count || 0} COMPACT RETURNS · ${globalStats.project_count || 0} SOURCE PROJECTS`;
  if (!data.global_events?.length) {
    const empty = document.createElement("p");
    empty.className = "microcopy";
    empty.textContent = "No cross-project continuity returns stored yet.";
    elements.globalMemoryEvents.replaceChildren(empty);
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const event of data.global_events) {
    const entry = document.createElement("details");
    entry.className = "memory-loop-event global-memory-event";
    const heading = document.createElement("summary");
    heading.textContent = `RETURN ${event.sequence} · ${event.source_project_name || event.source_project_id || "UNKNOWN PROJECT"}`;
    const trigger = document.createElement("p");
    const triggerLabel = document.createElement("strong");
    triggerLabel.textContent = "TRIGGER SUMMARY: ";
    trigger.append(triggerLabel, document.createTextNode(event.trigger_summary || ""));
    const returned = document.createElement("p");
    const returnLabel = document.createElement("strong");
    returnLabel.textContent = "RETURN SUMMARY: ";
    returned.append(returnLabel, document.createTextNode(event.return_summary || ""));
    entry.append(heading, trigger, returned);
    if (event.actions?.length) {
      const actions = document.createElement("p");
      actions.textContent = `ACTIONS: ${event.actions.map((action) => `${action.tool}${action.path ? ` (${action.path})` : ""}`).join(" · ")}`;
      entry.append(actions);
    }
    fragment.append(entry);
  }
  elements.globalMemoryEvents.replaceChildren(fragment);
}

async function loadSharedContext() {
  const data = await api("/api/shared-context");
  elements.sharedContextEditor.value = data.markdown_text || "";
}

async function loadProposals() {
  const proposals = await api("/api/proposals");
  elements.proposalCount.textContent = String(proposals.length);
  elements.proposalList.replaceChildren();
  for (const proposal of proposals.slice(0, UI_DEFAULTS.proposalListMaxItems)) {
    const item = document.createElement("div");
    item.className = "proposal";
    item.textContent = `${proposal.agent_id}: ${proposal.reason} (${proposal.changes?.length || 0} proposed changes)`;
    elements.proposalList.append(item);
  }
}

function providerProfiles(sessionData = state.session) {
  return Array.isArray(sessionData?.provider_catalog)
    ? sessionData.provider_catalog
    : [];
}

function populateProviderSelect(select, selectedProvider, profiles) {
  select.replaceChildren();
  if (!selectedProvider) {
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "SELECT A PROVIDER";
    placeholder.disabled = true;
    placeholder.selected = true;
    select.append(placeholder);
  }
  for (const profile of profiles) {
    const option = document.createElement("option");
    option.value = profile.id;
    option.textContent = profile.label || profile.id.toUpperCase();
    select.append(option);
  }
  if (selectedProvider && !profiles.some((profile) => profile.id === selectedProvider)) {
    const unavailable = document.createElement("option");
    unavailable.value = selectedProvider;
    unavailable.textContent = `${selectedProvider.toUpperCase()} (NOT ENABLED)`;
    select.append(unavailable);
  }
  select.value = selectedProvider || "";
}

function populateModelSuggestions(providerSelect, datalist, profiles) {
  datalist.replaceChildren();
  const profile = profiles.find((item) => item.id === providerSelect.value);
  for (const model of profile?.models || []) {
    const option = document.createElement("option");
    option.value = model;
    datalist.append(option);
  }
}

function renderProviderControls(sessionData) {
  const profiles = providerProfiles(sessionData);
  const runtimeProviders = sessionData.runtime?.providers || {};
  const controls = [
    [elements.providerA, elements.modelOptionsA, runtimeProviders.agent_a],
    [elements.providerB, elements.modelOptionsB, runtimeProviders.agent_b],
    [elements.providerC, elements.modelOptionsC, runtimeProviders.agent_c],
  ];
  for (const [providerSelect, modelOptions, selectedProvider] of controls) {
    populateProviderSelect(providerSelect, selectedProvider, profiles);
    populateModelSuggestions(providerSelect, modelOptions, profiles);
  }
}

function renderProviderStatus(sessionData) {
  const profiles = providerProfiles(sessionData);
  const statuses = sessionData.provider_status || {};
  elements.keyStatus.textContent = profiles
    .map((profile) => {
      if (statuses[profile.id]) {
        return `${String(profile.label || profile.id).toUpperCase()}: READY`;
      }
      return profile.api_key_required
        ? `${String(profile.label || profile.id).toUpperCase()}: NO KEY`
        : `${String(profile.label || profile.id).toUpperCase()}: UNAVAILABLE`;
    })
    .join(" · ") || "NO PROVIDERS ENABLED";
  elements.keyStatus.className = `key-status ${sessionData.key_configured ? "ok" : "error"}`;
}

function fillSetup(sessionData) {
  const runtime = sessionData.runtime || {
    ...sessionData.runtime_defaults,
    providers: {},
    models: {},
  };
  renderProviderControls(sessionData);
  elements.modelA.value = runtime.models.agent_a || "";
  elements.modelB.value = runtime.models.agent_b || "";
  elements.modelC.value = runtime.models.agent_c || "";
  elements.defaultResearchMode.value = runtime.default_research_mode || "auto";
  elements.researchMode.value = runtime.default_research_mode || "auto";
  elements.temperature.value = runtime.temperature;
  elements.maxTokens.value = runtime.max_tokens;
  elements.demoArguments.maxLength = Number(
    sessionData.runner_arguments_max_bytes
      || elements.demoArguments.maxLength,
  );
  elements.demoStdin.maxLength = Number(
    sessionData.runner_input_message_max_bytes
      || elements.demoStdin.maxLength,
  );
  renderProviderStatus(sessionData);
  elements.setupWarning.classList.toggle("hidden", !setupIncomplete(sessionData));
}

async function loadProviderCatalog() {
  const catalog = await api("/api/provider-catalog");
  elements.providerCatalogEditor.value = catalog.yaml_text || "";
  return catalog;
}

async function refreshSession() {
  const session = await api("/api/session");
  setSessionToken(session.token);
  state.session = session;
  state.projects = session.projects;
  state.agents = session.agents;
  if (!state.projects.some((project) => project.id === state.currentProject)) {
    state.currentProject = state.projects[0]?.id || "general";
  }
  renderProjects();
  renderAgentOptions();
  fillSetup(session);
  setBusy(state.busy);
}

async function initialize() {
  try {
    loadPanelLayout();
    setStatus("CONNECTING");
    await refreshSession();
    await Promise.all([
      loadMessages(),
      loadPersona(),
      loadMemoryLoop(),
      loadMotifs(),
      loadSharedContext(),
      loadFiles(),
      loadSources(),
      loadProposals(),
      loadProviderCatalog(),
      loadTurnRecovery(),
    ]);
    setStatus("READY", "ok");
  } catch (error) {
    setStatus("STARTUP ERROR", "error");
    showToast(error.message, true);
  }
}

elements.toggleLeftPanel.addEventListener("click", () => {
  state.leftCollapsed = !state.leftCollapsed;
  applyPanelLayout();
  savePanelLayout();
});

elements.toggleRightPanel.addEventListener("click", () => {
  state.rightCollapsed = !state.rightCollapsed;
  applyPanelLayout();
  savePanelLayout();
});

function activateTab(name) {
  $$(".tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === name));
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${name}`));
  document.querySelector(`.tab[data-tab="${name}"]`)?.scrollIntoView({
    block: "nearest",
    inline: "nearest",
  });
}

$$(".tab").forEach((button) => button.addEventListener("click", async () => {
  activateTab(button.dataset.tab);
  try {
    if (button.dataset.tab === "setup") await refreshSession();
    if (button.dataset.tab === "motifs") await loadMotifs();
  } catch (error) {
    showToast(`Could not refresh ${button.dataset.tab}: ${error.message}`, true);
  }
}));

elements.motifAgentFilter.addEventListener("change", () => {
  renderMotifTrajectories();
  renderMotifCheckpoints();
  renderMotifPatterns();
  renderMotifTags();
});
elements.motifStatusFilter.addEventListener("change", renderMotifTags);

elements.projectSelect.addEventListener("change", async () => {
  state.currentProject = elements.projectSelect.value;
  renderProjects();
  elements.filePreview.textContent = "Select a file.";
  elements.sourcePreview.textContent = "Select a source.";
  try {
    await Promise.all([
      loadMessages(),
      loadMemoryLoop(),
      loadMotifs(),
      loadFiles(),
      loadSources(),
      loadTurnRecovery(),
    ]);
    if (window.matchMedia(UI_DEFAULTS.narrowViewportQuery).matches) {
      document.querySelector(".room")?.scrollIntoView({ block: "start" });
    }
    elements.messageInput.focus({ preventScroll: true });
  } catch (error) {
    showToast(error.message, true);
  }
});

elements.downloadLog.addEventListener("click", () => {
  const project = state.projects.find((item) => item.id === state.currentProject);
  if (!project) {
    showToast("Choose a project before downloading its conversation.", true);
    return;
  }
  const download = document.createElement("a");
  download.href = (
    `/api/projects/${encodeURIComponent(project.id)}/conversation.md`
  );
  download.download = `motif-feedback-${project.id}-conversation.md`;
  document.body.append(download);
  download.click();
  download.remove();
  showToast("Conversation log download started.");
});

elements.createProject.addEventListener("click", async () => {
  const name = elements.newProjectName.value.trim();
  if (!name) return;
  try {
    const project = await api("/api/projects", { method: "POST", body: JSON.stringify({ name }) });
    state.projects.unshift(project);
    state.currentProject = project.id;
    elements.newProjectName.value = "";
    renderProjects();
    await Promise.all([
      loadMessages(),
      loadMemoryLoop(),
      loadMotifs(),
      loadFiles(),
      loadSources(),
      loadTurnRecovery(),
    ]);
    showToast("Project created.");
  } catch (error) {
    showToast(error.message, true);
  }
});

elements.deleteProject.addEventListener("click", async () => {
  if (state.busy || state.promptQueue.length) return;
  const project = state.projects.find((item) => item.id === state.currentProject);
  if (!project) return;
  const confirmed = window.confirm(
    `Permanently delete "${project.name}"?\n\nThis removes its conversations, files, sources, local memory, and cross-project memory references. This cannot be undone.`
  );
  if (!confirmed) return;

  elements.deleteProject.disabled = true;
  try {
    const result = await api(`/api/projects/${encodeURIComponent(project.id)}`, {
      method: "DELETE",
    });
    state.projects = result.projects || [];
    state.currentProject = state.projects[0]?.id || "";
    state.activeDemo = null;
    elements.demoOverlay.classList.add("hidden");
    elements.demoFrame.srcdoc = "";
    elements.demoOutput.textContent = "";
    elements.filePreview.textContent = "Select a file.";
    elements.sourcePreview.textContent = "Select a source.";
    renderProjects();
    await Promise.all([
      loadMessages(),
      loadMemoryLoop(),
      loadMotifs(),
      loadFiles(),
      loadSources(),
      loadProposals(),
      loadTurnRecovery(),
    ]);
    showToast(`Deleted ${project.name} and all of its stored project data.`);
    elements.messageInput.focus({ preventScroll: true });
  } catch (error) {
    showToast(error.message, true);
  } finally {
    elements.deleteProject.disabled = (
      state.busy
      || state.promptQueue.length > 0
      || !state.currentProject
    );
  }
});

function appendOptimisticPrompt(turn) {
  if (state.currentProject !== turn.projectId) return;
  const optimistic = {
    role: "user",
    content: turn.message,
    created_at: turn.queuedAt,
    annotations: [],
    metadata: {},
  };
  const currentNodes = [...elements.messages.querySelectorAll(".message")];
  if (!currentNodes.length) elements.messages.replaceChildren();
  const optimisticViewport = captureMessagesViewport();
  elements.messages.append(renderMessage(optimistic));
  restoreMessagesViewport(optimisticViewport);
}

function updateResearchIndicator(result) {
  const research = result.research;
  const sourceCount = result.web_sources?.length || 0;
  elements.researchIndicator.textContent = research.evidence_status === "unavailable"
    ? "NO WEB EVIDENCE · AGENTS SKIPPED"
    : sourceCount
    ? `WEB: ${sourceCount} SHARED SOURCE${sourceCount === 1 ? "" : "S"}`
    : research.search_fallback_agent
      ? `SEARCH FALLBACK / ${agentMeta[research.search_fallback_agent]?.name?.toUpperCase() || research.search_fallback_agent.toUpperCase()}`
    : research.needs_search
      ? `SEARCH DISCOVERY UNAVAILABLE${research.lead_agent ? ` / ${agentMeta[research.lead_agent].name.toUpperCase()}` : ""}`
      : "NO SEARCH";
}

async function executeQueuedPrompt(turn) {
  appendOptimisticPrompt(turn);
  const refreshState = createTurnRefreshState();
  try {
    if (state.currentProject === turn.projectId) beginTurnProgress();
    const result = await streamApi("/api/chat/stream", {
      method: "POST",
      body: JSON.stringify({
        turn_id: turn.turnId,
        project_id: turn.projectId,
        message: turn.message,
        participants: turn.participants,
        research_mode: turn.researchMode,
      }),
    }, (event) => {
      observeTurnRefreshEvent(refreshState, event);
      if (state.currentProject === turn.projectId) updateTurnProgress(event);
    });
    if (!result) throw new Error("The server closed the response before returning a result.");
    if (state.currentProject === turn.projectId) {
      observeTurnRefreshResult(refreshState, result);
      updateResearchIndicator(result);
      await refreshChangedTurnPanels(refreshState);
      renderAgentFailures(result.agent_failures || []);
    } else {
      showToast(`Queued turn completed in ${turn.projectName}.`);
    }
  } catch (error) {
    showToast(error.message, true);
    if (state.currentProject === turn.projectId) {
      await Promise.all([loadMessages(), loadMotifs(), loadTurnRecovery()]);
      renderRequestError(error.message);
    }
  } finally {
    state.progressNode?.remove();
    state.progressNode = null;
  }
}

async function drainPromptQueue() {
  if (state.busy) return;
  const turn = state.promptQueue.shift();
  if (!turn) {
    setBusy(false);
    renderPromptQueue();
    return;
  }
  state.activePrompt = turn;
  setBusy(true);
  renderPromptQueue();
  try {
    await executeQueuedPrompt(turn);
  } catch (error) {
    showToast(`Queued turn could not finish: ${error.message}`, true);
  } finally {
    state.activePrompt = null;
    state.busy = false;
    renderPromptQueue();
    if (state.promptQueue.length) {
      void drainPromptQueue();
    } else {
      setBusy(false);
    }
    elements.messageInput.focus({ preventScroll: true });
  }
}

elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = elements.messageInput.value.trim();
  if (!message) return;
  const participants = currentParticipants();
  if (!participants.length) {
    showToast("Select at least one agent.", true);
    return;
  }
  const project = state.projects.find((item) => item.id === state.currentProject);
  if (!project) {
    showToast("Choose a project before adding a prompt.", true);
    return;
  }

  try {
    const turnId = window.crypto?.randomUUID?.()
      || `turn-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    state.promptQueue.enqueue({
      turnId,
      projectId: project.id,
      projectName: project.name,
      message,
      participants,
      researchMode: elements.researchMode.value,
      queuedAt: new Date().toISOString(),
    });
  } catch (error) {
    showToast(error.message, true);
    return;
  }
  elements.messageInput.value = "";
  renderPromptQueue();
  setBusy(state.busy);
  if (state.busy) {
    showToast(`Prompt queued. ${state.promptQueue.length} waiting.`);
  }
  void drainPromptQueue();
});

elements.promptQueue.addEventListener("click", (event) => {
  const remove = event.target.closest("[data-queued-turn-id]");
  if (!remove) return;
  if (state.promptQueue.remove(remove.dataset.queuedTurnId)) {
    renderPromptQueue();
    setBusy(state.busy);
  }
});

elements.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});

elements.personaSelect.addEventListener("change", async () => {
  state.currentPersona = elements.personaSelect.value;
  try {
    await Promise.all([loadPersona(), loadMemoryLoop()]);
  } catch (error) {
    showToast(error.message, true);
  }
});

elements.reloadPersona.addEventListener("click", async () => {
  try {
    await loadPersona();
    showToast("Persona reloaded.");
  } catch (error) {
    showToast(error.message, true);
  }
});

elements.savePersona.addEventListener("click", async () => {
  try {
    await api(`/api/personas/${state.currentPersona}`, {
      method: "PUT",
      body: JSON.stringify({ yaml_text: elements.personaEditor.value }),
    });
    await refreshSession();
    await loadPersona();
    showToast("Persona saved with a versioned snapshot.");
  } catch (error) {
    showToast(error.message, true);
  }
});

elements.reloadSharedContext.addEventListener("click", async () => {
  try {
    await loadSharedContext();
    showToast("Shared context reloaded.");
  } catch (error) {
    showToast(error.message, true);
  }
});

elements.saveSharedContext.addEventListener("click", async () => {
  try {
    await api("/api/shared-context", {
      method: "PUT",
      body: JSON.stringify({ markdown_text: elements.sharedContextEditor.value }),
    });
    await loadSharedContext();
    showToast("Shared context saved with a versioned snapshot.");
  } catch (error) {
    showToast(error.message, true);
  }
});

[
  [elements.providerA, elements.modelOptionsA],
  [elements.providerB, elements.modelOptionsB],
  [elements.providerC, elements.modelOptionsC],
].forEach(([providerSelect, modelOptions]) => {
  providerSelect.addEventListener("change", () => {
    populateModelSuggestions(providerSelect, modelOptions, providerProfiles());
  });
});

elements.reloadProviderCatalog.addEventListener("click", async () => {
  try {
    await loadProviderCatalog();
    showToast("Provider catalog reloaded.");
  } catch (error) {
    showToast(error.message, true);
  }
});

elements.saveProviderCatalog.addEventListener("click", async () => {
  try {
    await api("/api/provider-catalog", {
      method: "PUT",
      body: JSON.stringify({ yaml_text: elements.providerCatalogEditor.value }),
    });
    await refreshSession();
    await loadProviderCatalog();
    showToast("Provider catalog saved.");
  } catch (error) {
    showToast(error.message, true);
  }
});

elements.setupForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const payload = {
      providers: {
        agent_a: elements.providerA.value,
        agent_b: elements.providerB.value,
        agent_c: elements.providerC.value,
      },
      models: {
        agent_a: elements.modelA.value.trim(),
        agent_b: elements.modelB.value.trim(),
        agent_c: elements.modelC.value.trim(),
      },
      default_research_mode: elements.defaultResearchMode.value,
      room_default_participants: ["agent_a", "agent_b", "agent_c"],
      temperature: Number(elements.temperature.value),
      max_tokens: Number(elements.maxTokens.value),
    };
    await api("/api/setup", { method: "PUT", body: JSON.stringify(payload) });
    await refreshSession();
    showToast("Setup saved.");
  } catch (error) {
    showToast(error.message, true);
  }
});

elements.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = elements.fileInput.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  try {
    const uploaded = await api(`/api/files/${encodeURIComponent(state.currentProject)}/upload`, { method: "POST", body: form });
    elements.fileInput.value = "";
    await loadFiles();
    await openProjectFile(uploaded.path);
    showToast("File uploaded into the current project folder.");
  } catch (error) {
    showToast(error.message, true);
  }
});

elements.demoCollapse.addEventListener("click", () => {
  const collapsed = elements.demoOverlay.classList.toggle("collapsed");
  elements.demoCollapse.textContent = collapsed ? "EXPAND" : "COLLAPSE";
  elements.demoCollapse.setAttribute("aria-expanded", String(!collapsed));
});

elements.demoClose.addEventListener("click", () => {
  cancelLiveRun();
  elements.demoOverlay.classList.add("hidden");
  elements.demoFrame.srcdoc = "";
  elements.demoOutput.textContent = "";
  showRoleSignals([]);
  state.activeDemo = null;
});

elements.demoRefresh.addEventListener("click", refreshActiveDemo);
elements.demoStart.addEventListener("click", () => {
  const demo = state.activeDemo;
  if (demo?.type === "python") runPythonDemo(demo.path);
});
elements.demoCancel.addEventListener("click", () => {
  if (!state.activeRunController) return;
  cancelLiveRun();
  elements.demoCancel.disabled = true;
});
elements.demoSendInput.addEventListener("click", sendLiveRunInput);
elements.demoSendEof.addEventListener("click", closeLiveRunInput);
elements.demoStdin.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey || elements.demoSendInput.disabled) return;
  event.preventDefault();
  sendLiveRunInput();
});

elements.newReturns.addEventListener("click", () => {
  elements.messages.scrollTop = elements.messages.scrollHeight;
  elements.newReturns.classList.add("hidden");
});

elements.messages.addEventListener("scroll", () => {
  if (messagesNearBottom()) elements.newReturns.classList.add("hidden");
});

initialize();
