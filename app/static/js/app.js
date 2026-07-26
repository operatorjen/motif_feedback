import { api, setSessionToken, streamApi } from "./api.js";
import { renderCodeViewer } from "./code_viewer.js";
import { appendRoleSignal, createDemoController } from "./demo_controller.js";
import { renderMarkdown } from "./markdown.js";

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
  researchMode: $("#research-mode"),
  researchIndicator: $("#research-indicator"),
  setupWarning: $("#setup-warning"),
  messages: $("#messages"),
  composer: $("#composer"),
  messageInput: $("#message-input"),
  sendButton: $("#send-button"),
  personaSelect: $("#persona-select"),
  personaSummary: $("#persona-summary"),
  memoryLoopSummary: $("#memory-loop-summary"),
  memoryLoopEvents: $("#memory-loop-events"),
  globalMemorySummary: $("#global-memory-summary"),
  globalMemoryEvents: $("#global-memory-events"),
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
  elements.sendButton.disabled = busy || !setupComplete;
  elements.messageInput.disabled = busy || !setupComplete;
  elements.deleteProject.disabled = busy || !state.currentProject;
  elements.sendButton.textContent = busy ? "THINKING..." : "SEND ↵";
}

function currentParticipants() {
  return $$('.agent-toggle input[type="checkbox"]:checked').map((input) => input.value);
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
  elements.deleteProject.disabled = state.busy || !current;
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
  const failures = toolEvents.filter((event, index) => {
    if (event.result?.ok !== false) return false;
    const retryable = event.result?.retryable
      || /(?:unterminated string|invalid json|json.*(?:decode|delimiter))/i.test(event.result?.error || "");
    if (!retryable) return true;
    return !toolEvents.slice(index + 1).some(
      (later) => later.tool === event.tool && later.result?.ok !== false,
    );
  });
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
    for (const failure of sourceFailures) {
      const row = document.createElement("div");
      row.textContent = `SOURCE BLOCKED: ${failure.url || "URL"} — ${failure.detail || "Could not load the page."}`;
      failureBox.append(row);
    }
    article.append(failureBox);
  }

  return article;
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
  row.className = `turn-progress-event ${["agent_empty", "agent_timeout", "agent_no_response", "agent_provider_error", "source_fetch_error"].includes(event.type) ? "error" : ""}`;
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
  elements.messages.replaceChildren();
  if (!messages.length) {
    const empty = document.createElement("div");
    empty.className = "empty-room";
    const title = document.createElement("h2");
    title.textContent = "THE ROOM IS QUIET.";
    const copy = document.createElement("p");
    copy.textContent = "Speak to everyone, address one agent by name, ask for current research, or ask them to create a file in this project.";
    empty.append(title, copy);
    elements.messages.append(empty);
    state.renderedProject = state.currentProject;
    elements.newReturns.classList.add("hidden");
    return;
  }
  for (const message of messages) elements.messages.append(renderMessage(message));
  state.renderedProject = state.currentProject;
  restoreMessagesViewport(viewport, !projectChanged);
}

async function loadMessages() {
  const messages = await api(`/api/projects/${encodeURIComponent(state.currentProject)}/messages`);
  renderMessages(messages);
}

async function loadFiles() {
  const files = await api(`/api/files/${encodeURIComponent(state.currentProject)}`);
  elements.fileList.replaceChildren();
  if (!files.length) {
    const empty = document.createElement("p");
    empty.className = "microcopy";
    empty.textContent = "No project files yet.";
    elements.fileList.append(empty);
    return;
  }
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
    elements.fileList.append(row);
  }
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
    details.textContent = `${source.content_type} · ${source.byte_count} BYTES · ${source.char_count} CHARACTERS${source.truncated ? " · TRUNCATED" : ""}`;
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
  elements.sourceList.replaceChildren();
  if (!sources.length) {
    const empty = document.createElement("p");
    empty.className = "microcopy";
    empty.textContent = "No supplied web pages stored in this project yet.";
    elements.sourceList.append(empty);
    return;
  }
  for (const source of sources) {
    const row = document.createElement("div");
    row.className = "source-row";
    const open = document.createElement("button");
    open.type = "button";
    open.className = "source-open";
    const title = document.createElement("span");
    title.textContent = source.title || source.final_url;
    const url = document.createElement("small");
    url.textContent = `${source.final_url} · ${source.char_count} CHARS${source.truncated ? " · TRUNCATED" : ""}`;
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
    elements.sourceList.append(row);
  }
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

  elements.memoryLoopEvents.replaceChildren();
  if (!data.events?.length) {
    const empty = document.createElement("p");
    empty.className = "microcopy";
    empty.textContent = "No returns stored for this agent in this project yet.";
    elements.memoryLoopEvents.append(empty);
  } else {
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
      elements.memoryLoopEvents.append(entry);
    }
  }

  const globalStats = data.global_stats || {};
  elements.globalMemorySummary.textContent = `${globalStats.event_count || 0} COMPACT RETURNS · ${globalStats.project_count || 0} SOURCE PROJECTS`;
  elements.globalMemoryEvents.replaceChildren();
  if (!data.global_events?.length) {
    const empty = document.createElement("p");
    empty.className = "microcopy";
    empty.textContent = "No cross-project continuity returns stored yet.";
    elements.globalMemoryEvents.append(empty);
    return;
  }
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
    elements.globalMemoryEvents.append(entry);
  }
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
      loadSharedContext(),
      loadFiles(),
      loadSources(),
      loadProposals(),
      loadProviderCatalog(),
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
}

$$(".tab").forEach((button) => button.addEventListener("click", async () => {
  activateTab(button.dataset.tab);
  if (button.dataset.tab !== "setup") return;
  try {
    await refreshSession();
  } catch (error) {
    showToast(`Could not refresh provider status: ${error.message}`, true);
  }
}));

elements.projectSelect.addEventListener("change", async () => {
  state.currentProject = elements.projectSelect.value;
  renderProjects();
  elements.filePreview.textContent = "Select a file.";
  elements.sourcePreview.textContent = "Select a source.";
  try {
    await Promise.all([loadMessages(), loadMemoryLoop(), loadFiles(), loadSources()]);
    if (window.matchMedia(UI_DEFAULTS.narrowViewportQuery).matches) {
      document.querySelector(".room")?.scrollIntoView({ block: "start" });
    }
    elements.messageInput.focus({ preventScroll: true });
  } catch (error) {
    showToast(error.message, true);
  }
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
    await Promise.all([loadMessages(), loadMemoryLoop(), loadFiles(), loadSources()]);
    showToast("Project created.");
  } catch (error) {
    showToast(error.message, true);
  }
});

elements.deleteProject.addEventListener("click", async () => {
  if (state.busy) return;
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
      loadFiles(),
      loadSources(),
      loadProposals(),
    ]);
    showToast(`Deleted ${project.name} and all of its stored project data.`);
    elements.messageInput.focus({ preventScroll: true });
  } catch (error) {
    showToast(error.message, true);
  } finally {
    elements.deleteProject.disabled = state.busy || !state.currentProject;
  }
});

elements.composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy) return;
  const message = elements.messageInput.value.trim();
  if (!message) return;
  const participants = currentParticipants();
  if (!participants.length) {
    showToast("Select at least one agent.", true);
    return;
  }

  setBusy(true);
  const optimistic = {
    role: "user",
    content: message,
    created_at: new Date().toISOString(),
    annotations: [],
    metadata: {},
  };
  const currentNodes = [...elements.messages.querySelectorAll(".message")];
  if (!currentNodes.length) elements.messages.replaceChildren();
  const optimisticViewport = captureMessagesViewport();
  elements.messages.append(renderMessage(optimistic));
  restoreMessagesViewport(optimisticViewport);
  elements.messageInput.value = "";

  try {
    beginTurnProgress();
    const result = await streamApi("/api/chat/stream", {
      method: "POST",
      body: JSON.stringify({
        project_id: state.currentProject,
        message,
        participants,
        research_mode: elements.researchMode.value,
      }),
    }, updateTurnProgress);
    if (!result) throw new Error("The server closed the response before returning a result.");
    const research = result.research;
    const sourceCount = result.web_sources?.length || 0;
    elements.researchIndicator.textContent = sourceCount
      ? `WEB: ${sourceCount} SHARED SOURCE${sourceCount === 1 ? "" : "S"}`
      : research.needs_search
        ? `SEARCH DISCOVERY UNAVAILABLE${research.lead_agent ? ` / ${agentMeta[research.lead_agent].name.toUpperCase()}` : ""}`
        : "NO SEARCH";
    await Promise.all([loadMessages(), loadMemoryLoop(), loadFiles(), loadSources(), loadProposals()]);
    renderAgentFailures(result.agent_failures || []);
    state.progressNode = null;
  } catch (error) {
    showToast(error.message, true);
    await loadMessages();
    state.progressNode = null;
    renderRequestError(error.message);
  } finally {
    setBusy(false);
    elements.messageInput.focus();
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
