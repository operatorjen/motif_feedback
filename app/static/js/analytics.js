import { api, setSessionToken } from "./api.js";

const AGENTS = {
  agent_a: {
    name: "The Phenomenologist",
    short: "Phenomenologist",
    className: "agent-a",
    color: "#ef8354",
  },
  agent_b: {
    name: "The Cyberneticist",
    short: "Cyberneticist",
    className: "agent-b",
    color: "#58a6ff",
  },
  agent_c: {
    name: "The Game Theorist",
    short: "Game Theorist",
    className: "agent-c",
    color: "#c792ea",
  },
};

const FEEDBACK = {
  useful_difference: "USEFUL DIFFERENCE",
  repetitive: "REPETITIVE",
  off_lens: "OFF-LENS",
  unsupported: "UNSUPPORTED",
};

const CONTEXT_LABELS = {
  recent_message: "Recent messages",
  same_turn_message: "Same-turn responses",
  local_memory: "Project memory",
  global_memory: "Cross-project memory",
  own_motif: "Own motifs",
  other_observer_motif: "Other observers’ motifs",
  pattern_checkpoint: "Pattern checkpoints",
  web_source: "Web sources",
  role_signal: "Role signals",
};

const elements = {
  project: document.querySelector("#analytics-project"),
  refresh: document.querySelector("#analytics-refresh"),
  status: document.querySelector("#analytics-status"),
  coverageNote: document.querySelector("#coverage-note"),
  coverage: document.querySelector("#coverage-grid"),
  activity: document.querySelector("#activity-chart"),
  agents: document.querySelector("#agent-comparison"),
  reliability: document.querySelector("#reliability-grid"),
  contexts: document.querySelector("#context-chart"),
  motifExposure: document.querySelector("#motif-exposure"),
  motifLifecycle: document.querySelector("#motif-lifecycle"),
  projectActivity: document.querySelector("#project-activity"),
  review: document.querySelector("#response-review"),
  toast: document.querySelector("#analytics-toast"),
};

let snapshot = null;

function showToast(message, error = false) {
  elements.toast.textContent = message;
  elements.toast.classList.toggle("error", error);
  elements.toast.classList.remove("hidden");
  window.setTimeout(() => elements.toast.classList.add("hidden"), 2600);
}

function setStatus(text, error = false) {
  elements.status.textContent = text;
  elements.status.classList.toggle("error", error);
}

function number(value) {
  return new Intl.NumberFormat().format(Number(value || 0));
}

function formatDuration(milliseconds) {
  if (!Number.isFinite(milliseconds)) return "NO TIMING DATA";
  if (milliseconds < 1000) return `${Math.round(milliseconds)} MS`;
  return `${(milliseconds / 1000).toFixed(1)} S`;
}

function metricCard(value, label, detail = "") {
  const card = document.createElement("article");
  card.className = "metric-card";
  const strong = document.createElement("strong");
  strong.textContent = number(value);
  const span = document.createElement("span");
  span.textContent = detail ? `${label} · ${detail}` : label;
  card.append(strong, span);
  return card;
}

function renderCoverage() {
  const coverage = snapshot.coverage;
  elements.coverage.replaceChildren(
    metricCard(coverage.agent_responses, "agent responses", `${coverage.messages} total messages`),
    metricCard(coverage.memory_events, "project memory events"),
    metricCard(coverage.chat_turns, "durable room turns"),
    metricCard(coverage.prompt_runs, "instrumented agent beats"),
    metricCard(coverage.context_exposures, "recorded context exposures"),
    metricCard(coverage.motifs, "observer-owned motifs", `${coverage.feedback_events} feedback events`),
  );
  const scopeName = snapshot.scope.project_name || "all projects";
  elements.coverageNote.textContent = `${scopeName.toUpperCase()} · prompt instrumentation applies to newly executed turns.`;
}

function svgElement(name, attributes = {}, text = "") {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
  if (text) element.textContent = text;
  return element;
}

function renderActivity() {
  elements.activity.replaceChildren();
  const data = snapshot.activity;
  if (!data.length) {
    renderEmpty(elements.activity, "No stored agent responses exist in this scope.");
    return;
  }

  const width = 920;
  const height = 310;
  const margin = { top: 18, right: 18, bottom: 48, left: 48 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const maximum = Math.max(
    1,
    ...data.flatMap((row) => Object.keys(AGENTS).map((agentId) => row[agentId] || 0)),
  );
  const ceiling = Math.ceil(maximum / 10) * 10 || 10;
  const svg = svgElement("svg", {
    class: "analytics-svg",
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Daily agent response counts",
  });

  for (let tick = 0; tick <= 4; tick += 1) {
    const value = Math.round((ceiling / 4) * tick);
    const y = margin.top + plotHeight - (value / ceiling) * plotHeight;
    svg.append(
      svgElement("line", {
        x1: margin.left,
        x2: width - margin.right,
        y1: y,
        y2: y,
        class: tick === 0 ? "axis-line" : "grid-line",
      }),
      svgElement(
        "text",
        { x: margin.left - 8, y: y + 4, "text-anchor": "end" },
        String(value),
      ),
    );
  }

  const groupWidth = plotWidth / data.length;
  const barWidth = Math.max(6, Math.min(24, (groupWidth - 12) / 3));
  data.forEach((row, index) => {
    Object.keys(AGENTS).forEach((agentId, agentIndex) => {
      const value = row[agentId] || 0;
      const barHeight = (value / ceiling) * plotHeight;
      const groupStart = margin.left + index * groupWidth;
      const x = groupStart + (groupWidth - barWidth * 3 - 6) / 2 + agentIndex * (barWidth + 3);
      const y = margin.top + plotHeight - barHeight;
      const bar = svgElement("rect", {
        x,
        y,
        width: barWidth,
        height: barHeight,
        class: `bar-${agentId}`,
        fill: AGENTS[agentId].color,
      });
      bar.append(svgElement("title", {}, `${row.day} · ${AGENTS[agentId].name}: ${value}`));
      svg.append(bar);
    });
    const label = row.day.slice(5);
    svg.append(
      svgElement(
        "text",
        {
          x: margin.left + index * groupWidth + groupWidth / 2,
          y: height - 20,
          "text-anchor": "middle",
        },
        label,
      ),
    );
  });
  const legend = document.createElement("div");
  legend.className = "chart-legend";
  Object.entries(AGENTS).forEach(([agentId, meta]) => {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = `legend-swatch ${meta.className}`;
    item.append(swatch, document.createTextNode(meta.name));
    legend.append(item);
  });
  elements.activity.append(svg, legend);
}

function renderAgents() {
  elements.agents.replaceChildren();
  snapshot.agents.forEach((agent) => {
    const meta = AGENTS[agent.agent_id];
    const row = document.createElement("article");
    row.className = `agent-row ${meta.className}`;
    const identity = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = meta.name;
    const coverage = document.createElement("small");
    coverage.textContent = `${number(agent.prompt_runs)} instrumented beats`;
    identity.append(name, coverage);
    const responses = document.createElement("div");
    responses.append(
      document.createTextNode(number(agent.responses)),
      Object.assign(document.createElement("small"), { textContent: " responses" }),
    );
    const average = document.createElement("div");
    average.className = "agent-secondary";
    average.append(
      document.createTextNode(number(Math.round(agent.average_chars))),
      Object.assign(document.createElement("small"), { textContent: " avg chars" }),
    );
    const useful = document.createElement("div");
    useful.append(
      document.createTextNode(number(agent.feedback.useful_difference)),
      Object.assign(document.createElement("small"), { textContent: " useful" }),
    );
    const positions = document.createElement("div");
    positions.className = "agent-secondary";
    positions.append(
      document.createTextNode(
        Object.entries(agent.speaker_positions)
          .map(([position, count]) => `${position}:${count}`)
          .join(" · ") || "—",
      ),
      Object.assign(document.createElement("small"), { textContent: " positions" }),
    );
    row.append(identity, responses, average, useful, positions);
    elements.agents.append(row);
  });
}

function debugRow(label, value) {
  const row = document.createElement("div");
  row.className = "debug-row";
  const span = document.createElement("span");
  span.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = value;
  row.append(span, strong);
  return row;
}

function renderReliability() {
  const reliability = snapshot.reliability;
  const statuses = Object.entries(reliability.turn_statuses)
    .map(([status, count]) => `${status.toUpperCase()} ${count}`)
    .join(" · ") || "NO DURABLE TURNS";
  const usage = reliability.provider_usage || {};
  elements.reliability.replaceChildren(
    debugRow("TURN STATUS", statuses),
    debugRow("AVERAGE TURN DURATION", formatDuration(reliability.average_duration_ms)),
    debugRow("PROVIDER REQUESTS", number(reliability.provider_requests)),
    debugRow("PROMPT TOKENS", number(usage.prompt_tokens)),
    debugRow("COMPLETION TOKENS", number(usage.completion_tokens)),
    debugRow("TOTAL TOKENS", number(usage.total_tokens)),
  );
}

function renderContexts() {
  elements.contexts.replaceChildren();
  if (!snapshot.contexts.length) {
    renderEmpty(
      elements.contexts,
      "No context manifests have been recorded yet. The next room turn will begin this dataset.",
    );
    return;
  }
  const grouped = new Map();
  snapshot.contexts.forEach((item) => {
    const record = grouped.get(item.context_kind) || {
      context_kind: item.context_kind,
      agent_a: 0,
      agent_b: 0,
      agent_c: 0,
    };
    record[item.agent_id] = item.count;
    grouped.set(item.context_kind, record);
  });
  const rows = [...grouped.values()].sort(
    (a, b) =>
      b.agent_a + b.agent_b + b.agent_c - (a.agent_a + a.agent_b + a.agent_c),
  );
  const maximum = Math.max(...rows.map((row) => row.agent_a + row.agent_b + row.agent_c), 1);
  rows.forEach((item) => {
    const total = item.agent_a + item.agent_b + item.agent_c;
    const row = document.createElement("div");
    row.className = "context-row";
    const label = document.createElement("div");
    label.className = "context-label";
    const strong = document.createElement("strong");
    strong.textContent = CONTEXT_LABELS[item.context_kind] || item.context_kind;
    label.append(strong);
    const bar = document.createElement("div");
    bar.className = "context-bar";
    Object.keys(AGENTS).forEach((agentId) => {
      const segment = document.createElement("span");
      segment.className = `context-segment ${AGENTS[agentId].className}`;
      segment.style.width = `${(item[agentId] / maximum) * 100}%`;
      segment.setAttribute("aria-label", `${AGENTS[agentId].name}: ${item[agentId]}`);
      bar.append(segment);
    });
    const value = document.createElement("div");
    value.className = "context-value";
    value.textContent = number(total);
    row.append(label, bar, value);
    elements.contexts.append(row);
  });
  const legend = document.createElement("div");
  legend.className = "chart-legend";
  Object.entries(AGENTS).forEach(([agentId, meta]) => {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = `legend-swatch ${meta.className}`;
    item.append(swatch, document.createTextNode(meta.short));
    legend.append(item);
  });
  elements.contexts.append(legend);
}

function renderMotifExposure() {
  const exposure = snapshot.motifs.return_exposure;
  const total = exposure.prompted + exposure.unprompted + exposure.unknown;
  elements.motifExposure.replaceChildren();
  if (!total) {
    renderEmpty(elements.motifExposure, "No agent-authored motif observations exist in this scope.");
    return;
  }
  const bar = document.createElement("div");
  bar.className = "exposure-bar";
  ["prompted", "unprompted", "unknown"].forEach((kind) => {
    const segment = document.createElement("span");
    segment.className = `exposure-segment ${kind}`;
    segment.style.width = `${(exposure[kind] / total) * 100}%`;
    segment.setAttribute("aria-label", `${kind}: ${exposure[kind]}`);
    bar.append(segment);
  });
  const summary = document.createElement("div");
  summary.className = "exposure-summary";
  [
    ["prompted", "MOTIF WAS IN PROMPT"],
    ["unprompted", "NOT IN PROMPT"],
    ["unknown", "PRE-INSTRUMENTATION"],
  ].forEach(([kind, label]) => {
    const article = document.createElement("article");
    const strong = document.createElement("strong");
    strong.textContent = number(exposure[kind]);
    const span = document.createElement("span");
    span.textContent = label;
    article.append(strong, span);
    summary.append(article);
  });
  elements.motifExposure.append(bar, summary);
}

function renderMotifLifecycle() {
  elements.motifLifecycle.replaceChildren();
  const grouped = new Map();
  snapshot.motifs.statuses.forEach((item) => {
    const row = grouped.get(item.status) || { status: item.status, agent_a: 0, agent_b: 0, agent_c: 0 };
    row[item.agent_id] = item.count;
    grouped.set(item.status, row);
  });
  if (!grouped.size) {
    renderEmpty(elements.motifLifecycle, "No motifs exist in this scope.");
    return;
  }
  [...grouped.values()].forEach((item) => {
    const row = document.createElement("div");
    row.className = "lifecycle-row";
    const status = document.createElement("strong");
    status.className = "lifecycle-status";
    status.textContent = item.status.toUpperCase();
    row.append(status);
    Object.keys(AGENTS).forEach((agentId) => {
      const value = document.createElement("span");
      value.className = `lifecycle-agent ${AGENTS[agentId].className}`;
      const label = document.createElement("span");
      label.textContent = AGENTS[agentId].short;
      const count = document.createElement("strong");
      count.textContent = number(item[agentId]);
      value.append(label, count);
      row.append(value);
    });
    elements.motifLifecycle.append(row);
  });
}

function renderProjects() {
  elements.projectActivity.replaceChildren();
  const rows = snapshot.project_activity.filter((item) => item.responses > 0);
  if (!rows.length) {
    renderEmpty(elements.projectActivity, "No project response activity exists.");
    return;
  }
  const maximum = Math.max(...rows.map((row) => row.responses), 1);
  rows.forEach((item) => {
    const row = document.createElement("div");
    row.className = "project-row";
    const label = document.createElement("strong");
    label.textContent = item.project_name;
    const bar = document.createElement("div");
    bar.className = "project-bar";
    const segment = document.createElement("span");
    segment.className = "project-segment agent-b";
    segment.style.width = `${(item.responses / maximum) * 100}%`;
    bar.append(segment);
    const value = document.createElement("div");
    value.className = "project-value";
    value.textContent = number(item.responses);
    row.append(label, bar, value);
    elements.projectActivity.append(row);
  });
}

function renderReview() {
  elements.review.replaceChildren();
  if (!snapshot.recent_responses.length) {
    renderEmpty(elements.review, "No agent responses are available for review.");
    return;
  }
  snapshot.recent_responses.forEach((response) => {
    const meta = AGENTS[response.agent_id] || {
      name: response.agent_id || "Agent",
      className: "",
    };
    const card = document.createElement("article");
    card.className = `review-card ${meta.className}`;
    const head = document.createElement("div");
    head.className = "review-head";
    const name = document.createElement("strong");
    name.textContent = meta.name;
    const context = document.createElement("span");
    const position = response.speaker_position ? ` · POSITION ${response.speaker_position}` : "";
    context.textContent = `${response.project_name}${position} · ${new Date(response.created_at).toLocaleString()}`;
    head.append(name, context);
    const excerpt = document.createElement("p");
    excerpt.textContent = response.excerpt;
    const actions = document.createElement("div");
    actions.className = "review-actions";
    Object.entries(FEEDBACK).forEach(([kind, label]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      const active = response.feedback.includes(kind);
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
      button.addEventListener("click", () => toggleFeedback(response, kind, button));
      actions.append(button);
    });
    card.append(head, excerpt, actions);
    elements.review.append(card);
  });
}

async function toggleFeedback(response, feedbackType, button) {
  const active = button.getAttribute("aria-pressed") !== "true";
  button.disabled = true;
  try {
    await api("/api/analytics/feedback", {
      method: "POST",
      body: JSON.stringify({
        project_id: response.project_id,
        message_id: response.id,
        feedback_type: feedbackType,
        active,
      }),
    });
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
    response.feedback = active
      ? [...new Set([...response.feedback, feedbackType])]
      : response.feedback.filter((item) => item !== feedbackType);
    showToast(active ? "FEEDBACK RECORDED" : "FEEDBACK REMOVED");
  } catch (error) {
    showToast(error.message || "Could not save feedback.", true);
  } finally {
    button.disabled = false;
  }
}

function renderEmpty(container, message) {
  const empty = document.createElement("div");
  empty.className = "empty-measurement";
  empty.textContent = message;
  container.replaceChildren(empty);
}

function render() {
  renderCoverage();
  renderActivity();
  renderAgents();
  renderReliability();
  renderContexts();
  renderMotifExposure();
  renderMotifLifecycle();
  renderProjects();
  renderReview();
}

async function loadAnalytics() {
  elements.refresh.disabled = true;
  setStatus("LOADING DATA");
  try {
    const projectId = elements.project.value;
    const query = projectId ? `?project_id=${encodeURIComponent(projectId)}` : "";
    snapshot = await api(`/api/analytics${query}`);
    render();
    setStatus("DATA CURRENT");
  } catch (error) {
    setStatus("LOAD FAILED", true);
    showToast(error.message || "Could not load analytics.", true);
  } finally {
    elements.refresh.disabled = false;
  }
}

async function initialize() {
  try {
    const session = await api("/api/session");
    setSessionToken(session.token);
    session.projects.forEach((project) => {
      const option = document.createElement("option");
      option.value = project.id;
      option.textContent = project.name.toUpperCase();
      elements.project.append(option);
    });
    elements.project.addEventListener("change", loadAnalytics);
    elements.refresh.addEventListener("click", loadAnalytics);
    await loadAnalytics();
  } catch (error) {
    setStatus("INITIALIZATION FAILED", true);
    showToast(error.message || "Could not initialize analytics.", true);
  }
}

initialize();
