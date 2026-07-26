import { api } from "./api.js";
import { sandboxErrorDocument, sandboxHtmlDocument } from "./demo_sandbox.js";

export function appendRoleSignal(container, signal, badgeClass) {
  const entry = document.createElement("div");
  entry.className = "role-signal-entry";
  const badge = document.createElement("span");
  badge.className = badgeClass;
  badge.textContent = `${signal.label} → ${signal.target} · ${signal.intensity}`;
  entry.append(badge);
  if (signal.observations && Object.keys(signal.observations).length) {
    const observations = document.createElement("code");
    observations.textContent = `observations: ${JSON.stringify(signal.observations)}`;
    entry.append(observations);
  }
  container.append(entry);
}

export function createDemoController({
  state,
  elements,
  defaults,
  userDisplayName,
  showToast,
  loadMessages,
}) {
  function showDemoOverlay(path, type) {
    state.activeDemo = { projectId: state.currentProject, path, type };
    elements.demoTitle.textContent = `${type === "python" ? "ISOLATED RUN" : "SANDBOXED DEMO"} · ${path}`;
    elements.demoOverlay.classList.remove("hidden", "collapsed");
    elements.demoCollapse.textContent = "COLLAPSE";
    elements.demoCollapse.setAttribute("aria-expanded", "true");
  }

  function showRoleSignals(signals = []) {
    elements.demoRoleSignals.replaceChildren();
    elements.demoRoleSignals.classList.toggle("hidden", !signals.length);
    for (const signal of signals) {
      appendRoleSignal(elements.demoRoleSignals, signal, "demo-role-signal");
    }
  }

  async function openHtmlDemo(path) {
    showDemoOverlay(path, "html");
    elements.demoRunControls.classList.add("hidden");
    elements.demoFrame.classList.remove("hidden");
    elements.demoOutput.classList.add("hidden");
    elements.demoFrame.srcdoc = "<p style=\"font-family:monospace\">Loading demo…</p>";
    try {
      const data = await api(
        `/api/files/${encodeURIComponent(state.currentProject)}/read?path=${encodeURIComponent(path)}`,
      );
      elements.demoFrame.srcdoc = sandboxHtmlDocument(data.content || "");
    } catch (error) {
      elements.demoFrame.srcdoc = sandboxErrorDocument(error.message || error);
      showToast(error.message, true);
    }
  }

  function preparePythonDemo(path) {
    showDemoOverlay(path, "python");
    elements.demoArguments.value = "";
    elements.demoStdin.value = "";
    elements.demoRunControls.classList.remove("hidden");
    elements.demoFrame.classList.add("hidden");
    elements.demoOutput.classList.remove("hidden");
    elements.demoOutput.textContent = [
      "READY FOR AN ISOLATED RUN — NO EXTERNAL DOCKER NETWORK ATTACHMENT",
      "",
      "Optional arguments are one per line. Press START, then answer prompts with SEND INPUT.",
      "Enter sends input; Shift+Enter adds a line. SEND EOF closes the script's input stream.",
      "The run ends on completion, its hard timeout, CANCEL, or closing this overlay.",
    ].join("\n");
    showRoleSignals([]);
    elements.demoStart.disabled = false;
    elements.demoCancel.disabled = true;
    elements.demoSendInput.disabled = true;
    elements.demoSendEof.disabled = true;
  }

  function appendLiveRunEvent(event) {
    const output = elements.demoOutput;
    const nearBottom = (
      output.scrollHeight - output.scrollTop - output.clientHeight
      < defaults.liveOutputFollowThresholdPx
    );
    if (event.type === "stdin") {
      const submitted = String(event.text || "").replace(/[\r\n]+$/, "");
      const leadingBreak = output.textContent && !output.textContent.endsWith("\n") ? "\n" : "";
      output.textContent += `${leadingBreak}[${userDisplayName().toUpperCase()} INPUT]\n› ${submitted.replace(/\r?\n/g, "\n› ")}\n`;
      state.activeRunPromptTail = "";
      if (nearBottom) output.scrollTop = output.scrollHeight;
      return;
    }
    const prefix = event.type === "stderr" ? "\n[STDERR]\n" : "";
    const visibleText = String(event.text || "").replace(
      /^__MOTIF_ROLE_SIGNAL__.*(?:\n|$)/gm,
      "",
    );
    output.textContent += `${prefix}${visibleText}`;
    if (nearBottom) output.scrollTop = output.scrollHeight;
    if (event.type === "stdout" && visibleText) {
      state.activeRunPromptTail = `${state.activeRunPromptTail}${visibleText}`.slice(
        -defaults.livePromptTailChars,
      );
      const lastNewline = Math.max(
        state.activeRunPromptTail.lastIndexOf("\n"),
        state.activeRunPromptTail.lastIndexOf("\r"),
      );
      if (lastNewline >= 0) {
        state.activeRunPromptTail = state.activeRunPromptTail.slice(lastNewline + 1);
      }
      if (
        state.activeRunPromptTail.trim()
        && !visibleText.endsWith("\n")
        && !state.activeRunInputClosed
      ) {
        window.requestAnimationFrame(() => {
          if (
            state.activeRunId
            && !state.activeRunInputClosed
            && !elements.demoOverlay.classList.contains("hidden")
            && !elements.demoOverlay.classList.contains("collapsed")
          ) {
            elements.demoStdin.focus({ preventScroll: true });
          }
        });
      }
    }
  }

  async function pollLiveRun(projectId, runId) {
    if (state.activeRunId !== runId) return;
    try {
      const status = await api(
        `/api/code-runs/${encodeURIComponent(projectId)}/${encodeURIComponent(runId)}?after=${state.activeRunCursor}`,
      );
      if (state.activeRunId !== runId) return;
      for (const event of status.events || []) appendLiveRunEvent(event);
      state.activeRunCursor = Number(status.next || state.activeRunCursor);
      if (!state.activeRunInputClosed) {
        elements.demoSendInput.disabled = false;
        elements.demoSendEof.disabled = false;
      }
    } catch {
      if (state.activeRunId !== runId) return;
    }
    if (state.activeRunId === runId) {
      state.activeRunPollTimer = window.setTimeout(
        () => pollLiveRun(projectId, runId),
        defaults.liveRunPollMs,
      );
    }
  }

  async function runPythonDemo(path) {
    showDemoOverlay(path, "python");
    elements.demoRunControls.classList.remove("hidden");
    elements.demoFrame.classList.add("hidden");
    elements.demoOutput.classList.remove("hidden");
    elements.demoOutput.textContent = "LIVE OUTPUT\n-----------\n";
    showRoleSignals([]);
    const argumentsList = elements.demoArguments.value
      .split("\n")
      .map((argument) => argument.trim())
      .filter(Boolean);
    const controller = new AbortController();
    const runId = globalThis.crypto?.randomUUID?.()
      || `run-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const projectId = state.currentProject;
    state.activeRunController = controller;
    state.activeRunId = runId;
    state.activeRunCursor = 0;
    state.activeRunInputClosed = false;
    state.activeRunPromptTail = "";
    elements.demoStart.disabled = true;
    elements.demoRefresh.disabled = true;
    elements.demoCancel.disabled = false;
    elements.demoSendInput.disabled = true;
    elements.demoSendEof.disabled = true;
    try {
      const resultPromise = api(`/api/code-runs/${encodeURIComponent(projectId)}`, {
        method: "POST",
        signal: controller.signal,
        body: JSON.stringify({
          run_id: runId,
          path,
          arguments: argumentsList,
          stdin: "",
        }),
      });
      state.activeRunPollTimer = window.setTimeout(
        () => pollLiveRun(projectId, runId),
        defaults.liveRunInitialPollMs,
      );
      const result = await resultPromise;
      if (result.error) throw new Error(result.error);
      const status = result.canceled
        ? "CANCELED"
        : result.timed_out
        ? "TIMED OUT"
        : result.return_code === 0
          ? "COMPLETED"
          : `EXIT ${result.return_code}`;
      elements.demoOutput.textContent = [
        `${status} · NETWORK ${String(result.network || "disabled").toUpperCase()} · TEMPORARY COPY`,
        result.output_truncated ? "OUTPUT TRUNCATED" : "",
        result.transcript ? `\nINTERACTIVE TRANSCRIPT\n----------------------\n${result.transcript}` : "",
        !result.transcript && result.stdout ? `\nSTDOUT\n------\n${result.stdout}` : "",
        !result.transcript && result.stderr ? `\nSTDERR\n------\n${result.stderr}` : "",
        !result.transcript && !result.stdout && !result.stderr ? "\n[No output.]" : "",
      ].filter(Boolean).join("\n");
      showRoleSignals(result.role_signals || []);
      await loadMessages();
    } catch (error) {
      if (error.name === "AbortError") {
        elements.demoOutput.textContent = "CANCELING ISOLATED RUN…";
        window.setTimeout(
          () => loadMessages().catch(() => {}),
          defaults.canceledRunRefreshDelayMs,
        );
      } else {
        elements.demoOutput.textContent = `RUNNER UNAVAILABLE\n\n${error.message}`;
        showToast(error.message, true);
      }
    } finally {
      if (state.activeRunController === controller) state.activeRunController = null;
      if (state.activeRunId === runId) state.activeRunId = null;
      state.activeRunInputClosed = false;
      state.activeRunPromptTail = "";
      window.clearTimeout(state.activeRunPollTimer);
      state.activeRunPollTimer = null;
      elements.demoStart.disabled = false;
      elements.demoRefresh.disabled = false;
      elements.demoCancel.disabled = true;
      elements.demoSendInput.disabled = true;
      elements.demoSendEof.disabled = true;
    }
  }

  async function sendLiveRunInput() {
    const demo = state.activeDemo;
    const runId = state.activeRunId;
    const text = elements.demoStdin.value;
    if (!demo || demo.type !== "python" || !runId || !text) return;
    try {
      await api(
        `/api/code-runs/${encodeURIComponent(demo.projectId)}/${encodeURIComponent(runId)}/input`,
        {
          method: "POST",
          body: JSON.stringify({ text, append_newline: true, eof: false }),
        },
      );
      elements.demoStdin.value = "";
      elements.demoStdin.focus();
    } catch (error) {
      showToast(error.message, true);
    }
  }

  async function closeLiveRunInput() {
    const demo = state.activeDemo;
    const runId = state.activeRunId;
    if (!demo || demo.type !== "python" || !runId) return;
    try {
      await api(
        `/api/code-runs/${encodeURIComponent(demo.projectId)}/${encodeURIComponent(runId)}/input`,
        {
          method: "POST",
          body: JSON.stringify({ text: "", append_newline: false, eof: true }),
        },
      );
      elements.demoSendInput.disabled = true;
      elements.demoSendEof.disabled = true;
      state.activeRunInputClosed = true;
      appendLiveRunEvent({ type: "stdout", text: "\n[INPUT CLOSED]\n" });
    } catch (error) {
      showToast(error.message, true);
    }
  }

  async function cancelLiveRun() {
    const demo = state.activeDemo;
    const runId = state.activeRunId;
    if (demo?.type === "python" && runId) {
      api(
        `/api/code-runs/${encodeURIComponent(demo.projectId)}/${encodeURIComponent(runId)}`,
        { method: "DELETE" },
      ).catch(() => {});
    }
    state.activeRunController?.abort();
  }

  async function refreshActiveDemo() {
    const demo = state.activeDemo;
    if (!demo || demo.projectId !== state.currentProject) return;
    if (demo.type === "python") await runPythonDemo(demo.path);
    else await openHtmlDemo(demo.path);
  }

  async function refreshHtmlDemoInPlace(path) {
    const demo = state.activeDemo;
    if (
      !demo
      || demo.type !== "html"
      || demo.projectId !== state.currentProject
      || demo.path !== path
      || elements.demoOverlay.classList.contains("hidden")
    ) return;
    const wasCollapsed = elements.demoOverlay.classList.contains("collapsed");
    await openHtmlDemo(path);
    if (wasCollapsed) {
      elements.demoOverlay.classList.add("collapsed");
      elements.demoCollapse.textContent = "EXPAND";
      elements.demoCollapse.setAttribute("aria-expanded", "false");
    }
  }

  return {
    cancelLiveRun,
    closeLiveRunInput,
    openHtmlDemo,
    preparePythonDemo,
    refreshActiveDemo,
    refreshHtmlDemoInPlace,
    runPythonDemo,
    sendLiveRunInput,
    showRoleSignals,
  };
}
