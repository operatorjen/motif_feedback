let sessionToken = "";
let reconnectPromise = null;

const SESSION_ERROR = "Missing or invalid local session token.";

export function setSessionToken(token) {
  sessionToken = token || "";
}

async function reconnectSession() {
  if (!reconnectPromise) {
    reconnectPromise = (async () => {
      const response = await fetch("/api/session", {
        method: "GET",
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      if (!response.ok) throw new Error("Could not reconnect to the local server.");
      const session = await response.json();
      if (!session.token) throw new Error("The local server did not return a new session token.");
      setSessionToken(session.token);
      if (typeof window !== "undefined") {
        window.dispatchEvent(new CustomEvent("motif:session-reconnected"));
      }
      return session;
    })().finally(() => {
      reconnectPromise = null;
    });
  }
  return reconnectPromise;
}

function requestHeaders(options) {
  const headers = new Headers(options.headers || {});
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (sessionToken && !["GET", "HEAD"].includes((options.method || "GET").toUpperCase())) {
    headers.set("X-Motif-Token", sessionToken);
  }
  return headers;
}

function responseDetail(payload) {
  return typeof payload === "object" ? payload.detail || JSON.stringify(payload) : payload;
}

export async function api(path, options = {}, retried = false) {
  const headers = requestHeaders(options);
  const response = await fetch(path, { ...options, headers });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = responseDetail(payload);
    if (!retried && response.status === 403 && detail === SESSION_ERROR) {
      await reconnectSession();
      return api(path, options, true);
    }
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return payload;
}

export async function streamApi(path, options = {}, onEvent = () => {}, retried = false) {
  const headers = requestHeaders(options);
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = responseDetail(payload);
    if (!retried && response.status === 403 && detail === SESSION_ERROR) {
      await reconnectSession();
      return streamApi(path, options, onEvent, true);
    }
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  if (!response.body) throw new Error("This browser did not provide a streaming response body.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult = null;

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      onEvent(event);
      if (event.type === "error") throw new Error(event.detail || "The agent request failed.");
      if (event.type === "result") finalResult = event;
    }
    if (done) break;
  }
  return finalResult;
}
