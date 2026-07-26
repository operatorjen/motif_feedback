export const DEMO_CSP = [
  "default-src 'none'",
  "base-uri 'none'",
  "connect-src 'none'",
  "font-src 'none'",
  "form-action 'none'",
  "frame-src 'none'",
  "img-src data: blob:",
  "media-src 'none'",
  "object-src 'none'",
  "script-src 'unsafe-inline'",
  "style-src 'unsafe-inline'",
].join("; ");

function installPolicy(documentNode) {
  const policy = documentNode.createElement("meta");
  policy.httpEquiv = "Content-Security-Policy";
  policy.content = DEMO_CSP;
  documentNode.head.prepend(policy);

  for (const meta of documentNode.querySelectorAll("meta[http-equiv]")) {
    if (meta !== policy && meta.httpEquiv.toLowerCase() === "refresh") meta.remove();
  }
}

function serialize(documentNode) {
  installPolicy(documentNode);
  return `<!doctype html>\n${documentNode.documentElement.outerHTML}`;
}

export function sandboxHtmlDocument(content) {
  const parsed = new DOMParser().parseFromString(content, "text/html");
  return serialize(parsed);
}

export function sandboxErrorDocument(message) {
  const parsed = document.implementation.createHTMLDocument("Demo error");
  const error = parsed.createElement("pre");
  error.textContent = String(message || "The demo could not be opened.");
  parsed.body.replaceChildren(error);
  return serialize(parsed);
}
