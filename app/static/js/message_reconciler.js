export function messageRenderSignature(message) {
  return JSON.stringify([
    message.role || null,
    message.agent_id || null,
    message.content || "",
    message.annotations || [],
    message.metadata || {},
    message.created_at || null,
  ]);
}

export function decorateMessageNode(node, message) {
  if (!message.id) return node;
  node.dataset.messageId = String(message.id);
  node.dataset.messageSignature = messageRenderSignature(message);
  return node;
}

export function reconcileMessageNodes(
  container,
  messages,
  renderMessage,
  { reuseExisting = true } = {},
) {
  const existing = new Map();
  if (reuseExisting) {
    for (const node of container.children) {
      if (node.dataset?.messageId) existing.set(node.dataset.messageId, node);
    }
  }

  let created = 0;
  let reused = 0;
  const nodes = messages.map((message) => {
    const id = message.id ? String(message.id) : "";
    const signature = messageRenderSignature(message);
    const prior = id ? existing.get(id) : null;
    if (prior?.dataset?.messageSignature === signature) {
      reused += 1;
      return prior;
    }
    created += 1;
    return decorateMessageNode(renderMessage(message), message);
  });
  container.replaceChildren(...nodes);
  return { created, reused };
}
