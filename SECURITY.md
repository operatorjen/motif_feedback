# Security Model

Motif Feedback is a single-user localhost application for personal experimentation. It is not
designed to authenticate or isolate multiple users, and it does not make arbitrary code
risk-free.

The design assumes that the local user, other local processes, Docker, the host kernel,
application dependencies, and selected model-provider endpoints are trusted. Model output,
project files, fetched pages, HTML demos, and user-approved Python code may be untrusted.

## Core boundaries

### Local application and containers

- Compose publishes the application only on `127.0.0.1`.
- `X-Motif-Token` protects state-changing browser requests from cross-origin request forgery. It
  is not user authentication: any local process that can reach the application can request the
  token and control the application.
- With the default Compose values, both containers run as UID/GID `10001`, drop all Linux
  capabilities, use read-only root filesystems, and mount non-executable temporary filesystems.
  Do not set `APP_UID` or `APP_GID` to `0`.
- The application container's only writable host bind mount is `./workspace`. The runner has no
  host-project mount, Docker socket, home-directory mount, or provider-key environment.

### Agent tools and project files

- Agent tools accept only confined project-relative paths. They reject path traversal and
  symbolic links, restrict file types and sizes, and use atomic writes.
- Agents may create files and revise their own files. Editing another agent's file requires the
  user to enable sharing for that file. Agents cannot overwrite uploaded user files or delete
  files and projects.
- Agent tools expose no shell, package installer, Python executor, browser automation, desktop
  control, or Docker control.
- Project files are data, not trusted instructions. Their contents can still attempt prompt
  injection when an agent reads them. Tool confinement limits what a persuaded model can do, but
  it does not guarantee that the model will ignore instructions embedded in a file.
- Raster uploads receive basic filename-signature validation. SVG files reject scripts,
  embedded HTML, event handlers, and external references before storage and preview.

### User-approved Python runner

- Agents may write Python files but have no runner tool. In the interface, a run begins only when
  the local user selects one `.py` file and presses **RUN**.
- The application sends a bounded text copy of the selected project to a separate runner over a
  private Unix socket. The runner receives no database, provider keys, application environment,
  Docker socket, or host-project mount.
- The runner uses `network_mode: none`, which provides no external Docker network attachment.
  Code cannot reach provider endpoints or public websites.
- Python runs without a shell using a fixed isolated invocation, a disposable working copy, a
  minimal environment, and CPU, wall-clock, memory, process, file, descriptor, input, and output
  limits. Cancellation kills the process group and adopted descendants.
- Standard input, stdout, and stderr are bounded. Submitted input and output may be retained in
  the project conversation so agents can observe the result later.
- A script may emit only fixed, validated conversational-attention signals. Script-authored
  observation text is retained for inspection but is not inserted into agent prompts.

The runner remains an ordinary Docker container sharing the host kernel. Use a disposable VM or
microVM when code may be hostile or stronger isolation is required.

### Browser previews and supplied URLs

- Model text is rendered through application-owned DOM construction rather than injected HTML.
- Self-contained HTML demos run in a sandboxed, opaque-origin iframe. Content policies block
  subresource networking, forms, nested frames, objects, media, and external scripts or styles.
  A demo can still attempt to navigate its own iframe document, so this is not equivalent to the
  runner's network isolation.
- URLs pasted by the user are fetched by a read-only server broker. It accepts HTTP(S) on ports
  80/443, can send one or two locally configured user-agent profiles but no application cookies
  or provider credentials, does not execute JavaScript, validates redirects, limits response size
  and time, and blocks local, private, reserved, link-local, and Docker-host destinations.
- If every configured direct attempt returns HTTP 403 and research is enabled, the exact failed
  URL may be included in one selected agent request with provider-native search enabled. This
  sends the URL and related prompt context to the selected provider and its search infrastructure.
  Providers without a declared compatible search mode are not asked to search.
- Provider search output is usable evidence only when it includes at least one returned URL
  citation. If no direct snapshot and no cited search evidence are available, ordinary agent turns
  are skipped and uncited search text is not persisted as an agent response.
- Direct snapshots and search-derived answers remain distinct. Stored provenance identifies the
  direct retrieval method and attempt count or, for search fallback, the provider/model, failed
  URL/status, and provider-returned citations. A citation does not prove that the provider opened
  the original page; agents are instructed to disclose substitute evidence.
- Fetched page text is marked as untrusted evidence in agent prompts. Prompt injection is still
  possible at the model level, including through provider search results. DNS and URL checks
  reduce SSRF risk for direct retrieval but are not a substitute for an outbound proxy or
  firewall when stronger egress control is required.

### Personas, persistence, and deletion

- Agent persona updates pass through a validated server path. An agent cannot update another
  persona or its own user-owned `core_motif`. Every cited evidence ID must resolve to that
  agent's project-local memory or to a provenance-carried return from another project.
- Fast-changing peripheral state may commit within bounded fields. Relationship memory requires
  repeated verified returns. Structural changes remain dormant records: repetition accumulates
  evidence, conflicting records supersede older ones, and only a matching manual persona edit
  incorporates one into active identity. No agent consensus or voting process applies changes.
- Detailed memory remains project-local. Compact, provenance-labeled continuity summaries from
  successful turns may be supplied to the same agent in another project. Raw beat records remain
  available for inspection; prompt preparation groups beats from one agent turn and favors
  relevant records without repeating exchanges already present in the room transcript.
- The interface asks for confirmation before deleting a file or project, but confirmation is a
  browser safeguard rather than a separate server authorization step. Deletion requests still
  require the session token.
- Deleting a project removes its confined directory, project-scoped database records, and
  cross-project continuity records derived from that project.

### Providers and secrets

- Provider keys are loaded from `.env` into the application container. They are not copied into
  the image or stored in the provider catalog. The application exposes readiness and key-variable
  names to the browser, not the configured key values.
- Provider catalog entries are schema-validated and cannot embed credentials in their URLs.
- Selecting a provider authorizes the application to send it the relevant system prompts,
  persona state, conversation, tool definitions, continuity context, and any project or web
  content used in the turn. Custom and local-network endpoints receive the same material.
- Provider responses and tool-call loops are bounded. Provider requests ignore proxy variables,
  and stored public tool-event metadata omits generated file, source, YAML, and Markdown bodies.
- When a streaming browser response closes before a room turn completes, the unfinished
  provider task is canceled and awaited so it does not continue as detached background work.
  The started turn is marked interrupted; any agent messages committed before interruption
  remain part of the conversation rather than being rolled back.
- Prompts submitted during an active room turn wait in a bounded in-memory browser queue. They are
  sent to the server sequentially and are not written to browser storage. Reloading or closing the
  tab discards prompts that have not started.
- A prompt receives a unique turn identifier when it enters the browser queue. Once that prompt
  starts, its request fingerprint, lifecycle, completed result, bounded progress timings, provider
  and model labels, URLs involved in retrieval, and provider-reported token counts may be retained
  in SQLite. Reusing a completed identifier replays its stored result; reusing the identifier for
  different content is rejected. Trace records omit prompt text, response bodies, tool arguments,
  file contents, and source text, although the completed result duplicates messages already stored
  in the conversation.

## What this does not protect against

- Another local process accessing the loopback application and obtaining its session token.
- A compromised Docker daemon, container runtime, host kernel, browser profile, dependency, or
  provider endpoint.
- A container escape or kernel exploit attempted by approved Python code.
- Prompt injection contained in project files, uploaded text, or fetched pages.
- Disclosure of prompts, conversations, persona state, continuity summaries, tool definitions,
  or relevant project content to a selected provider.
- A sandboxed HTML demo navigating its own iframe document.
- Unsafe future tools or mounts added without equivalent confinement.

Do not expose the application port beyond loopback. Do not mount the Docker socket, host root,
home directory, SSH directory, or cloud credentials into either container. Keep `.env` out of
source control.
