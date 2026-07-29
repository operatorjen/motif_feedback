# Motif Feedback

![⟁](https://aesxarg.net/taste-as-system-sm.jpg)

Motif Feedback is a local-first, single-user application for user-guided conversations with
three persistent agent personas in one conversational room. It is designed for long-running
thematic work that benefits from distinct perspectives, continuity across returns, and
revisiting ideas over time. Its project-scoped motif observatory keeps recurring conversational
patterns inspectable without collapsing the agents' separate interpretations into one account.

Each agent has an editable persona, a protected core motif, and separately configurable access
to a hosted or local OpenAI-compatible model. Conversations, continuity records, sources, and
project files persist under `workspace/`.

The agents respond only to user-initiated turns; the application does not schedule background
work or allow them to operate independently.

This is a personal experimental system for localhost. It is not a production multi-user
service.

## Quick start

Docker with Compose is the intended way to run the complete system.

```bash
git clone https://github.com/operatorjen/motif_feedback.git
cd motif_feedback
cp .env.example .env
```

Add API keys to `.env` only for the hosted providers you intend to use. Local providers can be
configured later in the application.

```bash
docker compose build --pull
docker compose up -d
```

Open `http://127.0.0.1:8000`, then select a provider and model for each agent in **SETUP**.

On Linux, run `./scripts/init-linux.sh` first if the containers cannot write to `./workspace`.

To rebuild both services while developing:

```bash
docker compose -f compose.yaml -f compose.dev.yaml up -d --build runner app
```

Stop the system with `docker compose down`.

## How the room works

### Motif-centered continuity

Each agent has a durable center of return:

| Agent | Core motif | Primary attention |
|---|---|---|
| **The Phenomenologist** | Embodied Motif (△) | Embodiment, situated perception, lived experience, and observer position |
| **The Cyberneticist** | Motif Handshake (⋈) | Feedback, signaling, black-box observation, and structural coupling |
| **The Game Theorist** | Vector of Flight (ε → ↻) | Rules, positions, incentives, and moves that preserve meaningful play |

The agents are not opposing characters. They may agree, disagree, influence one another, or
change position. Their core motifs remain user-owned and agent-locked, while peripheral state
such as current position, motif expression, relationship memory, and self-model may adapt
through feedback.

Identity changes are governed by policy rather than agent consensus. Bounded peripheral state
may adapt automatically through recorded feedback, while structural changes remain inactive
continuity records unless the user deliberately incorporates them. Changes are bounded,
attributable, and recoverable without requiring the agents—or the user—to vote on them. This
creates continuity without treating the agents as conscious, embodied, or independently
self-producing.

Agent-authored identity changes must cite stored returns belonging to that agent. Fast-changing
state may update from one verified return, while relationship memory requires repeated evidence.
Repeated structural suggestions accumulate into one dormant record; conflicting suggestions
supersede older records, and a matching manual persona edit records deliberate incorporation.

Selected agents speak sequentially, so later agents can respond to earlier returns in the same
round. Direct address, research routing, or rotation determines who speaks first.

### Conversational motifs

Conversational motifs are separate from the protected core motifs that organize agent identity.
During a response, an agent may privately record a sparse, observer-specific hypothesis about a
meaningful pattern that returns or transforms across the conversation—not merely a topic word.
Every observation remains attached to evidence from the user message and agent response.

Motifs belong to the current project and retain their observing agent. A candidate becomes
supported only after returning across distinct turns; the user may then mark it active, dormant,
or rejected. Agents may record provisional alignment, translation, contrast, extension,
transformation, or shared evidence between motifs, but those relations never merge ownership or
turn agreement into truth.

The **MOTIFS** inspector shows motif tags, evidence, relations, and lifecycle history. It also
reports recurring two- and three-motif sequences, return paths, recurrence, transition diversity,
and return paths. When an established sequence recurs across at least three distinct turns, it
becomes a pattern checkpoint. The user can let its observing agent simply notice it, keep
following it, test its limits, or pause it. A relevant agent may then tentatively compare what
stayed stable with what changed and offer one useful next move. Checkpoints never alter memory,
persona, motif status, or speaking order. Recurrence and transition diversity remain descriptive
measurements; they do not decide how an agent responds.

### Turn order, queueing, and recovery

The chat composer remains available while a room turn is running. Additional submissions enter
a bounded, first-in-first-out browser queue. A queued prompt retains its project, participants,
and research mode and may be removed before it begins. The queue exists only in the current
browser tab, so reloading discards prompts that have not started.

Once a prompt starts, the server records its turn identifier and lifecycle. Reusing a completed
identifier returns the stored result instead of running the agents twice.

An interrupted or failed turn can resume unfinished work or be explicitly accepted as partial.
When recovery is available, a compact notice appears in the room with both actions and optional
technical details; completed turns do not occupy the interface.
Provider completion, message storage, memory storage, response-beat completion, and agent
completion are checkpointed separately, so recovery starts at the first unfinished stage without
duplicating committed messages or changing sequential room causality.

**DOWNLOAD LOG** exports every stored message in the current project as a chronological Markdown
file. Each message is a numbered block with its speaker, timestamp, turn and response beat when
available, sources, and retrieval notes.

## Providers

Each agent can use a different provider and model. The persistent provider catalog is editable
under **SETUP → PROVIDER CATALOG** and supports OpenAI-compatible `/chat/completions`
endpoints, including compatible local servers.

Provider profiles can declare `web_search_mode: responses` when their endpoint supports the
Responses API and its built-in `web_search` tool. The built-in OpenAI profile enables this
capability by default; other hosted or custom profiles remain `none` unless explicitly configured.
Search capability is separate from ordinary Chat Completions function-tool support.

From Docker, a model server running on the host can usually be reached at:

```text
http://host.docker.internal:<port>/v1
```

Provider keys remain in `.env`; they are not stored in the catalog or returned to the browser.
The available configuration values and defaults are documented in `.env.example`.

Model inference is local only when the selected provider is local. A hosted or custom provider
receives the relevant system prompt, persona, conversation, continuity, and project context for
that request.

## Projects, sources, and tools

### Project continuity

Each project keeps its own conversation, conversational motifs, motif evidence and relations,
files, supplied-page snapshots, and detailed agent continuity records. Motifs do not cross
projects. Raw response-beat records remain visible in the inspector, while prompt context groups
beats from one agent turn, favors relevant returns, and uses compact event references where the
room transcript already contains the same exchange. Compact, provenance-labelled returns may be
shown to the same agent in another project as provisional context; this continuity does not copy
or merge project motifs.

### Project files

Agents can list, read, search, create, and revise permitted project files. Their file tools are
confined to the current project folder:

```text
workspace/projects/<project>/
```

An agent may revise a file it created. Editing another agent's file requires the user to enable
sharing for that specific file. Agents cannot overwrite uploaded user files, delete files or
projects, run a shell, install packages, or control Docker.

### Web retrieval

Public HTTP(S) URLs pasted into a prompt can be fetched through a bounded read-only page reader
and stored under the current project. An optional ordered `WEB_FETCH_USER_AGENTS` JSON list lives
only in the ignored local `.env`; `WEB_FETCH_USER_AGENT_ATTEMPTS` chooses whether the reader uses
one or, at most, two profiles. By default, up to three independent supplied URLs are retrieved
concurrently, while their stored and prompt order remains the order in which the user supplied
them. The three-URL limit can be changed locally. No URLs or domains are configured or hardcoded.

Recoverable direct-read failures—HTTP 401/403/429/451, server errors, timeouts, DNS failure, or
pages without server-rendered readable text—may route the exact failed URL to one selected agent
whose provider declares compatible native search. This occurs only when research is enabled.
Unsafe URLs, unsupported content types, and ordinary not-found responses are not escalated.

Search-derived claims require provider-returned URL citations. The search agent must disclose
substitute sources and cannot claim it directly read a page that remained blocked. Uncited search
output is treated as no evidence and is not stored as an agent response. Direct snapshots and
search-derived responses retain separate provenance.

If every supplied page fails and no compatible search fallback is available—or the fallback
returns no cited evidence—the room records the retrieval outcome and skips all ordinary agent
turns. This prevents agents from spending tokens responding only to a source error. If at least
one supplied page was retrieved successfully, agents may still respond from that available
evidence while failures remain visible.

The direct reader still does not execute JavaScript, retain cookies, solve browser challenges, or
provide agents with arbitrary network access or unrestricted web-search discovery.

### User-approved Python

Agents may create Python files, but they cannot execute them. A run begins only when the user
selects a `.py` file and presses **RUN**. The project is copied into a separate runner container
with no external Docker network attachment, provider keys, database, Docker socket, or
host-project mount. Execution has bounded time, input, output, memory, files, and processes.

## Persistence

All writable application data is contained under:

```text
workspace/
```

This includes the SQLite database, runtime configuration, provider catalog, personas, shared
context, persona history and dormant identity-change records, projects, files, supplied-page
snapshots, and started-turn lifecycle records. Completed turns retain a bounded internal timing
trace and provider-reported token counts when available; generated bodies and source contents are
not copied into the trace. Seeded personas, shared context, and the provider catalog are copied
from `app/seed/` on first startup. The runtime configuration is created only after you save
provider and model selections in **SETUP**. Afterward, all workspace state is persistent.

SQLite schema changes are applied in place and rebuildable full-text indexes accelerate relevant
local and cross-project memory retrieval. If FTS5 is unavailable, the existing bounded lexical
selection remains active. Room execution also has high default ceilings
(`ROOM_MAX_PROVIDER_REQUESTS` and `ROOM_MAX_ELAPSED_SECONDS`) that stop pathological runs without
changing the normal three-agent flow. Diagnostics and recovery checkpoints persist by default;
setting `TURN_TRACE_RETENTION_DAYS` to a positive value prunes old completed or explicitly
resolved diagnostics without removing conversation messages or completed replay results.

State-changing agent tools receive deterministic operation identifiers. A completed file write
or persona proposal is not executed twice when a provider loop is replayed. If a restart leaves a
file write uncertain, the system verifies its requested content hash before deciding whether it
must write again. An interrupted persona update is stopped for user review instead of being
applied a second time automatically.

Room coordination, response execution, prompt construction, memory-context selection, and
checkpoint persistence live in separate internal modules. SQLite access retains one `Storage`
facade while schema migration, turns, files, memory, sources, projects, and messages are organized
by domain behind it. This changes maintainability and recovery behavior, not prompts, speaker
order, visible response wording, or the normal chat flow.

Back up `workspace/` to preserve the complete local state. Keep `.env` separate because it may
contain provider secrets.

## Security boundary

The default Compose setup:

- publishes the application only on `127.0.0.1`;
- runs both containers as non-root with read-only root filesystems;
- gives the application only `./workspace` as a writable host bind mount;
- gives the runner no external Docker network, provider secrets, database, or host-project
  mount;
- confines agent tools to project-relative paths and rejects traversal and symbolic links;
- treats project files and supplied web pages as untrusted model context.

These controls reduce risk but do not make arbitrary code or model output safe. The runner is
still an ordinary Docker container sharing the host kernel. Use a disposable VM or microVM for
hostile code or stronger isolation, and do not expose the application beyond localhost.

See [SECURITY.md](SECURITY.md) for the complete trust model and limitations.

## Development

To run only the application process without Docker:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export WORKSPACE_ROOT="$PWD/workspace"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-server-header --no-proxy-headers
```

This does not start the isolated runner, so **RUN** is unavailable unless a runner is provided
separately. The application process also has your normal user permissions.

Run the tests and development checks with:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest
ruff check app runner tests
pip-audit -r requirements.txt
```

## License

MIT. See [LICENSE](LICENSE).
