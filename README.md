# Motif Feedback

![⟁](https://aesxarg.net/taste-as-system-sm.jpg)

Motif Feedback is a local-first, single-user application for user-guided conversations with
three persistent agent personas in one conversational room. It is designed for long-running
thematic work that benefits from distinct perspectives, continuity across returns, and
revisiting ideas over time.

Each agent has an editable persona, a protected core motif, and separately configurable access
to a hosted or local OpenAI-compatible model. Conversations, continuity records, sources, and
project files persist under `workspace/`.

The agents respond only to user-initiated turns; the application does not schedule background
work or allow them to operate independently.

This is a personal experimental system for localhost. It is not a production multi-user
service.

## Motif-centered continuity

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

Open:

```text
http://127.0.0.1:8000
```

In **SETUP**, select a provider and model for each agent.

On Linux, run the following first if the containers cannot write to `./workspace`:

```bash
./scripts/init-linux.sh
```

To rebuild both services while developing:

```bash
docker compose -f compose.yaml -f compose.dev.yaml up -d --build runner app
```

To stop the system:

```bash
docker compose down
```

## Providers

Each agent can use a different provider and model. The persistent provider catalog is editable
under **SETUP → PROVIDER CATALOG** and supports OpenAI-compatible `/chat/completions`
endpoints, including compatible local servers.

From Docker, a model server running on the host can usually be reached at:

```text
http://host.docker.internal:<port>/v1
```

Provider keys remain in `.env`; they are not stored in the catalog or returned to the browser.
The available configuration values and defaults are documented in `.env.example`.

Model inference is local only when the selected provider is local. A hosted or custom provider
receives the relevant system prompt, persona, conversation, continuity, and project context for
that request.

## Projects and tools

Each project keeps its own conversation, files, supplied-page snapshots, and detailed agent
continuity records. Raw response-beat records remain visible in the inspector, while prompt
context groups beats from one agent turn, favors relevant returns, and uses compact event
references where the room transcript already contains the same exchange. Compact,
provenance-labelled returns may be shown to the same agent in another project as provisional
context.

Agents can list, read, search, create, and revise permitted project files. Their file tools are
confined to the current project folder:

```text
workspace/projects/<project>/
```

An agent may revise a file it created. Editing another agent's file requires the user to enable
sharing for that specific file. Agents cannot overwrite uploaded user files, delete files or
projects, run a shell, install packages, or control Docker.

Public HTTP(S) URLs pasted into a prompt can be fetched through a bounded read-only page reader
and stored under the current project. The application does not provide agents with arbitrary
network access or general web-search discovery.

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
context, persona history and dormant identity-change records, projects, files, and supplied-page
snapshots. Seeded personas, shared context, and the provider catalog are copied from `app/seed/`
on first startup. The runtime configuration is created only after you save provider and model
selections in **SETUP**. Afterward, all workspace state is persistent.

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
