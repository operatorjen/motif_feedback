# Analytics and debugging

The analytics page shows how Motif Feedback behaves over time. It describes activity, runtime,
context exposure, motif recurrence, and user review; it is not an automatic quality score.

Open **ANALYTICS / DEBUG** in the main masthead or visit
`http://127.0.0.1:8000/analytics`. Use the project selector to narrow the data.

## Coverage

Check coverage before interpreting a graph:

| Measurement | Meaning |
|---|---|
| Agent responses | Stored agent messages |
| Project memory events | Project-scoped continuity records |
| Durable room turns | Turns with a persisted lifecycle and runtime trace |
| Instrumented agent beats | Provider attempts with a prompt manifest |
| Context exposures | Context records included in those manifests |
| Observer-owned motifs | Motifs retained by their observing agent |
| Feedback events | Append-only user review actions |

Prompt instrumentation starts with turns run after this feature was installed. Older messages,
memories, turns, and motifs still appear, but missing historical prompt context remains unknown.

## Dashboard

| Section | What it shows | Main caution |
|---|---|---|
| Agent responses over time | Daily responses by agent | Activity is not quality |
| Agent comparison | Response count, average length, feedback, and speaking position | Sequence can affect results |
| Efficiency and reliability | Turn status, duration, provider requests, and tokens | Providers may omit token data |
| Prompt context exposure | Context types included in each agent prompt | Exposure does not prove causation |
| Motif return exposure | Whether an observed motif appeared in the prompt | “Unprompted” does not prove discovery |
| Motif lifecycle | Motif status by observing agent | Status remains user-governed |
| Project activity | Agent-response volume by project | Large projects can dominate |
| Recent response feedback | User review of recent responses | Feedback is subjective and sparse |

A response beat counts separately when an agent continues the same turn. Duration covers the
room workflow, not only model inference. Speaking position matters because later agents can see
earlier responses from the same turn.

## Prompt context

Each instrumented beat records which sources entered the prompt:

| Context kind | Source |
|---|---|
| Recent messages | Existing room transcript |
| Same-turn responses | Earlier responses in the current turn |
| Project memory | Continuity from the current project |
| Cross-project memory | Provisional continuity from another project |
| Own motifs | Motifs owned by the responding observer |
| Other observers’ motifs | Supported or active motifs owned by another observer |
| Pattern checkpoints | Established motif sequences |
| Web sources | User-supplied sources |
| Role signals | Bounded scripted role signals |

The manifest stores identifiers, project, prompt section, rank, selection reason, version hash,
and estimated size. It does not copy source text or the assembled prompt. Prompt template,
persona revision, and context selector hashes distinguish runs made under different conditions.

## Motif return exposure

| Category | Meaning |
|---|---|
| Prompted | The observed motif was in that beat’s context manifest |
| Unprompted | The beat was instrumented, but the motif was not in its manifest |
| Pre-instrumentation | No matching manifest exists |

These categories show exposure, not causation. A user message or related context can still suggest
an “unprompted” motif.

## Feedback

| Label | Meaning |
|---|---|
| Useful difference | Added a distinct lens or move |
| Repetitive | Repeated material already in the room |
| Off-lens | Did not speak convincingly from the intended perspective |
| Unsupported | Made an important claim without adequate evidence |

Selecting an active label again deactivates it by appending another event. Feedback is user-owned
and does not alter prompts, personas, memory, motifs, or speaking order.

## Tuning workflow

1. Choose one question.
2. Filter to one project and check coverage.
3. Add feedback while reading responses in context.
4. Change one condition: prompt, persona, model, order, or context selection.
5. Collect new turns, then compare the same measurements.

Useful comparisons include speaking position against repetition, context size against tokens or
duration, and prompted against unprompted motif returns. Do not collapse the dashboard into one
score: cheaper, shorter, more agreeable, or more recurrent is not automatically better.

## Storage and API

Analytics stays in the local SQLite database under `workspace/`:

| Table | Contents |
|---|---|
| `agent_prompt_runs` | One manifest per attempted agent beat |
| `context_exposures` | Context sources associated with a prompt run |
| `interaction_feedback_events` | Append-only review actions |

The dashboard also reads existing messages, memory, motifs, motif events, and turn traces.
Deleting a project deletes its analytics rows.

```text
GET  /api/analytics
GET  /api/analytics?project_id=<project-id>
POST /api/analytics/feedback
```

The feedback endpoint requires the current session token.
