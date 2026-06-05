# DSG Viewer Design

Date: 2026-06-05

## Purpose

Build a local DSG inspection workbench for the DSG-SpatialQA project. The tool should help with three workflows in one interface:

- Debugging predicted DSG quality, including missing evidence, noisy relations, object location issues, and evidence readiness failures.
- Demonstrating DSG structure and QA evidence paths in a readable browser UI.
- Analyzing experiments by linking predicted/oracle graph deltas, QA errors, graph metrics, and evidence reports.

The first version is a read-only local web app. It does not modify graph artifacts, call external services, or run detector/simulator integrations.

## Chosen Approach

Use an Inspector Workbench layout:

- Left rail: data source status, mode tabs, search, and filters.
- Center panel: metrics cards, toolbar, and interactive graph canvas.
- Right rail: selected node/edge details, state history, oracle delta, linked QA cases, and diagnostics.

This layout is dense enough for debugging and analysis, while still supporting a clean demo mode by hiding advanced filters and focusing on selected evidence paths.

## Architecture

Add a local server entrypoint:

```bash
python scripts/serve_dsg_viewer.py --workspace handoffs/ai2thor-real-small
```

The server is local-only and deterministic:

- It binds to localhost by default.
- It reads only files under an explicit `--workspace` path or explicitly supplied artifact paths.
- It performs no network calls.
- It does not read current time or generate random output.
- It rejects paths outside the allowed project/workspace root.

Backend responsibilities:

- Load local JSON/JSONL artifacts.
- Validate known schema versions where existing validators are available.
- Normalize artifacts into one viewer payload.
- Serve the static viewer page and payload endpoints.

Frontend responsibilities:

- Render graph canvas, filters, metrics cards, and details panels.
- Keep graph selection, QA selection, and report diagnostics linked.
- Provide Debug, Demo, and Analysis views over the same payload.

## Data Inputs

The first version supports these artifact types:

- Predicted graph JSON, such as `outputs/predicted-dsg/predicted-graph.json`.
- Oracle graph JSON, optional but needed for deltas.
- QA dataset JSONL.
- QA eval report JSON.
- Graph eval report JSON.
- Predicted DSG evidence report JSON.

The server should accept either a workspace preset or explicit file arguments. A workspace preset should discover common current project paths under `handoffs/ai2thor-real-small`.

## Viewer Payload

The backend normalizes artifacts into a compact payload with these sections:

- `graph`: nodes, edges, object states, state histories, agent history, graph summary.
- `oracle`: optional oracle graph summary and predicted-to-oracle matches when graph eval is supplied.
- `qa`: cases, predictions/eval rows, success/failure status, target object ids, missing evidence when available.
- `metrics`: graph counts, relation counts, object recall, relation precision/recall/F1, evidence readiness.
- `diagnostics`: evidence gate checks, failed checks, source counts, and report digests.

The payload should preserve original artifact paths and digests so a user can trace any displayed item back to the saved file.

## Core Interactions

Graph interactions:

- Show nodes by type: object, room, region, state, agent.
- Show edges by relation.
- Filter by node type, label, relation, step range, and error category.
- Search node ids, labels, relation names, and QA case ids.
- Click a node or edge to populate the right detail rail.

QA interactions:

- Select a QA case from the linked QA list.
- Highlight target object nodes and evidence edges.
- Show missing graph evidence when available.
- Show prediction status, exact match status, semantic match status, and failure reason when available.

Comparison interactions:

- When oracle/graph eval data exists, mark matched, missing, and extra objects/relations.
- Show per-selection oracle match and delta details.
- Provide a Show Delta toggle in the center toolbar.

Mode tabs:

- Debug: filters and diagnostics are prominent.
- Demo: graph canvas and human-readable explanation are prominent.
- Analysis: metrics, QA table, and predicted/oracle deltas are prominent.

## Error Handling

The viewer should make artifact problems explicit instead of failing silently:

- Missing optional file: show a disabled panel with the missing path.
- Invalid schema: show the validator error and keep other valid panels usable.
- Path outside workspace: reject the request.
- Malformed JSON/JSONL: show file path and parsing error.
- Missing linkage fields: load the artifact but mark affected joins as unavailable.

## Testing

Use test-first implementation for production code.

Recommended tests:

- Path allowlist rejects files outside the workspace.
- Workspace preset resolves expected `ai2thor-real-small` artifact paths.
- Graph JSON normalization produces stable node/edge/state counts.
- QA/eval linkage associates object-location cases with target object ids.
- Evidence report normalization exposes readiness and failed checks.
- Missing optional artifacts do not block loading valid artifacts.
- CLI/server argument parsing rejects ambiguous or invalid inputs.

Full verification remains:

```bash
python scripts/verify.py
```

## Non-Goals For First Version

- No graph editing.
- No annotation persistence.
- No multi-user or remote deployment.
- No external AI, detector, simulator, or service calls.
- No large server-side layout engine.
- No replacement for graph eval or QA eval; the viewer only displays and links existing artifacts.

## Implementation Constraint

The first implementation should stay dependency-light: Python standard-library HTTP serving plus a static HTML/JavaScript viewer committed under the project. Do not add new package dependencies unless a failing test or concrete browser compatibility issue shows that the standard-library/static approach cannot meet the first-version requirements.
