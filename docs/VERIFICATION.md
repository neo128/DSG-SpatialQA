# Verification Record

## Modification Summary

- Created a minimal deterministic Python package for DSG-SpatialQA Lab.
- Implemented in-memory Dynamic Scene Graph nodes/edges/history for agents, objects, rooms, regions, states, actions, and events.
- Implemented deterministic 3D relation evaluation, GraphTool queries, SpatialQAEngine intents, and VLA anchor planner outputs.
- Added deterministic tabletop scene fixture plus JSON dict/string/file import and export helpers for reproducible experiments.
- Added deterministic scene fixture metadata manifests with tag filtering for experiment discovery without graph loading.
- Added structured `GraphQuery` support for deterministic node, edge, relation, visibility, text, and step-window filtering.
- Added direct `agent_location`, `object_location`, and `object_status` QA intents for agent pose, object pose, visibility, confidence, state evidence, current containment location, last-seen memory, and re-observation decisions.
- Added deterministic nearest-object distance diagnostics through `GraphTool.nearest_distances()`.
- Added candidate-constrained `nearest_object` QA support that reuses deterministic candidate filtering and returns selected distance plus stable candidate distance diagnostics.
- Added built-in deterministic candidate-constrained nearest-object evaluation coverage for retrieval and distance-diagnostic regressions.
- Added deterministic `label_candidates` QA support for semantic same-label candidate discovery with stable state evidence and ambiguity flags.
- Added deterministic `AgentPoseState` history, `STATE_CHANGED` agent pose state nodes/edges, scene IO round-trip support, and an `agent_history` QA intent.
- Added deterministic `GraphTool.agent_timeline()` and `agent_timeline` QA intent for explicit-step agent pose audits with per-step evidence edges.
- Added deterministic `GraphTool.relation_timeline()` and `relation_timeline` QA intent for explicit-step relation-edge audits with evidence edges.
- Added deterministic `GraphTool.recent_events()` for action/event nodes and step-window change evidence, with QA `recent_events` reusing the graph tool path.
- Added deterministic `GraphTool.world_state()` for current agent pose, object states, containment locations, and evidence IDs, with QA `world_state` reusing the graph tool path.
- Added `world_state` and `recent_events` QA intents for current scene snapshots and step-window event/change summaries.
- Added built-in deterministic world-state evaluation coverage for current dynamic scene state regressions.
- Added built-in deterministic foundation QA evaluation coverage for agent location, object location evidence, missing-object error diagnostics, object status, object history, and direct relative-relation regressions.
- Added built-in deterministic QA error-path evaluation coverage for missing-object object-location failures and reversed explicit-step scene-delta windows.
- Added built-in deterministic next-action validity QA evaluation coverage for stale-action regressions over dynamic scenes.
- Added a deterministic `graph_query` QA intent backed by `GraphTool.query_graph()` for structured node/edge retrieval through the QA layer.
- Added a deterministic `retrieve_subgraph` QA intent backed by `GraphTool.retrieve_subgraph()` for text-seeded graph retrieval through the QA layer.
- Added built-in deterministic graph query evaluation coverage for structured node/edge retrieval regressions.
- Added built-in deterministic retrieve_subgraph evaluation coverage for text-seeded graph retrieval regressions.
- Added deterministic explicit-step scene snapshots through `GraphTool.scene_snapshot()` and the `scene_snapshot` QA intent.
- Added built-in deterministic scene snapshot evaluation coverage for explicit-step state reconstruction regressions.
- Added deterministic explicit-step scene deltas through `GraphTool.scene_delta()` and the `scene_delta` QA intent.
- Added structured VLA planner validation details for re-observation and stale-action replan causes.
- Added deterministic VLA ambiguity details with same-label candidate pose, visibility, confidence, last-seen step, and re-observation status.
- Added deterministic `place_relative` validation that treats target pose as a planned anchor and returns `stale_reference_state` when reference-object movement invalidates the anchor.
- Added a deterministic scene fixture registry with static `tabletop`, dynamic `moved_mug`, `multi_room_rearrangement`, `needs_reobserve`, `ambiguous_mugs`, `ambiguous_plates`, and `relation_shift` fixtures.
- Added a deterministic evaluation harness for QA and VLA regression cases over scene fixtures.
- Added deterministic evaluation case metadata manifests for built-in and custom experiment discovery without case execution.
- Added scene fixture descriptions and tags to evaluation metadata manifests without loading scene graphs.
- Added baseline scene fixture descriptions and tags to stale-action evaluation metadata manifests.
- Added structured question copies to evaluation metadata manifests for QA dry-run auditability.
- Added deterministic evaluation mismatch diagnostics with path, reason, category, expected, and actual fields.
- Added deterministic QA and VLA runtime `error_category` diagnostics for non-empty evaluation errors, including missing objects, unsupported relations, ambiguity, invalid time windows, re-observation, and stale-state planner failures.
- Added deterministic suite/report `runtime_error_categories` aggregates with stable category counts and affected case names, included them in report comparison and suite digests, and added category-path drift differences for compact-report comparison.
- Added deterministic evaluation suite `selected_cases` summaries for experiment auditability.
- Added deterministic evaluation suite `failed_cases` summaries for faster failed regression triage.
- Added deterministic evaluation suite `breakdown` summaries grouped by case kind, QA question type, scene fixture, and tag.
- Added deterministic evaluation suite SHA-256 digests for reproducible experiment record comparison.
- Added deterministic VLA `vla_pick` and `vla_place_relative` evaluation cases for anchor planner regression coverage.
- Added built-in deterministic VLA error-path evaluation coverage for missing pick targets, missing place references, and unsupported place relations that must return structured planner errors without commands.
- Added deterministic VLA stale place-relative evaluation coverage for reference-object movement and `stale_reference_state` regressions.
- Added deterministic VLA needs-reobserve pick evaluation coverage for invisible low-confidence targets that must not produce commands.
- Added built-in deterministic VLA ambiguous-label pick evaluation coverage over the `ambiguous_mugs` scene fixture.
- Added built-in deterministic VLA ambiguous-reference place-relative evaluation coverage over the `ambiguous_plates` scene fixture.
- Added built-in deterministic QA label-candidate ambiguity evaluation coverage over the `ambiguous_mugs` scene fixture.
- Added label-based VLA pick evaluation cases for semantic target resolution and ambiguous-label regression coverage.
- Added label-based VLA place-relative planning and evaluation cases for semantic target/reference resolution.
- Added deterministic `GraphTool.object_timeline()` and `object_timeline` QA intent for explicit-step object state audits with pose, visibility, confidence, location, and per-step evidence edges.
- Added built-in deterministic timeline evaluation cases for `agent_timeline` and `object_timeline` QA regressions.
- Added built-in deterministic relation timeline evaluation coverage for explicit relation-edge regression audits.
- Added deterministic `GraphTool.reobserve_targets()` and `reobserve_targets` QA intent for current invisible low-confidence object discovery.
- Added built-in deterministic `needs_reobserve_targets` evaluation coverage over the `needs_reobserve` scene fixture.
- Added a deterministic `scene_delta` evaluation case and tag-based evaluation suite filtering for focused experiment runs.
- Added deterministic evaluation filtering by case kind for QA-only and VLA-only experiment runs.
- Added deterministic evaluation filtering by QA question type for intent-focused experiment runs and metadata discovery.
- Added deterministic explicit-name evaluation filtering for fixed built-in and custom experiment slices while preserving caller order.
- Added deterministic custom `EvaluationCase` runners for local experiment cases outside the built-in registry.
- Added deterministic caller-supplied evaluation scene loaders for custom in-memory experiment scenes.
- Added deterministic multi-room rearrangement fixture and evaluation coverage for kitchen-to-pantry relocation, region containment changes, low-confidence occlusion, re-observation QA, scene deltas, and recent-event windows.
- Added deterministic relation-shift fixture and evaluation coverage for explicit-step relation changes from `LEFT_OF` to `RIGHT_OF`.
- Improved relation geometry with bbox surface-distance `NEAR`, support-overlap `ON`, and agent-yaw egocentric coverage.
- Added `GraphTool.update_spatial_relations()` to deterministically infer current spatial relation edges from object bboxes and agent pose for explicit caller-supplied steps.
- Added a deterministic `ObservationIngestor` for structured mock perception frames with explicit steps, optional agent pose, room/region nodes, object observations, and optional relation inference.
- Added `scripts/verify.py` as a deterministic local verification entrypoint for install, lint, typecheck, tests, build, and built-in evaluation suite checks.
- Extended the local verification typecheck gate to cover `scripts` as well as `src` and `tests`, and added the package `py.typed` marker for typed CLI imports.
- Added `scripts/check_determinism.py` as a deterministic local source scanner for current-time, random, network, and external model client boundaries, and wired it into the local verification gate.
- Added README quickstart, development baseline, MVP capability, project boundary, and roadmap sections for handoff-ready onboarding.
- Added deterministic evaluation report helpers for compact benchmark metrics, including case counts, pass/failure rates, grouped count/rate metrics, failed-case mismatch-path summaries, report-level failure-path summaries, failure-reason summaries, stable failure-category summaries, stable JSON payloads, and explicit-path report saving.
- Added `scripts/evaluate.py` as a deterministic offline evaluation report CLI with exact-name, tag, kind, question-type, stdout, and explicit local file output support.
- Added deterministic evaluation report loading and comparison helpers plus CLI `--compare-report` support to detect current-code drift from compact reports, including nested summary, failed-case, metric-path, breakdown, runtime-error category, failure-reason, failure-category, and failure-path differences.
- Added deterministic evaluation manifest helpers and CLI `--manifest` output for filtered case manifests, fixture manifests, coverage counts, and digests without running cases.
- Added deterministic evaluation manifest loading and validation helpers plus CLI `--validate-manifest` support for explicit local manifest files, including digest and coverage summary consistency checks with nested coverage-path differences.
- Added deterministic evaluation bundle helpers and CLI `--bundle` output for reproducible local artifacts that include filters, scene fixture manifests, evaluation case manifests, coverage counts, full suite results, compact reports, and digests.
- Added deterministic evaluation bundle loading and validation helpers plus CLI `--validate-bundle` support for explicit local bundle files, including report consistency, suite-backed case manifest metadata consistency, case-backed scene fixture metadata consistency, and coverage summary checks with nested path differences and stable compact-report failed-case paths.
- Added deterministic evaluation bundle comparison helpers plus CLI `--compare-bundle` support to detect current-code drift in suite digest, compact report, coverage, case manifest, and scene fixture manifest, including nested report, coverage, and manifest metadata path differences.
- Added deterministic evaluation manifest comparison helpers plus CLI `--compare-manifest` support to detect current-code metadata drift without running evaluation cases, including nested coverage and manifest metadata path differences.
- Added deterministic evaluation CLI invalid artifact diagnostics for explicit local report, manifest, and bundle validation/comparison, returning non-zero structured JSON with `valid: false` instead of tracebacks.
- Added deterministic graph digest and graph summary helpers, including object visibility, low-confidence, re-observation candidate, object-label, node-type, and edge-relation counts, plus `scripts/scene.py` for explicit scene fixture graph export and explicit local graph file validation.
- Added deterministic scene graph fixture comparison helpers plus CLI support to detect drift between an explicit local graph JSON file and the current built-in fixture digest/summary, including nested summary-path differences.
- Added deterministic scene CLI invalid graph diagnostics for explicit local validation and fixture comparison, returning non-zero structured JSON with `valid: false` instead of tracebacks.
- Added a GitHub Actions workflow that runs the same local `python scripts/verify.py` gate on pushes and pull requests.
- Added deterministic tests for spatial memory, dynamic graph updates, graph retrieval, nearest-object distance diagnostics, spatial QA, VLA planning, ambiguity, stale actions, and re-observation.
- Added deterministic tests for the local verification entrypoint command order, install skipping, failure short-circuiting, package type marker declaration, and deterministic boundary scan gate.
- Added deterministic tests for the local determinism scanner's clean-file result, blocked-token diagnostics, and generated metadata skipping.
- Added deterministic tests for evaluation mismatch categories, expanded grouped report metrics, failed-case mismatch-path summaries, report-level failure-path summaries, failure-reason summaries, failure-category summaries, stable JSON serialization, and explicit report file writes.
- Added deterministic tests for report-vs-current comparison success, summary/failed-case/metric/breakdown path drift diagnostics, runtime-error category drift diagnostics, failure-reason/category/path drift diagnostics, and CLI comparison exit status.
- Added deterministic tests for the offline evaluation report CLI filters, explicit report file writes, and invalid report artifact diagnostics.
- Added deterministic tests for evaluation manifest JSON stability, explicit manifest file writes, coverage summaries, manifest validation coverage-path diagnostics, invalid manifest artifact diagnostics, and CLI manifest output/validation.
- Added deterministic tests for manifest-vs-current metadata comparison success, coverage-path drift diagnostics, case-manifest metadata path diagnostics, invalid manifest artifact diagnostics, and CLI comparison exit status.
- Added deterministic tests for evaluation bundle JSON stability, explicit bundle file writes, coverage summaries, and CLI bundle output.
- Added deterministic tests for explicit bundle loading, validation success, tampered digest, report-path diagnostics, failed-case report-path diagnostics, case-manifest metadata diagnostics, scene-fixture metadata diagnostics, coverage-path diagnostics, invalid bundle artifact diagnostics, and CLI bundle validation exit status.
- Added deterministic tests for bundle-vs-current comparison success, report-path drift diagnostics, coverage-path drift diagnostics, invalid bundle artifact diagnostics, and CLI comparison exit status.
- Added deterministic tests for graph digest/summary helpers, object-state summary counts, and the scene export/validate/compare CLI, including invalid graph diagnostics and summary-path drift diagnostics.
- Added deterministic tests for dynamic relation-shift fixture metadata and relation timeline evaluation coverage.
- Added deterministic tests for foundation QA evaluation metadata, QA error-path evaluation coverage, temporal error-path filtering/execution, label-candidate ambiguity filtering/execution, VLA error-path/ambiguity diagnostics, runtime error categories and aggregates, and relation manifest coverage.
- Added deterministic tests that keep the CI workflow pointed at the local verification entrypoint instead of duplicating gate commands.

## Verification Environment

- Workspace: `/home/user/Code/DSG-SpatialQA`
- OS: Linux `baai` `6.8.0-111-generic` x86_64
- Python: `Python 3.13.12`
- pip: `pip 26.0.1` from `/home/user/miniconda3/lib/python3.13/site-packages/pip`

## Commands and Results

| Category | Command | Result |
| --- | --- | --- |
| All gates | `python scripts/verify.py` | Passed. Runs editable dev install, lint, typecheck, determinism scan, unit tests, package build, and built-in evaluation suite in order. |
| Install | `python -m pip install -e ".[dev]"` | Passed. Editable package built and installed as `dsg-spatialqa-lab-0.1.0`; dev tools resolved from `pyproject.toml`. |
| Lint | `python -m ruff check .` | Passed: `All checks passed!` |
| Typecheck | `python -m mypy src tests scripts` | Passed: `Success: no issues found in 29 source files`. |
| Determinism scan | `python scripts/check_determinism.py` | Passed: `{"matches": [], "valid": true}`. |
| Test | `python -m pytest -q` | Passed: `232 passed`. |
| Build | `python -m build` | Passed. Built `dsg_spatialqa_lab-0.1.0.tar.gz` and `dsg_spatialqa_lab-0.1.0-py3-none-any.whl`; build artifacts were removed after verification because they are reproducible. |
| Evaluation suite | `python -c "from dsg_spatialqa_lab import run_evaluation_suite; suite = run_evaluation_suite(); print(suite['summary']); print(suite['digest'])"` | Passed: `34` selected cases, `34` passed, `0` failed; digest `ee1056ed7be66acfbe0c2b15a250814ad674e187e453b664a20046be239cd057`. |

## Skips or Substitutions

- None. The default minimal Python project commands were used.
- Runtime code depends only on the Python standard library. `pytest`, `ruff`, `mypy`, and `build` are dev-only dependencies recorded in `pyproject.toml` to support verification.
- `scripts/verify.py` shells out to local Python module commands only; it does not add runtime dependencies or service integrations.
- `scripts/check_determinism.py` uses only the Python standard library and scans explicit local project paths; it skips generated package metadata and build/cache directories.
- CI delegates to `scripts/verify.py`; workflow maintenance should update the local verifier first.
- Evaluation report saving, loading, and comparison read or write only explicit caller-supplied local paths.
- Evaluation bundle saving writes only to an explicit caller-supplied local path.
- Evaluation manifest loading, validation, and comparison read only explicit caller-supplied local paths.
- Evaluation bundle loading, validation, and comparison read only explicit caller-supplied local paths.
- Scene graph export, validation, and comparison read or write only explicit caller-supplied local paths.
- `scripts/evaluate.py` shells out to no services; it only runs local deterministic evaluation code, writes reports, manifests, or bundles when an explicit local path is provided, and validates or compares only explicit local report, manifest, or bundle files. Invalid explicit report, manifest, or bundle files return structured local JSON diagnostics and a non-zero status.
- `scripts/scene.py` shells out to no services; it only exports built-in deterministic fixtures, validates explicit local graph JSON files, or compares explicit graph JSON files with current built-in fixture baselines. Invalid explicit graph files return structured local JSON diagnostics and a non-zero status.

## Known Limits and Follow-up Work

- MVP is intentionally in-memory only; there is no persistence layer.
- VLM/LLM, robot, simulator, and network integrations are not implemented and are represented by deterministic local logic only.
- Relation thresholds are configurable but simple; future work can add richer geometry, semantic priors, and sensor evidence models.
- `ObservationIngestor` accepts structured local observations only; real sensor, simulator, or model adapters remain intentionally out of scope.
- VLA planner returns structured skill anchors only; it does not perform motion planning or robot control.
