# Verification Record

## Modification Summary

- Created a minimal deterministic Python package for DSG-SpatialQA Lab.
- Implemented in-memory Dynamic Scene Graph nodes/edges/history for agents, objects, rooms, regions, states, actions, and events.
- Implemented deterministic 3D relation evaluation, GraphTool queries, SpatialQAEngine intents, and VLA anchor planner outputs.
- Added deterministic tabletop scene fixture plus JSON dict/string/file import and export helpers for reproducible experiments.
- Added deterministic scene fixture metadata manifests with schema version, digest, and tag filtering for experiment discovery without graph loading.
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
- Added deterministic `GraphTool.current_room()` for room-level containment resolution with path and evidence edge diagnostics.
- Added deterministic `object_room` QA intent for room-level containment answers with room id/label, path nodes, and evidence edges.
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
- Added built-in deterministic multi-room `object_room` evaluation coverage for relocated-object room resolution.
- Added deterministic relation-shift fixture and evaluation coverage for explicit-step relation changes from `LEFT_OF` to `RIGHT_OF`.
- Improved relation geometry with bbox surface-distance `NEAR`, support-overlap `ON`, and agent-yaw egocentric coverage.
- Added `GraphTool.update_spatial_relations()` to deterministically infer current spatial relation edges from object bboxes and agent pose for explicit caller-supplied steps.
- Added a deterministic `ObservationIngestor` for structured mock perception frames with explicit steps, optional agent pose, room/region nodes, object observations, and optional relation inference.
- Added deterministic `SceneObservation` JSON/save/load helpers for stable offline mock perception input artifacts with explicit caller-supplied local paths.
- Added deterministic `SceneObservation` sequence JSON/save/load helpers for stable ordered multi-frame mock perception input artifacts with explicit step lists and observation counts.
- Added deterministic observation sequence digests plus observation ingest report digest, saving, loading, validation, and comparison helpers. Report validation checks the report digest, explicit input and graph paths, plus nested graph report path consistency. `scripts/observations.py` supports explicit local observation sequence ingest, graph JSON export, ingest report validation, current-code re-ingest drift comparison, and exported graph-file drift comparison with stable sequence digest, graph-file summary, and per-step result difference paths.
- Added `scripts/verify.py` as a deterministic local verification entrypoint for install, lint, typecheck, tests, build, and built-in evaluation suite checks.
- Extended the local verification typecheck gate to cover `scripts` as well as `src` and `tests`, and added the package `py.typed` marker for typed CLI imports.
- Added `scripts/check_determinism.py` as a deterministic local source scanner for current-time, random, network, and external model client boundaries, and wired it into the local verification gate.
- Added README quickstart, development baseline, MVP capability, project boundary, and roadmap sections for handoff-ready onboarding.
- Added deterministic evaluation report helpers for compact benchmark metrics, including case counts, pass/failure rates, grouped count/rate metrics, failed-case mismatch-path summaries, report-level failure-path summaries, failure-reason summaries, stable failure-category summaries, stable JSON payloads, compact report digest validation, and explicit-path report saving.
- Added deterministic compact-report `evidence_metrics` for evidence-node, evidence-edge, VLA command evidence, evidence-covered case, and average evidence-item counts, including grouped summaries by kind, question type, scene fixture, and tag.
- Added deterministic compact-report `case_digests` for per-case SHA-256 result summaries with case name, kind, question type, scene fixture, and pass status.
- Added `scripts/evaluate.py` as a deterministic offline evaluation report CLI with exact-name, tag, kind, question-type, stdout, explicit local file output support, and `--list-cases` case metadata discovery with stable listing digests, validation, and current-metadata comparison without running evaluation cases.
- Added deterministic evaluation report loading, validation, and comparison helpers plus CLI `--validate-report` and `--compare-report` support to detect report artifact tampering and current-code drift from compact reports, including nested summary, failed-case, metric-path, breakdown, runtime-error category, failure-reason, failure-category, and failure-path differences.
- Added deterministic compact-report comparison checks for `evidence_metrics` drift with stable nested evidence metric paths.
- Added deterministic compact-report comparison checks for per-case digest drift with stable case digest paths.
- Added deterministic evaluation manifest helpers and CLI `--manifest` output for filtered case manifests, fixture manifests, coverage counts, public manifest digest recomputation, and digests without running cases.
- Added deterministic evaluation manifest loading and validation helpers plus CLI `--validate-manifest` support for explicit local manifest files, including digest and coverage summary consistency checks with nested coverage-path differences.
- Added deterministic evaluation bundle helpers and CLI `--bundle` output for reproducible local artifacts that include filters, scene fixture manifests, evaluation case manifests, coverage counts, full suite results, compact reports, suite digests, and bundle artifact digests.
- Added deterministic evaluation bundle loading and validation helpers plus CLI `--validate-bundle` support for explicit local bundle files, including bundle digest tamper checks, report consistency, suite-backed case manifest metadata consistency, case-backed scene fixture metadata consistency, and coverage summary checks with nested path differences and stable compact-report failed-case paths.
- Added deterministic evaluation bundle comparison helpers plus CLI `--compare-bundle` support to detect current-code drift in suite digest, bundle artifact digest, compact report, coverage, case manifest, and scene fixture manifest, including nested report, coverage, and manifest metadata path differences.
- Added deterministic evaluation manifest comparison helpers plus CLI `--compare-manifest` support to detect current-code metadata drift without running evaluation cases, including nested coverage and manifest metadata path differences.
- Added deterministic evaluation CLI invalid artifact diagnostics for explicit local report, manifest, and bundle validation/comparison, returning non-zero structured JSON with `valid: false` instead of tracebacks.
- Added deterministic scene CLI `--list-fixtures` fixture metadata discovery with schema version, digest, repeated `--tag` filtering, optional explicit local `--output`, and no graph loading.
- Added deterministic scene fixture manifest JSON and save helpers so Python handoffs can emit the same stable explicit-path fixture manifest artifacts as the scene CLI.
- Added deterministic scene fixture manifest loading and validation helpers plus CLI `--validate-fixture-manifest` support for explicit local fixture metadata manifest files, including schema version, digest, and fixture-count consistency checks.
- Added deterministic scene fixture manifest comparison helpers plus CLI `--compare-fixture-manifest` support to detect current-code metadata drift from explicit local fixture metadata manifest files, including stable fixture metadata `differences` paths.
- Added deterministic graph digest and graph summary helpers, including object visibility, low-confidence, re-observation candidate, current-location, current-room, object-label, node-type, and edge-relation counts, plus `scripts/scene.py` for explicit scene fixture graph export and explicit local graph file validation.
- Added deterministic graph report helpers plus scene CLI `--report` support so graph export/validation payloads can be saved as stable explicit-path local artifacts with report schema version, graph digest, report digest, and summary.
- Added deterministic graph report loading, validation, and comparison helpers plus scene CLI `--validate-report` and `--compare-report` support to audit explicit graph report artifacts against the current built-in fixture baseline, including report digest tamper checks and stable summary drift `differences` paths.
- Added deterministic graph report-to-file comparison helpers plus scene CLI `--compare-report-graph` support to audit explicit graph report artifacts against caller-supplied graph JSON files, including digest drift and stable summary drift `differences` paths.
- Added deterministic scene graph fixture comparison helpers plus CLI support to detect drift between an explicit local graph JSON file and the current built-in fixture digest/summary, including nested summary-path differences.
- Added deterministic scene CLI invalid graph diagnostics for explicit local validation and fixture comparison, returning non-zero structured JSON with `valid: false` instead of tracebacks.
- Added a GitHub Actions workflow that runs the same local `python scripts/verify.py` gate on pushes and pull requests.
- Added deterministic tests for spatial memory, dynamic graph updates, graph retrieval, current-room and object-room resolution diagnostics, nearest-object distance diagnostics, spatial QA, VLA planning, ambiguity, stale actions, and re-observation.
- Added deterministic tests for `SceneObservation` JSON round-trips, ordered observation sequence JSON round-trips and digests, explicit observation file saves/loads, observation ingestion boundaries, observation sequence CLI graph/report export, observation ingest report digest, save/load, explicit artifact path validation, validation/comparison, invalid observation artifact diagnostics, and observation ingest report sequence, graph-file, and output drift diagnostics.
- Added deterministic tests for the local verification entrypoint command order, install skipping, failure short-circuiting, package type marker declaration, and deterministic boundary scan gate.
- Added deterministic tests for the local determinism scanner's clean-file result, blocked-token diagnostics, and generated metadata skipping.
- Added deterministic tests for evaluation mismatch categories, expanded grouped report metrics, failed-case mismatch-path summaries, report-level failure-path summaries, failure-reason summaries, failure-category summaries, stable JSON serialization, compact report digest validation, and explicit report file writes.
- Added deterministic tests for report-vs-current comparison success, summary/failed-case/case-digest/metric/evidence-metric/breakdown path drift diagnostics, runtime-error category drift diagnostics, failure-reason/category/path drift diagnostics, and CLI validation/comparison exit status.
- Added deterministic tests for the offline evaluation report CLI filters, lightweight case metadata listing with stable digests, explicit listing validation/comparison, explicit report file writes, and invalid report artifact diagnostics.
- Added deterministic tests for evaluation manifest JSON stability, explicit manifest file writes, public manifest digest recomputation, coverage summaries, manifest validation coverage-path diagnostics, invalid manifest artifact diagnostics, and CLI manifest output/validation.
- Added deterministic tests for manifest-vs-current metadata comparison success, coverage-path drift diagnostics, case-manifest metadata path diagnostics, invalid manifest artifact diagnostics, and CLI comparison exit status.
- Added deterministic tests for evaluation bundle JSON stability, explicit bundle file writes, coverage summaries, and CLI bundle output.
- Added deterministic tests for explicit bundle loading, validation success, tampered suite and bundle digests, report-path diagnostics, failed-case report-path diagnostics, case-manifest metadata diagnostics, scene-fixture metadata diagnostics, coverage-path diagnostics, invalid bundle artifact diagnostics, and CLI bundle validation exit status.
- Added deterministic tests for bundle-vs-current comparison success, suite and bundle digest comparison diagnostics, report-path drift diagnostics, coverage-path drift diagnostics, invalid bundle artifact diagnostics, and CLI comparison exit status.
- Added deterministic tests for graph digest/summary/report helpers, graph report digest tamper checks, graph report explicit file writes, graph report loading/validation/comparison, graph report-to-file comparison, object-state, current-location, current-room summary counts, scene fixture manifest digests, fixture manifest JSON/save helpers, fixture manifest loading/validation/comparison, and the scene fixture metadata listing/export/validate/compare CLI, including graph report summary drift diagnostics, graph report-to-file graph drift diagnostics, fixture manifest metadata drift diagnostics, invalid fixture manifest diagnostics, invalid graph diagnostics, and summary-path drift diagnostics.
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
| Typecheck | `python -m mypy src tests scripts` | Passed: `Success: no issues found in 31 source files`. |
| Determinism scan | `python scripts/check_determinism.py` | Passed: `{"matches": [], "valid": true}`. |
| Test | `python -m pytest -q` | Passed: `279 passed`. |
| Build | `python -m build` | Passed. Built `dsg_spatialqa_lab-0.1.0.tar.gz` and `dsg_spatialqa_lab-0.1.0-py3-none-any.whl`; build artifacts were removed after verification because they are reproducible. |
| Evaluation suite | `python -c "from dsg_spatialqa_lab import run_evaluation_suite; suite = run_evaluation_suite(); print(suite['summary']); print(suite['digest'])"` | Passed: `35` selected cases, `35` passed, `0` failed; digest `717a61fd8757270fedb454f99c41b6a801e0ee237428bb839d5d8452bc4421ee`. |

## Skips or Substitutions

- None. The default minimal Python project commands were used.
- Runtime code depends only on the Python standard library. `pytest`, `ruff`, `mypy`, and `build` are dev-only dependencies recorded in `pyproject.toml` to support verification.
- `scripts/verify.py` shells out to local Python module commands only; it does not add runtime dependencies or service integrations.
- `scripts/check_determinism.py` uses only the Python standard library and scans explicit local project paths; it skips generated package metadata and build/cache directories.
- CI delegates to `scripts/verify.py`; workflow maintenance should update the local verifier first.
- Evaluation report saving, loading, validation, and comparison read or write only explicit caller-supplied local paths. Compact report validation checks the saved report digest before handoff.
- Evaluation case listing, manifest, and bundle saving write only to explicit caller-supplied local paths, create parent directories when needed, and use deterministic digest helpers for handoff verification.
- Evaluation case listing loading, validation, and comparison read only explicit caller-supplied local paths. Listing validation checks the saved listing digest and case count before handoff; listing comparison regenerates current metadata without running evaluation cases.
- Evaluation manifest loading, validation, and comparison read only explicit caller-supplied local paths.
- Evaluation bundle loading, validation, and comparison read only explicit caller-supplied local paths. Bundle validation checks the saved bundle digest before handoff.
- Scene graph fixture listing, fixture manifest validation/comparison, graph export, graph report validation/comparison, graph report-to-file comparison, graph validation, and graph comparison read or write only explicit caller-supplied local paths. Graph report validation also checks the saved report digest before handoff.
- Observation sequence CLI ingestion reads only an explicit caller-supplied local sequence JSON file, writes graph JSON only to an explicit caller-supplied local path, writes the ingest report only when an explicit local `--report` path is supplied, and validates or compares only explicit local observation ingest report artifacts. Python observation ingest report saving/loading also uses explicit caller-supplied local paths. Observation ingest report validation checks the report digest, explicit input and graph paths, plus nested graph report path consistency. Comparison re-ingests only the explicit sequence path recorded in the report, reads only the explicit graph path recorded in the report, and compares saved/current sequence plus graph-file digests.
- `scripts/evaluate.py` shells out to no services; it only runs local deterministic evaluation code, writes reports, manifests, or bundles when an explicit local path is provided, and validates or compares only explicit local report, manifest, or bundle files. Invalid explicit report, manifest, or bundle files return structured local JSON diagnostics and a non-zero status.
- `scripts/scene.py` shells out to no services; it only lists built-in scene fixture metadata manifests, validates or compares explicit local fixture metadata manifest files, exports built-in deterministic fixtures, validates or compares explicit local graph report files, compares explicit graph report files to explicit graph JSON files, validates explicit local graph JSON files, or compares explicit graph JSON files with current built-in fixture baselines. Fixture metadata listing writes only to an explicit local `--output` path when one is supplied and does not load graph objects. Structured stdout payloads are also written only when an explicit local `--report` path is supplied. Invalid or drifted explicit fixture manifest, graph report, or graph files return structured local JSON diagnostics and a non-zero status.

## Known Limits and Follow-up Work

- MVP is intentionally in-memory only; there is no persistence layer.
- VLM/LLM, robot, simulator, and network integrations are not implemented and are represented by deterministic local logic only.
- Relation thresholds are configurable but simple; future work can add richer geometry, semantic priors, and sensor evidence models.
- `ObservationIngestor` accepts structured local observations only; `SceneObservation` JSON helpers read or write only explicit caller-supplied local paths. Real sensor, simulator, or model adapters remain intentionally out of scope.
- VLA planner returns structured skill anchors only; it does not perform motion planning or robot control.
