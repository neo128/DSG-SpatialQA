# DSG-SpatialQA Lab Validation Platform Development Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve DSG-SpatialQA Lab from a deterministic DSG-SpatialQA core into a validation platform for simulator episodes, oracle/predicted DSGs, benchmarks, baselines, metrics, active tasks, and review dashboards.

**Architecture:** Preserve the current deterministic in-memory core and add optional adapter layers around it. Build the oracle benchmark loop first, then graph construction evaluation, predicted DSG skeletons, error attribution, active EQA, and dashboard tooling.

**Tech Stack:** Python package under `src/dsg_spatialqa_lab`, standard-library runtime by default, optional simulator/dashboard extras, deterministic JSON/JSONL artifacts, CLI scripts, pytest, ruff, mypy, and `scripts/verify.py`.

---

## Current State

The project already has:

- `DynamicSceneGraph`, `GraphTool`, `SpatialQAEngine`, and `VLAAnchorPlanner` deterministic local core.
- In-memory graph structures, object state history, relation edges, QA intents, and VLA pick/place-relative prototypes.
- Deterministic episode JSONL, oracle graph building, AI2-THOR mock adapter
  boundary, extended relation support, and oracle QA JSONL generation.
- `scripts/verify.py` local verification gate covering install, ruff, mypy, determinism scan, pytest, build, and evaluation suite.

## Current Gaps

1. Missing AI2-THOR / Habitat environment adapters.
2. Missing unified episode JSONL data protocol.
3. Missing simulator metadata to oracle DSG builder.
4. Missing large-scale 1000+ QA benchmark assembly and QA prediction metrics.
5. Missing VLM-only / caption memory / graph-text / graph-tool baseline runner.
6. Missing real experiment metrics: QA accuracy, relation F1, evidence consistency, task success, SPL/action cost.
7. Missing predicted DSG: segmentation, depth projection, object fusion, tracking, relation inference.
8. Missing oracle vs predicted graph alignment evaluation and error attribution.
9. Missing active EQA / navigation / next-best-view interactive tasks.
10. Missing dashboard for graph, trajectory, RGB/depth, answers, evidence, and error-source human review.

## Development Goal

Keep the deterministic core stable while gradually adding the validation platform capabilities required by the attached proposal. Prioritize the oracle pipeline and benchmark loop first, then expand into predicted DSG and active exploration.

The core research target is to verify whether Dynamic Scene Graphs improve:

- spatial QA,
- dynamic memory,
- `GraphTool` queries,
- interactive task performance.

## Hard Constraints

- Do not break existing APIs.
- Do not break existing tests.
- Runtime core must not introduce network calls, random sources, or current-time dependencies.
- New modules must safely skip or use mocks when AI2-THOR, Habitat, VLM, or perception dependencies are absent.
- Default tests must run in a pure local environment.
- External simulator, VLM, and perception dependencies must live in optional extras or adapter layers.
- External dependencies must not become package runtime dependencies.
- Every PR must pass `python scripts/verify.py`.
- New features must include typed implementation, unit tests, CLI or Python API examples, deterministic artifact validation/comparison where applicable, and README or docs updates.

## PR Plan

### PR 1: Deterministic Episode JSONL Schema And IO

**Status:** Implemented in the current deterministic baseline with API, CLI,
tests, README/runbook coverage, and verifier evidence.

**Task name:** Implement deterministic episode JSONL schema and IO helpers.

**Goal:** Define a unified episode data format for AI2-THOR, Habitat, and mock environment collection without connecting a real simulator.

**Files:**
- Create: `src/dsg_spatialqa_lab/episodes.py`
- Create: `scripts/episodes.py`
- Create: `tests/test_episodes.py`
- Modify: `README.md` or create `docs/episode_schema.md`

**Implementation:**
- Define dataclasses: `EpisodeFrame`, `EpisodeAction`, `EpisodeMetadata`, `EpisodeRecord`, `EpisodeSequence`.
- `EpisodeFrame` fields:
  - `episode_id: str`
  - `scene_id: str`
  - `step: int`
  - `rgb_path: str | None`
  - `depth_path: str | None`
  - `segmentation_path: str | None`
  - `agent_id: str`
  - `agent_pose: Pose3D`
  - `action: str | None`
  - `visible_object_ids: tuple[str, ...]`
  - `metadata: Mapping[str, Any]`
- JSONL format: one frame record per line, serialized with `sort_keys=True`.
- Add API:
  - `episode_frame_to_dict()`
  - `episode_frame_from_dict()`
  - `episode_sequence_to_jsonl()`
  - `episode_sequence_from_jsonl()`
  - `save_episode_sequence()`
  - `load_episode_sequence()`
  - `episode_sequence_digest()`
  - `validate_episode_sequence()`
  - `compare_episode_sequence()`
- Add CLI:
  - `python scripts/episodes.py --validate path.jsonl`
  - `python scripts/episodes.py --compare path.jsonl`
  - `python scripts/episodes.py --summary path.jsonl`
- Tests:
  - stable JSONL round trip,
  - stable digest,
  - structured invalid schema errors,
  - deterministic step ordering,
  - duplicate step validation,
  - missing required field validation.

**Acceptance:**
- `python scripts/verify.py` passes.
- Mock episode JSONL produces a stable digest.
- Invalid files return non-zero structured JSON.

### PR 2: Mock Episode To Oracle DSG Builder

**Status:** Implemented in the current deterministic baseline with API, CLI,
tests, README/runbook coverage, and verifier evidence.

**Task name:** Build oracle DSG from deterministic episode metadata.

**Goal:** Build oracle `DynamicSceneGraph` from mock episode metadata without AI2-THOR.

**Files:**
- Create: `src/dsg_spatialqa_lab/oracle.py`
- Create: `scripts/build_oracle_graph.py`
- Create: `tests/test_oracle_graph_builder.py`

**Implementation:**
- Define `OracleObjectRecord` with:
  - `object_id`
  - `label`
  - `pose`
  - `bbox`
  - `confidence`
  - `visible`
  - `room_id`
  - `region_id`
  - `states: dict[str, Any]`
- In `EpisodeFrame.metadata`, support:
  - `objects`
  - `rooms`
  - `regions`
  - `relations`
- Add API:
  - `build_oracle_graph_from_episode(sequence)`
  - `oracle_graph_report()`
  - `oracle_graph_report_digest()`
  - `validate_oracle_graph_report()`
  - `compare_oracle_graph_report()`
- Graph rules:
  - rooms to `graph.add_room`,
  - regions to `graph.add_region`,
  - agent pose to `graph.set_agent_pose`,
  - objects to `graph.upsert_object`,
  - explicit relations to `graph.add_edge`,
  - containment to `IN_ROOM` / `IN_REGION`,
  - invisible low-confidence objects preserve last-seen info,
  - moved objects create `STATE_CHANGED` and optionally `MOVED_FROM` / `MOVED_TO`.
- Add CLI:
  - `python scripts/build_oracle_graph.py --input episode.jsonl --output-graph oracle-graph.json --report oracle-report.json`
  - `python scripts/build_oracle_graph.py --validate-report oracle-report.json`
  - `python scripts/build_oracle_graph.py --compare-report oracle-report.json`
- Tests:
  - one-frame episode builds graph,
  - multi-frame moved object builds timeline,
  - room/region containment works,
  - explicit relations are preserved,
  - report digest is stable,
  - compare detects drift.

**Acceptance:**
- Mock episode builds graph JSON.
- `graph_summary` includes object, room, region, state, action/event baseline stats.
- `python scripts/verify.py` passes.

### PR 3: AI2-THOR Adapter Skeleton And Mock Fallback

**Status:** Implemented as a deterministic adapter boundary with explicit-step
mock collection, missing-dependency diagnostics, CLI output, and tests. Real
simulator collection remains omitted from the default runtime.

**Task name:** Add optional AI2-THOR adapter with deterministic mock fallback.

**Goal:** Establish simulator adapter boundaries without making AI2-THOR a default dependency.

**Files:**
- Create: `src/dsg_spatialqa_lab/adapters/__init__.py`
- Create: `src/dsg_spatialqa_lab/adapters/ai2thor.py`
- Create: `scripts/collect_ai2thor.py`
- Create: `tests/test_ai2thor_adapter.py`
- Modify: `pyproject.toml`

**Implementation:**
- Keep `ai2thor` out of default runtime dependencies.
- Add optional extra:
  - `ai2thor = ["ai2thor>=..."]`
- Add adapter API:
  - `AI2ThorAdapterConfig`
  - `AI2ThorEpisodeCollector`
  - `collect_episode()`
  - `convert_ai2thor_event_to_episode_frame()`
- If `ai2thor` is absent:
  - adapter import must not fail,
  - real collector call returns: `AI2-THOR optional dependency is not installed. Install with .[ai2thor].`
- Add `build_mock_ai2thor_episode()` for deterministic tests and oracle builder input.
- Add CLI:
  - `python scripts/collect_ai2thor.py --mock --scene FloorPlan1 --episode-id ai2thor_mock_001 --step 1 --step 2 --action Initialize --action MoveAhead --output mock-ai2thor.jsonl`
  - `python scripts/collect_ai2thor.py --scene FloorPlan1 --episode-id ai2thor_real_001 --step 1 --episodes 1 --output out.jsonl`
- Tests:
  - import without AI2-THOR,
  - mock collection stable,
  - CLI mock output validates,
  - missing optional dependency message.

**Acceptance:**
- Default environment passes verify without AI2-THOR.
- `--mock` generates valid episode JSONL.

### PR 4: Extended Spatial Relation Engine

**Status:** Implemented for deterministic metric distance reports, computed
`INSIDE` / `SUPPORTS` geometry, explicit placeholder relation querying, and
unsupported computed-frame diagnostics. Real visibility, reachability, and
occlusion inference remains out of scope and must stay explicit.

**Task name:** Extend deterministic spatial relation engine for metric, containment, visibility, and reachability placeholders.

**Goal:** Expand relation support while preserving deterministic behavior.

**Files:**
- Modify: `src/dsg_spatialqa_lab/relations.py`
- Modify: `src/dsg_spatialqa_lab/graph_tool.py`
- Modify: `src/dsg_spatialqa_lab/memory.py`
- Modify: `tests/test_relations.py`
- Modify: `tests/test_graph_tool.py`

**Relations:**
- `DISTANCE_LT` / `DISTANCE`
  - Add `GraphTool.compute_distance(src, dst)`.
- `INSIDE`
  - Use bbox centroid inside container bbox.
  - Optional containment ratio threshold.
- `SUPPORTS`
  - Inverse of valid `ON`.
- `VISIBLE_FROM`
  - MVP placeholder based on explicit edge or metadata flag.
  - No real frustum/depth calculation yet.
- `REACHABLE_FROM`
  - MVP placeholder based on explicit edge or metadata flag.
  - Habitat/navmesh support comes later.
- `OCCLUDES`
  - Explicit metadata edge only.
- `reference_frame` support:
  - `world`
  - `agent`
  - `image`
  - `object`
  - MVP validates schema and supports explicit edge query for image/object frames.

**Tests:**
- `ON` / `SUPPORTS` consistency.
- `INSIDE` true/false.
- Stable rounded distance.
- Unsupported frame raises structured `SpatialQAError`.
- Explicit `VISIBLE_FROM` / `REACHABLE_FROM` can be queried.

**Acceptance:**
- Existing tests still pass.
- Docs explain computed relations vs explicit-placeholder relations.

### PR 5: Oracle DSG QA Generator MVP

**Status:** Implemented in the current deterministic baseline with API, CLI,
tests, README/runbook coverage, dataset validation, and current-graph
comparison. The implementation keeps QA IO in
`src/dsg_spatialqa_lab/benchmark/qa_generator.py`; no separate `qa_io.py` was
needed for the MVP.

**Task name:** Implement oracle DSG based QA generator.

**Goal:** Generate spatial QA cases from oracle graphs instead of only hand-written cases.

**Files:**
- Create: `src/dsg_spatialqa_lab/benchmark/__init__.py`
- Create: `src/dsg_spatialqa_lab/benchmark/qa_generator.py`
- Create: `src/dsg_spatialqa_lab/benchmark/qa_io.py`
- Create: `scripts/generate_qa.py`
- Create: `tests/test_qa_generator.py`

**QA Types:**
1. `object_location`
2. `object_room`
3. `relative_relation`
4. `nearest_object`
5. `relation_timeline`
6. `scene_delta`
7. `reobserve_targets`
8. `next_action_validity` placeholder

**QA Case Fields:**
- `id`
- `scene_id`
- `episode_id`
- `graph_digest`
- `step`
- `question`
- `question_type`
- `answer`
- `answer_type`
- `choices`
- `reference_frame`
- `required_nodes`
- `required_edges`
- `difficulty`
- `tags`

**API:**
- `generate_qa_cases(graph, *, scene_id, episode_id, max_cases=None, tags=None)`
- `qa_case_to_dict()`
- `qa_case_from_dict()`
- `qa_dataset_jsonl()`
- `qa_dataset_digest()`
- `validate_qa_dataset()`
- `compare_qa_dataset()`

**Generation Rules:**
- Only generate unique-answer samples.
- `nearest_object` requires a sufficient margin between closest and second closest.
- Relation QA requires explicit edge or deterministic geometry support.
- `scene_delta` requires `from_step < to_step`.
- Every QA records `required_nodes` and `required_edges`.
- Generation order is deterministic.

**CLI:**
- `python scripts/generate_qa.py --graph oracle-graph.json --scene-id mock_scene --episode-id mock_ep --output qa.jsonl --max-cases 100`
- `python scripts/generate_qa.py --validate qa.jsonl`
- `python scripts/generate_qa.py --compare qa.jsonl --graph oracle-graph.json`

**Tests:**
- Stable ordering.
- Stable digest.
- Generated cases answerable by `SpatialQAEngine`.
- Invalid dataset structured errors.
- Margin filtering works.
- Required evidence present.

**Acceptance:**
- Mock oracle graph can generate QA JSONL.
- Every QA can be checked by current `SpatialQAEngine` or `GraphTool`.

### PR 6: QA Metrics And Evaluation Runner

**Task name:** Add QA metrics and benchmark evaluation runner.

**Goal:** Evaluate QA predictions with accuracy, evidence consistency, and stable breakdowns.

**Files:**
- Create: `src/dsg_spatialqa_lab/eval/__init__.py`
- Create: `src/dsg_spatialqa_lab/eval/qa_metrics.py`
- Create: `src/dsg_spatialqa_lab/eval/evidence_metrics.py`
- Create: `scripts/run_qa_eval.py`
- Create: `tests/test_qa_metrics.py`

**Prediction Format:**
- `id`
- `answer`
- `evidence_nodes`
- `evidence_edges`
- `confidence`
- `error`

**Metrics:**
- `exact_match`
- `multiple_choice_accuracy`
- `numeric_mae`
- `evidence_node_recall`
- `evidence_edge_recall`
- `answer_graph_consistency`
- by-question-type breakdown
- by-tag breakdown
- by-reference-frame breakdown

**API:**
- `evaluate_qa_predictions(gold_cases, predictions)`
- `qa_eval_report()`
- `qa_eval_report_digest()`
- `validate_qa_eval_report()`
- `compare_qa_eval_report()`

**CLI:**
- `python scripts/run_qa_eval.py --gold qa.jsonl --pred predictions.jsonl --report qa-eval-report.json`
- `python scripts/run_qa_eval.py --validate-report qa-eval-report.json`
- `python scripts/run_qa_eval.py --compare-report qa-eval-report.json`

**Tests:**
- Exact match.
- Wrong answer.
- Missing prediction.
- Partial evidence recall.
- Stable breakdown.
- Stable digest.

**Acceptance:**
- Deterministic baseline predictions can be evaluated.
- Output includes compact and full report.

### PR 7: Baseline Runner First Version

**Task name:** Implement deterministic baseline runner skeleton.

**Goal:** Add baseline framework without real VLM calls, plus a working `GraphTool` baseline.

**Files:**
- Create: `src/dsg_spatialqa_lab/agents/__init__.py`
- Create: `src/dsg_spatialqa_lab/agents/base.py`
- Create: `src/dsg_spatialqa_lab/agents/random_agent.py`
- Create: `src/dsg_spatialqa_lab/agents/graph_tool_agent.py`
- Create: `src/dsg_spatialqa_lab/agents/graph_text_agent.py`
- Create: `src/dsg_spatialqa_lab/agents/caption_memory_agent.py`
- Create: `scripts/run_baselines.py`
- Create: `tests/test_baselines.py`

**Baselines:**
- B0 Random / majority:
  - deterministic selection,
  - no `random` module.
- B4 Static graph text:
  - graph summary to answer placeholder,
  - stable unsupported output for complex questions.
- B6 Dynamic DSG + GraphTool:
  - call `SpatialQAEngine` / `GraphTool`,
  - output answer and evidence.
- B1/B2/B3/B5/B7/B8:
  - define interface only,
  - `ExternalModelAgent` raises clear error unless provider configured,
  - default tests do not require API keys.

**CLI:**
- `python scripts/run_baselines.py --baseline graph_tool --graph oracle-graph.json --qa qa.jsonl --pred predictions.jsonl`
- `python scripts/run_baselines.py --list-baselines`

**Tests:**
- `graph_tool` answers generated QA.
- Random baseline deterministic.
- External model baseline disabled by default.
- Prediction JSONL validates.

**Acceptance:**
- End-to-end chain works:
  - `generate_qa.py`
  - `run_baselines.py --baseline graph_tool`
  - `run_qa_eval.py`
- GraphTool baseline reaches high accuracy on oracle QA; failures expose case details.

### PR 8: Graph Metrics And Oracle/Predicted Alignment

**Task name:** Add graph construction metrics and oracle-vs-predicted comparison interface.

**Goal:** Establish graph evaluation API before predicted DSG is complete.

**Files:**
- Create: `src/dsg_spatialqa_lab/eval/graph_metrics.py`
- Create: `scripts/evaluate_graphs.py`
- Create: `tests/test_graph_metrics.py`

**Metrics:**
- object precision / recall,
- object label accuracy,
- relation precision / recall / F1,
- state accuracy placeholder,
- 3D bbox center error,
- by-relation breakdown,
- by-object-label breakdown.

**Matching Rules:**
- Default object ID exact match.
- Optional label + nearest center matching placeholder.
- Deterministic ordering.

**API:**
- `compare_graphs(oracle_graph, predicted_graph)`
- `graph_eval_report()`
- `graph_eval_report_digest()`
- `validate_graph_eval_report()`
- `compare_graph_eval_report()`

**CLI:**
- `python scripts/evaluate_graphs.py --oracle oracle-graph.json --predicted predicted-graph.json --report graph-eval-report.json`

**Tests:**
- Perfect graph score equals `1.0`.
- Missing object reduces recall.
- Extra object reduces precision.
- Wrong relation affects relation F1.
- Stable digest.

**Acceptance:**
- Two graph JSON artifacts can be compared.
- Report contains basic fields needed for error attribution.

### PR 9: Predicted DSG Skeleton

**Task name:** Add predicted DSG builder skeleton with deterministic mock perception.

**Goal:** Define predicted graph pipeline boundaries for segmentation, depth projection, fusion, tracking, and relation inference using mock perception first.

**Files:**
- Create: `src/dsg_spatialqa_lab/perception/__init__.py`
- Create: `src/dsg_spatialqa_lab/perception/mock.py`
- Create: `src/dsg_spatialqa_lab/perception/depth_projector.py`
- Create: `src/dsg_spatialqa_lab/perception/object_fusion.py`
- Create: `src/dsg_spatialqa_lab/perception/object_tracker.py`
- Create: `src/dsg_spatialqa_lab/predicted.py`
- Create: `scripts/build_predicted_graph.py`
- Create: `tests/test_predicted_graph_builder.py`

**Implementation:**
- Define `Detection2D`.
- Define `Instance3D`.
- Add `MockSegmenter`.
- Add `MockDepthProjector`.
- Add `SimpleObjectFusion`.
- Add `SimpleObjectTracker`.
- Add `build_predicted_graph_from_episode()`.
- Do not use real SAM/Open3D dependencies.
- Mock perception reads `EpisodeFrame.metadata["mock_detections"]`.

**CLI:**
- `python scripts/build_predicted_graph.py --input episode.jsonl --output-graph predicted-graph.json --report predicted-report.json --mock`

**Tests:**
- Mock detections build objects.
- Object ID remains stable across frames.
- Missing detection creates invisible/low-confidence state.
- Relation inference deterministic.
- Graph metrics compare oracle vs predicted.

**Acceptance:**
- Typed predicted graph pipeline skeleton exists.
- Default verify does not need real perception models.

### PR 10: Error Attribution Report

**Task name:** Implement QA error attribution between oracle graph, predicted graph, and model prediction.

**Goal:** Attribute QA failures to benchmark/engine, graph construction, retrieval/evidence, or answer reasoning.

**Files:**
- Create: `src/dsg_spatialqa_lab/eval/error_attribution.py`
- Create: `scripts/attribute_errors.py`
- Create: `tests/test_error_attribution.py`

**Inputs:**
- gold QA,
- oracle graph,
- predicted graph,
- prediction JSONL,
- optional oracle GraphTool prediction.

**Output Fields:**
- `case_id`
- `answer_correct`
- `oracle_answer`
- `predicted_answer`
- `model_answer`
- `required_nodes_present`
- `required_edges_present`
- `error_category`

**Error Categories:**
- `correct`
- `missing_object`
- `wrong_object_label`
- `missing_relation`
- `wrong_relation`
- `missing_state`
- `retrieval_error`
- `reasoning_error`
- `ambiguous_question`
- `unsupported_question_type`
- `benchmark_or_engine_error`
- `graph_construction`
- `reasoning_or_tool_use`
- `evidence_missing`

**Rules:**
- If oracle GraphTool is wrong, classify as `benchmark_or_engine_error`.
- If oracle GraphTool is correct and predicted GraphTool is wrong, classify as `graph_construction`.
- If predicted GraphTool is correct and model is wrong, classify as `reasoning_or_tool_use`.
- If required evidence is absent from predicted graph, classify as `evidence_missing`.

**CLI:**
- `python scripts/attribute_errors.py --gold qa.jsonl --oracle-graph oracle.json --predicted-graph predicted.json --predictions predictions.jsonl --report error-attribution.json`

**Tests:**
- Missing node attribution.
- Missing relation attribution.
- Reasoning error attribution.
- Benchmark/engine error path.
- Stable breakdown.

**Acceptance:**
- Report includes by-error-category breakdown.

### PR 11: Dashboard MVP

**Task name:** Add lightweight static dashboard export and optional Streamlit app.

**Goal:** Export a static dashboard bundle for per-sample QA/error review; keep Streamlit optional.

**Files:**
- Create: `src/dsg_spatialqa_lab/visualization/__init__.py`
- Create: `src/dsg_spatialqa_lab/visualization/dashboard_export.py`
- Create: `scripts/export_dashboard.py`
- Create: `visualization/app.py`
- Create: `tests/test_dashboard_export.py`

**Implementation:**
- Dashboard bundle JSON contains:
  - `qa_case`
  - `prediction`
  - `eval_result`
  - `error_attribution`
  - `evidence_subgraph`
  - frame paths when present
  - graph summary
- HTML export:
  - one `index.html`,
  - embedded JSON or linked `dashboard.json`,
  - table filters for question type, tag, correctness, and error category.
- Optional Streamlit:
  - do not import Streamlit unless running app,
  - missing Streamlit gives clear install message.

**CLI:**
- `python scripts/export_dashboard.py --qa qa.jsonl --pred predictions.jsonl --eval-report qa-eval-report.json --graph oracle-graph.json --output dashboard/`

**Tests:**
- Stable bundle.
- Missing optional files handled.
- Evidence subgraph extracted.
- No Streamlit dependency needed for verify.

**Acceptance:**
- `dashboard/index.html` is generated.
- Gold, prediction, evidence, and error are viewable per sample.

### PR 12: Active EQA Task Skeleton

**Task name:** Add active EQA task and policy skeleton.

**Goal:** Define active exploration task interfaces with mock environment and deterministic policies.

**Files:**
- Create: `src/dsg_spatialqa_lab/tasks/__init__.py`
- Create: `src/dsg_spatialqa_lab/tasks/active_eqa.py`
- Create: `src/dsg_spatialqa_lab/agents/active_graph_agent.py`
- Create: `src/dsg_spatialqa_lab/eval/task_metrics.py`
- Create: `scripts/run_active_tasks.py`
- Create: `tests/test_active_eqa.py`

**Data Structure:**
- `ActiveEQATask`
  - `id`
  - `scene_id`
  - `episode_id`
  - `initial_step`
  - `question`
  - `gold_answer`
  - `success_conditions`
  - `max_actions`
  - `required_evidence`

**Mock Environment:**
- `reset(task)`
- `observe()`
- `step(action)`
- `current_graph()`
- `done()`

**Policies:**
- `direct_answer`
- `random_explore_deterministic`
- `graph_uncertainty_policy` placeholder
- `oracle_evidence_policy` for tests

**Metrics:**
- `task_success`
- `answer_accuracy`
- `action_count`
- `evidence_coverage`
- `answer_graph_consistency`

**CLI:**
- `python scripts/run_active_tasks.py --tasks active-tasks.jsonl --policy direct_answer --report active-report.json`

**Tests:**
- Direct answer works if graph is sufficient.
- Exploration collects evidence in mock env.
- Stable action count.
- Evidence coverage computed.
- `max_actions` enforced.

**Acceptance:**
- Mock active EQA tasks run.
- Later AI2-THOR/Habitat env adapters can plug in.

### PR 13: Habitat Adapter Skeleton

**Task name:** Add optional Habitat adapter skeleton.

**Goal:** Reserve Habitat / ReplicaCAD environment interface without making Habitat a default dependency.

**Files:**
- Create: `src/dsg_spatialqa_lab/adapters/habitat.py`
- Create: `scripts/collect_habitat.py`
- Create: `tests/test_habitat_adapter.py`
- Modify: `pyproject.toml`

**Implementation:**
- Add `HabitatAdapterConfig`.
- Add `HabitatEpisodeCollector` interface.
- Add `convert_habitat_observation_to_episode_frame()`.
- Add mock Habitat episode generator.
- Add optional dependency handling.
- Add optional extra:
  - `habitat = []`

**Tests:**
- Mock collection works.
- Missing dependency error is clear.
- API aligns with AI2-THOR adapter.

**Acceptance:**
- Habitat adapter skeleton produces `EpisodeSequence` compatible with oracle builder.
- Default verify does not install Habitat.

### PR 14: Benchmark Scale Tooling

**Task name:** Add benchmark manifest and dataset generation tooling.

**Goal:** Support larger benchmark manifests while default tests use small fixtures.

**Files:**
- Create: `src/dsg_spatialqa_lab/benchmark/manifest.py`
- Create: `scripts/build_benchmark.py`
- Create: `tests/test_benchmark_manifest.py`

**BenchmarkManifest Fields:**
- `schema_version`
- `dataset_name`
- `scene_count`
- `episode_count`
- `qa_count`
- `task_count`
- `graph_digests`
- `qa_dataset_digests`
- `filters`
- `coverage`

**Coverage:**
- by question type,
- by scene,
- by episode,
- by reference frame,
- by tag,
- dynamic/static,
- oracle/predicted.

**CLI:**
- `python scripts/build_benchmark.py --episodes data/episodes/*.jsonl --output-dir data/benchmark --max-qa-per-episode 100 --manifest benchmark-manifest.json`
- Add validate/compare manifest commands.

**Tests:**
- Small benchmark deterministic.
- Coverage correct.
- Stable digest.
- Compare detects changed QA.

**Acceptance:**
- Episode to oracle graph to QA to manifest chain works.

### PR 15: README And Roadmap Docs

**Task name:** Update project documentation for full DSG-SpatialQA Lab roadmap.

**Goal:** Clearly distinguish current deterministic core, oracle DSG pipeline, predicted DSG pipeline, baselines, dashboard, active EQA, and optional dependencies.

**Files:**
- Modify: `README.md`
- Create: `docs/architecture.md`
- Create: `docs/roadmap.md`
- Create: `docs/benchmark_format.md`

**Content:**
- Keep existing verify quickstart.
- Add mock end-to-end:
  - collect mock episode,
  - build oracle graph,
  - generate QA,
  - run GraphTool baseline,
  - evaluate QA,
  - export dashboard.
- Add optional simulator instructions:
  - AI2-THOR extra,
  - Habitat extra.
- Add data format docs:
  - episode JSONL,
  - graph JSON,
  - QA JSONL,
  - prediction JSONL,
  - eval report,
  - dashboard bundle.
- Add project boundaries:
  - default runtime deterministic,
  - no default external AI calls,
  - predicted DSG uses optional mock-first pipeline,
  - real VLM/simulator adapters are optional.

**Acceptance:**
- README mock commands run locally.
- Docs do not claim real benchmark, predicted DSG, or active EQA completion before implementation and tests exist.

## Recommended End-To-End Target After PR 1-7

```bash
python scripts/collect_ai2thor.py \
  --mock \
  --scene FloorPlan1 \
  --episode-id ai2thor_mock_001 \
  --step 1 \
  --step 2 \
  --action Initialize \
  --action MoveAhead \
  --output mock-episode.jsonl

python scripts/episodes.py --validate mock-episode.jsonl

python scripts/build_oracle_graph.py \
  --input mock-episode.jsonl \
  --output-graph oracle-graph.json \
  --report oracle-report.json

python scripts/generate_qa.py \
  --graph oracle-graph.json \
  --scene-id mock_scene \
  --episode-id mock_ep \
  --output qa.jsonl \
  --max-cases 100

python scripts/run_baselines.py \
  --baseline graph_tool \
  --graph oracle-graph.json \
  --qa qa.jsonl \
  --pred predictions.jsonl

python scripts/run_qa_eval.py \
  --gold qa.jsonl \
  --pred predictions.jsonl \
  --report qa-eval-report.json

python scripts/export_dashboard.py \
  --qa qa.jsonl \
  --pred predictions.jsonl \
  --eval-report qa-eval-report.json \
  --graph oracle-graph.json \
  --output dashboard

python scripts/verify.py
```

## Development Priority

Follow this order unless dependencies require adjustment:

1. P0: Episode JSONL + IO
2. P1: Mock episode to oracle DSG
3. P2: AI2-THOR adapter skeleton + mock fallback
4. P3: Relation engine extension
5. P4: QA generator
6. P5: QA metrics
7. P6: Baseline runner
8. P7: Graph metrics
9. P8: Predicted DSG skeleton
10. P9: Error attribution
11. P10: Dashboard MVP
12. P11: Active EQA skeleton
13. P12: Habitat adapter skeleton
14. P13: Benchmark manifest tooling
15. P14: README/docs

## Prohibited Changes

Do not:

- Add `ai2thor`, `habitat`, `open3d`, `torch`, `sam`, or similar heavy packages as default dependencies.
- Call network APIs from runtime core.
- Depend on real simulators in default tests.
- Use `random`, `datetime`, or `time` to produce unstable results.
- Introduce unsorted dict/list output that makes digests unstable.
- Hard-code model API keys or provider assumptions.
- Break `scripts/verify.py`.
- Delete existing evaluation cases.
- Make dashboard dependencies default install dependencies.
- Overstate real predicted DSG or active EQA completion in README.

## Definition Of Done For Every PR

A PR is complete only when:

1. All new public APIs have type annotations.
2. All JSON output uses `sort_keys=True` and stable digests.
3. All CLI invalid inputs return structured JSON and non-zero exit.
4. New tests cover happy path, invalid path, and determinism.
5. `python scripts/verify.py` passes.
6. README or docs include minimal usage.
7. Optional dependencies do not affect default installation.
8. Skeleton features explicitly document placeholder boundaries in docstrings and README.

## Milestones

### Milestone A: Oracle QA MVP

Complete PR 1-7.

**Acceptance:**
- Mock episode to oracle graph to QA to GraphTool baseline to QA eval runs end-to-end.
- At least 100 mock QA are generated.
- GraphTool baseline reaches high accuracy on oracle-generated QA.
- Dashboard can inspect each QA's gold answer, prediction, and evidence.

### Milestone B: Graph Construction Evaluation

Complete PR 8-10.

**Acceptance:**
- Oracle graph and predicted mock graph can be compared.
- Object recall, relation F1, and error attribution are reported.
- QA errors can be attributed to missing object, missing relation, or reasoning.

### Milestone C: Interactive Skeleton

Complete PR 11-12.

**Acceptance:**
- Mock active EQA task runs.
- Task success, action count, and evidence coverage are reported.
- Dashboard can show active task results.

### Milestone D: Simulator Extension

Complete PR 13-14.

**Acceptance:**
- AI2-THOR and Habitat adapters both have optional skeletons.
- Benchmark manifest summarizes multiple episodes, graphs, and QA datasets.

## First Codex Task

Start with PR 1: Implement deterministic episode JSONL schema and IO helpers.

Requirements:

- Add `src/dsg_spatialqa_lab/episodes.py`.
- Add `scripts/episodes.py`.
- Add `tests/test_episodes.py`.
- Update README quickstart with an episode JSONL validation example.
- Keep runtime dependencies empty.
- Run `python scripts/verify.py` and fix every failure.

After PR 1-7, the project should have the first complete oracle benchmark loop:

```text
mock/AI2-THOR episode -> oracle DSG -> automatic QA -> GraphTool baseline -> QA evaluation -> dashboard
```
