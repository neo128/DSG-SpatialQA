# DSG-SpatialQA Lab Validation Platform Development Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic validation platform that can test whether
Dynamic Scene Graphs improve spatial QA, dynamic memory, `GraphTool` queries,
and interactive task performance.

**Architecture:** Preserve the deterministic in-memory core and build a local
evaluation loop around it: explicit episodes -> oracle DSG -> predicted DSG ->
QA/task datasets -> baselines -> metrics -> error attribution -> review
artifacts. Optional simulator, perception, and dashboard adapters stay outside
the default runtime and must fail closed or use deterministic mocks.

**Tech Stack:** Python package under `src/dsg_spatialqa_lab`, standard-library runtime by default, optional simulator/dashboard extras, deterministic JSON/JSONL artifacts, CLI scripts, pytest, ruff, mypy, and `scripts/verify.py`.

---

## Next-Stage Goal: VLM Gate Then DSG Memory/Query Experiments

**Goal:** Treat VLM-only semantic success rate >= 50% as the entry gate for
the next DSG experiment stage. After the gate is satisfied, optimize DSG memory
storage contents, storage method, query contents, and query method before
claiming any DSG-vs-VLM result.

Current gate status:

- VLM-only P26 semantic match: `49 / 60 = 0.816667`.
- VLM-only P26 strict exact match: `0 / 60 = 0.000000`.
- Gate decision: semantic >= 50% is satisfied; strict exact remains a
  secondary answer-normalization metric, not the stage entry metric.
- Current strict detector-only DSG P30 semantic match: `25 / 60 = 0.416667`.
- Current DSG gap to VLM P26: `-24` semantic matches.
- Next-stage goal report:
  `handoffs/ai2thor-real-small/outputs/diagnostics/p32-next-stage-goal-vlm-gate-dsg-memory-query.json`.

DSG experiment axes for the next stage:

- [ ] **Memory storage contents:** store detector-visible object states,
  support-surface candidates, object-id/label provenance, temporal last-seen
  facts, and relation hypotheses with confidence/evidence paths.
- [ ] **Memory storage method:** keep append-only observation memory before
  collapsing into graph edges; separate observed facts from inferred location
  hypotheses; retain top-k support candidates and ambiguity/abstain flags.
- [ ] **Query contents:** retrieve target state, visible-frame evidence,
  support candidates, room fallback, missing-evidence status, and answer
  rationale together.
- [ ] **Query method:** rank explicit detector current-location evidence first,
  then inferred support hypotheses, then room fallback; return structured
  missing-evidence blockers when the graph cannot answer.
- [ ] **P32 detector return loop:** use
  `handoffs/ai2thor-real-small/inputs/predicted-dsg/p31-detector-recall-handoff.json`
  to obtain external detector JSONL without gold answers, import it, rebuild
  observation-backed predicted DSG, and rerun DSG candidate/eval.
- [ ] **P33 memory/query ablation:** compare baseline GraphTool lookup against
  memory-first retrieval and support-candidate ranking, recording wins/losses
  against VLM P26.

## Next-Stage Todo: Conclusive DSG-vs-Control Evidence

**Goal:** Turn the ready real-small handoff into an explicit research decision:
either DSG is superior to VLM/video-memory controls under pre-registered gates,
or it is not superior for the current package.

Completed in this stage:

- [x] Add a deterministic research conclusion report that consumes existing
  readiness, offline-control result, QA eval, graph eval, predicted DSG
  evidence, and error-attribution artifacts.
- [x] Use paired case-level exact-match comparisons and a one-sided sign test
  instead of relying only on aggregate exact-match deltas.
- [x] Require practical superiority floors for candidate exact-match rate,
  exact-match rate delta, statistical paired lift, and predicted graph object
  recall.
- [x] Generate current handoff conclusion artifacts:
  `handoffs/ai2thor-real-small/outputs/research-conclusion.json` and
  `handoffs/ai2thor-real-small/outputs/research-conclusion.zh.md`.
- [x] Record the current formal verdict:
  `dsg_not_superior` for the current ready package.
- [x] Attach QA observability context to the conclusion layer so the report
  records full-oracle versus observation-aware scope and evidence-observable
  QA counts.
- [x] Block observation-aware superiority claims unless QA evals are run on the
  evidence-observable QA slice and the slice has enough cases.

Next development priorities:

- [ ] Improve the real detector/RGB-D predicted graph, not the oracle or
  metadata-assisted diagnostic graph, until object recall and relation quality
  pass the conclusion gates.
- [ ] Produce separate observation-aware QA eval, four-control delta reports,
  and conclusion artifacts once evidence-observable coverage reaches the
  configured sample floor.
- [ ] Re-run the four required external controls with the structured JSON
  prompt and preserved evidence traces.
- [ ] Expand observation-aware QA coverage with dynamic-memory and
  GraphTool-query cases that are actually observable from the detector/RGB-D
  sequence.
- [ ] Re-run `scripts/conclude_experiment.py` after each candidate update and
  treat `dsg_not_superior` as the default conclusion until all gates pass.
- [ ] Only claim DSG superiority when the conclusion report returns
  `dsg_superior` with `dsg_superiority_claim_allowed=true`.

## Current State

The project already has the deterministic foundation needed for an offline
evaluation loop:

- `DynamicSceneGraph`, `GraphTool`, `SpatialQAEngine`, and `VLAAnchorPlanner` deterministic local core.
- In-memory graph structures, object state history, relation edges, QA intents, and VLA pick/place-relative prototypes.
- Deterministic episode JSONL, oracle graph building, AI2-THOR mock adapter
  boundary, extended relation support, oracle QA JSONL generation, QA metrics,
  local baseline predictions, graph metrics, and predicted DSG mock construction.
- Deterministic predicted DSG skeleton with mock segmentation, depth
  projection, object tracking/fusion, hidden missed-object updates, relation
  inference, prediction-source metadata propagation, detection-source report
  summaries, reports, and CLI validation/comparison.
- Predicted DSG construction can also read explicit local
  `SceneObservation` sequence artifacts, so RGB-D or detector outputs produced
  outside the deterministic runtime can be converted into predicted graph JSON
  and standard predicted graph reports without reading `mock_detections`.
- Oracle-vs-predicted graph evaluation with exact object-id matching by
  default, optional label+nearest-center object matching, and room-aware
  label+nearest-center matching that remaps relation edges through matched
  object pairs, reports duplicate-track / ID-fragmentation diagnostics, and
  includes confidence-weighted object/relation metrics plus prediction-source
  confidence precision slices.
- QA error attribution, static dashboard export with optional active-task
  review panels and local Research Axis / Evidence Source filtering, and
  deterministic mock active EQA task reports with task success, answer
  accuracy, action count, evidence coverage, and answer-graph consistency metrics plus
  budget-vs-success analysis by max-action budget, per-action evidence
  snapshots, QA delta diagnostic slices by scene, episode, question type, tag,
  and reference frame, and a deterministic `next_best_view` policy placeholder
  that targets missing required evidence.
- Experiment summary artifacts now retain graph construction diagnostics from
  graph eval reports: object recall, relation F1, matched-object state
  accuracy, duplicate-track / ID-fragmentation counts, and prediction-source
  precision slices.
- Experiment summary artifacts now retain error attribution diagnostics from
  QA attribution reports: graph-construction, evidence-missing,
  reasoning/tool-use, benchmark/engine, research-axis, and predicted-evidence
  source failure summaries.
- Experiment summary artifacts now retain failure-linkage diagnostics that
  match attribution reports to graph eval reports by oracle/predicted graph
  digest, connecting graph quality metrics to QA failure causes for the same
  predicted graph.
- Offline external prediction import tooling that converts local VLM/caption
  memory style prediction records into standard `QAPrediction` JSONL plus
  stable import reports for QA evaluation, attribution, and dashboard review,
  including derived source profiles for side-by-side source review. The
  deterministic four-way import handoff can optionally take a local candidate
  DSG GraphTool prediction JSONL file and write candidate QA eval plus
  DSG-vs-control QA delta reports for every imported control source. It can
  now consume either raw `OfflinePredictionRecord` JSONL or already-normalized
  standard `QAPrediction` JSONL source files, and it can now run from one
  explicit local import manifest that records the QA, source, candidate,
  output, matrix, and real-source metadata paths. This reduces hand conversion
  and repeated CLI argument drift before real VLM/LLM prediction artifacts
  enter the matrix.
- Detector/RGB-D observation import tooling that converts explicit local
  detector JSONL records into standard `SceneObservationSequence` artifacts,
  retaining RGB/depth/segmentation paths and detector metadata as object
  evidence attributes so observation-backed predicted DSG construction can
  start from externally produced detector outputs without running detector
  models in the default runtime. The detector handoff can now run this import,
  predicted graph construction, predicted graph reporting, and predicted DSG
  evidence reporting in one deterministic local command.
- AI2-THOR and Habitat optional adapter skeletons with deterministic mock
  episode generation and missing-dependency diagnostics for non-mock collection.
- Benchmark manifest tooling that builds oracle graph and QA artifacts from
  explicit episode JSONL files and records stable coverage/digest metadata plus
  QA, active-task, dashboard, graph-eval, error-attribution, and
  predicted-graph experiment artifact refs.
- Real experiment readiness reporting that checks whether a benchmark manifest
  has enough local artifact evidence for real DSG-vs-control evaluation:
  explicit real data declaration, minimum episode/scene/QA coverage, dynamic
  and GraphTool-style QA coverage, offline control imports, observation-backed
  predicted DSG reports, graph eval, attribution, active-task, and dashboard
  review artifacts. It now loads and validates graph-eval and error-attribution
  reports, loads/validates predicted-graph reports against their local
  observation/graph files, and requires all predicted-graph digests to align
  before accepting them as real predicted-DSG evidence. It also loads and
  validates dashboard bundles before accepting them as review evidence.
- `scripts/verify.py` local verification gate covering install, ruff, mypy, determinism scan, pytest, build, and evaluation suite.

## Current Gaps

The remaining gaps are now experimental-evidence gaps. The framework is mature
enough for local artifact validation, but it cannot yet prove the DSG route is
better than VLM-only without real small-scale experiments:

1. Real data: AI2-THOR or Habitat collection must produce enough local episode
   and benchmark artifacts to cover spatial QA, dynamic memory, GraphTool query,
   and interactive-task cases beyond mock scenes.
2. Real controls: VLM-only, multi-frame VLM, caption-memory, and graph-text LLM
   predictions must be imported as offline prediction JSONL artifacts and
   evaluated side by side against DSG-based answers. The local import/eval
   handoff exists, but the actual real prediction files still need to be
   produced outside the deterministic runtime.
3. Real predicted DSG: RGB-D or detector outputs must be converted into
   predicted graph artifacts. The current code now supports an explicit
   `SceneObservation` sequence boundary for those outputs, but it still does
   not run detectors, depth estimators, or simulator collection inside the
   default runtime.

The current pilot handoff root is `handoffs/ai2thor-real-smoke/`. It now
contains the real-data, real-control, and predicted-DSG child manifests plus
ready request bundles for AI2-THOR real collection and detector/RGB-D predicted
DSG intake. Real controls remain intentionally blocked until the same handoff
receives the gold QA JSONL, candidate DSG prediction JSONL, and four external
offline prediction files.

## Development Goal

Keep the deterministic core stable while shifting the next phase from building
more framework skeletons to running real small-scale experiments. New simulator,
VLM, and detector work should write local artifacts first, then reuse the
existing QA, graph, attribution, manifest, summary, and dashboard evaluators.

The core research target is to verify whether Dynamic Scene Graphs improve:

- spatial QA,
- dynamic memory,
- `GraphTool` queries,
- interactive task performance.

## Evaluation Questions And Evidence Chain

### RQ1: Spatial QA

Compare oracle DSG `GraphTool` answers, predicted DSG `GraphTool` answers,
graph-text/choice baselines, and future external-model predictions over the
same deterministic QA JSONL cases. Evidence comes from exact-match accuracy,
multiple-choice accuracy, numeric MAE, evidence node/edge recall, and
question-type/tag/reference-frame and research-axis breakdowns, plus
source-level attribution for predicted evidence used by failed QA cases.

### RQ2: Dynamic Memory

Use episode steps, object/agent timelines, scene deltas, moved-object events,
last-seen state, hidden low-confidence updates, and re-observation targets.
Evidence comes from QA cases and future task metrics that require temporal
state, not only static spatial relations.

### RQ3: GraphTool Query Utility

Compare direct graph-tool answers with graph-text and future VLM/caption-memory
baselines. Evidence comes from answer accuracy, evidence recall, graph-query
case coverage, and error attribution that separates graph construction failures
from reasoning/tool-use failures.

### RQ4: Interactive Task Ability

Use mock active EQA tasks after attribution and dashboard support. Evidence
comes from task success, answer accuracy, action count, evidence coverage, and
answer-graph consistency under explicit max-action budgets, including
budget-vs-success curves and per-action evidence snapshots across those
budgets. The mock `next_best_view` policy provides deterministic missing-
evidence action targets for policy comparison without claiming real navigation.

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

**Status:** Implemented in the current deterministic baseline with prediction
JSONL IO, QA eval metrics, research-axis breakdowns for RQ1-RQ3, stable
reports, validation/comparison helpers, candidate-vs-baseline delta reports,
CLI, tests, README/runbook coverage, and verifier evidence. The MVP keeps
evidence recall metrics in
`src/dsg_spatialqa_lab/eval/qa_metrics.py`; a separate `evidence_metrics.py`
split can wait until graph/task evidence metrics diverge.

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
- by-research-axis breakdown for `spatial_qa`, `dynamic_memory`, and
  `graph_tool_query`

**API:**
- `evaluate_qa_predictions(gold_cases, predictions)`
- `qa_eval_report()`
- `qa_eval_report_digest()`
- `validate_qa_eval_report()`
- `compare_qa_eval_report()`
- `qa_eval_delta_report()`
- `validate_qa_eval_delta_report()`
- `compare_qa_eval_delta_report()`

**CLI:**
- `python scripts/run_qa_eval.py --gold qa.jsonl --pred predictions.jsonl --report qa-eval-report.json`
- `python scripts/run_qa_eval.py --validate-report qa-eval-report.json`
- `python scripts/run_qa_eval.py --compare-report qa-eval-report.json`
- `python scripts/run_qa_eval.py --candidate-report graph-tool-report.json --baseline-report majority-report.json --delta-report qa-delta-report.json`
- `python scripts/run_qa_eval.py --validate-delta-report qa-delta-report.json`
- `python scripts/run_qa_eval.py --compare-delta-report qa-delta-report.json`

**Tests:**
- Exact match.
- Wrong answer.
- Missing prediction.
- Partial evidence recall.
- Stable breakdown.
- Stable digest.
- Stable candidate-vs-baseline delta report.

**Acceptance:**
- Deterministic baseline predictions can be evaluated.
- Output includes compact and full report.

### PR 7: Baseline Runner First Version

**Status:** Implemented in the current deterministic baseline with local agent
interfaces, `graph_tool`, `majority`, `graph_text`, disabled `caption_memory`,
prediction JSONL output, CLI listing/running, tests, README/runbook coverage,
and verifier evidence. The deterministic first-choice baseline lives in
`src/dsg_spatialqa_lab/agents/majority_agent.py` instead of a file name that
would trip the local determinism scanner.

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

**Status:** Implemented in the current deterministic baseline with exact
object-id and exact relation-edge matching by default, optional label+nearest
center matching, room-aware label+nearest center matching with relation
remapping through matched object pairs,
object/relation metrics, confidence-weighted metrics, bbox center error,
object-label, relation, and prediction-source confidence
breakdowns, stable reports, validation/comparison helpers, CLI, tests,
README/runbook coverage, and verifier evidence.

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
- Optional label + nearest center matching.
- Optional room-aware label + nearest center matching.
- Relation keys are remapped through matched object pairs when non-exact object
  IDs are matched.
- Deterministic ordering.

**API:**
- `compare_graphs(oracle_graph, predicted_graph)`
- `graph_eval_report()`
- `graph_eval_report_digest()`
- `validate_graph_eval_report()`
- `compare_graph_eval_report()`

**CLI:**
- `python scripts/evaluate_graphs.py --oracle oracle-graph.json --predicted predicted-graph.json --report graph-eval-report.json`
- `python scripts/evaluate_graphs.py --oracle oracle-graph.json --predicted predicted-graph.json --matching label_center --center-distance-threshold 0.25 --report graph-eval-report.json`
- `python scripts/evaluate_graphs.py --oracle oracle-graph.json --predicted predicted-graph.json --matching label_center_room --center-distance-threshold 0.25 --report graph-eval-report.json`

**Tests:**
- Perfect graph score equals `1.0`.
- Missing object reduces recall.
- Extra object reduces precision.
- Wrong relation affects relation F1.
- Label+center matching handles changed predicted object IDs and remapped
  relation edges.
- Room-aware label+center matching rejects same-label center matches when the
  current room differs.
- Stable digest.

**Acceptance:**
- Two graph JSON artifacts can be compared.
- Report contains basic fields needed for error attribution.

### PR 9: Predicted DSG Skeleton

**Status:** Implemented in the current deterministic baseline with mock
perception pipeline APIs, CLI, tests, README/runbook coverage, and verifier
evidence.

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
- Detection source metadata is propagated to predicted graph nodes, inferred
  relation edges are source-marked, and predicted reports summarize detections
  by source.
- Default verify does not need real perception models.

### PR 10: Error Attribution Report

**Status:** Implemented in the current deterministic baseline with API, CLI,
tests, README/runbook coverage, predicted-evidence source summaries, stable
report digests, validation/comparison, and verifier evidence.

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
- `predicted_evidence_sources`
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
- Summaries group cases and errors by predicted evidence source, falling back
  to `missing_predicted_evidence` when no required predicted evidence source is
  present.

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

**Status:** Implemented in the current deterministic baseline with static bundle
and HTML export APIs, active task, active delta, and experiment summary review
sections, CLI, tests, README/runbook coverage, optional Streamlit stub, stable
digests, and verifier evidence.

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
  - predicted evidence source summaries
  - `evidence_subgraph`
  - frame paths when present
  - graph summary
  - optional active task review panels
  - optional active task delta review tables
  - optional experiment summary review rows
- HTML export:
  - one `index.html`,
  - embedded JSON or linked `dashboard.json`,
  - table filters for question type, tag, correctness, error category, and
    predicted evidence source.
  - active task delta tables for candidate-vs-baseline task metrics and
    max-action budget lift.
  - experiment summary table for RQ1-RQ4 lift.
- Optional Streamlit:
  - do not import Streamlit unless running app,
  - missing Streamlit gives clear install message.

**CLI:**
- `python scripts/export_dashboard.py --qa qa.jsonl --pred predictions.jsonl --eval-report qa-eval-report.json --graph oracle-graph.json --output dashboard/`

**Tests:**
- Stable bundle.
- Missing optional files handled.
- Evidence subgraph extracted.
- Predicted evidence source summaries are exposed in bundle, HTML review rows,
  and local filter metadata.
- Research-axis attribution summaries are exposed in bundle, HTML review rows,
  validation, and local filter metadata.
- Experiment-summary failure-linkage rows are exposed in the dashboard review,
  connecting graph primary metrics and graph diagnostics to QA attribution
  summaries.
- No Streamlit dependency needed for verify.

**Acceptance:**
- `dashboard/index.html` is generated.
- Gold, prediction, evidence, research-axis attribution, source attribution,
  and error are viewable per sample.

### PR 12: Active EQA Task Skeleton

**Status:** Implemented in the current deterministic baseline with task JSONL
helpers, mock graph-step environment, active policies, task metrics, report
digests, `next_best_view` target metadata, budget-vs-success analysis,
candidate-vs-baseline active delta reports, CLI output/validation/comparison,
tests, README/runbook coverage, and verifier evidence.

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
- `sweep_explore`
- `graph_uncertainty`
- `next_best_view`
- `oracle_evidence`

**Metrics:**
- `task_success`
- `answer_accuracy`
- `action_count`
- `evidence_coverage`
- `answer_graph_consistency`
- `budget_analysis` by max-action budget

**API:**
- `active_task_report()`
- `validate_active_task_report()`
- `compare_active_task_report()`
- `active_task_delta_report()`
- `validate_active_task_delta_report()`
- `compare_active_task_delta_report()`

**CLI:**
- `python scripts/run_active_tasks.py --tasks active-tasks.jsonl --graph oracle-graph.json --policy direct_answer --report active-report.json`
- `python scripts/run_active_tasks.py --validate-report active-report.json`
- `python scripts/run_active_tasks.py --compare-report active-report.json`
- `python scripts/run_active_tasks.py --candidate-report next-best-view-report.json --baseline-report direct-answer-report.json --delta-report active-delta-report.json`
- `python scripts/run_active_tasks.py --validate-delta-report active-delta-report.json`
- `python scripts/run_active_tasks.py --compare-delta-report active-delta-report.json`

**Tests:**
- Direct answer works if graph is sufficient.
- Exploration collects evidence in mock env.
- Stable action count.
- Evidence coverage computed.
- `max_actions` enforced.
- Stable candidate-vs-baseline active delta report.

**Acceptance:**
- Mock active EQA tasks run.
- Later AI2-THOR/Habitat env adapters can plug in.

### PR 13: Habitat Adapter Skeleton

**Status:** Implemented in the current deterministic baseline with optional
adapter config, deterministic mock episode generation, observation conversion,
missing-dependency diagnostics, empty optional extra metadata, CLI output,
tests, README/runbook coverage, and verifier evidence.

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

**Status:** Implemented in the current deterministic baseline with explicit
episode-to-oracle-graph-to-QA artifact building, manifest digests, coverage
metadata, optional QA/active/dashboard/graph-eval/predicted-graph experiment artifact refs,
save/load/validation/comparison helpers, CLI build/validate/compare output,
tests, README/runbook coverage, and verifier evidence.

**Task name:** Add benchmark manifest and dataset generation tooling.

**Goal:** Support larger benchmark manifests while default tests use small fixtures.

**Files:**
- Create: `src/dsg_spatialqa_lab/benchmark/manifest.py`
- Create: `src/dsg_spatialqa_lab/benchmark/experiment_summary.py`
- Create: `scripts/build_benchmark.py`
- Create: `scripts/summarize_experiment.py`
- Create: `tests/test_benchmark_manifest.py`
- Create: `tests/test_experiment_summary.py`

**BenchmarkManifest Fields:**
- `schema_version`
- `dataset_name`
- `scene_count`
- `episode_count`
- `qa_count`
- `task_count`
- `graph_digests`
- `qa_dataset_digests`
- optional `experiment_artifacts`
- optional `experiment_artifact_digests`
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
- `python scripts/build_benchmark.py --episodes data/episodes/*.jsonl --output-dir data/benchmark --max-qa-per-episode 100 --qa-eval-report qa-eval-report.json --qa-eval-delta-report qa-delta-report.json --active-task-report active-report.json --active-task-delta-report active-delta-report.json --dashboard-bundle dashboard/dashboard.json --error-attribution-report error-attribution.json --graph-eval-report graph-eval-report.json --predicted-graph-report predicted-report.json --manifest benchmark-manifest.json`
- Add validate/compare manifest commands.
- `python scripts/summarize_experiment.py --manifest benchmark-manifest.json --report experiment-summary.json`
- Add validate/compare experiment summary commands.
- `python scripts/record_experiment.py --summary-report experiment-summary.json --record experiment-record.json`
- Add validate/compare experiment record commands.
- `python scripts/run_mock_experiment.py --output-dir data/mock-experiment --dataset-name mock_experiment --max-qa-per-episode 3 --episode-count 2 --qa-baseline majority --qa-baseline graph_text`

**Tests:**
- Small benchmark deterministic.
- Coverage correct.
- Stable digest.
- Compare detects changed QA.
- Compare detects changed experiment report/dashboard/graph artifacts.
- Experiment summary reports RQ1 spatial QA, RQ2 dynamic memory, RQ3 GraphTool
  query, and RQ4 interactive task lift from manifest-linked delta artifacts.
- Experiment summary validation rejects research-question metric or summary
  count drift after dependent report digests are recomputed.
- Experiment summary validation rejects embedded QA/active delta comparison
  rows that no longer match source artifact keys, paths, and digests.
- Experiment summary validation rejects drifted graph construction diagnostics
  rebuilt from embedded graph eval summaries.
- Experiment summary validation rejects drifted error attribution diagnostics
  rebuilt from embedded attribution summaries, including research-axis failure
  breakdowns.
- Experiment summary validation rejects drifted failure-linkage diagnostics
  rebuilt from graph construction and attribution diagnostics.
- Experiment summary comparison detects referenced QA/active delta and
  graph-eval/attribution drift.
- Experiment summary records deterministic readiness coverage, marking the
  experiment `ready` only when RQ1-RQ4 all have candidate-vs-baseline evidence
  and listing missing research questions or source artifact types otherwise.
- Experiment summary records deterministic per-RQ verdicts from primary metric
  deltas, plus summary verdict counts for improved, unchanged, regressed, and
  inconclusive research questions.
- Experiment record projects a validated summary into a compact final handoff
  ledger with manifest/summary digests, readiness, RQ verdict rows, verdict
  counts, diagnostic ledger counts/keys, source artifact digests, optional
  dashboard metadata, and drift comparison.
- Mock experiment pipeline writes a deterministic local artifact chain from
  one or more mock episodes through final experiment record for smoke-test
  handoffs.
- Mock experiment pipeline writes deterministic predicted graph reports and
  oracle-vs-predicted graph eval reports for each episode and records them in
  the final manifest, summary, dashboard-linked record, and digest ledger.
- Mock experiment pipeline writes a deterministic QA agent matrix, comparing
  the `graph_tool` candidate against one or more local QA baselines with one
  manifest-linked QA delta report per baseline.
- Mock experiment pipeline writes deterministic predicted GraphTool QA
  predictions/reports and an oracle-vs-predicted GraphTool QA delta, so RQ1-RQ3
  measurements include graph-construction impact alongside baseline lift.
- Mock experiment pipeline writes deterministic predicted-graph active-task
  reports and an oracle-vs-predicted active-task delta, so RQ4 measurements
  include graph-construction impact alongside direct-answer baseline lift.
- Benchmark manifests can record offline prediction import reports, and
  experiment summaries/final records project their derived source profiles into
  a compact `source_profile_matrix` for side-by-side imported-source review.
- Dashboard experiment summary review exposes verdicts, readiness, and a
  per-measurement matrix alongside RQ1-RQ4 lift, plus source-profile rows and
  Source Profile filtering for imported prediction sources.
- Experiment record projects per-measurement research-question matrix rows for
  final handoff review and drift comparison.

**Acceptance:**
- Episode to oracle graph to predicted graph to graph eval to oracle/predicted
  GraphTool QA and oracle/predicted active-task evaluation to manifest chain
  works, with optional evaluation reports and dashboard bundles tied into the
  same manifest.
- Manifest-linked experiment summary answers the four project research
  questions with deterministic source artifact digests and drift checks.
- Summary validation proves saved RQ rows are still derivable from embedded
  candidate-vs-baseline delta comparisons.
- Summary validation proves embedded candidate-vs-baseline comparison rows still
  trace back to recorded source artifacts.
- Summary validation proves saved readiness status and missing-RQ lists are
  still derivable from manifest-linked QA/active delta evidence.
- Summary validation proves saved per-RQ verdicts and verdict counts are still
  derivable from manifest-linked QA/active delta evidence.
- Experiment record validation and comparison prove saved final handoff verdict
  rows remain synchronized with the explicit summary/dashboard artifacts.
- Mock experiment pipeline produces a ready final record and dashboard without
  real simulator, network, VLM, current-time, or random dependencies.

### PR 15: README And Roadmap Docs

**Status:** Implemented in the current deterministic baseline with
architecture, roadmap, and benchmark/artifact format docs; README
documentation map; current-state roadmap separation; verification record
coverage; and passing `python scripts/verify.py` evidence for this docs-only PR.

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

## Recommended End-To-End Target After PR 9

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

python scripts/build_predicted_graph.py \
  --mock \
  --input mock-episode.jsonl \
  --output-graph predicted-graph.json \
  --report predicted-report.json

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

python scripts/evaluate_graphs.py \
  --oracle oracle-graph.json \
  --predicted predicted-graph.json \
  --report graph-eval-report.json

python scripts/attribute_errors.py \
  --gold qa.jsonl \
  --oracle-graph oracle-graph.json \
  --predicted-graph predicted-graph.json \
  --predictions predictions.jsonl \
  --report error-attribution.json

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

1. P0: Episode JSONL + IO - complete.
2. P1: Mock episode to oracle DSG - complete.
3. P2: AI2-THOR adapter skeleton + mock fallback - complete.
4. P3: Relation engine extension - complete.
5. P4: QA generator - complete.
6. P5: QA metrics - complete.
7. P6: Baseline runner - complete.
8. P7: Graph metrics - complete.
9. P8: Predicted DSG skeleton - complete.
10. P9: Error attribution - complete.
11. P10: Dashboard MVP - complete.
12. P11: Active EQA skeleton and task metrics - complete.
13. P12: Habitat adapter skeleton - complete.
14. P13: Benchmark manifest and experiment summary tooling - complete.
15. P14: Roadmap and architecture docs hardening - complete.
16. P15: Predicted DSG from explicit detector/RGB-D observation artifacts -
    complete.
17. P16: Real small benchmark readiness gate for AI2-THOR/Habitat artifact
    packages - complete.
18. P17: Real experiment package assembler for explicit local episode and
    report artifacts - complete.
19. P18: Offline real baseline prediction control matrix gate - complete.
20. P19: Predicted DSG RGB-D/detector evidence gate - complete.
21. P20: Real collection evidence gate for AI2-THOR/Habitat package imports -
    complete.
22. P21: Real package final record handoff for ready artifact imports -
    complete.
23. P22: Deterministic real package run/import handoff command for explicit
    local artifacts - complete.
24. P23: Run/import a real small AI2-THOR or Habitat benchmark artifact package
    - next.
25. P24: Deterministic four-way offline baseline prediction import handoff -
    complete.
26. P25: Offline control same-QA readiness gate - complete.
27. P26: Offline control import quality readiness gate - complete.
28. P27: Offline control matrix artifact handoff into real packages - complete.
29. P28: Offline control matrix required-kind readiness alignment - complete.
30. P29a: Offline control real-source metadata readiness gate - complete.
31. P29b: Offline control import-run real-source readiness alignment - complete.
32. P30a: QA delta real-control baseline coverage readiness gate - complete.
33. P30b: Active-task delta readiness evidence gate - complete.
34. P30c: Graph-eval and error-attribution readiness evidence gate - complete.
35. P30d: Predicted-graph report readiness and diagnostic digest alignment gate
    - complete.
36. P30e: Dashboard bundle readiness evidence gate - complete.
37. P29c: Offline control import QA eval/delta handoff - complete.
38. P29d: Offline control standard `QAPrediction` source input support -
    complete.
39. P31a: Detector/RGB-D JSONL to observation sequence import handoff -
    complete.
40. P31b: Detector/RGB-D JSONL to predicted DSG evidence run handoff -
    complete.
41. P29e: Offline control import manifest handoff - complete.
42. P29f: Real experiment run consumes offline-control import manifest -
    complete.
43. P31c: Real experiment run consumes predicted DSG detector-run manifest -
    complete.
44. P32a: Top-level real experiment run manifest handoff - complete.
45. P32b: Real experiment run manifest preflight gap report - complete.
46. P29g: Offline control import manifest content preflight - complete.
47. P31d: Predicted DSG detector-run manifest content preflight - complete.
48. P31e: Predicted DSG detector artifact contract save/load handoff -
    complete.
49. P31f: Predicted DSG detector-run ledger save/load/validate/compare handoff
    - complete.
50. P29h: Offline control result report for four-way DSG-vs-control deltas -
    complete.
51. P29i: Real package consumes offline control result reports - complete.
52. P29j: Standalone offline prediction import CLI accepts standard
    `QAPrediction` JSONL inputs - complete.
53. P32c: Real experiment handoff manifest-set writer - complete.
54. P32d: Real experiment handoff writer saves preflight checklist reports -
    complete.
55. P32e: Real experiment handoff writer saves compact artifact checklists -
    complete.
56. P29k: Offline control preflight exposes four-way prediction artifact
    contracts - complete.
57. P29l: Offline control artifact contracts save/load handoff - complete.
58. P29m: Offline control artifact contract validation and manifest drift
    compare - complete.
59. P29n: Offline control manifest-import run ledger save/load/validate/compare
    handoff - complete.
60. P32f: Real experiment handoff carries child run ledger paths - complete.
61. P32g: Real experiment artifact checklist track summary for real data,
    controls, predicted DSG, review, and run outputs - complete.
62. P32h: Real experiment handoff saves external artifact contracts for
    collection producers - complete.
63. P32i: Real experiment external artifact contracts digest/load/validate/
    compare handoff - complete.
64. P32j: Real experiment external artifact launch readiness report - complete.
65. P29o: Offline control artifact launch readiness report - complete.
66. P31g: Predicted DSG detector artifact launch readiness report - complete.
67. P31h: Predicted DSG detector launch report handles missing/invalid current
    detector inputs - complete.
68. P29p: Offline control launch report includes candidate DSG prediction
    blockers - complete.
69. P32k: Real experiment launch report exposes child offline-control and
    predicted-DSG launch gates - complete.
70. P32l: Real experiment launch report exposes real collection child gate
    commands with AI2-THOR/Habitat source-kind and minimum-frame thresholds -
    complete.
71. P29q: Offline control launch source rows expose single-source normalization
    commands and original source metadata - complete.
72. P31i: Predicted DSG detector launch report exposes explicit detector JSONL
    build command - complete.
73. P31j: Predicted DSG detector launch report exposes build plan and
    actionable blockers - complete.
74. P31k: Predicted DSG detector launch report exposes external detector
    intake plan - complete.
75. P31l: Predicted DSG detector preflight verifies frame asset receipt -
    complete.
76. P32m: Real experiment launch report exposes actionable blocker rows with
    child gates - complete.
77. P32n: Real experiment launch report exposes review-artifact child gate
    commands - complete.
78. P29r: Offline control launch report exposes source import plan and
    actionable blockers - complete.
79. P32o: Real experiment launch report exposes ordered external artifact
    intake plan - complete.
80. P32p: Real experiment launch report exposes real-data collection intake
    plan - complete.
81. P32q: Real experiment launch report exposes primary evidence intake
    plan - complete.
82. P32r: Real collection report verifies frame asset receipt - complete.
83. P32s: Real experiment launch report projects real collection receipt
    failures - complete.
84. P32t: Real experiment launch report projects offline-control receipt
    failures - complete.
85. P32u: Real experiment launch report projects predicted-DSG detector
    receipt failures - complete.
86. P32v: Real experiment launch readiness is gated by primary evidence
    receipts - complete.
87. P29s: Offline control launch report exposes external prediction intake
    plan - complete.
88. P29t: Offline control manifest exports external prediction request bundle
    templates - complete.
89. P31m: Predicted DSG detector manifest exports external detector request
    bundle templates - complete.
90. P32w: Real collection intake exports external collection request bundle
    templates - complete.
91. P29u: Offline control manifest exports returned prediction receipt bundles -
    complete.
92. P31n: Predicted DSG detector manifest exports returned detector receipt
    bundles - complete.
93. P29: Import real four-way offline baseline prediction result files - next
    external-artifact milestone.

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

Complete PR 1-7. This milestone is complete in the current deterministic
baseline.

**Acceptance:**
- Mock episode to oracle graph to QA to GraphTool baseline to QA eval runs end-to-end.
- At least 100 mock QA are generated.
- GraphTool baseline reaches high accuracy on oracle-generated QA.
- GraphTool baseline reaches high accuracy on oracle-generated QA.

### Milestone B: Graph Construction Evaluation

Complete PR 8-10. This milestone is complete in the current deterministic
baseline.

**Acceptance:**
- Oracle graph and predicted mock graph can be compared.
- Object recall, relation F1, and error attribution are reported.
- QA errors can be attributed to missing object, missing relation, or reasoning.

### Milestone C: Interactive Skeleton

Complete PR 11-12. The deterministic dashboard, mock active task report, and
static active-task review panels are implemented, including budget-vs-success
analysis by max-action budget, per-action evidence snapshots, and
`next_best_view` action-target metadata; media previews and real
simulator-backed next-best-view execution remain follow-up work.

**Acceptance:**
- Mock active EQA task runs.
- Task success, action count, and evidence coverage are reported.
- Dashboard can show active task results, budget analysis, and action evidence
  snapshots.
- `next_best_view` can deterministically target missing required evidence.

### Milestone D: Benchmark And Simulator Extension

Complete PR 13-14.

**Acceptance:**
- AI2-THOR and Habitat adapters both have optional skeletons.
- Benchmark manifest summarizes multiple episodes, graphs, and QA datasets.

## Current Codex Task

Continue beyond the completed deterministic validation-loop MVP by prioritizing
real experiment evidence over additional framework scaffolding. The three most
important remaining tracks are:

1. Collect a small but real AI2-THOR or Habitat benchmark and record explicit
   episode, oracle graph, QA, active-task, graph-eval, attribution, manifest,
   summary, and dashboard artifacts.
2. Import real offline predictions for VLM-only, multi-frame VLM,
   caption-memory, and graph-text LLM controls, then compare them with DSG
   GraphTool and predicted-DSG GraphTool rows through the existing
   `source_profile_matrix` and dashboard filters.
3. Build predicted DSGs from explicit RGB-D or detector-output artifacts. The
   completed P15 slice adds `SceneObservation` sequence input to
   `scripts/build_predicted_graph.py`, making the default runtime ready to
   evaluate detector-produced predicted graphs without importing detector
   models.
4. Check whether a candidate real experiment package is actually ready to
   answer the research question. The completed P16 slice adds
   `scripts/check_real_experiment.py`, which fails mock, undersized, missing
   control, missing predicted-DSG, missing attribution, or missing review
   artifact packages instead of letting them be mistaken for evidence.
5. Assemble externally collected real experiment artifacts into the canonical
   manifest plus readiness-report pair. The completed P17 slice adds
   `scripts/assemble_real_experiment.py`, which accepts explicit episode JSONL
   files plus local QA/graph/active/offline-prediction/predicted-graph reports,
   writes a benchmark manifest, writes a readiness report, and returns non-zero
   unless the package passes the readiness gate.
6. Gate the real offline control matrix before treating imported predictions as
   evidence. The completed P18 slice adds `scripts/check_offline_controls.py`,
   which requires VLM-only, multi-frame VLM, caption-memory, and graph-text LLM
   import reports with complete gold-case coverage, clean import diagnostics,
   unique source keys, and a shared QA digest.
7. Gate observation-sequence predicted DSGs before treating them as real
   RGB-D/detector evidence. The completed P19 slice adds
   `scripts/check_predicted_dsg.py`, which requires a multi-frame observation
   sequence, detector/RGB/depth evidence, digest consistency, and no mock
   sources, then wires that evidence report into manifest, package assembly,
   and readiness checks.
8. Gate externally collected episode JSONL files before treating them as real
   AI2-THOR/Habitat data. The completed P20 slice adds
   `scripts/check_real_collection.py`, which requires supported source kind,
   real collection metadata, RGB/depth/segmentation evidence on each frame,
   minimum scene/episode/frame counts, clean episode digests, and no mock
   markers, then wires that report into manifest, package assembly, and
   readiness checks.
9. Seal ready real experiment packages into the final experiment record without
   rerunning evaluation. The completed P21 slice extends
   `scripts/record_experiment.py` with `--real-readiness-report`, carries the
   linked readiness report digest/status, failed checks, missing groups, and
   manifest digest into `experiment-record.json`, and makes record comparison
   detect drift in the referenced readiness artifact.
10. Run the deterministic import handoff for explicit local real packages. The
   completed P22 slice adds `scripts/run_real_experiment.py` and
   `run_real_experiment_package`, composing assembly, readiness, experiment
   summary, and final record generation. It writes summary/record only when the
   real readiness report is ready, links the readiness digest into the record,
   and returns non-zero diagnostics for unready packages.
11. Import the real four-way offline baseline control set through one
   deterministic local-artifact handoff. The completed P24 slice adds
   `scripts/run_offline_controls.py` and `run_offline_control_imports`, which
   consume explicit VLM-only, multi-frame VLM, caption-memory, and graph-text
   prediction JSONL files, write per-source `QAPrediction` and import reports,
   and write the offline control matrix report without calling providers.
12. Enforce same-QA evidence alignment for real offline controls. The
   completed P25 slice adds manifest-level `qa_digest` and makes
   `scripts/check_real_experiment.py` fail packages whose offline control
   import report `qa_digest` does not match the benchmark manifest `qa_digest`.
13. Reject incomplete or diagnostic-bearing offline controls at real readiness.
   The completed P26 slice makes `scripts/check_real_experiment.py` fail
   packages whose offline control import reports miss gold QA cases, import
   fewer predictions than the gold QA count, or contain unknown, duplicate, or
   error prediction diagnostics.
14. Carry the offline control matrix report through real package handoff. The
   completed P27 slice adds `offline_control_matrix_report` manifest artifacts,
   `--offline-control-matrix-report` support in benchmark/real package CLIs,
   and a real-readiness check requiring the matrix report itself to be ready.
15. Require the offline control matrix to gate the same controls as readiness.
   The completed P28 slice makes `scripts/check_real_experiment.py` fail
   packages whose offline control matrix `required_source_kinds` do not cover
   the requested real-readiness control kinds.
16. Reject placeholder offline control source identities before accepting a
   package as real evidence. The completed P29a slice makes
   `scripts/check_real_experiment.py` fail packages whose offline prediction
   import reports are missing real-source `model_id`, `prompt_id`, or
   `dataset_id` metadata, or whose source names/profiles still contain obvious
   fixture/mock/placeholder/synthetic markers. This prevents local test
   fixtures from passing the real offline-control gate while the actual four-way
   prediction JSONL files remain to be imported.
17. Align the deterministic four-way offline import command with real-readiness
   source identity rules. The completed P29b slice makes
   `scripts/run_offline_controls.py` return aggregate readiness that includes
   both the offline control matrix and per-source real metadata checks, exposes
   `matrix_readiness` and `source_metadata_summary` separately for audit, and
   returns non-zero when imported controls still lack `model_id`, `prompt_id`,
   or `dataset_id` metadata, or when their source identity still looks like a
   fixture/mock/placeholder handoff.
18. Require QA delta evidence to cover the same controls requested by real
   readiness. The completed P30a slice makes
   `scripts/check_real_experiment.py` load and validate every
   `qa_eval_delta_report`, reject invalid reports, case-count mismatches, and
   placeholder candidate/baseline names, and fail packages whose QA delta
   baselines do not cover the readiness-required controls. This prevents a
   package with only one DSG-vs-control delta from being accepted as evidence
   for a multi-control real experiment.
19. Require active-task delta evidence to be a valid interactive-task
   comparison before accepting a real package. The completed P30b slice makes
   `scripts/check_real_experiment.py` load and validate each
   `active_task_delta_report`, reject invalid reports, task-count mismatches,
   and placeholder candidate/baseline names, and expose active delta
   candidate/baseline names plus readiness diagnostics in the artifact summary.
20. Require graph-eval and error-attribution evidence to be valid and aligned
   before accepting a predicted-DSG real package. The completed P30c slice makes
   `scripts/check_real_experiment.py` load and validate every
   `graph_eval_report` and `error_attribution_report`, reject invalid reports,
   expose oracle/predicted/gold/prediction digests plus not-ready paths in the
   artifact summary, and require graph-eval predicted digests to match
   error-attribution predicted graph digests.
21. Require predicted-graph reports to be valid, reproducible from their local
   observation/graph files, and aligned with the diagnostic artifacts before
   accepting a predicted-DSG real package. The completed P30d slice makes
   `scripts/check_real_experiment.py` load, validate, and compare each
   `predicted_graph_report`, expose predicted graph/report digests plus
   not-ready paths in the artifact summary, and require predicted graph digests
   to match both graph-eval and error-attribution predicted graph digests.
22. Require dashboard review artifacts to be valid before accepting a real
   package as ready for review. The completed P30e slice makes
   `scripts/check_real_experiment.py` load and validate every
   `dashboard_bundle`, reject invalid dashboard summaries or digest drift, and
   expose dashboard digests, case counts, readiness, and not-ready paths in the
   artifact summary.
23. Generate QA eval and DSG-vs-control delta handoff artifacts directly from
   the real offline control import command. The completed P29c slice lets
   `scripts/run_offline_controls.py` accept an explicit local
   `--candidate-prediction` JSONL file and write the candidate
   `qa_eval_report`, each control baseline `qa_eval_report`, and one
   `qa_eval_delta_report` per imported source. This does not replace real
   VLM-only, multi-frame VLM, caption-memory, and graph-text prediction
   collection; it removes the manual eval/delta stitching step after those
   files exist.
24. Accept already-normalized standard `QAPrediction` JSONL files as real
   offline control source inputs. The completed P29d slice adds per-source
   `input_format` support to `run_offline_control_imports` and
   `--source-input-format NAME qa_prediction` to
   `scripts/run_offline_controls.py`, while keeping
   `offline_prediction_record` as the default. Import reports record and
   compare the original standard-prediction input digest, so externally
   generated VLM/LLM prediction files can be audited without a separate
   conversion artifact.
25. Run the real-source offline control import handoff from one local manifest.
   The completed P29e slice adds
   `run_offline_control_import_manifest`,
   `OFFLINE_CONTROL_IMPORT_MANIFEST_SCHEMA_VERSION`, manifest loading and digest
   helpers, and `scripts/run_offline_controls.py --manifest`. The manifest
   anchors relative paths to its own directory and records the four control
   source files, source formats, real-source metadata, candidate DSG prediction,
   output directory, QA eval output, and matrix report path. This still does not
   collect VLM/LLM predictions; it makes the real files auditable once they
   exist.
26. Feed offline-control import manifests directly into the real experiment
   run handoff. The completed P29f slice lets
   `run_real_experiment_package` and `scripts/run_real_experiment.py` accept
   `--offline-control-import-manifest`, execute that explicit local manifest
   first, and merge the generated per-source import reports, offline control
   matrix, QA eval reports, and DSG-vs-control QA delta reports into benchmark
   package assembly. This removes another manual path-copy step between
   collecting real prediction files and producing a readiness-gated summary and
   final record.
27. Audit the four-way offline control import manifest contents before writing
   generated artifacts. The completed P29g slice adds
   `OFFLINE_CONTROL_IMPORT_PREFLIGHT_SCHEMA_VERSION`,
   `offline_control_import_manifest_preflight`, and
   `scripts/run_offline_controls.py --preflight-manifest`. The preflight reads
   only the explicit manifest, QA file, four source prediction files, and
   optional candidate prediction file; it reports source coverage,
   missing/unknown/duplicate/error predictions, matrix readiness,
   real-source metadata checks, and planned output paths without writing
   predictions, import reports, QA eval reports, deltas, or the matrix. This
   still does not collect VLM/LLM predictions; it makes the actual P29 import
   run auditable before the real files are committed to the package.
28. Summarize four-way offline control import results into a stable local
   result artifact. The completed P29h slice adds
   `OFFLINE_CONTROL_RESULT_REPORT_SCHEMA_VERSION`,
   `offline_control_result_report`, save/load/validate/compare helpers,
   `scripts/run_offline_controls.py --result-report`, and
   `scripts/check_offline_controls.py --validate-result-report` /
   `--compare-result-report`. The result report links the offline control
   matrix, candidate QA eval report, and four QA delta reports, then records
   per-source candidate exact-match lift/regression rows plus readiness checks
   for required-source-kind delta coverage, candidate digest consistency, and
   matched QA case counts. This still does not collect VLM/LLM predictions; it
   makes externally produced predictions auditable as one DSG-vs-real-controls
   comparison artifact once those files are present.
29. Carry the offline control result report through real package readiness.
   The completed P29i slice lets benchmark manifests record
   `offline_control_result_report` experiment artifacts, lets
   `assemble_real_experiment_package`, `run_real_experiment_package`, and their
   CLIs accept explicit result report paths, merges result reports generated by
   offline-control import manifests into real package assembly, lists result
   reports in top-level run preflight planned outputs, and adds readiness gates
   for ready result reports whose delta rows cover the required control kinds.
   This still does not collect VLM/LLM predictions; it closes the handoff gap
   between importing external predictions and treating their DSG-vs-control
   deltas as real experiment evidence.
30. Import standard `QAPrediction` JSONL through the standalone prediction
   import CLI. The completed P29j slice adds
   `--input-format qa_prediction` to `scripts/import_predictions.py`, exports
   `import_qa_prediction_inputs` plus the input-format constants through the
   public package API, and reports `input_format` in the CLI result payload.
   This still does not collect VLM/LLM predictions; it removes the last
   standalone CLI conversion mismatch when real VLM-only, multi-frame VLM,
   caption-memory, or graph-text LLM tools already emit standard
   `QAPrediction` JSONL.
31. Convert externally produced detector/RGB-D outputs into the existing
   observation-backed predicted DSG path. The completed P31a slice adds
   deterministic detector observation JSONL parsing, detector import reports
   with compare/validate support, and
   `scripts/observations.py --import-detector-jsonl ... --output-sequence ...`.
   This still does not run detectors or depth estimators; it gives real
   detector outputs a local artifact contract that can feed
   `scripts/build_predicted_graph.py --input-kind observation_sequence`.
32. Run the detector/RGB-D predicted-DSG artifact handoff in one local command.
   The completed P31b slice adds `run_predicted_dsg_from_detector_jsonl` and
   `scripts/run_predicted_dsg.py`, which write the observation sequence,
   detector import report, predicted graph JSON, predicted graph report, and
   predicted DSG evidence report from explicit detector JSONL inputs. This
   still depends on externally collected detector outputs; it removes manual
   stitching once those real outputs exist.
33. Feed predicted DSG detector-run manifests directly into the real experiment
   run handoff. The completed P31c slice adds
   `PREDICTED_DSG_DETECTOR_RUN_MANIFEST_SCHEMA_VERSION`, manifest load/digest
   helpers, `run_predicted_dsg_detector_run_manifest`,
   `scripts/run_predicted_dsg.py --manifest`, and
   `scripts/run_real_experiment.py --predicted-dsg-detector-run-manifest`.
   Real runs can now execute the explicit detector JSONL handoff first and
   merge the generated predicted graph report plus predicted DSG evidence
   report into package assembly. This still does not run detectors or depth
   estimators; it removes another manual path-copy step once externally
   collected detector outputs exist.
34. Audit predicted DSG detector-run manifest contents before writing generated
   artifacts. The completed P31d slice adds
   `PREDICTED_DSG_DETECTOR_PREFLIGHT_SCHEMA_VERSION`,
   `predicted_dsg_detector_run_manifest_preflight`, and
   `scripts/run_predicted_dsg.py --preflight-manifest`. The preflight reads
   only the explicit detector-run manifest and detector JSONL file, converts
   observations, builds the predicted graph and graph report in memory, applies
   the same detector/RGB/depth evidence and observation-count checks, and
   reports planned output paths without writing the observation sequence,
   predicted graph, detector import report, predicted graph report, or
   predicted DSG evidence report. This still does not run detectors or depth
   estimators; it makes real detector-output readiness auditable before the
   actual handoff writes package artifacts.
35. Save and reload a compact predicted-DSG detector artifact contract from
   detector-run manifest preflight. The completed P31e slice adds
   `PREDICTED_DSG_DETECTOR_ARTIFACT_CONTRACT_SCHEMA_VERSION`, stable contract
   digests, `save_predicted_dsg_detector_artifact_contract`, and
   `load_predicted_dsg_detector_artifact_contract`, and lets
   `scripts/run_predicted_dsg.py --preflight-manifest ... --artifact-contract ...`
   write the detector/RGB-D input schema, detector input status, build
   thresholds, evidence requirements, readiness summary, and planned
   observation/graph/report output paths. This still does not run detectors or
   depth estimators; it makes externally collected detector JSONL artifacts
   easier to hand off before building a real predicted DSG.
36. Save and audit a compact ledger after a manifest-driven predicted DSG
   detector run. The completed P31f slice adds
   `PREDICTED_DSG_DETECTOR_RUN_LEDGER_SCHEMA_VERSION`,
   `predicted_dsg_detector_run_ledger`, stable ledger digests,
   save/load/validation/current-file comparison helpers, and
   `scripts/run_predicted_dsg.py --manifest ... --run-ledger ...` plus
   `--validate-run-ledger` / `--compare-run-ledger`. The ledger binds the
   detector-run manifest, detector JSONL input, observation sequence,
   predicted graph, detector import report, predicted graph report, and
   predicted DSG evidence report without rebuilding the graph during
   comparison. This still does not run detectors or depth estimators; it makes
   a completed real predicted DSG handoff reproducible and drift-checkable
   before being used as graph-construction evidence.
37. Summarize launch readiness from the predicted DSG detector artifact
   contract. The completed P31g slice adds
   `PREDICTED_DSG_DETECTOR_ARTIFACT_LAUNCH_REPORT_SCHEMA_VERSION`,
   `predicted_dsg_detector_artifact_launch_report`, stable report digests, and
   `scripts/run_predicted_dsg.py --artifact-launch-report ... --manifest ...`.
   The report compares the saved detector artifact contract with the current
   detector-run manifest preflight and summarizes detector/RGB-D input blockers
   before writing observation sequence, graph, report, or evidence artifacts.
   This still does not run detectors or depth estimators; it gives the real
   predicted-DSG intake a current launch gate before the manifest build becomes
   graph-construction evidence.
38. Report missing or invalid current detector/RGB-D inputs through the same
   predicted DSG launch-readiness schema. The completed P31h slice extends
   `predicted_dsg_detector_artifact_launch_report` so a missing or unparsable
   detector JSONL produces a non-ready launch report with `detector_input`
   blockers, a failed current-contract digest, and preserved planned output
   paths instead of a generic exception payload. This keeps the first real
   detector handoff auditable while external detector/RGB-D artifacts are still
   arriving.
39. Expose an explicit detector JSONL build command in predicted DSG launch
   readiness. The completed P31i slice adds `build_command` to
   `predicted_dsg_detector_artifact_launch_report`, spelling out the direct
   `scripts/run_predicted_dsg.py --detector-jsonl ...` command with planned
   observation sequence, predicted graph, detector import report, predicted
   graph report, evidence report, relation/frame options, thresholds, and
   required evidence kinds. This lets detector/RGB-D producers smoke-test the
   current artifact without losing `next_commands.build` as the manifest-driven
   build path.
40. Expose a predicted-DSG detector build plan and actionable blockers. The
   completed P31j slice adds `build_plan` to
   `predicted_dsg_detector_artifact_launch_report`, collecting the detector
   input status, explicit build command, manifest build command, preflight
   command, build requirements, and planned outputs in one section. It also
   adds `actionable_blockers`, listing only detector-input and build-readiness
   blockers, so real detector/RGB-D handoffs can move from launch report to the
   next missing file, readiness threshold, or command without scanning every
   report row. The completed P31k slice adds
   `external_detector_intake_plan`, a detector/RGB-D input checklist with
   detector JSONL status, required schema, readiness state, build thresholds,
   evidence requirements, planned outputs, and the final preflight/build
   commands. The completed P31l slice adds frame asset receipt checks to
   detector-run preflight and launch reports: detector JSONL-declared
   RGB/depth/segmentation paths are anchored beside the detector JSONL,
   summarized as `asset_summary`, and missing assets fail the
   `frame_assets_present` readiness check without opening image/depth files or
   running detectors. The completed P31m slice adds
   `predicted_dsg_detector_request_bundle` plus
   `scripts/run_predicted_dsg.py --detector-request-bundle`, producing a
   manifest-only detector/RGB-D producer bundle with the target detector JSONL
   path, expected detector-observation schema, frame asset fields, build
   thresholds, planned graph/report outputs, a minimal detector-observation
   record template, stable bundle digests, save/load helpers, and a launch
   report command. It does not read detector JSONL or frame assets, so it can
   run before external predicted-DSG evidence files exist. The completed P31n
   slice adds `predicted_dsg_detector_receipt_bundle` plus
   `scripts/run_predicted_dsg.py --detector-receipt-bundle`, producing a
   returned-file detector/RGB-D receipt bundle with manifest digests, detector
   JSONL input digests, observation/object-observation counts, observation
   sequence digests, frame asset receipt summaries, build requirements,
   planned outputs, readiness, stable receipt bundle digests, save/load
   helpers, `validate_predicted_dsg_detector_receipt_bundle`, and a
   launch-report command without writing graph/report/evidence or ledger
   artifacts.
38. Run a full real experiment local handoff from one top-level manifest. The
   completed P32a slice adds
   `REAL_EXPERIMENT_RUN_MANIFEST_SCHEMA_VERSION`,
   `load_real_experiment_run_manifest`,
   `real_experiment_run_manifest_digest`,
   `run_real_experiment_manifest`, and
   `scripts/run_real_experiment.py --run-manifest`. The manifest anchors
   relative paths to its own directory and records episodes, output paths,
   readiness thresholds, required controls, required predicted input kinds,
   review artifacts, the optional offline-control import manifest, and the
   optional predicted DSG detector-run manifest. This still does not collect
   data, produce VLM/LLM predictions, or run detectors; it makes a real small
   experiment package reproducible from one explicit local handoff file once
   those external artifacts exist.
38. Audit a top-level real experiment run manifest before execution. The
   completed P32b slice adds `REAL_EXPERIMENT_PREFLIGHT_SCHEMA_VERSION`,
   `real_experiment_run_manifest_preflight`, and
   `scripts/run_real_experiment.py --preflight-run-manifest`. The preflight
   reads only the explicit run manifest and child manifests, groups declared
   input paths by real collection, offline controls, predicted DSG, and review
   artifacts, reports missing or invalid inputs plus undeclared required
   evidence groups, and lists planned outputs. It does not import predictions,
   build predicted DSGs, assemble benchmark manifests, or write final records;
   it makes the next real-experiment work item auditable by showing exactly
   which externally collected files are still missing.
39. Write the real experiment manifest set from one handoff root. The completed
   P32c slice adds `REAL_EXPERIMENT_HANDOFF_SCHEMA_VERSION`,
   `write_real_experiment_handoff_manifests`, and
   `scripts/run_real_experiment.py --write-handoff-manifests`. The writer
   creates the top-level real experiment run manifest plus the offline-control
   import manifest and predicted-DSG detector-run manifest under one explicit
   local root, using portable relative paths for real episodes, gold QA,
   VLM-only / multi-frame VLM / caption-memory / graph-text prediction files,
   candidate DSG predictions, detector/RGB-D JSONL, real collection reports,
   graph eval, attribution, active-task delta, dashboard review artifacts, and
   planned outputs. It does not collect data, import predictions, build graphs,
   or write review artifacts; it gives the existing run-manifest preflight a
   concrete missing-file checklist for the first real small-scale experiment.
40. Save a real experiment handoff preflight checklist when writing manifests.
   The completed P32d slice makes
   `write_real_experiment_handoff_manifests` call the existing top-level
   run-manifest preflight after writing the child manifests, save the resulting
   `real-experiment-preflight.json` under the handoff root, and return its
   ready status and summary counts in the CLI/API payload. The handoff template
   also records explicit per-source normalized prediction and import-report
   planned output paths for the four offline controls. This still does not
   collect real data or model predictions; it makes the first external-artifact
   collection pass easier to audit from one generated checklist file.
41. Save a compact real experiment artifact checklist from the handoff
   preflight report. The completed P32e slice adds
   `real-experiment-artifact-checklist.json` to the handoff writer output. It
   converts preflight required inputs into `input_artifacts` with
   present/missing status, converts planned outputs into
   `planned_output_artifacts`, preserves source metadata for the four offline
   controls, and returns checklist summary counts through the CLI/API payload.
   This still does not create real artifacts; it makes the external collection
   work queue easier to inspect than the full preflight report.
42. Carry child run ledgers through the top-level real experiment handoff. The
   completed P32f slice lets real experiment run manifests declare
   `offline_control_import_run_ledger_path` and
   `predicted_dsg_detector_run_ledger_path`, includes those paths in preflight
   and artifact-checklist planned outputs, and saves both ledgers when the
   top-level real run executes the offline-control import manifest and
   predicted-DSG detector-run manifest. The real run result now reports each
   generated ledger path and digest, so a small real experiment can prove its
   imported controls and predicted DSG artifacts are reproducible before they
   are used as DSG-vs-control evidence.
43. Summarize the real experiment artifact checklist by external evidence
   track. The completed P32g slice adds `track` to each handoff checklist row
   and `track_summary` / `artifact_track_summary` aggregates for `real_data`,
   `real_controls`, `predicted_dsg`, `review_artifacts`, and `run_outputs`.
   This maps the generated missing-file checklist directly onto the three
   remaining real-experiment gaps: real AI2-THOR/Habitat data, real
   VLM/LLM-control predictions, and real detector/RGB-D predicted DSG inputs.
   It does not create those external artifacts; it makes the first real
   small-scale experiment collection pass easier to drive and audit.
44. Save a static real experiment external artifact contract file from the
   handoff writer. The completed P32h slice adds
   `REAL_EXPERIMENT_EXTERNAL_ARTIFACT_CONTRACTS_SCHEMA_VERSION` and writes
   `real-experiment-external-artifact-contracts.json` beside the generated
   run manifest, preflight report, and checklist. The contract is derived only
   from the generated manifests and checklist, so it does not read missing
   episode, QA, VLM/LLM prediction, detector, or review files. It records the
   real-data requirements, four control source input paths and planned import
   outputs, detector/RGB-D input requirements and planned predicted-DSG
   outputs, review artifact paths, run outputs, and the same track summary.
   This makes the external artifact production handoff shareable before the
   first real small-scale experiment has been collected.
45. Validate and compare the static real experiment external artifact contract.
   The completed P32i slice adds stable contract digests,
   `load_real_experiment_external_artifact_contracts`,
   `validate_real_experiment_external_artifact_contracts`,
   `compare_real_experiment_external_artifact_contracts`, and
   `scripts/run_real_experiment.py --validate-external-artifact-contracts` /
   `--compare-external-artifact-contracts`. Validation checks schema, digest,
   required tracks, source counts, and summary consistency; comparison rebuilds
   the contract from the saved run manifest, child manifests, and checklist
   paths recorded inside the contract, without reading missing real input
   files. This makes the external collection contract drift-checkable before
   real episodes, model predictions, detector outputs, or review artifacts
   have been produced.
46. Summarize launch readiness from the external artifact contract. The
   completed P32j slice adds
   `real_experiment_external_artifact_launch_report`, stable report digests,
   and `scripts/run_real_experiment.py --external-artifact-launch-report`.
   The report validates and compares the saved contract, reruns the current
   top-level run-manifest preflight, and groups remaining blockers under
   `real_data`, `real_controls`, `predicted_dsg`, `review_artifacts`, and
   `run_outputs`. This turns the static handoff into a current launch gate as
   real episode files, VLM/LLM predictions, detector/RGB-D files, and review
   artifacts arrive, without running models or writing experiment outputs.
47. Link top-level real launch blockers to the child handoff launch gates. The
   completed P32k slice adds `child_launch_gates` to
   `real_experiment_external_artifact_launch_report`, pointing to the
   offline-control artifact-contract preflight/launch-report commands and the
   predicted-DSG detector artifact-contract preflight/launch-report commands.
   This lets a single top-level handoff report route `real_controls` and
   `predicted_dsg` blockers directly into the P29/P31 intake checks.
48. Link top-level real data blockers to the real collection intake gate. The
   completed P32l slice adds `child_launch_gates.real_data`, carrying the exact
   `scripts/check_real_collection.py` collection, validate, and compare
   commands, plus explicit AI2-THOR/Habitat source-kind and minimum-frame
   thresholds from the handoff run manifest. This lets the same launch report
   route `real_data` blockers into the P32 real-data intake check before the
   top-level experiment can run. The completed P32r slice adds frame asset
   receipt checks to `real_collection_report`: episode-declared
   RGB/depth/segmentation paths are anchored beside each episode JSONL,
   summarized as `asset_summary`, and missing assets fail the
   `frame_assets_present` readiness check without opening image/depth files.
   The completed P32s slice adds `collection_report_receipt` to
   `real_data_collection_intake_plan`, projecting saved real collection report
   readiness, failed checks, digest, and `asset_summary` into the top-level
   launch report so missing frame assets remain visible after the report file
   itself exists. The completed P32w slice adds
   `real_collection_request_bundle` plus
   `scripts/check_real_collection.py --request-bundle`, producing a
   manifest-only AI2-THOR/Habitat collection request bundle with target
   episode/report paths, required RGB/depth/segmentation frame fields, a
   minimal episode-frame template, stable bundle digest, save/load helpers, and
   launch-report request-bundle commands without reading episode files or
   frame assets. The completed P33x slice makes that real-data request handoff
   auditable before collection starts. `validate_real_collection_request_bundle`
   checks schema, digest, supported source kind, field shape, episode template,
   and command consistency; `compare_real_collection_request_bundle` rebuilds
   the request bundle from its recorded fields to detect drift; and
   `scripts/check_real_collection.py --validate-request-bundle` /
   `--compare-request-bundle` expose those checks without reading episode
   JSONL files, frame assets, simulators, or services.
49. Expose actionable top-level blockers for the real experiment handoff. The
   completed P32m slice adds `actionable_blockers` to
   `real_experiment_external_artifact_launch_report`, listing only non-ready
   tracks with blocking roles, missing/invalid inputs, missing requirements,
   and the matching child launch gate for `real_data`, `real_controls`,
   `predicted_dsg`, and `review_artifacts`. The completed P32n slice adds
   review-artifact child launch gates with explicit active-task delta,
   dashboard bundle, error-attribution, and graph-eval validate/compare
   commands. This lets a single top-level launch report route each current
   real-data, offline-control, predicted-DSG, or review-artifact blocker
   directly to the next intake command without manual cross-referencing. The
   completed P32o slice adds `external_artifact_intake_plan`, ordering the
   blocked tracks, carrying each matching child gate plus recommended command
   keys, and keeping final top-level preflight/run commands in the same launch
   report. The completed P32p slice adds
   `real_data_collection_intake_plan`, a real-data checklist with
   AI2-THOR/Habitat dataset/source identity, episode and collection-report
   paths, minimum episode/scene/frame/QA thresholds, current missing or invalid
   real-data inputs, and request-bundle/collection/validate/compare commands.
   The completed
   P32q slice adds `primary_evidence_intake_plan`, grouping `real_data`,
   `real_controls`, and `predicted_dsg` as the three research-evidence launch
   tracks, with child gates, recommended command keys, readiness, blockers,
   and final top-level commands while excluding review artifacts and run
   outputs. The completed P32t slice adds
   `real_controls_prediction_intake_plan`, projecting the saved
   offline-control artifact contract receipt plus child launch report summary,
   actionable blockers, and `external_prediction_intake_plan` into the
   top-level launch report so incomplete or diagnostically invalid VLM-only,
   multi-frame VLM, caption-memory, and graph-text prediction files remain
   visible after their JSONL paths exist. The completed P32u slice adds
   `predicted_dsg_detector_intake_plan`, projecting the saved predicted-DSG
   detector artifact contract receipt plus child launch report summary,
   actionable blockers, frame `asset_summary`, and
   `external_detector_intake_plan` into the top-level launch report so
   incomplete or diagnostically invalid detector/RGB-D files remain visible
   after the detector JSONL path exists. The completed P32v slice adds
   `primary_evidence_receipt_gate`, making top-level `ready_to_run` require
   all three primary evidence receipts to be ready, while
   `preflight_ready_to_run` still reports the narrower path/manifest preflight
   status. This prevents a launch report from declaring readiness when all
   paths exist but a real-data, real-control, or predicted-DSG child receipt
   is still not ready. The completed P32x slice adds top-level offline
   prediction request/receipt bundle commands and predicted-DSG detector
   request/receipt bundle commands to the child launch gates and recommended
   primary-evidence command lists, so a single real-experiment launch report
   can move directly from requesting real VLM/LLM or detector/RGB-D artifacts
   to auditing their returned receipt bundles without manually cross-referencing
   the child P29/P31 manifests. The completed P32y slice makes the top-level
   launch report read the saved offline-control prediction receipt bundle and
   predicted-DSG detector receipt bundle, validate each bundle's digest, child
   manifest path, and internal receipt consistency, project their
   ready/missing/invalid/not-ready status
   into the real-controls and predicted-DSG intake plans, and keep
   `primary_evidence_receipt_gate` / `ready_to_run` blocked until those
   returned-file bundles are ready. This prevents a handoff whose paths and
   artifact contracts are ready, but whose returned external-file receipts were
   never saved or drifted, from being treated as launch-ready real evidence.
   The completed P32z slice promotes the top-level external-artifact launch
   report into a saved audit artifact with stable digests, save/load helpers,
   validation, current-contract/manifest/receipt comparison, and
   `scripts/run_real_experiment.py --launch-report-output` /
   `--validate-external-artifact-launch-report` /
   `--compare-external-artifact-launch-report` CLI coverage. This gives the
   first real small-scale experiment a reproducible launch-readiness record
   that can be archived and drift-checked before the final run manifest is
   executed.
   The completed P33a slice adds a deterministic real-experiment execution
   packet generated from a saved launch report. The packet freezes the saved
   launch-report digest, current launch-report validation/comparison result,
   primary-evidence blocker summary, audit commands, and final preflight/run
   commands; it leaves `execution_commands` empty until the launch report is
   still current and `ready_to_run` is true. Public save/load/validate/compare
   helpers plus `scripts/run_real_experiment.py --execution-packet` /
   `--execution-packet-output` / `--validate-execution-packet` /
   `--compare-execution-packet` make the first ready local real small-scale
   experiment executable from one archived packet without fabricating real
   data, VLM/LLM predictions, or detector/RGB-D artifacts.
   The completed P33b slice adds a post-run real-experiment execution receipt
   generated from a saved execution packet. It reads the packet's run manifest
   and audits the benchmark manifest, real readiness report, experiment
   summary, experiment record, output directory, offline-control import run
   ledger, and predicted-DSG detector run ledger with existing digest and
   validation helpers. Public save/load/validate/compare helpers plus
   `scripts/run_real_experiment.py --execution-receipt` /
   `--execution-receipt-output` / `--validate-execution-receipt` /
   `--compare-execution-receipt` make the first completed real small-scale
   run reviewable and drift-checkable without rerunning the experiment or
   fabricating missing outputs.
   The completed P33c slice adds a deterministic real-experiment smoke-run
   checklist generated from a saved execution packet. The checklist orders
   packet validation/comparison, launch-report audit commands, final
   preflight/run commands, and post-run execution-receipt
   generation/validation/comparison while keeping not-ready packets audit-only.
   Public save/load/validate/compare helpers plus
   `scripts/run_real_experiment.py --smoke-run-checklist` /
   `--smoke-run-checklist-output` /
   `--smoke-run-checklist-receipt-output` /
   `--validate-smoke-run-checklist` / `--compare-smoke-run-checklist` give the
   first ready real small-scale experiment an archived command checklist before
   any local run is launched.
   The completed P33d slice adds a post-receipt real-experiment research
   review packet. It reads the saved execution receipt plus the validated
   experiment summary and record, then projects RQ1-RQ4 measurement
   availability, verdict counts, source-profile evidence, graph-construction
   diagnostics, and failure-linkage diagnostics without changing the
   experiment conclusions. Public save/load/validate/compare helpers plus
   `scripts/run_real_experiment.py --research-review` /
   `--research-review-output` / `--validate-research-review` /
   `--compare-research-review` make the first real smoke run auditable as
   research evidence, not merely as a set of completed output files.
   The completed P33e slice adds a post-review real-experiment claim-readiness
   report. It reads the saved research review, reloads the execution receipt's
   benchmark manifest, and checks whether the run is still only a pilot or can
   support a claim-ready DSG-vs-control conclusion under explicit thresholds.
   The default gate requires at least 3 episodes, 1 scene, 30 QA cases, and 1
   dynamic QA case plus ready RQ/evidence diagnostics; saved thresholds are
   included in the report so comparison reuses the same claim policy. Public
   save/load/validate/compare helpers plus
   `scripts/run_real_experiment.py --claim-readiness` /
   `--claim-readiness-output` / `--validate-claim-readiness` /
   `--compare-claim-readiness` prevent a tiny real smoke run from being
   mistaken for sufficient benchmark evidence.
   The completed P33f slice adds claim-gap follow-up guidance inside that
   claim-readiness report. `claim_gap_summary` records exact saved-policy scale
   deficits, such as missing episodes or QA cases, and `next_actions` groups
   failed checks back to the real-data, real-controls, predicted-DSG, or
   review-artifact track. This turns a `pilot_only` verdict into an explicit
   next-run expansion target instead of leaving the user to infer which real
   evidence stream should be collected or rerun.
   The completed P33g slice connects those claim gaps to a deterministic next
   handoff plan. `next_handoff_plan` reloads the saved run manifest recorded
   by the execution receipt, compares its current handoff thresholds with the
   saved claim policy, and emits a `write-handoff-manifests` command rooted at
   `next-claim-ready-handoff`. This gives the next real collection/control/
   detector cycle concrete target thresholds without rerunning or mutating the
   completed pilot.
   The completed P33h slice adds deterministic episode collection slots to that
   plan. `episode_collection_plan` keeps the existing episode paths and adds
   placeholder paths for the missing real collection episodes under the next
   handoff root, then includes all of those paths in the generated
   `write-handoff-manifests` command. This moves the next real-data step from
   an abstract episode-count deficit to explicit local files the external
   AI2-THOR/Habitat collector must return.
   The completed P33i slice adds external artifact slots to the same next
   handoff plan. `external_artifact_slots` now exposes deterministic local
   input paths for the candidate GraphTool prediction file, each required
   offline-control prediction file, and the detector/RGB-D JSONL input, and the
   generated `write-handoff-manifests` command carries the candidate and
   detector paths forward. This moves the next real-controls and predicted-DSG
   work from abstract blockers to explicit local files while still avoiding
   fake prediction or detector output generation.
   The completed P33j slice adds an after-write intake plan to the same
   handoff guidance. `after_write_intake_plan` lists the next handoff's run
   manifest, external contract, launch-report, request-bundle, and
   receipt-bundle paths, and emits deterministic commands for real collection
   request/report, offline-control request/receipt bundles, predicted-DSG
   detector request/receipt bundles, run-manifest preflight, contract
   validation, and launch-report validation/comparison. This gives the next
   real small-scale experiment a command sequence immediately after the
   handoff manifests are written, without creating fake external artifacts.
   The completed P33k slice adds a next-run review plan after the launch audit.
   `next_run_review_plan` records deterministic execution-packet, smoke-run
   checklist, execution-receipt, research-review, and claim-readiness paths,
   carries the saved claim thresholds, and emits the matching
   build/validate/compare commands. This makes the next real run auditable from
   launch readiness through post-run claim recheck without redefining success
   around smoke-run thresholds.
   The completed P33l slice adds a consolidated operator checklist to the next
   handoff plan. `operator_checklist.steps` orders the handoff writer, external
   request bundles, returned-artifact audits, launch-report validation,
   execution packet, smoke-run checklist, execution receipt, research review,
   and claim-readiness recheck into one deterministic sequence. This reduces
   operator error for the first real small-scale run while still executing
   nothing during claim-readiness generation.
   The completed P33m slice persists the post-write operator checklist inside
   the generated handoff directory as
   `real-experiment-operator-checklist.json`. That saved checklist starts after
   handoff-manifest writing and orders contract validation, request bundles,
   returned receipt audits, launch audit, execution packet, smoke checklist,
   execution receipt, research review, and claim-readiness recheck. This lets
   the first real small-scale experiment handoff travel with one local command
   queue while still not fabricating real data, real VLM/LLM predictions, or
   detector/RGB-D outputs.
   The completed P33n slice makes that saved operator checklist auditable.
   `real_experiment_operator_checklist_digest`,
   `load_real_experiment_operator_checklist`,
   `validate_real_experiment_operator_checklist`, and
   `compare_real_experiment_operator_checklist` now check schema, digest,
   phase/step counts, consecutive step order, and current command-queue drift.
   `scripts/run_real_experiment.py --validate-operator-checklist` and
   `--compare-operator-checklist` expose the same gate for local operators
   before they execute the real collection/control/predicted-DSG handoff.
   The completed P33o slice adds a deterministic operator progress report for
   that saved checklist. `real_experiment_operator_progress_report` maps each
   checklist step to its target local artifact path, records present/missing
   target status, summarizes counts by track, and identifies the next missing
   step. `scripts/run_real_experiment.py --operator-progress-report` /
   `--operator-progress-output` gives operators a resume view for partially
   filled real handoffs without executing commands or reading missing real
   data, prediction, detector, or review inputs.
   The completed P33p slice makes that saved progress view auditable.
   `load_real_experiment_operator_progress_report`,
   `validate_real_experiment_operator_progress_report`, and
   `compare_real_experiment_operator_progress_report` now check schema, digest,
   target-count consistency, target-status rows, the next missing step, and
   track summaries. `scripts/run_real_experiment.py --validate-operator-progress-report`
   and
   `--compare-operator-progress-report` let operators detect stale resume
   reports after local artifact files appear or disappear, still without
   executing handoff commands or reading missing real inputs.
   The completed P33q slice adds a saved primary-evidence status artifact for
   the three evidence tracks that decide whether a real pilot can launch:
   `real_data`, `real_controls`, and `predicted_dsg`.
   `real_experiment_primary_evidence_status` summarizes launch-report receipt
   state, per-track blockers, recommended commands, and the next blocked track;
   its digest/load/validate/compare helpers and
   `scripts/run_real_experiment.py --primary-evidence-status` /
   `--validate-primary-evidence-status` / `--compare-primary-evidence-status`
   make that reduced three-track view auditable after real collection,
   VLM/LLM-control, or detector/RGB-D files arrive.
   The completed P33r slice adds a top-level primary-evidence request package
   for the same three external producer tracks.
   `real_experiment_primary_evidence_request_package` embeds the
   real-collection request bundle, offline-control prediction request bundle,
   and predicted-DSG detector request bundle when their local inputs are
   available, while blocked rows record the missing-input error and request
   command instead of fabricating QA or prediction templates. Its
   digest/load/validate/compare helpers and
   `scripts/run_real_experiment.py --primary-evidence-request-package` /
   `--validate-primary-evidence-request-package` /
   `--compare-primary-evidence-request-package` give the first real small-scale
   experiment one auditable package to hand to AI2-THOR/Habitat, VLM/LLM, and
   detector/RGB-D producers. The completed P33y slice tightens that top-level
   package by saving a compact `request_bundle_validation` summary for every
   ready child bundle and recomputing those summaries during package
   validation. This prevents a saved primary-evidence request package from
   accepting internally inconsistent real-data, offline-control, or
   predicted-DSG child request bundles merely because their child digest and
   row digest were updated together. The completed P33z slice materializes the
   same verified child requests as producer-facing files.
   `write_real_experiment_primary_evidence_request_bundles` and
   `scripts/run_real_experiment.py --write-primary-evidence-request-bundles`
   write ready embedded real-collection, offline-control prediction, and
   predicted-DSG detector request bundles to their declared local paths, skip
   blocked tracks with explicit status rows, and do not collect episodes,
   generate VLM/LLM predictions, build predicted DSGs, or call external
   services.
   The completed P33s slice adds a primary-evidence return checklist derived
   from that saved request package. `real_experiment_primary_evidence_return_checklist`
   reloads the package's saved launch report, keeps blocked rows pointed at the
   request-bundle command, and turns ready rows into ordered return-artifact
   paths plus receipt/report commands for real collection, offline-control
   prediction receipt bundles, and predicted-DSG detector receipt bundles. Its
   digest/load/validate/compare helpers and
   `scripts/run_real_experiment.py --primary-evidence-return-checklist` /
   `--validate-primary-evidence-return-checklist` /
   `--compare-primary-evidence-return-checklist` make the first real small-scale
   experiment's returned evidence acceptance path auditable before the launch
   report, execution packet, and smoke run are refreshed.
   The completed P33t slice adds a primary-evidence return progress report on
   top of that checklist. `real_experiment_primary_evidence_return_progress_report`
   checks only the explicit returned-artifact paths recorded by the checklist,
   summarizes present/missing counts by the three evidence tracks, preserves
   blocked request rows, and points at the next missing returned artifact or
   request command. Its digest/load/validate/compare helpers and
   `scripts/run_real_experiment.py --primary-evidence-return-progress-report` /
   `--validate-primary-evidence-return-progress-report` /
   `--compare-primary-evidence-return-progress-report` give operators a local
   resume view before they refresh launch readiness from the returned receipt
   bundles.
   The completed P34a slice adds a focused primary-evidence acceptance report
   after that path-presence view.
   `real_experiment_primary_evidence_acceptance_report` reloads the saved
   return progress, return checklist, request package, and current launch
   receipt projections for the three primary evidence tracks, then records
   per-track digest, validation, and manifest-match acceptance status. Its
   digest/load/validate/compare helpers and
   `scripts/run_real_experiment.py --primary-evidence-acceptance-report` /
   `--validate-primary-evidence-acceptance-report` /
   `--compare-primary-evidence-acceptance-report` give operators a single
   three-track content gate before refreshing the launch report, execution
   packet, and smoke run. It still does not collect episodes, generate VLM/LLM
   predictions, run detectors, build predicted DSGs, or call services.
   The completed P34b slice wires those primary-evidence gates into generated
   handoff workflows. `write_real_experiment_handoff_manifests`,
   claim-readiness `next_handoff_plan.after_write_intake_plan`, and the saved
   operator checklist now declare stable paths and ordered commands for
   primary-evidence status, request package creation, child request-bundle
   materialization, return checklist, return progress, and acceptance report
   validation/comparison before execution-packet generation. Operator progress
   maps those new steps to their target files, so the default handoff resume
   view points at the next missing primary-evidence gate instead of leaving
   P33/P34 commands as manual follow-ups. This remains local-only and does not
   run external producers.
   The completed P34c slice makes the same acceptance gate a machine-enforced
   execution-packet prerequisite. `real_experiment_execution_packet` now loads
   the saved primary-evidence acceptance report from the default handoff-root
   path, or from an explicit
   `--execution-packet-primary-evidence-acceptance-report` CLI override,
   records its digest / validation / current-comparison readiness, adds
   acceptance validate/compare audit commands, and leaves final preflight/run
   execution commands empty when the acceptance report is missing, invalid,
   stale, or not fully accepted. This prevents a ready launch report from
   bypassing the three-track primary-evidence content gate before a real pilot
   is authorized.
   The completed P34d slice tightens saved execution-packet validation around
   that same gate. `validate_real_experiment_execution_packet` now rejects
   packets that omit the primary-evidence acceptance validate/compare audit
   commands, even when the packet digest has been recomputed. This keeps a
   tampered or hand-edited packet from preserving `ready_to_execute` while
   hiding the mandatory acceptance-report recheck from the smoke-run checklist.
   The completed P34e slice binds the final run-manifest execution command to
   that approved packet. `real_experiment_execution_packet` now records its
   intended saved packet path and appends
   `--approved-execution-packet <packet>` to the generated
   `run_real_experiment` command; `run_real_experiment_manifest` and
   `scripts/run_real_experiment.py --run-manifest` can enforce that the saved
   packet validates, is ready, and points at the same run manifest before the
   real package assembly is allowed to run. Direct manifest execution remains
   available for legacy/local use, but reports `execution_approval.required` as
   false instead of pretending a packet gate was checked.
   The completed P34f slice makes that approved execution auditable after the
   run. Real run manifests can now declare a top-level
   `real_experiment_run_ledger_path`; `run_real_experiment_manifest` and
   `scripts/run_real_experiment.py --run-manifest` write a deterministic
   `real_experiment_run_ledger` for ready runs, with save/load/validate/compare
   helpers and stable digests. Generated handoff manifests and launch-report
   run commands include `--run-ledger-output`, and execution receipts now treat
   that ledger as a required run-output artifact when declared. A direct
   manifest run can still produce outputs for local use, but its ledger records
   `execution_approval.required=false`, so the post-run receipt will not become
   review-ready unless the ledger proves that an approved execution packet gate
   was actually used.
   The completed P34g slice makes that top-level run ledger a first-class
   smoke-run audit step. `scripts/run_real_experiment.py` now exposes
   `--validate-run-ledger` and `--compare-run-ledger`, and generated
   smoke-run checklists insert those commands immediately after
   `run_real_experiment` and before execution-receipt writing when
   `real_experiment_run_ledger_path` is declared. This keeps the first real
   pilot's post-run evidence chain independently checkable before research
   review or claim-readiness summaries are generated.
   The completed P34h slice carries the same run-ledger audit into the
   next-run command surfaces. Claim-readiness `next_run_review_plan` now
   records the planned top-level run-ledger path plus validate/compare
   commands, and generated operator checklists place those commands after the
   smoke-run checklist and before execution-receipt generation. Operator
   progress reports map both commands to the same planned ledger path, so a
   partially executed real handoff can resume without treating receipt
   generation as the first post-run evidence check.
   The completed P34i slice exposes the actual smoke-run command sequence as a
   saved runbook instead of leaving it buried inside the checklist JSON.
   `real_experiment_smoke_run_runbook` derives a digest-stable command runbook
   from a saved smoke-run checklist, with save/load/validate/compare helpers
   and `scripts/run_real_experiment.py --smoke-run-runbook` /
   `--validate-smoke-run-runbook` / `--compare-smoke-run-runbook`. Generated
   next-run plans and operator checklists now add runbook build/validate/compare
   steps after smoke-checklist validation and before run-ledger auditing. This
   gives the real pilot operator an explicit preflight/run/ledger/receipt
   command queue while still executing nothing automatically.
   The completed P34j slice makes that saved runbook part of the operator
   resume signal instead of treating it as path-only progress.
   `real_experiment_operator_progress_report` now marks smoke-run runbook
   validate/compare steps with `target_ready`, `target_status`, and compact
   validation/comparison audit payloads, adds `all_targets_ready`,
   ready/not-ready counts, and `next_not_ready_step`, and reports stale
   runbooks when the saved checklist has changed. This keeps a real pilot
   resume view from advancing past an outdated command queue while still
   reading only local JSON artifacts and executing no commands.
   The completed P34k slice extends that content-aware progress gate to the
   post-run ledger. Operator progress now marks `validate_run_ledger` and
   `compare_run_ledger` with compact target audits, keeps invalid ledgers
   not-ready, and reports a stale ledger when the saved ledger still validates
   but its recorded run manifest no longer matches current local files. This
   protects the first real pilot's post-run evidence chain before execution
   receipt generation, while still reading only local JSON artifacts and
   executing no commands.
   The completed P34l slice extends the same operator progress content audit to
   the execution receipt. `validate_execution_receipt` and
   `compare_execution_receipt` steps now report compact target audits,
   invalid receipts stay not-ready, and stale receipts are flagged when the
   saved receipt still validates but no longer matches the current execution
   packet/run-manifest artifact state. This keeps research review generation
   from appearing resumable on a path-only receipt after real pilot outputs or
   manifests have drifted.
   The completed P34m slice extends that operator progress content audit to
   research review packets. `validate_research_review` and
   `compare_research_review` steps now load the saved review, record compact
   target audits, keep invalid reviews not-ready, and flag stale reviews when
   the saved review still validates but the referenced execution receipt has
   changed. This keeps claim-readiness rechecks from appearing resumable on a
   path-only research review after the first pilot evidence chain has drifted.
   The completed P34n slice closes the same content-audit chain at the
   claim-readiness report. `validate_claim_readiness` and
   `compare_claim_readiness` steps now load the saved claim report, expose
   compact validation/comparison audits, keep invalid reports not-ready, and
   mark stale reports when the saved claim still validates but the referenced
   research review has changed. This prevents the final post-pilot resume view
   from treating an outdated claim gate as ready just because the JSON file is
   present.
   The completed P34o slice moves that content-aware operator progress audit
   back to the primary-evidence request stage. `validate_primary_evidence_request_package`
   and `compare_primary_evidence_request_package` now load the saved request
   package, report compact validation/comparison audits, keep invalid packages
   not-ready, and flag stale packages when the real-data, real-control, or
   predicted-DSG receipt state has changed since the package was written. This
   prevents the operator from materializing external request bundles from a
   path-only package after the launch evidence has drifted.
   The completed P34p slice moves the same content-aware progress audit one
   step further upstream to the external artifact launch report.
   `validate_external_artifact_launch_report` and
   `compare_external_artifact_launch_report` now load the saved launch report,
   record compact validation/comparison audits, keep invalid reports
   not-ready, and mark stale reports when the current real-data, real-control,
   or predicted-DSG receipt gate differs from the saved report. This prevents
   the real pilot operator from entering primary-evidence status/request
   package steps from a path-only launch audit after external evidence has
   changed.
   The completed P34q slice extends the content-aware progress audit to the
   primary-evidence acceptance report. `validate_primary_evidence_acceptance_report`
   and `compare_primary_evidence_acceptance_report` now load the saved
   acceptance report, expose compact validation/comparison audits, keep
   invalid reports not-ready, and mark stale reports when the saved
   acceptance report still validates but current returned real-data,
   real-control, or predicted-DSG evidence has changed. This prevents
   execution-packet generation from appearing resumable from a path-only
   acceptance gate after returned primary evidence has drifted.
   The completed P34r slice fills the operator-progress gap between the
   primary-evidence request package and acceptance report.
   `validate_primary_evidence_return_checklist`,
   `compare_primary_evidence_return_checklist`,
   `validate_primary_evidence_return_progress_report`, and
   `compare_primary_evidence_return_progress_report` now load the saved return
   checklist/progress artifacts, expose compact validation/comparison audits,
   keep invalid artifacts not-ready, and mark stale artifacts when refreshed
   request packages or missing returned real-data artifacts change the current
   return state. This keeps operators from generating an acceptance report from
   path-only return artifacts after external evidence has drifted.
   The completed P34s slice closes the remaining primary-evidence
   operator-progress gap at the status artifact.
   `validate_primary_evidence_status` and
   `compare_primary_evidence_status` now load the saved status report, expose
   compact validation/comparison audits, keep invalid reports not-ready, and
   mark stale reports when the current launch receipt gate changes after real
   data, real-control, or predicted-DSG evidence drifts. This prevents the
   real pilot operator from entering request-package generation from a
   path-only primary-evidence status report.
   The completed P34t slice moves the first launch-audit operator progress
   step from path presence to contract validation.
   `validate_external_artifact_contracts` now loads the saved external
   artifact contracts file, exposes a compact validation audit, keeps tampered
   or invalid contracts not-ready, and leaves later launch-report generation
   blocked until the contract handoff itself validates. This prevents the
   first real small-scale experiment from starting its launch audit from a
   path-only external contract file after the contract digest has drifted.
   The completed P34u slice adds the companion comparison gate for saved
   external artifact contracts. `compare_external_artifact_contracts` is now
   part of generated operator checklists and operator-progress target mapping,
   loads the saved contract file, exposes a compact comparison audit, keeps
   invalid comparisons not-ready, and marks stale contracts when the run
   manifest or child manifest/checklist state changes before launch-report
   generation. This prevents the first launch report from being generated
   from a valid-but-outdated contract handoff.
   The completed P34v slice moves the external returned-evidence operator
   progress rows from path presence to content validation.
   `real_collection_report`, `offline_control_prediction_receipt_bundle`, and
   `predicted_dsg_detector_receipt_bundle` now load the saved returned
   evidence artifact, expose a compact validation audit, keep tampered or
   internally invalid returns not-ready, and only mark the external receipt
   row ready when the local validator accepts the saved content. This keeps
   the first real small-scale experiment from advancing through returned real
   data, VLM/LLM control predictions, or detector/RGB-D evidence using
   path-only receipts.
   The completed P34w slice closes the matching outbound request-bundle
   operator-progress gap. The aggregate
   `write_primary_evidence_request_bundles` row now audits the three saved
   child request bundles for real data, real controls, and predicted DSG,
   marks the row missing until all expected bundles exist, marks it invalid
   when any saved request bundle fails its local schema/action/digest
   validation, and exposes per-track request-bundle statuses in the compact
   audit. This keeps external collectors, VLM/LLM prediction producers, and
   detector/RGB-D producers from receiving path-only or tampered request
   templates as the basis for the first real small-scale experiment.
   The completed P34x slice extends content-aware operator progress into the
   smoke-run checklist handoff. `validate_smoke_run_checklist` and
   `compare_smoke_run_checklist` now load the saved checklist, expose compact
   validation/comparison audits, keep invalid checklists not-ready, and mark
   stale checklists when the saved execution packet changes before runbook
   generation. This prevents the first real pilot runbook from being generated
   from a path-only smoke-run checklist after execution approval or run
   commands have drifted.
   The completed P34y slice tightens the first real-evidence intake gates from
   mock-only rejection to broader non-real marker rejection. Real collection
   reports now fail `non_real_markers_absent` when episode ids, scene ids,
   asset paths, or metadata contain synthetic / placeholder / fake / dummy /
   mock markers, and predicted DSG evidence plus detector-run preflight now
   fail `non_real_sources_absent` for synthetic or placeholder detector
   sources. This keeps realistic-looking local artifacts from being counted as
   real pilot data or real RGB-D/detector DSG evidence when they are still
   synthetic handoff material.
   The completed P34z slice tightens the top-level real experiment readiness
   summary for the same evidence tracks. `real_experiment_readiness_report`
   now validates saved real collection reports and predicted DSG evidence
   reports before accepting their embedded `readiness.ready` flag, and marks
   those artifact paths not-ready when the report digest or internal shape is
   invalid. This prevents a tampered local evidence report from being counted
   as real data or real predicted-DSG evidence merely because its saved
   readiness block still says ready.
   The completed P35a slice applies the same package-level validation rule to
   real offline controls. `real_experiment_readiness_report` now validates each
   saved offline prediction import report before accepting its source metadata,
   coverage summary, diagnostics, or QA digest, exposes
   `offline_control_invalid_source_keys`, and fails
   `offline_control_import_reports_valid` when a VLM-only, multi-frame,
   caption-memory, or graph-text import report has a tampered digest or invalid
   internal shape. This prevents corrupted real-control prediction imports from
   being counted as DSG-vs-control evidence while their summary still looks
   complete.
   The completed P35b slice binds predicted-DSG evidence reports to the
   predicted-graph report artifacts they claim to summarize. Top-level
   readiness now exposes
   `predicted_dsg_evidence_predicted_report_digests`, compares those digests
   against the manifest's saved `predicted_graph_report` digests via
   `predicted_dsg_evidence_report_digest_alignment`, and treats evidence
   reports as not-ready when their saved comparison against the current
   predicted-graph report fails. This prevents a real predicted-DSG evidence
   report from one graph from being combined with graph eval, attribution, or
   GraphTool QA evidence from another graph while still looking internally
   ready.
   The completed P35c slice extends current-file comparison into the real
   control QA delta layer. `real_experiment_readiness_report` now runs
   `compare_qa_eval_delta_report` for every saved DSG-vs-control QA delta,
   exposes `qa_delta_stale_paths`, and keeps `qa_delta_reports_ready` false
   when a candidate or baseline QA eval report has changed without rebuilding
   the corresponding delta. This prevents stale VLM-only, multi-frame,
   caption-memory, or graph-text comparison deltas from supporting a
   GraphTool/DSG lift claim after the underlying eval evidence has drifted.
   The completed P35d slice applies the same current-file rule to the aggregate
   offline-control result report. `real_experiment_readiness_report` now runs
   `compare_offline_control_result_report`, exposes
   `offline_control_result_stale_paths`, and keeps
   `offline_control_result_ready` false when the saved matrix, candidate QA
   eval, or per-source delta files have drifted without rebuilding the result
   summary. This prevents a stale source-result matrix from presenting old
   VLM/LLM-vs-DSG comparison rows as current real-control evidence.
   The completed P35e slice extends the current-file rule to interactive-task
   lift evidence. `real_experiment_readiness_report` now runs
   `compare_active_task_delta_report`, exposes `active_delta_stale_paths`, and
   keeps `active_delta_reports_ready` false when the candidate or baseline
   active-task report has changed without rebuilding the delta. This prevents
   stale interactive-task deltas from supporting an RQ4 DSG improvement claim
   after the underlying active-task evidence has drifted.
   The completed P35f slice extends current-file comparison to the
   offline-control matrix itself. `real_experiment_readiness_report` now runs
   `compare_offline_control_matrix_report`, exposes
   `offline_control_matrix_stale_paths`, and keeps
   `offline_control_matrix_ready` false when a saved matrix no longer matches
   the current offline prediction import reports. This prevents stale VLM-only,
   multi-frame, caption-memory, or graph-text source-profile coverage from
   supporting real-control readiness after an import report has changed.
   The completed P35g slice extends current-file comparison to graph
   construction evaluation evidence. `real_experiment_readiness_report` now
   runs `compare_graph_eval_report`, exposes `graph_eval_stale_paths`, and
   keeps `graph_eval_reports_ready` false when a saved graph eval report no
   longer matches the current oracle or predicted graph files. This prevents
   stale predicted-DSG quality metrics from supporting RQ1/RQ3 conclusions
   after the underlying graph artifact has drifted.
   The completed P35h slice extends current-file comparison to QA failure
   attribution evidence. `real_experiment_readiness_report` now runs
   `compare_error_attribution_report`, exposes
   `error_attribution_stale_paths`, and keeps
   `error_attribution_reports_ready` false when a saved attribution report no
   longer matches the current gold QA, oracle graph, predicted graph, or
   prediction files. This prevents stale graph-construction / evidence-missing
   / reasoning attribution rows from supporting RQ1-RQ3 diagnosis after the
   underlying QA or graph artifacts have drifted.
   The completed P35i slice extends current-file comparison to real-data
   collection evidence. `real_experiment_readiness_report` now runs
   `compare_real_collection_report`, exposes `real_collection_stale_paths`, and
   keeps `real_collection_ready` false when a saved collection report no longer
   matches the current episode files. This prevents stale AI2-THOR/Habitat
   collection summaries from supporting a real-data readiness claim after the
   underlying RGB-D / segmentation episode records have drifted.
   The completed P35j slice extends current-file comparison to each real
   offline-control import report. `real_experiment_readiness_report` now runs
   `compare_offline_prediction_import_report`, exposes
   `offline_control_stale_source_keys`, and keeps
   `offline_control_import_reports_valid` false when a saved VLM-only,
   multi-frame, caption-memory, or graph-text import no longer matches the
   current QA, raw prediction input, or normalized prediction file. This
   prevents stale per-source control predictions from supporting DSG-vs-control
   readiness before the matrix or aggregate result layers are rebuilt.
   The completed P35k slice exposes current-file drift for predicted graph
   reports in top-level real-experiment readiness. `real_experiment_readiness_report`
   already rebuilds predicted graph reports via `compare_predicted_graph_report`;
   it now also exposes `predicted_graph_stale_paths` when a saved
   observation-sequence or episode-backed predicted graph report no longer
   matches its current input or exported graph file. This makes detector/RGB-D
   predicted-DSG evidence drift auditable separately from malformed report
   JSON before graph-eval, attribution, and evidence summaries are trusted.
   The completed P35l slice extends the current-file rule to review dashboard
   bundles. `dashboard_bundle` can now record explicit QA, prediction, QA eval,
   graph, and optional review source paths, `compare_dashboard_bundle` rebuilds
   the current bundle from those paths, and
   `real_experiment_readiness_report` exposes `dashboard_bundle_stale_paths`
   while keeping stale dashboards not-ready. This prevents an old dashboard
   from presenting outdated RQ1-RQ4 review rows, source-profile summaries, or
   failure-linkage panels after the underlying evidence files have changed.
   The completed P35m slice closes the remaining manifest-ledger gap in
   top-level readiness. `real_experiment_readiness_report` now recomputes every
   manifest-declared experiment artifact digest, exposes
   `benchmark_manifest_artifact_digest_mismatch_paths`, and fails
   `benchmark_manifest_artifact_digests_current` when the manifest's artifact
   digest ledger has been re-signed or left stale while the underlying report
   file has a different current digest. This prevents an internally consistent
   but stale benchmark manifest from laundering old artifact identities into a
   real-evidence package.
   The completed P35n slice makes the final claim gate more actionable when
   the experiment can run but still cannot answer all four research questions.
   `real_experiment_claim_readiness` now projects missing and inconclusive RQ
   keys plus saved verdict snapshots into the `research_question_availability`
   check and `claim_gap_summary.research_question_gaps`, while validation
   checks the new gap count. This lets operators distinguish scale-only pilots
   from specific dynamic-memory, GraphTool, interactive-task, or spatial-QA
   evidence gaps before expanding the next handoff.
   The completed P35o slice carries those RQ gaps into the next-action layer.
   When `research_question_availability` blocks the final claim gate,
   `next_actions` now includes per-RQ `evidence_targets` with the gap type,
   expected source artifact type, saved verdict, and real-control /
   predicted-DSG / review-artifact tracks that need expansion. The next
   handoff plan folds those target tracks into `tracks_to_expand`, so a
   GraphTool or dynamic-memory verdict gap no longer looks like a generic
   review rerun with no evidence-track guidance.
   The completed P35p slice makes a claim-ready report answer "whether DSG
   improved" instead of merely saying the gate passed. Claim readiness now
   carries `research_question_verdicts` and `claim_conclusion_summary`, with
   per-RQ improved/regressed/unchanged key lists, verdict counts, and a
   conclusion label such as `mixed_improvement`, `regression`, `no_change`,
   `all_improved`, or `pilot_only`. Validation recomputes that summary from
   the saved verdicts and RQ gaps, so a claim-ready smoke threshold can support
   a negative or mixed conclusion without being mistaken for all-RQ
   improvement.
   The completed P35q slice makes that conclusion auditable without reopening
   summary artifacts. Claim readiness now includes `claim_conclusion_evidence`,
   a per-RQ table copied from the saved research review with availability,
   measurement count, primary metric, source artifact type, and verdict.
   Validation checks the evidence table's keys, metric shape, and verdict
   consistency with `research_question_verdicts`, so the conclusion summary can
   be traced back to its saved measurement rows inside the claim artifact.
   The completed P35r slice makes the effect direction and magnitude easier to
   scan inside the same claim artifact. Claim readiness now emits
   `claim_effect_matrix`, one row per RQ with the saved primary metric name and
   value, availability, measurement count, source artifact type, and verdict.
   Validation rebuilds this matrix from `claim_conclusion_evidence`, so a
   recomputed-digest edit to a metric value is rejected before an operator uses
   the final claim to argue for DSG improvement, no-change, or regression.
   The completed P35s slice tightens the same final-claim audit by checking
   that conclusive verdict labels match metric direction. Claim readiness now
   includes `claim_effect_direction_summary`, grouping positive, zero, negative,
   and missing metric rows plus any verdict/metric mismatches. Validation
   rebuilds that summary from `claim_effect_matrix` and fails when, for
   example, an `unchanged` verdict is paired with a positive primary-metric
   delta, even if the saved claim digest has been recomputed.
   The completed P35t slice makes the target hypothesis explicit inside the
   final claim artifact. Claim readiness now includes
   `claim_hypothesis_assessment`, derived from the conclusion and effect
   direction summaries, with a fixed hypothesis string, positive/neutral/
   negative evidence keys, and an assessment such as
   `supported_all_capabilities`, `partial_improvement_observed`,
   `contradicted_by_regression`, `no_improvement_observed`, or `pilot_only`.
   Validation recomputes the assessment, so an edited claim-ready artifact
   cannot relabel a regression-heavy result as full DSG benefit by changing
   only the hypothesis assessment fields.
   The completed P35u slice separates smoke-threshold readiness from
   full-scale benchmark readiness inside the same claim artifact.
   `claim_scope_assessment` now compares the saved claim thresholds with the
   default benchmark thresholds, records default-scale deficits and any
   below-default threshold fields, and exposes a scope label such as
   `pilot_only`, `smoke_threshold_ready`, or `full_scale_claim_ready`.
   Validation recomputes the assessment from saved scale counts and thresholds,
   so an artifact produced with intentionally lowered smoke thresholds cannot
   be edited to permit full-scale DSG benefit claims.
   The completed P35v slice adds deterministic full-scale guidance for
   smoke-threshold-ready claim artifacts. `claim_scope_next_actions` now records
   the default benchmark expansion action, current scale, default thresholds,
   and default-scale deficits when a claim can conclude under saved smoke
   thresholds but still cannot support full-scale DSG benefit claims. Validation
   recomputes that guidance from the saved scope assessment and scale counts,
   so deleting the follow-up action from a recomputed-digest artifact is
   rejected.
   The completed P35w slice turns that scope guidance into an executable
   default-scale handoff plan. `claim_scope_handoff_plan` now records a
   `next-full-scale-claim-handoff` root, default target thresholds, planned
   episode slots, handoff writer command, after-write intake commands,
   next-run review commands, and an operator checklist whenever a
   smoke-threshold-ready artifact still lacks default benchmark scale.
   Validation checks that the scoped plan remains required, points at the
   default deficits/thresholds/tracks, and keeps its handoff command and
   checklist instead of letting a recomputed-digest artifact drop the full-scale
   follow-up.
   The completed P35x slice tightens that scoped handoff validation from
   command presence to command-detail consistency. Claim-readiness validation
   now checks the `write_handoff_manifests` command against the saved
   full-scale handoff root, episode plan, candidate prediction slot, detector
   JSONL slot, and default claim thresholds, and also checks the next
   claim-readiness command thresholds plus the mirrored operator-checklist
   commands. A recomputed-digest artifact that edits the handoff command back
   toward smoke scale is rejected.
   The completed P35y slice binds the same scoped handoff plan to the default
   episode deficit. Claim-readiness validation now checks that the scoped
   `episode_collection_plan` records the expected current episode count,
   target episode count, deficit, and deterministic planned episode paths for
   the `next-full-scale-claim-handoff` root. A recomputed-digest artifact that
   leaves the full-scale command present but removes the planned missing
   episode slots is rejected.
   The completed P35z slice binds the scoped full-scale handoff plan to its
   deterministic external artifact slots. Claim-readiness validation now checks
   that `claim_scope_handoff_plan.external_artifact_slots` keeps the candidate
   GraphTool path, detector/RGB-D JSONL path, offline-control prediction paths,
   and track order under the saved `next-full-scale-claim-handoff` root, and
   that the scoped writer command still names the declared offline-control
   kinds. A recomputed-digest artifact that removes the scoped offline-control
   slots is rejected.
   The completed P35aa slice binds the scoped full-scale handoff plan to its
   downstream intake and review subplans. Claim-readiness validation now checks
   that `claim_scope_handoff_plan.after_write_intake_plan` and
   `claim_scope_handoff_plan.next_run_review_plan` keep deterministic artifact
   paths, command keys, command fragments, track/phase order, claim thresholds,
   and operator-checklist command mirrors for the saved
   `next-full-scale-claim-handoff` root. A recomputed-digest artifact that
   clears the scoped after-write intake commands is rejected.
   The completed P35ab slice binds the scoped full-scale handoff plan to its
   current-threshold metadata and derived threshold-update summary.
   Claim-readiness validation now recomputes
   `claim_scope_handoff_plan.threshold_updates` from saved
   `current_handoff_thresholds` and the default target thresholds, so a
   recomputed-digest artifact that removes the visible default-scale threshold
   increases is rejected.
   The completed P35ac slice binds scoped full-scale handoff provenance.
   Claim-readiness validation now checks that `source_run_manifest_path`
   matches the sibling handoff plan, deterministically owns the
   `next-full-scale-claim-handoff` root, and keeps a SHA-256 source digest. A
   recomputed-digest artifact that points the scoped plan at a different source
   manifest is rejected.
   The completed P35ad slice binds scoped full-scale handoff dataset identity.
   Claim-readiness validation now checks that
   `claim_scope_handoff_plan.dataset_name` matches the sibling handoff plan, so
   a recomputed-digest artifact that rewrites the scoped full-scale expansion
   commands and planned episode filenames to a different dataset is rejected.
   The completed P35ae slice binds scoped full-scale handoff current-threshold
   metadata to the sibling handoff plan. Claim-readiness validation now checks
   that `claim_scope_handoff_plan.current_handoff_thresholds` still matches
   the source handoff thresholds before deriving `threshold_updates`, so a
   recomputed-digest artifact that raises the saved current QA threshold and
   removes the visible QA threshold increase is rejected.
   The completed P35af slice binds scoped full-scale handoff offline-control
   slots to the sibling handoff control-kind set. Claim-readiness validation
   now checks that `claim_scope_handoff_plan.external_artifact_slots` preserves
   every source offline-control kind, so a recomputed-digest artifact that
   drops one control source and removes the matching writer-command fragment is
   rejected.
   The completed P35ag slice binds scoped full-scale handoff predicted-input
   requirements to the sibling handoff plan. Claim-readiness validation now
   checks `claim_scope_handoff_plan.required_predicted_input_kinds` and the
   matching writer-command fragments, so a recomputed-digest artifact that
   drops `--required-predicted-input-kind observation_sequence` from the scoped
   expansion command is rejected.
   The completed P35ah slice binds the sibling next handoff to its full
   offline-control kind set. `next_handoff_plan.required_control_kinds` now
   records the source control kinds, and claim-readiness validation checks that
   the next-handoff offline-control slots and writer-command fragments still
   preserve every required control, so a recomputed-digest artifact that drops
   `graph_text` from the next expansion plan is rejected.
   The completed P35ai slice binds the sibling next handoff to deterministic
   external artifact slot paths. Claim-readiness validation now checks that the
   next-handoff candidate GraphTool prediction, detector/RGB-D JSONL,
   offline-control prediction paths, and track order remain under the saved
   `next-claim-ready-handoff` root, so a recomputed-digest artifact that
   redirects the detector input to a different JSONL file is rejected.
   The completed P35aj slice binds the sibling next handoff's after-write
   intake plan to deterministic artifact paths and command fragments.
   Claim-readiness validation now checks that the next-handoff launch-audit,
   primary-evidence, request-bundle, receipt-bundle, and real-collection
   commands still point at the saved `next-claim-ready-handoff` root, so a
   recomputed-digest artifact that redirects the launch-report output path is
   rejected.
   The completed P36a pilot slice creates
   `handoffs/ai2thor-real-smoke/` as the concrete next experiment handoff and
   fixes relative handoff-root path anchoring in top-level launch/preflight
   helpers. Launch reports now keep child manifest and planned-output paths
   under the saved handoff root instead of duplicating
   `handoffs/ai2thor-real-smoke/`, so the real-data and predicted-DSG request
   bundles can be handed to external collectors/producers without phantom
   missing-file blockers.
   The completed P33u slice tightens the same launch gate for real data:
   top-level `collection_report_receipt` now records `digest_valid` and
   `validation_valid`, and `real_data_collection_intake_plan.ready` remains
   false when a real collection report has a stale or tampered
   `report_digest`, even if the saved readiness block says ready. This keeps
   the first real small-scale experiment from treating an edited collection
   report as accepted primary evidence.
47. Expose per-source artifact contracts for the four real offline controls
   during import-manifest preflight. The completed P29k slice adds
   `dsg-spatialqa-lab.offline-control-artifact-contracts.v1` rows to
   `offline_control_import_manifest_preflight` and
   `scripts/run_offline_controls.py --preflight-manifest`. The preflight now
   reports each VLM-only, multi-frame VLM, caption-memory, and graph-text
   source's expected input schema, input status, source metadata readiness,
   diagnostics, normalized prediction/import output paths, and planned
   candidate-vs-source QA eval/delta paths without writing artifacts. This
   still does not produce VLM/LLM predictions; it makes the real prediction
   result files easier to check before the atomic four-way import.
47. Save and reload offline-control artifact contracts as explicit local handoff
   files. The completed P29l slice adds stable contract digests plus
   `offline_control_artifact_contracts_json`,
   `save_offline_control_artifact_contracts`, and
   `load_offline_control_artifact_contracts`, and lets
   `scripts/run_offline_controls.py --preflight-manifest ... --artifact-contracts ...`
   write just the contract JSON to a caller-supplied path. This still does not
   generate external VLM/LLM predictions; it makes the four-way prediction-file
   contract shareable with the external prediction runner and auditable before
   import.
48. Validate and compare saved offline-control artifact contract files. The
   completed P29m slice adds `validate_offline_control_artifact_contracts`,
   `compare_offline_control_artifact_contracts`, and
   `scripts/check_offline_controls.py --validate-artifact-contracts` /
   `--compare-artifact-contracts ... --manifest ...`. Validation checks schema,
   digest, source-count, ready-count, and source-key consistency; comparison
   rebuilds current contracts from the explicit import manifest preflight and
   reports digest/payload drift. This still does not run external predictors;
   it makes the saved prediction-file contract a first-class auditable artifact
   before the real four-way import.
49. Save and audit a compact ledger after a manifest-driven four-way offline
   control import. The completed P29n slice adds
   `OFFLINE_CONTROL_IMPORT_RUN_LEDGER_SCHEMA_VERSION`,
   `offline_control_import_run_ledger`, stable ledger digests,
   save/load/validation/current-file comparison helpers, and
   `scripts/run_offline_controls.py --manifest ... --run-ledger ...` plus
   `scripts/check_offline_controls.py --validate-run-ledger` /
   `--compare-run-ledger`. The ledger binds the manifest, QA digest, four
   source input files, normalized predictions, import reports, matrix/result
   reports, candidate QA eval report, and per-source QA eval/delta reports
   without rerunning imports during comparison. This still does not generate
   external VLM/LLM predictions; it makes a completed real four-way import
   reproducible and drift-checkable before being used as DSG-vs-control
   evidence.
50. Summarize launch readiness from the four-way offline-control artifact
   contract. The completed P29o slice adds
   `OFFLINE_CONTROL_ARTIFACT_LAUNCH_REPORT_SCHEMA_VERSION`,
   `offline_control_artifact_launch_report`, stable report digests, and
   `scripts/check_offline_controls.py --artifact-launch-report ... --manifest ...`.
   The report validates and compares the saved artifact contract, reruns the
   current import-manifest preflight, and summarizes source-level blockers for
   VLM-only, multi-frame VLM, caption-memory, and graph-text prediction files.
   This still does not create VLM/LLM predictions; it gives the real four-way
   prediction intake a current launch gate before the atomic import command is
   allowed to become evidence.
51. Report candidate DSG prediction blockers in the same offline-control launch
   readiness artifact. The completed P29p slice extends the offline-control
   artifact contract candidate row with `status` and `error`, then projects it
   into `offline_control_artifact_launch_report` with candidate
   `blocking_reasons` and summary counts. This makes missing or invalid
   candidate DSG prediction files visible beside the four external VLM/LLM
   control sources before the real DSG-vs-control import runs.
52. Expose source-scoped normalization commands in offline-control launch
   readiness. The completed P29q slice carries each source's original
   `source_metadata` into the artifact contract and launch report, and adds a
   per-source `source_import_command` for
   `scripts/import_predictions.py`. This gives external VLM/LLM producers a
   deterministic single-source smoke check while preserving
   `next_commands.import` as the atomic four-way manifest import.
53. Expose an offline-control source import plan and actionable blockers. The
   completed P29r slice adds `source_import_plan` to
   `offline_control_artifact_launch_report`, collecting the ordered
   single-source import commands, candidate DSG prediction status, preflight
   command, and atomic manifest import command in one section. It also adds
   `actionable_blockers`, listing only currently blocked source rows or
   candidate prediction rows, so the real four-way VLM/LLM control import can
   move from launch report to the next file or command without scanning every
   source row. The completed P29s slice adds
   `external_prediction_intake_plan`, a four-control prediction-file intake
   checklist with required source kinds, required metadata fields, per-source
   file status, planned normalized outputs, blocked sources, and the final
   preflight/import commands. The completed P29t slice adds
   `offline_control_prediction_request_bundle` plus
   `scripts/run_offline_controls.py --prediction-request-bundle`, producing a
   no-gold-answer request bundle with case IDs, questions, answer types,
   per-source output paths and metadata, empty `qa_prediction` /
   `offline_prediction_record` templates, stable bundle digests, save/load
   helpers, and a launch-report command for external VLM/LLM producers. The
   completed P29u slice adds `offline_control_prediction_receipt_bundle` plus
   `scripts/run_offline_controls.py --prediction-receipt-bundle`, producing a
   returned-file receipt bundle with manifest/QA digests, per-source input
   digests, prediction counts, input readiness, source metadata, planned
   normalized outputs, candidate prediction digest/status, stable receipt
   bundle digests, save/load helpers,
   `validate_offline_control_prediction_receipt_bundle`, and a launch-report
   command without writing normalized predictions, import reports, QA eval
   reports, deltas, or matrix artifacts.
   The completed P33v slice tightens the top-level returned-receipt gate for
   real controls and predicted DSG. The offline-control receipt validator checks
   summary/source/candidate consistency, the predicted-DSG receipt validator
   checks detector/readiness/summary consistency, and the top-level launch
   report now records `validation_valid` for both returned receipt bundles.
   `ready_to_run` stays false when a bundle has a valid digest and matching
   child manifest path but internally inconsistent receipt content.
   The completed P33w slice exposes those receipt validators through child
   handoff CLIs. `scripts/run_offline_controls.py
   --validate-prediction-receipt-bundle` and `scripts/run_predicted_dsg.py
   --validate-detector-receipt-bundle` validate saved returned receipt bundles
   directly, return non-zero for inconsistent content, and do not import
   predictions, build graphs, or write downstream artifacts.

After PR 12, the project has the first complete mock active-task loop:

```text
mock/AI2-THOR episode -> oracle DSG -> predicted DSG -> automatic QA ->
baseline/model prediction -> QA evaluation -> graph evaluation -> error attribution ->
active EQA task report -> static review dashboard
```

## P32 Progress: DSG Memory/Query Evidence Gate

The next-stage VLM gate is now evidence-backed:

- VLM-only P26 semantic success is 49 / 60 = 0.816667, so the >= 50%
  semantic gate is satisfied.
- Strict exact remains 0 / 60 and is tracked as a normalization issue, not as
  the entry gate for DSG optimization.
- Detector-only DSG P30 remains below the VLM baseline at 25 / 60 = 0.416667.

The new P32 evidence report is:

```text
handoffs/ai2thor-real-small/outputs/diagnostics/p32-dsg-memory-query-optimization-evidence.json
```

It records the current DSG failure split:

- 16 failures are relation degradation to `IN_ROOM`;
- 19 failures are target objects missing from the predicted graph.

It also records the diagnostic positive control: support-rich, metadata-backed
coverage DSG P22 reaches 60 / 60 on the same 60-case slice and has +11 paired
wins / 0 losses against VLM P26. This is useful evidence for the memory/query
optimization direction, but it is not a final external-detector-only research
conclusion.

P32/P33 development should therefore focus on reproducing the P22 memory/query
behavior with explicit external detector/RGB-D artifacts:

1. Store target and support observations as append-only evidence before graph
   collapse.
2. Preserve `current_location_id` / `current_location_relation` as detector
   current-location evidence when supplied.
3. Query in this order: explicit detector location, support hypothesis, room
   fallback, structured missing-evidence blocker.
4. Use the P31 detector recall handoff as the next external detector input
   contract.

## P33 Progress: Detector Current-Location Label Alias

The first concrete DSG memory/query code change is now in place:

```text
handoffs/ai2thor-real-small/outputs/diagnostics/p33-detector-current-location-label-alias-report.json
```

External detector/RGB-D producers may return `current_location_id` as a stable
object id or as a unique same-frame support label. For `ON` and `INSIDE`
relations, the observation ingestor now resolves a label alias such as
`countertop` to the unique same-frame observed object id such as
`countertop_1`. If the alias is ambiguous, ingestion raises a structured
`SpatialQAError` instead of guessing.

This improves the memory storage path needed for DSG:

- support/container current-location evidence is less likely to be rejected;
- GraphTool can query explicit detector current-location edges before room
  fallback;
- no QA gold answers, oracle required edges, or evaluator-only fields are read.

Verification completed for this slice:

- `python -m pytest -q tests/test_predicted_graph_builder.py -k current_location_label_alias`
- `python -m pytest -q tests/test_predicted_graph_builder.py`
- `python -m pytest -q tests/test_observations.py tests/test_observations_script.py`
- `python -m pytest -q tests/test_spatial_qa.py -k 'support_fallback or current_location or support_like'`

This does not change the already-saved P30 score by itself. It prepares the
P31/P33 external detector return path so the next detector-only predicted DSG
run can preserve support-rich current-location memory instead of degrading to
`IN_ROOM`.

## P34 Progress: Distance-Based Alias Disambiguation

The detector current-location alias path now handles a more realistic external
producer output pattern:

```text
handoffs/ai2thor-real-small/outputs/diagnostics/p34-detector-current-location-alias-distance-report.json
```

If an external detector returns a common support label such as `countertop` for
`current_location_id`, and multiple same-label supports are present in the same
frame, the observation ingestor resolves the alias only when one candidate is
clearly nearest by bbox surface distance. Near ties still raise
`SpatialQAError`; the producer must return a stable support id or stronger
evidence in those cases.

This is a DSG memory/query improvement, not a score claim:

- it keeps explicit `detector_current_location` support edges from being
  dropped when the returned support label is common but spatially clear;
- it still uses only same-frame detector/RGB-D observations;
- it does not read gold answers, oracle required edges, or evaluator-only
  fields;
- it does not change already-saved P30 metrics until a new detector-only
  observation sequence is imported and evaluated.
- full verification passed with `python scripts/verify.py` after this change
  (`796` pytest cases passed, evaluation suite `52/52` passed).

## P35 Progress: Object-Location Query Diagnostics

The DSG query path now exposes why an object-location query fell back to a
room-level answer when diagnostics are explicitly requested:

```text
handoffs/ai2thor-real-small/outputs/diagnostics/p35-object-location-query-diagnostics-report.json
```

`object_location` accepts `include_diagnostics=true` and then returns a
`query_diagnostics` object with:

- `location_evidence_status`;
- `missing_evidence`;
- whether room fallback or support fallback was applied;
- same-frame detector/RGB-D support candidate count;
- stable support candidate rows with id, label, distance, and evidence kinds.

Default answers are unchanged, so saved prediction JSONL and existing QA eval
schemas are not polluted by diagnostic-only fields. The new diagnostics
separate two important DSG failure modes that previously looked identical:

- no detector support candidate was stored for the target;
- multiple same-frame support candidates existed but were too ambiguous to
  choose safely.

This is a query observability improvement, not a score claim. It should be used
on the next detector-only DSG rerun to decide whether failures are primarily
detector recall problems, memory storage problems, or query disambiguation
problems before changing the answer policy.

Full verification passed with `python scripts/verify.py` after this change
(`798` pytest cases passed, evaluation suite `52/52` passed).

## P36 Progress: Apply Query Diagnostics To P30 Detector-Only DSG

The P35 diagnostic path has now been applied to the current detector-only P30
run:

```text
handoffs/ai2thor-real-small/outputs/diagnostics/p36-dsg-object-location-query-diagnostics.json
```

The new reusable API is:

```text
object_location_query_diagnostic_report
```

It consumes an explicit predicted graph, QA dataset, and optional semantic eval
report. It reruns only local GraphTool object-location queries with
`include_diagnostics=true`, then writes a stable digest-backed report. It does
not call a model, detector, simulator, or network service.

P36 splits the 60-case object-location slice as follows:

- `query_error`: 19 cases;
- `support_fallback_missing`: 35 cases;
- `support_fallback_applied`: 4 cases;
- `explicit_location_edge`: 2 cases.

The semantic mismatch split is more useful for the next DSG optimization:

- 19 mismatches are `query_error`, matching the target-object-missing failure
  mode;
- 15 mismatches are `support_fallback_missing`, meaning the target exists but no
  same-frame detector/RGB-D support candidate was available to query;
- 1 mismatch has an explicit location edge but still disagrees semantically.

This makes the next optimization priority concrete:

1. Improve detector/RGB-D target recall for the 19 missing target cases.
2. Improve support object recall and current-location storage for the 15
   support-missing mismatch cases.
3. Inspect the single explicit-edge mismatch before changing query policy.

Full verification passed after P36 with `python scripts/verify.py` (`799`
pytest cases passed, evaluation suite `52/52` passed).

## P37 Progress: Target-Alias Query Robustness

The current next-stage goal is explicit: VLM-only must first be a meaningful
baseline, then DSG optimization should focus on memory content, storage,
query content, and query method.

The VLM gate is already satisfied by the P26 semantic evaluator:

- VLM-only P26 semantic match: `49 / 60 = 0.816667`.
- VLM-only P26 strict exact match: `0 / 60 = 0.000000`.
- Stage decision: semantic match is the entry gate; strict exact remains an
  answer-normalization diagnostic.

P37 adds a conservative target-alias query path for detector-only DSG:

- GraphTool baseline now enriches `object_location` questions with the
  case-level `scene_id` and `step`.
- `SpatialQAEngine` can resolve a missing requested object id to exactly one
  detector/RGB-D object in the same scene and step when the normalized label
  matches and the candidate has `rgb/depth/detector` evidence.
- Ambiguous same-label candidates raise `SpatialQAError`; the query path does
  not guess.
- The path does not read gold answers, oracle required edges, or
  evaluator-only fields.

P37 artifacts:

```text
handoffs/ai2thor-real-small/inputs/candidate/predicted-graph-tool-independent-p37-target-alias.jsonl
handoffs/ai2thor-real-small/outputs/diagnostics/dsg-candidate-semantic-eval-p37-target-alias-independent.json
handoffs/ai2thor-real-small/outputs/diagnostics/p37-dsg-target-alias-query-diagnostics.json
```

P37 result:

- DSG semantic match remains `25 / 60 = 0.416667`.
- Strict exact match remains `0 / 60 = 0.000000`.
- Query errors drop from `19` to `15`.
- `support_fallback_missing` rises from `35` to `38`, showing that some
  formerly missing targets can now be resolved, but still lack enough support
  or current-location evidence to answer correctly.

P37 is therefore a robustness improvement, not a DSG success result. The next
optimization should not be more answer formatting. It should target:

1. Detector/RGB-D target recall for the remaining `15` query-error mismatches.
2. Support/current-location memory for the `18` support-missing semantic
   mismatches in P37.
3. Explicit-edge correctness for the `2` explicit-location mismatches.

Verification after P37:

```text
python -m ruff check src/dsg_spatialqa_lab/qa.py src/dsg_spatialqa_lab/agents/graph_tool_agent.py src/dsg_spatialqa_lab/eval/dsg_query_diagnostics.py tests/test_spatial_qa.py tests/test_baselines.py tests/test_dsg_query_diagnostics.py
python -m mypy src/dsg_spatialqa_lab/qa.py src/dsg_spatialqa_lab/agents/graph_tool_agent.py src/dsg_spatialqa_lab/eval/dsg_query_diagnostics.py tests/test_spatial_qa.py tests/test_baselines.py tests/test_dsg_query_diagnostics.py
python -m pytest -q tests/test_spatial_qa.py tests/test_baselines.py tests/test_dsg_query_diagnostics.py
python scripts/verify.py
```

Fresh full verification passed with `802` pytest cases and evaluation suite
`52/52` passed.

## P38 Progress: Query-Diagnostic Detector Recall Handoff

P37 showed that target aliasing reduces `query_error` but does not improve
semantic accuracy because the newly resolved cases still lack support or
current-location evidence. P38 turns that failure split into the next
detector/RGB-D intake request.

New entry point:

```text
dsg_detector_recall_handoff_from_query_diagnostics
scripts/build_dsg_detector_recall_handoff.py --query-diagnostic-report ...
```

The builder consumes the P37 object-location query diagnostic report and a
local frame-index JSONL. It includes only unresolved semantic mismatches whose
query status is:

- `query_error`;
- `support_fallback_missing`.

It emits the same detector recall handoff schema as the existing P31 path. The
output is still gold-free: validation checks reject evaluator-only fields such
as `gold_answer`, `gold_support_label`, `required_edges`, and `evidence_nodes`.

P38 artifact:

```text
handoffs/ai2thor-real-small/inputs/predicted-dsg/p38-detector-recall-handoff-from-p37-query-diagnostics.json
```

P38 handoff summary:

- `case_count`: 33;
- `frame_count`: 16;
- `frames_with_support_labels`: 14;
- `missing_frame_case_count`: 0;
- `requested_detection_label_count`: 52;
- `support_label_count`: 20;
- `target_label_count`: 33.

This is the next external detector/RGB-D request: rerun detector outputs on
these 16 frames, returning stable target object ids plus support/current
location evidence where visible. After import, rebuild the observation-backed
predicted graph and rerun the P37 semantic eval plus paired delta against VLM
P26.

Verification after P38:

```text
python -m pytest -q tests/test_dsg_detector_recall_handoff.py
python -m ruff check src/dsg_spatialqa_lab/eval/dsg_detector_recall.py scripts/build_dsg_detector_recall_handoff.py tests/test_dsg_detector_recall_handoff.py
python -m mypy src/dsg_spatialqa_lab/eval/dsg_detector_recall.py scripts/build_dsg_detector_recall_handoff.py tests/test_dsg_detector_recall_handoff.py
python scripts/build_dsg_detector_recall_handoff.py --validate-report handoffs/ai2thor-real-small/inputs/predicted-dsg/p38-detector-recall-handoff-from-p37-query-diagnostics.json
```

## P39 Progress: Compatible-Step Target Alias

P38 identified a remaining query-method defect: some observation-aware QA case
steps use encoded values such as `100040`, while the detector-only predicted
graph stores the same frame as step `40`. P37 required exact step equality, so
valid same-scene/same-label detector nodes were still treated as missing.

P39 updates target alias resolution for `object_location` queries:

- candidate target nodes may match either `step` or `step % 100000`;
- matching still requires same `scene_id`, normalized label match, and
  `rgb/depth/detector` evidence;
- more than one compatible candidate remains an ambiguity error;
- no gold answer, oracle edge, or evaluator-only field is read.

P39 artifacts:

```text
handoffs/ai2thor-real-small/inputs/candidate/predicted-graph-tool-independent-p39-compatible-step-target-alias.jsonl
handoffs/ai2thor-real-small/outputs/diagnostics/dsg-candidate-semantic-eval-p39-compatible-step-target-alias-independent.json
handoffs/ai2thor-real-small/outputs/diagnostics/p39-dsg-compatible-step-target-alias-query-diagnostics.json
handoffs/ai2thor-real-small/outputs/diagnostics/dsg-p39-vs-p37-target-alias-semantic-delta.json
handoffs/ai2thor-real-small/outputs/diagnostics/dsg-p39-vs-vlm-p26-affordance-option-fallback-semantic-delta.json
```

P39 results:

- semantic match improves from P37 `25 / 60 = 0.416667` to
  `27 / 60 = 0.450000`;
- query errors drop from P37 `15` to `13`;
- P39 vs P37 paired delta: `2` wins, `0` losses, `58` ties;
- P39 vs VLM P26 remains negative: DSG `27 / 60`, VLM `49 / 60`;
- P39 vs VLM P26 paired delta: `4` wins, `26` losses, `30` ties.

This is a real DSG query-method improvement, but still not a DSG superiority
result. The next bottleneck is evidence coverage, not query formatting:

- `13` semantic mismatches remain `query_error`;
- `18` semantic mismatches remain `support_fallback_missing`;
- `2` semantic mismatches have explicit location edges and need edge-quality
  review.

Targeted verification after P39:

```text
python -m pytest -q tests/test_spatial_qa.py -k 'compatible_step_label'
python -m pytest -q tests/test_spatial_qa.py -k 'object_location or support_fallback or current_location'
python -m ruff check src/dsg_spatialqa_lab/qa.py tests/test_spatial_qa.py
python -m mypy src/dsg_spatialqa_lab/qa.py tests/test_spatial_qa.py
python -m pytest -q tests/test_spatial_qa.py tests/test_baselines.py tests/test_dsg_query_diagnostics.py tests/test_dsg_detector_recall_handoff.py
```
