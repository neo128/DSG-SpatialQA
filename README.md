# DSG-SpatialQA Lab

Deterministic minimal verification package for Dynamic Scene Graph spatial QA.

The lab is intentionally local and in-memory. It is meant to make spatial QA,
scene graph retrieval, temporal state audits, and VLA anchor planning
reproducible without calling live AI services, robot stacks, simulators, clocks,
or random sources.

## Documentation

- [Architecture](docs/architecture.md) explains the validation loop, research
  questions, implemented components, and deterministic boundaries.
- [Benchmark And Artifact Formats](docs/benchmark_format.md) summarizes episode
  JSONL, graph JSON, QA JSONL, prediction JSONL, reports, dashboard bundles,
  active task artifacts, and benchmark manifests.
- [Roadmap](docs/roadmap.md) separates the completed deterministic MVP from
  future optional simulator, perception, VLM, and active-task extensions.
- [AI Runbook](docs/AI_RUNBOOK.md) lists handoff commands and operational
  guardrails for agentic development.
- [Verification Record](docs/VERIFICATION.md) records the local verification
  gates, current command results, and known limits.

## Quickstart

Install the package with development tools:

```bash
python -m pip install -e ".[dev]"
```

Run a minimal deterministic spatial QA example:

```python
from dsg_spatialqa_lab import GraphTool, SpatialQAEngine, load_scene_fixture

graph = load_scene_fixture("tabletop")
qa = SpatialQAEngine(GraphTool(graph))

response = qa.answer({"type": "object_location", "object_id": "mug_1"})
print(response.answer)
```

Run the complete local verification gate:

```bash
python scripts/verify.py
```

Run a deterministic evaluation report or reproducibility bundle from the shell:

```bash
python scripts/evaluate.py --tag qa --tag dynamic --report evaluation-report.json
python scripts/evaluate.py --validate-report evaluation-report.json
python scripts/evaluate.py --compare-report evaluation-report.json
python scripts/evaluate.py --list-cases --tag qa --question-type object_room
python scripts/evaluate.py --manifest --tag qa --tag relations --report relations-manifest.json
python scripts/evaluate.py --validate-manifest relations-manifest.json
python scripts/evaluate.py --compare-manifest relations-manifest.json
python scripts/evaluate.py --bundle --tag qa --tag reobserve --report reobserve-bundle.json
python scripts/evaluate.py --validate-bundle reobserve-bundle.json
python scripts/evaluate.py --compare-bundle reobserve-bundle.json
```

Evaluation artifact validation and comparison return non-zero structured JSON
with `valid: false` when an explicit report, case listing, manifest, or bundle file is
unreadable or fails artifact loading.

Validate deterministic episode JSONL artifacts from the shell:

```bash
python scripts/episodes.py --validate mock-episode.jsonl
python scripts/episodes.py --summary mock-episode.jsonl
python scripts/episodes.py --compare mock-episode.jsonl
```

Episode JSONL validation reads only the explicit local file, checks frame schema,
episode/step ordering, duplicate episode steps, and emits stable digest and
summary metadata. Invalid files return non-zero structured JSON with
`valid: false`.

Generate a deterministic mock AI2-THOR episode JSONL artifact without launching
a simulator:

```bash
python scripts/collect_ai2thor.py \
  --mock \
  --scene FloorPlan1 \
  --episode-id ai2thor_mock_001 \
  --step 1 \
  --step 2 \
  --action Initialize \
  --action MoveAhead \
  --output mock-ai2thor.jsonl
```

The adapter CLI requires explicit caller-supplied steps. In the default local
environment, non-mock collection returns non-zero structured JSON with the
missing optional dependency message instead of importing or calling a simulator.

Generate a deterministic mock Habitat episode JSONL artifact without launching
Habitat:

```bash
python scripts/collect_habitat.py \
  --mock \
  --scene apartment_0 \
  --episode-id habitat_mock_001 \
  --step 1 \
  --step 2 \
  --action reset \
  --action turn_left \
  --output mock-habitat.jsonl
```

The Habitat adapter follows the same deterministic boundary: explicit steps,
local mock metadata, oracle-builder-compatible episode JSONL, and structured
missing optional dependency diagnostics for non-mock collection.

Build a deterministic oracle DSG from mock episode metadata:

```bash
python scripts/build_oracle_graph.py \
  --input mock-episode.jsonl \
  --output-graph oracle-graph.json \
  --report oracle-report.json
python scripts/build_oracle_graph.py --validate-report oracle-report.json
python scripts/build_oracle_graph.py --compare-report oracle-report.json
```

Oracle graph building reads only the explicit episode JSONL file, consumes
metadata-provided rooms, regions, objects, and relations, writes graph JSON only
to `--output-graph`, and writes a stable report only to `--report`. Report
validation checks digest, path, and nested graph-report consistency; comparison
rebuilds from the recorded episode path and checks current graph/report drift
plus exported graph-file drift.

Generate a deterministic oracle QA dataset from an explicit graph artifact:

```bash
python scripts/generate_qa.py \
  --graph oracle-graph.json \
  --scene-id mock_scene \
  --episode-id mock_episode \
  --max-cases 100 \
  --output qa.jsonl
python scripts/generate_qa.py --validate qa.jsonl
python scripts/generate_qa.py --compare qa.jsonl --graph oracle-graph.json
```

Generated QA cases are JSONL records with stable IDs, graph digests, questions,
oracle answers, evidence node IDs, evidence edge IDs, choices, tags, and
question types. Validation checks dataset shape and stable digests; comparison
replays each case through the current `SpatialQAEngine` against the supplied
graph and reports answer/evidence drift.

Evaluate deterministic QA prediction JSONL artifacts against a gold QA dataset:

```bash
python scripts/run_qa_eval.py \
  --gold qa.jsonl \
  --pred predictions.jsonl \
  --report qa-eval-report.json
python scripts/run_qa_eval.py --validate-report qa-eval-report.json
python scripts/run_qa_eval.py --compare-report qa-eval-report.json
python scripts/run_qa_eval.py \
  --candidate-report graph-tool-report.json \
  --baseline-report majority-report.json \
  --candidate-name graph_tool \
  --baseline-name majority \
  --delta-report qa-delta-report.json
python scripts/run_qa_eval.py --validate-delta-report qa-delta-report.json
python scripts/run_qa_eval.py --compare-delta-report qa-delta-report.json
```

QA evaluation reads only explicit local gold and prediction files. Reports
include exact match, multiple-choice accuracy, numeric MAE, evidence node/edge
recall, answer-graph consistency, scene/episode/question-type/tag/reference-frame
breakdowns, research-axis breakdowns for spatial QA, dynamic memory, and
GraphTool query coverage, stable report digests, current-file comparison, and
pairwise delta reports for candidate-vs-baseline accuracy/evidence lift. Delta
reports retain scene, episode, question-type, tag, and reference-frame slice
deltas for failure-mode triage.

Run deterministic local baselines over a QA dataset:

```bash
python scripts/run_baselines.py --list-baselines
python scripts/run_baselines.py \
  --baseline graph_tool \
  --graph oracle-graph.json \
  --qa qa.jsonl \
  --pred predictions.jsonl
python scripts/run_qa_eval.py \
  --gold qa.jsonl \
  --pred predictions.jsonl \
  --report graph-tool-qa-eval-report.json
```

The `graph_tool` baseline replays questions through the local
`SpatialQAEngine`; `majority` is a deterministic first-choice baseline for
choice-style cases; `graph_text` and `caption_memory` provide stable offline
interface placeholders. No baseline calls an external model or service.

Import offline external prediction records into the standard QA prediction
JSONL format:

```bash
python scripts/import_predictions.py \
  --qa qa.jsonl \
  --input offline-predictions.jsonl \
  --source-name vlm_fixture \
  --source-kind vlm \
  --metadata prompt_id=spatial-qa-v1 \
  --pred vlm-predictions.jsonl \
  --report vlm-import-report.json
python scripts/import_predictions.py --validate-report vlm-import-report.json
python scripts/import_predictions.py --compare-report vlm-import-report.json
python scripts/run_qa_eval.py \
  --gold qa.jsonl \
  --pred vlm-predictions.jsonl \
  --report vlm-qa-eval-report.json
```

Offline import reads only local JSONL records, skips unknown case IDs with
report diagnostics, records missing gold cases, preserves source metadata,
derives a stable `source_profile` with source key, adapter, model, prompt,
dataset, metadata keys, and capability axes, and writes deterministic
`QAPrediction` JSONL for the existing QA evaluation, attribution, and
dashboard pipeline.

Build a deterministic predicted DSG from mock perception detections in episode
metadata:

```bash
python scripts/build_predicted_graph.py \
  --mock \
  --input mock-episode.jsonl \
  --output-graph predicted-graph.json \
  --report predicted-report.json
python scripts/build_predicted_graph.py --validate-report predicted-report.json
python scripts/build_predicted_graph.py --compare-report predicted-report.json
```

Predicted graph building reads only `EpisodeFrame.metadata["mock_detections"]`
from the explicit episode JSONL file. The mock pipeline defines segmentation,
depth projection, object tracking, object fusion, missing-detection hidden
state updates, and deterministic relation inference without importing real
perception models. Detection `attributes.source`, `source_name`, or
`source_kind` are propagated to object nodes, inferred relations are marked as
`geometry_inference`, and predicted reports summarize detections by source.

Compare two deterministic graph JSON artifacts:

```bash
python scripts/evaluate_graphs.py \
  --oracle oracle-graph.json \
  --predicted predicted-graph.json \
  --report graph-eval-report.json
python scripts/evaluate_graphs.py \
  --oracle oracle-graph.json \
  --predicted predicted-graph.json \
  --matching label_center \
  --center-distance-threshold 0.25 \
  --report graph-eval-label-center-report.json
python scripts/evaluate_graphs.py --validate-report graph-eval-report.json
python scripts/evaluate_graphs.py --compare-report graph-eval-report.json
```

Graph evaluation uses exact object-id and exact relation-edge keys by default.
The optional `label_center` mode matches same-label objects by nearest bbox
center under the supplied threshold and remaps relation edges through matched
object pairs. Use `label_center_room` when same-label, nearby predictions must
also share the oracle object's current room. Reports include object
precision/recall, object label accuracy, relation precision/recall/F1,
confidence-weighted object/relation precision, recall, and F1, matched-object
state accuracy, bbox center error, object-label, relation, and prediction-source
confidence breakdowns,
duplicate-track / ID-fragmentation diagnostics for label+center matching,
stable report digests, saved matching settings, and current-file comparison.

Attribute QA errors across oracle graphs, predicted graphs, and prediction
JSONL artifacts:

```bash
python scripts/attribute_errors.py \
  --gold qa.jsonl \
  --oracle-graph oracle-graph.json \
  --predicted-graph predicted-graph.json \
  --predictions predictions.jsonl \
  --report error-attribution.json
python scripts/attribute_errors.py --validate-report error-attribution.json
python scripts/attribute_errors.py --compare-report error-attribution.json
```

Error attribution replays each QA case through the oracle graph and predicted
graph, compares model/baseline predictions with gold answers, checks required
evidence nodes and edges in the predicted graph, and reports stable categories
such as `benchmark_or_engine_error`, `evidence_missing`, `graph_construction`,
and `reasoning_or_tool_use`. Each case also records predicted evidence sources,
and the summary groups errors by research axis and predicted evidence source so
RQ-level and graph-source quality can be connected to QA failures.

Build a deterministic benchmark manifest from explicit episode JSONL files:

```bash
python scripts/build_benchmark.py \
  --episodes mock-ai2thor.jsonl \
  --episodes mock-habitat.jsonl \
  --dataset-name mock_benchmark \
  --output-dir data/benchmark \
  --max-qa-per-episode 100 \
  --qa-eval-report qa-eval-report.json \
  --qa-eval-delta-report qa-delta-report.json \
  --active-task-report active-report.json \
  --active-task-delta-report active-delta-report.json \
  --dashboard-bundle dashboard/dashboard.json \
  --error-attribution-report error-attribution.json \
  --graph-eval-report graph-eval-report.json \
  --predicted-graph-report predicted-report.json \
  --manifest benchmark-manifest.json
python scripts/build_benchmark.py --validate-manifest benchmark-manifest.json
python scripts/build_benchmark.py --compare-manifest benchmark-manifest.json
```

Benchmark building turns each explicit episode into an oracle graph JSON and QA
JSONL artifact, then writes a manifest with graph digests, QA dataset digests,
summary counts, and coverage by scene, episode, question type, reference frame,
tag, dynamic/static split, and oracle/predicted source. Optional experiment
artifact paths let the same manifest record QA eval reports, QA delta reports,
active task reports, active delta reports, dashboard bundles, oracle-vs-
predicted graph eval reports, error attribution reports, and predicted graph
reports with stable digests for current-file drift checks.

Run the deterministic mock experiment pipeline when you want a local
end-to-end final record in one command. It defaults to one episode and can
aggregate multiple deterministic mock episodes with `--episode-count`. Repeat
`--qa-baseline` to compare `graph_tool` against multiple local QA agents in the
same manifest-linked experiment matrix:

```bash
python scripts/run_mock_experiment.py \
  --output-dir data/mock-experiment \
  --dataset-name mock_experiment \
  --max-qa-per-episode 3 \
  --episode-count 2 \
  --qa-baseline majority \
  --qa-baseline graph_text
```

The pipeline writes mock episode files, oracle graphs, per-episode QA datasets,
a deterministic predicted graph and graph eval report for each episode, a
combined QA dataset, oracle GraphTool, predicted GraphTool, and baseline
prediction/report files, one QA delta report per requested baseline, an
additional oracle-vs-predicted GraphTool QA delta for graph-construction impact,
per-episode error attribution reports for predicted GraphTool failures,
mock active-task delta report, an oracle-vs-predicted active-task delta for
interactive graph-construction impact, benchmark manifest, experiment summary,
static dashboard, and final experiment record. It uses only deterministic local
mocks and explicit output paths.

Summarize the manifest's experiment artifacts into the four project research
questions:

```bash
python scripts/summarize_experiment.py \
  --manifest benchmark-manifest.json \
  --report experiment-summary.json
python scripts/summarize_experiment.py --validate-report experiment-summary.json
python scripts/summarize_experiment.py --compare-report experiment-summary.json
python scripts/record_experiment.py \
  --summary-report experiment-summary.json \
  --record experiment-record.json
python scripts/record_experiment.py --validate-record experiment-record.json
python scripts/record_experiment.py --compare-record experiment-record.json
```

The experiment summary report records stable source artifact digests, QA
candidate-vs-baseline lift for spatial QA, dynamic memory, and GraphTool query
axes, QA diagnostic slices by scene, episode, question type, tag, and reference
frame, graph-construction diagnostics from oracle-vs-predicted graph eval
reports, error attribution diagnostics from QA failure attribution reports,
plus active-task lift for interactive task ability. Each research question also
gets a deterministic `verdict` of `improved`, `unchanged`, `regressed`, or
`inconclusive` from the primary metric delta. The graph diagnostic block
records object recall, relation F1, state accuracy, duplicate-track /
ID-fragmentation counts, and prediction-source precision slices. The
attribution diagnostic block records failure categories such as
`graph_construction`, `evidence_missing`, and `reasoning_or_tool_use`, plus
research-axis and predicted-evidence source summaries, so QA and task lift can
be reviewed beside predicted DSG quality and failure causes. The failure-linkage
block matches
attribution reports to graph eval reports by oracle/predicted graph digest, so
the same summary row can show graph quality metrics beside the failure
categories for that predicted graph. The report records a `readiness` block
that marks the experiment `ready` only when RQ1-RQ4 all have
candidate-vs-baseline evidence, otherwise listing missing research questions
and missing source artifact types. Comparison reloads the manifest path stored
in the summary and detects drift in the referenced local QA, active delta,
graph-eval, and attribution artifacts.

The experiment record projects the saved summary into a compact final handoff:
manifest and summary digests, readiness status, RQ1-RQ4 verdict rows,
`verdict_counts`, a per-measurement `research_question_matrix`,
`diagnostic_ledger` counts/keys for QA slices, graph construction, attribution,
and failure-linkage pairs, source artifact digests, and optional dashboard bundle digest when
`--dashboard-bundle dashboard/dashboard.json` is supplied. Record comparison
reloads only the explicit summary/dashboard paths stored in the record and
reports current-file drift.

Export a deterministic static dashboard for per-sample review:

```bash
python scripts/export_dashboard.py \
  --qa qa.jsonl \
  --pred predictions.jsonl \
  --eval-report qa-eval-report.json \
  --graph oracle-graph.json \
  --error-attribution error-attribution.json \
  --active-task-report active-report.json \
  --active-task-delta-report active-delta-report.json \
  --experiment-summary-report experiment-summary.json \
  --output dashboard/
```

Dashboard export writes `dashboard/dashboard.json` and `dashboard/index.html`
from explicit local artifacts. The bundle contains QA cases, predictions,
per-case QA eval results, optional error attribution rows, evidence subgraphs,
research-axis attribution summaries, predicted evidence source summaries, frame
paths when present, graph summary, optional active-task review panels with task
transcripts, action evidence snapshots, evidence coverage, budget analysis,
optional active-task delta review tables for candidate-vs-baseline RQ4 lift,
optional experiment-summary review rows for RQ1-RQ4 lift, a per-measurement
matrix for multi-baseline QA deltas, failure-linkage rows connecting graph
quality to QA failure causes, verdicts, and experiment readiness, and a stable
bundle digest. The HTML table includes local Research Axis and Evidence Source
filters when attribution data is present.

Run deterministic mock active EQA tasks against an explicit graph artifact:

```bash
python scripts/run_active_tasks.py \
  --tasks active-tasks.jsonl \
  --graph oracle-graph.json \
  --policy direct_answer \
  --report active-report.json
python scripts/run_active_tasks.py --validate-report active-report.json
python scripts/run_active_tasks.py --compare-report active-report.json
python scripts/run_active_tasks.py \
  --candidate-report next-best-view-report.json \
  --baseline-report direct-answer-report.json \
  --candidate-name next_best_view \
  --baseline-name direct_answer \
  --delta-report active-delta-report.json
python scripts/run_active_tasks.py --validate-delta-report active-delta-report.json
python scripts/run_active_tasks.py --compare-delta-report active-delta-report.json
```

Active task reports score task success, answer accuracy, action count, evidence
coverage, answer-graph consistency, per-action evidence snapshots, and
budget-vs-success analysis under explicit max-action budgets. Report comparison
reloads the recorded explicit task and graph paths and reruns the recorded
policy to detect current artifact drift. Active delta reports compare a
candidate policy against a baseline policy and record task-success,
answer-accuracy, evidence-coverage, action-count, and budget-curve lift for
RQ4 review. The `next_best_view` policy
deterministically targets missing required evidence in
the local mock loop. The mock environment only switches between caller-supplied
graph steps; it does not launch a simulator or navigation stack.

Export and validate a deterministic scene fixture graph from the shell:

```bash
python scripts/scene.py --list-fixtures --tag multi_room --output multi-room-fixtures.json
python scripts/scene.py --validate-fixture-manifest multi-room-fixtures.json
python scripts/scene.py --compare-fixture-manifest multi-room-fixtures.json
python scripts/scene.py --fixture tabletop --output tabletop-scene.json --report tabletop-report.json
python scripts/scene.py --validate-report tabletop-report.json
python scripts/scene.py --compare-report tabletop-report.json
python scripts/scene.py --compare-report-graph tabletop-report.json --input tabletop-scene.json
python scripts/scene.py --validate tabletop-scene.json
python scripts/scene.py --compare-fixture tabletop --input tabletop-scene.json
```

Fixture listing emits a filtered scene fixture manifest with schema version,
metadata digest, fixture count, and scene names/descriptions/tags; it does not
load graph objects or compute graph JSON digests. When `--output` is supplied,
the same stable JSON is written to that explicit local path and printed to stdout.
Fixture manifest validation reads only the explicit local file and checks schema
version, digest, and fixture count consistency. Fixture manifest comparison
also checks the saved metadata against the current built-in fixture registry and
reports stable metadata `differences` paths such as
`multi_room_rearrangement.tags`.

Scene validation, fixture manifest validation, fixture manifest comparison, and
fixture comparison return non-zero structured JSON when an explicit file is
unreadable, fails schema/digest validation, or drifts from current metadata.
For graph export and validation workflows, `--report` writes the same structured
stdout JSON to an explicit local path; graph reports include a report schema
version, graph digest, report digest, and deterministic graph summary. Graph
report validation checks the explicit report artifact and report digest, and
graph report comparison checks that its saved graph digest and summary still
match the current built-in fixture named in the report. Graph report-to-file
comparison checks the saved report against an explicit caller-supplied graph JSON
artifact.

Ingest a deterministic mock perception observation sequence from an explicit
local file and export the resulting graph/report artifacts:

```bash
python scripts/observations.py --validate-sequence mock-observation-sequence.json
python scripts/observations.py \
  --summarize-sequence mock-observation-sequence.json \
  --report mock-observation-sequence-summary.json
python scripts/observations.py --validate-sequence-summary mock-observation-sequence-summary.json
python scripts/observations.py --compare-sequence-summary mock-observation-sequence-summary.json
python scripts/observations.py \
  --input mock-observation-sequence.json \
  --output-graph mock-observation-graph.json \
  --report mock-observation-ingest-report.json
python scripts/observations.py --validate-report mock-observation-ingest-report.json
python scripts/observations.py --compare-report mock-observation-ingest-report.json
```

Observation sequence validation reads only the explicit sequence JSON file,
checks the sequence schema/count/step shape, and reports the stable sequence
digest plus graph-free summary without building a graph. Observation sequence
summarization reads only the explicit sequence JSON file and reports stable
sequence digest, step, object, visibility, confidence, and label-count metadata.
Summary validation checks the explicit summary artifact fingerprint and internal
count consistency; summary comparison reads only the sequence path recorded in
that artifact and detects current sequence drift. Observation ingestion reads
only the explicit sequence JSON file, writes graph JSON only to `--output-graph`,
and writes the structured ingest report only when `--report` is supplied.
Observation ingest report validation and comparison read only explicit local
report, sequence, and graph artifacts, then compare saved sequence digest, graph
digest, graph-file digest, summaries, and per-step ingest results against a
current deterministic re-ingest. Invalid or drifted sequence, graph, summary, or
report files return non-zero structured JSON with `valid: false` or
`matches: false`.

If the editable development install is already current, skip that first gate:

```bash
python scripts/verify.py --skip-install
```

## MVP Capabilities

- In-memory Dynamic Scene Graph state for agents, objects, rooms, regions,
  actions, events, current state, and explicit-step history.
- Deterministic spatial relations and graph retrieval through `GraphTool`,
  including bbox surface-distance `NEAR`, centroid containment `INSIDE`,
  support-overlap `ON` / `SUPPORTS`, stable metric distance reports, and
  explicit placeholder edges for visibility, reachability, and occlusion.
- Structured QA intents for object state, agent state, label-candidate
  ambiguity, room-level containment, timelines, scene snapshots, scene deltas,
  world state, recent events, graph queries, and re-observation targets.
- Deterministic VLA anchor planning for pick and place-relative commands,
  including ambiguity candidate diagnostics, stale-action, re-observation, and
  target/reference visibility-confidence precondition handling.
- Built-in scene fixtures and evaluation cases with stable suite summaries,
  failure diagnostics, and SHA-256 digests for experiment records.
- Deterministic oracle QA JSONL generation from explicit graph artifacts,
  including answer/evidence replay validation and current-graph comparison.
- Deterministic QA prediction JSONL evaluation with stable accuracy, evidence,
  research-axis breakdown, validation, current-file comparison, and
  candidate-vs-baseline delta reports.
- Deterministic local baseline runner with `graph_tool`, `majority`,
  `graph_text`, and disabled `caption_memory` interfaces that emit prediction
  JSONL without external calls.
- Deterministic offline prediction import tooling for local VLM/caption-memory
  style outputs, with source metadata, unknown/missing case diagnostics, stable
  prediction JSONL, and import report validation/comparison.
- Deterministic oracle-vs-predicted graph metrics with explicit graph JSON
  inputs, stable digest reports, validation, and current-file comparison.
- Deterministic predicted DSG builder skeleton with mock perception detections,
  depth projection, stable object IDs, hidden low-confidence updates, relation
  inference, graph/report digests, and explicit-path validation/comparison.
- Deterministic QA error attribution across oracle graph answers, predicted
  graph answers, model/baseline predictions, required evidence presence,
  research-axis failure summaries, and predicted-evidence source summaries.
- Static dashboard export for per-sample QA, prediction, evaluation,
  attribution, research-axis and predicted-evidence source filtering,
  graph-summary, and evidence-subgraph review without default dashboard
  dependencies.
- Deterministic active EQA task schema, mock environment, local active
  `GraphTool` policies, task metrics, action evidence snapshots, and
  budget-vs-success analysis for explicit graph-step handoffs, including
  active report comparison and a deterministic `next_best_view`
  missing-evidence policy placeholder.
- Deterministic benchmark manifest tooling that builds oracle graph and QA
  artifacts from explicit episodes and records stable coverage/digest metadata.
- Offline evaluation report, manifest, and bundle CLI entrypoints with
  structured invalid-file diagnostics for reproducible handoffs.
- Deterministic episode JSONL schema and CLI validation for simulator/mock
  collection handoffs without connecting to a simulator.
- Optional AI2-THOR adapter boundary with deterministic mock episode generation,
  explicit-step collection config, and structured missing-dependency diagnostics
  for non-mock collection.
- Optional Habitat adapter boundary with deterministic mock episode generation,
  explicit-step collection config, and structured missing-dependency diagnostics
  for non-mock collection.
- Deterministic oracle DSG builder for episode metadata, including room/region
  containment, explicit relations, moved-object evidence, stable graph/report
  digests, and explicit-path validation/comparison.
- Offline observation sequence summary and ingest CLI that turns explicit mock
  perception JSON artifacts into stable dataset summaries, graph JSON, and
  digest/summary reports.
- Dynamic fixture coverage for multi-room relocation, stable room containment,
  occlusion, re-observation, relation shifts, temporal deltas, and step-window
  event audits.
- JSON graph import/export helpers, graph digests, graph summaries with
  object-state, current-location, current-room, node-type, edge-relation, and
  object-label counts, and a scene fixture metadata/export/validate/compare CLI
  with structured invalid-file diagnostics for reproducible local experiments.

## Project Boundaries

- Runtime code must stay deterministic: callers supply explicit steps and the
  package does not read wall-clock time or produce random output.
- No runtime network calls are allowed.
- Real LLM/VLM, robot, simulator, database, and service integrations are out of
  scope for the MVP and should be mocked or omitted.
- Data structures remain in-memory until a real integration justifies
  persistence.
- Development tooling may install declared dev dependencies, but package runtime
  dependencies remain standard-library only.

## Development Baseline

Use the one-command verifier before handing off changes:

```bash
python scripts/verify.py
```

It runs the project gates in order: editable dev install, lint, typecheck,
determinism scan, unit tests, package build, and the built-in evaluation suite.
The individual commands remain available when focused feedback is useful:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m mypy src tests scripts
python scripts/check_determinism.py
python -m pytest -q
python -m build
```

GitHub Actions uses the same local verifier through
`.github/workflows/verify.yml`, so CI and local handoff checks stay aligned.
The package declares a `py.typed` marker so CLI scripts can be typechecked
against the installed package during the verifier run.
The determinism scan is a local source check over `.github`, `scripts`, `src`,
and `tests` for current-time, random, network, or external model client
boundaries.

The built-in evaluation suite can also be run directly:

```bash
python scripts/evaluate.py --name tabletop_object_location
python scripts/evaluate.py --kind vla_pick --report evaluation-report.json
python scripts/evaluate.py --tag qa --tag dynamic --question-type scene_delta
python scripts/evaluate.py --list-cases --tag vla --tag error --report vla-error-listing.json
python scripts/evaluate.py --validate-listing vla-error-listing.json
python scripts/evaluate.py --compare-listing vla-error-listing.json
python scripts/evaluate.py --compare-report evaluation-report.json
python scripts/evaluate.py --bundle --tag qa --tag reobserve
python scripts/evaluate.py --validate-bundle reobserve-bundle.json
python scripts/evaluate.py --compare-bundle reobserve-bundle.json
python scripts/evaluate.py --compare-manifest relations-manifest.json
```

The CLI prints stable JSON to stdout and, when `--report` is provided, writes
the selected report, case listing, manifest, or bundle to an explicit local
path. `--list-cases` emits only the listing schema version, filtered case
metadata, case count, and stable digest without running cases, which is useful
for discovering focused benchmark slices.
Case listing validation checks the explicit schema version, listing digest, and
case count, plus required case metadata shape and unique case names. Case
listing comparison regenerates current filtered metadata without running
evaluation cases and reports stable case metadata drift by case name.
Manifest output includes the filter manifest, selected scene fixtures, selected
evaluation cases, deterministic coverage counts, and a digest without running
cases. Manifest validation reads only the explicit local JSON file and checks
schema version, digest, required case metadata shape, unique case names, scene
fixture coverage, case-backed scene fixture metadata consistency, and coverage
summary consistency.
`evaluation_manifest_digest()` recomputes the saved metadata fingerprint for
Python handoffs. Fixture metadata and coverage summary validation/comparison
include stable nested `differences` entries with paths such as
`tabletop.tags` and `by_scene_fixture.tabletop`.
Manifest comparison reads the saved filters from an explicit local manifest,
regenerates the current deterministic metadata without running cases, and
returns a non-zero status when digest, coverage, case manifest, or fixture
manifest drift is detected. Coverage, case manifest, and fixture manifest drift
include stable nested `differences` entries with paths such as `by_tag.qa` and
`tabletop_relation_timeline.tags`.
Report comparison reads the selected case names from an explicit local compact
report, reruns that deterministic case slice, and returns a non-zero status when
digest, case selection metadata, summary, metrics, runtime error metrics,
failure diagnostics, or breakdown drift is detected.
Report validation checks the compact report schema version, suite digest format,
case selection digest, case selection entry metadata shape, case selection
consistency with `summary.selected_cases`, failed-case detail consistency with
`summary.failed_cases`, failed-case entry metadata shape, case digest
consistency with `summary.selected_cases`, summary case-list shape and
failed-case membership in selected cases, summary count consistency with the
selected/failed case lists, breakdown count consistency with each grouped
entry's selected/failed case lists, breakdown case-list consistency with
`case_selection` metadata, metric consistency with summary/breakdown, top-level
and grouped evidence metric internal consistency with summary/breakdown counts,
evidence metric value ranges, runtime error category entry shape, runtime error
category count/case consistency with selected cases, runtime error metric
consistency with category aggregates, per-case digest format, per-case digest
metadata consistency with `case_selection`, per-case digest pass/fail status
consistency with `summary.failed_cases`, failure diagnostic aggregate
consistency with `failed_cases`, and saved report digest before handoff.
Summary, failed-case, metric, and breakdown drift checks include stable nested
`differences` entries with paths such as `failed`,
`tabletop_object_location`, `by_tag.qa.pass_rate`, and `by_tag.qa.failed` for
quick report triage.
Runtime error category drift checks include stable nested `differences` entries
with category paths such as `missing_object.count`.
Failure diagnostic drift checks also include stable nested `differences` entries
with category or reason paths such as `value_mismatch`, and mismatch paths such
as `answer.visible`.
Bundle output includes
the filter manifest, selected scene fixtures, selected evaluation cases, full
suite results, compact report, coverage counts, and digest. Compact report
metrics include selected, passed, and failed case counts, pass rate, and failure
rate. Coverage counts are grouped by case kind, QA question type, case tag, scene
fixture, and scene tag.
Each bundle also carries `bundle_digest`, and `evaluation_bundle_digest()` plus
`validate_evaluation_bundle()` recompute and validate the full bundle artifact
fingerprint, including outer metadata such as filters.
Suite and report breakdowns also include QA question-type summaries for direct
intent-level triage.
Bundle validation reads only the explicit local JSON file, recomputes
deterministic consistency checks, and returns a non-zero status when the bundle
does not validate. Validation checks schema version, suite digest, bundle digest,
report consistency, case manifest names, required case metadata shape, unique
case names, suite-backed metadata, scene fixture coverage and case-backed
metadata, and coverage summary consistency. Report consistency, case manifest
metadata, case metadata shape, scene fixture metadata, and coverage summary
validation include stable nested `differences` entries for quick handoff triage,
including compact-report paths such as `failed_cases.tabletop_object_location`,
case metadata paths such as `evaluation_cases[0].question`, case manifest paths
such as `multi_room_rearrangement_reobserve_targets.tags`, and fixture manifest paths
such as `needs_reobserve.tags` when summaries or metadata drift.
Bundle comparison reads the saved filters from an explicit local bundle, reruns
the current deterministic suite, and returns a non-zero status when digest,
compact report, coverage, case manifest, or fixture manifest drift is detected.
Comparison reports both saved/current suite digests and saved/current bundle
artifact digests, with a dedicated `bundle_digest_matches_current` check for
handoff artifact triage.
Compact report drift includes stable nested `differences` paths such as
`metrics.by_tag.qa.pass_rate`. Case and fixture manifest metadata drift is keyed
by manifest entry name for handoff triage.
Evaluation artifact validation and comparison commands mark invalid explicit
report, manifest, or bundle files with `valid: false`, a stable `error` string,
and a non-zero exit status; comparison error payloads also include
`matches: false`.
Scene graph comparison reads an explicit local graph JSON file and compares its
digest and summary counts with a freshly generated built-in fixture graph.
Graph summaries include total counts, visible/hidden object counts,
low-confidence and re-observation candidate counts, current containment counts
such as `by_current_location`, resolved room counts such as `by_current_room`,
plus `by_node_type`, `by_edge_relation`, and `by_object_label` counts. Summary
drift checks include stable nested `differences` entries such as
`by_current_room.pantry`, `by_current_location.pantry_shelf`,
`by_node_type.region`, and `node_count`. Scene validation and comparison mark valid explicit graph inputs
with `valid: true`; invalid explicit graph files return `valid: false`, a stable
`error` string, and a non-zero exit status.

```python
from dsg_spatialqa_lab import (
    compare_evaluation_bundle,
    compare_evaluation_case_listing,
    compare_evaluation_manifest,
    compare_evaluation_report,
    compare_graph_file_to_fixture,
    compare_graph_report,
    compare_graph_report_to_file,
    compare_graph_to_fixture,
    evaluation_bundle,
    evaluation_bundle_digest,
    evaluation_bundle_json,
    evaluation_case_listing,
    evaluation_case_listing_digest,
    evaluation_case_listing_json,
    evaluation_manifest,
    evaluation_manifest_digest,
    evaluation_manifest_json,
    evaluation_report,
    evaluation_report_digest,
    evaluation_report_json,
    load_evaluation_bundle,
    load_evaluation_case_listing,
    load_evaluation_manifest,
    load_evaluation_report,
    run_evaluation_suite,
    save_evaluation_bundle,
    save_evaluation_case_listing,
    save_evaluation_manifest,
    save_evaluation_report,
    validate_evaluation_bundle,
    validate_evaluation_case_listing,
    validate_evaluation_manifest,
    validate_evaluation_report,
)

suite = run_evaluation_suite()
print(suite["summary"])
print(suite["digest"])

case_listing = evaluation_case_listing(tags=("qa",), question_types=("object_room",))
print(evaluation_case_listing_digest(case_listing))
save_evaluation_case_listing(
    "object-room-listing.json",
    tags=("qa",),
    question_types=("object_room",),
)
loaded_case_listing = load_evaluation_case_listing("object-room-listing.json")
case_listing_validation = validate_evaluation_case_listing(loaded_case_listing)
case_listing_comparison = compare_evaluation_case_listing(loaded_case_listing)

report = evaluation_report(suite)
print(evaluation_report_json(report))
print(evaluation_report_digest(report))
save_evaluation_report("evaluation-report.json", suite)
loaded_report = load_evaluation_report("evaluation-report.json")
report_validation = validate_evaluation_report(loaded_report)
report_comparison = compare_evaluation_report(loaded_report)

manifest = evaluation_manifest(tags=("qa", "relations"))
print(evaluation_manifest_json(manifest))
print(evaluation_manifest_digest(manifest))
save_evaluation_manifest("relations-manifest.json", tags=("qa", "relations"))
loaded_manifest = load_evaluation_manifest("relations-manifest.json")
manifest_validation = validate_evaluation_manifest(loaded_manifest)
manifest_comparison = compare_evaluation_manifest(loaded_manifest)

bundle = evaluation_bundle(tags=("qa", "reobserve"))
print(evaluation_bundle_json(bundle))
print(bundle["coverage"])
print(evaluation_bundle_digest(bundle))
save_evaluation_bundle("reobserve-bundle.json", tags=("qa", "reobserve"))
loaded_bundle = load_evaluation_bundle("reobserve-bundle.json")
bundle_validation = validate_evaluation_bundle(loaded_bundle)
bundle_comparison = compare_evaluation_bundle(loaded_bundle)
fixture_comparison = compare_graph_file_to_fixture("tabletop-scene.json", "tabletop")
```

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the maintained roadmap and milestone
status.

- Keep CI and local verification aligned as new benchmark gates are added.
- Add optional deterministic dataset import adapters once real experiment
  formats are chosen.
- Expand relation geometry and sensor evidence models while preserving explicit
  caller-supplied steps.
- Extend offline report outputs with any new benchmark metrics before adding
  external integrations.
- Add persistence only after there is a concrete integration requirement.
- Keep external AI, robot, and simulator adapters outside the deterministic core
  behind mocked or offline boundaries.

## Full API Example

```python
from dsg_spatialqa_lab import (
    AI2THOR_MISSING_DEPENDENCY_MESSAGE,
    AI2ThorAdapterConfig,
    AI2ThorEpisodeCollector,
    BBox3D,
    compare_evaluation_bundle,
    compare_evaluation_case_listing,
    compare_evaluation_manifest,
    compare_evaluation_report,
    compare_graph_file_to_fixture,
    compare_graph_report,
    compare_graph_report_to_file,
    compare_graph_to_fixture,
    compare_episode_sequence,
    compare_observation_ingest_report,
    compare_oracle_graph_report,
    compare_scene_observation_sequence_summary,
    build_oracle_graph_from_episode,
    build_mock_ai2thor_episode,
    build_relation_shift_scene,
    DynamicSceneGraph,
    EpisodeFrame,
    EvaluationCase,
    compare_scene_fixture_manifest,
    evaluation_bundle,
    evaluation_bundle_digest,
    evaluation_bundle_json,
    evaluation_case_listing,
    evaluation_case_listing_digest,
    evaluation_case_listing_json,
    evaluation_cases_metadata,
    evaluation_manifest,
    evaluation_manifest_digest,
    evaluation_manifest_json,
    evaluation_report,
    evaluation_report_digest,
    evaluation_report_json,
    GraphQuery,
    GraphTool,
    ObjectObservation,
    ObservationIngestor,
    Pose3D,
    SceneObservation,
    SpatialQAEngine,
    VLAAnchorPlanner,
    graph_from_json,
    graph_json_digest,
    graph_report,
    graph_report_digest,
    graph_report_json,
    graph_summary,
    graph_to_json,
    episode_frame_from_dict,
    episode_frame_to_dict,
    episode_sequence_digest,
    episode_sequence_from_jsonl,
    episode_sequence_summary,
    episode_sequence_to_jsonl,
    ingest_scene_observation_sequence,
    load_evaluation_bundle,
    load_evaluation_case_listing,
    load_evaluation_manifest,
    load_evaluation_report,
    load_graph_report,
    load_graph_json,
    load_observation_ingest_report,
    load_oracle_graph_report,
    load_episode_sequence,
    load_scene_observation,
    load_scene_observation_sequence,
    load_scene_observation_sequence_summary,
    load_scene_fixture_manifest,
    list_scene_fixture_metadata,
    list_scene_fixtures,
    load_scene_fixture,
    list_evaluation_case_metadata,
    run_evaluation_cases,
    run_evaluation_suite,
    observation_ingest_report,
    observation_ingest_report_digest,
    observation_ingest_report_json,
    oracle_graph_report,
    oracle_graph_report_digest,
    oracle_graph_report_json,
    oracle_graph_summary,
    save_observation_ingest_report,
    save_oracle_graph_report,
    save_episode_sequence,
    save_evaluation_bundle,
    save_evaluation_case_listing,
    save_evaluation_manifest,
    save_evaluation_report,
    save_graph_report,
    save_graph_json,
    save_scene_observation,
    save_scene_observation_sequence,
    save_scene_observation_sequence_summary,
    save_scene_fixture_manifest,
    scene_observation_from_dict,
    scene_observation_from_json,
    scene_observation_sequence_from_dict,
    scene_observation_sequence_from_json,
    scene_observation_sequence_digest,
    scene_observation_sequence_summary,
    scene_observation_sequence_summary_digest,
    scene_observation_sequence_summary_json,
    scene_observation_sequence_to_dict,
    scene_observation_sequence_to_json,
    scene_observation_to_dict,
    scene_observation_to_json,
    scene_fixture_manifest,
    scene_fixture_manifest_json,
    validate_evaluation_bundle,
    validate_evaluation_case_listing,
    validate_evaluation_manifest,
    validate_evaluation_report,
    validate_episode_sequence,
    validate_graph_report,
    validate_observation_ingest_report,
    validate_oracle_graph_report,
    validate_scene_observation_sequence_payload,
    validate_scene_observation_sequence_summary,
    validate_scene_fixture_manifest,
    convert_ai2thor_event_to_episode_frame,
)

available_scenes = list_scene_fixtures()
scene_manifest = list_scene_fixture_metadata()
scene_fixture_handoff = scene_fixture_manifest(tags=("reobserve",))
scene_fixture_payload = scene_fixture_manifest_json(scene_fixture_handoff)
save_scene_fixture_manifest("reobserve-fixtures.json", tags=("reobserve",))
scene_fixture_validation = validate_scene_fixture_manifest(scene_fixture_handoff)
scene_fixture_comparison = compare_scene_fixture_manifest(scene_fixture_handoff)
ambiguity_scene_manifest = list_scene_fixture_metadata(tags=("ambiguity",))
reobserve_scene_manifest = list_scene_fixture_metadata(tags=("reobserve",))
graph = load_scene_fixture("tabletop")
tool = GraphTool(graph)
qa = SpatialQAEngine(tool)
agent_location = qa.answer({"type": "agent_location"})
agent_history = qa.answer({"type": "agent_history"})
agent_timeline = qa.answer({"type": "agent_timeline"})
response = qa.answer({"type": "object_location", "object_id": "mug_1"})
object_status = qa.answer({"type": "object_status", "object_id": "mug_1"})
object_timeline = qa.answer({"type": "object_timeline", "object_id": "mug_1"})
relation_timeline = qa.answer(
    {
        "type": "relation_timeline",
        "src": "mug_1",
        "dst": "plate_1",
        "reference_frame": "agent",
    }
)
reobserve_targets = qa.answer({"type": "reobserve_targets"})
nearest_plate = qa.answer(
    {"type": "nearest_object", "src": "mug_1", "candidates": ["plate_1"]}
)
label_candidates = qa.answer(
    {"type": "label_candidates", "label": "mug", "visible": True}
)
snapshot = qa.answer({"type": "scene_snapshot", "step": 1, "visible": True})
delta = qa.answer({"type": "scene_delta", "from_step": 1, "to_step": 2})
world_state = qa.answer({"type": "world_state", "visible": True})
recent_events = qa.answer({"type": "recent_events", "since_step": 1})
multi_room_tool = GraphTool(load_scene_fixture("multi_room_rearrangement"))
room_response = SpatialQAEngine(multi_room_tool).answer(
    {"type": "object_room", "object_id": "cereal_box_1"}
)
graph_query_response = qa.answer(
    {
        "type": "graph_query",
        "query": {
            "node_types": ["object"],
            "labels": ["mug", "plate"],
            "relations": ["LEFT_OF", "NEAR"],
            "reference_frame": "agent",
        },
    }
)
retrieved_subgraph = qa.answer(
    {"type": "retrieve_subgraph", "query": "mug", "max_nodes": 3, "hops": 1}
)
planner = VLAAnchorPlanner(tool)
pick = planner.plan_pick(target_object="mug_1")
place = planner.plan_place_relative("mug_1", "plate_1", "RIGHT_OF")
place_validity = planner.validate(place.command) if place.command is not None else place

subgraph = tool.query_graph(
    GraphQuery(
        node_types=("object",),
        labels=("mug", "plate"),
        visible=True,
        relations=("LEFT_OF", "NEAR"),
        reference_frame="agent",
    )
)
agent_timeline = tool.agent_timeline("agent")
timeline = tool.object_timeline("mug_1")
relation_timeline = tool.relation_timeline(src="mug_1", reference_frame="agent")
current_room = tool.current_room("mug_1")
current_world = tool.world_state(visible=True)
recent_event_trace = tool.recent_events(since_step=1)
targets_to_reobserve = tool.reobserve_targets()
metric_distance = tool.compute_distance("mug_1", "plate_1")

inferred_edges = tool.update_spatial_relations(
    step=2,
    object_ids=("mug_1", "plate_1", "table_1"),
    relations=("LEFT_OF", "RIGHT_OF", "NEAR", "INSIDE", "SUPPORTS"),
    reference_frames=("world", "agent"),
)

episode_frames = (
    EpisodeFrame(
        episode_id="episode_001",
        scene_id="FloorPlan1",
        step=1,
        rgb_path="rgb/0001.png",
        depth_path="depth/0001.npy",
        segmentation_path=None,
        agent_id="agent",
        agent_pose=Pose3D(0.0, 0.0, 0.0),
        action="MoveAhead",
        visible_object_ids=("mug_1",),
        metadata={
            "rooms": [{"room_id": "kitchen", "label": "Kitchen"}],
            "objects": [
                {
                    "object_id": "mug_1",
                    "label": "mug",
                    "pose": {"x": 0.0, "y": 1.0, "z": 0.75, "yaw": 0.0},
                    "bbox": {
                        "center": {"x": 0.0, "y": 1.0, "z": 0.75, "yaw": 0.0},
                        "size": [0.1, 0.1, 0.1],
                    },
                    "confidence": 0.9,
                    "visible": True,
                    "room_id": "kitchen",
                    "states": {"clean": True},
                }
            ],
        },
    ),
)
episode_jsonl = episode_sequence_to_jsonl(episode_frames)
episode_round_trip = episode_sequence_from_jsonl(episode_jsonl)
episode_frame_payload = episode_frame_to_dict(episode_round_trip[0])
episode_frame = episode_frame_from_dict(episode_frame_payload)
episode_digest = episode_sequence_digest(episode_round_trip)
episode_summary = episode_sequence_summary(episode_round_trip)
episode_validation = validate_episode_sequence(episode_round_trip)
episode_comparison = compare_episode_sequence(episode_round_trip)
save_episode_sequence(episode_round_trip, "mock-episode.jsonl")
loaded_episode = load_episode_sequence("mock-episode.jsonl")
ai2thor_config = AI2ThorAdapterConfig(
    scene_id="FloorPlan1",
    episode_id="ai2thor_mock_001",
    steps=(1, 2),
    actions=("Initialize", "MoveAhead"),
    artifact_root="artifacts/ai2thor",
)
ai2thor_mock_episode = build_mock_ai2thor_episode(ai2thor_config)
ai2thor_frame = convert_ai2thor_event_to_episode_frame(
    {
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        "visible_object_ids": ["mug_1"],
        "metadata": {"objects": []},
    },
    config=ai2thor_config,
    step=3,
    action="LookDown",
)
ai2thor_missing_message = AI2THOR_MISSING_DEPENDENCY_MESSAGE
ai2thor_real_collector = AI2ThorEpisodeCollector(ai2thor_config)
oracle_graph = build_oracle_graph_from_episode(loaded_episode)
oracle_summary_payload = oracle_graph_summary(oracle_graph, loaded_episode)
save_graph_json(oracle_graph, "oracle-graph.json")
oracle_graph_report_payload = oracle_graph_report(
    input_path="mock-episode.jsonl",
    graph_path="oracle-graph.json",
    graph=oracle_graph,
    frames=loaded_episode,
)
oracle_graph_report_payload_json = oracle_graph_report_json(
    oracle_graph_report_payload
)
oracle_graph_report_payload_digest = oracle_graph_report_digest(
    oracle_graph_report_payload
)
save_oracle_graph_report(oracle_graph_report_payload, "oracle-report.json")
loaded_oracle_graph_report = load_oracle_graph_report("oracle-report.json")
oracle_graph_report_validation = validate_oracle_graph_report(
    loaded_oracle_graph_report
)
oracle_graph_report_comparison = compare_oracle_graph_report(
    loaded_oracle_graph_report
)

ingest_result = ObservationIngestor(graph).ingest(
    SceneObservation(
        step=3,
        agent_pose=Pose3D(0.0, 0.0, 0.0, yaw=0.0),
        objects=(
            ObjectObservation(
                "mug_1",
                "mug",
                Pose3D(-0.4, 1.0, 0.78),
                BBox3D(center=Pose3D(-0.4, 1.0, 0.78), size=(0.12, 0.12, 0.16)),
                confidence=0.95,
                visible=True,
            ),
        ),
    ),
    infer_relations=("NEAR",),
)
scene_observation_payload = scene_observation_to_json(
    SceneObservation(
        step=4,
        agent_pose=Pose3D(0.0, 0.0, 0.0, yaw=0.0),
        objects=(
            ObjectObservation(
                "mug_1",
                "mug",
                Pose3D(-0.4, 1.0, 0.78),
                BBox3D(center=Pose3D(-0.4, 1.0, 0.78), size=(0.12, 0.12, 0.16)),
                confidence=0.95,
                visible=True,
            ),
        ),
    )
)
restored_scene_observation = scene_observation_from_json(scene_observation_payload)
save_scene_observation(restored_scene_observation, "mock-observation-step4.json")
loaded_scene_observation = load_scene_observation("mock-observation-step4.json")
scene_observation_sequence_payload = scene_observation_sequence_to_json(
    (loaded_scene_observation, restored_scene_observation)
)
restored_scene_observation_sequence = scene_observation_sequence_from_json(
    scene_observation_sequence_payload
)
save_scene_observation_sequence(
    restored_scene_observation_sequence,
    "mock-observation-sequence.json",
)
loaded_scene_observation_sequence = load_scene_observation_sequence(
    "mock-observation-sequence.json"
)
observation_sequence_digest = scene_observation_sequence_digest(
    loaded_scene_observation_sequence
)
observation_sequence_validation = validate_scene_observation_sequence_payload(
    scene_observation_sequence_to_dict(loaded_scene_observation_sequence)
)
observation_sequence_summary = scene_observation_sequence_summary(
    loaded_scene_observation_sequence
)
observation_sequence_summary_digest = scene_observation_sequence_summary_digest(
    observation_sequence_summary
)
observation_sequence_summary_payload = {
    "action": "summarize_observation_sequence",
    "path": "mock-observation-sequence.json",
    "valid": True,
    **observation_sequence_summary,
}
observation_sequence_summary_json = scene_observation_sequence_summary_json(
    observation_sequence_summary_payload
)
save_scene_observation_sequence_summary(
    observation_sequence_summary_payload,
    "mock-observation-sequence-summary.json",
)
loaded_observation_sequence_summary = load_scene_observation_sequence_summary(
    "mock-observation-sequence-summary.json"
)
observation_sequence_summary_validation = validate_scene_observation_sequence_summary(
    loaded_observation_sequence_summary
)
observation_sequence_summary_comparison = compare_scene_observation_sequence_summary(
    loaded_observation_sequence_summary
)
observation_graph, observation_ingest_results = ingest_scene_observation_sequence(
    loaded_scene_observation_sequence,
    source_path="mock-observation-sequence.json",
)
observation_ingest_report_payload = observation_ingest_report(
    input_path="mock-observation-sequence.json",
    graph_path="mock-observation-graph.json",
    graph=observation_graph,
    ingest_results=observation_ingest_results,
    sequence_digest=observation_sequence_digest,
)
observation_ingest_report_payload_json = observation_ingest_report_json(
    observation_ingest_report_payload
)
observation_ingest_report_payload_digest = observation_ingest_report_digest(
    observation_ingest_report_payload
)
save_observation_ingest_report(
    observation_ingest_report_payload,
    "mock-observation-ingest-report.json",
)
loaded_observation_ingest_report = load_observation_ingest_report(
    "mock-observation-ingest-report.json"
)
observation_ingest_report_validation = validate_observation_ingest_report(
    loaded_observation_ingest_report
)
observation_ingest_report_comparison = compare_observation_ingest_report(
    loaded_observation_ingest_report
)

payload = graph_to_json(graph)
graph_digest = graph_json_digest(graph)
graph_counts = graph_summary(graph)
graph_report_payload = graph_report(
    graph,
    action="export_fixture",
    graph_path="tabletop-scene.json",
    fixture="tabletop",
)
graph_report_json_payload = graph_report_json(graph_report_payload)
graph_report_payload_digest = graph_report_digest(graph_report_payload)
restored = graph_from_json(payload)
save_graph_json(graph, "tabletop-scene.json")
save_graph_report(
    graph,
    "tabletop-report.json",
    action="export_fixture",
    graph_path="tabletop-scene.json",
    fixture="tabletop",
)
loaded_graph_report = load_graph_report("tabletop-report.json")
graph_report_validation = validate_graph_report(loaded_graph_report)
graph_report_comparison = compare_graph_report(loaded_graph_report)
restored_from_path = load_graph_json("tabletop-scene.json")
graph_report_file_comparison = compare_graph_report_to_file(
    loaded_graph_report,
    "tabletop-scene.json",
)
graph_fixture_comparison = compare_graph_to_fixture(restored_from_path, "tabletop")
graph_file_comparison = compare_graph_file_to_fixture("tabletop-scene.json", "tabletop")
case_listing = evaluation_case_listing(tags=("qa",), question_types=("object_room",))
case_listing_digest = evaluation_case_listing_digest(case_listing)
case_listing_json = evaluation_case_listing_json(case_listing)
case_listing_path = save_evaluation_case_listing(
    "object-room-listing.json",
    tags=("qa",),
    question_types=("object_room",),
)
loaded_case_listing = load_evaluation_case_listing(case_listing_path)
case_listing_validation = validate_evaluation_case_listing(loaded_case_listing)
case_listing_comparison = compare_evaluation_case_listing(loaded_case_listing)
suite = run_evaluation_suite()
suite_digest = suite["digest"]
suite_report = evaluation_report(suite)
suite_report_digest = evaluation_report_digest(suite_report)
suite_report_json = evaluation_report_json(suite_report)
suite_manifest = evaluation_manifest(tags=("qa", "relations"))
suite_manifest_digest = evaluation_manifest_digest(suite_manifest)
suite_manifest_json = evaluation_manifest_json(suite_manifest)
manifest_path = save_evaluation_manifest(
    "relations-manifest.json",
    tags=("qa", "relations"),
)
loaded_manifest = load_evaluation_manifest(manifest_path)
manifest_validation = validate_evaluation_manifest(loaded_manifest)
manifest_comparison = compare_evaluation_manifest(loaded_manifest)
suite_bundle = evaluation_bundle(tags=("qa", "reobserve"))
suite_bundle_digest = evaluation_bundle_digest(suite_bundle)
suite_bundle_json = evaluation_bundle_json(suite_bundle)
bundle_validation = validate_evaluation_bundle(suite_bundle)
named_suite = run_evaluation_suite(
    names=("tabletop_object_location", "moved_mug_recent_events")
)
report_path = save_evaluation_report("evaluation-report.json", named_suite)
loaded_report = load_evaluation_report(report_path)
report_validation = validate_evaluation_report(loaded_report)
report_comparison = compare_evaluation_report(loaded_report)
bundle_path = save_evaluation_bundle("reobserve-bundle.json", tags=("qa", "reobserve"))
loaded_bundle = load_evaluation_bundle(bundle_path)
bundle_comparison = compare_evaluation_bundle(loaded_bundle)
dynamic_qa_suite = run_evaluation_suite(tags=("qa", "dynamic"))
relation_shift_suite = run_evaluation_suite(names=("relation_shift_relation_timeline",))
action_validity_suite = run_evaluation_suite(tags=("qa", "action_validity"))
world_state_suite = run_evaluation_suite(tags=("qa", "world_state"))
temporal_qa_suite = run_evaluation_suite(tags=("qa", "temporal"))
retrieval_qa_suite = run_evaluation_suite(tags=("qa", "retrieval"))
nearest_intent_suite = run_evaluation_suite(question_types=("nearest_object",))
label_candidates_suite = run_evaluation_suite(tags=("qa", "label", "ambiguity"))
retrieve_subgraph_suite = run_evaluation_suite(question_types=("retrieve_subgraph",))
snapshot_qa_suite = run_evaluation_suite(tags=("qa", "snapshot"))
reobserve_suite = run_evaluation_suite(tags=("qa", "reobserve"))
vla_dynamic_suite = run_evaluation_suite(tags=("vla", "dynamic"))
vla_ambiguity_suite = run_evaluation_suite(tags=("vla", "label", "ambiguity"))
vla_reobserve_suite = run_evaluation_suite(tags=("vla", "reobserve"))
vla_anchor_manifest = list_evaluation_case_metadata(
    kinds=("vla_pick", "vla_place_relative")
)
named_manifest = list_evaluation_case_metadata(
    names=("tabletop_mug_pick", "tabletop_object_location")
)

def build_custom_scene():
    custom = DynamicSceneGraph()
    custom.set_agent_pose("agent", Pose3D(0.0, 0.0, 0.0), step=1)
    custom.upsert_object(
        "cube_1",
        "cube",
        Pose3D(0.25, -0.5, 0.4),
        BBox3D(center=Pose3D(0.25, -0.5, 0.4), size=(0.2, 0.2, 0.2)),
        confidence=0.82,
        visible=True,
        step=1,
    )
    return custom

custom_suite = run_evaluation_cases(
    (
        EvaluationCase(
            name="custom_cube_location",
            scene_fixture="custom_counter",
            kind="qa",
            tags=("qa", "custom"),
            question={"type": "object_location", "object_id": "cube_1"},
            expected={"answer": {"object_id": "cube_1"}, "error": None},
        ),
    ),
    scene_loaders={"custom_counter": build_custom_scene},
)
custom_named_suite = run_evaluation_cases(
    (
        EvaluationCase(
            name="custom_cube_location",
            scene_fixture="custom_counter",
            kind="qa",
            tags=("qa", "custom"),
            question={"type": "object_location", "object_id": "cube_1"},
            expected={"answer": {"object_id": "cube_1"}, "error": None},
        ),
    ),
    names=("custom_cube_location",),
    scene_loaders={"custom_counter": build_custom_scene},
)
custom_manifest = evaluation_cases_metadata(
    (
        EvaluationCase(
            name="custom_cube_location",
            scene_fixture="custom_counter",
            kind="qa",
            tags=("qa", "custom"),
            question={"type": "object_location", "object_id": "cube_1"},
            expected={"answer": {"object_id": "cube_1"}, "error": None},
        ),
    ),
    tags=("qa",),
)
```

The fixture registry currently includes `"ambiguous_mugs"`, `"ambiguous_plates"`,
`"tabletop"`, `"moved_mug"`, `"multi_room_rearrangement"`,
`"needs_reobserve"`, and `"relation_shift"` scenes.
`list_scene_fixture_metadata()` returns deterministic scene fixture metadata
with each fixture's name, description, and tags, and supports tag filters without
loading scene graphs. `scene_fixture_manifest()` wraps that metadata in a
standalone manifest with schema version, filters, fixture count, and a stable
SHA-256 digest for fixture-only handoffs. `scene_fixture_manifest_json()` and
`save_scene_fixture_manifest()` produce the same stable JSON format used by the
scene CLI. `load_scene_fixture_manifest()`, `validate_scene_fixture_manifest()`,
and `compare_scene_fixture_manifest()` read, validate, and compare explicit
local fixture manifest artifacts without loading scene graphs.
Each call to `load_scene_fixture()` returns a fresh in-memory graph.
The evaluation suite returns deterministic dictionaries with a top-level
`digest` for experiment records, pass/fail summary, including stable
`selected_cases` and `failed_cases` lists, per-case
actual/expected outputs, per-case tags, and a `breakdown` grouped by case kind,
QA question type, tag, and scene fixture for experiment triage.
The digest is a SHA-256 hash of the suite summary, breakdown, and results, so a
fixed case slice can be compared across deterministic runs without relying on
wall-clock data.
Each result includes deterministic `mismatches` with path, reason, category,
expected, and actual fields, so failed regression cases are reproducible and
inspectable.
QA and VLA actual outputs with a non-empty `error` also include stable
`error_category` values such as `missing_object`, `missing_label`,
`missing_reference`, `missing_target`, `invalid_time_window`,
`invalid_question`, `ambiguous_label`, `needs_reobserve`, `target_not_visible`,
`low_confidence`, `unsupported_question`, and `unsupported_relation` for
machine-comparable runtime diagnostics.
Suites and compact reports also include deterministic `runtime_error_categories`
aggregates with category counts and affected case names, so an experiment can
compare its runtime error mix without reprocessing every result. Report
validation checks that saved category entries are non-empty, positive-count
aggregates and that category counts match unique affected selected cases.
Report comparison surfaces drift in those aggregates with stable category
paths such as `missing_object.count`.
Compact reports additionally include deterministic `runtime_error_metrics` with
total case count, runtime-error case count, clean case count, runtime-error
rate, and per-category case rates for quick benchmark triage. Report validation
checks those metrics against the saved runtime error category aggregates, and
comparison surfaces metric drift with stable nested paths such as
`by_category.missing_target.case_rate`.
Compact reports preserve failed-case `mismatch_paths`, report-level
`failure_paths`, raw `failure_reasons`, and stable `failure_categories` such as
`missing_output`, `cardinality_mismatch`, `schema_mismatch`, and
`value_mismatch` for benchmark-level triage. Report validation checks that
those failure diagnostic aggregates match the saved `failed_cases` summaries.
Report comparison surfaces failure-reason, failure-category, and failure-path
drift with stable paths such as `value_mismatch` and `answer.visible`.
Their `metrics` include `case_count`, `passed_case_count`, `failed_case_count`,
`pass_rate`, and `failure_rate` for direct experiment comparison, plus grouped
count/rate metrics under `by_kind`, `by_question_type`, `by_scene_fixture`, and
`by_tag`.
Their `evidence_metrics` summarize evidence-node counts, evidence-edge counts,
VLA command evidence counts, evidence-covered cases, and average evidence items
per case, with the same grouped views for offline explainability audits.
Compact reports also include `case_selection`, a compact ordered summary of
selected case name, kind, question type, scene fixture, and tags, plus
`case_selection_digest` for validating the experiment slice. They also include
`case_digests`, a per-case SHA-256 digest summary with case name, kind, question
type, scene fixture, and pass status so a changed suite digest can be narrowed
to individual deterministic cases. Each compact report also carries the report
schema version and `report_digest`, and
`evaluation_report_digest()` plus `validate_evaluation_report()` can recompute
and validate that artifact fingerprint, case selection and case digest
consistency, per-case digest format, per-case digest metadata and pass/fail
status, summary metrics, runtime error metrics, and evidence metric internal
consistency plus non-negative value ranges across top-level and grouped
summaries before comparison.
Evaluation manifests expose `evaluation_manifest_digest()` for recomputing their
saved metadata digest, while evaluation bundles expose
`evaluation_bundle_digest()` and `bundle_digest` for validating the complete
handoff artifact before accepting it.
Evaluation cases can be filtered by tags, for example
`run_evaluation_suite(tags=("qa", "dynamic"))` selects dynamic QA regression
cases including stale next-action validity, explicit object timelines, scene
deltas, multi-room rearrangement event windows, dynamic relation shifts, and
current world state audits, while
`run_evaluation_suite(tags=("qa", "action_validity"))` selects stale-action QA
regressions that derive an old action from a deterministic baseline scene.
`run_evaluation_suite(tags=("qa", "world_state"))` selects current-scene
world-state regressions for dynamic spatial state checks.
`run_evaluation_suite(tags=("qa", "foundation"))` selects basic QA contract
regressions for agent location, agent pose history, missing-object error
diagnostics, object status, object history, and direct relative-relation checks
over the tabletop fixture.
`run_evaluation_suite(tags=("qa", "error"))` selects structured QA error-path
regressions, including missing-object, invalid-question-field,
unsupported-question, and reversed time-window diagnostics, that should remain
deterministic and local.
`run_evaluation_suite(tags=("qa", "temporal"))` selects agent history,
timeline, and delta regressions, including relation timelines, for explicit-step
memory audits.
`run_evaluation_suite(tags=("qa", "relations"))` selects static and dynamic
relation regressions, including direct relative checks and the `relation_shift`
fixture.
`run_evaluation_suite(tags=("qa", "retrieval"))` selects structured graph
retrieval regressions for the `graph_query` QA path and candidate-constrained
nearest-object QA, plus text-seeded `retrieve_subgraph` QA.
`run_evaluation_suite(tags=("qa", "snapshot"))` selects explicit-step scene
snapshot regressions for state reconstruction audits.
`run_evaluation_suite(tags=("qa", "reobserve"))` selects re-observation target
and low-confidence label-candidate regressions over the deterministic
`needs_reobserve` scene.
`run_evaluation_suite(tags=("qa", "label", "ambiguity"))` selects direct QA
candidate-list regressions for same-label object ambiguity over the deterministic
`ambiguous_mugs` scene.
`run_evaluation_suite(tags=("vla", "anchor"))` selects deterministic VLA pick
and place-relative anchor regressions. They can also be filtered by kind, for
example `run_evaluation_suite(kinds=("vla_pick", "vla_place_relative"))` runs
only anchor-planner cases.
`run_evaluation_suite(tags=("vla", "dynamic"))` selects stale VLA action
regressions, including stale pick targets and stale place-relative reference
anchors.
`run_evaluation_suite(tags=("vla", "reobserve"))` selects VLA planner
regressions where low-confidence invisible pick targets, place-relative
targets, or place-relative references must return `needs_reobserve` without
emitting a command.
`run_evaluation_suite(tags=("vla", "error"))` selects VLA planner error-path
regressions, including missing pick/place target inputs, missing object pick
targets, missing semantic-label targets, invisible pick targets and
place-relative targets/references, visible low-confidence pick targets and
place-relative targets/references, missing place references, missing reference
inputs, and unsupported place relations that must return structured diagnostics
without emitting a command.
`run_evaluation_suite(tags=("vla", "label", "ambiguity"))` selects semantic
target and reference ambiguity regressions where the planner must return
`ambiguous` instead of choosing among same-label objects, with deterministic
candidate details for offline diagnosis. QA intents can be filtered directly with
`question_types`, for example
`run_evaluation_suite(question_types=("nearest_object",))` or
`run_evaluation_suite(question_types=("label_candidates",))` or
`run_evaluation_suite(question_types=("retrieve_subgraph",))` or
`run_evaluation_suite(question_types=("next_action_validity",))` or
`run_evaluation_suite(question_types=("relative_relation",))`.
Tag, kind, and question-type filters can be combined.
Use `names=` on `list_evaluation_cases()`, `list_evaluation_case_metadata()`,
`run_evaluation_suite()`, `run_evaluation_cases()`, or
`evaluation_cases_metadata()` to select exact built-in or custom cases while
preserving the caller-supplied order. This is useful for publishing a fixed
regression slice as part of an experiment record.
Callers can also pass temporary `EvaluationCase` objects to
`run_evaluation_cases()` to run local experiment cases without modifying the
built-in registry. Custom cases may use caller-supplied `scene_loaders`, so local
experiments can provide deterministic in-memory scenes outside the built-in
fixture registry.
`list_evaluation_case_metadata()` returns deterministic case metadata without
loading scenes or executing QA/VLA logic. `evaluation_case_listing()` wraps that
metadata with the listing schema version, applied filters, case count, and digest;
`evaluation_case_listing_digest()` recomputes the fingerprint for filtered
discovery handoffs. `save_evaluation_case_listing()`,
`load_evaluation_case_listing()`, `validate_evaluation_case_listing()`, and
`compare_evaluation_case_listing()` support explicit-path listing artifacts and
current-code metadata drift checks. Listing validation checks the schema
version, saved digest, and case count before handoff. Each entry includes the
case name, scene fixture, fixture description/tags when available, kind, tags,
structured question copy, question type, baseline fixture metadata for
stale-action cases, action target fields, relation, and expected top-level keys.
Listing helpers support the same tag, kind, exact-name, and `question_types`
filters for deterministic QA intent discovery. Caller-supplied fixture names that are not in
the built-in scene registry keep `scene_description=None` and `scene_tags=[]`.
Use `evaluation_cases_metadata()` for the same deterministic manifest over
caller-supplied custom `EvaluationCase` sequences.
For VLA pick and place-relative evaluation cases, callers may provide explicit
object ids (`target_object`, `reference_object`) or semantic labels
(`target_label`, `reference_label`) for deterministic target resolution.
Label-based cases preserve planner outcomes such as `ambiguous` when multiple
visible objects share the same label; ambiguous results include stable candidate
details with pose, visibility, confidence, last-seen step, and re-observation
status. The built-in `ambiguous_mugs` and `ambiguous_plates` fixtures provide
deterministic same-label scenes for those regressions.

`SpatialQAEngine` supports direct structured intents for agent location, agent
history, agent timeline, object location, object status, relative relations,
nearest object, label candidates, object history, object timeline, relation timeline,
re-observation targets, action validity, explicit-step scene snapshots, world
state, and recent events. It also exposes deterministic `graph_query`
answers backed by `GraphTool.query_graph()` so experiments can retrieve
structured graph nodes and edges through the QA layer, plus `retrieve_subgraph`
for text-seeded graph retrieval backed by `GraphTool.retrieve_subgraph()`.
Object location exposes pose, visibility, confidence, state step, current
containment location, and containment plus `STATE_CHANGED` evidence. Object
room QA resolves the current room-level containment path when one exists,
returning room id/label, path nodes, and evidence edge IDs for multi-room
audits. Object status exposes visibility, confidence, last-seen pose/step, and
whether the target needs re-observation. Nearest-object QA can optionally constrain the
deterministic search with a caller-supplied `candidates` list, and returns the
selected distance plus stable candidate distance diagnostics for offline
triage. Label-candidate QA returns stable object-id ordered candidates for a
semantic label, marks
same-label ambiguity, and includes per-object `STATE_CHANGED` evidence. Object
timeline returns each explicit object state step with
pose, visibility, confidence, last-seen memory, current containment location, and
per-step evidence edges. Relation timeline returns deterministic relation-edge
records filtered by source, relation, destination, reference frame, and explicit
step windows. Agent timeline returns each explicit agent pose step with per-step
`STATE_CHANGED` evidence edges. Re-observation targets list current invisible
low-confidence objects, optionally filtered by label, without requiring an agent
pose or external model.

`GraphTool.scene_snapshot(step=...)` and the matching QA intent reconstruct the
latest agent pose, object states, and containment locations at or before an
explicit step. `GraphTool.scene_delta(from_step=..., to_step=...)` compares two
snapshots and reports changed agent pose, changed objects, changed fields, and
window evidence. `GraphTool.world_state(visible=...)` returns the current agent
pose, current object states, current containment locations, and evidence IDs.
`GraphTool.current_room(object_id)` resolves the object's latest containment
path to a room when possible, returning the room id, room label, containment
path, and evidence edge IDs for local multi-room audits.
`GraphTool.recent_events(since_step=..., until_step=...)`
returns action/event nodes, step-window change edges, and evidence IDs for
auditing what just happened. This makes temporal regression cases deterministic
without clocks or simulators.

Agent poses are stored as current state plus deterministic `AgentPoseState`
history. Each `set_agent_pose(..., step=...)` call creates a `state:<agent>:<step>`
node and `STATE_CHANGED` edge, so graph export/import preserves where the agent
was across explicit steps.

Spatial relations are computed geometrically. `NEAR` uses 3D bbox surface
distance, `ON` requires vertical contact plus configurable support-area overlap,
`SUPPORTS` is the inverse of a valid `ON` relation, `INSIDE` checks whether the
source bbox centroid lies within the destination bbox, and egocentric relations
respect the agent yaw. `GraphTool.compute_distance()` returns a stable rounded
world-frame distance payload with source/destination poses for offline metrics.
`GraphTool.update_spatial_relations()` can append inferred relation edges for an
explicit caller-supplied step while preserving previous relation history.
`VISIBLE_FROM`, `REACHABLE_FROM`, and `OCCLUDES` are explicit placeholder edges:
they can be saved and queried, including `image` and `object:*` reference-frame
strings, but the MVP does not infer them from RGB/depth, frustums, or navmeshes.

Episode JSONL helpers define a deterministic simulator/mock collection boundary
without importing simulator packages. Each `EpisodeFrame` line records explicit
episode id, scene id, step, optional RGB/depth/segmentation artifact paths,
agent pose, optional action, visible object ids, and metadata. The
`episode_sequence_*` helpers support stable JSONL round-trips, SHA-256 digests,
summary metadata, explicit-path save/load, validation for episode/step ordering
and duplicate episode steps, and canonical round-trip comparison.
`scripts/episodes.py --validate`, `--summary`, and `--compare` provide the same
local handoff checks from the shell and return structured non-zero JSON for
invalid explicit files.

The optional AI2-THOR adapter is a boundary, not a simulator integration in the
default runtime. `AI2ThorAdapterConfig` requires caller-supplied `steps`;
`build_mock_ai2thor_episode()` creates deterministic episode frames with local
artifact paths and metadata that can feed the oracle builder; and
`AI2ThorEpisodeCollector.collect_episode()` raises the missing optional
dependency diagnostic unless a future integration supplies an external collector
layer. `scripts/collect_ai2thor.py --mock` writes valid Episode JSONL to an
explicit `--output` path, while non-mock collection fails closed with local JSON
diagnostics.

The oracle builder turns mock episode metadata into a local `DynamicSceneGraph`
without importing simulator packages. `EpisodeFrame.metadata` may include
`rooms`, `regions`, `objects`, and explicit `relations`; object records include
pose, bbox, confidence, visibility, optional `room_id`/`region_id`, and arbitrary
state keys. `build_oracle_graph_from_episode()` adds agent poses, containment,
explicit relation edges, action nodes, event nodes for moved objects, and
`MOVED_FROM` / `MOVED_TO` evidence while preserving last-seen pose/step for
hidden low-confidence objects through the existing graph memory semantics.
`oracle_graph_report()`, `oracle_graph_report_digest()`,
`save_oracle_graph_report()`, `load_oracle_graph_report()`,
`validate_oracle_graph_report()`, and `compare_oracle_graph_report()` provide
stable explicit-path graph/report handoffs. `scripts/build_oracle_graph.py`
offers the same build, validation, and comparison flow from the shell.

`ObservationIngestor` is the deterministic boundary for mock perception frames:
callers provide explicit `SceneObservation.step`, optional agent pose, room/region
nodes, object observations, and optional relation inference settings. It writes
only in-memory graph state and never calls models, clocks, random sources, or
network services. Object observation `source`, `source_name`, or `source_kind`
metadata is normalized onto object nodes, and inferred relation edges are marked
as `geometry_inference`. `scene_observation_to_dict()`, `scene_observation_to_json()`,
`scene_observation_from_dict()`, `scene_observation_from_json()`,
`save_scene_observation()`, and `load_scene_observation()` provide stable JSON
round-trips and explicit-path local files for offline mock perception handoffs.
The `scene_observation_sequence_*` helpers provide the same stable JSON
round-trip for ordered multi-frame observation streams, including explicit step
lists and observation counts. `validate_scene_observation_sequence_payload()`
checks raw sequence schema/count/step shape and returns the sequence digest plus
graph-free summary for pre-ingest gates. `scene_observation_sequence_summary()`
and `scene_observation_sequence_summary_digest()` provide graph-free dataset
handoff metadata for ordered observation streams, including step spans,
object-observation counts, unique object IDs, label counts, visibility counts,
low-confidence counts, and re-observation candidate counts.
`scene_observation_sequence_summary_json()`,
`save_scene_observation_sequence_summary()`,
`load_scene_observation_sequence_summary()`,
`validate_scene_observation_sequence_summary()`, and
`compare_scene_observation_sequence_summary()` support explicit-path summary
artifacts, digest checks, internal count consistency checks, and current
sequence drift checks before graph ingest. `scripts/observations.py` can
summarize such a sequence with `--summarize-sequence`, validate or compare the
summary artifact, or ingest it into a new in-memory graph, export graph JSON to
an explicit `--output-graph` path, and emit a stable ingest report with per-step
`IngestResult` data plus the input sequence digest, graph digest/summary, and
report digest.
`scene_observation_sequence_digest()`, `observation_ingest_report_digest()`,
`save_observation_ingest_report()`, `load_observation_ingest_report()`,
`validate_observation_ingest_report()`, and
`compare_observation_ingest_report()` support explicit-path report validation,
path consistency checks, digest tamper checks, and deterministic input,
graph-file, and output drift checks.

Planner results include deterministic `details` for ambiguity and replan causes,
including same-label candidate state, expected/current pose, last-seen steps,
current location, and evidence edge IDs when an action becomes stale.
`place_relative` validation treats the target pose as the planned anchor, not the
target object's current pose, and returns `stale_reference_state` when the
reference object moves enough to invalidate that anchor.
`VLAAnchorPlanner.plan_place_relative()` also accepts `target_label` and
`reference_label` keyword arguments so semantic place requests can resolve to
object ids without invoking any model.

All built-in scene and graph IO helpers are deterministic and avoid network,
current-time, random, model, robot, and simulator dependencies.
