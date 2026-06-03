# DSG-SpatialQA Lab Roadmap

This roadmap tracks the deterministic validation-loop MVP and the remaining
research extensions for verifying Dynamic Scene Graph benefits.

## Completed MVP

The current project implements the offline deterministic platform needed to
evaluate the four target capabilities:

- spatial QA,
- dynamic memory,
- `GraphTool` queries,
- mock interactive task ability.

Completed layers:

1. Episode JSONL schema and IO.
2. Mock episode to oracle DSG builder.
3. AI2-THOR adapter skeleton with deterministic mock fallback.
4. Extended deterministic spatial relations.
5. Oracle QA generator.
6. QA metrics and report validation.
7. Deterministic local baseline runner.
8. Oracle-vs-predicted graph metrics.
9. Predicted DSG skeleton from mock perception.
10. QA error attribution.
11. Static dashboard export.
12. Active EQA task skeleton and task metrics.
13. Habitat adapter skeleton.
14. Benchmark manifest tooling.
15. Architecture, roadmap, and artifact format documentation.
16. Real experiment readiness reporting over manifest-linked local artifacts.

## Milestones

### Milestone A: Oracle QA MVP

Status: complete.

The chain from mock episode to oracle graph to generated QA to GraphTool
baseline to QA evaluation runs locally and deterministically.

### Milestone B: Graph Construction Evaluation

Status: complete.

Oracle and predicted mock graphs can be compared. Object recall, relation F1,
matched-state metrics, and QA error attribution are available.

### Milestone C: Interactive Skeleton

Status: complete for deterministic mock tasks.

Mock active EQA tasks run over explicit graph steps. Reports include task
success, answer accuracy, action count, evidence coverage, answer-graph
consistency, per-action evidence snapshots, and budget-vs-success analysis by
max-action budget. The local `next_best_view` policy deterministically targets
missing required evidence without real navigation.

Not included: real navigation, simulator-backed next-best-view execution, real
robot interaction, active-task media previews, or rich interactive per-action
review.

### Milestone D: Benchmark And Simulator Extension

Status: complete for deterministic mock adapters and manifests.

AI2-THOR and Habitat have optional adapter skeletons with mock collection.
Benchmark manifests summarize multiple episodes, oracle graphs, and QA
datasets.

Not included: real simulator collection in the default runtime.

## Near-Term Research Extensions

### Richer Graph Matching

Current graph evaluation uses exact object IDs and exact relation-edge keys by
default. Optional label+center and room-aware label+center matching are
implemented for changed predicted object IDs, including relation remapping
through matched object pairs, duplicate-track / ID-fragmentation diagnostics,
confidence-weighted object/relation metrics, and prediction-source confidence
precision slices.
Next steps:

- real adapter confidence calibration against source-specific slices.

### Active Task Review

Current active EQA is a mock graph-step loop. The static dashboard can include
active task report panels with task transcripts and required/observed evidence
IDs, per-action evidence snapshots, and report-level budget analysis. Next
steps:

- simulator-backed next-best-view execution behind deterministic adapter
  boundaries,
- explicit integration points for future simulator-backed task execution.

### External Prediction Imports

Status: complete for deterministic offline imports.

Default runtime must not call external models. The project now imports local
offline prediction records into standard `QAPrediction` JSONL with source
metadata, missing/unknown case diagnostics, report validation, and current-file
comparison. QA delta and experiment summary artifacts retain question-type,
tag, reference-frame, scene, and episode diagnostic slices so imported-source
regressions can be localized without reloading full per-case reports.
Experiment summaries also retain compact graph construction diagnostics from
graph eval artifacts, so imported-source QA lift can be reviewed beside
predicted DSG object recall, relation F1, state accuracy, track-fragmentation,
and prediction-source precision evidence. They also retain compact attribution
diagnostics from QA attribution reports, including research-axis failure
summaries, so imported-source regressions can be separated into graph
construction, evidence missing, benchmark/engine, or reasoning/tool-use failure
categories. The final summary links attribution reports to graph eval reports
by graph digest, giving each predicted graph a compact quality-plus-failure
view. Final experiment records retain a diagnostic ledger of QA, graph,
attribution, and linkage coverage, and can seal a linked real package readiness
digest/status when externally collected artifacts pass the readiness gate.
Offline import reports also derive stable
source profiles from explicit metadata, including source key, adapter, model,
prompt, dataset, metadata keys, and capability axes. This supports VLM-only or
caption-memory style outputs without provider calls during default
verification. Benchmark manifests can now record offline prediction import
reports, and experiment summaries/final records project them into
`source_profile_matrix` rows for side-by-side source review. Static dashboard
exports show those rows and expose a Source Profile filter for imported-source
review. `scripts/check_offline_controls.py` now gates the real control matrix
itself, requiring VLM-only, multi-frame VLM, caption-memory, and graph-text LLM
source kinds with complete gold-case coverage and a shared QA digest.
`scripts/run_offline_controls.py` now imports all four local control prediction
files and writes the matrix report in one deterministic handoff.
Real experiment readiness then compares that shared offline-control digest with
the benchmark manifest `qa_digest` before accepting the controls as aligned
evidence, and rejects offline controls with incomplete gold-case coverage or
import diagnostics. The matrix report itself is now recorded as a manifest
artifact and must be ready, with required source kinds covering the requested
controls, before a real package can pass readiness.

Next steps:

- provide real offline prediction JSONL files for VLM-only, multi-frame,
  caption-memory, and graph-text LLM controls;
- pass the offline control matrix gate before using source profiles and
  dashboard filters to compare those controls against DSG GraphTool and
  predicted-DSG GraphTool outputs.

### Predicted DSG Integration

The current predicted DSG path includes mock episode perception plus an
explicit `SceneObservation` sequence input for detector/RGB-D outputs produced
outside the deterministic runtime. Both paths preserve source metadata on graph
object nodes, mark inferred relation edges as `geometry_inference`, and write
stable predicted graph reports. `scripts/check_predicted_dsg.py` now gates
observation-sequence predicted graphs as detector/RGB-D evidence by requiring
multi-frame object observations plus detector, RGB, and depth evidence before
readiness treats them as real predicted DSG artifacts. Future work can add
optional real perception adapters only behind explicit boundaries:

- local file inputs,
- optional extras,
- deterministic mocked tests,
- no default model imports,
- stable graph and report artifacts.

Next steps:

- run a small detector/RGB-D pipeline outside the default runtime;
- write its outputs as `SceneObservation` sequences;
- build predicted graph reports with `--input-kind observation_sequence`;
- pass the predicted DSG evidence gate;
- compare those predicted graphs against oracle DSGs and QA/task outcomes.

### Real Experiment Readiness

Status: complete for deterministic readiness gating.

`scripts/check_real_experiment.py` checks whether a benchmark manifest and its
linked local artifacts are enough to begin answering whether DSG improves over
controls. It requires an explicit `real` data declaration, minimum
episode/scene/QA coverage, spatial/dynamic/GraphTool-style QA coverage,
real collection reports, offline VLM-only, multi-frame VLM, caption-memory,
and graph-text controls, observation-backed predicted DSG reports, graph eval,
attribution, active-task, dashboard review artifacts, complete offline-control
coverage, clean offline-control import diagnostics, and matching QA digests
between offline controls and the benchmark manifest, plus a ready offline
control matrix report artifact whose required source kinds cover the requested
controls. Mock or underspecified packages fail the gate instead of being treated
as real evidence.

Next steps:

- run a small AI2-THOR or Habitat artifact package;
- pass the real collection evidence gate;
- import real offline control predictions;
- pass the readiness gate before summarizing DSG lift.

### Real Experiment Package Assembly

Status: complete for deterministic package assembly from explicit local
artifacts.

`scripts/assemble_real_experiment.py` takes caller-supplied episode JSONL files
plus local QA, graph, active-task, dashboard, offline-prediction, and
predicted-graph reports, including real collection reports, writes a benchmark
manifest, writes the matching real experiment readiness report, and exits
non-zero if the readiness gate is not ready. It does not collect simulator data
or run perception/model inference.

Next steps:

- use the assembler on a small real AI2-THOR or Habitat package;
- require real VLM-only, multi-frame, caption-memory, and graph-text offline
  prediction imports;
- run the deterministic real package handoff once the package is ready.

### Real Experiment Package Run/Import

Status: complete for deterministic run/import handoff from explicit local
artifacts.

`scripts/run_real_experiment.py` composes package assembly, real readiness,
experiment summary, and final experiment record generation. It does not collect
simulator data, run detectors, or call VLM/LLM providers. It writes the summary
and final record only when the linked real readiness report is ready; otherwise
it returns non-zero structured diagnostics and leaves the final evidence record
unwritten.

Next steps:

- supply a small real AI2-THOR or Habitat artifact package;
- import real VLM-only, multi-frame, caption-memory, and graph-text offline
  predictions;
- compare DSG rows only after the assembled package passes readiness.
- run the package handoff and inspect DSG-vs-control lift.

## Long-Term Research Extensions

### Real Simulator Collection

AI2-THOR and Habitat real collection can be added only as optional adapter
paths. Default tests should continue to use mock collection and missing
dependency diagnostics.

Required properties:

- explicit scene and episode ids,
- explicit action or step sequences,
- no current-time generated IDs,
- no hidden network calls,
- artifact validation before graph construction.

### Real Interactive Tasks

Real interactive tasks should reuse the active EQA task schema where possible.
The active task report should remain the common metric surface so mock and real
task runs can be compared.

Potential metrics:

- task success,
- answer accuracy,
- action count,
- evidence coverage,
- graph consistency,
- re-observation count,
- path length or simulator action cost when explicitly available.

### Persistent Datasets

The MVP stays in memory and explicit local files. Add persistence only when a
real benchmark or integration needs it.

Any persistence layer should preserve:

- stable artifact digests,
- explicit dataset versioning,
- reproducible local export,
- validation and comparison commands.

## Development Rules For Future Work

- Keep runtime dependencies empty unless a feature is explicitly optional.
- Keep real simulators, VLMs, LLMs, robots, and perception models outside the
  default verification path.
- Add typed APIs, focused tests, and CLI or Python examples for each new
  artifact surface.
- Add validation and comparison helpers for new saved artifacts.
- Run `python scripts/verify.py` before handoff.
- Update `README.md`, `docs/AI_RUNBOOK.md`, `docs/VERIFICATION.md`, and this
  roadmap when a feature changes the experiment workflow.

## Completion Criteria For The Current Objective

The project is aligned with the stated objective when the following remain
true:

- spatial QA can be generated, predicted, evaluated, and attributed;
- dynamic memory is represented by explicit-step graph history and QA cases;
- `GraphTool` query utility is evaluated against baselines;
- interactive task ability has at least a deterministic mock task loop and
  metrics;
- benchmark manifests aggregate episode, graph, QA, and coverage artifacts;
- all default verification gates pass without external services.
