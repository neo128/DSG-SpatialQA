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
comparison. This supports VLM-only or caption-memory style outputs without
provider calls during default verification.

Next steps:

- richer source metadata conventions,
- side-by-side eval manifests for multiple imported sources,
- optional real adapter boundaries that write offline files only.

### Predicted DSG Integration

The current predicted DSG path is mock perception with deterministic
source-metadata propagation from detections to graph object nodes, inferred
relation edges marked as `geometry_inference`, and predicted graph report
summaries by detection source. Future work can add optional real perception
adapters only behind explicit boundaries:

- local file inputs,
- optional extras,
- deterministic mocked tests,
- no default model imports,
- stable graph and report artifacts.

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
