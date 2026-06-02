# Benchmark And Artifact Formats

This document summarizes the local artifact formats used by DSG-SpatialQA Lab.
All paths are explicit caller-supplied local paths. Validation and comparison
commands return structured JSON and non-zero exit status for invalid or drifted
artifacts.

## Artifact Chain

```text
episode.jsonl
-> oracle-graph.json
-> predicted-graph.json
-> qa.jsonl
-> predictions.jsonl
-> qa-eval-report.json
-> graph-eval-report.json
-> error-attribution.json
-> dashboard/
-> active-tasks.jsonl and active-report.json
-> benchmark-manifest.json
```

## Episode JSONL

Schema version: `dsg-spatialqa-lab.episode-frame.v1`

One frame per line. Each frame records:

- `episode_id`
- `scene_id`
- `step`
- `rgb_path`
- `depth_path`
- `segmentation_path`
- `agent_id`
- `agent_pose`
- `action`
- `visible_object_ids`
- `metadata`

Core helpers:

- `episode_sequence_to_jsonl()`
- `episode_sequence_from_jsonl()`
- `episode_sequence_digest()`
- `save_episode_sequence()`
- `load_episode_sequence()`
- `validate_episode_sequence()`
- `compare_episode_sequence()`

Baseline CLI:

```bash
python scripts/episodes.py --validate mock-episode.jsonl
python scripts/episodes.py --summary mock-episode.jsonl
python scripts/episodes.py --compare mock-episode.jsonl
```

## Graph JSON

Schema version: `1`

Graph JSON records nodes, edges, object states, object state history, agent
poses, and agent pose history.

Core helpers:

- `graph_to_json()`
- `graph_from_json()`
- `graph_json_digest()`
- `graph_summary()`
- `save_graph_json()`
- `load_graph_json()`
- `validate_graph_report()`
- `compare_graph_report()`

CLI examples:

```bash
python scripts/build_oracle_graph.py \
  --input mock-episode.jsonl \
  --output-graph oracle-graph.json \
  --report oracle-report.json

python scripts/scene.py --validate oracle-graph.json
```

## QA JSONL

Schema version: `dsg-spatialqa-lab.qa-case.v1`

Each QA case records:

- stable case id,
- scene and episode ids,
- graph digest,
- explicit step,
- question payload,
- gold answer,
- answer type,
- choices,
- reference frame,
- required evidence nodes and edges,
- tags and difficulty.

Core helpers:

- `generate_qa_cases()`
- `qa_dataset_jsonl()`
- `qa_dataset_digest()`
- `save_qa_dataset()`
- `load_qa_dataset()`
- `validate_qa_dataset()`
- `compare_qa_dataset()`

CLI:

```bash
python scripts/generate_qa.py \
  --graph oracle-graph.json \
  --scene-id mock_scene \
  --episode-id mock_episode \
  --max-cases 100 \
  --output qa.jsonl
```

## Prediction JSONL

Schema version: `dsg-spatialqa-lab.qa-prediction.v1`

Each prediction records:

- QA case id,
- predicted answer,
- evidence nodes,
- evidence edges,
- confidence,
- optional error.

Core helpers:

- `qa_predictions_jsonl()`
- `qa_predictions_digest()`
- `save_qa_predictions()`
- `load_qa_predictions()`

## Offline Prediction Import

Record schema version: `dsg-spatialqa-lab.offline-prediction-record.v1`

Import report schema version:
`dsg-spatialqa-lab.offline-prediction-import-report.v1`

Offline records provide local external predictions without calling a provider:

- QA case id,
- structured answer object,
- optional evidence node IDs,
- optional evidence edge IDs,
- confidence,
- optional error,
- optional per-record metadata.

The import report records source name/kind/metadata, a derived
`source_profile` with source key, adapter, model, prompt, dataset, metadata
keys, and capability axes, QA/input/prediction paths, artifact digests,
imported/unknown/missing/duplicate counts, missing case IDs, unknown case IDs,
and per-record import diagnostics. Validation recomputes the profile from the
explicit source metadata so a recomputed report digest cannot hide profile
drift.

CLI:

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
```

Baseline CLI:

```bash
python scripts/run_baselines.py \
  --baseline graph_tool \
  --graph oracle-graph.json \
  --qa qa.jsonl \
  --pred predictions.jsonl
```

## QA Evaluation Report

Schema version: `dsg-spatialqa-lab.qa-eval-report.v1`

Reports include:

- exact-match accuracy,
- multiple-choice accuracy,
- numeric MAE,
- evidence node and edge recall,
- answer-graph consistency,
- breakdowns by scene id, episode id, question type, tag, reference frame, and
  research axis,
- report digest.

The research-axis breakdown maps QA cases to `spatial_qa`,
`dynamic_memory`, and `graph_tool_query` groups so RQ1-RQ3 evidence can be
reviewed without reinterpreting raw tags.

Pairwise delta reports use schema version
`dsg-spatialqa-lab.qa-eval-delta-report.v1`. They compare a candidate QA eval
report against a baseline QA eval report and record summary, metric,
scene, episode, question-type, tag, reference-frame, and research-axis deltas
while preserving the source report digests and explicit report paths for
validation and current-file comparison.

CLI:

```bash
python scripts/run_qa_eval.py \
  --gold qa.jsonl \
  --pred predictions.jsonl \
  --report qa-eval-report.json
python scripts/run_qa_eval.py \
  --candidate-report graph-tool-report.json \
  --baseline-report majority-report.json \
  --candidate-name graph_tool \
  --baseline-name majority \
  --delta-report qa-delta-report.json
python scripts/run_qa_eval.py --validate-delta-report qa-delta-report.json
python scripts/run_qa_eval.py --compare-delta-report qa-delta-report.json
```

## Graph Evaluation Report

Schema version: `dsg-spatialqa-lab.graph-eval-report.v1`

Reports compare oracle and predicted graph JSON files using exact object IDs
and exact relation-edge keys by default. Optional `label_center` matching
matches same-label objects by nearest bbox center under a caller-supplied
threshold, remaps relation edges through matched object pairs, and reports
duplicate-track / ID-fragmentation diagnostics. Optional `label_center_room`
matching applies the same label+center rule and also requires current-room
agreement.

Metrics include:

- object precision and recall,
- object label accuracy,
- relation precision, recall, and F1,
- confidence-weighted object and relation precision, recall, and F1,
- matched-object state accuracy,
- bbox center error,
- duplicate-track and ID-fragmentation diagnostics,
- object-label, relation, and prediction-source confidence breakdowns.

Predicted graph reports also summarize mock detections by source. Graph JSON
object nodes may carry normalized `source`, `source_name`, or `source_kind`
metadata, and deterministic inferred relation edges use
`source: geometry_inference`.

CLI:

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
python scripts/evaluate_graphs.py \
  --oracle oracle-graph.json \
  --predicted predicted-graph.json \
  --matching label_center_room \
  --center-distance-threshold 0.25 \
  --report graph-eval-label-center-room-report.json
```

Non-default reports include a `matching` object so validation and comparison
can replay the saved mode and threshold.

## Error Attribution Report

Schema version: `dsg-spatialqa-lab.error-attribution-report.v1`

Attribution connects QA cases, oracle graph answers, predicted graph answers,
and prediction JSONL outputs.

Current top-level categories include:

- `benchmark_or_engine_error`
- `evidence_missing`
- `graph_construction`
- `reasoning_or_tool_use`

Case rows include `predicted_evidence_sources`, derived from required nodes and
edges present in the predicted graph. The summary includes `by_research_axis`
for `spatial_qa`, `dynamic_memory`, and `graph_tool_query`, plus
`by_predicted_evidence_source` with case counts, error counts, error-category
counts, and evidence-error-category counts per source.

CLI:

```bash
python scripts/attribute_errors.py \
  --gold qa.jsonl \
  --oracle-graph oracle-graph.json \
  --predicted-graph predicted-graph.json \
  --predictions predictions.jsonl \
  --report error-attribution.json
```

## Dashboard Bundle

Schema version: `dsg-spatialqa-lab.dashboard-bundle.v1`

The static dashboard export writes:

- `dashboard.json`
- `index.html`

Bundle content includes QA cases, predictions, eval rows, optional attribution
rows, research-axis attribution summaries, predicted-evidence source summaries,
evidence subgraphs, frame paths when present, graph summary, optional active
task review panels, optional active task delta review tables, optional
experiment summary review rows, and bundle digest.
The HTML table exposes local Research Axis and Evidence Source filters for
per-case RQ-level and source-level review. The `active_task_review` section
records the source report digest, summary, metrics, budget analysis, per-task
case result, transcript, action evidence snapshots, required evidence IDs,
observed evidence IDs, and missing evidence IDs. The
`active_task_delta_review` section records the delta report digest,
candidate/baseline names and report digests, source report paths, summary
deltas, metric deltas, and max-action budget deltas.
The `experiment_summary_review` section records the experiment summary digest,
manifest path and digest, source artifact digests, RQ1-RQ4 research-question
metrics, per-measurement matrix rows, failure-linkage rows that connect graph
quality metrics to QA failure attribution, and summary counts.

CLI:

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

## Active EQA Task JSONL

Schema version: `dsg-spatialqa-lab.active-eqa-task.v1`

Each active task records:

- task id,
- scene and episode ids,
- initial graph step,
- question,
- gold answer,
- success conditions,
- max action budget,
- required evidence.

Reports use schema version `dsg-spatialqa-lab.active-task-report.v1` and score
task success, answer accuracy, action count, evidence coverage,
answer-graph consistency, per-action evidence snapshots, and budget-vs-success
analysis by max-action budget.

Supported deterministic local policies include `direct_answer`, `sweep_explore`,
`graph_uncertainty`, `oracle_evidence`, and `next_best_view`. The
`next_best_view` policy records missing-required-evidence action targets; it
does not perform simulator navigation.

CLI:

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

Report comparison reloads the recorded explicit task and graph paths, reruns
the recorded policy, and compares summary, metrics, budget analysis, tasks,
results, and case rows against the saved artifact.

Active task delta reports use schema version
`dsg-spatialqa-lab.active-task-delta-report.v1`. They compare a candidate
active task report against a baseline active task report, preserve both source
report digests and explicit report paths, and record:

- summary deltas for task count, success count, failure count, and total action
  count,
- metric deltas for task success, answer accuracy, answer-graph consistency,
  evidence coverage, and action count,
- budget deltas by `max_actions`, including success-rate,
  evidence-coverage, and action-count lift.

Delta comparison reloads only the recorded explicit candidate and baseline
report paths and detects current-file drift in summary, metric, and budget
deltas.

## Benchmark Manifest

Schema version: `dsg-spatialqa-lab.benchmark-manifest.v1`

Benchmark manifests aggregate explicit episode-derived artifacts. Each artifact
entry records:

- `episode_id`
- `scene_id`
- `episode_path`
- `episode_step_count`
- `graph_path`
- `graph_digest`
- `qa_path`
- `qa_count`
- `qa_dataset_digest`
- `source`

Top-level manifest fields include:

- `dataset_name`
- `scene_count`
- `episode_count`
- `qa_count`
- `task_count`
- `graph_digests`
- `qa_dataset_digests`
- `filters`
- `coverage`
- `summary`
- optional `experiment_artifacts`
- optional `experiment_artifact_digests`
- `manifest_digest`

Coverage includes:

- by question type,
- by scene,
- by episode,
- by reference frame,
- by tag,
- dynamic/static split,
- oracle/predicted source split.

When optional experiment artifact paths are provided, `experiment_artifacts`
records explicit local QA eval reports, QA eval delta reports, active task
reports, active task delta reports, dashboard bundles, graph eval reports, and
predicted graph reports. Each entry stores:

- `artifact_type`
- `path`
- `schema_version`
- `digest`

`experiment_artifact_digests` maps `<artifact_type>:<filename>` to the stored
digest. Manifest comparison reloads these explicit local artifact paths and
recomputes their schema-aware digests to detect current-file drift.

CLI:

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

## Experiment Summary Reports

Schema version: `dsg-spatialqa-lab.experiment-summary-report.v1`

Experiment summary reports turn a benchmark manifest's recorded experiment
artifacts into a compact four-question project summary:

- `spatial_qa`: QA delta exact-match lift for spatial QA cases.
- `dynamic_memory`: QA delta exact-match and evidence-recall lift for dynamic
  memory cases.
- `graph_tool_query`: QA delta exact-match lift for GraphTool-backed query
  cases.
- `interactive_task`: active-task success-rate lift for interactive task cases.

Each research-question row includes the primary metric, supporting metrics,
measurements, and a deterministic `verdict`: `improved` for positive primary
metric deltas, `unchanged` for zero deltas, `regressed` for negative deltas, and
`inconclusive` when evidence is missing or case counts do not match.

Top-level fields include:

- `manifest_path`
- `manifest_digest`
- `source_artifact_digests`
- `source_artifacts`
- `qa_delta_comparisons`
- `qa_diagnostic_slices`
- `active_task_delta_comparisons`
- `graph_eval_summaries`
- `graph_construction_diagnostics`
- `error_attribution_summaries`
- `error_attribution_diagnostics`
- `failure_linkage_diagnostics`
- `research_questions`
- `readiness`
- `summary`
- `report_digest`

`qa_diagnostic_slices` groups each QA delta artifact by scene id, episode id,
question type, tag, and reference frame so experiment reviews can localize
spatial QA, memory, and GraphTool lift or regression without reloading full
per-case reports.

`graph_eval_summaries` and `graph_construction_diagnostics` retain compact
oracle-vs-predicted graph quality evidence for each graph eval artifact:
object recall, relation F1, matched-object state accuracy, duplicate-track /
ID-fragmentation counts, graph summary counts, oracle/predicted graph digests,
and prediction-source precision slices. This lets final reviews compare RQ lift
against predicted DSG construction quality without embedding full graph eval
reports.

`error_attribution_summaries` and `error_attribution_diagnostics` retain compact
QA failure attribution evidence for each attribution artifact: input digests,
answer-correct counts, oracle/predicted GraphTool correctness counts, error
category counts, evidence-error category counts, and predicted-evidence source
summaries. Each attribution summary also includes `by_research_axis` rows for
`spatial_qa`, `dynamic_memory`, and `graph_tool_query`. This lets final reviews
connect RQ lift and graph construction quality to failure causes such as
`graph_construction`, `evidence_missing`, and `reasoning_or_tool_use`.

`failure_linkage_diagnostics` links each attribution artifact to a graph eval
artifact when their oracle and predicted graph digests match. Each row includes
the attribution summary, linked graph eval artifact key, graph primary metrics,
duplicate-track / ID-fragmentation diagnostics, graph digests, and whether the
link was matched or left unmatched. This gives the final handoff a compact
per-predicted-graph view of graph quality and QA failure causes.

`readiness` is a deterministic coverage gate for final experiment handoff. It
lists required, available, and missing research questions, required and missing
source artifact types, per-RQ measurement checks, and a `status` of `ready` only
when spatial QA, dynamic memory, GraphTool query, and interactive task evidence
are all present.

`summary.verdict_counts` aggregates the four RQ verdicts so automation can see
how many project questions improved, stayed unchanged, regressed, or remained
inconclusive.

CLI:

```bash
python scripts/summarize_experiment.py \
  --manifest benchmark-manifest.json \
  --report experiment-summary.json
python scripts/summarize_experiment.py --validate-report experiment-summary.json
python scripts/summarize_experiment.py --compare-report experiment-summary.json
```

Summary comparison reloads the saved `manifest_path`, re-reads the explicit
local artifact paths recorded by that manifest, and reports drift in source
artifact digests, research-question metrics, or summary counts.
Summary validation also rebuilds `research_questions` and `summary` from the
embedded QA and active-task delta comparisons, and rebuilds
`graph_construction_diagnostics` from embedded graph eval summaries, and
`error_attribution_diagnostics` from embedded attribution summaries, then
rebuilds `failure_linkage_diagnostics` from those diagnostic blocks, so metric
or count edits are rejected even if `report_digest` is recomputed after
tampering. It also checks that embedded QA, active delta, graph eval, and
attribution summary rows match the corresponding `source_artifacts` keys,
paths, and digests, and rebuilds `readiness` and verdict counts so missing-RQ,
ready/incomplete, or improved/regressed edits cannot be accepted by recomputing
the report digest.

## Experiment Record JSON

Schema version: `dsg-spatialqa-lab.experiment-record.v1`

Experiment records are compact final handoff ledgers derived from a saved
experiment summary report. They do not rerun evaluation. They record the summary
report path and digest, manifest path and digest, readiness status, RQ1-RQ4
verdict rows, per-measurement research-question matrix rows, verdict counts,
diagnostic ledger counts/keys for QA diagnostic slices, graph construction,
error attribution, and failure-linkage pairs, source artifact digests, and
optional dashboard bundle digest metadata.

Top-level fields include:

- `summary_report_path`
- `summary_report_digest`
- `manifest_path`
- `manifest_digest`
- `readiness_status`
- `readiness`
- `research_question_verdicts`
- `research_question_matrix`
- `verdict_counts`
- `diagnostic_ledger`
- `source_artifact_digests`
- optional `dashboard_bundle`
- `record_digest`

CLI:

```bash
python scripts/record_experiment.py \
  --summary-report experiment-summary.json \
  --record experiment-record.json
python scripts/record_experiment.py --validate-record experiment-record.json
python scripts/record_experiment.py --compare-record experiment-record.json
```

Record validation checks the saved record digest, readiness status consistency,
research-question verdict count, research-question matrix count, and
verdict-count consistency, plus diagnostic ledger count/key consistency.
Record comparison reloads only the explicit
summary/dashboard paths recorded in the record and reports drift in summary
digest, readiness, verdict counts, research-question verdict rows,
research-question matrix rows, diagnostic ledger rows, source artifact digests,
or dashboard metadata.

## Mock Experiment Result JSON

Schema version: `dsg-spatialqa-lab.mock-experiment-result.v1`

`scripts/run_mock_experiment.py` writes a deterministic local pipeline under an
explicit output directory. It defaults to one mock episode and can aggregate
multiple deterministic mock episodes with `--episode-count`. Repeated
`--qa-baseline` arguments compare the fixed `graph_tool` candidate against
multiple local QA agents. Its stdout payload is a path manifest for generated
artifacts: mock episodes, oracle graphs, per-episode and combined QA datasets,
predicted graph JSON files, predicted graph reports, graph eval reports,
oracle GraphTool, predicted GraphTool, and baseline prediction JSONL files, QA
reports, active-task reports, delta reports, benchmark manifest, experiment
summary, dashboard files, and final experiment record. The payload includes
singular compatibility path fields for the first artifact plus `episode_paths`,
`graph_paths`, `predicted_graph_paths`, `predicted_graph_report_paths`,
`graph_eval_report_paths`, `qa_paths`, `combined_qa_path`, `qa_candidate_name`,
`qa_baseline_names`, `qa_graph_construction_baseline_name`,
`qa_prediction_paths`, `qa_report_paths`, `qa_delta_report_paths`,
`predicted_graph_tool_prediction_path`, `predicted_graph_tool_report_path`,
`qa_graph_construction_delta_report_path`,
`active_predicted_graph_report_path`,
`active_graph_construction_delta_report_path`, `readiness_status`,
`verdict_counts`, and the final record digest for quick automation checks.

CLI:

```bash
python scripts/run_mock_experiment.py \
  --output-dir data/mock-experiment \
  --dataset-name mock_experiment \
  --max-qa-per-episode 3 \
  --episode-count 2 \
  --qa-baseline majority \
  --qa-baseline graph_text
```

## Digest Rules

- JSON output is sorted and stable.
- JSONL records are emitted in deterministic order.
- Report and manifest digests omit only their own digest field.
- Compare commands recompute current artifacts from the explicit saved paths.
- Drift is reported as structured JSON rather than as an unhandled exception.
