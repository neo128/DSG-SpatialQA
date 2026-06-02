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

The import report records source name/kind/metadata, QA/input/prediction paths,
artifact digests, imported/unknown/missing/duplicate counts, missing case IDs,
unknown case IDs, and per-record import diagnostics.

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
- breakdowns by question type, tag, and reference frame,
- report digest.

CLI:

```bash
python scripts/run_qa_eval.py \
  --gold qa.jsonl \
  --pred predictions.jsonl \
  --report qa-eval-report.json
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
edges present in the predicted graph. The summary includes
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
rows, predicted-evidence source summaries, evidence subgraphs, frame paths when
present, graph summary, optional active task review panels, and bundle digest.
The HTML table exposes an Evidence Source column for per-case source-level
review. The `active_task_review` section records the source report digest,
summary, metrics, budget analysis, per-task case result, transcript, action
evidence snapshots, required evidence IDs, observed evidence IDs, and missing
evidence IDs.

CLI:

```bash
python scripts/export_dashboard.py \
  --qa qa.jsonl \
  --pred predictions.jsonl \
  --eval-report qa-eval-report.json \
  --graph oracle-graph.json \
  --error-attribution error-attribution.json \
  --active-task-report active-report.json \
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
```

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
- `manifest_digest`

Coverage includes:

- by question type,
- by scene,
- by episode,
- by reference frame,
- by tag,
- dynamic/static split,
- oracle/predicted source split.

CLI:

```bash
python scripts/build_benchmark.py \
  --episodes mock-ai2thor.jsonl \
  --episodes mock-habitat.jsonl \
  --dataset-name mock_benchmark \
  --output-dir data/benchmark \
  --max-qa-per-episode 100 \
  --manifest benchmark-manifest.json
python scripts/build_benchmark.py --validate-manifest benchmark-manifest.json
python scripts/build_benchmark.py --compare-manifest benchmark-manifest.json
```

## Digest Rules

- JSON output is sorted and stable.
- JSONL records are emitted in deterministic order.
- Report and manifest digests omit only their own digest field.
- Compare commands recompute current artifacts from the explicit saved paths.
- Drift is reported as structured JSON rather than as an unhandled exception.
