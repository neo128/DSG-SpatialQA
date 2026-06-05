# Real Small Experiment

The real-small runner is a thin manifest entrypoint over the existing
DSG-SpatialQA artifact APIs. It does not call simulators, detectors, VLMs,
LLMs, or network services. It only consumes explicit local files and either
assembles a ready package or returns structured blockers.

```bash
python scripts/run_real_small_experiment.py \
  --manifest examples/real_small_experiment/real-small-run-manifest.template.json \
  --output-dir /tmp/dsg-real-small \
  --report /tmp/dsg-real-small/run-report.json
```

The template is expected to fail until real local artifacts are supplied. A
failed run writes `ready=false`, `research_ready=false`, `blockers`,
`next_missing_artifacts`, and `final_record_written=false`.

## Directory Layout

Use one local package root for the first pilot:

```text
data/real-small/
  episodes/
  raw/
  detector/
  observations/
  graphs/
  qa/
  predictions/
  reports/
  dashboard/
  final/
```

Relative paths inside manifests are resolved from the manifest file location.
Keep generated outputs under the same package root so digests and compare
commands remain easy to audit.

## Required Inputs

A research-ready real-small package requires all of these local artifacts:

- real episode JSONL from AI2-THOR or Habitat
- real collection report from `scripts/check_real_collection.py`
- oracle benchmark manifest plus generated QA dataset
- GraphTool candidate prediction JSONL
- offline-control import manifest with all four source kinds: `vlm`,
  `multi_frame_vlm`, `caption_memory`, and `graph_text`
- four external offline prediction JSONL files
- detector/RGB-D JSONL or a detector-run manifest that builds an
  observation-sequence-backed predicted DSG
- predicted DSG evidence report requiring `depth`, `detector`, and `rgb`
  evidence kinds
- QA eval and candidate-vs-control delta reports
- graph eval report
- error attribution report
- dashboard bundle
- active-task delta report, which may be mock only if the artifact says so
- final readiness report with `ready=true`

Without all four controls, do not report DSG-vs-control lift. Without
observation-sequence-backed predicted DSG evidence, do not report a real
predicted DSG result.

## External Producers

Simulator producers provide:

- episode JSONL
- raw RGB/depth/segmentation files referenced by each frame
- frame metadata with `adapter="ai2thor"` or `adapter="habitat"`,
  `source_kind="real_simulator"`, `simulator`, and `collection_kind="real"`

Perception producers provide detector/RGB-D JSONL. The supported external frame
schema is shown in
`examples/real_small_experiment/external-detector-observations.example.jsonl`.
Each detection should include stable object or track IDs, confidence, 2D/3D
box data, RGB/depth/mask paths when available, `detector_name`, and
`evidence_kinds` containing `rgb`, `depth`, and `detector`. Location and state
evidence are expected under each detection's `attributes`: use
`current_location_id` plus `current_location_relation` (`IN_REGION`, `IN_ROOM`,
`INSIDE`, or `ON`) for queryable object placement, and use `states` for
per-step dynamic evidence such as `isOpen`, `isDirty`, `isToggled`, or
`isFilledWithLiquid`. If the detector uses its own track ID instead of the
handoff `object_id`, return one of `coverage_target_object_id`,
`target_object_id`, or `collection_target_object_id` in `attributes`; acceptance
reports will match it back to the requested target while still requiring visible
detector/RGB-D evidence. For visible detector observations with `rgb`, `depth`,
and `detector` evidence, the graph builder can create missing `IN_REGION` or
`IN_ROOM` destination nodes from `current_location_id` and optional
`current_location_label`. `ON` and `INSIDE` still require the destination object
or container to be present in the observation sequence. Valid detector-backed
`states` are copied into the `state:<object>:<step>` node and the
`STATE_CHANGED` edge, with local RGB/depth/mask paths preserved as evidence.
The coverage acceptance gates require
`source_kind="detector"` after import; simulator metadata or hidden planning
metadata cannot satisfy predicted DSG evidence.

VLM/LLM producers provide only prediction JSONL files. The request bundle must
include case ID, scene/episode IDs, question text, answer type, choices when
applicable, source metadata, and an output schema template. It must not include
gold answers, gold evidence nodes, gold evidence edges, or hidden evaluator
fields.

For VLM-only, producers should treat `question_text` and `target.label` as the
primary visual query. The structured `question` object is retained for stable
ids, but opaque simulator ids are not enough for a visual model to identify the
target. If the target is not visible in the single RGB frame, the prediction
should return a structured uncertainty error such as `target_not_observed`
instead of guessing a hidden object location. The request bundle may provide
`primary_frame` and per-episode `frames` with local RGB/depth/segmentation paths
and digests; VLM-only should use the primary RGB frame, while multi-frame VLM
may inspect the frame list. The bundle must not expose simulator
`visible_object_ids` or `visible_object_labels` as VLM-only evidence.

## Command Sequence

Collect or import real episodes:

```bash
python scripts/collect_ai2thor.py \
  --scene FloorPlan1 \
  --episode-id ai2thor_real_smoke_001 \
  --step 1 --step 2 --step 3 \
  --action Initialize --action MoveAhead --action RotateRight \
  --artifact-root data/real-small/raw/ai2thor_real_smoke_001 \
  --output data/real-small/episodes/ai2thor_real_smoke_001.jsonl
```

For AI2-THOR real collection, each frame must save RGB, depth, instance
segmentation, and the segmentation color map in frame metadata. The color map
links visible segmentation colors to simulator object IDs. It is required so a
later visible segmentation/RGB-D detector importer can prove that object
evidence came from local frame artifacts instead of hidden metadata.

If the real episode contains `metadata.segmentation_color_map`, convert local
RGB/depth/segmentation artifacts into detector-source JSONL before importing the
observation sequence:

```bash
python scripts/observations.py \
  --episode-segmentation-detector-jsonl data/real-small/episodes/ai2thor_real_smoke_001.jsonl \
  --output-detector-jsonl data/real-small/detector/visible-segmentation-rgbd.jsonl \
  --mask-root data/real-small/detector/masks \
  --detector-name ai2thor_visible_segmentation_rgbd
```

This path sets `source_kind="detector"` only for objects whose pixels appear in
the local segmentation artifact. It adds RGB/depth/detector evidence kinds,
image-space masks, a deterministic depth-ray geometry estimate, and an explicit
`IN_REGION` current-location edge to a visible frame region. Depth artifacts may
be JSON matrices or `.npy` arrays written by the AI2-THOR collector; `.npy`
reading uses the standard library and does not add a default runtime dependency.

Check real collection evidence:

```bash
python scripts/check_real_collection.py \
  --episode data/real-small/episodes/ai2thor_real_smoke_001.jsonl \
  --report data/real-small/reports/real-collection-report.json \
  --min-frame-count 3 \
  --required-adapter ai2thor
```

Build oracle graph and QA artifacts:

```bash
python scripts/build_oracle_graph.py \
  --input data/real-small/episodes/ai2thor_real_smoke_001.jsonl \
  --output-graph data/real-small/graphs/oracle-graph.json \
  --report data/real-small/reports/oracle-graph-report.json
python scripts/generate_qa.py \
  --graph data/real-small/graphs/oracle-graph.json \
  --scene-id FloorPlan1 \
  --episode-id ai2thor_real_smoke_001 \
  --max-cases 50 \
  --output data/real-small/qa/qa.jsonl
```

Run the local GraphTool candidate and generate no-gold external control
requests:

```bash
python scripts/run_baselines.py \
  --baseline graph_tool \
  --graph data/real-small/graphs/oracle-graph.json \
  --qa data/real-small/qa/qa.jsonl \
  --pred data/real-small/predictions/graph-tool-predictions.jsonl
python scripts/run_offline_controls.py \
  --prediction-request-bundle data/real-small/offline-control-import-manifest.json \
  --request-bundle-output data/real-small/reports/offline-control-request-bundle.json
```

After the four external prediction files arrive, import and evaluate controls:

```bash
python scripts/run_offline_controls.py \
  --manifest data/real-small/offline-control-import-manifest.json
```

For an explicit VLM-only rerun through an OpenAI-compatible endpoint, use the
separate project-scoped key variable. Do not rely on or modify a system
`DASHSCOPE_API_KEY`:

```bash
DSG_SPATIALQA_DASHSCOPE_API_KEY=... \
python external_tools/run_vlm_controls.py \
  --request-bundle data/real-small/offline-control-prediction-request-bundle.json \
  --source-kind vlm \
  --model qwen3.7-plus \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --allow-network \
  --output data/real-small/predictions/vlm.jsonl \
  --trace-output data/real-small/reports/vlm-interactions.jsonl
```

Without `--allow-network`, the command returns a structured
`network_not_allowed` error and writes no predictions.

Import detector/RGB-D observations and build predicted DSG evidence:

```bash
python scripts/observations.py \
  --import-detector-jsonl data/real-small/detector/external-detector-observations.jsonl \
  --output-sequence data/real-small/observations/detector-observation-sequence.json \
  --report data/real-small/reports/detector-observation-import-report.json
```

If coverage-driven collection returns an additional detector/RGB-D sequence,
merge it with the base observation sequence before rebuilding the predicted DSG:

```bash
python scripts/observations.py \
  --merge-sequence data/real-small/observations/detector-observation-sequence.json \
  --merge-sequence data/real-small/observations/coverage-return-observation-sequence.json \
  --output-sequence data/real-small/observations/detector-observation-sequence-merged.json \
  --report data/real-small/reports/detector-observation-merge-report.json
```

```bash
python scripts/run_predicted_dsg.py \
  --manifest data/real-small/predicted-dsg-detector-run-manifest.json
```

Run graph evaluation, attribution, dashboard export, assembly, readiness, and
final recording through the manifest runner:

```bash
python scripts/run_real_small_experiment.py \
  --manifest data/real-small/real-small-run-manifest.json \
  --output-dir data/real-small \
  --report data/real-small/final/run-report.json
```

If readiness is false, the runner does not write a final claim record. Use
`blockers` and `next_missing_artifacts` as the next handoff checklist.

For DSG-vs-control conclusions, first generate a QA observability report
against the predicted DSG, then pass it into the conclusion layer:

```bash
python scripts/analyze_qa_observability.py \
  --qa data/real-small/qa/qa.jsonl \
  --graph data/real-small/graphs/predicted-graph.json \
  --report data/real-small/reports/qa-observability.json \
  --evidence-observable-qa data/real-small/qa/evidence-observable-qa.jsonl \
  --target-observable-relation-missing-qa \
    data/real-small/qa/target-observable-relation-missing-qa.jsonl
python scripts/conclude_experiment.py \
  --real-readiness-report data/real-small/reports/real-experiment-readiness.json \
  --offline-control-result-report data/real-small/reports/offline-control-result.json \
  --predicted-dsg-evidence-report data/real-small/reports/predicted-dsg-evidence.json \
  --graph-eval-report data/real-small/reports/graph-eval-report.json \
  --error-attribution-report data/real-small/reports/error-attribution-report.json \
  --qa-observability-report data/real-small/reports/qa-observability.json \
  --evaluation-scope full_oracle \
  --report data/real-small/final/research-conclusion.json
```

Use `--evaluation-scope observation_aware` only with QA eval and control delta
artifacts generated from the evidence-observable QA slice. The conclusion gate
rejects observation-aware claims when the slice is too small or when the QA eval
case count does not match the evidence-observable case count. It also checks
that every candidate/control QA eval report has the same `gold_digest` as
`qa-observability.json` records under
`split_qa_digests.evidence_observable`, so a full-oracle or unrelated QA
dataset cannot be reused as an observation-aware result. The candidate must
meet both the configured exact-match rate floor and the absolute exact-match
count floor, which defaults to `--min-candidate-exact-match-count 15`.

## Mock, Synthetic, And Real

Mock artifacts are deterministic demos and must not pass real collection
readiness. Synthetic fixtures may exercise the mechanical package gate only
when their manifest sets `data_source_kind="synthetic_test_fixture"` and
`not_real_research_result=true`.

`data_source_kind="real"` requires valid real collection reports whose frames
come from `source_kind="real_simulator"` and have local RGB/depth/segmentation
evidence. Only a real package with readiness `ready=true`, all four controls,
candidate-vs-control delta reports, graph eval, attribution, dashboard, and
predicted DSG evidence may support a DSG-vs-control research conclusion.
