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
`evidence_kinds` containing `rgb`, `depth`, and `detector`.

VLM/LLM producers provide only prediction JSONL files. The request bundle must
include case ID, scene/episode IDs, question text, answer type, choices when
applicable, source metadata, and an output schema template. It must not include
gold answers, gold evidence nodes, gold evidence edges, or hidden evaluator
fields.

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

Import detector/RGB-D observations and build predicted DSG evidence:

```bash
python scripts/observations.py \
  --import-detector-jsonl data/real-small/detector/external-detector-observations.jsonl \
  --output-sequence data/real-small/observations/detector-observation-sequence.json \
  --report data/real-small/reports/detector-observation-import-report.json
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
