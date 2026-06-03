# Real Small Experiment

The real-small runner is a thin manifest entrypoint over the existing
DSG-SpatialQA artifact APIs. It does not call simulators, detectors, VLMs, LLMs,
or network services. It only consumes explicit local files.

```bash
python scripts/run_real_small_experiment.py \
  --manifest examples/real_small_experiment/real-small-run-manifest.template.json \
  --output-dir /tmp/dsg-real-small \
  --report /tmp/dsg-real-small/run-report.json
```

The template is expected to fail until real local artifacts are supplied. A
failed run writes `ready=false`, `research_ready=false`, `blockers`, and
`next_missing_artifacts`.

Required real inputs:

- episode JSONL from AI2-THOR or Habitat
- real collection report from `scripts/check_real_collection.py`
- offline-control import manifest with `vlm`, `multi_frame_vlm`,
  `caption_memory`, and `graph_text`
- predicted DSG detector-run manifest or ready predicted DSG report paths
- active-task delta, graph eval, error attribution, and dashboard bundle

Detector/RGB-D outputs can be imported with the existing observation CLI:

```bash
python scripts/observations.py \
  --import-detector-jsonl data/real-small/detector/external-detector-observations.jsonl \
  --output-sequence data/real-small/observations/detector-observation-sequence.json
```

Synthetic fixtures may exercise the mechanical gate only when their manifest
sets `data_source_kind="synthetic_test_fixture"` and
`not_real_research_result=true`. They must not be reported as DSG improvement
evidence.

