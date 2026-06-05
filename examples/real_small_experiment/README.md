# Real Small Experiment Template

This directory contains handoff templates for the first real-small
DSG-vs-control experiment package. The files are templates only. They do not
contain real episode data, real VLM/LLM predictions, detector outputs, or
research results.

Run the template as a failure smoke:

```bash
python scripts/run_real_small_experiment.py \
  --manifest examples/real_small_experiment/real-small-run-manifest.template.json \
  --output-dir /tmp/dsg-real-small \
  --report /tmp/dsg-real-small/run-report.json
```

Expected result: non-zero exit, `ready=false`, `research_ready=false`,
`final_record_written=false`, and a structured `next_missing_artifacts` list.

External producers fill the template paths under `data/real-small/`:

- simulator collection writes episode JSONL plus referenced RGB/depth/
  segmentation files
- `scripts/check_real_collection.py` writes the real collection report
- VLM/LLM producers return the four prediction JSONL files without receiving
  gold answers or gold evidence
- perception producers return detector/RGB-D JSONL matching
  `external-detector-observations.example.jsonl`
  with `evidence_kinds=["rgb","depth","detector"]`,
  `attributes.coverage_target_object_id` or another documented handoff target
  alias when the detector object ID is an external track ID,
  `attributes.current_location_id`,
  `attributes.current_location_relation`, and `attributes.states` whenever the
  handoff task requests location or dynamic-state evidence
- `IN_REGION` and `IN_ROOM` current-location destinations may be created by the
  graph builder from strict visible detector evidence; `ON` and `INSIDE`
  destinations should be returned as observed support/container objects
- local DSG-SpatialQA CLIs import controls, build predicted DSG evidence, run
  evals, export dashboard artifacts, and run readiness

Synthetic smoke fixtures must use `data_source_kind="synthetic_test_fixture"`
and `not_real_research_result=true`. Mock or synthetic outputs must not be
claimed as real DSG-vs-control evidence.
