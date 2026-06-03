# Real Small Experiment Template

This directory contains handoff templates for the first real-small DSG-vs-control
experiment package. The files are templates only. They do not contain real
episode data, real VLM/LLM predictions, detector outputs, or research results.

Run the template as a failure smoke:

```bash
python scripts/run_real_small_experiment.py \
  --manifest examples/real_small_experiment/real-small-run-manifest.template.json \
  --output-dir /tmp/dsg-real-small \
  --report /tmp/dsg-real-small/run-report.json
```

Expected result: non-zero exit, `ready=false`, and a structured
`next_missing_artifacts` list.

