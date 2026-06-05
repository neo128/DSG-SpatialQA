# Real-Small Experiment Status

This page records the current handoff state for the manifest-driven
real-small experiment path. It is intentionally narrow: it describes which
commands work today, which artifacts are still needed for a new real package,
and why mock or synthetic artifacts cannot be used as real research evidence.

## Working Commands

The template manifest is a failure smoke test. It does not include real
artifacts and should return a non-zero exit code:

```bash
python scripts/run_real_small_experiment.py \
  --manifest examples/real_small_experiment/real-small-run-manifest.template.json \
  --output-dir /tmp/dsg-real-small \
  --report /tmp/dsg-real-small/run-report.json
```

Current observed behavior:

- `ready=false`
- `research_ready=false`
- `final_record_written=false`
- `blockers` includes `required_artifacts_present`
- `next_missing_artifacts` lists 8 missing local artifact paths

The current real-small regression tests exercise missing manifests, invalid
schemas, missing controls, QA digest mismatch, predicted DSG evidence failure,
synthetic mechanical pass, synthetic mislabeled-as-real failure, mock real
collection rejection, and no-gold request bundles:

```bash
python -m pytest \
  tests/test_real_small_experiment_package.py \
  tests/test_ai2thor_adapter.py \
  tests/test_observations.py \
  tests/test_offline_control_manifest_run.py \
  tests/test_predicted_dsg_evidence.py \
  -q
```

The full repository gate remains:

```bash
python scripts/verify.py
```

## Current Template Blockers

The example template is expected to fail until these local files exist under
`examples/real_small_experiment/data/real-small/` or equivalent manifest paths:

- `episodes/ai2thor_real_smoke_001.jsonl`
- `offline-control-import-manifest.json`
- `predicted-dsg-detector-run-manifest.json`
- `reports/real-collection-report.json`
- `reports/active-task-delta.json`
- `dashboard/dashboard.json`
- `reports/error-attribution-report.json`
- `reports/graph-eval-report.json`

The runner writes these blockers into `/tmp/dsg-real-small/run-report.json`
instead of throwing an unstructured stack trace.

## External Producer Inputs

Simulator producers must provide:

- real AI2-THOR or Habitat episode JSONL
- local RGB, depth, and segmentation artifacts referenced by episode frames
- frame metadata with `adapter="ai2thor"` or `adapter="habitat"`
- frame metadata with `source_kind="real_simulator"`
- frame metadata with `collection_kind="real"`

Perception producers must provide:

- detector/RGB-D JSONL accepted by
  `scripts/observations.py --import-detector-jsonl`
- stable object or track IDs
- detector source name
- confidence values
- RGB/depth/mask paths when available
- object attributes preserving `source_kind="detector"`,
  `source_name`, `evidence_kinds`, `rgb_path`, `depth_path`, and `mask_path`

VLM/LLM producers must provide four local offline prediction JSONL files:

- `vlm`
- `multi_frame_vlm`
- `caption_memory`
- `graph_text`

The external request bundle must not contain gold answers, gold evidence nodes,
gold evidence edges, or hidden evaluator-only fields.

The current offline-control request bundle now includes VLM-readable
`question_text`, non-gold `target` labels, `answer_schema_hint`, `primary_frame`,
and per-episode `frames` fields. The saved handoff bundle is:

- `handoffs/ai2thor-real-small/offline-control-prediction-request-bundle.json`
- digest: `b98160f3edb26b4504595fb2b1dd9d78373c010bea08a63d684acc84c314d16c`
- frame index: 50 local RGB/depth/segmentation frame refs from
  `handoffs/ai2thor-real-small/inputs/traces/frame-index.jsonl`

VLM-only producers should use `question_text` and `target.label` as the visual
query. If a target is not visible in the single RGB frame, the expected behavior
is a structured uncertainty error such as `target_not_observed`, not a guessed
hidden object location. The bundle intentionally omits `visible_object_ids` and
`visible_object_labels` so VLM-only results are not boosted by detector or
simulator visibility metadata.

The project now has an explicit optional VLM runner for this handoff:

```bash
DSG_SPATIALQA_DASHSCOPE_API_KEY=... \
python external_tools/run_vlm_controls.py \
  --request-bundle handoffs/ai2thor-real-small/offline-control-prediction-request-bundle.json \
  --source-kind vlm \
  --model qwen3.7-plus \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --allow-network \
  --output handoffs/ai2thor-real-small/inputs/offline-controls/vlm.jsonl \
  --trace-output handoffs/ai2thor-real-small/inputs/traces/vlm-interactions-rerun.jsonl
```

The runner refuses to call the network unless `--allow-network` is present and a
project-scoped `DSG_SPATIALQA_DASHSCOPE_API_KEY` is set. It does not modify or
print the system `DASHSCOPE_API_KEY`.

## Mock, Synthetic, And Real

Mock artifacts are deterministic demos. They must not pass real collection
readiness.

Synthetic fixtures may pass only mechanical smoke gates. They must use:

- `data_source_kind="synthetic_test_fixture"`
- `not_real_research_result=true`

The current synthetic mechanical fixture is generated inside
`tests/test_real_small_experiment_package.py`; it is not a real benchmark
package and cannot be labeled as a real research result. If the same synthetic
fixture is mislabeled as `data_source_kind="real"`, the runner rejects it.

Real packages must use `data_source_kind="real"` and must include valid real
collection reports whose frames come from `source_kind="real_simulator"`.
They also require all four controls, observation-sequence-backed predicted DSG
evidence with `rgb`, `depth`, and `detector`, QA eval and delta reports, graph
eval, error attribution, dashboard, readiness, and final record artifacts.

## Research Claim Status

Do not claim “DSG improves over VLM/video memory” from a template failure, a
mock run, or a synthetic fixture.

The current `handoffs/ai2thor-real-small` package has a complete experiment
artifact trail, but the latest predicted DSG evidence gate no longer treats its
`ai2thor` metadata-backed observation sequence as real detector/RGB-D evidence.
Its formal full-oracle conclusion report remains useful as a failure diagnosis:

- conclusion report:
  `handoffs/ai2thor-real-small/outputs/research-conclusion.json`
- verdict: `dsg_not_superior`
- candidate exact match: `1/60`
- required controls passing superiority gates: `0/4`
- predicted graph object recall: `0.159091`
- predicted graph unlocated objects: `0`
- predicted graph relation F1: `0.047650`
- evaluation scope: `full_oracle`
- evidence-observable QA: `3/60`

Current predicted DSG evidence readiness is now `ready=false`:

- evidence report:
  `handoffs/ai2thor-real-small/outputs/predicted-dsg/predicted-dsg-evidence.json`
- failed check: `non_real_sources_absent`
- evidence source counts: `{"ai2thor": 123}`
- evidence report digest:
  `56d9b7168e27b58b3f09b9fd29a6d84356d9f179ac415acab230dc42ddd4ae0d`

This is intentional. AI2-THOR metadata can guide coverage planning, but it must
not be counted as external detector evidence for a detector-only predicted DSG
claim.

An observation-aware conclusion attempt is also saved at
`handoffs/ai2thor-real-small/outputs/diagnostics/research-conclusion-observation-aware.json`.
It currently returns `inconclusive_not_ready` because only `3/60` QA cases are
evidence-observable, below the configured floor of `30`.

A diagnostic coverage run is also saved under the `coverage-v1` artifact names.
It is useful for root-cause analysis, but it is not a final detector-only
research result because it converts saved AI2-THOR episode metadata, including
hidden objects, into detector-style observations:

- coverage observability:
  `handoffs/ai2thor-real-small/outputs/diagnostics/qa-observability-coverage-v1.json`
- evidence-observable QA: `49/60`
- object recall: `1.0`
- evidence-observable exact match: `22/49`
- missing state cases: `0`
- relation precision: `0.056883`

This diagnostic run shows that DSG improves substantially when object coverage
and state timelines are available. It also shows the next bottleneck: relation
over-generation. Because hidden metadata is used, it must be treated as a
coverage/state-timeline diagnosis, not as final evidence that a detector-only
DSG beats VLM or video memory controls.

The current visible-only detector manifest now enables containment and a
per-object NEAR cap:

- manifest:
  `handoffs/ai2thor-real-small/predicted-dsg-detector-run-manifest.json`
- `infer_containment=true`
- `containment_axis="z"`
- `relation_top_k=3`

After rerunning that manifest and graph eval, the current predicted graph has
`unlocated_object_count=0`, and the conclusion gate now requires that count to
remain zero before any DSG-superiority claim is allowed. QA target coverage
remains too low, and the evidence gate rejects the current source as non-real
detector evidence:
`evidence_observable=3/60` and target node recall remains `0.161616`.
The latest ON semantic filter keeps only one best support surface per object,
reducing visible-only ON edges from `8` to `4` and raising current graph
relation precision to `0.395738`; this does not change the object coverage
blocker.
This confirms that coverage-driven visible RGB-D collection is the next real
blocker.

A coverage-driven collection plan now exists:

- `handoffs/ai2thor-real-small/outputs/diagnostics/coverage-collection-plan-p1.json`
- current evidence-observable QA: `3/60`
- missing target nodes: `83`
- planned collection targets: `83`
- unresolved targets: `0`
- relation evidence targets: `11`

The plan explicitly marks episode metadata as collection-planning-only:

- `episode_metadata_used_for="collection_planning_only"`
- `not_predicted_graph_evidence=true`
- `requires_visible_detector_evidence_before_claim=true`

It can be regenerated with:

```bash
python scripts/plan_coverage_collection.py \
  --qa-observability-report handoffs/ai2thor-real-small/outputs/diagnostics/qa-observability-p1.json \
  --episode handoffs/ai2thor-real-small/inputs/episodes/ai2thor-real-small-episode-001.jsonl \
  --episode handoffs/ai2thor-real-small/inputs/episodes/ai2thor-real-small-episode-002.jsonl \
  --episode handoffs/ai2thor-real-small/inputs/episodes/ai2thor-real-small-episode-003.jsonl \
  --episode handoffs/ai2thor-real-small/inputs/episodes/ai2thor-real-small-episode-004.jsonl \
  --episode handoffs/ai2thor-real-small/inputs/episodes/ai2thor-real-small-episode-005.jsonl \
  --output handoffs/ai2thor-real-small/outputs/diagnostics/coverage-collection-plan-p1.json \
  --target-evidence-observable-count 30 \
  --target-node-recall-floor 0.5
```

The plan has also been materialized as a producer-facing coverage collection
request bundle:

- `handoffs/ai2thor-real-small/inputs/predicted-dsg/coverage-collection-request-p1.json`
- target count: `83`
- relation evidence target count: `11`
- state evidence target count: `45`
- request bundle digest:
  `831ac9f0a402890ea76bd8cef2848aa0a57d6cfb0e5ceb502b0d9934aaac8af5`
- each target includes up to 3 deterministic `viewpoint_hints` with the nearest
  episode-frame agent pose, distance to target, and suggested yaw toward the
  object
- planned detector JSONL return path:
  `handoffs/ai2thor-real-small/inputs/predicted-dsg/detector-rgbd-coverage-targets-p1.jsonl`
- planned observation sequence path:
  `handoffs/ai2thor-real-small/outputs/predicted-dsg/detector-observations-coverage-targets-p1.json`

It can be regenerated with:

```bash
python scripts/plan_coverage_collection.py \
  --request-plan handoffs/ai2thor-real-small/outputs/diagnostics/coverage-collection-plan-p1.json \
  --request-bundle handoffs/ai2thor-real-small/inputs/predicted-dsg/coverage-collection-request-p1.json \
  --detector-jsonl-output handoffs/ai2thor-real-small/inputs/predicted-dsg/detector-rgbd-coverage-targets-p1.jsonl \
  --observation-sequence-output handoffs/ai2thor-real-small/outputs/predicted-dsg/detector-observations-coverage-targets-p1.json \
  --acceptance-report-output handoffs/ai2thor-real-small/outputs/diagnostics/coverage-collection-acceptance-p1.json
```

The request bundle contains no `gold_answer` or `gold_evidence` fields. It is
not predicted DSG evidence; it is only a collection/detector handoff input.
The viewpoint hints are planning metadata only; returned visible RGB-D/detector
records must still pass the acceptance gate before rebuilding the predicted DSG.
State evidence targets ask the producer to return per-step object state evidence
under `attributes.states`; the request does not include gold state values.
Each target contract also asks the producer to return
`attributes.current_location_id` and `attributes.current_location_relation`
with one of `IN_REGION`, `IN_ROOM`, `INSIDE`, or `ON`, so visible detections can
become queryable location edges in the predicted DSG.
The request bundle also contains `viewpoint_batches`, which groups each target's
primary viewpoint by episode, scene, and step. The current P1 request groups 83
targets into 13 primary batches. Each batch now includes unique
`related_case_ids`, `related_case_count`, `priority_rank`,
`cumulative_target_count`, and `cumulative_related_case_count`. The current
target case gap is 27, and the first 5 priority batches cumulatively cover 31
related QA cases. The highest-yield batch covers 33 targets and 7 related QA
cases from one agent pose. These fields are collection priorities only; returned
RGB-D/detector records must still pass the visible-evidence acceptance gate
before they can support predicted DSG results.
Each `viewpoint_batch` now also carries an `execution_plan` with
`TeleportFull`, `RotateToTargetYaw`, and `CaptureVisibleRgbdDetection` actions,
plus the top-batch return report command hint. The execution plan is collection
metadata only and is marked `not_predicted_graph_evidence=true`.

The first 5 priority batches have also been exported as a line-oriented target
task handoff for external collection/perception producers:

- `handoffs/ai2thor-real-small/inputs/predicted-dsg/coverage-collection-top-batches-p1.jsonl`
- task count: `54`
- batch count: `5`
- related QA case count: `31`
- digest: `26b949cd2843f8ad98cdc24355e0c404210690058fc5a499e04375cdaee1aa10`

It can be regenerated and validated with:

```bash
python scripts/plan_coverage_collection.py \
  --top-batch-handoff-request-bundle handoffs/ai2thor-real-small/inputs/predicted-dsg/coverage-collection-request-p1.json \
  --top-batch-handoff-jsonl handoffs/ai2thor-real-small/inputs/predicted-dsg/coverage-collection-top-batches-p1.jsonl \
  --max-priority-batches 5

python scripts/plan_coverage_collection.py \
  --validate-top-batch-handoff handoffs/ai2thor-real-small/inputs/predicted-dsg/coverage-collection-top-batches-p1.jsonl
```

The handoff JSONL is not predicted DSG evidence. It contains no `gold_answer` or
`gold_evidence` fields. Every task line now includes `batch_execution_plan`, so
a producer that receives only the JSONL still has the batch-level simulator
action plan and acceptance command hint. A direct alignment check shows the existing
`detector-rgbd.jsonl` currently matches `0/54` of these top-priority targets, so
the next success-rate improvement requires returned visible detector/RGB-D
records for this handoff rather than more graph-rule tuning alone.

A top-batch return report now checks whether a returned detector/RGB-D file
actually satisfies those 54 handoff tasks:

- `handoffs/ai2thor-real-small/outputs/diagnostics/coverage-collection-top-batch-return-p1.json`
- accepted target tasks: `0/54`
- accepted location tasks: `0/54`
- accepted state tasks: `0/26`
- related QA case count: `31`
- return ready: `false`
- digest: `aa2577f7f4c0e57b5da5c44c8f2b5f0a5ee1dbd146f74be67bdee5aff6cf6e14`

It can be regenerated and validated with:

```bash
python scripts/plan_coverage_collection.py \
  --top-batch-return-tasks handoffs/ai2thor-real-small/inputs/predicted-dsg/coverage-collection-top-batches-p1.jsonl \
  --detector-jsonl handoffs/ai2thor-real-small/inputs/predicted-dsg/detector-rgbd.jsonl \
  --top-batch-return-report handoffs/ai2thor-real-small/outputs/diagnostics/coverage-collection-top-batch-return-p1.json

python scripts/plan_coverage_collection.py \
  --validate-top-batch-return-report handoffs/ai2thor-real-small/outputs/diagnostics/coverage-collection-top-batch-return-p1.json
```

This report is still a return-quality/gap report, not predicted DSG evidence. It
is meant to be rerun after the producer returns
`detector-rgbd-coverage-targets-p1.jsonl`; only accepted visible target,
current-location, and state evidence should flow into the predicted DSG rebuild.

The corresponding acceptance report checks whether detector/RGB-D observations
actually cover those planned targets and state evidence targets with visible
evidence:

- `handoffs/ai2thor-real-small/outputs/diagnostics/coverage-collection-acceptance-p1.json`
- accepted visible detector targets: `0/83`
- unaccepted targets: `83`
- target acceptance rate: `0.0`
- target evidence ready: `false`
- accepted location evidence targets: `0/83`
- unaccepted location evidence targets: `83`
- location evidence acceptance rate: `0.0`
- location evidence ready: `false`
- accepted state evidence targets: `0/45`
- unaccepted state evidence targets: `45`
- state evidence acceptance rate: `0.0`
- state evidence ready: `false`

It can be regenerated with:

```bash
python scripts/plan_coverage_collection.py \
  --acceptance-plan handoffs/ai2thor-real-small/outputs/diagnostics/coverage-collection-plan-p1.json \
  --detector-jsonl handoffs/ai2thor-real-small/inputs/predicted-dsg/detector-rgbd.jsonl \
  --acceptance-report handoffs/ai2thor-real-small/outputs/diagnostics/coverage-collection-acceptance-p1.json
```

The acceptance gate rejects hidden metadata coverage and simulator metadata
sources. Planned targets count only after they appear in visible detector/RGB-D
observations with `source_kind="detector"` and `rgb`, `depth`, and `detector`
evidence. Location evidence targets count only after the accepted visible
detector observation also returns `attributes.current_location_id` and
`attributes.current_location_relation`. State evidence targets count only after
the accepted visible detector observation also returns per-step object state
evidence under `attributes.states`.

Returned detector files may use external track IDs. The acceptance gate first
matches the planned `object_id`, then a visible detector observation may match
through an explicit `attributes.ai2thor_object_id` alias or through handoff
target aliases: `attributes.coverage_target_object_id`,
`attributes.target_object_id`, or `attributes.collection_target_object_id`.
Accepted targets record `matched_by` and `observed_object_ids`, so alias
matches remain auditable. Alias matching does not bypass the visible
RGB-D/detector evidence requirements.

A future package may claim DSG superiority only when readiness is true, all
four controls are present, candidate-vs-control delta reports are available,
and the conclusion report returns `dsg_superior` with
`dsg_superiority_claim_allowed=true`.

## Next Development Focus

The next implementation path is方案 A + 方案 B:

- Improve predicted DSG quality: raise QA target object coverage, add stable
  `IN_ROOM`, `IN_REGION`, `ON`, current-location, and state-timeline evidence,
  and reduce relation over-generation.
- Run formal split comparisons: keep the full-oracle QA conclusion, but also
  produce separate QA evals, control deltas, and conclusion reports for the
  evidence-observable QA slice.
- Treat observation-aware superiority as invalid unless the QA eval case count
  matches the evidence-observable slice count, every candidate/control
  `gold_digest` matches `qa_observability.split_qa_digests.evidence_observable`,
  the slice has enough cases, and the candidate reaches the absolute
  exact-match floor, currently `15`.

Current visible-only observation-aware smoke reports are available under
`handoffs/ai2thor-real-small/outputs/offline-controls/qa-eval-observation-aware-p1/`.
They use the 3-case `evidence-observable-p1-qa.jsonl` slice and compare
`predicted_graph_tool` against all four controls with matching case counts.
The candidate is `1/3`; every control is `0/3`. This is useful for checking the
formal split-comparison path, but it is not enough for a research claim because
the slice is below the 30-case observation-aware floor.

## VLM-only Rerun Hardening

The current checked-in VLM-only predictions are still the old control output and
remain `0/60`. They were not overwritten with synthetic or gold-assisted
answers.

The rerun path is now stricter. `external_tools/run_vlm_controls.py` sends an
OpenAI-compatible `response_format={"type":"json_object"}` request, uses a
system prompt that requires evaluator-ready JSON fields, and records per-case
trace diagnostics. If a model returns a free-text answer such as
`{"text":"on the countertop"}` without a structured object-location answer, the
runner marks the prediction with `answer_schema_mismatch` instead of letting it
look like a valid structured prediction.

This improves the next real VLM-only run without changing the current score. A
real rerun still requires the project-scoped
`DSG_SPATIALQA_DASHSCOPE_API_KEY`; the runner must be invoked with
`--allow-network`, and the trace must be kept with the prediction file.

A five-case smoke attempt was made by mapping the existing system
`DASHSCOPE_API_KEY` into the process-local project key for that command only.
It returned `HTTP Error 401: Unauthorized`, wrote one failure trace to
`handoffs/ai2thor-real-small/inputs/traces/reruns/vlm-qwen37-plus-structured-limit5-trace.jsonl`,
and wrote no prediction file. The trace contains the failed case id, image
references, and `external_call_failed` diagnostics, and it does not contain an
API key or bearer token. This is an authentication blocker for rerunning the
control, not evidence about VLM spatial QA ability.

## VLM-only Observable Semantic Calibration

A VLM-only calibration pass now separates single-frame observable QA from hidden
or memory-heavy QA. It does not overwrite the original `vlm.jsonl` and does not
rewrite natural-language VLM answers into oracle object ids.

Command:

```bash
python scripts/eval_vlm_calibration.py \
  --qa handoffs/ai2thor-real-small/inputs/qa.jsonl \
  --predictions handoffs/ai2thor-real-small/inputs/offline-controls/vlm.jsonl \
  --request-bundle handoffs/ai2thor-real-small/offline-control-prediction-request-bundle.json \
  --observable-qa-output handoffs/ai2thor-real-small/inputs/traces/vlm-observable-qa-v1.jsonl \
  --observable-request-bundle-output handoffs/ai2thor-real-small/inputs/traces/vlm-observable-request-bundle-v1.json \
  --observable-slice-report handoffs/ai2thor-real-small/outputs/diagnostics/vlm-observable-slice-v1.json \
  --semantic-eval-report handoffs/ai2thor-real-small/outputs/diagnostics/vlm-semantic-eval-v1.json
```

Current result:

- full QA cases: 60
- single-frame VLM-observable cases: 5
- observable semantic match: 4/5
- observable strict exact match: 0/5
- filtered request bundle digest:
  `854958fed1b63215217a256edd314b338217d2164474b4e2a5707d77227d4853`
- observable slice report digest:
  `29e73529b5da97874eaf378dcd1b67fc9b56493377525dc30c0390484b0dc729`
- semantic eval report digest:
  `6b521a2d4102956745ce105d9b547c389861812f52a7091b5d7ce3f845c70393`

The filtered VLM request bundle contains no `gold_answer`, `gold_evidence`,
`required_nodes`, or `required_edges`. This makes the VLM baseline more
diagnosable without making a research claim. It also shows why the next real
DSG step is still coverage-driven collection: only 5/60 cases are currently
single-frame observable, far below the 30-case observation-aware target.

The VLM-only request path has now been tightened further: when generating
`offline-control-prediction-request-bundle.json`, the bundle selects a
`primary_frame` where the target object is visible if `frame-index.jsonl`
contains such a frame for the same episode and scene. The visible-object lists
are used only internally for frame selection and are not written to the request
case.

Current refreshed VLM input-quality diagnostics:

- request bundle digest:
  `aec7e1c6c40350b8783615072a9869f52ca9615ab54972f7f4fc71eaa1561859`
- primary-frame visibility report:
  `handoffs/ai2thor-real-small/outputs/diagnostics/vlm-primary-frame-visibility-v1.json`
- primary-frame visibility report digest:
  `de20daafdea3f6ee86d53c0f80933ac9029ce7ff60459f309d57f3811608ad32`
- object-location cases: 50
- primary frame missing: 0
- target visible in VLM primary frame: 9/50
- gold-visible semantic calibration remains: 4/5 semantic match, 0/5 strict
  exact match

This improves the next VLM-only rerun input quality without changing the old
`vlm.jsonl` score. The remaining bottleneck is still collection coverage: 9
target-visible single-frame inputs are better than 5, but still below the
30-case observation-aware comparison target.

Additional concrete next steps:

- Generate a visible-only collection target plan from missing QA target nodes.
- Recollect or import detector/RGB-D observations for those target objects.
- Merge returned coverage observations with the base observation sequence before
  rebuilding the predicted DSG:

```bash
python scripts/observations.py \
  --merge-sequence handoffs/ai2thor-real-small/outputs/predicted-dsg/detector-observations.json \
  --merge-sequence handoffs/ai2thor-real-small/outputs/predicted-dsg/detector-observations-coverage-targets-p1.json \
  --output-sequence handoffs/ai2thor-real-small/outputs/predicted-dsg/detector-observations-merged-p1.json \
  --report handoffs/ai2thor-real-small/outputs/diagnostics/detector-observation-merge-p1.json
```

- When the detector/RGB-D producer can localize an object to a visible region,
  room, or support surface, include `current_location_id` and
  `current_location_relation` in the object attributes. The graph builder now
  turns those fields into explicit containment edges only when the observation
  is visible detector evidence with `source_kind="detector"` plus `rgb`,
  `depth`, and `detector` evidence kinds.
- Keep hidden AI2-THOR metadata out of predicted graph evidence for final
  detector-only claims.
- Add a second-stage relation verifier to raise relation precision after
  coverage improves.
- Require state evidence in visible detector records before claiming dynamic
  memory improvement.
