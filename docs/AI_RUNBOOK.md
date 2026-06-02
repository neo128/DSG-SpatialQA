# AI Runbook

## Workflow
1. Plan the change before editing code.
2. Make the smallest scoped changes that satisfy the plan.
3. Verify with `python scripts/verify.py` or the focused deterministic command
   for the file you changed.
4. Keep `.github/workflows/verify.yml` aligned with `scripts/verify.py` when
   changing project verification gates.
5. Summarize the result, including commands run, known limits, and next steps.

## Guardrails
- Read `AGENTS.md` and `docs/VERIFICATION.md` before implementation.
- Do not call real model, robot, simulator, or network services.
- Keep tests deterministic and free of current-time or random dependencies.
- The local verifier typechecks `src`, `tests`, and `scripts`; keep the
  package's `py.typed` marker packaged so CLI entrypoints are checked against
  typed installed imports.
- The local verifier runs `python scripts/check_determinism.py` over `.github`,
  `scripts`, `src`, and `tests`; update that explicit scanner if a new
  deterministic boundary needs to be allowed or forbidden.
- Use `python scripts/verify.py --skip-install` only when the editable dev
  install is already current.
- For experiment handoff, generate deterministic suite reports through
  `python scripts/evaluate.py`, `evaluation_report()`,
  `evaluation_report_json()`, `evaluation_bundle()`,
  `evaluation_bundle_digest(bundle)`, `evaluation_bundle_json()`,
  `evaluation_case_listing()`, `evaluation_case_listing_digest(listing)`,
  `evaluation_case_listing_json(listing)`, `evaluation_manifest()`,
  `evaluation_manifest_digest(manifest)`,
  `evaluation_manifest_json()`, `save_evaluation_report(path, suite)`,
  `load_evaluation_report(path)`, `evaluation_report_digest(report)`,
  `validate_evaluation_report(report)`, `compare_evaluation_report(report)`,
  `save_evaluation_case_listing(path, ...)`,
  `load_evaluation_case_listing(path)`,
  `validate_evaluation_case_listing(listing)`,
  `compare_evaluation_case_listing(listing)`,
  `save_evaluation_manifest(path, ...)`, `load_evaluation_manifest(path)`,
  `validate_evaluation_manifest(manifest)`, `compare_evaluation_manifest(manifest)`,
  or `save_evaluation_bundle(path, ...)`
  with an explicit caller-supplied path.
  Suite and report breakdowns include case kind, QA question type, case tag,
  and scene fixture summaries for deterministic triage. Reports keep failed-case
  mismatch paths, report-level failure path summaries, raw mismatch reasons, and
  stable failure categories for faster benchmark-level triage.
- Use `python scripts/evaluate.py --tag qa --report evaluation-report.json` for
  shell-based offline report handoff, and combine `--name`, `--tag`, `--kind`,
  and `--question-type` only when a focused deterministic slice is needed.
- Use `python scripts/evaluate.py --list-cases --tag qa --question-type object_room`
  when a handoff only needs filtered case metadata for discovery; this emits
  case names, tags, question copies, expected keys, scene fixture metadata, and
  a schema-versioned stable listing digest without running evaluation cases.
- Use `python scripts/evaluate.py --validate-listing case-listing.json` before
  accepting a saved case listing; validation reads only that explicit local file
  and checks the schema version, listing digest, case count, required case
  metadata shape, and unique case names. Use
  `python scripts/evaluate.py --compare-listing case-listing.json` to detect
  current-code metadata drift without running evaluation cases.
- Use `python scripts/episodes.py --validate mock-episode.jsonl`,
  `python scripts/episodes.py --summary mock-episode.jsonl`, or
  `python scripts/episodes.py --compare mock-episode.jsonl` when a handoff needs
  deterministic simulator/mock episode JSONL checks before graph construction.
  Episode JSONL files use one explicit-step frame per line, stable sorted-key
  serialization, SHA-256 digests, summary metadata, episode/step ordering
  validation, and duplicate episode-step diagnostics without connecting to a
  simulator.
- Use `python scripts/collect_ai2thor.py --mock --scene FloorPlan1 --episode-id ai2thor_mock_001 --step 1 --step 2 --action Initialize --action MoveAhead --output mock-ai2thor.jsonl`
  when a handoff needs an AI2-THOR-shaped mock episode without launching a
  simulator. Steps must be supplied explicitly with repeated `--step` values.
  Non-mock collection fails closed in the default local environment with
  structured JSON containing the missing optional dependency message, so default
  verification never imports or calls a simulator.
- Use `python scripts/build_oracle_graph.py --input mock-episode.jsonl --output-graph oracle-graph.json --report oracle-report.json`
  when a mock episode metadata handoff needs an oracle `DynamicSceneGraph`
  artifact. The builder consumes only explicit episode JSONL metadata for rooms,
  regions, objects, and relations; it emits graph JSON and report JSON only to
  caller-supplied local paths. Use
  `python scripts/build_oracle_graph.py --validate-report oracle-report.json`
  before accepting a saved oracle report, and
  `python scripts/build_oracle_graph.py --compare-report oracle-report.json` to
  rebuild from the recorded episode path and detect current graph/report or
  exported graph-file drift.
- Use `python scripts/generate_qa.py --graph oracle-graph.json --scene-id mock_scene --episode-id mock_episode --output qa.jsonl --max-cases 100`
  when an oracle graph handoff needs deterministic QA JSONL cases. The generator
  reads only that explicit graph file, records graph digest, stable case IDs,
  oracle answers, evidence node IDs, evidence edge IDs, choices, and tags, then
  writes only to the caller-supplied output path. Use
  `python scripts/generate_qa.py --validate qa.jsonl` before accepting a saved
  QA dataset, and
  `python scripts/generate_qa.py --compare qa.jsonl --graph oracle-graph.json`
  to replay saved questions through the current `SpatialQAEngine` and detect
  answer, evidence, or graph-digest drift.
- Use `scene_observation_to_json()`, `scene_observation_from_json()`,
  `save_scene_observation(path)`, or `load_scene_observation(path)` when a
  mock perception handoff needs stable offline `SceneObservation` input files.
  Use the `scene_observation_sequence_*` helpers for ordered multi-frame
  observation streams. Observation files must use explicit caller-supplied local
  paths and explicit `SceneObservation.step` values.
- Use `python scripts/observations.py --validate-sequence mock-observation-sequence.json`
  when a handoff needs the lightest graph-free gate for raw sequence schema,
  observation count, step consistency, stable digest, and summary metadata.
- Use `python scripts/observations.py --summarize-sequence mock-observation-sequence.json --report mock-observation-sequence-summary.json`
  when a handoff needs graph-free sequence audit metadata before ingesting. The
  summary reads only that explicit sequence file and reports stable sequence
  digest, step spans, object IDs, label counts, visibility counts,
  low-confidence counts, and re-observation candidate counts. Use
  `python scripts/observations.py --validate-sequence-summary mock-observation-sequence-summary.json`
  before sharing the summary artifact, and
  `python scripts/observations.py --compare-sequence-summary mock-observation-sequence-summary.json`
  to detect sequence-file drift from the explicit path recorded in that summary.
- Use `python scripts/observations.py --input mock-observation-sequence.json --output-graph mock-observation-graph.json --report mock-observation-ingest-report.json`
  when a handoff needs to turn an explicit local observation sequence into
  graph JSON plus a stable ingest report with sequence digest, graph digest,
  summary, and per-step ingest results. Use
  `python scripts/observations.py --validate-report mock-observation-ingest-report.json`
  before sharing the report, and
  `python scripts/observations.py --compare-report mock-observation-ingest-report.json`
  to detect input, exported-graph-file, or output drift by re-ingesting the
  explicit sequence path and reading the explicit graph path recorded in the
  report. Invalid or drifted artifacts return non-zero structured JSON with
  `valid: false` or `matches: false` instead of tracebacks.
  Use `save_observation_ingest_report(report, path)` and
  `load_observation_ingest_report(path)` for Python handoffs that need the same
  explicit local report artifact model. Use
  `observation_ingest_report_digest(report)` when a handoff needs to recompute
  the report fingerprint. Report validation also checks that the report digest,
  saved input path, graph path, and nested graph report path are present and
  consistent before handoff.
- Use `GraphTool.compute_distance(src, dst)` when an experiment needs a stable
  rounded world-frame metric distance payload with both endpoint poses.
  `GraphTool.update_spatial_relations()` can infer computed geometric
  relations such as `NEAR`, `ON`, `INSIDE`, and `SUPPORTS` for explicit steps.
  Treat `VISIBLE_FROM`, `REACHABLE_FROM`, and `OCCLUDES` as explicit placeholder
  edges only; they can be queried through `GraphTool.query_graph()` but are not
  inferred from sensors, frustums, depth, or navigation state in the MVP.
- Use `scene_fixture_manifest_json()` or `save_scene_fixture_manifest(path, ...)`
  when a Python handoff needs the same stable fixture manifest JSON produced by
  `python scripts/scene.py --list-fixtures`; saving always uses an explicit
  caller-supplied local path.
- Use `graph_report()`, `graph_report_digest()`, `graph_report_json()`,
  `save_graph_report(path, ...)`,
  or `python scripts/scene.py --fixture tabletop --output tabletop-scene.json --report tabletop-report.json`
  when a scene graph handoff needs a stable graph digest, report digest, and
  summary report alongside the exported graph JSON. Report saving always uses an
  explicit local path.
  Validate and compare saved graph reports with `load_graph_report()`,
  `validate_graph_report()`, `compare_graph_report()`,
  `python scripts/scene.py --validate-report tabletop-report.json`, or
  `python scripts/scene.py --compare-report tabletop-report.json`.
  Use `compare_graph_report_to_file(report, path)` or
  `python scripts/scene.py --compare-report-graph tabletop-report.json --input tabletop-scene.json`
  when the handoff should verify a saved report against a caller-supplied graph
  JSON artifact rather than the current built-in fixture baseline.
- Use `python scripts/evaluate.py --compare-report evaluation-report.json` to
  detect compact-report drift against the current code; comparison reads only
  the saved report's selected case names, reruns that deterministic local slice,
  and checks the saved report digest, suite digest, case selection digest,
  per-case digests, summary, metrics, evidence metrics, failure diagnostics,
  and breakdown. Use
  `python scripts/evaluate.py --validate-report evaluation-report.json` when a
  handoff only needs to validate the explicit local report artifact fingerprint;
  validation checks the report schema version, suite digest format, case
  selection digest, case selection entry metadata shape, case selection
  consistency with `summary.selected_cases`, failed-case detail consistency with
  `summary.failed_cases`, failed-case entry metadata shape, case digest
  consistency with `summary.selected_cases`, summary case-list shape and
  failed-case membership in selected cases, summary count consistency with the
  selected/failed case lists, breakdown count consistency with each grouped
  entry's selected/failed case lists, breakdown case-list consistency with
  `case_selection` metadata, metric consistency with summary/breakdown,
  top-level and grouped evidence metric internal consistency with
  summary/breakdown counts, evidence metric value ranges, runtime error category
  entry shape, runtime error category count/case consistency with selected
  cases, per-case digest format, per-case digest metadata consistency with
  `case_selection`, per-case digest pass/fail status consistency with
  `summary.failed_cases`, failure diagnostic aggregate consistency with
  `failed_cases`, and saved report digest.
  Summary, failed-case, case-digest, metric, evidence-metric, and breakdown
  drift include stable nested `differences` paths such as `failed`,
  `tabletop_object_location`,
  `tabletop_object_location.digest`,
  `by_question_type.object_location.pass_rate`,
  `by_question_type.object_location.evidence_edge_count`, and `by_tag.qa.failed`;
  runtime error category drift includes paths such as `missing_object.count`;
  failure diagnostic drift includes paths such as `value_mismatch` and
  `answer.visible`.
- Use `python scripts/evaluate.py --manifest --tag qa --report qa-manifest.json`
  when the handoff needs filtered case manifests, fixture manifests, and
  coverage counts without running the selected evaluation cases.
- Use `python scripts/evaluate.py --validate-manifest qa-manifest.json` before
  accepting a saved manifest; validation reads only that explicit local file
  and checks schema version, manifest digest, required case metadata shape,
  unique case names, fixture coverage, case-backed scene fixture metadata
  consistency, and coverage summary consistency. Case metadata, fixture
  metadata, and coverage summary drift include stable nested `differences`
  paths such as `evaluation_cases[0].name`, `tabletop.tags`, and `by_tag.qa`.
- Use `python scripts/evaluate.py --compare-manifest qa-manifest.json` to detect
  metadata drift against the current code without running cases; comparison
  reads only the saved manifest filters and checks digest, coverage, case
  manifest, and fixture manifest equality. Coverage and manifest metadata drift
  include stable nested `differences` paths such as `by_tag.qa` and
  `tabletop_relation_timeline.tags`.
- Use `python scripts/evaluate.py --tag qa --tag relations` when checking
  static and dynamic relation regressions, including direct relative checks and
  the `relation_shift` fixture.
- Use `python scripts/evaluate.py --tag qa --tag foundation` when checking basic
  QA contracts for agent location, agent pose history, object location evidence,
  deterministic missing-object errors, object status, object history, and direct
  relative-relation answers.
- Use `python scripts/evaluate.py --question-type object_room` or
  `python scripts/evaluate.py --tag qa --tag room` when checking multi-room
  containment resolution, including room id/label, path nodes, and evidence edge
  IDs for relocated and stable objects.
- Use `python scripts/evaluate.py --tag qa --tag error` when checking
  structured QA error-path regressions, including missing objects, unsupported
  question types, invalid question fields, and invalid explicit step windows.
  These outputs include stable `error_category` values for local drift triage.
- Use `python scripts/evaluate.py --question-type nearest_object` when checking
  nearest-object selection, caller-supplied candidate filtering, and stable
  candidate distance diagnostics.
- Use `python scripts/evaluate.py --tag qa --tag label --tag ambiguity` when
  checking direct QA same-label candidate listing and ambiguity evidence.
- Use `python scripts/evaluate.py --name needs_reobserve_spoon_label_candidates`
  when checking that low-confidence invisible objects can still be returned by
  semantic label lookup with state evidence and `needs_reobserve`.
- Use `python scripts/evaluate.py --name needs_reobserve_bowl_pick_target_not_visible`
  when checking that invisible high-confidence VLA pick targets return
  `target_not_visible` without emitting a command.
- Use `python scripts/evaluate.py --name needs_reobserve_cup_pick_low_confidence`
  when checking that visible low-confidence VLA pick targets return
  `low_confidence` without emitting a command.
- Use `python scripts/evaluate.py --name needs_reobserve_bowl_place_target_not_visible`
  when checking that an invisible high-confidence place-relative target object
  returns `target_not_visible` and target-object diagnostics.
- Use `python scripts/evaluate.py --name needs_reobserve_cup_place_target_low_confidence`
  when checking that a visible low-confidence place-relative target object
  returns `low_confidence` and target-object diagnostics.
- Use `python scripts/evaluate.py --name needs_reobserve_bowl_place_reference_target_not_visible`
  when checking that an invisible high-confidence place-relative reference
  object returns `target_not_visible` and reference-object diagnostics.
- Use `python scripts/evaluate.py --name needs_reobserve_cup_place_reference_low_confidence`
  when checking that a visible low-confidence place-relative reference object
  returns `low_confidence` and reference-object diagnostics.
- Use `python scripts/evaluate.py --name needs_reobserve_spoon_place_reference`
  when checking that a place-relative command refuses an invisible
  low-confidence reference object with `needs_reobserve` and reference-object
  diagnostics.
- Use `python scripts/evaluate.py --name needs_reobserve_spoon_place_target`
  when checking that a place-relative command refuses an invisible
  low-confidence target object with `needs_reobserve` and target-object
  diagnostics.
- Use `python scripts/evaluate.py --tag vla --tag label --tag ambiguity` when
  checking that VLA semantic ambiguity returns candidate diagnostics instead of
  choosing one same-label object.
- Use `python scripts/evaluate.py --tag vla --tag error` when checking VLA
  planner error-path regressions, including missing pick/place target inputs,
  missing object or semantic-label pick targets, invisible pick targets and
  place-relative targets/references, visible low-confidence pick targets and
  place-relative targets/references, missing place references, missing
  reference inputs, and unsupported place relations that must not emit commands.
  These outputs include stable `error_category` values for planner failure
  aggregation.
- Check `runtime_error_categories` in suite, report, or bundle output when the
  handoff needs category counts and affected case names without reprocessing
  every case result. Compact-report validation rejects empty zero-count category
  entries, checks category counts against unique affected selected cases, and
  comparison surfaces category-count or case drift with stable nested
  `differences` paths.
- Check `runtime_error_metrics` in compact reports and bundles when the handoff
  needs runtime-error case counts, clean case counts, runtime-error rate, and
  per-category case rates for benchmark triage. Compact-report validation
  recomputes those metrics from the saved category aggregates, and comparison
  surfaces drift with paths such as `by_category.missing_target.case_rate`.
- Check `evidence_metrics` in compact reports and bundles when comparing how
  much node, edge, or VLA command evidence a deterministic experiment produced;
  validation rejects negative counts, and the metrics include grouped summaries
  by kind, question type, scene fixture, and tag.
- Use `python scripts/evaluate.py --bundle --tag qa --report qa-bundle.json`
  when the handoff needs case manifests, fixture manifests, full suite results,
  compact report metrics (`case_count`, passed/failed counts, pass rate, and
  failure rate), grouped report metrics by kind, question type, scene fixture,
  and tag, deterministic coverage counts, suite digest, and bundle digest in
  one reproducible local artifact.
- Use `python scripts/evaluate.py --validate-bundle qa-bundle.json` before
  accepting a saved bundle; validation reads only that explicit local file and
  checks schema version, suite digest, bundle digest, report consistency, case
  manifest names, required case metadata shape, unique case names,
  suite-backed metadata, fixture manifest coverage and case-backed metadata, and
  coverage summary consistency. Coverage summary, case metadata shape, case
  manifest metadata, fixture manifest metadata, and compact report drift include
  stable nested `differences` paths, including report paths such as
  `failed_cases.tabletop_object_location`, case metadata paths such as
  `evaluation_cases[0].question`, case manifest paths such as
  `multi_room_rearrangement_reobserve_targets.tags`, and fixture manifest paths
  such as `needs_reobserve.tags`.
- Use `python scripts/evaluate.py --compare-bundle qa-bundle.json` to detect
  benchmark drift against the current code; comparison reads only the saved
  bundle filters, reruns the deterministic local suite, and checks digest,
  bundle artifact digest, compact report, coverage, case manifest, and fixture
  manifest equality.
  Compact report drift includes stable nested `differences` paths such as
  `metrics.by_tag.qa.pass_rate`; coverage and manifest metadata drift include
  stable nested `differences` paths for handoff triage.
- Evaluation artifact validation and comparison commands report invalid
  explicit report, manifest, or bundle files with a non-zero status and stable
  JSON containing `valid: false`, `path`, and `error` instead of a traceback;
  compare failures also include `matches: false`.
- Use `python scripts/scene.py --list-fixtures --tag multi_room --output multi-room-fixtures.json`
  to discover filtered built-in scene fixture metadata from the shell. The
  command reports schema version, metadata digest, filters, `fixture_count`, and
  scene names/descriptions/tags without loading graph objects or computing graph
  JSON digests. With `--output`, it writes the same stable JSON to that explicit
  local path and stdout.
- Use `python scripts/scene.py --validate-fixture-manifest multi-room-fixtures.json`
  before accepting a saved fixture-only handoff. Validation reads only that
  explicit local file and checks schema version, digest, and fixture count
  consistency without loading scene graphs.
- Use `python scripts/scene.py --compare-fixture-manifest multi-room-fixtures.json`
  to detect current-code drift in a saved fixture-only handoff. Comparison reads
  only the saved manifest filters, regenerates current fixture metadata without
  loading graph objects, and reports stable metadata `differences` paths such as
  `multi_room_rearrangement.tags`.
- Use `python scripts/scene.py --fixture tabletop --output tabletop-scene.json`
  to export a built-in deterministic scene graph, and
  `python scripts/scene.py --validate tabletop-scene.json` to load an explicit
  local graph file and report its stable digest, total summary counts, node-type
  counts, edge-relation counts, object-label counts, current-location and
  current-room counts, and visibility/re-observe candidate counts.
- Use `python scripts/scene.py --compare-fixture tabletop --input tabletop-scene.json`
  to detect drift between an explicit local graph file and the current
  deterministic built-in fixture digest and summary counts. Summary drift
  includes stable nested `differences` paths such as
  `by_current_room.pantry`, `by_current_location.pantry_shelf`, and
  `by_node_type.region`.
- Use `GraphTool.current_room(object_id)` in local experiments when an object
  needs a room-level explanation; it returns the resolved room id/label,
  containment path, and evidence edge IDs, or `None` when no room can be
  resolved.
- Scene graph validation and fixture comparison report `valid: true` for loaded
  explicit graph files. Invalid explicit graph JSON returns a non-zero status
  with stable JSON containing `valid: false`, `path`, and `error` instead of a
  traceback; compare failures also include `fixture` and `matches: false`.
