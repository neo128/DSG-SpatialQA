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
  `evaluation_bundle_json()`, `evaluation_manifest()`,
  `evaluation_manifest_json()`, `save_evaluation_report(path, suite)`,
  `load_evaluation_report(path)`, `compare_evaluation_report(report)`,
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
  case names, tags, question copies, expected keys, and scene fixture metadata
  without running evaluation cases.
- Use `python scripts/evaluate.py --compare-report evaluation-report.json` to
  detect compact-report drift against the current code; comparison reads only
  the saved report's selected case names, reruns that deterministic local slice,
  and checks digest, per-case digests, summary, metrics, evidence metrics,
  failure diagnostics, and breakdown. Summary, failed-case, case-digest, metric,
  evidence-metric, and breakdown drift include stable nested `differences` paths
  such as `failed`, `tabletop_object_location`,
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
  and checks schema version, manifest digest, fixture coverage, and coverage
  summary consistency. Coverage summary drift includes stable nested
  `differences` paths such as `by_tag.qa`.
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
  QA contracts for agent location, object location evidence, deterministic
  missing-object errors, object status, object history, and direct
  relative-relation answers.
- Use `python scripts/evaluate.py --question-type object_room` or
  `python scripts/evaluate.py --tag qa --tag room` when checking multi-room
  containment resolution, including room id/label, path nodes, and evidence edge
  IDs for relocated objects.
- Use `python scripts/evaluate.py --tag qa --tag error` when checking
  structured QA error-path regressions, including missing objects and invalid
  explicit step windows. These outputs include stable `error_category` values
  for local drift triage.
- Use `python scripts/evaluate.py --question-type nearest_object` when checking
  nearest-object selection, caller-supplied candidate filtering, and stable
  candidate distance diagnostics.
- Use `python scripts/evaluate.py --tag qa --tag label --tag ambiguity` when
  checking direct QA same-label candidate listing and ambiguity evidence.
- Use `python scripts/evaluate.py --tag vla --tag label --tag ambiguity` when
  checking that VLA semantic ambiguity returns candidate diagnostics instead of
  choosing one same-label object.
- Use `python scripts/evaluate.py --tag vla --tag error` when checking VLA
  planner error-path regressions, including missing pick targets and missing
  place references, plus unsupported place relations that must not emit
  commands. These outputs include stable `error_category` values for planner
  failure aggregation.
- Check `runtime_error_categories` in suite, report, or bundle output when the
  handoff needs category counts and affected case names without reprocessing
  every case result. Compact-report comparison surfaces category-count or case
  drift with stable nested `differences` paths.
- Check `evidence_metrics` in compact reports and bundles when comparing how
  much node, edge, or VLA command evidence a deterministic experiment produced;
  the metrics include grouped summaries by kind, question type, scene fixture,
  and tag.
- Use `python scripts/evaluate.py --bundle --tag qa --report qa-bundle.json`
  when the handoff needs case manifests, fixture manifests, full suite results,
  compact report metrics (`case_count`, passed/failed counts, pass rate, and
  failure rate), grouped report metrics by kind, question type, scene fixture,
  and tag, deterministic coverage counts, and digest in one reproducible local
  artifact.
- Use `python scripts/evaluate.py --validate-bundle qa-bundle.json` before
  accepting a saved bundle; validation reads only that explicit local file and
  checks schema version, suite digest, report consistency, case manifest names
  and suite-backed metadata, fixture manifest coverage and case-backed
  metadata, and coverage summary consistency. Coverage summary, case manifest
  metadata, fixture manifest metadata, and compact report drift include stable
  nested `differences` paths, including report paths such as
  `failed_cases.tabletop_object_location`, case manifest paths such as
  `multi_room_rearrangement_reobserve_targets.tags`, and fixture manifest paths
  such as `needs_reobserve.tags`.
- Use `python scripts/evaluate.py --compare-bundle qa-bundle.json` to detect
  benchmark drift against the current code; comparison reads only the saved
  bundle filters, reruns the deterministic local suite, and checks digest,
  compact report, coverage, case manifest, and fixture manifest equality.
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
