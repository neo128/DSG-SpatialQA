# DSG-SpatialQA Lab

Deterministic minimal verification package for Dynamic Scene Graph spatial QA.

The lab is intentionally local and in-memory. It is meant to make spatial QA,
scene graph retrieval, temporal state audits, and VLA anchor planning
reproducible without calling live AI services, robot stacks, simulators, clocks,
or random sources.

## Quickstart

Install the package with development tools:

```bash
python -m pip install -e ".[dev]"
```

Run a minimal deterministic spatial QA example:

```python
from dsg_spatialqa_lab import GraphTool, SpatialQAEngine, load_scene_fixture

graph = load_scene_fixture("tabletop")
qa = SpatialQAEngine(GraphTool(graph))

response = qa.answer({"type": "object_location", "object_id": "mug_1"})
print(response.answer)
```

Run the complete local verification gate:

```bash
python scripts/verify.py
```

Run a deterministic evaluation report or reproducibility bundle from the shell:

```bash
python scripts/evaluate.py --tag qa --tag dynamic --report evaluation-report.json
python scripts/evaluate.py --compare-report evaluation-report.json
python scripts/evaluate.py --list-cases --tag qa --question-type object_room
python scripts/evaluate.py --manifest --tag qa --tag relations --report relations-manifest.json
python scripts/evaluate.py --validate-manifest relations-manifest.json
python scripts/evaluate.py --compare-manifest relations-manifest.json
python scripts/evaluate.py --bundle --tag qa --tag reobserve --report reobserve-bundle.json
python scripts/evaluate.py --validate-bundle reobserve-bundle.json
python scripts/evaluate.py --compare-bundle reobserve-bundle.json
```

Evaluation artifact validation and comparison return non-zero structured JSON
with `valid: false` when an explicit report, manifest, or bundle file is
unreadable or fails artifact loading.

Export and validate a deterministic scene fixture graph from the shell:

```bash
python scripts/scene.py --list-fixtures --tag multi_room --output multi-room-fixtures.json
python scripts/scene.py --fixture tabletop --output tabletop-scene.json
python scripts/scene.py --validate tabletop-scene.json
python scripts/scene.py --compare-fixture tabletop --input tabletop-scene.json
```

Fixture listing emits a filtered scene fixture manifest with schema version,
metadata digest, fixture count, and scene names/descriptions/tags; it does not
load graph objects or compute graph JSON digests. When `--output` is supplied,
the same stable JSON is written to that explicit local path and printed to stdout.

Scene validation and fixture comparison return non-zero structured JSON with
`valid: false` when an explicit graph file is unreadable or fails schema
validation.

If the editable development install is already current, skip that first gate:

```bash
python scripts/verify.py --skip-install
```

## MVP Capabilities

- In-memory Dynamic Scene Graph state for agents, objects, rooms, regions,
  actions, events, current state, and explicit-step history.
- Deterministic spatial relations and graph retrieval through `GraphTool`.
- Structured QA intents for object state, agent state, label-candidate
  ambiguity, room-level containment, timelines, scene snapshots, scene deltas,
  world state, recent events, graph queries, and re-observation targets.
- Deterministic VLA anchor planning for pick and place-relative commands,
  including ambiguity candidate diagnostics, stale-action, and re-observation
  handling.
- Built-in scene fixtures and evaluation cases with stable suite summaries,
  failure diagnostics, and SHA-256 digests for experiment records.
- Offline evaluation report, manifest, and bundle CLI entrypoints with
  structured invalid-file diagnostics for reproducible handoffs.
- Dynamic fixture coverage for multi-room relocation, occlusion, re-observation,
  relation shifts, temporal deltas, and step-window event audits.
- JSON graph import/export helpers, graph digests, graph summaries with
  object-state, current-location, current-room, node-type, edge-relation, and
  object-label counts, and a scene fixture metadata/export/validate/compare CLI
  with structured invalid-file diagnostics for reproducible local experiments.

## Project Boundaries

- Runtime code must stay deterministic: callers supply explicit steps and the
  package does not read wall-clock time or produce random output.
- No runtime network calls are allowed.
- Real LLM/VLM, robot, simulator, database, and service integrations are out of
  scope for the MVP and should be mocked or omitted.
- Data structures remain in-memory until a real integration justifies
  persistence.
- Development tooling may install declared dev dependencies, but package runtime
  dependencies remain standard-library only.

## Development Baseline

Use the one-command verifier before handing off changes:

```bash
python scripts/verify.py
```

It runs the project gates in order: editable dev install, lint, typecheck,
determinism scan, unit tests, package build, and the built-in evaluation suite.
The individual commands remain available when focused feedback is useful:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m mypy src tests scripts
python scripts/check_determinism.py
python -m pytest -q
python -m build
```

GitHub Actions uses the same local verifier through
`.github/workflows/verify.yml`, so CI and local handoff checks stay aligned.
The package declares a `py.typed` marker so CLI scripts can be typechecked
against the installed package during the verifier run.
The determinism scan is a local source check over `.github`, `scripts`, `src`,
and `tests` for current-time, random, network, or external model client
boundaries.

The built-in evaluation suite can also be run directly:

```bash
python scripts/evaluate.py --name tabletop_object_location
python scripts/evaluate.py --kind vla_pick --report evaluation-report.json
python scripts/evaluate.py --tag qa --tag dynamic --question-type scene_delta
python scripts/evaluate.py --list-cases --tag vla --tag error
python scripts/evaluate.py --compare-report evaluation-report.json
python scripts/evaluate.py --bundle --tag qa --tag reobserve
python scripts/evaluate.py --validate-bundle reobserve-bundle.json
python scripts/evaluate.py --compare-bundle reobserve-bundle.json
python scripts/evaluate.py --compare-manifest relations-manifest.json
```

The CLI prints stable JSON to stdout and, when `--report` is provided, writes
the selected report, case listing, manifest, or bundle to an explicit local
path. `--list-cases` emits only filtered case metadata and a case count without
running cases, which is useful for discovering focused benchmark slices.
Manifest output includes the filter manifest, selected scene fixtures, selected
evaluation cases, deterministic coverage counts, and a digest without running
cases. Manifest validation reads only the explicit local JSON file and checks
schema version, digest, scene fixture coverage, and coverage summary
consistency. Coverage summary validation and comparison include stable nested
`differences` entries with paths such as `by_scene_fixture.tabletop`.
Manifest comparison reads the saved filters from an explicit local manifest,
regenerates the current deterministic metadata without running cases, and
returns a non-zero status when digest, coverage, case manifest, or fixture
manifest drift is detected. Coverage, case manifest, and fixture manifest drift
include stable nested `differences` entries with paths such as `by_tag.qa` and
`tabletop_relation_timeline.tags`.
Report comparison reads the selected case names from an explicit local compact
report, reruns that deterministic case slice, and returns a non-zero status when
digest, summary, metrics, failure diagnostics, or breakdown drift is detected.
Summary, failed-case, metric, and breakdown drift checks include stable nested
`differences` entries with paths such as `failed`,
`tabletop_object_location`, `by_tag.qa.pass_rate`, and `by_tag.qa.failed` for
quick report triage.
Runtime error category drift checks include stable nested `differences` entries
with category paths such as `missing_object.count`.
Failure diagnostic drift checks also include stable nested `differences` entries
with category or reason paths such as `value_mismatch`, and mismatch paths such
as `answer.visible`.
Bundle output includes
the filter manifest, selected scene fixtures, selected evaluation cases, full
suite results, compact report, coverage counts, and digest. Compact report
metrics include selected, passed, and failed case counts, pass rate, and failure
rate. Coverage counts are grouped by case kind, QA question type, case tag, scene
fixture, and scene tag.
Suite and report breakdowns also include QA question-type summaries for direct
intent-level triage.
Bundle validation reads only the explicit local JSON file, recomputes
deterministic consistency checks, and returns a non-zero status when the bundle
does not validate. Validation checks schema version, suite digest, report
consistency, case manifest names and suite-backed metadata, scene fixture
coverage and case-backed metadata, and coverage summary consistency. Report
consistency, case manifest metadata, scene fixture metadata, and coverage
summary validation include stable nested `differences` entries for quick
handoff triage, including compact-report paths such as
`failed_cases.tabletop_object_location`, case manifest paths such as
`multi_room_rearrangement_reobserve_targets.tags`, and fixture manifest paths
such as `needs_reobserve.tags` when summaries or metadata drift.
Bundle comparison reads the saved filters from an explicit local bundle, reruns
the current deterministic suite, and returns a non-zero status when digest,
compact report, coverage, case manifest, or fixture manifest drift is detected.
Compact report drift includes stable nested `differences` paths such as
`metrics.by_tag.qa.pass_rate`. Case and fixture manifest metadata drift is keyed
by manifest entry name for handoff triage.
Evaluation artifact validation and comparison commands mark invalid explicit
report, manifest, or bundle files with `valid: false`, a stable `error` string,
and a non-zero exit status; comparison error payloads also include
`matches: false`.
Scene graph comparison reads an explicit local graph JSON file and compares its
digest and summary counts with a freshly generated built-in fixture graph.
Graph summaries include total counts, visible/hidden object counts,
low-confidence and re-observation candidate counts, current containment counts
such as `by_current_location`, resolved room counts such as `by_current_room`,
plus `by_node_type`, `by_edge_relation`, and `by_object_label` counts. Summary
drift checks include stable nested `differences` entries such as
`by_current_room.pantry`, `by_current_location.pantry_shelf`,
`by_node_type.region`, and `node_count`. Scene validation and comparison mark valid explicit graph inputs
with `valid: true`; invalid explicit graph files return `valid: false`, a stable
`error` string, and a non-zero exit status.

```python
from dsg_spatialqa_lab import (
    compare_evaluation_bundle,
    compare_evaluation_manifest,
    compare_evaluation_report,
    compare_graph_file_to_fixture,
    compare_graph_to_fixture,
    evaluation_bundle,
    evaluation_bundle_json,
    evaluation_manifest,
    evaluation_manifest_json,
    evaluation_report,
    evaluation_report_json,
    load_evaluation_bundle,
    load_evaluation_manifest,
    load_evaluation_report,
    run_evaluation_suite,
    save_evaluation_bundle,
    save_evaluation_manifest,
    save_evaluation_report,
    validate_evaluation_bundle,
    validate_evaluation_manifest,
)

suite = run_evaluation_suite()
print(suite["summary"])
print(suite["digest"])

report = evaluation_report(suite)
print(evaluation_report_json(report))
save_evaluation_report("evaluation-report.json", suite)
loaded_report = load_evaluation_report("evaluation-report.json")
report_comparison = compare_evaluation_report(loaded_report)

manifest = evaluation_manifest(tags=("qa", "relations"))
print(evaluation_manifest_json(manifest))
save_evaluation_manifest("relations-manifest.json", tags=("qa", "relations"))
loaded_manifest = load_evaluation_manifest("relations-manifest.json")
manifest_validation = validate_evaluation_manifest(loaded_manifest)
manifest_comparison = compare_evaluation_manifest(loaded_manifest)

bundle = evaluation_bundle(tags=("qa", "reobserve"))
print(evaluation_bundle_json(bundle))
print(bundle["coverage"])
save_evaluation_bundle("reobserve-bundle.json", tags=("qa", "reobserve"))
loaded_bundle = load_evaluation_bundle("reobserve-bundle.json")
bundle_validation = validate_evaluation_bundle(loaded_bundle)
bundle_comparison = compare_evaluation_bundle(loaded_bundle)
fixture_comparison = compare_graph_file_to_fixture("tabletop-scene.json", "tabletop")
```

## Roadmap

- Keep CI and local verification aligned as new benchmark gates are added.
- Add optional deterministic dataset import adapters once real experiment
  formats are chosen.
- Expand relation geometry and sensor evidence models while preserving explicit
  caller-supplied steps.
- Extend offline report outputs with any new benchmark metrics before adding
  external integrations.
- Add persistence only after there is a concrete integration requirement.
- Keep external AI, robot, and simulator adapters outside the deterministic core
  behind mocked or offline boundaries.

## Full API Example

```python
from dsg_spatialqa_lab import (
    BBox3D,
    compare_evaluation_bundle,
    compare_evaluation_manifest,
    compare_evaluation_report,
    compare_graph_file_to_fixture,
    compare_graph_to_fixture,
    build_relation_shift_scene,
    DynamicSceneGraph,
    EvaluationCase,
    evaluation_bundle,
    evaluation_bundle_json,
    evaluation_cases_metadata,
    evaluation_manifest,
    evaluation_manifest_json,
    evaluation_report,
    evaluation_report_json,
    GraphQuery,
    GraphTool,
    ObjectObservation,
    ObservationIngestor,
    Pose3D,
    SceneObservation,
    SpatialQAEngine,
    VLAAnchorPlanner,
    graph_from_json,
    graph_json_digest,
    graph_summary,
    graph_to_json,
    load_evaluation_bundle,
    load_evaluation_manifest,
    load_evaluation_report,
    load_graph_json,
    list_scene_fixture_metadata,
    list_scene_fixtures,
    load_scene_fixture,
    list_evaluation_case_metadata,
    run_evaluation_cases,
    run_evaluation_suite,
    save_evaluation_bundle,
    save_evaluation_manifest,
    save_evaluation_report,
    save_graph_json,
    scene_fixture_manifest,
    validate_evaluation_bundle,
    validate_evaluation_manifest,
)

available_scenes = list_scene_fixtures()
scene_manifest = list_scene_fixture_metadata()
scene_fixture_handoff = scene_fixture_manifest(tags=("reobserve",))
ambiguity_scene_manifest = list_scene_fixture_metadata(tags=("ambiguity",))
reobserve_scene_manifest = list_scene_fixture_metadata(tags=("reobserve",))
graph = load_scene_fixture("tabletop")
tool = GraphTool(graph)
qa = SpatialQAEngine(tool)
agent_location = qa.answer({"type": "agent_location"})
agent_history = qa.answer({"type": "agent_history"})
agent_timeline = qa.answer({"type": "agent_timeline"})
response = qa.answer({"type": "object_location", "object_id": "mug_1"})
object_status = qa.answer({"type": "object_status", "object_id": "mug_1"})
object_timeline = qa.answer({"type": "object_timeline", "object_id": "mug_1"})
relation_timeline = qa.answer(
    {
        "type": "relation_timeline",
        "src": "mug_1",
        "dst": "plate_1",
        "reference_frame": "agent",
    }
)
reobserve_targets = qa.answer({"type": "reobserve_targets"})
nearest_plate = qa.answer(
    {"type": "nearest_object", "src": "mug_1", "candidates": ["plate_1"]}
)
label_candidates = qa.answer(
    {"type": "label_candidates", "label": "mug", "visible": True}
)
snapshot = qa.answer({"type": "scene_snapshot", "step": 1, "visible": True})
delta = qa.answer({"type": "scene_delta", "from_step": 1, "to_step": 2})
world_state = qa.answer({"type": "world_state", "visible": True})
recent_events = qa.answer({"type": "recent_events", "since_step": 1})
multi_room_tool = GraphTool(load_scene_fixture("multi_room_rearrangement"))
room_response = SpatialQAEngine(multi_room_tool).answer(
    {"type": "object_room", "object_id": "cereal_box_1"}
)
graph_query_response = qa.answer(
    {
        "type": "graph_query",
        "query": {
            "node_types": ["object"],
            "labels": ["mug", "plate"],
            "relations": ["LEFT_OF", "NEAR"],
            "reference_frame": "agent",
        },
    }
)
retrieved_subgraph = qa.answer(
    {"type": "retrieve_subgraph", "query": "mug", "max_nodes": 3, "hops": 1}
)
planner = VLAAnchorPlanner(tool)
pick = planner.plan_pick(target_object="mug_1")
place = planner.plan_place_relative("mug_1", "plate_1", "RIGHT_OF")
place_validity = planner.validate(place.command) if place.command is not None else place

subgraph = tool.query_graph(
    GraphQuery(
        node_types=("object",),
        labels=("mug", "plate"),
        visible=True,
        relations=("LEFT_OF", "NEAR"),
        reference_frame="agent",
    )
)
agent_timeline = tool.agent_timeline("agent")
timeline = tool.object_timeline("mug_1")
relation_timeline = tool.relation_timeline(src="mug_1", reference_frame="agent")
current_room = tool.current_room("mug_1")
current_world = tool.world_state(visible=True)
recent_event_trace = tool.recent_events(since_step=1)
targets_to_reobserve = tool.reobserve_targets()

inferred_edges = tool.update_spatial_relations(
    step=2,
    object_ids=("mug_1", "plate_1"),
    relations=("LEFT_OF", "RIGHT_OF", "NEAR"),
    reference_frames=("agent",),
)

ingest_result = ObservationIngestor(graph).ingest(
    SceneObservation(
        step=3,
        agent_pose=Pose3D(0.0, 0.0, 0.0, yaw=0.0),
        objects=(
            ObjectObservation(
                "mug_1",
                "mug",
                Pose3D(-0.4, 1.0, 0.78),
                BBox3D(center=Pose3D(-0.4, 1.0, 0.78), size=(0.12, 0.12, 0.16)),
                confidence=0.95,
                visible=True,
            ),
        ),
    ),
    infer_relations=("NEAR",),
)

payload = graph_to_json(graph)
graph_digest = graph_json_digest(graph)
graph_counts = graph_summary(graph)
restored = graph_from_json(payload)
save_graph_json(graph, "tabletop-scene.json")
restored_from_path = load_graph_json("tabletop-scene.json")
graph_fixture_comparison = compare_graph_to_fixture(restored_from_path, "tabletop")
graph_file_comparison = compare_graph_file_to_fixture("tabletop-scene.json", "tabletop")
suite = run_evaluation_suite()
suite_digest = suite["digest"]
suite_report = evaluation_report(suite)
suite_report_json = evaluation_report_json(suite_report)
suite_manifest = evaluation_manifest(tags=("qa", "relations"))
suite_manifest_json = evaluation_manifest_json(suite_manifest)
manifest_path = save_evaluation_manifest(
    "relations-manifest.json",
    tags=("qa", "relations"),
)
loaded_manifest = load_evaluation_manifest(manifest_path)
manifest_validation = validate_evaluation_manifest(loaded_manifest)
manifest_comparison = compare_evaluation_manifest(loaded_manifest)
suite_bundle = evaluation_bundle(tags=("qa", "reobserve"))
suite_bundle_json = evaluation_bundle_json(suite_bundle)
bundle_validation = validate_evaluation_bundle(suite_bundle)
named_suite = run_evaluation_suite(
    names=("tabletop_object_location", "moved_mug_recent_events")
)
report_path = save_evaluation_report("evaluation-report.json", named_suite)
loaded_report = load_evaluation_report(report_path)
report_comparison = compare_evaluation_report(loaded_report)
bundle_path = save_evaluation_bundle("reobserve-bundle.json", tags=("qa", "reobserve"))
loaded_bundle = load_evaluation_bundle(bundle_path)
bundle_comparison = compare_evaluation_bundle(loaded_bundle)
dynamic_qa_suite = run_evaluation_suite(tags=("qa", "dynamic"))
relation_shift_suite = run_evaluation_suite(names=("relation_shift_relation_timeline",))
action_validity_suite = run_evaluation_suite(tags=("qa", "action_validity"))
world_state_suite = run_evaluation_suite(tags=("qa", "world_state"))
temporal_qa_suite = run_evaluation_suite(tags=("qa", "temporal"))
retrieval_qa_suite = run_evaluation_suite(tags=("qa", "retrieval"))
nearest_intent_suite = run_evaluation_suite(question_types=("nearest_object",))
label_candidates_suite = run_evaluation_suite(tags=("qa", "label", "ambiguity"))
retrieve_subgraph_suite = run_evaluation_suite(question_types=("retrieve_subgraph",))
snapshot_qa_suite = run_evaluation_suite(tags=("qa", "snapshot"))
reobserve_suite = run_evaluation_suite(tags=("qa", "reobserve"))
vla_dynamic_suite = run_evaluation_suite(tags=("vla", "dynamic"))
vla_ambiguity_suite = run_evaluation_suite(tags=("vla", "label", "ambiguity"))
vla_reobserve_suite = run_evaluation_suite(tags=("vla", "reobserve"))
vla_anchor_manifest = list_evaluation_case_metadata(
    kinds=("vla_pick", "vla_place_relative")
)
named_manifest = list_evaluation_case_metadata(
    names=("tabletop_mug_pick", "tabletop_object_location")
)

def build_custom_scene():
    custom = DynamicSceneGraph()
    custom.set_agent_pose("agent", Pose3D(0.0, 0.0, 0.0), step=1)
    custom.upsert_object(
        "cube_1",
        "cube",
        Pose3D(0.25, -0.5, 0.4),
        BBox3D(center=Pose3D(0.25, -0.5, 0.4), size=(0.2, 0.2, 0.2)),
        confidence=0.82,
        visible=True,
        step=1,
    )
    return custom

custom_suite = run_evaluation_cases(
    (
        EvaluationCase(
            name="custom_cube_location",
            scene_fixture="custom_counter",
            kind="qa",
            tags=("qa", "custom"),
            question={"type": "object_location", "object_id": "cube_1"},
            expected={"answer": {"object_id": "cube_1"}, "error": None},
        ),
    ),
    scene_loaders={"custom_counter": build_custom_scene},
)
custom_named_suite = run_evaluation_cases(
    (
        EvaluationCase(
            name="custom_cube_location",
            scene_fixture="custom_counter",
            kind="qa",
            tags=("qa", "custom"),
            question={"type": "object_location", "object_id": "cube_1"},
            expected={"answer": {"object_id": "cube_1"}, "error": None},
        ),
    ),
    names=("custom_cube_location",),
    scene_loaders={"custom_counter": build_custom_scene},
)
custom_manifest = evaluation_cases_metadata(
    (
        EvaluationCase(
            name="custom_cube_location",
            scene_fixture="custom_counter",
            kind="qa",
            tags=("qa", "custom"),
            question={"type": "object_location", "object_id": "cube_1"},
            expected={"answer": {"object_id": "cube_1"}, "error": None},
        ),
    ),
    tags=("qa",),
)
```

The fixture registry currently includes `"ambiguous_mugs"`, `"ambiguous_plates"`,
`"tabletop"`, `"moved_mug"`, `"multi_room_rearrangement"`,
`"needs_reobserve"`, and `"relation_shift"` scenes.
`list_scene_fixture_metadata()` returns deterministic scene fixture metadata
with each fixture's name, description, and tags, and supports tag filters without
loading scene graphs. `scene_fixture_manifest()` wraps that metadata in a
standalone manifest with schema version, filters, fixture count, and a stable
SHA-256 digest for fixture-only handoffs.
Each call to `load_scene_fixture()` returns a fresh in-memory graph.
The evaluation suite returns deterministic dictionaries with a top-level
`digest` for experiment records, pass/fail summary, including stable
`selected_cases` and `failed_cases` lists, per-case
actual/expected outputs, per-case tags, and a `breakdown` grouped by case kind,
QA question type, tag, and scene fixture for experiment triage.
The digest is a SHA-256 hash of the suite summary, breakdown, and results, so a
fixed case slice can be compared across deterministic runs without relying on
wall-clock data.
Each result includes deterministic `mismatches` with path, reason, category,
expected, and actual fields, so failed regression cases are reproducible and
inspectable.
QA and VLA actual outputs with a non-empty `error` also include stable
`error_category` values such as `missing_object`, `invalid_time_window`,
`ambiguous_label`, `needs_reobserve`, and `unsupported_relation` for
machine-comparable runtime diagnostics.
Suites and compact reports also include deterministic `runtime_error_categories`
aggregates with category counts and affected case names, so an experiment can
compare its runtime error mix without reprocessing every result.
Report comparison surfaces drift in those aggregates with stable category
paths such as `missing_object.count`.
Compact reports preserve failed-case `mismatch_paths`, report-level
`failure_paths`, raw `failure_reasons`, and stable `failure_categories` such as
`missing_output`, `cardinality_mismatch`, `schema_mismatch`, and
`value_mismatch` for benchmark-level triage.
Report comparison surfaces failure-reason, failure-category, and failure-path
drift with stable paths such as `value_mismatch` and `answer.visible`.
Their `metrics` include `case_count`, `passed_case_count`, `failed_case_count`,
`pass_rate`, and `failure_rate` for direct experiment comparison, plus grouped
count/rate metrics under `by_kind`, `by_question_type`, `by_scene_fixture`, and
`by_tag`.
Their `evidence_metrics` summarize evidence-node counts, evidence-edge counts,
VLA command evidence counts, evidence-covered cases, and average evidence items
per case, with the same grouped views for offline explainability audits.
Compact reports also include `case_digests`, a per-case SHA-256 digest summary
with case name, kind, question type, scene fixture, and pass status so a changed
suite digest can be narrowed to individual deterministic cases.
Evaluation cases can be filtered by tags, for example
`run_evaluation_suite(tags=("qa", "dynamic"))` selects dynamic QA regression
cases including stale next-action validity, explicit object timelines, scene
deltas, multi-room rearrangement event windows, dynamic relation shifts, and
current world state audits, while
`run_evaluation_suite(tags=("qa", "action_validity"))` selects stale-action QA
regressions that derive an old action from a deterministic baseline scene.
`run_evaluation_suite(tags=("qa", "world_state"))` selects current-scene
world-state regressions for dynamic spatial state checks.
`run_evaluation_suite(tags=("qa", "foundation"))` selects basic QA contract
regressions for agent location, missing-object error diagnostics, object
status, object history, and direct relative-relation checks over the tabletop
fixture.
`run_evaluation_suite(tags=("qa", "error"))` selects structured QA error-path
regressions, including missing-object and reversed time-window diagnostics, that
should remain deterministic and local.
`run_evaluation_suite(tags=("qa", "temporal"))` selects timeline and delta
regressions, including relation timelines, for explicit-step memory audits.
`run_evaluation_suite(tags=("qa", "relations"))` selects static and dynamic
relation regressions, including direct relative checks and the `relation_shift`
fixture.
`run_evaluation_suite(tags=("qa", "retrieval"))` selects structured graph
retrieval regressions for the `graph_query` QA path and candidate-constrained
nearest-object QA, plus text-seeded `retrieve_subgraph` QA.
`run_evaluation_suite(tags=("qa", "snapshot"))` selects explicit-step scene
snapshot regressions for state reconstruction audits.
`run_evaluation_suite(tags=("qa", "reobserve"))` selects re-observation target
regressions over the deterministic `needs_reobserve` scene.
`run_evaluation_suite(tags=("qa", "label", "ambiguity"))` selects direct QA
candidate-list regressions for same-label object ambiguity over the deterministic
`ambiguous_mugs` scene.
`run_evaluation_suite(tags=("vla", "anchor"))` selects deterministic VLA pick
and place-relative anchor regressions. They can also be filtered by kind, for
example `run_evaluation_suite(kinds=("vla_pick", "vla_place_relative"))` runs
only anchor-planner cases.
`run_evaluation_suite(tags=("vla", "dynamic"))` selects stale VLA action
regressions, including stale pick targets and stale place-relative reference
anchors.
`run_evaluation_suite(tags=("vla", "reobserve"))` selects VLA planner
regressions where low-confidence invisible targets must return
`needs_reobserve` without emitting a command.
`run_evaluation_suite(tags=("vla", "error"))` selects VLA planner error-path
regressions, including missing pick targets, missing place references, and
unsupported place relations that must return structured `error` results without
emitting a command.
`run_evaluation_suite(tags=("vla", "label", "ambiguity"))` selects semantic
target and reference ambiguity regressions where the planner must return
`ambiguous` instead of choosing among same-label objects, with deterministic
candidate details for offline diagnosis. QA intents can be filtered directly with
`question_types`, for example
`run_evaluation_suite(question_types=("nearest_object",))` or
`run_evaluation_suite(question_types=("label_candidates",))` or
`run_evaluation_suite(question_types=("retrieve_subgraph",))` or
`run_evaluation_suite(question_types=("next_action_validity",))` or
`run_evaluation_suite(question_types=("relative_relation",))`.
Tag, kind, and question-type filters can be combined.
Use `names=` on `list_evaluation_cases()`, `list_evaluation_case_metadata()`,
`run_evaluation_suite()`, `run_evaluation_cases()`, or
`evaluation_cases_metadata()` to select exact built-in or custom cases while
preserving the caller-supplied order. This is useful for publishing a fixed
regression slice as part of an experiment record.
Callers can also pass temporary `EvaluationCase` objects to
`run_evaluation_cases()` to run local experiment cases without modifying the
built-in registry. Custom cases may use caller-supplied `scene_loaders`, so local
experiments can provide deterministic in-memory scenes outside the built-in
fixture registry.
`list_evaluation_case_metadata()` returns a deterministic manifest with each
case's name, scene fixture, fixture description/tags when available, kind, tags,
structured question copy, question type, baseline fixture metadata for
stale-action cases, action target fields, relation, and expected top-level keys.
It supports the same tag and kind filters as `list_evaluation_cases()`, and does
not load scenes or execute QA/VLA logic. Both functions also support
`question_types` filters for deterministic QA intent discovery. Caller-supplied
fixture names that are not in the built-in scene registry keep
`scene_description=None` and `scene_tags=[]`.
Use `evaluation_cases_metadata()` for the same deterministic manifest over
caller-supplied custom `EvaluationCase` sequences.
For VLA pick and place-relative evaluation cases, callers may provide explicit
object ids (`target_object`, `reference_object`) or semantic labels
(`target_label`, `reference_label`) for deterministic target resolution.
Label-based cases preserve planner outcomes such as `ambiguous` when multiple
visible objects share the same label; ambiguous results include stable candidate
details with pose, visibility, confidence, last-seen step, and re-observation
status. The built-in `ambiguous_mugs` and `ambiguous_plates` fixtures provide
deterministic same-label scenes for those regressions.

`SpatialQAEngine` supports direct structured intents for agent location, agent
history, agent timeline, object location, object status, relative relations,
nearest object, label candidates, object history, object timeline, relation timeline,
re-observation targets, action validity, explicit-step scene snapshots, world
state, and recent events. It also exposes deterministic `graph_query`
answers backed by `GraphTool.query_graph()` so experiments can retrieve
structured graph nodes and edges through the QA layer, plus `retrieve_subgraph`
for text-seeded graph retrieval backed by `GraphTool.retrieve_subgraph()`.
Object location exposes pose, visibility, confidence, state step, current
containment location, and containment plus `STATE_CHANGED` evidence. Object
room QA resolves the current room-level containment path when one exists,
returning room id/label, path nodes, and evidence edge IDs for multi-room
audits. Object status exposes visibility, confidence, last-seen pose/step, and
whether the target needs re-observation. Nearest-object QA can optionally constrain the
deterministic search with a caller-supplied `candidates` list, and returns the
selected distance plus stable candidate distance diagnostics for offline
triage. Label-candidate QA returns stable object-id ordered candidates for a
semantic label, marks
same-label ambiguity, and includes per-object `STATE_CHANGED` evidence. Object
timeline returns each explicit object state step with
pose, visibility, confidence, last-seen memory, current containment location, and
per-step evidence edges. Relation timeline returns deterministic relation-edge
records filtered by source, relation, destination, reference frame, and explicit
step windows. Agent timeline returns each explicit agent pose step with per-step
`STATE_CHANGED` evidence edges. Re-observation targets list current invisible
low-confidence objects, optionally filtered by label, without requiring an agent
pose or external model.

`GraphTool.scene_snapshot(step=...)` and the matching QA intent reconstruct the
latest agent pose, object states, and containment locations at or before an
explicit step. `GraphTool.scene_delta(from_step=..., to_step=...)` compares two
snapshots and reports changed agent pose, changed objects, changed fields, and
window evidence. `GraphTool.world_state(visible=...)` returns the current agent
pose, current object states, current containment locations, and evidence IDs.
`GraphTool.current_room(object_id)` resolves the object's latest containment
path to a room when possible, returning the room id, room label, containment
path, and evidence edge IDs for local multi-room audits.
`GraphTool.recent_events(since_step=..., until_step=...)`
returns action/event nodes, step-window change edges, and evidence IDs for
auditing what just happened. This makes temporal regression cases deterministic
without clocks or simulators.

Agent poses are stored as current state plus deterministic `AgentPoseState`
history. Each `set_agent_pose(..., step=...)` call creates a `state:<agent>:<step>`
node and `STATE_CHANGED` edge, so graph export/import preserves where the agent
was across explicit steps.

Spatial relations are computed geometrically. `NEAR` uses 3D bbox surface
distance, `ON` requires vertical contact plus configurable support-area overlap,
and egocentric relations respect the agent yaw. `GraphTool.update_spatial_relations`
can append inferred relation edges for an explicit caller-supplied step while
preserving previous relation history.

`ObservationIngestor` is the deterministic boundary for mock perception frames:
callers provide explicit `SceneObservation.step`, optional agent pose, room/region
nodes, object observations, and optional relation inference settings. It writes
only in-memory graph state and never calls models, clocks, random sources, or
network services.

Planner results include deterministic `details` for ambiguity and replan causes,
including same-label candidate state, expected/current pose, last-seen steps,
current location, and evidence edge IDs when an action becomes stale.
`place_relative` validation treats the target pose as the planned anchor, not the
target object's current pose, and returns `stale_reference_state` when the
reference object moves enough to invalidate that anchor.
`VLAAnchorPlanner.plan_place_relative()` also accepts `target_label` and
`reference_label` keyword arguments so semantic place requests can resolve to
object ids without invoking any model.

All built-in scene and graph IO helpers are deterministic and avoid network,
current-time, random, model, robot, and simulator dependencies.
