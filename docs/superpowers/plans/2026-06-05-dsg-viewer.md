# DSG Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only local DSG inspection workbench that serves a browser UI and linked JSON payload for graph, QA, eval, and evidence artifacts.

**Architecture:** Add a dependency-light viewer module under `src/dsg_spatialqa_lab/visualization/` that normalizes local artifacts into one payload. Add `scripts/serve_dsg_viewer.py` to serve a static HTML/JavaScript app and `/payload.json` from localhost. The implementation is deterministic and local-only, with explicit workspace path checks.

**Tech Stack:** Python standard library HTTP server, existing DSG-SpatialQA JSON loaders, static HTML/CSS/JavaScript, pytest, mypy, ruff.

---

### Task 1: Viewer Payload Builder

**Files:**
- Create: `src/dsg_spatialqa_lab/visualization/dsg_viewer.py`
- Modify: `src/dsg_spatialqa_lab/visualization/__init__.py`
- Modify: `src/dsg_spatialqa_lab/__init__.py`
- Test: `tests/test_dsg_viewer.py`

- [ ] **Step 1: Write the failing payload test**

Add this test to `tests/test_dsg_viewer.py`:

```python
from __future__ import annotations

import dsg_spatialqa_lab as lab


def _single_object_graph() -> lab.DynamicSceneGraph:
    graph = lab.DynamicSceneGraph()
    graph.set_agent_pose("agent", lab.Pose3D(0.0, 0.0, 0.0), step=1)
    graph.add_room("room_1", "kitchen", step=1)
    graph.upsert_object(
        "mug_1",
        "mug",
        lab.Pose3D(1.0, 0.8, 2.0),
        lab.BBox3D(center=lab.Pose3D(1.0, 0.8, 2.0), size=(0.2, 0.2, 0.2)),
        confidence=0.9,
        visible=True,
        step=1,
        attributes={"source_kind": "detector", "evidence_kinds": ["rgb", "depth", "detector"]},
    )
    graph.add_edge(
        "mug_1",
        "IN_ROOM",
        "room_1",
        "world",
        0.9,
        step=1,
        evidence=["frame:1"],
        attributes={"source": "test"},
    )
    return graph


def test_dsg_viewer_payload_exposes_graph_metrics_and_selection_indexes() -> None:
    assert hasattr(lab, "dsg_viewer_payload")
    graph = _single_object_graph()
    evidence_report = {
        "readiness": {"ready": False, "failed_check_count": 1, "failed_checks": ["non_real_sources_absent"]},
        "evidence_summary": {"source_counts": {"ai2thor": 1}, "evidence_kind_counts": {"rgb": 1, "depth": 1, "detector": 1}},
        "report_digest": "e" * 64,
    }

    payload = lab.dsg_viewer_payload(
        predicted_graph=graph,
        predicted_graph_path="predicted-graph.json",
        evidence_report=evidence_report,
        evidence_report_path="evidence.json",
    )

    assert payload["schema_version"] == "dsg-spatialqa-lab.dsg-viewer-payload.v1"
    assert payload["artifacts"]["predicted_graph_path"] == "predicted-graph.json"
    assert payload["graph"]["summary"]["object_count"] == 1
    assert payload["graph"]["nodes"][0]["id"] == "agent"
    assert payload["graph"]["edges"][0]["relation"] == "IN_ROOM"
    assert payload["metrics"]["object_count"] == 1
    assert payload["metrics"]["edge_count"] == 3
    assert payload["metrics"]["evidence_ready"] is False
    assert payload["diagnostics"]["failed_checks"] == ["non_real_sources_absent"]
    assert payload["indexes"]["qa_case_ids_by_object_id"] == {}
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
python -m pytest tests/test_dsg_viewer.py::test_dsg_viewer_payload_exposes_graph_metrics_and_selection_indexes -q
```

Expected: FAIL because `dsg_viewer_payload` is not exported.

- [ ] **Step 3: Implement minimal payload builder**

Create `src/dsg_spatialqa_lab/visualization/dsg_viewer.py` with `DSG_VIEWER_PAYLOAD_SCHEMA_VERSION`, `dsg_viewer_payload`, `dsg_viewer_payload_digest`, `dsg_viewer_payload_json`, `save_dsg_viewer_payload`, and `load_dsg_viewer_payload`.

The payload builder must:

- Convert nodes and edges from `graph.nodes` and `graph.edges`.
- Include `graph_summary(predicted_graph)`.
- Copy evidence readiness into `metrics.evidence_ready` and `diagnostics.failed_checks`.
- Build empty QA indexes when no QA cases are provided.
- Compute a stable `payload_digest`.

Export these symbols from `src/dsg_spatialqa_lab/visualization/__init__.py` and `src/dsg_spatialqa_lab/__init__.py`.

- [ ] **Step 4: Run the payload test**

Run:

```bash
python -m pytest tests/test_dsg_viewer.py::test_dsg_viewer_payload_exposes_graph_metrics_and_selection_indexes -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/dsg_spatialqa_lab/visualization/dsg_viewer.py src/dsg_spatialqa_lab/visualization/__init__.py src/dsg_spatialqa_lab/__init__.py tests/test_dsg_viewer.py
git commit -m "add dsg viewer payload builder"
```

### Task 2: QA, Eval, Oracle, And Report Linkage

**Files:**
- Modify: `src/dsg_spatialqa_lab/visualization/dsg_viewer.py`
- Test: `tests/test_dsg_viewer.py`

- [ ] **Step 1: Write the failing linkage test**

Add:

```python
def test_dsg_viewer_payload_links_qa_cases_and_oracle_delta() -> None:
    graph = _single_object_graph()
    case = lab.QACase(
        id="case-1",
        scene_id="FloorPlan1",
        episode_id="episode-1",
        graph_digest="graph-digest",
        step=1,
        question={"type": "object_location", "object_id": "mug_1"},
        question_type="object_location",
        answer={"object_id": "mug_1"},
        answer_type="object_location",
    )
    qa_eval_report = {
        "cases": [
            {
                "case_id": "case-1",
                "exact_match": False,
                "prediction": {"object_id": "mug_1"},
                "gold": {"object_id": "mug_1"},
            }
        ]
    }
    graph_eval_report = {
        "comparison": {
            "object_matches": [{"oracle_object_id": "mug_1", "predicted_object_id": "mug_1"}],
            "missing_relations": [{"src": "mug_1", "relation": "ON", "dst": "table_1"}],
            "extra_relations": [],
        },
        "summary": {
            "object_recall": {"rate": 1.0},
            "relation_precision": {"rate": 0.5},
            "relation_recall": {"rate": 0.25},
            "relation_f1": 0.333333,
        },
    }

    payload = lab.dsg_viewer_payload(
        predicted_graph=graph,
        qa_cases=(case,),
        qa_eval_report=qa_eval_report,
        graph_eval_report=graph_eval_report,
    )

    assert payload["qa"]["cases"][0]["case_id"] == "case-1"
    assert payload["qa"]["cases"][0]["target_object_ids"] == ["mug_1"]
    assert payload["indexes"]["qa_case_ids_by_object_id"] == {"mug_1": ["case-1"]}
    assert payload["oracle"]["object_matches_by_predicted_id"] == {
        "mug_1": {"oracle_object_id": "mug_1", "predicted_object_id": "mug_1"}
    }
    assert payload["oracle"]["missing_relation_count"] == 1
    assert payload["metrics"]["relation_f1"] == 0.333333
```

- [ ] **Step 2: Run the failing linkage test**

Run:

```bash
python -m pytest tests/test_dsg_viewer.py::test_dsg_viewer_payload_links_qa_cases_and_oracle_delta -q
```

Expected: FAIL because QA and oracle linkage is not implemented.

- [ ] **Step 3: Implement linkage**

Update `dsg_viewer_payload` to:

- Accept optional `qa_cases`, `qa_eval_report`, `graph_eval_report`, and `oracle_graph`.
- Extract target object ids from object-location QA question fields.
- Join QA eval rows by `case_id`.
- Build `qa_case_ids_by_object_id`.
- Normalize graph eval object matches and relation metrics.
- Preserve missing and extra relation counts.

- [ ] **Step 4: Run viewer tests**

Run:

```bash
python -m pytest tests/test_dsg_viewer.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add src/dsg_spatialqa_lab/visualization/dsg_viewer.py tests/test_dsg_viewer.py
git commit -m "link dsg viewer qa and graph eval data"
```

### Task 3: Workspace Preset And Path Safety

**Files:**
- Modify: `src/dsg_spatialqa_lab/visualization/dsg_viewer.py`
- Test: `tests/test_dsg_viewer.py`

- [ ] **Step 1: Write failing workspace tests**

Add:

```python
def test_dsg_viewer_workspace_preset_resolves_known_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "ai2thor-real-small"
    (workspace / "outputs" / "predicted-dsg").mkdir(parents=True)
    (workspace / "outputs" / "benchmark" / "graphs").mkdir(parents=True)
    (workspace / "inputs").mkdir()
    (workspace / "outputs" / "offline-controls" / "qa-eval-observation-aware-p4-target60").mkdir(parents=True)
    expected_graph = workspace / "outputs" / "predicted-dsg" / "predicted-graph.json"
    expected_graph.write_text("{}", encoding="utf-8")

    preset = lab.dsg_viewer_workspace_preset(workspace)

    assert preset["workspace_path"] == str(workspace)
    assert preset["paths"]["predicted_graph_path"] == str(expected_graph)


def test_dsg_viewer_path_guard_rejects_paths_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    try:
        lab.dsg_viewer_resolve_workspace_path(workspace, outside)
    except lab.SpatialQAError as exc:
        assert "outside workspace" in str(exc)
    else:
        raise AssertionError("expected outside workspace path to be rejected")
```

- [ ] **Step 2: Run failing workspace tests**

Run:

```bash
python -m pytest tests/test_dsg_viewer.py::test_dsg_viewer_workspace_preset_resolves_known_paths tests/test_dsg_viewer.py::test_dsg_viewer_path_guard_rejects_paths_outside_workspace -q
```

Expected: FAIL because preset/path guard functions are not exported.

- [ ] **Step 3: Implement workspace helpers**

Add and export:

- `dsg_viewer_resolve_workspace_path(workspace, path)`.
- `dsg_viewer_workspace_preset(workspace)`.

The path guard must use `Path.resolve()` and reject any resolved path outside the resolved workspace. The preset must return known project paths only when files exist.

- [ ] **Step 4: Run workspace tests**

Run:

```bash
python -m pytest tests/test_dsg_viewer.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add src/dsg_spatialqa_lab/visualization/dsg_viewer.py src/dsg_spatialqa_lab/visualization/__init__.py src/dsg_spatialqa_lab/__init__.py tests/test_dsg_viewer.py
git commit -m "add dsg viewer workspace path safety"
```

### Task 4: Local Server CLI

**Files:**
- Create: `scripts/serve_dsg_viewer.py`
- Test: `tests/test_dsg_viewer.py`

- [ ] **Step 1: Write failing CLI tests**

Add:

```python
import importlib.util
import json
import sys
from types import ModuleType
from typing import Any, Protocol, cast


ROOT = Path(__file__).resolve().parents[1]
SERVE_DSG_VIEWER_SCRIPT = ROOT / "scripts" / "serve_dsg_viewer.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_serve_dsg_viewer_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("serve_dsg_viewer_script", SERVE_DSG_VIEWER_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_serve_dsg_viewer_cli_writes_payload_without_starting_server(tmp_path: Path, capsys: object) -> None:
    module = load_serve_dsg_viewer_script()
    main = cast(MainFn, getattr(module, "main"))
    graph_path = tmp_path / "predicted-graph.json"
    lab.save_graph_json(_single_object_graph(), graph_path)
    output_path = tmp_path / "payload.json"

    assert main(["--graph", str(graph_path), "--write-payload", str(output_path)]) == 0

    output = json.loads(cast(Any, capsys).readouterr().out)
    payload = lab.load_dsg_viewer_payload(output_path)
    assert output["action"] == "write_dsg_viewer_payload"
    assert output["payload_path"] == str(output_path)
    assert payload["metrics"]["object_count"] == 1
```

- [ ] **Step 2: Run failing CLI test**

Run:

```bash
python -m pytest tests/test_dsg_viewer.py::test_serve_dsg_viewer_cli_writes_payload_without_starting_server -q
```

Expected: FAIL because `scripts/serve_dsg_viewer.py` does not exist.

- [ ] **Step 3: Implement CLI**

Create `scripts/serve_dsg_viewer.py` with:

- `--workspace`
- `--graph`
- `--oracle-graph`
- `--qa`
- `--qa-eval-report`
- `--graph-eval-report`
- `--evidence-report`
- `--write-payload`
- `--host`, default `127.0.0.1`
- `--port`, default `8765`

If `--write-payload` is supplied, write payload JSON and exit. If not, serve viewer HTML and `/payload.json`.

- [ ] **Step 4: Run CLI tests**

Run:

```bash
python -m pytest tests/test_dsg_viewer.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add scripts/serve_dsg_viewer.py tests/test_dsg_viewer.py
git commit -m "add local dsg viewer server"
```

### Task 5: Static Viewer UI And Verification

**Files:**
- Create: `src/dsg_spatialqa_lab/visualization/static/dsg_viewer.html`
- Modify: `scripts/serve_dsg_viewer.py`
- Test: `tests/test_dsg_viewer.py`

- [ ] **Step 1: Write failing static UI test**

Add:

```python
def test_dsg_viewer_static_html_contains_workbench_regions() -> None:
    html = lab.dsg_viewer_html()

    assert "DSG Viewer" in html
    assert 'id="data-rail"' in html
    assert 'id="graph-canvas"' in html
    assert 'id="detail-rail"' in html
    assert "Debug" in html
    assert "Demo" in html
    assert "Analysis" in html
    assert "fetch('payload.json')" in html
```

- [ ] **Step 2: Run failing static UI test**

Run:

```bash
python -m pytest tests/test_dsg_viewer.py::test_dsg_viewer_static_html_contains_workbench_regions -q
```

Expected: FAIL because `dsg_viewer_html` is not exported.

- [ ] **Step 3: Implement static viewer HTML**

Add `dsg_viewer_html()` to `dsg_viewer.py` and export it. Store the HTML at `src/dsg_spatialqa_lab/visualization/static/dsg_viewer.html`.

The UI must include:

- Left `data-rail`.
- Center `graph-canvas`.
- Right `detail-rail`.
- Debug, Demo, Analysis tabs.
- Metrics cards.
- Search/filter controls.
- JavaScript that fetches `payload.json`, renders nodes/edges as SVG, and updates detail panels on click.

- [ ] **Step 4: Run targeted and package tests**

Run:

```bash
python -m pytest tests/test_dsg_viewer.py -q
python -m ruff check src/dsg_spatialqa_lab/visualization/dsg_viewer.py scripts/serve_dsg_viewer.py tests/test_dsg_viewer.py
python -m mypy src tests scripts
```

Expected: all pass.

- [ ] **Step 5: Run full verification**

Run:

```bash
python scripts/verify.py
```

Expected: all checks pass.

- [ ] **Step 6: Commit Task 5**

Run:

```bash
git add src/dsg_spatialqa_lab/visualization/static/dsg_viewer.html src/dsg_spatialqa_lab/visualization/dsg_viewer.py src/dsg_spatialqa_lab/visualization/__init__.py src/dsg_spatialqa_lab/__init__.py scripts/serve_dsg_viewer.py tests/test_dsg_viewer.py
git commit -m "add dsg viewer static workbench"
```

---

## Self-Review

Spec coverage:

- Local server: Task 4.
- Read-only local artifact loading: Tasks 1, 3, 4.
- Workspace path safety: Task 3.
- Predicted/oracle graph, QA/eval/report payload linkage: Tasks 1 and 2.
- Debug/Demo/Analysis UI: Task 5.
- No network, no external AI, no simulator/detector integration: Tasks 3 and 4 keep inputs local and read-only.
- Tests and full verification: all tasks include targeted verification; Task 5 includes full verification.

Placeholder scan: each task contains concrete file paths, commands, and expected outcomes.

Type consistency: public functions use the `dsg_viewer_*` prefix and are exported through visualization and package `__init__` modules.
