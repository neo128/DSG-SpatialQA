from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest
from _pytest.capture import CaptureFixture

from dsg_spatialqa_lab import (
    graph_json_digest,
    load_graph_json,
    load_scene_fixture,
    save_graph_json,
)


ROOT = Path(__file__).resolve().parents[1]
SCENE_SCRIPT = ROOT / "scripts" / "scene.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_scene_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("scene_script", SCENE_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def expected_multi_room_fixture_manifest() -> dict[str, object]:
    payload_without_digest: dict[str, object] = {
        "schema_version": "dsg-spatialqa-lab.scene-fixture-manifest.v1",
        "filters": {
            "tags": ["multi_room"],
        },
        "fixture_count": 1,
        "scene_fixtures": [
            {
                "name": "multi_room_rearrangement",
                "description": (
                    "Dynamic kitchen-to-pantry scene with relocated cereal and an occluded fork."
                ),
                "tags": ["dynamic", "multi_room", "move", "occlusion", "reobserve"],
            },
        ],
    }
    return {
        **payload_without_digest,
        "digest": hashlib.sha256(
            json.dumps(
                payload_without_digest,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }


def test_scene_cli_exports_fixture_to_explicit_graph_json(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_scene_script()
    main = cast(MainFn, getattr(module, "main"))
    graph_path = tmp_path / "fixtures" / "tabletop.json"

    assert main(["--fixture", "tabletop", "--output", str(graph_path)]) == 0

    exported_graph = load_graph_json(graph_path)
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "action": "export_fixture",
        "fixture": "tabletop",
        "path": str(graph_path),
        "digest": graph_json_digest(load_scene_fixture("tabletop")),
        "summary": {
            "schema_version": 1,
            "node_count": 9,
            "edge_count": 8,
            "object_count": 3,
            "agent_count": 1,
            "object_history_count": 3,
            "agent_history_count": 1,
            "visible_object_count": 3,
            "hidden_object_count": 0,
            "low_confidence_object_count": 0,
            "reobserve_candidate_count": 0,
            "unlocated_object_count": 2,
            "unroomed_object_count": 3,
            "by_node_type": {
                "agent": 1,
                "object": 3,
                "room": 1,
                "state": 4,
            },
            "by_edge_relation": {
                "LEFT_OF": 1,
                "NEAR": 1,
                "ON": 1,
                "RIGHT_OF": 1,
                "STATE_CHANGED": 4,
            },
            "by_object_label": {
                "mug": 1,
                "plate": 1,
                "table": 1,
            },
            "by_current_location": {
                "table_1": 1,
            },
            "by_current_room": {},
        },
    }
    assert graph_json_digest(exported_graph) == report["digest"]


def test_scene_cli_validates_explicit_graph_json_file(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_scene_script()
    main = cast(MainFn, getattr(module, "main"))
    graph_path = tmp_path / "tabletop.json"

    assert main(["--fixture", "tabletop", "--output", str(graph_path)]) == 0
    capsys.readouterr()

    assert main(["--validate", str(graph_path)]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["action"] == "validate_graph"
    assert report["path"] == str(graph_path)
    assert report["valid"] is True
    assert report["digest"] == graph_json_digest(load_scene_fixture("tabletop"))
    assert report["summary"]["node_count"] == 9


def test_scene_cli_reports_invalid_graph_json_for_validation(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_scene_script()
    main = cast(MainFn, getattr(module, "main"))
    graph_path = tmp_path / "invalid.json"
    graph_path.write_text('{"schema_version": 999}\n', encoding="utf-8")

    try:
        result = main(["--validate", str(graph_path)])
    except Exception as exc:
        pytest.fail(f"expected structured validation report, got {type(exc).__name__}: {exc}")

    assert result == 1
    assert json.loads(capsys.readouterr().out) == {
        "action": "validate_graph",
        "path": str(graph_path),
        "valid": False,
        "error": "Unsupported scene schema version: 999",
    }


def test_scene_cli_lists_filtered_fixture_metadata_without_loading_graphs(
    capsys: CaptureFixture[str],
) -> None:
    module = load_scene_script()
    main = cast(MainFn, getattr(module, "main"))

    def fail_load_scene_fixture(name: str) -> object:
        raise AssertionError(f"unexpected graph load for {name}")

    setattr(module, "load_scene_fixture", fail_load_scene_fixture)

    assert main(["--list-fixtures", "--tag", "multi_room"]) == 0

    assert json.loads(capsys.readouterr().out) == expected_multi_room_fixture_manifest()


def test_scene_cli_writes_filtered_fixture_metadata_to_explicit_output(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_scene_script()
    main = cast(MainFn, getattr(module, "main"))
    manifest_path = tmp_path / "manifests" / "multi-room-fixtures.json"

    assert (
        main(
            [
                "--list-fixtures",
                "--tag",
                "multi_room",
                "--output",
                str(manifest_path),
            ]
        )
        == 0
    )

    expected = expected_multi_room_fixture_manifest()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == expected
    assert json.loads(capsys.readouterr().out) == expected


def test_scene_cli_compares_explicit_graph_json_to_current_fixture(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_scene_script()
    main = cast(MainFn, getattr(module, "main"))
    graph_path = tmp_path / "tabletop.json"

    assert main(["--fixture", "tabletop", "--output", str(graph_path)]) == 0
    capsys.readouterr()

    assert main(["--compare-fixture", "tabletop", "--input", str(graph_path)]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["action"] == "compare_fixture"
    assert report["fixture"] == "tabletop"
    assert report["path"] == str(graph_path)
    assert report["valid"] is True
    assert report["matches"] is True
    assert report["graph_digest"] == report["fixture_digest"]
    assert [check["passed"] for check in report["checks"]] == [True, True]


def test_scene_cli_reports_invalid_graph_json_for_fixture_compare(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_scene_script()
    main = cast(MainFn, getattr(module, "main"))
    graph_path = tmp_path / "invalid.json"
    graph_path.write_text('{"schema_version": 999}\n', encoding="utf-8")

    try:
        result = main(["--compare-fixture", "tabletop", "--input", str(graph_path)])
    except Exception as exc:
        pytest.fail(f"expected structured compare report, got {type(exc).__name__}: {exc}")

    assert result == 1
    assert json.loads(capsys.readouterr().out) == {
        "action": "compare_fixture",
        "fixture": "tabletop",
        "path": str(graph_path),
        "valid": False,
        "matches": False,
        "error": "Unsupported scene schema version: 999",
    }


def test_scene_cli_returns_nonzero_when_graph_json_drifts_from_fixture(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_scene_script()
    main = cast(MainFn, getattr(module, "main"))
    graph_path = tmp_path / "tabletop-drift.json"
    graph = load_scene_fixture("tabletop")
    graph.add_region("drift_region", "Drift Region", step=9)
    save_graph_json(graph, graph_path)

    assert main(["--compare-fixture", "tabletop", "--input", str(graph_path)]) == 1

    report = json.loads(capsys.readouterr().out)
    assert report["action"] == "compare_fixture"
    assert report["matches"] is False
    assert report["graph_digest"] != report["fixture_digest"]
    assert report["checks"][1]["actual"]["node_count"] == 10
    assert report["checks"][1]["differences"] == [
        {"path": "by_node_type.region", "expected": None, "actual": 1},
        {"path": "node_count", "expected": 9, "actual": 10},
    ]
