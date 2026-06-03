from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from _pytest.capture import CaptureFixture

from dsg_spatialqa_lab import (
    BBox3D,
    DynamicSceneGraph,
    ObjectObservation,
    Pose3D,
    SceneObservation,
    graph_json_digest,
    graph_report_digest,
    load_graph_json,
    load_scene_observation_sequence,
    observation_ingest_report_digest,
    save_scene_observation_sequence,
    save_graph_json,
    scene_observation_sequence_digest,
    scene_observation_sequence_summary,
)


ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS_SCRIPT = ROOT / "scripts" / "observations.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_observations_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "observations_script",
        OBSERVATIONS_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_observations_cli_summarizes_explicit_sequence_without_graph_ingest(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_observations_script()
    main = cast(MainFn, getattr(module, "main"))
    sequence_path = tmp_path / "observations" / "mock-sequence.json"
    summary_path = tmp_path / "reports" / "mock-sequence-summary.json"
    graph_path = tmp_path / "graphs" / "unused-graph.json"
    observations = (
        SceneObservation(
            step=1,
            agent_pose=Pose3D(0.0, 0.0, 0.0),
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
        SceneObservation(
            step=2,
            objects=(
                ObjectObservation(
                    "spoon_1",
                    "spoon",
                    Pose3D(0.2, 1.2, 0.78),
                    BBox3D(center=Pose3D(0.2, 1.2, 0.78), size=(0.2, 0.04, 0.02)),
                    confidence=0.2,
                    visible=False,
                ),
            ),
        ),
    )
    save_scene_observation_sequence(observations, sequence_path)

    assert (
        main(
            [
                "--summarize-sequence",
                str(sequence_path),
                "--report",
                str(summary_path),
            ]
        )
        == 0
    )

    stdout_summary = json.loads(capsys.readouterr().out)
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary == stdout_summary
    assert stdout_summary == {
        "action": "summarize_observation_sequence",
        "path": str(sequence_path),
        "valid": True,
        **scene_observation_sequence_summary(observations),
    }
    assert not graph_path.exists()


def test_observations_cli_validates_explicit_sequence_without_graph_ingest(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_observations_script()
    main = cast(MainFn, getattr(module, "main"))
    sequence_path = tmp_path / "observations" / "mock-sequence.json"
    validation_path = tmp_path / "reports" / "mock-sequence-validation.json"
    graph_path = tmp_path / "graphs" / "unused-graph.json"
    observations = (
        SceneObservation(
            step=1,
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
    )
    save_scene_observation_sequence(observations, sequence_path)

    assert (
        main(
            [
                "--validate-sequence",
                str(sequence_path),
                "--report",
                str(validation_path),
            ]
        )
        == 0
    )

    stdout_validation = json.loads(capsys.readouterr().out)
    saved_validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert saved_validation == stdout_validation
    assert stdout_validation["action"] == "validate_observation_sequence"
    assert stdout_validation["path"] == str(sequence_path)
    assert stdout_validation["valid"] is True
    assert stdout_validation["digest"] == scene_observation_sequence_digest(observations)
    assert stdout_validation["summary"] == scene_observation_sequence_summary(observations)
    assert [check["passed"] for check in stdout_validation["checks"]] == [True] * 5
    assert not graph_path.exists()


def test_observations_cli_returns_nonzero_for_invalid_sequence_validation(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_observations_script()
    main = cast(MainFn, getattr(module, "main"))
    sequence_path = tmp_path / "invalid-sequence.json"
    sequence_path.write_text('{"schema_version": "invalid"}\n', encoding="utf-8")

    assert main(["--validate-sequence", str(sequence_path)]) == 1

    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_observation_sequence"
    assert validation["path"] == str(sequence_path)
    assert validation["valid"] is False
    assert validation["schema_version"] == "invalid"
    assert validation["digest"] is None
    assert validation["summary"] is None
    assert validation["error"] == "Unsupported scene observation sequence schema version: invalid"


def test_observations_cli_validates_and_compares_sequence_summary(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_observations_script()
    main = cast(MainFn, getattr(module, "main"))
    sequence_path = tmp_path / "observations" / "mock-sequence.json"
    summary_path = tmp_path / "reports" / "mock-sequence-summary.json"
    observations = (
        SceneObservation(
            step=1,
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
    )
    save_scene_observation_sequence(observations, sequence_path)
    assert (
        main(
            [
                "--summarize-sequence",
                str(sequence_path),
                "--report",
                str(summary_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["--validate-sequence-summary", str(summary_path)]) == 0

    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_observation_sequence_summary"
    assert validation["path"] == str(summary_path)
    assert validation["valid"] is True
    assert [check["passed"] for check in validation["checks"]] == [True] * 18

    assert main(["--compare-sequence-summary", str(summary_path)]) == 0

    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_observation_sequence_summary"
    assert comparison["path"] == str(summary_path)
    assert comparison["matches"] is True
    assert comparison["sequence_path"] == str(sequence_path)
    assert comparison["saved_sequence_digest"] == scene_observation_sequence_digest(
        observations
    )
    assert comparison["current_sequence_digest"] == scene_observation_sequence_digest(
        observations
    )
    assert [check["passed"] for check in comparison["checks"]] == [True, True, True]


def test_observations_cli_imports_detector_jsonl_to_sequence(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_observations_script()
    main = cast(MainFn, getattr(module, "main"))
    input_path = tmp_path / "detector" / "rgbd-detections.jsonl"
    sequence_path = tmp_path / "observations" / "detector-sequence.json"
    report_path = tmp_path / "reports" / "detector-import-report.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in _detector_records())
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--import-detector-jsonl",
                str(input_path),
                "--output-sequence",
                str(sequence_path),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    observations = load_scene_observation_sequence(sequence_path)
    assert output == saved_report
    assert output["action"] == "import_detector_observation_jsonl"
    assert output["path"] == str(input_path)
    assert output["output_sequence_path"] == str(sequence_path)
    assert output["valid"] is True
    assert output["sequence_digest"] == scene_observation_sequence_digest(observations)
    assert output["summary"] == scene_observation_sequence_summary(observations)
    assert [observation.step for observation in observations] == [1, 2]
    assert observations[0].objects[0].attributes["source"] == "detector_rgbd"
    assert observations[0].objects[0].attributes["rgb_path"] == "rgb/0001.png"


def test_observations_cli_returns_nonzero_for_sequence_summary_drift(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_observations_script()
    main = cast(MainFn, getattr(module, "main"))
    sequence_path = tmp_path / "observations" / "mock-sequence.json"
    summary_path = tmp_path / "reports" / "mock-sequence-summary.json"
    first_sequence = (
        SceneObservation(
            step=1,
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
    )
    drifted_sequence = (
        SceneObservation(
            step=1,
            objects=(
                ObjectObservation(
                    "mug_1",
                    "mug",
                    Pose3D(-0.4, 1.0, 0.78),
                    BBox3D(center=Pose3D(-0.4, 1.0, 0.78), size=(0.12, 0.12, 0.16)),
                    confidence=0.95,
                    visible=True,
                ),
                ObjectObservation(
                    "cup_1",
                    "cup",
                    Pose3D(0.1, 1.0, 0.78),
                    BBox3D(center=Pose3D(0.1, 1.0, 0.78), size=(0.1, 0.1, 0.12)),
                    confidence=0.9,
                    visible=True,
                ),
            ),
        ),
    )
    save_scene_observation_sequence(first_sequence, sequence_path)
    assert (
        main(
            [
                "--summarize-sequence",
                str(sequence_path),
                "--report",
                str(summary_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    save_scene_observation_sequence(drifted_sequence, sequence_path)

    assert main(["--compare-sequence-summary", str(summary_path)]) == 1

    comparison = json.loads(capsys.readouterr().out)
    checks = {check["name"]: check for check in comparison["checks"]}
    assert comparison["action"] == "compare_observation_sequence_summary"
    assert comparison["matches"] is False
    assert checks["sequence_digest_matches_current"]["passed"] is False
    assert {
        "path": "unique_object_ids",
        "expected": ["mug_1"],
        "actual": ["cup_1", "mug_1"],
    } in checks["sequence_summary_matches_current"]["differences"]


def test_observations_cli_ingests_explicit_sequence_to_graph_json_and_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_observations_script()
    main = cast(MainFn, getattr(module, "main"))
    sequence_path = tmp_path / "observations" / "mock-sequence.json"
    graph_path = tmp_path / "graphs" / "mock-graph.json"
    report_path = tmp_path / "reports" / "mock-ingest-report.json"
    observations = (
        SceneObservation(
            step=1,
            agent_pose=Pose3D(0.0, 0.0, 0.0),
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
        SceneObservation(
            step=2,
            objects=(
                ObjectObservation(
                    "mug_1",
                    "mug",
                    Pose3D(0.2, 1.2, 0.78),
                    BBox3D(center=Pose3D(0.2, 1.2, 0.78), size=(0.12, 0.12, 0.16)),
                    confidence=0.2,
                    visible=False,
                ),
            ),
        ),
    )
    save_scene_observation_sequence(observations, sequence_path)

    assert (
        main(
            [
                "--input",
                str(sequence_path),
                "--output-graph",
                str(graph_path),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )

    graph = load_graph_json(graph_path)
    stdout_report = json.loads(capsys.readouterr().out)
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report == stdout_report
    expected_report = {
        "schema_version": "dsg-spatialqa-lab.observation-ingest-report.v1",
        "action": "ingest_observation_sequence",
        "path": str(sequence_path),
        "graph_path": str(graph_path),
        "valid": True,
        "sequence_digest": scene_observation_sequence_digest(observations),
        "options": {
            "infer_relations": [],
            "reference_frames": ["world"],
        },
        "observation_count": 2,
        "steps": [1, 2],
        "ingest_results": [
            {
                "step": 1,
                "node_ids": ["agent", "mug_1"],
                "object_ids": ["mug_1"],
                "state_edge_ids": ["mug_1-STATE_CHANGED-state:mug_1:1-1"],
                "inferred_edge_ids": [],
            },
            {
                "step": 2,
                "node_ids": ["mug_1"],
                "object_ids": ["mug_1"],
                "state_edge_ids": ["mug_1-STATE_CHANGED-state:mug_1:2-2"],
                "inferred_edge_ids": [],
            },
        ],
        "graph_report": {
            "schema_version": "dsg-spatialqa-lab.graph-report.v1",
            "action": "ingest_observation_sequence",
            "path": str(graph_path),
            "digest": graph_json_digest(graph),
            "summary": {
                "schema_version": 1,
                "node_count": 5,
                "edge_count": 3,
                "object_count": 1,
                "agent_count": 1,
                "object_history_count": 2,
                "agent_history_count": 1,
                "visible_object_count": 0,
                "hidden_object_count": 1,
                "low_confidence_object_count": 1,
                "reobserve_candidate_count": 1,
                "unlocated_object_count": 1,
                "unroomed_object_count": 1,
                "by_node_type": {
                    "agent": 1,
                    "object": 1,
                    "state": 3,
                },
                "by_edge_relation": {
                    "STATE_CHANGED": 3,
                },
                "by_object_label": {
                    "mug": 1,
                },
                "by_current_location": {},
                "by_current_room": {},
            },
        },
    }
    graph_report_payload = cast(dict[str, object], expected_report["graph_report"])
    graph_report_payload["report_digest"] = graph_report_digest(graph_report_payload)
    assert stdout_report == {
        **expected_report,
        "digest": observation_ingest_report_digest(expected_report),
    }


def test_observations_cli_reports_invalid_sequence_json(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_observations_script()
    main = cast(MainFn, getattr(module, "main"))
    sequence_path = tmp_path / "invalid-sequence.json"
    graph_path = tmp_path / "graph.json"
    sequence_path.write_text('{"schema_version": "invalid"}\n', encoding="utf-8")

    assert (
        main(
            [
                "--input",
                str(sequence_path),
                "--output-graph",
                str(graph_path),
            ]
        )
        == 1
    )

    assert json.loads(capsys.readouterr().out) == {
        "action": "ingest_observation_sequence",
        "path": str(sequence_path),
        "graph_path": str(graph_path),
        "valid": False,
        "error": "Unsupported scene observation sequence schema version: invalid",
    }
    assert not graph_path.exists()


def test_observations_cli_validates_and_compares_explicit_ingest_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_observations_script()
    main = cast(MainFn, getattr(module, "main"))
    sequence_path = tmp_path / "observations" / "mock-sequence.json"
    graph_path = tmp_path / "graphs" / "mock-graph.json"
    report_path = tmp_path / "reports" / "mock-ingest-report.json"
    observations = (
        SceneObservation(
            step=1,
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
        SceneObservation(
            step=2,
            objects=(
                ObjectObservation(
                    "mug_1",
                    "mug",
                    Pose3D(0.2, 1.2, 0.78),
                    BBox3D(center=Pose3D(0.2, 1.2, 0.78), size=(0.12, 0.12, 0.16)),
                    confidence=0.2,
                    visible=False,
                ),
            ),
        ),
    )
    save_scene_observation_sequence(observations, sequence_path)

    assert (
        main(
            [
                "--input",
                str(sequence_path),
                "--output-graph",
                str(graph_path),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["--validate-report", str(report_path)]) == 0

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    validation = json.loads(capsys.readouterr().out)
    assert validation == {
        "action": "validate_observation_ingest_report",
        "path": str(report_path),
        "valid": True,
        "schema_version": "dsg-spatialqa-lab.observation-ingest-report.v1",
        "digest": saved_report["digest"],
        "checks": [
            {
                "name": "schema_version",
                "passed": True,
                "expected": "dsg-spatialqa-lab.observation-ingest-report.v1",
                "actual": "dsg-spatialqa-lab.observation-ingest-report.v1",
            },
            {
                "name": "action",
                "passed": True,
                "expected": "ingest_observation_sequence",
                "actual": "ingest_observation_sequence",
            },
            {
                "name": "report_valid",
                "passed": True,
                "expected": True,
                "actual": True,
            },
            {
                "name": "report_digest",
                "passed": True,
                "expected": saved_report["digest"],
                "actual": saved_report["digest"],
            },
            {
                "name": "input_path_present",
                "passed": True,
                "expected": "non-empty explicit local path",
                "actual": str(sequence_path),
            },
            {
                "name": "graph_path_present",
                "passed": True,
                "expected": "non-empty explicit local path",
                "actual": str(graph_path),
            },
            {
                "name": "graph_report_path_matches",
                "passed": True,
                "expected": str(graph_path),
                "actual": str(graph_path),
            },
            {
                "name": "sequence_digest_format",
                "passed": True,
                "expected": "64 lowercase sha256 hex characters",
                "actual": scene_observation_sequence_digest(observations),
            },
            {
                "name": "observation_count_matches_results",
                "passed": True,
                "expected": 2,
                "actual": 2,
            },
            {
                "name": "steps_match_results",
                "passed": True,
                "expected": [1, 2],
                "actual": [1, 2],
            },
            {
                "name": "graph_report_valid",
                "passed": True,
            },
        ],
    }

    assert main(["--compare-report", str(report_path)]) == 0

    graph = load_graph_json(graph_path)
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_observation_ingest_report"
    assert comparison["path"] == str(report_path)
    assert comparison["sequence_path"] == str(sequence_path)
    assert comparison["graph_path"] == str(graph_path)
    assert comparison["matches"] is True
    assert comparison["saved_sequence_digest"] == scene_observation_sequence_digest(
        observations
    )
    assert comparison["current_sequence_digest"] == scene_observation_sequence_digest(
        observations
    )
    assert comparison["saved_digest"] == graph_json_digest(graph)
    assert comparison["current_digest"] == graph_json_digest(graph)
    assert [check["passed"] for check in comparison["checks"]] == [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]


def test_observations_cli_returns_nonzero_for_ingest_report_drift(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_observations_script()
    main = cast(MainFn, getattr(module, "main"))
    sequence_path = tmp_path / "observations" / "mock-sequence.json"
    graph_path = tmp_path / "graphs" / "mock-graph.json"
    report_path = tmp_path / "reports" / "mock-ingest-report.json"
    observations = (
        SceneObservation(
            step=1,
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
    )
    save_scene_observation_sequence(observations, sequence_path)
    assert (
        main(
            [
                "--input",
                str(sequence_path),
                "--output-graph",
                str(graph_path),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["graph_report"]["summary"]["hidden_object_count"] = 7
    report["ingest_results"][0]["object_ids"] = []
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    assert main(["--compare-report", str(report_path)]) == 1

    comparison = json.loads(capsys.readouterr().out)
    checks = {check["name"]: check for check in comparison["checks"]}
    assert comparison["action"] == "compare_observation_ingest_report"
    assert comparison["matches"] is False
    assert checks["graph_summary_matches_current"]["differences"] == [
        {
            "path": "hidden_object_count",
            "expected": 7,
            "actual": 0,
        }
    ]
    assert checks["ingest_results_match_current"]["differences"] == [
        {
            "path": "0.object_ids",
            "expected": [],
            "actual": ["mug_1"],
        }
    ]


def test_observations_cli_returns_nonzero_for_sequence_digest_drift(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_observations_script()
    main = cast(MainFn, getattr(module, "main"))
    sequence_path = tmp_path / "observations" / "mock-sequence.json"
    graph_path = tmp_path / "graphs" / "mock-graph.json"
    report_path = tmp_path / "reports" / "mock-ingest-report.json"
    first_sequence = (
        SceneObservation(
            step=1,
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
    )
    drifted_sequence = (
        SceneObservation(
            step=1,
            objects=(
                ObjectObservation(
                    "mug_1",
                    "mug",
                    Pose3D(0.1, 1.0, 0.78),
                    BBox3D(center=Pose3D(0.1, 1.0, 0.78), size=(0.12, 0.12, 0.16)),
                    confidence=0.95,
                    visible=True,
                ),
            ),
        ),
    )
    save_scene_observation_sequence(first_sequence, sequence_path)
    assert (
        main(
            [
                "--input",
                str(sequence_path),
                "--output-graph",
                str(graph_path),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    save_scene_observation_sequence(drifted_sequence, sequence_path)

    assert main(["--compare-report", str(report_path)]) == 1

    comparison = json.loads(capsys.readouterr().out)
    checks = {check["name"]: check for check in comparison["checks"]}
    assert comparison["matches"] is False
    assert comparison["saved_sequence_digest"] == scene_observation_sequence_digest(
        first_sequence
    )
    assert comparison["current_sequence_digest"] == scene_observation_sequence_digest(
        drifted_sequence
    )
    assert checks["sequence_digest_matches_current"] == {
        "name": "sequence_digest_matches_current",
        "passed": False,
        "expected": scene_observation_sequence_digest(first_sequence),
        "actual": scene_observation_sequence_digest(drifted_sequence),
    }


def test_observations_cli_returns_nonzero_for_exported_graph_file_drift(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_observations_script()
    main = cast(MainFn, getattr(module, "main"))
    sequence_path = tmp_path / "observations" / "mock-sequence.json"
    graph_path = tmp_path / "graphs" / "mock-graph.json"
    report_path = tmp_path / "reports" / "mock-ingest-report.json"
    observations = (
        SceneObservation(
            step=1,
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
    )
    save_scene_observation_sequence(observations, sequence_path)
    assert (
        main(
            [
                "--input",
                str(sequence_path),
                "--output-graph",
                str(graph_path),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    drifted_graph = DynamicSceneGraph()
    drifted_graph.add_region("drift_region", "Drift Region", step=9)
    save_graph_json(drifted_graph, graph_path)

    assert main(["--compare-report", str(report_path)]) == 1

    comparison = json.loads(capsys.readouterr().out)
    checks = {check["name"]: check for check in comparison["checks"]}
    assert comparison["matches"] is False
    assert checks["graph_file_digest_matches_report"]["passed"] is False
    assert checks["graph_file_summary_matches_report"]["passed"] is False
    assert {
        "path": "by_node_type.region",
        "expected": None,
        "actual": 1,
    } in checks["graph_file_summary_matches_report"]["differences"]
    assert {
        "path": "object_count",
        "expected": 1,
        "actual": 0,
    } in checks["graph_file_summary_matches_report"]["differences"]


def _detector_records() -> tuple[dict[str, object], ...]:
    return (
        {
            "schema_version": "dsg-spatialqa-lab.detector-observation-record.v1",
            "step": 2,
            "agent_id": "agent",
            "rgb_path": "rgb/0002.png",
            "depth_path": "depth/0002.npy",
            "metadata": {
                "detector_id": "owlvit-real-trial",
                "source": "detector_rgbd",
            },
            "detections": [
                {
                    "object_id": "track_mug_1",
                    "label": "mug",
                    "pose": {"x": 0.2, "y": 1.2, "z": 0.78, "yaw": 0.0},
                    "bbox": {
                        "center": {"x": 0.2, "y": 1.2, "z": 0.78, "yaw": 0.0},
                        "size": [0.12, 0.12, 0.16],
                    },
                    "confidence": 0.41,
                    "visible": False,
                    "attributes": {"track_id": "track_mug_1"},
                }
            ],
        },
        {
            "schema_version": "dsg-spatialqa-lab.detector-observation-record.v1",
            "step": 1,
            "agent_id": "agent",
            "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "rgb_path": "rgb/0001.png",
            "depth_path": "depth/0001.npy",
            "segmentation_path": "seg/0001.png",
            "metadata": {
                "detector_id": "owlvit-real-trial",
                "source": "detector_rgbd",
            },
            "detections": [
                {
                    "object_id": "track_mug_1",
                    "label": "mug",
                    "pose": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
                    "bbox": {
                        "center": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
                        "size": [0.12, 0.12, 0.16],
                    },
                    "confidence": 0.93,
                    "visible": True,
                    "attributes": {"track_id": "track_mug_1"},
                }
            ],
        },
    )
