from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from _pytest.capture import CaptureFixture

import dsg_spatialqa_lab as lab
from dsg_spatialqa_lab.schema import Pose3D


ROOT = Path(__file__).resolve().parents[1]
PREDICTED_DSG_SCRIPT = ROOT / "scripts" / "check_predicted_dsg.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_predicted_dsg_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_predicted_dsg_script",
        PREDICTED_DSG_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_predicted_dsg_evidence_report_accepts_detector_rgbd_observations(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "predicted_dsg_evidence_report")
    assert hasattr(lab, "predicted_dsg_evidence_report_digest")
    assert hasattr(lab, "save_predicted_dsg_evidence_report")
    assert hasattr(lab, "load_predicted_dsg_evidence_report")
    assert hasattr(lab, "validate_predicted_dsg_evidence_report")
    assert hasattr(lab, "compare_predicted_dsg_evidence_report")
    predicted_report_path = _write_predicted_report(tmp_path, _rgbd_observations())
    predicted_report = lab.load_predicted_graph_report(predicted_report_path)
    evidence_path = tmp_path / "predicted-dsg-evidence.json"

    report = lab.predicted_dsg_evidence_report(
        predicted_report,
        predicted_graph_report_path=predicted_report_path,
    )
    saved_path = lab.save_predicted_dsg_evidence_report(report, evidence_path)
    loaded = lab.load_predicted_dsg_evidence_report(evidence_path)
    validation = lab.validate_predicted_dsg_evidence_report(loaded)
    comparison = lab.compare_predicted_dsg_evidence_report(loaded)

    assert report["schema_version"] == (
        "dsg-spatialqa-lab.predicted-dsg-evidence-report.v1"
    )
    assert report["predicted_graph_report_path"] == str(predicted_report_path)
    assert report["predicted_graph_report_digest"] == predicted_report["digest"]
    assert report["readiness"] == {
        "ready": True,
        "failed_check_count": 0,
        "failed_checks": [],
    }
    assert report["evidence_summary"] == {
        "detector_names": ["detic_fixture"],
        "evidence_kind_counts": {
            "depth": 3,
            "detector": 3,
            "rgb": 2,
        },
        "hidden_object_observation_count": 1,
        "input_kind": "observation_sequence",
        "invalid_state_evidence_object_ids": [],
        "object_observation_count": 3,
        "observation_count": 2,
        "observation_sequence_digest": lab.scene_observation_sequence_digest(
            _rgbd_observations()
        ),
        "source_counts": {
            "rgbd_detector": 2,
            "rgbd_tracker": 1,
        },
        "state_evidence_object_observation_count": 0,
        "visible_object_observation_count": 2,
    }
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["input_kind_observation_sequence"]["passed"] is True
    assert checks["required_evidence_kinds_present"]["passed"] is True
    assert checks["observation_sequence_digest_matches_report"]["passed"] is True
    assert saved_path == evidence_path
    assert loaded == report
    assert validation["valid"] is True
    assert comparison["matches"] is True


def test_predicted_dsg_evidence_report_rejects_non_rgbd_or_mock_inputs(
    tmp_path: Path,
) -> None:
    predicted_report_path = _write_predicted_report(
        tmp_path,
        _non_rgbd_observations(),
    )
    predicted_report = lab.load_predicted_graph_report(predicted_report_path)

    report = lab.predicted_dsg_evidence_report(
        predicted_report,
        predicted_graph_report_path=predicted_report_path,
    )

    assert report["readiness"]["ready"] is False
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["required_evidence_kinds_present"]["passed"] is False
    assert checks["required_evidence_kinds_present"]["missing"] == [
        "depth",
        "detector",
        "rgb",
    ]


def test_predicted_dsg_evidence_report_rejects_mock_detector_sources(
    tmp_path: Path,
) -> None:
    predicted_report_path = _write_predicted_report(
        tmp_path,
        _mock_rgbd_observations(),
    )
    predicted_report = lab.load_predicted_graph_report(predicted_report_path)

    report = lab.predicted_dsg_evidence_report(
        predicted_report,
        predicted_graph_report_path=predicted_report_path,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["readiness"]["ready"] is False
    assert checks["required_evidence_kinds_present"]["passed"] is True
    assert checks["mock_sources_absent"]["passed"] is False
    assert checks["mock_sources_absent"]["actual"] == ["mock_rgbd_detector"]


def test_predicted_dsg_evidence_report_rejects_synthetic_detector_sources(
    tmp_path: Path,
) -> None:
    predicted_report_path = _write_predicted_report(
        tmp_path,
        _synthetic_rgbd_observations(),
    )
    predicted_report = lab.load_predicted_graph_report(predicted_report_path)

    report = lab.predicted_dsg_evidence_report(
        predicted_report,
        predicted_graph_report_path=predicted_report_path,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["readiness"]["ready"] is False
    assert "non_real_sources_absent" in report["readiness"]["failed_checks"]
    assert checks["non_real_sources_absent"] == {
        "name": "non_real_sources_absent",
        "passed": False,
        "actual": ["SyntheticRGBDDetector"],
    }


def test_predicted_dsg_evidence_report_rejects_ai2thor_metadata_sources(
    tmp_path: Path,
) -> None:
    predicted_report_path = _write_predicted_report(
        tmp_path,
        _ai2thor_metadata_observations(),
    )
    predicted_report = lab.load_predicted_graph_report(predicted_report_path)

    report = lab.predicted_dsg_evidence_report(
        predicted_report,
        predicted_graph_report_path=predicted_report_path,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["readiness"]["ready"] is False
    assert "non_real_sources_absent" in report["readiness"]["failed_checks"]
    assert checks["required_evidence_kinds_present"]["passed"] is True
    assert checks["non_real_sources_absent"] == {
        "name": "non_real_sources_absent",
        "passed": False,
        "actual": ["ai2thor"],
    }


def test_predicted_dsg_evidence_report_rejects_hidden_detector_state_evidence(
    tmp_path: Path,
) -> None:
    predicted_report_path = _write_predicted_report(
        tmp_path,
        _hidden_state_rgbd_observations(),
    )
    predicted_report = lab.load_predicted_graph_report(predicted_report_path)

    report = lab.predicted_dsg_evidence_report(
        predicted_report,
        predicted_graph_report_path=predicted_report_path,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["readiness"]["ready"] is False
    assert "detector_state_evidence_valid" in report["readiness"]["failed_checks"]
    assert checks["detector_state_evidence_valid"] == {
        "name": "detector_state_evidence_valid",
        "passed": False,
        "invalid_object_ids": ["mug_1"],
        "state_evidence_object_observation_count": 1,
    }
    assert report["evidence_summary"]["state_evidence_object_observation_count"] == 1
    assert report["evidence_summary"]["invalid_state_evidence_object_ids"] == ["mug_1"]


def test_predicted_dsg_evidence_cli_writes_valid_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_predicted_dsg_script()
    main = cast(MainFn, getattr(module, "main"))
    predicted_report_path = _write_predicted_report(tmp_path, _rgbd_observations())
    evidence_path = tmp_path / "predicted-dsg-evidence.json"

    assert main(
        [
            "--predicted-report",
            str(predicted_report_path),
            "--report",
            str(evidence_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    report = lab.load_predicted_dsg_evidence_report(evidence_path)
    assert output["action"] == "predicted_dsg_evidence"
    assert output["path"] == str(evidence_path)
    assert output["ready"] is True
    assert output["report_digest"] == report["report_digest"]

    assert main(["--validate-report", str(evidence_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_predicted_dsg_evidence_report"
    assert validation["valid"] is True

    assert main(["--compare-report", str(evidence_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_predicted_dsg_evidence_report"
    assert comparison["matches"] is True


def test_predicted_dsg_evidence_cli_returns_structured_json_for_missing_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_predicted_dsg_script()
    main = cast(MainFn, getattr(module, "main"))
    missing_path = tmp_path / "missing-predicted-report.json"
    evidence_path = tmp_path / "predicted-dsg-evidence.json"

    assert main(
        [
            "--predicted-report",
            str(missing_path),
            "--report",
            str(evidence_path),
        ]
    ) == 1

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "predicted_dsg_evidence"
    assert output["path"] == str(evidence_path)
    assert output["valid"] is False
    assert "missing-predicted-report.json" in output["error"]


def test_predicted_dsg_evidence_cli_returns_structured_json_for_missing_args(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_predicted_dsg_script()
    main = cast(MainFn, getattr(module, "main"))
    evidence_path = tmp_path / "predicted-dsg-evidence.json"

    assert main(["--report", str(evidence_path)]) == 1
    missing_predicted = json.loads(capsys.readouterr().out)
    assert missing_predicted["action"] == "predicted_dsg_evidence"
    assert missing_predicted["path"] == str(evidence_path)
    assert missing_predicted["valid"] is False
    assert missing_predicted["error"] == "--predicted-report is required"

    assert main(["--predicted-report", str(tmp_path / "predicted-report.json")]) == 1
    missing_output = json.loads(capsys.readouterr().out)
    assert missing_output["action"] == "predicted_dsg_evidence"
    assert missing_output["path"] == ""
    assert missing_output["valid"] is False
    assert missing_output["error"] == "--report is required"


def _write_predicted_report(
    tmp_path: Path,
    observations: tuple[lab.SceneObservation, ...],
) -> Path:
    observation_path = tmp_path / "detector-observations.json"
    graph_path = tmp_path / "predicted-graph.json"
    predicted_report_path = tmp_path / "predicted-report.json"
    lab.save_scene_observation_sequence(observations, observation_path)
    graph = lab.build_predicted_graph_from_observations(
        observations,
        source_path=observation_path,
    )
    lab.save_graph_json(graph, graph_path)
    predicted_report = lab.predicted_graph_report_from_observations(
        input_path=observation_path,
        graph_path=graph_path,
        graph=graph,
        observations=observations,
    )
    lab.save_predicted_graph_report(predicted_report, predicted_report_path)
    return predicted_report_path


def _rgbd_observations() -> tuple[lab.SceneObservation, ...]:
    return (
        lab.SceneObservation(
            step=1,
            agent_pose=Pose3D(0.0, 0.0, 0.0),
            objects=(
                _object(
                    "mug_1",
                    "mug",
                    source="rgbd_detector",
                    confidence=0.91,
                    visible=True,
                    extra={
                        "depth_path": "frames/000001.depth.png",
                        "detector": "detic_fixture",
                        "rgb_path": "frames/000001.rgb.png",
                    },
                ),
                _object(
                    "plate_1",
                    "plate",
                    source="rgbd_detector",
                    confidence=0.87,
                    visible=True,
                    extra={
                        "depth_path": "frames/000001.depth.png",
                        "detector": "detic_fixture",
                        "rgb_path": "frames/000001.rgb.png",
                    },
                ),
            ),
        ),
        lab.SceneObservation(
            step=2,
            agent_pose=Pose3D(0.1, 0.0, 0.0),
            objects=(
                _object(
                    "mug_1",
                    "mug",
                    source="rgbd_tracker",
                    confidence=0.35,
                    visible=False,
                    extra={
                        "depth_path": "frames/000002.depth.png",
                        "detector": "detic_fixture",
                        "hidden_reason": "not_detected_in_frame",
                    },
                ),
            ),
        ),
    )


def _non_rgbd_observations() -> tuple[lab.SceneObservation, ...]:
    return (
        lab.SceneObservation(
            step=1,
            objects=(
                _object(
                    "mug_1",
                    "mug",
                    source="caption_memory",
                    confidence=0.8,
                    visible=True,
                    extra={"caption": "mug on table"},
                ),
            ),
        ),
        lab.SceneObservation(
            step=2,
            objects=(
                _object(
                    "mug_1",
                    "mug",
                    source="caption_memory",
                    confidence=0.8,
                    visible=True,
                    extra={"caption": "mug still on table"},
                ),
            ),
        ),
    )


def _mock_rgbd_observations() -> tuple[lab.SceneObservation, ...]:
    return (
        lab.SceneObservation(
            step=1,
            agent_pose=Pose3D(0.0, 0.0, 0.0),
            objects=(
                _object(
                    "mug_1",
                    "mug",
                    source="mock_rgbd_detector",
                    confidence=0.91,
                    visible=True,
                    extra={
                        "depth_path": "frames/000001.depth.png",
                        "detector": "detic_fixture",
                        "rgb_path": "frames/000001.rgb.png",
                    },
                ),
                _object(
                    "plate_1",
                    "plate",
                    source="mock_rgbd_detector",
                    confidence=0.87,
                    visible=True,
                    extra={
                        "depth_path": "frames/000001.depth.png",
                        "detector": "detic_fixture",
                        "rgb_path": "frames/000001.rgb.png",
                    },
                ),
            ),
        ),
        lab.SceneObservation(
            step=2,
            agent_pose=Pose3D(0.1, 0.0, 0.0),
            objects=(
                _object(
                    "mug_1",
                    "mug",
                    source="mock_rgbd_detector",
                    confidence=0.35,
                    visible=False,
                    extra={
                        "depth_path": "frames/000002.depth.png",
                        "detector": "detic_fixture",
                    },
                ),
            ),
        ),
    )


def _synthetic_rgbd_observations() -> tuple[lab.SceneObservation, ...]:
    return (
        lab.SceneObservation(
            step=1,
            agent_pose=Pose3D(0.0, 0.0, 0.0),
            objects=(
                _object(
                    "mug_1",
                    "mug",
                    source="SyntheticRGBDDetector",
                    confidence=0.91,
                    visible=True,
                    extra={
                        "depth_path": "frames/000001.depth.png",
                        "detector": "detic_real_trial",
                        "rgb_path": "frames/000001.rgb.png",
                    },
                ),
                _object(
                    "plate_1",
                    "plate",
                    source="SyntheticRGBDDetector",
                    confidence=0.87,
                    visible=True,
                    extra={
                        "depth_path": "frames/000001.depth.png",
                        "detector": "detic_real_trial",
                        "rgb_path": "frames/000001.rgb.png",
                    },
                ),
            ),
        ),
        lab.SceneObservation(
            step=2,
            agent_pose=Pose3D(0.1, 0.0, 0.0),
            objects=(
                _object(
                    "mug_1",
                    "mug",
                    source="SyntheticRGBDDetector",
                    confidence=0.35,
                    visible=False,
                    extra={
                        "depth_path": "frames/000002.depth.png",
                        "detector": "detic_real_trial",
                    },
                ),
            ),
        ),
    )


def _ai2thor_metadata_observations() -> tuple[lab.SceneObservation, ...]:
    return (
        lab.SceneObservation(
            step=1,
            agent_pose=Pose3D(0.0, 0.0, 0.0),
            objects=(
                _object(
                    "mug_1",
                    "mug",
                    source="ai2thor",
                    confidence=0.91,
                    visible=True,
                    extra={
                        "depth_path": "frames/000001.depth.png",
                        "detector": "ai2thor_metadata_visible_objects",
                        "rgb_path": "frames/000001.rgb.png",
                    },
                ),
            ),
        ),
        lab.SceneObservation(
            step=2,
            agent_pose=Pose3D(0.1, 0.0, 0.0),
            objects=(
                _object(
                    "plate_1",
                    "plate",
                    source="ai2thor",
                    confidence=0.87,
                    visible=True,
                    extra={
                        "depth_path": "frames/000002.depth.png",
                        "detector": "ai2thor_metadata_visible_objects",
                        "rgb_path": "frames/000002.rgb.png",
                    },
                ),
            ),
        ),
    )


def _hidden_state_rgbd_observations() -> tuple[lab.SceneObservation, ...]:
    return (
        lab.SceneObservation(
            step=1,
            agent_pose=Pose3D(0.0, 0.0, 0.0),
            objects=(
                _object(
                    "mug_1",
                    "mug",
                    source="rgbd_detector",
                    confidence=0.91,
                    visible=False,
                    extra={
                        "depth_path": "frames/000001.depth.png",
                        "detector": "detic_fixture",
                        "evidence_kinds": ["depth", "detector", "rgb"],
                        "rgb_path": "frames/000001.rgb.png",
                        "states": {"isOpen": False},
                    },
                ),
                _object(
                    "plate_1",
                    "plate",
                    source="rgbd_detector",
                    confidence=0.87,
                    visible=True,
                    extra={
                        "depth_path": "frames/000001.depth.png",
                        "detector": "detic_fixture",
                        "rgb_path": "frames/000001.rgb.png",
                    },
                ),
            ),
        ),
        lab.SceneObservation(
            step=2,
            agent_pose=Pose3D(0.1, 0.0, 0.0),
            objects=(
                _object(
                    "plate_1",
                    "plate",
                    source="rgbd_detector",
                    confidence=0.87,
                    visible=True,
                    extra={
                        "depth_path": "frames/000002.depth.png",
                        "detector": "detic_fixture",
                        "rgb_path": "frames/000002.rgb.png",
                    },
                ),
            ),
        ),
    )


def _object(
    object_id: str,
    label: str,
    *,
    source: str,
    confidence: float,
    visible: bool,
    extra: dict[str, object],
) -> lab.ObjectObservation:
    return lab.ObjectObservation(
        object_id,
        label,
        Pose3D(0.3, 1.0, 0.82),
        lab.BBox3D(
            center=Pose3D(0.3, 1.0, 0.82),
            size=(0.2, 0.2, 0.3),
        ),
        confidence=confidence,
        visible=visible,
        attributes={"source": source, **extra},
    )
