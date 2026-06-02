import json
import hashlib
from pathlib import Path

import pytest

import dsg_spatialqa_lab as lab
from dsg_spatialqa_lab import (
    BBox3D,
    DynamicSceneGraph,
    GraphTool,
    NodeObservation,
    ObjectObservation,
    ObservationIngestor,
    Pose3D,
    SceneObservation,
    SpatialQAError,
)


def test_scene_observation_json_round_trip_and_save_loads_explicit_file(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "scene_observation_to_dict")
    assert hasattr(lab, "scene_observation_from_dict")
    assert hasattr(lab, "scene_observation_to_json")
    assert hasattr(lab, "scene_observation_from_json")
    assert hasattr(lab, "save_scene_observation")
    assert hasattr(lab, "load_scene_observation")
    observation = SceneObservation(
        step=7,
        agent_pose=Pose3D(0.0, 0.0, 0.0, yaw=0.5),
        rooms=(NodeObservation("kitchen", "Kitchen", attributes={"floor": "1"}),),
        regions=(NodeObservation("counter_region", "Counter region"),),
        objects=(
            ObjectObservation(
                "mug_1",
                "mug",
                Pose3D(-0.4, 1.0, 0.78),
                BBox3D(center=Pose3D(-0.4, 1.0, 0.78), size=(0.12, 0.12, 0.16)),
                confidence=0.95,
                visible=True,
                attributes={"color": "blue"},
            ),
        ),
    )

    payload = lab.scene_observation_to_dict(observation)
    payload_json = lab.scene_observation_to_json(observation)
    repeated_payload_json = lab.scene_observation_to_json(observation)
    observation_path = tmp_path / "observations" / "mock-frame-7.json"
    saved_path = lab.save_scene_observation(observation, observation_path)
    loaded_observation = lab.load_scene_observation(observation_path)

    assert payload == {
        "schema_version": "dsg-spatialqa-lab.scene-observation.v1",
        "step": 7,
        "agent_id": "agent",
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.5},
        "rooms": [
            {
                "node_id": "kitchen",
                "label": "Kitchen",
                "attributes": {"floor": "1"},
            },
        ],
        "regions": [
            {
                "node_id": "counter_region",
                "label": "Counter region",
                "attributes": {},
            },
        ],
        "objects": [
            {
                "object_id": "mug_1",
                "label": "mug",
                "pose": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
                "bbox": {
                    "center": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
                    "size": [0.12, 0.12, 0.16],
                },
                "confidence": 0.95,
                "visible": True,
                "attributes": {"color": "blue"},
            },
        ],
    }
    assert payload_json == repeated_payload_json
    assert payload_json.endswith("\n")
    assert json.loads(payload_json) == payload
    assert lab.scene_observation_from_dict(payload) == observation
    assert lab.scene_observation_from_json(payload_json) == observation
    assert saved_path == observation_path
    assert loaded_observation == observation
    assert json.loads(observation_path.read_text(encoding="utf-8")) == payload


def test_scene_observation_sequence_json_round_trip_and_save_loads_explicit_file(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "scene_observation_sequence_to_dict")
    assert hasattr(lab, "scene_observation_sequence_from_dict")
    assert hasattr(lab, "scene_observation_sequence_to_json")
    assert hasattr(lab, "scene_observation_sequence_from_json")
    assert hasattr(lab, "scene_observation_sequence_digest")
    assert hasattr(lab, "save_scene_observation_sequence")
    assert hasattr(lab, "load_scene_observation_sequence")
    assert hasattr(lab, "load_observation_ingest_report")
    assert hasattr(lab, "validate_observation_ingest_report")
    assert hasattr(lab, "compare_observation_ingest_report")
    assert hasattr(lab, "save_observation_ingest_report")
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
                    confidence=0.8,
                    visible=False,
                ),
            ),
        ),
    )

    payload = lab.scene_observation_sequence_to_dict(observations)
    payload_json = lab.scene_observation_sequence_to_json(observations)
    repeated_payload_json = lab.scene_observation_sequence_to_json(observations)
    sequence_path = tmp_path / "observations" / "mock-sequence.json"
    saved_path = lab.save_scene_observation_sequence(observations, sequence_path)
    loaded_observations = lab.load_scene_observation_sequence(sequence_path)

    assert payload["schema_version"] == "dsg-spatialqa-lab.scene-observation-sequence.v1"
    assert payload["observation_count"] == 2
    assert payload["steps"] == [1, 2]
    assert payload["observations"] == [
        lab.scene_observation_to_dict(observations[0]),
        lab.scene_observation_to_dict(observations[1]),
    ]
    assert payload_json == repeated_payload_json
    assert payload_json.endswith("\n")
    assert json.loads(payload_json) == payload
    assert lab.scene_observation_sequence_digest(observations) == hashlib.sha256(
        payload_json.encode("utf-8")
    ).hexdigest()
    assert lab.scene_observation_sequence_from_dict(payload) == observations
    assert lab.scene_observation_sequence_from_json(payload_json) == observations
    assert saved_path == sequence_path
    assert loaded_observations == observations
    assert json.loads(sequence_path.read_text(encoding="utf-8")) == payload


def test_scene_observation_sequence_payload_validation_reports_digest_and_summary() -> None:
    assert hasattr(lab, "validate_scene_observation_sequence_payload")
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
    payload = lab.scene_observation_sequence_to_dict(observations)

    validation = lab.validate_scene_observation_sequence_payload(payload)

    assert validation == {
        "valid": True,
        "schema_version": "dsg-spatialqa-lab.scene-observation-sequence.v1",
        "digest": lab.scene_observation_sequence_digest(observations),
        "summary": lab.scene_observation_sequence_summary(observations),
        "checks": [
            {
                "name": "schema_version",
                "passed": True,
                "expected": "dsg-spatialqa-lab.scene-observation-sequence.v1",
                "actual": "dsg-spatialqa-lab.scene-observation-sequence.v1",
            },
            {
                "name": "observations_sequence_present",
                "passed": True,
                "expected": "sequence",
                "actual": "sequence",
            },
            {
                "name": "observation_count_matches_observations",
                "passed": True,
                "expected": 2,
                "actual": 2,
            },
            {
                "name": "steps_match_observations",
                "passed": True,
                "expected": [1, 2],
                "actual": [1, 2],
            },
            {
                "name": "sequence_loads",
                "passed": True,
            },
        ],
    }


def test_scene_observation_sequence_payload_validation_reports_shape_errors() -> None:
    payload = {
        "schema_version": "dsg-spatialqa-lab.scene-observation-sequence.v1",
        "observation_count": 2,
        "steps": [1],
        "observations": [
            lab.scene_observation_to_dict(
                SceneObservation(
                    step=1,
                    objects=(
                        ObjectObservation(
                            "mug_1",
                            "mug",
                            Pose3D(-0.4, 1.0, 0.78),
                            BBox3D(
                                center=Pose3D(-0.4, 1.0, 0.78),
                                size=(0.12, 0.12, 0.16),
                            ),
                            confidence=0.95,
                            visible=True,
                        ),
                    ),
                )
            ),
        ],
    }

    validation = lab.validate_scene_observation_sequence_payload(payload)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert validation["digest"] is None
    assert validation["summary"] is None
    assert validation["error"] == "observation_count must match observations length"
    assert checks["observation_count_matches_observations"] == {
        "name": "observation_count_matches_observations",
        "passed": False,
        "expected": 2,
        "actual": 1,
    }
    assert checks["sequence_loads"] == {
        "name": "sequence_loads",
        "passed": False,
        "error": "observation_count must match observations length",
    }


def test_scene_observation_sequence_summary_is_deterministic_and_digestable() -> None:
    assert hasattr(lab, "scene_observation_sequence_summary")
    assert hasattr(lab, "scene_observation_sequence_summary_digest")
    observations = (
        SceneObservation(
            step=3,
            agent_pose=Pose3D(0.0, 0.0, 0.0),
            rooms=(NodeObservation("kitchen", "Kitchen"),),
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
                    Pose3D(0.0, 1.0, 0.78),
                    BBox3D(center=Pose3D(0.0, 1.0, 0.78), size=(0.1, 0.1, 0.12)),
                    confidence=0.2,
                    visible=False,
                ),
            ),
        ),
        SceneObservation(
            step=5,
            regions=(NodeObservation("counter", "Counter"),),
            objects=(
                ObjectObservation(
                    "mug_1",
                    "mug",
                    Pose3D(-0.2, 1.0, 0.78),
                    BBox3D(center=Pose3D(-0.2, 1.0, 0.78), size=(0.12, 0.12, 0.16)),
                    confidence=0.8,
                    visible=False,
                ),
                ObjectObservation(
                    "plate_1",
                    "plate",
                    Pose3D(0.4, 1.0, 0.72),
                    BBox3D(center=Pose3D(0.4, 1.0, 0.72), size=(0.26, 0.26, 0.04)),
                    confidence=0.4,
                    visible=True,
                ),
            ),
        ),
    )

    summary = lab.scene_observation_sequence_summary(observations)
    repeated_summary = lab.scene_observation_sequence_summary(observations)
    expected_without_digest = {
        "schema_version": "dsg-spatialqa-lab.scene-observation-sequence-summary.v1",
        "sequence_digest": lab.scene_observation_sequence_digest(observations),
        "low_confidence_threshold": 0.5,
        "observation_count": 2,
        "steps": [3, 5],
        "first_step": 3,
        "last_step": 5,
        "agent_pose_count": 1,
        "room_observation_count": 1,
        "region_observation_count": 1,
        "object_observation_count": 4,
        "unique_object_count": 3,
        "unique_object_ids": ["cup_1", "mug_1", "plate_1"],
        "visible_object_observation_count": 2,
        "hidden_object_observation_count": 2,
        "low_confidence_object_observation_count": 2,
        "reobserve_candidate_observation_count": 1,
        "by_object_label": {
            "cup": 1,
            "mug": 2,
            "plate": 1,
        },
        "by_step": [
            {
                "step": 3,
                "agent_pose": True,
                "room_count": 1,
                "region_count": 0,
                "object_count": 2,
                "object_ids": ["cup_1", "mug_1"],
                "visible_object_count": 1,
                "hidden_object_count": 1,
                "low_confidence_object_count": 1,
                "reobserve_candidate_count": 1,
            },
            {
                "step": 5,
                "agent_pose": False,
                "room_count": 0,
                "region_count": 1,
                "object_count": 2,
                "object_ids": ["mug_1", "plate_1"],
                "visible_object_count": 1,
                "hidden_object_count": 1,
                "low_confidence_object_count": 1,
                "reobserve_candidate_count": 0,
            },
        ],
    }
    expected_digest = lab.scene_observation_sequence_summary_digest(
        expected_without_digest
    )

    assert summary == {**expected_without_digest, "digest": expected_digest}
    assert repeated_summary == summary
    assert lab.scene_observation_sequence_summary_digest(summary) == expected_digest


def test_scene_observation_sequence_summary_artifact_validates_and_compares(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "scene_observation_sequence_summary_json")
    assert hasattr(lab, "save_scene_observation_sequence_summary")
    assert hasattr(lab, "load_scene_observation_sequence_summary")
    assert hasattr(lab, "validate_scene_observation_sequence_summary")
    assert hasattr(lab, "compare_scene_observation_sequence_summary")
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
    sequence_path = tmp_path / "observations" / "mock-sequence.json"
    summary_path = tmp_path / "reports" / "mock-sequence-summary.json"
    lab.save_scene_observation_sequence(observations, sequence_path)
    summary = {
        "action": "summarize_observation_sequence",
        "path": str(sequence_path),
        "valid": True,
        **lab.scene_observation_sequence_summary(observations),
    }

    payload = lab.scene_observation_sequence_summary_json(summary)
    saved_path = lab.save_scene_observation_sequence_summary(summary, summary_path)
    loaded_summary = lab.load_scene_observation_sequence_summary(summary_path)
    validation = lab.validate_scene_observation_sequence_summary(loaded_summary)
    comparison = lab.compare_scene_observation_sequence_summary(loaded_summary)

    assert json.loads(payload) == summary
    assert saved_path == summary_path
    assert loaded_summary == summary
    assert summary_path.read_text(encoding="utf-8").endswith("\n")
    assert validation == {
        "valid": True,
        "schema_version": "dsg-spatialqa-lab.scene-observation-sequence-summary.v1",
        "digest": summary["digest"],
        "checks": [
            {
                "name": "schema_version",
                "passed": True,
                "expected": "dsg-spatialqa-lab.scene-observation-sequence-summary.v1",
                "actual": "dsg-spatialqa-lab.scene-observation-sequence-summary.v1",
            },
            {
                "name": "action",
                "passed": True,
                "expected": "summarize_observation_sequence",
                "actual": "summarize_observation_sequence",
            },
            {
                "name": "summary_valid",
                "passed": True,
                "expected": True,
                "actual": True,
            },
            {
                "name": "summary_digest",
                "passed": True,
                "expected": summary["digest"],
                "actual": summary["digest"],
            },
            {
                "name": "sequence_path_present",
                "passed": True,
                "expected": "non-empty explicit local path",
                "actual": str(sequence_path),
            },
            {
                "name": "sequence_digest_format",
                "passed": True,
                "expected": "64 lowercase sha256 hex characters",
                "actual": lab.scene_observation_sequence_digest(observations),
            },
            {
                "name": "by_step_entries_valid",
                "passed": True,
            },
            {
                "name": "observation_count_matches_by_step",
                "passed": True,
                "expected": 2,
                "actual": 2,
            },
            {
                "name": "steps_match_by_step",
                "passed": True,
                "expected": [1, 2],
                "actual": [1, 2],
            },
            {
                "name": "first_step_matches_steps",
                "passed": True,
                "expected": 1,
                "actual": 1,
            },
            {
                "name": "last_step_matches_steps",
                "passed": True,
                "expected": 2,
                "actual": 2,
            },
            {
                "name": "object_observation_count_matches_steps",
                "passed": True,
                "expected": 2,
                "actual": 2,
            },
            {
                "name": "visible_object_observation_count_matches_steps",
                "passed": True,
                "expected": 1,
                "actual": 1,
            },
            {
                "name": "hidden_object_observation_count_matches_steps",
                "passed": True,
                "expected": 1,
                "actual": 1,
            },
            {
                "name": "low_confidence_object_observation_count_matches_steps",
                "passed": True,
                "expected": 1,
                "actual": 1,
            },
            {
                "name": "reobserve_candidate_observation_count_matches_steps",
                "passed": True,
                "expected": 1,
                "actual": 1,
            },
            {
                "name": "unique_object_count_matches_ids",
                "passed": True,
                "expected": 2,
                "actual": 2,
            },
            {
                "name": "object_label_counts_valid",
                "passed": True,
            },
        ],
    }
    assert comparison["matches"] is True
    assert comparison["sequence_path"] == str(sequence_path)
    assert comparison["saved_sequence_digest"] == lab.scene_observation_sequence_digest(
        observations
    )
    assert comparison["current_sequence_digest"] == lab.scene_observation_sequence_digest(
        observations
    )
    assert comparison["saved_digest"] == summary["digest"]
    assert comparison["current_digest"] == summary["digest"]
    assert [check["passed"] for check in comparison["checks"]] == [True, True, True]


def test_scene_observation_sequence_summary_validation_and_compare_report_drift(
    tmp_path: Path,
) -> None:
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
    drifted_observations = (
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
    sequence_path = tmp_path / "observations" / "mock-sequence.json"
    lab.save_scene_observation_sequence(observations, sequence_path)
    summary = {
        "action": "summarize_observation_sequence",
        "path": str(sequence_path),
        "valid": True,
        **lab.scene_observation_sequence_summary(observations),
    }
    tampered_summary = json.loads(lab.scene_observation_sequence_summary_json(summary))
    tampered_summary["digest"] = "0" * 64

    validation = lab.validate_scene_observation_sequence_summary(tampered_summary)
    checks = {check["name"]: check for check in validation["checks"]}
    assert validation["valid"] is False
    assert checks["summary_digest"] == {
        "name": "summary_digest",
        "passed": False,
        "expected": summary["digest"],
        "actual": "0" * 64,
    }

    lab.save_scene_observation_sequence(drifted_observations, sequence_path)
    comparison = lab.compare_scene_observation_sequence_summary(summary)
    compare_checks = {check["name"]: check for check in comparison["checks"]}
    assert comparison["matches"] is False
    assert compare_checks["sequence_digest_matches_current"]["passed"] is False
    assert {
        "path": "object_observation_count",
        "expected": 1,
        "actual": 2,
    } in compare_checks["sequence_summary_matches_current"]["differences"]
    assert {
        "path": "unique_object_ids",
        "expected": ["mug_1"],
        "actual": ["cup_1", "mug_1"],
    } in compare_checks["sequence_summary_matches_current"]["differences"]


def test_observation_ingest_report_save_loads_explicit_file(
    tmp_path: Path,
) -> None:
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
    sequence_path = tmp_path / "observations" / "mock-sequence.json"
    graph_path = tmp_path / "graphs" / "mock-graph.json"
    report_path = tmp_path / "reports" / "mock-ingest-report.json"
    lab.save_scene_observation_sequence(observations, sequence_path)
    graph, ingest_results = lab.ingest_scene_observation_sequence(
        observations,
        source_path=sequence_path,
    )
    report = lab.observation_ingest_report(
        input_path=sequence_path,
        graph_path=graph_path,
        graph=graph,
        ingest_results=ingest_results,
        sequence_digest=lab.scene_observation_sequence_digest(observations),
    )

    saved_path = lab.save_observation_ingest_report(report, report_path)
    loaded_report = lab.load_observation_ingest_report(report_path)

    assert saved_path == report_path
    assert loaded_report == report
    assert report_path.read_text(encoding="utf-8").endswith("\n")
    assert lab.validate_observation_ingest_report(loaded_report)["valid"] is True


def test_observation_ingest_report_includes_stable_digest_and_validates_tampering(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "observation_ingest_report_digest")
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
    sequence_path = tmp_path / "observations" / "mock-sequence.json"
    graph_path = tmp_path / "graphs" / "mock-graph.json"
    graph, ingest_results = lab.ingest_scene_observation_sequence(
        observations,
        source_path=sequence_path,
    )
    report = lab.observation_ingest_report(
        input_path=sequence_path,
        graph_path=graph_path,
        graph=graph,
        ingest_results=ingest_results,
        sequence_digest=lab.scene_observation_sequence_digest(observations),
    )
    report_without_digest = {key: value for key, value in report.items() if key != "digest"}
    expected_digest = hashlib.sha256(
        json.dumps(report_without_digest, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()

    assert report["digest"] == expected_digest
    assert lab.observation_ingest_report_digest(report) == expected_digest
    validation = lab.validate_observation_ingest_report(report)
    checks = {check["name"]: check for check in validation["checks"]}
    assert validation["digest"] == expected_digest
    assert checks["report_digest"] == {
        "name": "report_digest",
        "passed": True,
        "expected": expected_digest,
        "actual": expected_digest,
    }

    tampered_report = json.loads(lab.observation_ingest_report_json(report))
    tampered_report["digest"] = "0" * 64

    tampered_validation = lab.validate_observation_ingest_report(tampered_report)
    tampered_checks = {check["name"]: check for check in tampered_validation["checks"]}
    assert tampered_validation["valid"] is False
    assert tampered_checks["report_digest"] == {
        "name": "report_digest",
        "passed": False,
        "expected": expected_digest,
        "actual": "0" * 64,
    }


def test_observation_ingest_report_validation_requires_explicit_artifact_paths(
    tmp_path: Path,
) -> None:
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
    sequence_path = tmp_path / "observations" / "mock-sequence.json"
    graph_path = tmp_path / "graphs" / "mock-graph.json"
    graph, ingest_results = lab.ingest_scene_observation_sequence(
        observations,
        source_path=sequence_path,
    )
    report = lab.observation_ingest_report(
        input_path=sequence_path,
        graph_path=graph_path,
        graph=graph,
        ingest_results=ingest_results,
        sequence_digest=lab.scene_observation_sequence_digest(observations),
    )
    report["path"] = ""
    report["graph_path"] = ""
    report["graph_report"]["path"] = str(graph_path)

    validation = lab.validate_observation_ingest_report(report)
    checks = {check["name"]: check for check in validation["checks"]}

    assert validation["valid"] is False
    assert checks["input_path_present"] == {
        "name": "input_path_present",
        "passed": False,
        "expected": "non-empty explicit local path",
        "actual": "",
    }
    assert checks["graph_path_present"] == {
        "name": "graph_path_present",
        "passed": False,
        "expected": "non-empty explicit local path",
        "actual": "",
    }
    assert checks["graph_report_path_matches"] == {
        "name": "graph_report_path_matches",
        "passed": False,
        "expected": "",
        "actual": str(graph_path),
    }


def test_scene_observation_ingestion_updates_graph_and_infers_relations() -> None:
    graph = DynamicSceneGraph()
    ingestor = ObservationIngestor(graph)

    result = ingestor.ingest(
        SceneObservation(
            step=7,
            agent_pose=Pose3D(0.0, 0.0, 0.0, yaw=0.0),
            rooms=(NodeObservation("kitchen", "Kitchen"),),
            regions=(NodeObservation("counter_region", "Counter region"),),
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
                    "plate_1",
                    "plate",
                    Pose3D(0.35, 1.0, 0.72),
                    BBox3D(center=Pose3D(0.35, 1.0, 0.72), size=(0.26, 0.26, 0.04)),
                    confidence=0.9,
                    visible=True,
                ),
                ObjectObservation(
                    "table_1",
                    "table",
                    Pose3D(0.0, 1.0, 0.35),
                    BBox3D(center=Pose3D(0.0, 1.0, 0.35), size=(1.2, 0.8, 0.7)),
                    confidence=1.0,
                    visible=True,
                ),
            ),
        ),
        infer_relations=("LEFT_OF", "RIGHT_OF", "NEAR", "ON"),
        reference_frames=("agent", "world"),
        relation_evidence=("mock_observation:7",),
    )

    tool = GraphTool(graph)
    assert result.step == 7
    assert result.node_ids == (
        "agent",
        "counter_region",
        "kitchen",
        "mug_1",
        "plate_1",
        "table_1",
    )
    assert result.object_ids == ("mug_1", "plate_1", "table_1")
    assert result.state_edge_ids == (
        "mug_1-STATE_CHANGED-state:mug_1:7-7",
        "plate_1-STATE_CHANGED-state:plate_1:7-7",
        "table_1-STATE_CHANGED-state:table_1:7-7",
    )
    assert graph.get_agent_pose("agent") == Pose3D(0.0, 0.0, 0.0, yaw=0.0)
    assert graph.nodes["kitchen"].type == "room"
    assert graph.nodes["counter_region"].type == "region"
    assert [edge.id for edge in tool.get_relation("mug_1", "LEFT_OF", "plate_1", "agent")] == [
        "mug_1-LEFT_OF-plate_1-7"
    ]
    assert [edge.evidence for edge in tool.get_relation("mug_1", "ON", "table_1", "world")] == [
        ["mock_observation:7"]
    ]
    assert "mug_1-ON-table_1-7" in result.inferred_edge_ids


def test_scene_observation_ingestion_preserves_last_seen_for_invisible_object() -> None:
    graph = DynamicSceneGraph()
    ingestor = ObservationIngestor(graph)
    visible_pose = Pose3D(0.2, 1.0, 0.4)

    ingestor.ingest(
        SceneObservation(
            step=1,
            objects=(
                ObjectObservation(
                    "spoon_1",
                    "spoon",
                    visible_pose,
                    BBox3D(center=visible_pose, size=(0.2, 0.04, 0.02)),
                    confidence=0.82,
                    visible=True,
                ),
            ),
        )
    )
    ingestor.ingest(
        SceneObservation(
            step=2,
            objects=(
                ObjectObservation(
                    "spoon_1",
                    "spoon",
                    Pose3D(0.6, 1.2, 0.4),
                    BBox3D(center=Pose3D(0.6, 1.2, 0.4), size=(0.2, 0.04, 0.02)),
                    confidence=0.2,
                    visible=False,
                ),
            ),
        )
    )

    state = graph.get_object_state("spoon_1")
    assert state.visible is False
    assert state.last_seen_step == 1
    assert state.last_seen_pose == visible_pose
    assert GraphTool(graph).needs_reobserve("spoon_1") is True


def test_scene_observation_ingestion_rejects_invalid_step() -> None:
    with pytest.raises(SpatialQAError, match="observation step must be an integer"):
        ObservationIngestor(DynamicSceneGraph()).ingest(SceneObservation(step=True))


def test_scene_observation_ingestion_requires_agent_pose_for_agent_relations() -> None:
    graph = DynamicSceneGraph()
    observation = SceneObservation(
        step=3,
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
                "plate_1",
                "plate",
                Pose3D(0.35, 1.0, 0.72),
                BBox3D(center=Pose3D(0.35, 1.0, 0.72), size=(0.26, 0.26, 0.04)),
                confidence=0.9,
                visible=True,
            ),
        ),
    )

    with pytest.raises(SpatialQAError, match="Agent pose not found: agent"):
        ObservationIngestor(graph).ingest(
            observation,
            infer_relations=("LEFT_OF",),
            reference_frames=("agent",),
        )
