from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from _pytest.capture import CaptureFixture

import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
COVERAGE_PLAN_SCRIPT = ROOT / "scripts" / "plan_coverage_collection.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_coverage_plan_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "plan_coverage_collection_script",
        COVERAGE_PLAN_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_coverage_collection_plan_targets_missing_objects_without_evidence_leak() -> None:
    assert hasattr(lab, "coverage_collection_plan")
    frames = _episode_frames()
    report = _qa_observability_report()

    plan = lab.coverage_collection_plan(
        report,
        frames,
        qa_observability_report_path="reports/qa-observability.json",
        episode_paths=("episodes/episode-001.jsonl",),
        target_evidence_observable_count=30,
        target_node_recall_floor=0.5,
    )
    validation = lab.validate_coverage_collection_plan(plan)

    assert validation["valid"] is True
    assert plan["schema_version"] == "dsg-spatialqa-lab.coverage-collection-plan.v1"
    assert plan["source_use_policy"] == {
        "episode_metadata_used_for": "collection_planning_only",
        "not_predicted_graph_evidence": True,
        "requires_visible_detector_evidence_before_claim": True,
    }
    assert plan["summary"] == {
        "case_count": 2,
        "current_evidence_observable_count": 1,
        "current_target_node_recall": 0.5,
        "missing_target_node_count": 1,
        "planned_target_count": 1,
        "relation_evidence_target_count": 1,
        "state_evidence_target_count": 1,
        "target_evidence_observable_count": 30,
        "target_node_recall_floor": 0.5,
        "unresolved_target_count": 0,
    }
    assert plan["collection_targets"] == [
        {
            "ai2thor_object_id": "Plate|-01.00|+00.90|+00.25",
            "current_visible_observation_count": 0,
            "episode_id": "episode-001",
            "first_metadata_step": 1,
            "label": "plate",
            "last_metadata_step": 2,
            "missing_reasons": ["target_missing"],
            "object_id": "plate_1",
            "pose": {"x": -1.0, "y": 0.9, "z": 0.25, "yaw": 0.0},
            "related_case_ids": ["case:plate"],
            "scene_id": "FloorPlan1",
            "suggested_action": "collect_visible_rgbd_detection",
            "viewpoint_hints": [
                {
                    "agent_pose": {"x": 0.0, "y": 0.9, "z": 0.25, "yaw": 0.0},
                    "distance_to_target": 1.0,
                    "episode_id": "episode-001",
                    "scene_id": "FloorPlan1",
                    "step": 2,
                    "suggested_yaw_to_target": 270.0,
                },
                {
                    "agent_pose": {"x": 0.0, "y": 0.9, "z": 0.0, "yaw": 0.0},
                    "distance_to_target": 1.030776,
                    "episode_id": "episode-001",
                    "scene_id": "FloorPlan1",
                    "step": 1,
                    "suggested_yaw_to_target": 284.036243,
                },
            ],
        }
    ]
    assert plan["relation_evidence_targets"] == [
        {
            "case_id": "case:mug",
            "missing_required_edge_relations": ["ON"],
            "missing_required_edges": ["mug_1-ON-table_1-1"],
            "target_nodes": ["mug_1"],
        }
    ]
    assert plan["state_evidence_targets"] == [
        {
            "case_id": "case:plate",
            "missing_state_edges": ["plate_1-STATE_CHANGED-state:plate_1:2-2"],
            "missing_state_nodes": ["state:plate_1:2"],
            "object_ids": ["plate_1"],
            "required_state_contract": {
                "evidence_kinds": ["depth", "detector", "rgb"],
                "state_attributes_field": "attributes.states",
                "visible": True,
            },
            "target_nodes": ["plate_1"],
        }
    ]


def test_coverage_collection_plan_cli_writes_valid_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_coverage_plan_script()
    main = cast(MainFn, getattr(module, "main"))
    report_path = tmp_path / "qa-observability.json"
    episode_path = tmp_path / "episode-001.jsonl"
    output_path = tmp_path / "coverage-plan.json"
    report_path.write_text(
        json.dumps(_qa_observability_report(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lab.save_episode_sequence(_episode_frames(), episode_path)

    assert main(
        [
            "--qa-observability-report",
            str(report_path),
            "--episode",
            str(episode_path),
            "--output",
            str(output_path),
            "--target-evidence-observable-count",
            "30",
            "--target-node-recall-floor",
            "0.5",
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    plan = lab.load_coverage_collection_plan(output_path)
    assert output["action"] == "coverage_collection_plan"
    assert output["valid"] is True
    assert output["summary"]["planned_target_count"] == 1
    assert output["digest"] == lab.coverage_collection_plan_digest(plan)
    assert lab.validate_coverage_collection_plan(plan)["valid"] is True

    assert main(["--validate-plan", str(output_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_coverage_collection_plan"
    assert validation["valid"] is True


def test_coverage_collection_acceptance_report_requires_visible_detector_evidence() -> None:
    assert hasattr(lab, "coverage_collection_acceptance_report")
    plan = lab.coverage_collection_plan(
        _qa_observability_report(),
        _episode_frames(),
        qa_observability_report_path="reports/qa-observability.json",
        episode_paths=("episodes/episode-001.jsonl",),
    )
    observations = (
        lab.SceneObservation(
            step=1,
            agent_id="agent",
            agent_pose=lab.Pose3D(0.0, 0.9, 0.0, yaw=0.0),
            objects=(
                _object_observation(
                    "plate_1",
                    visible=False,
                    attributes={
                        "coverage_source": "episode_metadata",
                        "source_kind": "ai2thor_metadata_coverage",
                        "evidence_kinds": ["rgb", "depth", "detector"],
                    },
                ),
            ),
        ),
        lab.SceneObservation(
            step=2,
            agent_id="agent",
            agent_pose=lab.Pose3D(0.0, 0.9, 0.0, yaw=0.0),
            objects=(
                _object_observation(
                    "plate_1",
                    visible=True,
                    attributes={
                        "source_kind": "detector",
                        "source_name": "grounded_sam2",
                        "evidence_kinds": ["rgb", "depth", "detector"],
                        "rgb_path": "rgb/0002.png",
                        "depth_path": "depth/0002.npy",
                    },
                ),
            ),
        ),
    )

    report = lab.coverage_collection_acceptance_report(
        plan,
        observations,
        observation_sequence_path="observations/visible-targets.json",
    )
    validation = lab.validate_coverage_collection_acceptance_report(report)

    assert validation["valid"] is True
    assert report["schema_version"] == (
        "dsg-spatialqa-lab.coverage-collection-acceptance-report.v1"
    )
    assert report["summary"] == {
        "accepted_location_evidence_target_count": 0,
        "accepted_state_evidence_target_count": 0,
        "accepted_target_count": 1,
        "location_evidence_acceptance_rate": 0.0,
        "location_evidence_ready": False,
        "location_evidence_target_count": 1,
        "planned_target_count": 1,
        "rejected_target_count": 1,
        "state_evidence_acceptance_rate": 0.0,
        "state_evidence_ready": False,
        "state_evidence_target_count": 1,
        "target_acceptance_rate": 1.0,
        "target_evidence_ready": True,
        "unaccepted_location_evidence_target_count": 1,
        "unaccepted_state_evidence_target_count": 1,
        "unaccepted_target_count": 0,
    }
    assert report["accepted_targets"] == [
        {
            "evidence_kinds": ["depth", "detector", "rgb"],
            "matched_by": ["object_id"],
            "object_id": "plate_1",
            "observation_steps": [2],
            "observed_object_ids": ["plate_1"],
            "source_kind": "detector",
            "source_name": "grounded_sam2",
        }
    ]
    assert report["rejected_observations"] == [
        {
            "object_id": "plate_1",
            "reason": "not_visible_detector_evidence",
            "step": 1,
        }
    ]


def test_coverage_collection_acceptance_report_matches_detector_aliases() -> None:
    plan = lab.coverage_collection_plan(
        _qa_observability_report(),
        _episode_frames(),
        qa_observability_report_path="reports/qa-observability.json",
        episode_paths=("episodes/episode-001.jsonl",),
    )
    observations = (
        lab.SceneObservation(
            step=3,
            agent_id="agent",
            agent_pose=lab.Pose3D(0.0, 0.9, 0.25, yaw=270.0),
            objects=(
                _object_observation(
                    "external_plate_track_7",
                    visible=True,
                    attributes={
                        "ai2thor_object_id": "Plate|-01.00|+00.90|+00.25",
                        "source_kind": "detector",
                        "source_name": "grounded_sam2",
                        "evidence_kinds": ["rgb", "depth", "detector"],
                        "rgb_path": "rgb/0003.png",
                        "depth_path": "depth/0003.npy",
                    },
                ),
            ),
        ),
    )

    report = lab.coverage_collection_acceptance_report(
        plan,
        observations,
        observation_sequence_path="observations/alias-targets.json",
    )

    assert report["summary"] == {
        "accepted_location_evidence_target_count": 0,
        "accepted_state_evidence_target_count": 0,
        "accepted_target_count": 1,
        "location_evidence_acceptance_rate": 0.0,
        "location_evidence_ready": False,
        "location_evidence_target_count": 1,
        "planned_target_count": 1,
        "rejected_target_count": 0,
        "state_evidence_acceptance_rate": 0.0,
        "state_evidence_ready": False,
        "state_evidence_target_count": 1,
        "target_acceptance_rate": 1.0,
        "target_evidence_ready": True,
        "unaccepted_location_evidence_target_count": 1,
        "unaccepted_state_evidence_target_count": 1,
        "unaccepted_target_count": 0,
    }
    assert report["accepted_targets"] == [
        {
            "evidence_kinds": ["depth", "detector", "rgb"],
            "matched_by": ["ai2thor_object_id"],
            "object_id": "plate_1",
            "observation_steps": [3],
            "observed_object_ids": ["external_plate_track_7"],
            "source_kind": "detector",
            "source_name": "grounded_sam2",
        }
    ]


def test_coverage_collection_acceptance_report_matches_handoff_target_alias() -> None:
    plan = lab.coverage_collection_plan(
        _qa_observability_report(),
        _episode_frames(),
        qa_observability_report_path="reports/qa-observability.json",
        episode_paths=("episodes/episode-001.jsonl",),
    )
    observations = (
        lab.SceneObservation(
            step=3,
            agent_id="agent",
            agent_pose=lab.Pose3D(0.0, 0.9, 0.25, yaw=270.0),
            objects=(
                _object_observation(
                    "producer_track_plate_7",
                    visible=True,
                    attributes={
                        "coverage_target_object_id": "plate_1",
                        "source_kind": "detector",
                        "source_name": "grounded_sam2",
                        "evidence_kinds": ["rgb", "depth", "detector"],
                        "rgb_path": "rgb/0003.png",
                        "depth_path": "depth/0003.npy",
                        "current_location_id": "counter_region",
                        "current_location_relation": "IN_REGION",
                        "states": {"isDirty": False},
                    },
                ),
            ),
        ),
    )

    report = lab.coverage_collection_acceptance_report(
        plan,
        observations,
        observation_sequence_path="observations/handoff-target-alias.json",
    )

    assert report["summary"]["accepted_target_count"] == 1
    assert report["summary"]["accepted_location_evidence_target_count"] == 1
    assert report["summary"]["accepted_state_evidence_target_count"] == 1
    assert report["accepted_targets"] == [
        {
            "evidence_kinds": ["depth", "detector", "rgb"],
            "matched_by": ["coverage_target_object_id"],
            "object_id": "plate_1",
            "observation_steps": [3],
            "observed_object_ids": ["producer_track_plate_7"],
            "source_kind": "detector",
            "source_name": "grounded_sam2",
        }
    ]
    assert report["accepted_location_evidence_targets"][0]["matched_by"] == [
        "coverage_target_object_id"
    ]
    assert report["accepted_state_evidence_targets"][0]["matched_by"] == [
        "coverage_target_object_id"
    ]


def test_coverage_collection_acceptance_report_rejects_non_detector_source_kind() -> None:
    plan = lab.coverage_collection_plan(
        _qa_observability_report(),
        _episode_frames(),
        qa_observability_report_path="reports/qa-observability.json",
        episode_paths=("episodes/episode-001.jsonl",),
    )
    observations = (
        lab.SceneObservation(
            step=2,
            agent_id="agent",
            agent_pose=lab.Pose3D(0.0, 0.9, 0.25, yaw=270.0),
            objects=(
                _object_observation(
                    "plate_1",
                    visible=True,
                    attributes={
                        "detector": "ai2thor_metadata_visible_objects",
                        "source_kind": "ai2thor",
                        "evidence_kinds": ["rgb", "depth", "detector"],
                        "rgb_path": "rgb/0002.png",
                        "depth_path": "depth/0002.npy",
                    },
                ),
            ),
        ),
    )

    report = lab.coverage_collection_acceptance_report(
        plan,
        observations,
        observation_sequence_path="observations/non-detector-source.json",
    )

    assert report["summary"]["accepted_target_count"] == 0
    assert report["summary"]["unaccepted_target_count"] == 1
    assert report["rejected_observations"] == [
        {
            "object_id": "plate_1",
            "reason": "not_visible_detector_evidence",
            "step": 2,
        }
    ]


def test_coverage_collection_acceptance_report_counts_state_evidence_only_with_states() -> None:
    plan = lab.coverage_collection_plan(
        _qa_observability_report(),
        _episode_frames(),
        qa_observability_report_path="reports/qa-observability.json",
        episode_paths=("episodes/episode-001.jsonl",),
    )
    observations = (
        lab.SceneObservation(
            step=2,
            agent_id="agent",
            agent_pose=lab.Pose3D(0.0, 0.9, 0.25, yaw=270.0),
            objects=(
                _object_observation(
                    "plate_1",
                    visible=True,
                    attributes={
                        "source_kind": "detector",
                        "source_name": "grounded_sam2",
                        "evidence_kinds": ["rgb", "depth", "detector"],
                        "rgb_path": "rgb/0002.png",
                        "depth_path": "depth/0002.npy",
                    },
                ),
            ),
        ),
        lab.SceneObservation(
            step=3,
            agent_id="agent",
            agent_pose=lab.Pose3D(0.0, 0.9, 0.25, yaw=270.0),
            objects=(
                _object_observation(
                    "plate_1",
                    visible=True,
                    attributes={
                        "source_kind": "detector",
                        "source_name": "grounded_sam2",
                        "evidence_kinds": ["rgb", "depth", "detector"],
                        "rgb_path": "rgb/0003.png",
                        "depth_path": "depth/0003.npy",
                        "states": {"isDirty": False},
                    },
                ),
            ),
        ),
    )

    report = lab.coverage_collection_acceptance_report(
        plan,
        observations,
        observation_sequence_path="observations/state-targets.json",
    )
    validation = lab.validate_coverage_collection_acceptance_report(report)

    assert validation["valid"] is True
    assert report["summary"]["accepted_target_count"] == 1
    assert report["summary"]["state_evidence_target_count"] == 1
    assert report["summary"]["accepted_state_evidence_target_count"] == 1
    assert report["summary"]["unaccepted_state_evidence_target_count"] == 0
    assert report["summary"]["state_evidence_acceptance_rate"] == 1.0
    assert report["summary"]["state_evidence_ready"] is True
    assert report["accepted_state_evidence_targets"] == [
        {
            "case_id": "case:plate",
            "matched_by": ["object_id"],
            "missing_state_edges": ["plate_1-STATE_CHANGED-state:plate_1:2-2"],
            "missing_state_nodes": ["state:plate_1:2"],
            "object_ids": ["plate_1"],
            "observation_steps": [3],
            "observed_object_ids": ["plate_1"],
            "state_attribute_keys": ["isDirty"],
        }
    ]
    assert report["unaccepted_state_evidence_targets"] == []


def test_coverage_collection_acceptance_report_counts_current_location_evidence() -> None:
    plan = lab.coverage_collection_plan(
        _qa_observability_report(),
        _episode_frames(),
        qa_observability_report_path="reports/qa-observability.json",
        episode_paths=("episodes/episode-001.jsonl",),
    )
    observations = (
        lab.SceneObservation(
            step=2,
            agent_id="agent",
            agent_pose=lab.Pose3D(0.0, 0.9, 0.25, yaw=270.0),
            objects=(
                _object_observation(
                    "plate_1",
                    visible=True,
                    attributes={
                        "source_kind": "detector",
                        "source_name": "grounded_sam2",
                        "evidence_kinds": ["rgb", "depth", "detector"],
                        "rgb_path": "rgb/0002.png",
                        "depth_path": "depth/0002.npy",
                    },
                ),
            ),
        ),
        lab.SceneObservation(
            step=3,
            agent_id="agent",
            agent_pose=lab.Pose3D(0.0, 0.9, 0.25, yaw=270.0),
            objects=(
                _object_observation(
                    "plate_1",
                    visible=True,
                    attributes={
                        "source_kind": "detector",
                        "source_name": "grounded_sam2",
                        "evidence_kinds": ["rgb", "depth", "detector"],
                        "rgb_path": "rgb/0003.png",
                        "depth_path": "depth/0003.npy",
                        "current_location_id": "counter_region",
                        "current_location_relation": "IN_REGION",
                    },
                ),
            ),
        ),
    )

    report = lab.coverage_collection_acceptance_report(
        plan,
        observations,
        observation_sequence_path="observations/location-targets.json",
    )
    validation = lab.validate_coverage_collection_acceptance_report(report)

    assert validation["valid"] is True
    assert report["summary"]["accepted_target_count"] == 1
    assert report["summary"]["location_evidence_target_count"] == 1
    assert report["summary"]["accepted_location_evidence_target_count"] == 1
    assert report["summary"]["unaccepted_location_evidence_target_count"] == 0
    assert report["summary"]["location_evidence_acceptance_rate"] == 1.0
    assert report["summary"]["location_evidence_ready"] is True
    assert report["accepted_location_evidence_targets"] == [
        {
            "current_location_id": "counter_region",
            "current_location_relation": "IN_REGION",
            "matched_by": ["object_id"],
            "object_id": "plate_1",
            "observation_steps": [3],
            "observed_object_ids": ["plate_1"],
        }
    ]
    assert report["unaccepted_location_evidence_targets"] == []


def test_coverage_collection_request_bundle_exposes_targets_without_gold_answers() -> None:
    assert hasattr(lab, "coverage_collection_request_bundle")
    plan = lab.coverage_collection_plan(
        _qa_observability_report(),
        _episode_frames(),
        qa_observability_report_path="reports/qa-observability.json",
        episode_paths=("episodes/episode-001.jsonl",),
    )

    bundle = lab.coverage_collection_request_bundle(
        plan,
        detector_jsonl_output_path="detector/coverage-targets.jsonl",
        observation_sequence_output_path="observations/coverage-targets.json",
        acceptance_report_output_path="reports/coverage-acceptance.json",
    )
    validation = lab.validate_coverage_collection_request_bundle(bundle)
    serialized = lab.coverage_collection_request_bundle_json(bundle)

    assert validation["valid"] is True
    assert bundle["schema_version"] == (
        "dsg-spatialqa-lab.coverage-collection-request-bundle.v1"
    )
    assert bundle["source_use_policy"] == {
        "episode_metadata_used_for": "collection_planning_only",
        "not_predicted_graph_evidence": True,
        "requires_visible_detector_evidence_before_claim": True,
    }
    assert bundle["planned_outputs"] == {
        "acceptance_report_output_path": "reports/coverage-acceptance.json",
        "detector_jsonl_output_path": "detector/coverage-targets.jsonl",
        "observation_sequence_output_path": "observations/coverage-targets.json",
    }
    assert bundle["required_evidence_kinds"] == ["depth", "detector", "rgb"]
    assert bundle["summary"]["target_count"] == 1
    assert bundle["summary"]["state_evidence_target_count"] == 1
    assert bundle["state_evidence_targets"] == [
        {
            "case_id": "case:plate",
            "missing_state_edges": ["plate_1-STATE_CHANGED-state:plate_1:2-2"],
            "missing_state_nodes": ["state:plate_1:2"],
            "object_ids": ["plate_1"],
            "required_state_contract": {
                "evidence_kinds": ["depth", "detector", "rgb"],
                "state_attributes_field": "attributes.states",
                "visible": True,
            },
            "target_nodes": ["plate_1"],
        }
    ]
    assert bundle["targets"] == [
        {
            "ai2thor_object_id": "Plate|-01.00|+00.90|+00.25",
            "episode_id": "episode-001",
            "label": "plate",
            "object_id": "plate_1",
            "pose": {"x": -1.0, "y": 0.9, "z": 0.25, "yaw": 0.0},
            "related_case_ids": ["case:plate"],
            "required_detection_contract": {
                "current_location_fields": [
                    "attributes.current_location_id",
                    "attributes.current_location_relation",
                ],
                "evidence_kinds": ["depth", "detector", "rgb"],
                "handoff_target_id_fields": [
                    "attributes.collection_target_object_id",
                    "attributes.coverage_target_object_id",
                    "attributes.target_object_id",
                ],
                "object_id": "plate_1",
                "source_kind": "detector",
                "supported_current_location_relations": [
                    "IN_REGION",
                    "IN_ROOM",
                    "INSIDE",
                    "ON",
                ],
                "visible": True,
            },
            "scene_id": "FloorPlan1",
            "suggested_action": "collect_visible_rgbd_detection",
            "viewpoint_hints": [
                {
                    "agent_pose": {"x": 0.0, "y": 0.9, "z": 0.25, "yaw": 0.0},
                    "distance_to_target": 1.0,
                    "episode_id": "episode-001",
                    "scene_id": "FloorPlan1",
                    "step": 2,
                    "suggested_yaw_to_target": 270.0,
                },
                {
                    "agent_pose": {"x": 0.0, "y": 0.9, "z": 0.0, "yaw": 0.0},
                    "distance_to_target": 1.030776,
                    "episode_id": "episode-001",
                    "scene_id": "FloorPlan1",
                    "step": 1,
                    "suggested_yaw_to_target": 284.036243,
                },
            ],
        }
    ]
    assert "gold_answer" not in serialized
    assert "gold_evidence" not in serialized


def test_coverage_collection_request_bundle_groups_targets_by_primary_viewpoint() -> None:
    plan = {
        "digest": "a" * 64,
        "summary": {
            "current_evidence_observable_count": 1,
            "target_evidence_observable_count": 4,
        },
        "collection_targets": [
            {
                "ai2thor_object_id": "Plate|-01.00|+00.90|+00.25",
                "episode_id": "episode-001",
                "label": "plate",
                "object_id": "plate_1",
                "pose": {"x": -1.0, "y": 0.9, "z": 0.25, "yaw": 0.0},
                "related_case_ids": ["case:shared", "case:plate"],
                "scene_id": "FloorPlan1",
                "suggested_action": "collect_visible_rgbd_detection",
                "viewpoint_hints": [
                    {
                        "agent_pose": {"x": 0.0, "y": 0.9, "z": 0.25, "yaw": 0.0},
                        "distance_to_target": 1.0,
                        "episode_id": "episode-001",
                        "scene_id": "FloorPlan1",
                        "step": 2,
                        "suggested_yaw_to_target": 270.0,
                    }
                ],
            },
            {
                "ai2thor_object_id": "Bowl|-01.20|+00.90|+00.30",
                "episode_id": "episode-001",
                "label": "bowl",
                "object_id": "bowl_1",
                "pose": {"x": -1.2, "y": 0.9, "z": 0.3, "yaw": 0.0},
                "related_case_ids": ["case:shared", "case:bowl"],
                "scene_id": "FloorPlan1",
                "suggested_action": "collect_visible_rgbd_detection",
                "viewpoint_hints": [
                    {
                        "agent_pose": {"x": 0.0, "y": 0.9, "z": 0.25, "yaw": 0.0},
                        "distance_to_target": 1.2,
                        "episode_id": "episode-001",
                        "scene_id": "FloorPlan1",
                        "step": 2,
                        "suggested_yaw_to_target": 272.0,
                    }
                ],
            },
        ],
        "relation_evidence_targets": [],
        "state_evidence_targets": [],
    }

    bundle = lab.coverage_collection_request_bundle(
        plan,
        detector_jsonl_output_path="detector/coverage-targets.jsonl",
        observation_sequence_output_path="observations/coverage-targets.json",
        acceptance_report_output_path="reports/coverage-acceptance.json",
    )

    assert bundle["summary"]["viewpoint_batch_count"] == 1
    assert bundle["summary"]["highest_yield_batch_related_case_count"] == 3
    assert bundle["summary"]["highest_yield_batch_target_count"] == 2
    assert bundle["summary"]["priority_batches_to_reach_target_case_count"] == 1
    assert bundle["summary"]["target_case_count_gap"] == 3
    assert bundle["viewpoint_batches"] == [
        {
            "agent_pose": {"x": 0.0, "y": 0.9, "z": 0.25, "yaw": 0.0},
            "batch_id": "episode-001:FloorPlan1:000002",
            "collection_action": "collect_visible_rgbd_detection",
            "cumulative_related_case_count": 3,
            "cumulative_target_count": 2,
            "episode_id": "episode-001",
            "execution_plan": {
                "ai2thor_actions": [
                    {
                        "action": "TeleportFull",
                        "position": {"x": 0.0, "y": 0.9, "z": 0.25},
                        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    },
                    {
                        "action": "RotateToTargetYaw",
                        "target_yaws": [270.0, 272.0],
                    },
                    {
                        "action": "CaptureVisibleRgbdDetection",
                        "required_evidence_kinds": ["depth", "detector", "rgb"],
                        "target_ids": ["bowl_1", "plate_1"],
                    },
                ],
                "acceptance_command_hint": (
                    "python scripts/plan_coverage_collection.py "
                    "--top-batch-return-tasks <handoff-jsonl> "
                    "--detector-jsonl <detector-jsonl> "
                    "--top-batch-return-report <report-json>"
                ),
                "not_predicted_graph_evidence": True,
            },
            "priority_rank": 1,
            "related_case_count": 3,
            "related_case_ids": ["case:bowl", "case:plate", "case:shared"],
            "scene_id": "FloorPlan1",
            "step": 2,
            "target_count": 2,
            "target_ids": ["bowl_1", "plate_1"],
            "targets": [
                {
                    "distance_to_target": 1.2,
                    "label": "bowl",
                    "object_id": "bowl_1",
                    "related_case_ids": ["case:bowl", "case:shared"],
                    "suggested_yaw_to_target": 272.0,
                },
                {
                    "distance_to_target": 1.0,
                    "label": "plate",
                    "object_id": "plate_1",
                    "related_case_ids": ["case:plate", "case:shared"],
                    "suggested_yaw_to_target": 270.0,
                },
            ],
        }
    ]
    assert lab.validate_coverage_collection_request_bundle(bundle)["valid"] is True


def test_coverage_collection_top_batch_handoff_exports_target_tasks_without_gold() -> None:
    bundle = _two_target_request_bundle()

    tasks = lab.coverage_collection_top_batch_handoff_tasks(
        bundle,
        max_priority_batches=1,
    )
    validation = lab.validate_coverage_collection_top_batch_handoff_tasks(tasks)
    serialized = lab.coverage_collection_top_batch_handoff_tasks_jsonl(tasks)

    assert validation["valid"] is True
    assert validation["task_count"] == 2
    assert validation["batch_count"] == 1
    assert validation["related_case_count"] == 3
    assert "gold_answer" not in serialized
    assert "gold_evidence" not in serialized
    assert [
        {
            key: value
            for key, value in task.items()
            if key not in {"task_digest"}
        }
        for task in tasks
    ] == [
        {
            "schema_version": (
                "dsg-spatialqa-lab.coverage-collection-target-task.v1"
            ),
            "action": "coverage_collection_target_task",
            "coverage_collection_request_bundle_digest": bundle["digest"],
            "source_use_policy": {
                "episode_metadata_used_for": "collection_planning_only",
                "not_predicted_graph_evidence": True,
                "requires_visible_detector_evidence_before_claim": True,
            },
            "task_id": "001:episode-001:FloorPlan1:000002:bowl_1",
            "priority_rank": 1,
            "batch_id": "episode-001:FloorPlan1:000002",
            "collection_action": "collect_visible_rgbd_detection",
            "episode_id": "episode-001",
            "scene_id": "FloorPlan1",
            "step": 2,
            "agent_pose": {"x": 0.0, "y": 0.9, "z": 0.25, "yaw": 0.0},
            "batch_execution_plan": {
                "ai2thor_actions": [
                    {
                        "action": "TeleportFull",
                        "position": {"x": 0.0, "y": 0.9, "z": 0.25},
                        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    },
                    {
                        "action": "RotateToTargetYaw",
                        "target_yaws": [270.0, 272.0],
                    },
                    {
                        "action": "CaptureVisibleRgbdDetection",
                        "required_evidence_kinds": ["depth", "detector", "rgb"],
                        "target_ids": ["bowl_1", "plate_1"],
                    },
                ],
                "acceptance_command_hint": (
                    "python scripts/plan_coverage_collection.py "
                    "--top-batch-return-tasks <handoff-jsonl> "
                    "--detector-jsonl <detector-jsonl> "
                    "--top-batch-return-report <report-json>"
                ),
                "not_predicted_graph_evidence": True,
            },
            "object_id": "bowl_1",
            "ai2thor_object_id": "Bowl|-01.20|+00.90|+00.30",
            "label": "bowl",
            "pose": {"x": -1.2, "y": 0.9, "z": 0.3, "yaw": 0.0},
            "distance_to_target": 1.2,
            "suggested_yaw_to_target": 272.0,
            "related_case_ids": ["case:bowl", "case:shared"],
            "planned_outputs": {
                "acceptance_report_output_path": "reports/coverage-acceptance.json",
                "detector_jsonl_output_path": "detector/coverage-targets.jsonl",
                "observation_sequence_output_path": "observations/coverage-targets.json",
            },
            "required_detection_contract": {
                "current_location_fields": [
                    "attributes.current_location_id",
                    "attributes.current_location_relation",
                ],
                "evidence_kinds": ["depth", "detector", "rgb"],
                "handoff_target_id_fields": [
                    "attributes.collection_target_object_id",
                    "attributes.coverage_target_object_id",
                    "attributes.target_object_id",
                ],
                "object_id": "bowl_1",
                "source_kind": "detector",
                "supported_current_location_relations": [
                    "IN_REGION",
                    "IN_ROOM",
                    "INSIDE",
                    "ON",
                ],
                "visible": True,
            },
            "state_evidence_required": False,
            "state_evidence_case_ids": [],
        },
        {
            "schema_version": (
                "dsg-spatialqa-lab.coverage-collection-target-task.v1"
            ),
            "action": "coverage_collection_target_task",
            "coverage_collection_request_bundle_digest": bundle["digest"],
            "source_use_policy": {
                "episode_metadata_used_for": "collection_planning_only",
                "not_predicted_graph_evidence": True,
                "requires_visible_detector_evidence_before_claim": True,
            },
            "task_id": "001:episode-001:FloorPlan1:000002:plate_1",
            "priority_rank": 1,
            "batch_id": "episode-001:FloorPlan1:000002",
            "collection_action": "collect_visible_rgbd_detection",
            "episode_id": "episode-001",
            "scene_id": "FloorPlan1",
            "step": 2,
            "agent_pose": {"x": 0.0, "y": 0.9, "z": 0.25, "yaw": 0.0},
            "batch_execution_plan": {
                "ai2thor_actions": [
                    {
                        "action": "TeleportFull",
                        "position": {"x": 0.0, "y": 0.9, "z": 0.25},
                        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    },
                    {
                        "action": "RotateToTargetYaw",
                        "target_yaws": [270.0, 272.0],
                    },
                    {
                        "action": "CaptureVisibleRgbdDetection",
                        "required_evidence_kinds": ["depth", "detector", "rgb"],
                        "target_ids": ["bowl_1", "plate_1"],
                    },
                ],
                "acceptance_command_hint": (
                    "python scripts/plan_coverage_collection.py "
                    "--top-batch-return-tasks <handoff-jsonl> "
                    "--detector-jsonl <detector-jsonl> "
                    "--top-batch-return-report <report-json>"
                ),
                "not_predicted_graph_evidence": True,
            },
            "object_id": "plate_1",
            "ai2thor_object_id": "Plate|-01.00|+00.90|+00.25",
            "label": "plate",
            "pose": {"x": -1.0, "y": 0.9, "z": 0.25, "yaw": 0.0},
            "distance_to_target": 1.0,
            "suggested_yaw_to_target": 270.0,
            "related_case_ids": ["case:plate", "case:shared"],
            "planned_outputs": {
                "acceptance_report_output_path": "reports/coverage-acceptance.json",
                "detector_jsonl_output_path": "detector/coverage-targets.jsonl",
                "observation_sequence_output_path": "observations/coverage-targets.json",
            },
            "required_detection_contract": {
                "current_location_fields": [
                    "attributes.current_location_id",
                    "attributes.current_location_relation",
                ],
                "evidence_kinds": ["depth", "detector", "rgb"],
                "handoff_target_id_fields": [
                    "attributes.collection_target_object_id",
                    "attributes.coverage_target_object_id",
                    "attributes.target_object_id",
                ],
                "object_id": "plate_1",
                "source_kind": "detector",
                "supported_current_location_relations": [
                    "IN_REGION",
                    "IN_ROOM",
                    "INSIDE",
                    "ON",
                ],
                "visible": True,
            },
            "state_evidence_required": True,
            "state_evidence_case_ids": ["case:plate-state"],
        },
    ]


def test_coverage_collection_cli_writes_top_batch_handoff_jsonl(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_coverage_plan_script()
    main = cast(MainFn, getattr(module, "main"))
    bundle_path = tmp_path / "coverage-request-bundle.json"
    handoff_path = tmp_path / "top-batches.jsonl"
    lab.save_coverage_collection_request_bundle(_two_target_request_bundle(), bundle_path)

    assert main(
        [
            "--top-batch-handoff-request-bundle",
            str(bundle_path),
            "--top-batch-handoff-jsonl",
            str(handoff_path),
            "--max-priority-batches",
            "1",
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    tasks = lab.load_coverage_collection_top_batch_handoff_tasks(handoff_path)
    assert output["action"] == "coverage_collection_top_batch_handoff"
    assert output["valid"] is True
    assert output["task_count"] == 2
    assert output["digest"] == lab.coverage_collection_top_batch_handoff_tasks_digest(
        tasks
    )

    assert main(["--validate-top-batch-handoff", str(handoff_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_coverage_collection_top_batch_handoff"
    assert validation["valid"] is True


def test_coverage_collection_top_batch_return_report_splits_target_location_and_state() -> None:
    tasks = lab.coverage_collection_top_batch_handoff_tasks(
        _two_target_request_bundle(),
        max_priority_batches=1,
    )
    observations = (
        lab.SceneObservation(
            step=2,
            agent_id="agent",
            agent_pose=lab.Pose3D(0.0, 0.9, 0.25, yaw=270.0),
            objects=(
                _object_observation(
                    "plate_1",
                    visible=True,
                    attributes={
                        "source_kind": "detector",
                        "source_name": "grounded_sam2",
                        "evidence_kinds": ["rgb", "depth", "detector"],
                        "rgb_path": "rgb/0002.png",
                        "depth_path": "depth/0002.npy",
                        "current_location_id": "counter_region",
                        "current_location_relation": "IN_REGION",
                        "states": {"isDirty": False},
                    },
                ),
                _object_observation(
                    "external_bowl_track",
                    visible=True,
                    attributes={
                        "ai2thor_object_id": "Bowl|-01.20|+00.90|+00.30",
                        "source_kind": "detector",
                        "source_name": "grounded_sam2",
                        "evidence_kinds": ["rgb", "depth", "detector"],
                        "rgb_path": "rgb/0002.png",
                        "depth_path": "depth/0002.npy",
                    },
                ),
            ),
        ),
    )

    report = lab.coverage_collection_top_batch_return_report(
        tasks,
        observations,
        observation_sequence_path="observations/top-batch-return.json",
    )
    validation = lab.validate_coverage_collection_top_batch_return_report(report)

    assert validation["valid"] is True
    assert report["schema_version"] == (
        "dsg-spatialqa-lab.coverage-collection-top-batch-return-report.v1"
    )
    assert report["summary"] == {
        "accepted_location_task_count": 1,
        "accepted_state_task_count": 1,
        "accepted_target_task_count": 2,
        "batch_count": 1,
        "location_evidence_ready": False,
        "location_required_task_count": 2,
        "related_case_count": 3,
        "return_ready": False,
        "state_evidence_ready": True,
        "state_required_task_count": 1,
        "target_acceptance_rate": 1.0,
        "target_evidence_ready": True,
        "task_count": 2,
        "unaccepted_location_task_count": 1,
        "unaccepted_state_task_count": 0,
        "unaccepted_target_task_count": 0,
    }
    assert report["accepted_target_tasks"] == [
        {
            "evidence_kinds": ["depth", "detector", "rgb"],
            "matched_by": ["ai2thor_object_id"],
            "object_id": "bowl_1",
            "observed_object_ids": ["external_bowl_track"],
            "observation_steps": [2],
            "task_id": "001:episode-001:FloorPlan1:000002:bowl_1",
        },
        {
            "evidence_kinds": ["depth", "detector", "rgb"],
            "matched_by": ["object_id"],
            "object_id": "plate_1",
            "observed_object_ids": ["plate_1"],
            "observation_steps": [2],
            "task_id": "001:episode-001:FloorPlan1:000002:plate_1",
        },
    ]
    assert report["unaccepted_location_tasks"] == [
        {
            "object_id": "bowl_1",
            "reason": "missing_current_location_evidence",
            "task_id": "001:episode-001:FloorPlan1:000002:bowl_1",
        }
    ]
    assert report["accepted_state_tasks"] == [
        {
            "object_id": "plate_1",
            "state_attribute_keys": ["isDirty"],
            "state_evidence_case_ids": ["case:plate-state"],
            "task_id": "001:episode-001:FloorPlan1:000002:plate_1",
        }
    ]


def test_coverage_collection_top_batch_return_report_matches_handoff_target_alias() -> None:
    tasks = lab.coverage_collection_top_batch_handoff_tasks(
        _two_target_request_bundle(),
        max_priority_batches=1,
    )
    observations = (
        lab.SceneObservation(
            step=2,
            agent_id="agent",
            agent_pose=lab.Pose3D(0.0, 0.9, 0.25, yaw=270.0),
            objects=(
                _object_observation(
                    "producer_track_bowl_12",
                    visible=True,
                    attributes={
                        "target_object_id": "bowl_1",
                        "source_kind": "detector",
                        "source_name": "grounded_sam2",
                        "evidence_kinds": ["rgb", "depth", "detector"],
                        "rgb_path": "rgb/0002.png",
                        "depth_path": "depth/0002.npy",
                        "current_location_id": "counter_region",
                        "current_location_relation": "IN_REGION",
                    },
                ),
            ),
        ),
    )

    report = lab.coverage_collection_top_batch_return_report(
        tasks,
        observations,
        observation_sequence_path="observations/top-batch-target-alias.json",
    )

    assert report["summary"]["accepted_target_task_count"] == 1
    assert report["summary"]["accepted_location_task_count"] == 1
    assert report["accepted_target_tasks"] == [
        {
            "evidence_kinds": ["depth", "detector", "rgb"],
            "matched_by": ["target_object_id"],
            "object_id": "bowl_1",
            "observed_object_ids": ["producer_track_bowl_12"],
            "observation_steps": [2],
            "task_id": "001:episode-001:FloorPlan1:000002:bowl_1",
        }
    ]
    assert report["accepted_location_tasks"] == [
        {
            "current_location_id": "counter_region",
            "current_location_relation": "IN_REGION",
            "object_id": "bowl_1",
            "task_id": "001:episode-001:FloorPlan1:000002:bowl_1",
        }
    ]


def test_coverage_collection_cli_writes_top_batch_return_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_coverage_plan_script()
    main = cast(MainFn, getattr(module, "main"))
    tasks_path = tmp_path / "top-batches.jsonl"
    sequence_path = tmp_path / "top-batch-observations.json"
    report_path = tmp_path / "top-batch-return-report.json"
    lab.save_coverage_collection_top_batch_handoff_tasks(
        lab.coverage_collection_top_batch_handoff_tasks(
            _two_target_request_bundle(),
            max_priority_batches=1,
        ),
        tasks_path,
    )
    lab.save_scene_observation_sequence(
        (
            lab.SceneObservation(
                step=2,
                agent_id="agent",
                agent_pose=lab.Pose3D(0.0, 0.9, 0.25, yaw=270.0),
                objects=(
                    _object_observation(
                        "plate_1",
                        visible=True,
                        attributes={
                            "source_kind": "detector",
                            "source_name": "grounded_sam2",
                            "evidence_kinds": ["rgb", "depth", "detector"],
                            "current_location_id": "counter_region",
                            "current_location_relation": "IN_REGION",
                            "states": {"isDirty": False},
                        },
                    ),
                ),
            ),
        ),
        sequence_path,
    )

    assert main(
        [
            "--top-batch-return-tasks",
            str(tasks_path),
            "--observation-sequence",
            str(sequence_path),
            "--top-batch-return-report",
            str(report_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    report = lab.load_coverage_collection_top_batch_return_report(report_path)
    assert output["action"] == "coverage_collection_top_batch_return_report"
    assert output["valid"] is True
    assert output["summary"]["accepted_target_task_count"] == 1
    assert output["summary"]["return_ready"] is False
    assert output["digest"] == lab.coverage_collection_top_batch_return_report_digest(
        report
    )

    assert main(["--validate-top-batch-return-report", str(report_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_coverage_collection_top_batch_return_report"
    assert validation["valid"] is True


def test_coverage_collection_cli_writes_acceptance_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_coverage_plan_script()
    main = cast(MainFn, getattr(module, "main"))
    plan = lab.coverage_collection_plan(
        _qa_observability_report(),
        _episode_frames(),
        qa_observability_report_path="reports/qa-observability.json",
        episode_paths=("episodes/episode-001.jsonl",),
    )
    plan_path = tmp_path / "coverage-plan.json"
    sequence_path = tmp_path / "detector-observations.json"
    report_path = tmp_path / "coverage-acceptance.json"
    lab.save_coverage_collection_plan(plan, plan_path)
    lab.save_scene_observation_sequence(
        (
            lab.SceneObservation(
                step=1,
                agent_id="agent",
                agent_pose=lab.Pose3D(0.0, 0.9, 0.0, yaw=0.0),
                objects=(
                    _object_observation(
                        "plate_1",
                        visible=True,
                        attributes={
                            "source_kind": "detector",
                            "source_name": "grounded_sam2",
                            "evidence_kinds": ["rgb", "depth", "detector"],
                        },
                    ),
                ),
            ),
        ),
        sequence_path,
    )

    assert main(
        [
            "--acceptance-plan",
            str(plan_path),
            "--observation-sequence",
            str(sequence_path),
            "--acceptance-report",
            str(report_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    report = lab.load_coverage_collection_acceptance_report(report_path)
    assert output["action"] == "coverage_collection_acceptance_report"
    assert output["valid"] is True
    assert output["summary"]["accepted_target_count"] == 1
    assert output["digest"] == lab.coverage_collection_acceptance_report_digest(report)


def test_coverage_collection_cli_writes_request_bundle(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_coverage_plan_script()
    main = cast(MainFn, getattr(module, "main"))
    plan = lab.coverage_collection_plan(
        _qa_observability_report(),
        _episode_frames(),
        qa_observability_report_path="reports/qa-observability.json",
        episode_paths=("episodes/episode-001.jsonl",),
    )
    plan_path = tmp_path / "coverage-plan.json"
    bundle_path = tmp_path / "coverage-request-bundle.json"
    lab.save_coverage_collection_plan(plan, plan_path)

    assert main(
        [
            "--request-plan",
            str(plan_path),
            "--request-bundle",
            str(bundle_path),
            "--detector-jsonl-output",
            "detector/coverage-targets.jsonl",
            "--observation-sequence-output",
            "observations/coverage-targets.json",
            "--acceptance-report-output",
            "reports/coverage-acceptance.json",
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    bundle = lab.load_coverage_collection_request_bundle(bundle_path)
    assert output["action"] == "coverage_collection_request_bundle"
    assert output["valid"] is True
    assert output["summary"]["target_count"] == 1
    assert output["digest"] == lab.coverage_collection_request_bundle_digest(bundle)


def _two_target_request_bundle() -> dict[str, object]:
    plan = {
        "digest": "a" * 64,
        "summary": {
            "current_evidence_observable_count": 1,
            "target_evidence_observable_count": 4,
        },
        "collection_targets": [
            {
                "ai2thor_object_id": "Plate|-01.00|+00.90|+00.25",
                "episode_id": "episode-001",
                "label": "plate",
                "object_id": "plate_1",
                "pose": {"x": -1.0, "y": 0.9, "z": 0.25, "yaw": 0.0},
                "related_case_ids": ["case:shared", "case:plate"],
                "scene_id": "FloorPlan1",
                "suggested_action": "collect_visible_rgbd_detection",
                "viewpoint_hints": [
                    {
                        "agent_pose": {"x": 0.0, "y": 0.9, "z": 0.25, "yaw": 0.0},
                        "distance_to_target": 1.0,
                        "episode_id": "episode-001",
                        "scene_id": "FloorPlan1",
                        "step": 2,
                        "suggested_yaw_to_target": 270.0,
                    }
                ],
            },
            {
                "ai2thor_object_id": "Bowl|-01.20|+00.90|+00.30",
                "episode_id": "episode-001",
                "label": "bowl",
                "object_id": "bowl_1",
                "pose": {"x": -1.2, "y": 0.9, "z": 0.3, "yaw": 0.0},
                "related_case_ids": ["case:shared", "case:bowl"],
                "scene_id": "FloorPlan1",
                "suggested_action": "collect_visible_rgbd_detection",
                "viewpoint_hints": [
                    {
                        "agent_pose": {"x": 0.0, "y": 0.9, "z": 0.25, "yaw": 0.0},
                        "distance_to_target": 1.2,
                        "episode_id": "episode-001",
                        "scene_id": "FloorPlan1",
                        "step": 2,
                        "suggested_yaw_to_target": 272.0,
                    }
                ],
            },
        ],
        "relation_evidence_targets": [],
        "state_evidence_targets": [
            {
                "case_id": "case:plate-state",
                "missing_state_edges": ["plate_1-STATE_CHANGED-state:plate_1:2-2"],
                "missing_state_nodes": ["state:plate_1:2"],
                "object_ids": ["plate_1"],
                "target_nodes": ["plate_1"],
            }
        ],
    }
    return lab.coverage_collection_request_bundle(
        plan,
        detector_jsonl_output_path="detector/coverage-targets.jsonl",
        observation_sequence_output_path="observations/coverage-targets.json",
        acceptance_report_output_path="reports/coverage-acceptance.json",
    )


def _qa_observability_report() -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": "dsg-spatialqa-lab.qa-observability-report.v1",
        "qa_path": "qa.jsonl",
        "qa_digest": "0" * 64,
        "graph_path": "predicted-graph.json",
        "graph_digest": "1" * 64,
        "summary": {
            "case_count": 2,
            "evidence_observable_count": 1,
            "missing_target_nodes": ["plate_1"],
            "target_node_recall": 0.5,
        },
        "splits": {
            "evidence_observable": ["case:cup"],
            "missing_evidence": ["case:plate", "case:mug"],
            "target_observable": ["case:cup", "case:mug"],
            "target_observable_relation_missing": ["case:mug"],
        },
        "cases": [
            {
                "case_id": "case:plate",
                "observability_status": "target_missing",
                "target_nodes": ["plate_1"],
                "missing_target_nodes": ["plate_1"],
                "missing_required_nodes": ["state:plate_1:2"],
                "missing_required_edges": [
                    "plate_1-STATE_CHANGED-state:plate_1:2-2"
                ],
                "missing_required_edge_relations": ["STATE_CHANGED"],
            },
            {
                "case_id": "case:mug",
                "observability_status": "target_observable_relation_missing",
                "target_nodes": ["mug_1"],
                "missing_target_nodes": [],
                "missing_required_edges": ["mug_1-ON-table_1-1"],
                "missing_required_edge_relations": ["ON"],
            },
        ],
    }
    report["report_digest"] = lab.qa_observability_report_digest(report)
    return report


def _episode_frames() -> tuple[lab.EpisodeFrame, ...]:
    metadata = {
        "adapter": "ai2thor",
        "source_kind": "real_simulator",
        "objects": [
            {
                "object_id": "plate_1",
                "label": "plate",
                "visible": False,
                "pose": {"x": -1.0, "y": 0.9, "z": 0.25, "yaw": 0.0},
                "attributes": {"ai2thor_object_id": "Plate|-01.00|+00.90|+00.25"},
            },
            {
                "object_id": "mug_1",
                "label": "mug",
                "visible": True,
                "pose": {"x": 0.0, "y": 0.9, "z": 0.0, "yaw": 0.0},
                "attributes": {"ai2thor_object_id": "Mug|+00.00|+00.90|+00.00"},
            },
        ],
    }
    return (
        lab.EpisodeFrame(
            episode_id="episode-001",
            scene_id="FloorPlan1",
            step=1,
            rgb_path="rgb/0001.png",
            depth_path="depth/0001.npy",
            segmentation_path="seg/0001.png",
            agent_id="agent",
            agent_pose=lab.Pose3D(0.0, 0.9, 0.0, yaw=0.0),
            action="Initialize",
            visible_object_ids=("mug_1",),
            metadata=metadata,
        ),
        lab.EpisodeFrame(
            episode_id="episode-001",
            scene_id="FloorPlan1",
            step=2,
            rgb_path="rgb/0002.png",
            depth_path="depth/0002.npy",
            segmentation_path="seg/0002.png",
            agent_id="agent",
            agent_pose=lab.Pose3D(0.0, 0.9, 0.25, yaw=0.0),
            action="MoveAhead",
            visible_object_ids=("mug_1",),
            metadata=metadata,
        ),
    )


def _object_observation(
    object_id: str,
    *,
    visible: bool,
    attributes: dict[str, object],
) -> lab.ObjectObservation:
    return lab.ObjectObservation(
        object_id=object_id,
        label="plate",
        pose=lab.Pose3D(-1.0, 0.9, 0.25, yaw=0.0),
        bbox=lab.BBox3D(
            center=lab.Pose3D(-1.0, 0.9, 0.25, yaw=0.0),
            size=(0.2, 0.2, 0.2),
        ),
        confidence=0.9,
        visible=visible,
        attributes=attributes,
    )
