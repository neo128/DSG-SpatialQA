from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from dsg_spatialqa_lab.eval.dsg_detector_recall import (
    dsg_detector_recall_handoff,
    dsg_detector_recall_handoff_from_query_diagnostics,
    dsg_detector_recall_handoff_json,
    validate_dsg_detector_recall_handoff,
)


ROOT = Path(__file__).resolve().parents[1]
DETECTOR_RECALL_SCRIPT = ROOT / "scripts" / "build_dsg_detector_recall_handoff.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_detector_recall_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_dsg_detector_recall_handoff_script",
        DETECTOR_RECALL_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dsg_detector_recall_handoff_uses_frame_visible_supports_without_gold() -> None:
    gap_report = {
        "schema_version": "dsg-spatialqa-lab.next-optimization-targets.v1",
        "report_digest": "a" * 64,
        "on_to_in_room_cases": [
            {
                "case_id": "case-apple",
                "episode_id": "episode-001",
                "gold_support_label": "countertop",
                "step": 100034,
                "target_label": "apple",
            }
        ],
        "on_to_none_cases": [
            {
                "case_id": "case-card",
                "episode_id": "episode-001",
                "gold_support_label": "drawer",
                "step": 100034,
                "target_label": "creditcard",
            }
        ],
    }
    frame_index = [
        {
            "detector_depth_path": "depth/000034.npy",
            "detector_rgb_path": "rgb/000034.ppm",
            "detector_segmentation_path": "segmentation/000034.ppm",
            "episode_id": "episode-001",
            "scene_id": "FloorPlan1",
            "step": 34,
            "visible_object_labels": ["apple", "chair", "floor"],
        }
    ]

    handoff = dsg_detector_recall_handoff(gap_report, frame_index)

    assert handoff["required_frames"] == [
        {
            "case_ids": ["case-apple", "case-card"],
            "depth_path": "depth/000034.npy",
            "episode_id": "episode-001",
            "frame_step": 34,
            "original_case_steps": [100034],
            "requested_detection_labels": ["apple", "chair", "creditcard"],
            "rgb_path": "rgb/000034.ppm",
            "scene_id": "FloorPlan1",
            "segmentation_path": "segmentation/000034.ppm",
            "support_labels": ["chair"],
            "target_labels": ["apple", "creditcard"],
        }
    ]
    assert handoff["summary"] == {
        "case_count": 2,
        "frame_count": 1,
        "frames_with_support_labels": 1,
        "missing_frame_case_count": 0,
        "requested_detection_label_count": 3,
        "support_label_count": 1,
        "target_label_count": 2,
    }
    serialized = dsg_detector_recall_handoff_json(handoff)
    assert "gold_support_label" not in serialized
    assert "countertop" not in serialized
    assert "drawer" not in serialized
    assert validate_dsg_detector_recall_handoff(handoff)["valid"] is True


def test_dsg_detector_recall_handoff_validation_rejects_evaluator_only_fields() -> None:
    handoff = {
        "schema_version": "dsg-spatialqa-lab.dsg-detector-recall-handoff.v1",
        "required_frames": [
            {
                "episode_id": "episode-001",
                "frame_step": 34,
                "gold_support_label": "countertop",
                "scene_id": "FloorPlan1",
            }
        ],
        "summary": {},
    }

    validation = validate_dsg_detector_recall_handoff(handoff)

    assert validation["valid"] is False
    assert "forbidden_evaluator_only_fields_absent" in [
        check["name"] for check in validation["checks"] if check["passed"] is False
    ]


def test_dsg_detector_recall_handoff_from_query_diagnostics_targets_unresolved_cases() -> None:
    query_report = {
        "schema_version": "dsg-spatialqa-lab.object-location-query-diagnostic-report.v1",
        "report_digest": "c" * 64,
        "cases": [
            {
                "case_id": "episode-001:FloorPlan1:0001:object_location:apple_00_47:observation_aware:100034",
                "location_evidence_status": "query_error",
                "object_id": "apple_00_47",
                "semantic_match": False,
            },
            {
                "case_id": "episode-001:FloorPlan1:0002:object_location:creditcard_01_94:observation_aware:100034",
                "location_evidence_status": "support_fallback_missing",
                "object_id": "creditcard_01_94",
                "semantic_match": False,
            },
            {
                "case_id": "episode-001:FloorPlan1:0003:object_location:book_00_15:observation_aware:100034",
                "location_evidence_status": "support_fallback_missing",
                "object_id": "book_00_15",
                "semantic_match": True,
            },
            {
                "case_id": "episode-001:FloorPlan1:0004:object_location:mug_01_00:observation_aware:100034",
                "location_evidence_status": "explicit_location_edge",
                "object_id": "mug_01_00",
                "semantic_match": False,
            },
        ],
    }
    frame_index = [
        {
            "asset_paths": {
                "depth": "depth/000034.npy",
                "rgb": "rgb/000034.ppm",
                "segmentation": "segmentation/000034.ppm",
            },
            "episode_id": "episode-001",
            "scene_id": "FloorPlan1",
            "step": 34,
            "visible_object_labels": ["countertop", "chair", "floor"],
        }
    ]

    handoff = dsg_detector_recall_handoff_from_query_diagnostics(
        query_report,
        frame_index,
    )

    assert handoff["gap_report_digest"] == "c" * 64
    assert handoff["query_diagnostic_source"] == {
        "included_statuses": ["query_error", "support_fallback_missing"],
        "report_digest": "c" * 64,
    }
    assert handoff["required_frames"] == [
        {
            "case_ids": [
                "episode-001:FloorPlan1:0001:object_location:apple_00_47:observation_aware:100034",
                "episode-001:FloorPlan1:0002:object_location:creditcard_01_94:observation_aware:100034",
            ],
            "depth_path": "depth/000034.npy",
            "episode_id": "episode-001",
            "frame_step": 34,
            "original_case_steps": [100034],
            "requested_detection_labels": [
                "apple",
                "chair",
                "countertop",
                "creditcard",
            ],
            "rgb_path": "rgb/000034.ppm",
            "scene_id": "FloorPlan1",
            "segmentation_path": "segmentation/000034.ppm",
            "support_labels": ["chair", "countertop"],
            "target_labels": ["apple", "creditcard"],
        }
    ]
    serialized = dsg_detector_recall_handoff_json(handoff)
    assert "gold" not in serialized
    assert validate_dsg_detector_recall_handoff(handoff)["valid"] is True


def test_dsg_detector_recall_handoff_cli_writes_and_validates_without_gold(
    tmp_path: Path,
) -> None:
    module = load_detector_recall_script()
    main = cast(MainFn, getattr(module, "main"))
    gap_report = tmp_path / "gap-report.json"
    frame_index = tmp_path / "frame-index.jsonl"
    output = tmp_path / "handoff.json"
    gap_report.write_text(
        json.dumps(
            {
                "on_to_in_room_cases": [
                    {
                        "case_id": "case-apple",
                        "episode_id": "episode-001",
                        "gold_support_label": "countertop",
                        "step": 100034,
                        "target_label": "apple",
                    }
                ],
                "report_digest": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    frame_index.write_text(
        json.dumps(
            {
                "detector_depth_path": "depth.npy",
                "detector_rgb_path": "rgb.ppm",
                "episode_id": "episode-001",
                "scene_id": "FloorPlan1",
                "step": 34,
                "visible_object_labels": ["apple", "countertop"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(
        [
            "--gap-report",
            str(gap_report),
            "--frame-index-jsonl",
            str(frame_index),
            "--output",
            str(output),
        ]
    ) == 0
    assert main(["--validate-report", str(output)]) == 0

    serialized = output.read_text(encoding="utf-8")
    assert "gold_support_label" not in serialized
    assert json.loads(serialized)["required_frames"][0]["support_labels"] == [
        "countertop"
    ]


def test_dsg_detector_recall_handoff_cli_accepts_query_diagnostic_report(
    tmp_path: Path,
) -> None:
    module = load_detector_recall_script()
    main = cast(MainFn, getattr(module, "main"))
    query_report = tmp_path / "query-diagnostics.json"
    frame_index = tmp_path / "frame-index.jsonl"
    output = tmp_path / "handoff.json"
    query_report.write_text(
        json.dumps(
            {
                "report_digest": "d" * 64,
                "cases": [
                    {
                        "case_id": "episode-001:FloorPlan1:0001:object_location:apple_00_47:observation_aware:100034",
                        "location_evidence_status": "query_error",
                        "object_id": "apple_00_47",
                        "semantic_match": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    frame_index.write_text(
        json.dumps(
            {
                "detector_depth_path": "depth.npy",
                "detector_rgb_path": "rgb.ppm",
                "episode_id": "episode-001",
                "scene_id": "FloorPlan1",
                "step": 34,
                "visible_object_labels": ["countertop"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(
        [
            "--query-diagnostic-report",
            str(query_report),
            "--frame-index-jsonl",
            str(frame_index),
            "--output",
            str(output),
        ]
    ) == 0

    handoff = json.loads(output.read_text(encoding="utf-8"))
    assert handoff["query_diagnostic_source"]["report_digest"] == "d" * 64
    assert handoff["required_frames"][0]["target_labels"] == ["apple"]
