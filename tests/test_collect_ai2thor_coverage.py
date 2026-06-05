from __future__ import annotations

from external_tools.collect_ai2thor_coverage import (
    coverage_capture_step,
    detection_from_ai2thor_object,
    nearest_reachable_positions,
    stable_ai2thor_object_id,
    yaw_to_target,
)


def test_stable_ai2thor_object_id_matches_benchmark_ids() -> None:
    assert (
        stable_ai2thor_object_id("Apple|-00.47|+01.15|+00.48")
        == "apple_00_47_01_15_00_48"
    )
    assert (
        stable_ai2thor_object_id("AlarmClock|-01.28|+01.24|+00.47")
        == "alarmclock_01_28_01_24_00_47"
    )
    assert (
        stable_ai2thor_object_id("Sink|-03.12|-00.01|+01.53|SinkBasin")
        == "sink_03_12_00_01_01_53_sinkbasin"
    )


def test_detection_from_ai2thor_object_preserves_visible_current_location() -> None:
    detection = detection_from_ai2thor_object(
        {
            "axisAlignedBoundingBox": {
                "center": {"x": -0.47, "y": 1.15, "z": 0.48},
                "size": {"x": 0.1, "y": 0.2, "z": 0.3},
            },
            "isBroken": False,
            "isOpen": False,
            "objectId": "Apple|-00.47|+01.15|+00.48",
            "objectType": "Apple",
            "parentReceptacles": ["CounterTop|-01.87|+00.95|-01.21"],
            "pickupable": True,
            "position": {"x": -0.47, "y": 1.15, "z": 0.48},
            "visible": True,
        },
        bbox_2d_xyxy=[1, 2, 10, 12],
        frame_paths={
            "depth_path": "depth/000001.npy",
            "mask_path": "masks/apple.ppm",
            "rgb_path": "rgb/000001.ppm",
            "segmentation_path": "segmentation/000001.ppm",
        },
        target_ids={"apple_00_47_01_15_00_48"},
    )

    assert detection["object_id"] == "apple_00_47_01_15_00_48"
    assert detection["label"] == "apple"
    assert detection["evidence_kinds"] == ["depth", "detector", "rgb"]
    assert detection["visible"] is True
    assert detection["bbox_2d_xyxy"] == [1, 2, 10, 12]
    assert detection["attributes"]["ai2thor_object_id"] == (
        "Apple|-00.47|+01.15|+00.48"
    )
    assert detection["attributes"]["current_location_id"] == (
        "countertop_01_87_00_95_01_21"
    )
    assert detection["attributes"]["current_location_relation"] == "ON"
    assert detection["attributes"]["collection_target_object_id"] == (
        "apple_00_47_01_15_00_48"
    )
    assert detection["attributes"]["states"] == {
        "isBroken": False,
        "isOpen": False,
        "pickupable": True,
    }


def test_yaw_to_target_matches_ai2thor_heading_convention() -> None:
    assert yaw_to_target({"x": 0.0, "z": -1.0}, {"x": 0.241, "z": -1.244}) == 135.35
    assert yaw_to_target({"x": -0.75, "z": -0.5}, {"x": -1.277, "z": -0.473}) == 272.93


def test_nearest_reachable_positions_are_sorted_stably() -> None:
    positions = [
        {"x": 2.0, "y": 0.9, "z": 1.0},
        {"x": 0.0, "y": 0.9, "z": -1.0},
        {"x": 0.5, "y": 0.9, "z": -1.0},
        {"x": 0.0, "y": 0.9, "z": -0.5},
    ]
    nearest = nearest_reachable_positions(
        positions,
        {"x": 0.24, "z": -1.24},
        limit=3,
    )

    assert nearest == [
        {"x": 0.0, "y": 0.9, "z": -1.0},
        {"x": 0.5, "y": 0.9, "z": -1.0},
        {"x": 0.0, "y": 0.9, "z": -0.5},
    ]


def test_coverage_capture_step_applies_positive_offset() -> None:
    assert coverage_capture_step(1, step_offset=100000) == 100001
    assert coverage_capture_step(48, step_offset=100000) == 100048
    assert coverage_capture_step(1, step_offset=0) == 1


def test_coverage_capture_step_rejects_invalid_values() -> None:
    import pytest

    with pytest.raises(ValueError, match="capture_index must be positive"):
        coverage_capture_step(0, step_offset=100000)
    with pytest.raises(ValueError, match="step_offset must be non-negative"):
        coverage_capture_step(1, step_offset=-1)
