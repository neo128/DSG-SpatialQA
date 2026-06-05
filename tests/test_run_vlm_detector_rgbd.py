from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from dsg_spatialqa_lab.observations import detector_observation_sequence_from_jsonl


ROOT = Path(__file__).resolve().parents[1]
RUN_VLM_DETECTOR_SCRIPT = ROOT / "external_tools" / "run_vlm_detector_rgbd.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_run_vlm_detector_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_vlm_detector_rgbd_script",
        RUN_VLM_DETECTOR_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_vlm_detector_requires_explicit_network_permission(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_detector_script()
    main = cast(MainFn, getattr(module, "main"))
    handoff = _handoff(tmp_path)

    exit_code = main(
        [
            "--handoff",
            str(handoff),
            "--output-detector-jsonl",
            str(tmp_path / "detector.jsonl"),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ready"] is False
    assert payload["error"] == "network_not_allowed"
    assert not (tmp_path / "detector.jsonl").exists()


def test_vlm_detector_writes_importable_external_detector_jsonl(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_detector_script()
    main = cast(MainFn, getattr(module, "main"))
    handoff = _handoff(tmp_path)
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")
    calls: list[dict[str, Any]] = []

    def fake_sender(
        payload: dict[str, Any],
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        calls.append({"api_key": api_key, "base_url": base_url, "payload": payload})
        assert timeout_seconds == 45.0
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "detections": [
                                    {
                                        "bbox_2d_xyxy": [0, 0, 1, 1],
                                        "confidence": 0.91,
                                        "label": "apple",
                                        "target_id": "apple_target_1",
                                        "visible": True,
                                    }
                                ],
                                "error": None,
                                "frame_id": "episode-001:FloorPlan1:0001",
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    output_path = tmp_path / "detector.jsonl"
    trace_path = tmp_path / "trace.jsonl"

    exit_code = main(
        [
            "--handoff",
            str(handoff),
            "--output-detector-jsonl",
            str(output_path),
            "--trace-output",
            str(trace_path),
            "--allow-network",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ready"] is True
    assert payload["frame_count"] == 1
    assert payload["detection_count"] == 1
    assert calls[0]["api_key"] == "test-key"
    request_text = calls[0]["payload"]["messages"][1]["content"][0]["text"]
    request_payload = json.loads(request_text)
    assert request_payload["image_dimensions"] == {"height": 2, "width": 2}

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == "dsg-spatialqa-lab.external-detector-frame.v1"
    assert record["detector_name"] == "qwen37_vlm_rgbd_detector"
    detection = record["detections"][0]
    assert detection["object_id"] == "apple_target_1"
    assert detection["bbox_3d_center"] == {
        "x": 1.0,
        "y": 0.9,
        "yaw": 0.0,
        "z": 0.0,
    }
    assert detection["evidence_kinds"] == ["depth", "detector", "rgb"]
    assert detection["attributes"]["detector_method"] == "vlm_bbox_depth_projection_v1"

    observations = detector_observation_sequence_from_jsonl(
        output_path.read_text(encoding="utf-8")
    )
    assert observations[0].objects[0].attributes["source_kind"] == "detector"
    assert observations[0].objects[0].attributes["source_name"] == "qwen37_vlm_rgbd_detector"
    assert "ai2thor" not in observations[0].objects[0].attributes["source_name"]


def test_vlm_detector_prompts_for_and_preserves_visible_support_objects(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_detector_script()
    main = cast(MainFn, getattr(module, "main"))
    handoff = _handoff(tmp_path, support_labels=["countertop", "sink"])
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")
    calls: list[dict[str, Any]] = []

    def fake_sender(
        payload: dict[str, Any],
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        calls.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "detections": [
                                    {
                                        "bbox_2d_xyxy": [0, 0, 1, 1],
                                        "confidence": 0.91,
                                        "label": "apple",
                                        "target_id": "apple_target_1",
                                        "visible": True,
                                    },
                                    {
                                        "bbox_2d_xyxy": [0, 1, 1, 1],
                                        "confidence": 0.75,
                                        "label": "countertop",
                                        "target_id": None,
                                        "visible": True,
                                    },
                                ],
                                "error": None,
                                "frame_id": "episode-001:FloorPlan1:0001",
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    output_path = tmp_path / "detector.jsonl"

    exit_code = main(
        [
            "--handoff",
            str(handoff),
            "--output-detector-jsonl",
            str(output_path),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
            "--allow-network",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["detection_count"] == 2
    request_text = calls[0]["messages"][1]["content"][0]["text"]
    request_payload = json.loads(request_text)
    assert request_payload["support_labels"] == ["countertop", "sink"]
    assert request_payload["contract"]["support_detection_rule"] == (
        "Also detect visible support surfaces/containers from support_labels. "
        "For support detections set target_id to null."
    )

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    support = records[0]["detections"][1]
    assert support["label"] == "countertop"
    assert support["object_id"] == "episode-001:FloorPlan1:000001:countertop:002"
    assert support["attributes"]["support_detection"] is True
    assert support["attributes"]["targeted_detection"] is False
    observations = detector_observation_sequence_from_jsonl(
        output_path.read_text(encoding="utf-8")
    )
    assert observations[0].objects[1].label == "countertop"


def test_vlm_detector_accepts_json_object_with_trailing_text(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_detector_script()
    main = cast(MainFn, getattr(module, "main"))
    handoff = _handoff(tmp_path)
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")

    def fake_sender(
        payload: dict[str, Any],
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        assert timeout_seconds == 10.0
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"detections":[{"bbox_2d_xyxy":[0,0,1,1],'
                            '"confidence":0.8,"label":"apple",'
                            '"target_id":"apple_target_1","visible":true}],'
                            '"error":null,"frame_id":"episode-001:FloorPlan1:0001"}'
                            "\nNotes: detection complete."
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    output_path = tmp_path / "detector.jsonl"

    exit_code = main(
        [
            "--handoff",
            str(handoff),
            "--output-detector-jsonl",
            str(output_path),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
            "--allow-network",
            "--request-timeout-seconds",
            "10",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ready"] is True
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["detections"][0]["object_id"] == "apple_target_1"


def test_vlm_detector_continue_on_error_writes_empty_frame_record(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_detector_script()
    main = cast(MainFn, getattr(module, "main"))
    handoff = _handoff(tmp_path)
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")

    def fake_sender(
        payload: dict[str, Any],
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        raise OSError("provider timeout")

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    output_path = tmp_path / "detector.jsonl"
    trace_path = tmp_path / "trace.jsonl"

    exit_code = main(
        [
            "--handoff",
            str(handoff),
            "--output-detector-jsonl",
            str(output_path),
            "--trace-output",
            str(trace_path),
            "--allow-network",
            "--continue-on-error",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ready"] is False
    assert payload["error_count"] == 1
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["detections"] == []
    assert records[0]["metadata"]["detector_error"] == "provider timeout"
    traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert traces[0]["error"] == "provider timeout"


def _handoff(tmp_path: Path, *, support_labels: list[str] | None = None) -> Path:
    rgb = tmp_path / "rgb.ppm"
    depth = tmp_path / "depth.json"
    rgb.write_text(
        "P3\n2 2\n255\n255 0 0 0 255 0\n0 0 255 255 255 255\n",
        encoding="ascii",
    )
    depth.write_text("[[1.0, 1.0], [1.0, 1.0]]\n", encoding="utf-8")
    frame: dict[str, object] = {
        "camera_pose": {"x": 0.0, "y": 0.9, "z": 0.0, "yaw": 90.0},
        "case_ids": ["case-001"],
        "depth_path": str(depth),
        "episode_id": "episode-001",
        "frame_id": "episode-001:FloorPlan1:0001",
        "frame_step": 1,
        "rgb_path": str(rgb),
        "scene_id": "FloorPlan1",
        "target_labels": ["apple"],
        "target_object_ids": ["apple_target_1"],
    }
    if support_labels is not None:
        frame["support_labels"] = support_labels
    payload = {
        "schema_version": "dsg-spatialqa-lab.independent-detector-rgbd-handoff.v1",
        "required_frames": [frame],
    }
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
