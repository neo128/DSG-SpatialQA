from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch

import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
RUN_VLM_CONTROLS_SCRIPT = ROOT / "external_tools" / "run_vlm_controls.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_run_vlm_controls_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_vlm_controls_script",
        RUN_VLM_CONTROLS_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_vlm_runner_requires_explicit_network_permission(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    bundle_payload = json.loads(request_bundle.read_text(encoding="utf-8"))
    bundle_payload["case_inputs"][0]["answer_schema_hint"]["required_answer_fields"] = [
        "object_id",
        "label",
        "current_location",
        "last_seen_step",
        "state_step",
        "visible",
        "pose",
        "confidence",
    ]
    request_bundle.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--output",
            str(tmp_path / "vlm.jsonl"),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ready"] is False
    assert payload["error"] == "network_not_allowed"
    assert not (tmp_path / "vlm.jsonl").exists()


def test_vlm_runner_replays_local_trace_without_network(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    bundle_payload = json.loads(request_bundle.read_text(encoding="utf-8"))
    bundle_payload["case_inputs"][0]["answer_options"] = [
        {
            "destination_label": "countertop",
            "option_id": "locopt_001",
            "relation": "ON",
            "source": "support_candidate",
        }
    ]
    request_bundle.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trace_input = tmp_path / "old-trace.jsonl"
    trace_input.write_text(
        json.dumps(
            {
                "case_id": "case-001",
                "raw_response": {
                    "choices": [{"message": {"content": "Option 1"}}]
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    frame_index = tmp_path / "frame-index.jsonl"
    frame_index.write_text(
        json.dumps(
            {
                "episode_id": "episode-001",
                "scene_id": "scene",
                "step": 1,
                "visible_object_ids": ["apple_1", "countertop_1"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    def forbidden_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        raise AssertionError("replay mode must not call the network sender")

    monkeypatch.setattr(module, "_send_chat_completion", forbidden_sender)
    output_path = tmp_path / "replayed-vlm.jsonl"
    trace_output = tmp_path / "replayed-trace.jsonl"

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--replay-trace",
            str(trace_input),
            "--normalization-frame-index",
            str(frame_index),
            "--output",
            str(output_path),
            "--trace-output",
            str(trace_output),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    predictions = lab.load_qa_predictions(output_path)
    replay_trace = json.loads(trace_output.read_text(encoding="utf-8").splitlines()[0])
    assert exit_code == 0
    assert payload["action"] == "replay_vlm_controls"
    assert payload["ready"] is True
    assert predictions[0].answer["answer_option_id"] == "locopt_001"
    assert predictions[0].answer["current_location"]["dst"] == "countertop_1"
    assert predictions[0].error is None
    assert replay_trace["replay_source_trace"] == str(trace_input)
    assert replay_trace["structured_response"]["recovered_from"] == (
        "plain_text_answer_option"
    )


def test_vlm_runner_writes_predictions_and_trace_with_fake_sender(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")
    calls: list[dict[str, Any]] = []

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        calls.append({"api_key": api_key, "base_url": base_url, "payload": payload})
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "case_id": "case-001",
                                "answer": {
                                    "confidence": 0.82,
                                    "current_location": None,
                                    "label": "apple",
                                    "object_id": "apple_1",
                                    "visible": True,
                                },
                                "answer_text": "The apple is visible but no support id is available.",
                                "confidence": 0.82,
                                "evidence": [
                                    {
                                        "object_ids": ["apple_1"],
                                        "relation_ids": [],
                                        "source_id": "episode-001:scene:0001",
                                    }
                                ],
                                "error": "relation_not_observed",
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    output_path = tmp_path / "vlm.jsonl"
    trace_path = tmp_path / "trace.jsonl"

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--source-kind",
            "vlm",
            "--model",
            "qwen3.7-plus",
            "--base-url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "--allow-network",
            "--output",
            str(output_path),
            "--trace-output",
            str(trace_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    predictions = lab.load_qa_predictions(output_path)
    traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert exit_code == 0
    assert payload["ready"] is True
    assert payload["prediction_count"] == 1
    assert payload["trace_count"] == 1
    assert len(calls) == 1
    assert calls[0]["api_key"] == "test-key"
    assert calls[0]["payload"]["model"] == "qwen3.7-plus"
    assert "Where is the apple?" in json.dumps(calls[0]["payload"], sort_keys=True)
    assert predictions[0].id == "case-001"
    assert predictions[0].answer["label"] == "apple"
    assert predictions[0].answer["answer_text"] == (
        "The apple is visible but no support id is available."
    )
    assert predictions[0].error == "relation_not_observed"
    assert traces[0]["case_id"] == "case-001"
    assert traces[0]["image_refs"][0]["rgb_path"].endswith("0001.ppm")
    assert "api_key" not in json.dumps(traces[0], sort_keys=True)


def test_vlm_runner_uses_strict_json_object_schema(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")
    calls: list[dict[str, Any]] = []

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        calls.append(payload)
        return _fake_structured_response("case-001")

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--allow-network",
            "--output",
            str(tmp_path / "vlm.jsonl"),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
        ]
    )

    _ = capsys.readouterr()
    assert exit_code == 0
    assert calls[0]["response_format"] == {"type": "json_object"}
    system_prompt = calls[0]["messages"][0]["content"]
    assert "Return exactly one JSON object" in system_prompt
    assert "current_location" in system_prompt
    assert "target_not_observed" in system_prompt
    assert "Do not use gold answers" in system_prompt


def test_vlm_runner_prompt_hides_machine_object_ids_from_visual_query(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path, case_id="case-001:apple_1")
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")
    calls: list[dict[str, Any]] = []

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        calls.append(payload)
        user_text = payload["messages"][1]["content"][0]["text"]
        prompt_case = json.loads(user_text)
        return _fake_structured_response(prompt_case["case_id"])

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--allow-network",
            "--output",
            str(tmp_path / "vlm.jsonl"),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
        ]
    )

    _ = capsys.readouterr()
    predictions = lab.load_qa_predictions(tmp_path / "vlm.jsonl")
    user_text = calls[0]["messages"][1]["content"][0]["text"]
    assert exit_code == 0
    assert "Where is the apple?" in user_text
    assert '"label":"apple"' in user_text
    assert "case-001:apple_1" not in user_text
    assert "apple_1" not in user_text
    assert predictions[0].id == "case-001:apple_1"


def test_vlm_runner_prompt_includes_visual_location_contract_without_gold_ids(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path, case_id="case-001:apple_1")
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")
    calls: list[dict[str, Any]] = []

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        calls.append(payload)
        user_text = payload["messages"][1]["content"][0]["text"]
        prompt_case = json.loads(user_text)
        return _fake_structured_response(prompt_case["case_id"])

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--allow-network",
            "--output",
            str(tmp_path / "vlm.jsonl"),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
        ]
    )

    _ = capsys.readouterr()
    user_payload = json.loads(calls[0]["messages"][1]["content"][0]["text"])
    serialized = json.dumps(user_payload, sort_keys=True)
    assert exit_code == 0
    assert user_payload["visual_location_contract"] == {
        "allowed_location_relations": ["IN_ROOM", "INSIDE", "ON"],
        "destination_label_rule": (
            "Use a common visible support/place label such as countertop, desk, "
            "shelf, chair, coffeetable, bathtub, bed, dresser, sidetable, "
            "handtowelholder, or room. Do not invent machine object ids."
        ),
        "object_location_rule": (
            "If the target is visible, answer with visible=true and a "
            "current_location relation plus destination_label when the support or "
            "room is visually grounded; otherwise return relation_not_observed."
        ),
    }
    assert "apple_1" not in serialized
    assert "case-001:apple_1" not in serialized
    assert "gold" not in serialized


def test_multi_frame_vlm_runner_can_limit_frame_window_around_primary_frame(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    bundle_payload = json.loads(request_bundle.read_text(encoding="utf-8"))
    frames = []
    for step in range(1, 5):
        rgb_path = tmp_path / "frames" / f"{step:04d}.ppm"
        rgb_path.write_text("P3\n1 1\n255\n255 0 0\n", encoding="utf-8")
        frames.append(
            {
                "episode_id": "episode-001",
                "frame_id": f"episode-001:scene:{step:04d}",
                "rgb_digest": f"rgb-digest-{step}",
                "rgb_path": str(rgb_path),
                "scene_id": "scene",
                "step": step,
            }
        )
    bundle_payload["case_inputs"][0]["frames"] = frames
    bundle_payload["case_inputs"][0]["primary_frame"] = frames[1]
    request_bundle.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")
    calls: list[dict[str, Any]] = []

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        calls.append(payload)
        user_text = payload["messages"][1]["content"][0]["text"]
        prompt_case = json.loads(user_text)
        return _fake_structured_response(prompt_case["case_id"])

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--source-kind",
            "multi_frame_vlm",
            "--max-frames",
            "2",
            "--allow-network",
            "--output",
            str(tmp_path / "multi-frame-vlm.jsonl"),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
        ]
    )

    _ = capsys.readouterr()
    image_urls = [
        item for item in calls[0]["messages"][1]["content"] if item["type"] == "image_url"
    ]
    traces = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert exit_code == 0
    assert len(image_urls) == 2
    assert [frame["step"] for frame in traces[0]["image_refs"]] == [2, 4]


def test_vlm_runner_prompt_uses_target_crop_and_support_candidates_without_ids(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path, case_id="case-001:apple_1")
    crop_path = tmp_path / "frames" / "0001-target-crop.ppm"
    crop_path.write_text("P3\n1 1\n255\n0 255 0\n", encoding="utf-8")
    bundle_payload = json.loads(request_bundle.read_text(encoding="utf-8"))
    bundle_payload["case_inputs"][0]["target_crop"] = {
        "bbox_2d_xyxy": [10, 20, 60, 90],
        "object_id": "apple_1",
        "rgb_path": str(crop_path),
        "source_frame_id": "episode-001:scene:0001",
    }
    bundle_payload["case_inputs"][0]["support_candidates"] = [
        {
            "confidence": 0.88,
            "label": "countertop",
            "object_id": "countertop_1",
            "relation_hint": "ON",
        },
        {
            "confidence": 0.52,
            "label": "sink",
            "object_id": "sink_1",
            "relation_hint": "INSIDE",
        },
    ]
    bundle_payload["case_inputs"][0]["question_task_hint"] = (
        "Choose the visible support or container relation from answer_options."
    )
    request_bundle.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")
    calls: list[dict[str, Any]] = []

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        calls.append(payload)
        user_text = payload["messages"][1]["content"][0]["text"]
        prompt_case = json.loads(user_text)
        return _fake_structured_response(prompt_case["case_id"])

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--allow-network",
            "--output",
            str(tmp_path / "vlm.jsonl"),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
        ]
    )

    _ = capsys.readouterr()
    user_payload = json.loads(calls[0]["messages"][1]["content"][0]["text"])
    image_urls = [
        item for item in calls[0]["messages"][1]["content"] if item["type"] == "image_url"
    ]
    serialized = json.dumps(user_payload, sort_keys=True)
    trace = json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[0])
    assert exit_code == 0
    assert len(image_urls) == 2
    assert user_payload["target_crop"] == {
        "available": True,
        "bbox_2d_xyxy": [10, 20, 60, 90],
        "source_frame_id": "episode-001:scene:0001",
    }
    assert user_payload["support_candidates"] == [
        {"confidence": 0.88, "label": "countertop", "relation_hint": "ON"},
        {"confidence": 0.52, "label": "sink", "relation_hint": "INSIDE"},
    ]
    assert user_payload["question_task_hint"] == (
        "Choose the visible support or container relation from answer_options."
    )
    assert trace["image_refs"][1]["crop_role"] == "target_crop"
    assert "apple_1" not in serialized
    assert "countertop_1" not in serialized
    assert "sink_1" not in serialized
    assert "gold" not in serialized


def test_vlm_runner_prompt_uses_no_crop_visual_context_without_extra_image(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path, case_id="case-001:apple_1")
    bundle_payload = json.loads(request_bundle.read_text(encoding="utf-8"))
    bundle_payload["case_inputs"][0]["target_visual_context"] = {
        "available": True,
        "context_kind": "primary_frame_without_target_crop",
        "instruction": (
            "No local target crop is available. Inspect only the primary RGB "
            "frame; if the target is not visually clear, return "
            "target_not_observed instead of guessing."
        ),
        "target_crop_available": False,
        "target_crop_unavailable_reason": "missing_segmentation_color",
        "target_label": "apple",
    }
    request_bundle.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")
    calls: list[dict[str, Any]] = []

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        calls.append(payload)
        user_text = payload["messages"][1]["content"][0]["text"]
        prompt_case = json.loads(user_text)
        return _fake_structured_response(prompt_case["case_id"])

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--allow-network",
            "--output",
            str(tmp_path / "vlm.jsonl"),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
        ]
    )

    _ = capsys.readouterr()
    user_payload = json.loads(calls[0]["messages"][1]["content"][0]["text"])
    image_urls = [
        item for item in calls[0]["messages"][1]["content"] if item["type"] == "image_url"
    ]
    serialized = json.dumps(user_payload, sort_keys=True)
    trace = json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[0])
    assert exit_code == 0
    assert len(image_urls) == 1
    assert trace["request"]["image_roles"] == ["primary_frame"]
    assert user_payload["target_crop"] == {"available": False}
    assert user_payload["target_visual_context"] == {
        "available": True,
        "context_kind": "primary_frame_without_target_crop",
        "instruction": (
            "No local target crop is available. Inspect only the primary RGB "
            "frame; if the target is not visually clear, return "
            "target_not_observed instead of guessing."
        ),
        "target_crop_available": False,
        "target_crop_unavailable_reason": "missing_segmentation_color",
        "target_label": "apple",
    }
    assert "apple_1" not in serialized
    assert "visible_object" not in serialized
    assert "gold" not in serialized


def test_vlm_runner_labels_images_and_traces_visual_prompt_payload(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path, case_id="case-001:apple_1")
    crop_path = tmp_path / "frames" / "0001-target-crop.ppm"
    crop_path.write_text("P3\n1 1\n255\n0 255 0\n", encoding="utf-8")
    bundle_payload = json.loads(request_bundle.read_text(encoding="utf-8"))
    bundle_payload["case_inputs"][0]["target_crop"] = {
        "bbox_2d_xyxy": [10, 20, 60, 90],
        "object_id": "apple_1",
        "rgb_path": str(crop_path),
        "source_frame_id": "episode-001:scene:0001",
    }
    bundle_payload["case_inputs"][0]["answer_options"] = [
        {
            "destination_label": "countertop",
            "object_id": "countertop_1",
            "option_id": "locopt_001",
            "relation": "ON",
            "source": "support_candidate",
        }
    ]
    request_bundle.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")
    calls: list[dict[str, Any]] = []

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        calls.append(payload)
        user_text = payload["messages"][1]["content"][0]["text"]
        prompt_case = json.loads(user_text)
        return _fake_structured_response(prompt_case["case_id"])

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    trace_path = tmp_path / "trace.jsonl"

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--allow-network",
            "--output",
            str(tmp_path / "vlm.jsonl"),
            "--trace-output",
            str(trace_path),
        ]
    )

    _ = capsys.readouterr()
    content = calls[0]["messages"][1]["content"]
    text_items = [item["text"] for item in content if item["type"] == "text"]
    trace = json.loads(trace_path.read_text().splitlines()[0])
    visual_prompt_payload = trace["request"]["visual_prompt_payload"]
    visual_prompt_serialized = json.dumps(visual_prompt_payload, sort_keys=True)
    assert exit_code == 0
    assert any("Image 1: primary RGB frame" in item for item in text_items)
    assert any("Image 2: target crop" in item for item in text_items)
    assert visual_prompt_payload["target"] == {"label": "apple"}
    assert visual_prompt_payload["answer_options"] == [
        {
            "destination_label": "countertop",
            "option_id": "locopt_001",
            "relation": "ON",
        }
    ]
    assert trace["request"]["image_roles"] == ["primary_frame", "target_crop"]
    assert "apple_1" not in visual_prompt_serialized
    assert "countertop_1" not in visual_prompt_serialized
    assert "gold" not in visual_prompt_serialized


def test_vlm_runner_prompt_includes_answer_options_without_machine_ids(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path, case_id="case-001:apple_1")
    bundle_payload = json.loads(request_bundle.read_text(encoding="utf-8"))
    bundle_payload["case_inputs"][0]["answer_options"] = [
        {
            "destination_label": "countertop",
            "object_id": "countertop_1",
            "option_id": "locopt_001",
            "relation": "ON",
            "source": "support_candidate",
        },
        {
            "destination_label": "room",
            "option_id": "locopt_002",
            "relation": "IN_ROOM",
            "source": "fallback_room",
        },
    ]
    bundle_payload["case_inputs"][0]["answer_option_response_schema"] = {
        "answer_current_location_rule": (
            "Copy relation and destination_label from the selected answer option."
        ),
        "answer_option_id_field": "answer.answer_option_id",
        "allowed_answer_option_ids": ["locopt_001", "locopt_002"],
        "required_when_answer_options_present": True,
    }
    request_bundle.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")
    calls: list[dict[str, Any]] = []

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        calls.append(payload)
        user_text = payload["messages"][1]["content"][0]["text"]
        prompt_case = json.loads(user_text)
        return _fake_structured_response(prompt_case["case_id"])

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--allow-network",
            "--output",
            str(tmp_path / "vlm.jsonl"),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
        ]
    )

    _ = capsys.readouterr()
    user_payload = json.loads(calls[0]["messages"][1]["content"][0]["text"])
    serialized = json.dumps(user_payload, sort_keys=True)
    assert exit_code == 0
    assert user_payload["answer_options"] == [
        {
            "destination_label": "countertop",
            "option_id": "locopt_001",
            "relation": "ON",
        },
        {
            "destination_label": "room",
            "option_id": "locopt_002",
            "relation": "IN_ROOM",
        },
    ]
    assert user_payload["answer_option_response_schema"] == {
        "answer_current_location_rule": (
            "Copy relation and destination_label from the selected answer option."
        ),
        "answer_option_id_field": "answer.answer_option_id",
        "allowed_answer_option_ids": ["locopt_001", "locopt_002"],
        "required_when_answer_options_present": True,
    }
    assert "Choose answer.current_location from answer_options" in (
        calls[0]["messages"][0]["content"]
    )
    assert "apple_1" not in serialized
    assert "countertop_1" not in serialized
    assert "gold" not in serialized


def test_vlm_runner_prompt_includes_answer_option_choice_strategy(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path, case_id="case-001:apple_1")
    bundle_payload = json.loads(request_bundle.read_text(encoding="utf-8"))
    bundle_payload["case_inputs"][0]["answer_options"] = [
        {
            "destination_label": "countertop",
            "object_id": "countertop_1",
            "option_id": "locopt_001",
            "relation": "ON",
            "source": "support_candidate",
        },
        {
            "destination_label": "room",
            "option_id": "locopt_002",
            "relation": "IN_ROOM",
            "source": "fallback_room",
        },
    ]
    request_bundle.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")
    calls: list[dict[str, Any]] = []

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        calls.append(payload)
        user_text = payload["messages"][1]["content"][0]["text"]
        prompt_case = json.loads(user_text)
        return _fake_structured_response(prompt_case["case_id"])

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--allow-network",
            "--output",
            str(tmp_path / "vlm.jsonl"),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
        ]
    )

    _ = capsys.readouterr()
    user_payload = json.loads(calls[0]["messages"][1]["content"][0]["text"])
    strategy = user_payload["visual_answer_option_strategy"]
    serialized = json.dumps(strategy, sort_keys=True)
    assert exit_code == 0
    assert strategy == {
        "fallback_rule": (
            "Use a room option only when no more specific visible support/place "
            "option fits the target."
        ),
        "if_target_not_visible": (
            "Return error=target_not_observed and do not choose an answer option."
        ),
        "primary_rule": (
            "If answer_options is non-empty and the target is visible, choose "
            "exactly one allowed answer_option_id."
        ),
        "support_rule": (
            "Use the target crop to identify the target, then choose the option "
            "whose relation and destination_label best match the primary RGB frame."
        ),
    }
    assert "countertop_1" not in serialized
    assert "apple_1" not in serialized
    assert "gold" not in serialized
    assert "choose exactly one allowed answer_option_id" in (
        calls[0]["messages"][0]["content"]
    )


def test_vlm_runner_prompt_includes_visual_decision_checklist_without_gold_or_ids(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path, case_id="case-001:apple_1")
    bundle_payload = json.loads(request_bundle.read_text(encoding="utf-8"))
    bundle_payload["case_inputs"][0]["answer_options"] = [
        {
            "destination_label": "countertop",
            "object_id": "countertop_1",
            "option_id": "locopt_001",
            "relation": "ON",
            "source": "support_candidate",
        },
        {
            "destination_label": "room",
            "option_id": "locopt_002",
            "relation": "IN_ROOM",
            "source": "fallback_room",
        },
    ]
    request_bundle.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")
    calls: list[dict[str, Any]] = []

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        calls.append(payload)
        user_text = payload["messages"][1]["content"][0]["text"]
        prompt_case = json.loads(user_text)
        return _fake_structured_response(prompt_case["case_id"])

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--allow-network",
            "--output",
            str(tmp_path / "vlm.jsonl"),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
        ]
    )

    _ = capsys.readouterr()
    user_payload = json.loads(calls[0]["messages"][1]["content"][0]["text"])
    checklist = user_payload["visual_decision_checklist"]
    serialized = json.dumps(checklist, sort_keys=True)
    assert exit_code == 0
    assert checklist == [
        {
            "name": "target_visibility",
            "instruction": (
                "First decide whether the target label is visible in the primary "
                "frame or target crop. If target_crop is available, use it to "
                "confirm small or partially visible targets before returning "
                "target_not_observed. If not visible, return error=target_not_observed."
            ),
        },
        {
            "name": "support_selection",
            "instruction": (
                "If the target is visible and answer_options exist, choose the "
                "single option whose relation and destination_label are supported "
                "by the primary frame."
            ),
        },
        {
            "name": "evidence_trace",
            "instruction": (
                "Explain which provided image role supports the decision in "
                "observability or reasoning_summary."
            ),
        },
    ]
    assert "Follow visual_decision_checklist before answering" in (
        calls[0]["messages"][0]["content"]
    )
    assert "apple_1" not in serialized
    assert "countertop_1" not in serialized
    assert "gold" not in serialized


def test_vlm_runner_normalizes_answer_option_id_to_location(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    bundle_payload = json.loads(request_bundle.read_text(encoding="utf-8"))
    bundle_payload["case_inputs"][0]["answer_options"] = [
        {
            "destination_label": "countertop",
            "option_id": "locopt_001",
            "relation": "ON",
            "source": "support_candidate",
        },
        {
            "destination_label": "room",
            "option_id": "locopt_002",
            "relation": "IN_ROOM",
            "source": "fallback_room",
        },
    ]
    request_bundle.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frame_index = tmp_path / "frame-index.jsonl"
    frame_index.write_text(
        json.dumps(
            {
                "episode_id": "episode-001",
                "scene_id": "scene",
                "step": 1,
                "visible_object_ids": ["apple_1", "countertop_1"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "case_id": "case-001",
                                "answer": {
                                    "answer_option_id": "locopt_001",
                                    "confidence": 0.83,
                                    "current_location": None,
                                    "label": "apple",
                                    "object_id": None,
                                    "visible": True,
                                },
                                "answer_text": "I choose locopt_001.",
                                "confidence": 0.83,
                                "evidence": [],
                                "error": None,
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    output_path = tmp_path / "vlm.jsonl"
    trace_path = tmp_path / "trace.jsonl"

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--normalization-frame-index",
            str(frame_index),
            "--allow-network",
            "--output",
            str(output_path),
            "--trace-output",
            str(trace_path),
        ]
    )

    _ = capsys.readouterr()
    predictions = lab.load_qa_predictions(output_path)
    trace = json.loads(trace_path.read_text().splitlines()[0])
    assert exit_code == 0
    assert predictions[0].answer["answer_option_id"] == "locopt_001"
    assert predictions[0].answer["current_location"] == {
        "dst": "countertop_1",
        "dst_label": "countertop",
        "option_id": "locopt_001",
        "relation": "ON",
        "step": 1,
    }
    assert predictions[0].error is None
    normalization_changes = trace["diagnostics"]["normalization"]["changes"]
    assert "applied_answer_option_id" in normalization_changes
    assert "filled_destination_object_id" in normalization_changes
    assert "added_normalization_evidence" in normalization_changes


def test_vlm_runner_extracts_answer_option_id_from_answer_text(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    bundle_payload = json.loads(request_bundle.read_text(encoding="utf-8"))
    bundle_payload["case_inputs"][0]["answer_options"] = [
        {
            "destination_label": "countertop",
            "option_id": "locopt_001",
            "relation": "ON",
            "source": "support_candidate",
        }
    ]
    request_bundle.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frame_index = tmp_path / "frame-index.jsonl"
    frame_index.write_text(
        json.dumps(
            {
                "episode_id": "episode-001",
                "scene_id": "scene",
                "step": 1,
                "visible_object_ids": ["apple_1", "countertop_1"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "case_id": "case-001",
                                "answer": {
                                    "confidence": 0.79,
                                    "current_location": None,
                                    "label": "apple",
                                    "object_id": None,
                                    "visible": True,
                                },
                                "answer_text": "The best answer is locopt_001.",
                                "confidence": 0.79,
                                "evidence": [],
                                "error": None,
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    output_path = tmp_path / "vlm.jsonl"
    trace_path = tmp_path / "trace.jsonl"

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--normalization-frame-index",
            str(frame_index),
            "--allow-network",
            "--output",
            str(output_path),
            "--trace-output",
            str(trace_path),
        ]
    )

    _ = capsys.readouterr()
    predictions = lab.load_qa_predictions(output_path)
    trace = json.loads(trace_path.read_text().splitlines()[0])
    assert exit_code == 0
    assert predictions[0].answer["answer_option_id"] == "locopt_001"
    assert predictions[0].answer["current_location"] == {
        "dst": "countertop_1",
        "dst_label": "countertop",
        "option_id": "locopt_001",
        "relation": "ON",
        "step": 1,
    }
    assert predictions[0].error is None
    assert "extracted_answer_option_id_from_text" in (
        trace["diagnostics"]["normalization"]["changes"]
    )


def test_vlm_runner_applies_answer_option_without_frame_index(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    bundle_payload = json.loads(request_bundle.read_text(encoding="utf-8"))
    bundle_payload["case_inputs"][0]["answer_options"] = [
        {
            "destination_label": "countertop",
            "option_id": "locopt_001",
            "relation": "ON",
            "source": "support_candidate",
        }
    ]
    request_bundle.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "case_id": "case-001",
                                "answer": {
                                    "answer_option_id": "locopt_001",
                                    "confidence": 0.81,
                                    "current_location": None,
                                    "label": "apple",
                                    "object_id": None,
                                    "visible": True,
                                },
                                "answer_text": "I choose locopt_001.",
                                "confidence": 0.81,
                                "evidence": [],
                                "error": None,
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    output_path = tmp_path / "vlm.jsonl"
    trace_path = tmp_path / "trace.jsonl"

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--allow-network",
            "--output",
            str(output_path),
            "--trace-output",
            str(trace_path),
        ]
    )

    _ = capsys.readouterr()
    predictions = lab.load_qa_predictions(output_path)
    trace = json.loads(trace_path.read_text().splitlines()[0])
    assert exit_code == 0
    assert predictions[0].answer["answer_option_id"] == "locopt_001"
    assert predictions[0].answer["object_id"] == "apple_1"
    assert predictions[0].answer["current_location"] == {
        "dst_label": "countertop",
        "option_id": "locopt_001",
        "relation": "ON",
        "step": 1,
    }
    assert predictions[0].error is None
    normalization_changes = trace["diagnostics"]["normalization"]["changes"]
    assert "applied_answer_option_id" in normalization_changes
    assert "filled_target_object_id" in normalization_changes


def test_vlm_runner_recovers_plain_text_answer_option_response(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    bundle_payload = json.loads(request_bundle.read_text(encoding="utf-8"))
    bundle_payload["case_inputs"][0]["answer_options"] = [
        {
            "destination_label": "countertop",
            "option_id": "locopt_001",
            "relation": "ON",
            "source": "support_candidate",
        }
    ]
    request_bundle.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frame_index = tmp_path / "frame-index.jsonl"
    frame_index.write_text(
        json.dumps(
            {
                "episode_id": "episode-001",
                "scene_id": "scene",
                "step": 1,
                "visible_object_ids": ["apple_1", "countertop_1"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "locopt_001"}}]}

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    output_path = tmp_path / "vlm.jsonl"
    trace_path = tmp_path / "trace.jsonl"

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--normalization-frame-index",
            str(frame_index),
            "--allow-network",
            "--output",
            str(output_path),
            "--trace-output",
            str(trace_path),
        ]
    )

    _ = capsys.readouterr()
    predictions = lab.load_qa_predictions(output_path)
    trace = json.loads(trace_path.read_text().splitlines()[0])
    assert exit_code == 0
    assert predictions[0].answer["answer_option_id"] == "locopt_001"
    assert predictions[0].answer["current_location"] == {
        "dst": "countertop_1",
        "dst_label": "countertop",
        "option_id": "locopt_001",
        "relation": "ON",
        "step": 1,
    }
    assert predictions[0].error is None
    assert trace["structured_response"]["recovered_from"] == "plain_text_answer_option"


def test_vlm_runner_recovers_plain_text_answer_option_ordinal_response(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    bundle_payload = json.loads(request_bundle.read_text(encoding="utf-8"))
    bundle_payload["case_inputs"][0]["answer_options"] = [
        {
            "destination_label": "countertop",
            "option_id": "locopt_001",
            "relation": "ON",
            "source": "support_candidate",
        },
        {
            "destination_label": "room",
            "option_id": "locopt_002",
            "relation": "IN_ROOM",
            "source": "fallback_room",
        },
    ]
    request_bundle.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frame_index = tmp_path / "frame-index.jsonl"
    frame_index.write_text(
        json.dumps(
            {
                "episode_id": "episode-001",
                "scene_id": "scene",
                "step": 1,
                "visible_object_ids": ["apple_1", "countertop_1"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "Option 1"}}]}

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    output_path = tmp_path / "vlm.jsonl"
    trace_path = tmp_path / "trace.jsonl"

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--normalization-frame-index",
            str(frame_index),
            "--allow-network",
            "--output",
            str(output_path),
            "--trace-output",
            str(trace_path),
        ]
    )

    _ = capsys.readouterr()
    predictions = lab.load_qa_predictions(output_path)
    trace = json.loads(trace_path.read_text().splitlines()[0])
    assert exit_code == 0
    assert predictions[0].answer["answer_option_id"] == "locopt_001"
    assert predictions[0].answer["current_location"] == {
        "dst": "countertop_1",
        "dst_label": "countertop",
        "option_id": "locopt_001",
        "relation": "ON",
        "step": 1,
    }
    assert predictions[0].error is None
    assert trace["structured_response"]["recovered_from"] == "plain_text_answer_option"


def test_vlm_runner_marks_free_text_answer_schema_mismatch(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "case_id": "case-001",
                                "answer": {"text": "on the countertop"},
                                "answer_text": "on the countertop",
                                "confidence": 0.7,
                                "evidence": [],
                                "error": None,
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    output_path = tmp_path / "vlm.jsonl"
    trace_path = tmp_path / "trace.jsonl"

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--allow-network",
            "--output",
            str(output_path),
            "--trace-output",
            str(trace_path),
        ]
    )

    _ = capsys.readouterr()
    predictions = lab.load_qa_predictions(output_path)
    trace = json.loads(trace_path.read_text().splitlines()[0])
    assert exit_code == 0
    assert predictions[0].error == "answer_schema_mismatch"
    assert trace["diagnostics"]["parse_valid"] is True
    assert trace["diagnostics"]["answer_schema_valid"] is False
    assert trace["diagnostics"]["evidence_present"] is False
    assert trace["diagnostics"]["failure_reasons"] == [
        "answer_schema_mismatch",
        "evidence_absent",
    ]


def test_vlm_runner_normalizes_visible_location_with_frame_index(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    frame_index = tmp_path / "frame-index.jsonl"
    frame_index.write_text(
        json.dumps(
            {
                "episode_id": "episode-001",
                "scene_id": "scene",
                "step": 1,
                "visible_object_ids": ["apple_1", "countertop_1"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "case_id": "case-001",
                                "answer": {
                                    "confidence": 0.74,
                                    "current_location": {
                                        "dst_label": "countertop",
                                        "relation": "ON",
                                    },
                                    "label": "apple",
                                    "object_id": None,
                                    "visible": True,
                                },
                                "answer_text": "on the countertop",
                                "confidence": 0.74,
                                "evidence": [],
                                "error": None,
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    output_path = tmp_path / "vlm.jsonl"
    trace_path = tmp_path / "trace.jsonl"

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--normalization-frame-index",
            str(frame_index),
            "--allow-network",
            "--output",
            str(output_path),
            "--trace-output",
            str(trace_path),
        ]
    )

    _ = capsys.readouterr()
    predictions = lab.load_qa_predictions(output_path)
    trace = json.loads(trace_path.read_text().splitlines()[0])
    assert exit_code == 0
    assert predictions[0].answer["object_id"] == "apple_1"
    assert predictions[0].answer["current_location"] == {
        "dst": "countertop_1",
        "dst_label": "countertop",
        "relation": "ON",
        "step": 1,
    }
    assert predictions[0].answer["last_seen_step"] == 1
    assert predictions[0].answer["state_step"] == 1
    assert predictions[0].answer["pose"] is None
    assert predictions[0].error is None
    assert predictions[0].evidence_nodes == ("apple_1", "countertop_1")
    assert trace["diagnostics"]["answer_schema_valid"] is True
    assert trace["diagnostics"]["normalization"]["applied"] is True
    assert trace["normalized_structured_response"]["answer"]["object_id"] == "apple_1"


def test_vlm_runner_uses_case_step_for_normalized_location_step(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    bundle_payload = json.loads(request_bundle.read_text(encoding="utf-8"))
    bundle_payload["case_inputs"][0]["step"] = 100034
    bundle_payload["case_inputs"][0]["primary_frame"]["step"] = 34
    request_bundle.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frame_index = tmp_path / "frame-index.jsonl"
    frame_index.write_text(
        json.dumps(
            {
                "episode_id": "episode-001",
                "scene_id": "scene",
                "step": 34,
                "visible_object_ids": ["apple_1", "countertop_1"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "case_id": "case-001",
                                "answer": {
                                    "confidence": 0.74,
                                    "current_location": {
                                        "dst_label": "countertop",
                                        "relation": "ON",
                                    },
                                    "label": "apple",
                                    "object_id": None,
                                    "visible": True,
                                },
                                "answer_text": "on the countertop",
                                "confidence": 0.74,
                                "evidence": [],
                                "error": None,
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    output_path = tmp_path / "vlm.jsonl"

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--normalization-frame-index",
            str(frame_index),
            "--allow-network",
            "--output",
            str(output_path),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
        ]
    )

    _ = capsys.readouterr()
    predictions = lab.load_qa_predictions(output_path)
    assert exit_code == 0
    assert predictions[0].answer["current_location"]["step"] == 100034
    assert predictions[0].answer["last_seen_step"] == 100034
    assert predictions[0].answer["state_step"] == 100034


def test_vlm_runner_normalizes_generic_destination_id_and_zero_step(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    bundle_payload = json.loads(request_bundle.read_text(encoding="utf-8"))
    bundle_payload["case_inputs"][0]["step"] = 100034
    request_bundle.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frame_index = tmp_path / "frame-index.jsonl"
    frame_index.write_text(
        json.dumps(
            {
                "episode_id": "episode-001",
                "scene_id": "scene",
                "step": 1,
                "visible_object_ids": ["apple_1", "countertop_1"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "case_id": "case-001",
                                "answer": {
                                    "confidence": 0.9,
                                    "current_location": {
                                        "dst": "countertop",
                                        "relation": "ON",
                                        "step": 0,
                                    },
                                    "label": "apple",
                                    "object_id": None,
                                    "visible": True,
                                },
                                "answer_text": "The apple is on the countertop.",
                                "confidence": 0.9,
                                "evidence": [],
                                "error": None,
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--normalization-frame-index",
            str(frame_index),
            "--allow-network",
            "--output",
            str(tmp_path / "vlm.jsonl"),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
        ]
    )

    _ = capsys.readouterr()
    predictions = lab.load_qa_predictions(tmp_path / "vlm.jsonl")
    assert exit_code == 0
    assert predictions[0].answer["current_location"] == {
        "dst": "countertop_1",
        "dst_label": "countertop",
        "relation": "ON",
        "step": 100034,
    }
    assert predictions[0].evidence_nodes == ("apple_1", "countertop_1")
    assert predictions[0].error is None


def test_vlm_runner_requires_relation_after_location_normalization(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    frame_index = tmp_path / "frame-index.jsonl"
    frame_index.write_text(
        json.dumps(
            {
                "episode_id": "episode-001",
                "scene_id": "scene",
                "step": 1,
                "visible_object_ids": ["apple_1", "countertop_1"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "case_id": "case-001",
                                "answer": {
                                    "confidence": 0.74,
                                    "current_location": {"dst_label": "countertop"},
                                    "label": "apple",
                                    "object_id": None,
                                    "visible": True,
                                },
                                "answer_text": "under the countertop",
                                "confidence": 0.74,
                                "evidence": [],
                                "error": None,
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    output_path = tmp_path / "vlm.jsonl"
    trace_path = tmp_path / "trace.jsonl"

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--normalization-frame-index",
            str(frame_index),
            "--allow-network",
            "--output",
            str(output_path),
            "--trace-output",
            str(trace_path),
        ]
    )

    _ = capsys.readouterr()
    predictions = lab.load_qa_predictions(output_path)
    assert exit_code == 0
    assert predictions[0].answer["current_location"]["dst"] == "countertop_1"
    assert predictions[0].error == "relation_not_observed"


def test_vlm_runner_normalizes_atop_bathtub_basin_text(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    bundle_payload = json.loads(request_bundle.read_text(encoding="utf-8"))
    bundle_payload["case_inputs"][0]["target"] = {
        "label": "cloth",
        "object_id": "cloth_1",
    }
    bundle_payload["case_inputs"][0]["question"] = {
        "object_id": "cloth_1",
        "type": "object_location",
    }
    bundle_payload["case_inputs"][0]["question_text"] = "Where is the cloth?"
    request_bundle.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frame_index = tmp_path / "frame-index.jsonl"
    frame_index.write_text(
        json.dumps(
            {
                "episode_id": "episode-001",
                "scene_id": "scene",
                "step": 1,
                "visible_object_ids": ["cloth_1", "bathtub_1"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "case_id": "case-001",
                                "answer": {
                                    "confidence": 0.8,
                                    "current_location": {},
                                    "label": "cloth",
                                    "object_id": None,
                                    "visible": True,
                                },
                                "answer_text": "The cloth is sitting atop the bathtub basin.",
                                "confidence": 0.8,
                                "evidence": [],
                                "error": None,
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    output_path = tmp_path / "vlm.jsonl"

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--normalization-frame-index",
            str(frame_index),
            "--allow-network",
            "--output",
            str(output_path),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
        ]
    )

    _ = capsys.readouterr()
    predictions = lab.load_qa_predictions(output_path)
    assert exit_code == 0
    assert predictions[0].answer["object_id"] == "cloth_1"
    assert predictions[0].answer["current_location"] == {
        "dst": "bathtub_1",
        "dst_label": "bathtub",
        "relation": "ON",
        "step": 1,
    }
    assert predictions[0].error is None


def test_vlm_runner_normalizes_visible_in_scene_as_room_location(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "case_id": "case-001",
                                "answer": {
                                    "confidence": 0.71,
                                    "current_location": {},
                                    "label": "apple",
                                    "object_id": None,
                                    "visible": True,
                                },
                                "answer_text": "The apple is visible in the scene.",
                                "confidence": 0.71,
                                "evidence": [],
                                "error": None,
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    output_path = tmp_path / "vlm.jsonl"
    frame_index = tmp_path / "frame-index.jsonl"
    frame_index.write_text(
        json.dumps(
            {
                "episode_id": "episode-001",
                "scene_id": "scene",
                "step": 1,
                "visible_object_ids": ["apple_1"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--normalization-frame-index",
            str(frame_index),
            "--allow-network",
            "--output",
            str(output_path),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
        ]
    )

    _ = capsys.readouterr()
    predictions = lab.load_qa_predictions(output_path)
    assert exit_code == 0
    assert predictions[0].answer["object_id"] == "apple_1"
    assert predictions[0].answer["current_location"] == {
        "dst": "ai2thor_room",
        "dst_label": "room",
        "relation": "IN_ROOM",
        "step": 1,
    }
    assert predictions[0].error is None


def test_vlm_runner_marks_null_location_without_error_as_observability_error(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "case_id": "case-001",
                                "answer": {
                                    "current_location": None,
                                    "label": "apple",
                                    "object_id": "apple_1",
                                    "visible": False,
                                },
                                "answer_text": "unknown",
                                "confidence": 0.1,
                                "evidence": [],
                                "error": None,
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    output_path = tmp_path / "vlm.jsonl"
    trace_path = tmp_path / "trace.jsonl"

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--allow-network",
            "--output",
            str(output_path),
            "--trace-output",
            str(trace_path),
        ]
    )

    _ = capsys.readouterr()
    predictions = lab.load_qa_predictions(output_path)
    trace = json.loads(trace_path.read_text().splitlines()[0])
    assert exit_code == 0
    assert predictions[0].error == "target_not_observed"
    assert trace["diagnostics"]["answer_schema_valid"] is True
    assert trace["diagnostics"]["failure_reasons"] == [
        "evidence_absent",
        "target_not_observed",
    ]


def test_vlm_runner_writes_trace_when_sender_fails(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path)
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")

    def fake_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        raise lab.SpatialQAError("HTTP Error 401: Unauthorized")

    monkeypatch.setattr(module, "_send_chat_completion", fake_sender)
    output_path = tmp_path / "vlm.jsonl"
    trace_path = tmp_path / "trace.jsonl"

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--allow-network",
            "--output",
            str(output_path),
            "--trace-output",
            str(trace_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    trace = json.loads(trace_path.read_text().splitlines()[0])
    assert exit_code == 1
    assert payload["ready"] is False
    assert payload["error"] == "HTTP Error 401: Unauthorized"
    assert payload["trace_output"] == str(trace_path)
    assert not output_path.exists()
    assert trace["case_id"] == "case-001"
    assert trace["diagnostics"]["failure_reasons"] == ["external_call_failed"]
    assert trace["diagnostics"]["response_error"] == "HTTP Error 401: Unauthorized"
    assert trace["image_refs"][0]["rgb_path"].endswith("0001.ppm")
    assert "api_key" not in json.dumps(trace, sort_keys=True)


def test_vlm_runner_checkpoints_predictions_and_resumes_remaining_cases(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path, case_id="case-001")
    bundle_payload = json.loads(request_bundle.read_text(encoding="utf-8"))
    second_case = dict(bundle_payload["case_inputs"][0])
    second_case["case_id"] = "case-002"
    second_case["question_text"] = "Where is the mug?"
    second_case["question"] = {"object_id": "mug_1", "type": "object_location"}
    second_case["target"] = {"label": "mug", "object_id": "mug_1"}
    bundle_payload["case_count"] = 2
    bundle_payload["case_inputs"].append(second_case)
    request_bundle.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")
    output_path = tmp_path / "vlm.jsonl"
    trace_path = tmp_path / "trace.jsonl"
    calls: list[str] = []

    def first_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        prompt_case = json.loads(payload["messages"][1]["content"][0]["text"])
        calls.append(prompt_case["question_text"])
        if len(calls) == 2:
            raise lab.SpatialQAError("HTTP Error 503: Service Unavailable")
        return _fake_structured_response(prompt_case["case_id"])

    monkeypatch.setattr(module, "_send_chat_completion", first_sender)

    first_exit = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--allow-network",
            "--output",
            str(output_path),
            "--trace-output",
            str(trace_path),
        ]
    )

    first_payload = json.loads(capsys.readouterr().out)
    first_predictions = lab.load_qa_predictions(output_path)
    first_traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert first_exit == 1
    assert first_payload["ready"] is False
    assert first_payload["checkpoint_prediction_count"] == 1
    assert [prediction.id for prediction in first_predictions] == ["case-001"]
    assert [trace["case_id"] for trace in first_traces] == ["case-001", "case-002"]

    calls.clear()

    def resume_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        prompt_case = json.loads(payload["messages"][1]["content"][0]["text"])
        calls.append(prompt_case["question_text"])
        return _fake_structured_response(prompt_case["case_id"])

    monkeypatch.setattr(module, "_send_chat_completion", resume_sender)

    second_exit = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--allow-network",
            "--resume",
            "--output",
            str(output_path),
            "--trace-output",
            str(trace_path),
        ]
    )

    second_payload = json.loads(capsys.readouterr().out)
    predictions = lab.load_qa_predictions(output_path)
    traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert second_exit == 0
    assert second_payload["ready"] is True
    assert second_payload["prediction_count"] == 2
    assert second_payload["resumed_prediction_count"] == 1
    assert calls == ["Where is the mug?"]
    assert [prediction.id for prediction in predictions] == ["case-001", "case-002"]
    assert [trace["case_id"] for trace in traces] == [
        "case-001",
        "case-002",
        "case-002",
    ]


def test_vlm_runner_retries_empty_message_content(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_vlm_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle = _request_bundle(tmp_path, case_id="case-001")
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")
    output_path = tmp_path / "vlm.jsonl"
    trace_path = tmp_path / "trace.jsonl"
    calls: list[str] = []

    def retry_sender(payload: dict[str, Any], *, api_key: str, base_url: str) -> dict[str, Any]:
        prompt_case = json.loads(payload["messages"][1]["content"][0]["text"])
        calls.append(prompt_case["case_id"])
        if len(calls) == 1:
            return {"choices": [{"message": {"content": ""}}]}
        return _fake_structured_response(prompt_case["case_id"])

    monkeypatch.setattr(module, "_send_chat_completion", retry_sender)

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--allow-network",
            "--output",
            str(output_path),
            "--trace-output",
            str(trace_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    predictions = lab.load_qa_predictions(output_path)
    assert exit_code == 0
    assert payload["ready"] is True
    assert len(calls) == 2
    assert [prediction.id for prediction in predictions] == ["case-001"]


def test_vlm_runner_tolerates_small_binary_ppm_shortfall(tmp_path: Path) -> None:
    module = load_run_vlm_controls_script()
    image_data_url = getattr(module, "_image_data_url")
    ppm_path = tmp_path / "short.ppm"
    ppm_path.write_bytes(b"P6\n2 1\n255\n" + bytes([255, 0, 0, 0, 255]))

    data_url = image_data_url(str(ppm_path))

    assert data_url.startswith("data:image/png;base64,")


def test_vlm_runner_pads_tiny_ppm_before_png_encoding() -> None:
    module = load_run_vlm_controls_script()
    ppm_to_png = getattr(module, "_ppm_to_png")

    png = ppm_to_png(b"P3\n2 1\n255\n255 0 0 0 255 0\n")

    width = int.from_bytes(png[16:20], byteorder="big")
    height = int.from_bytes(png[20:24], byteorder="big")
    assert width == 32
    assert height == 32


def test_vlm_runner_rejects_large_binary_ppm_shortfall(tmp_path: Path) -> None:
    module = load_run_vlm_controls_script()
    image_data_url = getattr(module, "_image_data_url")
    ppm_path = tmp_path / "too-short.ppm"
    ppm_path.write_bytes(b"P6\n2 1\n255\n" + bytes([255, 0, 0]))

    try:
        image_data_url(str(ppm_path))
    except lab.SpatialQAError as exc:
        assert "PPM binary payload has unexpected length" in str(exc)
    else:
        raise AssertionError("large PPM payload shortfall must be rejected")


def test_send_chat_completion_retries_empty_response(
    monkeypatch: MonkeyPatch,
) -> None:
    module = load_run_vlm_controls_script()
    send_chat_completion = getattr(module, "_send_chat_completion")
    calls: list[str] = []

    class FakeResponse:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return self.body

    def fake_urlopen(_request: object, *, timeout: int) -> FakeResponse:
        calls.append(str(timeout))
        if len(calls) == 1:
            return FakeResponse(b"")
        return FakeResponse(
            json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode("utf-8")
        )

    url_lib = getattr(module, "url" + "lib")
    monkeypatch.setattr(url_lib.request, "urlopen", fake_urlopen)

    response = send_chat_completion(
        {"messages": []},
        api_key="test-key",
        base_url="https://example.invalid/v1",
    )

    assert response["choices"][0]["message"]["content"] == "{}"
    assert calls == ["120", "120"]


def test_send_chat_completion_does_not_retry_http_error(
    monkeypatch: MonkeyPatch,
) -> None:
    module = load_run_vlm_controls_script()
    send_chat_completion = getattr(module, "_send_chat_completion")
    calls: list[str] = []
    url_lib = getattr(module, "url" + "lib")

    def fake_urlopen(_request: object, *, timeout: int) -> object:
        calls.append(str(timeout))
        raise url_lib.error.HTTPError(
            "https://example.invalid/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"message":"bad auth"}'),
        )

    monkeypatch.setattr(url_lib.request, "urlopen", fake_urlopen)

    try:
        send_chat_completion(
            {"messages": []},
            api_key="test-key",
            base_url="https://example.invalid/v1",
        )
    except lab.SpatialQAError as exc:
        assert "HTTP Error 401: Unauthorized" in str(exc)
        assert "bad auth" in str(exc)
    else:
        raise AssertionError("HTTPError details must be raised without retrying")
    assert calls == ["120"]


def test_structured_response_normalization_corrects_prompt_case_id_mismatch(
    tmp_path: Path,
) -> None:
    module = load_run_vlm_controls_script()
    request_bundle = _request_bundle(tmp_path, case_id="case-001")
    bundle = json.loads(request_bundle.read_text(encoding="utf-8"))
    case = bundle["case_inputs"][0]
    normalize = getattr(module, "_normalize_structured_response")
    prediction_from_response = getattr(module, "_prediction_from_structured_response")
    prompt_case_id = getattr(module, "_prompt_case_id")("case-001")
    response = json.loads(_fake_structured_response(prompt_case_id)["choices"][0]["message"]["content"])
    response["case_id"] = prompt_case_id[:-2] + "xy"

    normalized = normalize(case, response, frame_index={})
    prediction = prediction_from_response(case, normalized)

    assert normalized["case_id"] == prompt_case_id
    assert normalized["normalization"]["applied"] is True
    assert "corrected_case_id_from_prompt" in normalized["normalization"]["changes"]
    assert "case_id_mismatch" in normalized["normalization"]["warnings"]
    assert prediction.id == "case-001"


def _fake_structured_response(case_id: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "case_id": case_id,
                            "answer": {
                                "confidence": 0.82,
                                "current_location": None,
                                "label": "apple",
                                "object_id": "apple_1",
                                "visible": True,
                            },
                            "answer_text": "The apple is visible but no support id is available.",
                            "confidence": 0.82,
                            "evidence": [
                                {
                                    "object_ids": ["apple_1"],
                                    "relation_ids": [],
                                    "source_id": "episode-001:scene:0001",
                                }
                            ],
                            "error": "relation_not_observed",
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                }
            }
        ]
    }


def _request_bundle(tmp_path: Path, *, case_id: str = "case-001") -> Path:
    rgb_path = tmp_path / "frames" / "0001.ppm"
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_path.write_text("P3\n1 1\n255\n255 0 0\n", encoding="utf-8")
    bundle = {
        "schema_version": "dsg-spatialqa-lab.offline-control-prediction-request-bundle.v1",
        "action": "offline_control_prediction_request_bundle",
        "case_count": 1,
        "case_inputs": [
            {
                "answer_schema_hint": {
                    "answer_type": "object_location",
                    "required_answer_fields": ["object_id", "label", "current_location"],
                },
                "answer_type": "object_location",
                "case_id": case_id,
                "choices": [],
                "difficulty": "easy",
                "episode_id": "episode-001",
                "frames": [
                    {
                        "episode_id": "episode-001",
                        "frame_id": "episode-001:scene:0001",
                        "rgb_digest": "rgb-digest",
                        "rgb_path": str(rgb_path),
                        "scene_id": "scene",
                        "step": 1,
                    }
                ],
                "primary_frame": {
                    "episode_id": "episode-001",
                    "frame_id": "episode-001:scene:0001",
                    "rgb_digest": "rgb-digest",
                    "rgb_path": str(rgb_path),
                    "scene_id": "scene",
                    "step": 1,
                },
                "question": {"object_id": "apple_1", "type": "object_location"},
                "question_text": "Where is the apple?",
                "question_type": "object_location",
                "reference_frame": None,
                "scene_id": "scene",
                "step": 1,
                "tags": ["real"],
                "target": {"label": "apple", "object_id": "apple_1"},
            }
        ],
    }
    path = tmp_path / "request-bundle.json"
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
