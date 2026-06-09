from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch

import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
RUN_ADJUDICATION_SCRIPT = (
    ROOT / "external_tools" / "run_vlm_graph_adjudication_active.py"
)


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_run_adjudication_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_vlm_graph_adjudication_active_script",
        RUN_ADJUDICATION_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_adjudication_runner_requires_explicit_network_permission(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_adjudication_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle, vlm_path, graph_path = _inputs(tmp_path)

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--vlm-predictions",
            str(vlm_path),
            "--graph-predictions",
            str(graph_path),
            "--output",
            str(tmp_path / "adjudicated.jsonl"),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ready"] is False
    assert payload["error"] == "network_not_allowed"
    assert not (tmp_path / "adjudicated.jsonl").exists()


def test_adjudication_runner_writes_structured_predictions_without_gold_leakage(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_adjudication_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle, vlm_path, graph_path = _inputs(tmp_path)
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")
    calls: list[dict[str, Any]] = []

    def fake_sender(
        payload: dict[str, Any],
        *,
        api_key: str,
        base_url: str,
    ) -> dict[str, Any]:
        calls.append({"api_key": api_key, "base_url": base_url, "payload": payload})
        text_item = payload["messages"][1]["content"][0]
        prompt_payload = json.loads(text_item["text"])
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": {
                                    "confidence": 0.91,
                                    "current_location": {
                                        "dst_label": "countertop",
                                        "relation": "ON",
                                    },
                                    "decision": "accept_dsg",
                                    "evidence_summary": (
                                        "The DSG candidate has an explicit ON "
                                        "relation to countertop."
                                    ),
                                    "reasoning_summary": "Use the graph memory relation.",
                                },
                                "case_id": prompt_payload["case_id"],
                                "confidence": 0.91,
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
    output = tmp_path / "adjudicated.jsonl"
    trace_output = tmp_path / "trace.jsonl"

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--vlm-predictions",
            str(vlm_path),
            "--graph-predictions",
            str(graph_path),
            "--output",
            str(output),
            "--trace-output",
            str(trace_output),
            "--allow-network",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    predictions = lab.load_qa_predictions(output)
    trace = json.loads(trace_output.read_text(encoding="utf-8").splitlines()[0])
    request_text = calls[0]["payload"]["messages"][1]["content"][0]["text"]
    assert exit_code == 0
    assert payload["ready"] is True
    assert len(predictions) == 1
    assert predictions[0].id == "case-001"
    assert predictions[0].answer["decision"] == "accept_dsg"
    assert predictions[0].answer["current_location"] == {
        "dst": "countertop_001",
        "dst_label": "countertop",
        "relation": "ON",
        "step": 7,
    }
    assert predictions[0].evidence_edges == ("apple_001-ON-countertop_001-7",)
    assert trace["prediction"]["answer"]["decision"] == "accept_dsg"
    forbidden = (
        "gold_answer",
        "gold_evidence",
        "required_edges",
        "required_nodes",
        "visible_object_ids",
        "visible_object_labels",
    )
    assert not any(field in request_text for field in forbidden)


def test_adjudication_runner_accepts_batched_structured_responses(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_adjudication_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle, vlm_path, graph_path = _inputs(tmp_path, case_count=2)
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")
    call_count = 0

    def fake_sender(
        payload: dict[str, Any],
        *,
        api_key: str,
        base_url: str,
    ) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        prompt_payload = json.loads(payload["messages"][1]["content"][0]["text"])
        predictions = []
        for case in prompt_payload["cases"]:
            predictions.append(
                {
                    "answer": {
                        "confidence": 0.9,
                        "current_location": {
                            "dst_label": "countertop",
                            "relation": "ON",
                        },
                        "decision": "accept_dsg",
                        "evidence_summary": "The DSG candidate is specific.",
                    },
                    "case_id": case["case_id"],
                    "confidence": 0.9,
                    "error": None,
                }
            )
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"predictions": predictions},
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
            "--vlm-predictions",
            str(vlm_path),
            "--graph-predictions",
            str(graph_path),
            "--output",
            str(tmp_path / "adjudicated.jsonl"),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
            "--allow-network",
            "--batch-size",
            "2",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["prediction_count"] == 2
    assert call_count == 1


def test_adjudication_runner_retries_empty_message_content(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_adjudication_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle, vlm_path, graph_path = _inputs(tmp_path)
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")
    calls: list[str] = []

    def retry_sender(
        payload: dict[str, Any],
        *,
        api_key: str,
        base_url: str,
    ) -> dict[str, Any]:
        prompt_payload = json.loads(payload["messages"][1]["content"][0]["text"])
        calls.append(prompt_payload["case_id"])
        if len(calls) == 1:
            return {"choices": [{"message": {"content": ""}}]}
        return _adjudication_response(prompt_payload["case_id"])

    monkeypatch.setattr(module, "_send_chat_completion", retry_sender)
    output = tmp_path / "adjudicated.jsonl"

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--vlm-predictions",
            str(vlm_path),
            "--graph-predictions",
            str(graph_path),
            "--output",
            str(output),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
            "--allow-network",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    predictions = lab.load_qa_predictions(output)
    assert exit_code == 0
    assert payload["ready"] is True
    assert len(calls) == 2
    assert [prediction.id for prediction in predictions] == ["case-001"]


def test_adjudication_runner_corrects_single_response_case_id_mismatch(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_adjudication_script()
    main = cast(MainFn, getattr(module, "main"))
    request_bundle, vlm_path, graph_path = _inputs(tmp_path)
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "test-key")

    def mismatch_sender(
        payload: dict[str, Any],
        *,
        api_key: str,
        base_url: str,
    ) -> dict[str, Any]:
        prompt_payload = json.loads(payload["messages"][1]["content"][0]["text"])
        response = _adjudication_response(prompt_payload["case_id"])
        content = json.loads(response["choices"][0]["message"]["content"])
        content["case_id"] = "wrong_case_id"
        response["choices"][0]["message"]["content"] = json.dumps(content)
        return response

    monkeypatch.setattr(module, "_send_chat_completion", mismatch_sender)
    output = tmp_path / "adjudicated.jsonl"

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle),
            "--vlm-predictions",
            str(vlm_path),
            "--graph-predictions",
            str(graph_path),
            "--output",
            str(output),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
            "--allow-network",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    predictions = lab.load_qa_predictions(output)
    assert exit_code == 0
    assert payload["ready"] is True
    assert [prediction.id for prediction in predictions] == ["case-001"]


def _adjudication_response(prompt_case_id: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "answer": {
                                "confidence": 0.91,
                                "current_location": {
                                    "dst_label": "countertop",
                                    "relation": "ON",
                                },
                                "decision": "accept_dsg",
                                "evidence_summary": "The DSG candidate is grounded.",
                            },
                            "case_id": prompt_case_id,
                            "confidence": 0.91,
                            "error": None,
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                }
            }
        ]
    }


def _inputs(tmp_path: Path, *, case_count: int = 1) -> tuple[Path, Path, Path]:
    cases = []
    for index in range(1, case_count + 1):
        cases.append(
            {
                "answer_options": [
                    {
                        "destination_label": "countertop",
                        "option_id": "option_1",
                        "relation": "ON",
                    },
                    {
                        "destination_label": "FloorPlan1",
                        "option_id": "option_2",
                        "relation": "IN_ROOM",
                    },
                ],
                "case_id": f"case-{index:03d}",
                "episode_id": "episode-001",
                "primary_frame": {"rgb_path": "frame.ppm", "step": 7},
                "question_text": "Where is the apple?",
                "question_type": "object_location",
                "scene_id": "FloorPlan1",
                "target": {"label": "apple"},
            }
        )
    request_bundle = tmp_path / "vlm-request-bundle.json"
    request_bundle.write_text(
        json.dumps(
            {
                "leak_free": True,
                "prediction_cases": cases,
                "request_bundle_digest": "bundle-digest",
                "request_count": len(cases),
                "schema_version": (
                    "dsg-spatialqa-lab.active-qa-v2-vlm-request-bundle.v1"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    vlm_path = tmp_path / "vlm.jsonl"
    graph_path = tmp_path / "graph.jsonl"
    lab.save_qa_predictions(
        [
            lab.QAPrediction(
                id=f"case-{index:03d}",
                answer={
                    "answer_text": "I cannot see the support.",
                    "confidence": 0.25,
                    "current_location": {"relation": "UNKNOWN"},
                    "error": "relation_not_observed",
                    "reasoning_summary": "The frame is unclear.",
                },
                confidence=0.25,
                error="relation_not_observed",
            )
            for index in range(1, case_count + 1)
        ],
        vlm_path,
    )
    lab.save_qa_predictions(
        [
            lab.QAPrediction(
                id=f"case-{index:03d}",
                answer={
                    "confidence": 1.0,
                    "current_location": {
                        "dst": "countertop_001",
                        "dst_label": "countertop",
                        "relation": "ON",
                        "step": 7,
                    },
                    "source": "graph_tool_only_dsg",
                },
                evidence_edges=("apple_001-ON-countertop_001-7",),
                evidence_nodes=("apple_001", "countertop_001"),
                confidence=1.0,
            )
            for index in range(1, case_count + 1)
        ],
        graph_path,
    )
    return request_bundle, vlm_path, graph_path
