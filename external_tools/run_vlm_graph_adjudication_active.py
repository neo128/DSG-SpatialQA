from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast
import urllib.error
import urllib.request

from dsg_spatialqa_lab import (
    QAPrediction,
    SpatialQAError,
    load_qa_predictions,
    qa_predictions_digest,
    save_qa_predictions,
)


TRACE_SCHEMA_VERSION = "dsg-spatialqa-lab.vlm-graph-adjudication-run-trace.v1"
DEFAULT_API_KEY_ENV = "DSG_SPATIALQA_DASHSCOPE_API_KEY"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-plus"
ACTIVE_QA_V2_REQUEST_BUNDLE_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.active-qa-v2-vlm-request-bundle.v1"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Run external VLM/LLM adjudication over active QA v2 VLM-only and "
            "GraphTool-only DSG predictions. This opt-in runner consumes only "
            "explicit local files and never sends gold answers or required evidence."
        ),
    )
    parser.add_argument(
        "--qa-root",
        type=Path,
        default=Path("handoffs/ai2thor-real-small/inputs/qa-v2-active"),
    )
    parser.add_argument("--request-bundle", type=Path, action="append", default=[])
    parser.add_argument("--vlm-predictions", type=Path, required=True)
    parser.add_argument("--graph-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if not args.allow_network:
        _emit_json(
            {
                "action": "run_vlm_graph_adjudication_active",
                "ready": False,
                "error": "network_not_allowed",
                "message": "Pass --allow-network to run external adjudication requests.",
            }
        )
        return 1
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        _emit_json(
            {
                "action": "run_vlm_graph_adjudication_active",
                "ready": False,
                "api_key_env": args.api_key_env,
                "error": "api_key_env_unset",
                "message": f"Set {args.api_key_env} without modifying system keys.",
            }
        )
        return 1

    try:
        cases = _limited_cases(
            _load_prediction_cases(args.qa_root, args.request_bundle),
            args.limit,
        )
        case_ids = [_required_str(case, "case_id") for case in cases]
        vlm_by_id = {
            prediction.id: prediction for prediction in load_qa_predictions(args.vlm_predictions)
        }
        graph_by_id = {
            prediction.id: prediction for prediction in load_qa_predictions(args.graph_predictions)
        }
        missing_inputs = [
            case_id
            for case_id in case_ids
            if case_id not in vlm_by_id or case_id not in graph_by_id
        ]
        if missing_inputs:
            raise SpatialQAError(
                "Missing VLM or GraphTool predictions for cases: "
                + ", ".join(missing_inputs[:20])
            )
        if args.batch_size <= 0:
            raise SpatialQAError("--batch-size must be greater than zero")
        resumed = _load_resume_predictions(args.output, case_ids, args.resume)
        resumed_ids = {prediction.id for prediction in resumed}
        predictions: list[QAPrediction] = list(resumed)
        traces = _load_resume_traces(args.trace_output, args.resume)
        pending_cases = [
            case for case in cases if _required_str(case, "case_id") not in resumed_ids
        ]
        for batch in _chunks(pending_cases, args.batch_size):
            request_payload = _chat_completion_payload(
                batch,
                vlm_by_id,
                graph_by_id,
                model=args.model,
            )
            try:
                raw_response = _send_chat_completion(
                    request_payload,
                    api_key=api_key,
                    base_url=args.base_url,
                )
                structured = _extract_structured_response(raw_response)
                responses_by_prompt_id = _structured_responses_by_prompt_id(structured)
                batch_predictions: list[QAPrediction] = []
                for case in batch:
                    case_id = _required_str(case, "case_id")
                    prompt_id = _prompt_case_id(case_id)
                    response = responses_by_prompt_id.get(prompt_id)
                    if response is None:
                        raise SpatialQAError(
                            "Missing adjudication response for prompt case: "
                            f"{prompt_id}"
                        )
                    batch_predictions.append(
                        _prediction_from_response(
                            case,
                            response,
                            vlm_by_id[case_id],
                            graph_by_id[case_id],
                        )
                    )
            except (
                OSError,
                SpatialQAError,
                ValueError,
                json.JSONDecodeError,
                urllib.error.URLError,
            ) as exc:
                for case in batch:
                    case_id = _required_str(case, "case_id")
                    traces.append(
                        _failure_trace(
                            case,
                            vlm_by_id[case_id],
                            graph_by_id[case_id],
                            request_payload=request_payload,
                            model=args.model,
                            error=str(exc),
                        )
                    )
                if predictions:
                    save_qa_predictions(predictions, args.output)
                _save_trace_records(traces, args.trace_output)
                _emit_json(
                    {
                        "action": "run_vlm_graph_adjudication_active",
                        "checkpoint_prediction_count": len(predictions),
                        "checkpoint_trace_count": len(traces),
                        "error": str(exc),
                        "failed_case_ids": [
                            _required_str(case, "case_id") for case in batch
                        ],
                        "ready": False,
                    }
                )
                return 1
            predictions.extend(batch_predictions)
            for prediction, case in zip(batch_predictions, batch, strict=True):
                case_id = _required_str(case, "case_id")
                traces.append(
                    _trace_record(
                        case,
                        vlm_by_id[case_id],
                        graph_by_id[case_id],
                        request_payload=request_payload,
                        raw_response=raw_response,
                        structured_response=structured,
                        prediction=prediction,
                        model=args.model,
                    )
                )
            save_qa_predictions(predictions, args.output)
            _save_trace_records(traces, args.trace_output)
        save_qa_predictions(predictions, args.output)
        _save_trace_records(traces, args.trace_output)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
        _emit_json(
            {
                "action": "run_vlm_graph_adjudication_active",
                "ready": False,
                "error": str(exc),
            }
        )
        return 1

    _emit_json(
        {
            "action": "run_vlm_graph_adjudication_active",
            "batch_size": args.batch_size,
            "model": args.model,
            "output": str(args.output),
            "prediction_count": len(predictions),
            "prediction_digest": qa_predictions_digest(predictions),
            "ready": True,
            "resumed_prediction_count": len(resumed),
            "skipped_case_count": len(resumed_ids),
            "trace_count": len(traces),
            "trace_output": str(args.trace_output),
        }
    )
    return 0


def _load_prediction_cases(root: Path, request_bundles: Sequence[Path]) -> list[dict[str, Any]]:
    paths = list(request_bundles)
    if not paths:
        paths = sorted(root.glob("*/vlm-request-bundle.json"))
    if not paths:
        raise SpatialQAError("No active QA v2 request bundles found")
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise SpatialQAError(f"Request bundle must be an object: {path}")
        if payload.get("schema_version") != ACTIVE_QA_V2_REQUEST_BUNDLE_SCHEMA_VERSION:
            raise SpatialQAError(f"Unsupported active QA v2 request bundle: {path}")
        if payload.get("leak_free") is not True:
            raise SpatialQAError(f"Request bundle is not marked leak_free: {path}")
        for case in _mapping_sequence(payload.get("prediction_cases")):
            case_id = _required_str(case, "case_id")
            by_id[case_id] = dict(case)
    return [by_id[case_id] for case_id in sorted(by_id)]


def _limited_cases(cases: Sequence[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return list(cases)
    if limit < 0:
        raise SpatialQAError("--limit must be non-negative")
    return list(cases[:limit])


def _chunks(cases: Sequence[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [
        list(cases[index : index + batch_size])
        for index in range(0, len(cases), batch_size)
    ]


def _chat_completion_payload(
    cases: Sequence[Mapping[str, Any]],
    vlm_by_id: Mapping[str, QAPrediction],
    graph_by_id: Mapping[str, QAPrediction],
    *,
    model: str,
) -> dict[str, Any]:
    prompt_payload = _batch_prompt_payload(cases, vlm_by_id, graph_by_id)
    return {
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            prompt_payload,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    }
                ],
            },
        ],
        "model": model,
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "top_p": 1,
    }


def _system_prompt() -> str:
    return (
        "Return exactly one JSON object and no Markdown. You are adjudicating two "
        "local candidates for an active 3D spatial QA case: a VLM-only visual "
        "answer and a DSG/GraphTool memory answer. Use only the supplied prompt "
        "payload. Do not use gold answers, hidden evaluator fields, required "
        "nodes, required edges, or object ids hidden in case identifiers. The "
        "output must contain case_id, answer, confidence, and error. If the prompt "
        "has multiple cases, return an object with predictions, a list of per-case "
        "objects using that same schema. "
        "answer.decision must be one of accept_vlm, accept_dsg, reject_both, "
        "uncertain. answer.evidence_summary must be a non-empty sentence. "
        "answer.current_location must contain relation and dst_label when the "
        "chosen candidate has a grounded location; otherwise use relation UNKNOWN. "
        "Prefer the DSG candidate when it has a concrete non-room relation such as "
        "ON or INSIDE and the VLM candidate is uncertain, target_not_observed, or "
        "only room-level. Prefer the VLM candidate when the DSG candidate is "
        "UNKNOWN, implausible, or room-level while the visual answer is more "
        "specific. Preserve uncertainty instead of inventing a third answer."
    )


def _batch_prompt_payload(
    cases: Sequence[Mapping[str, Any]],
    vlm_by_id: Mapping[str, QAPrediction],
    graph_by_id: Mapping[str, QAPrediction],
) -> dict[str, Any]:
    if not cases:
        raise SpatialQAError("Cannot adjudicate an empty batch")
    case_payloads = []
    for case in cases:
        case_id = _required_str(case, "case_id")
        case_payloads.append(
            _adjudication_prompt_payload(case, vlm_by_id[case_id], graph_by_id[case_id])
        )
    if len(case_payloads) == 1:
        payload = dict(case_payloads[0])
        payload["cases"] = case_payloads
        payload["response_shape"] = "single_case_object_or_predictions_list"
        return payload
    return {
        "batch_schema_version": "dsg-spatialqa-lab.vlm-graph-adjudication-batch.v1",
        "cases": case_payloads,
        "response_shape": {"predictions": "list of one adjudication object per case"},
    }


def _adjudication_prompt_payload(
    case: Mapping[str, Any],
    vlm_prediction: QAPrediction,
    graph_prediction: QAPrediction,
) -> dict[str, Any]:
    return {
        "answer_options": _answer_options(case.get("answer_options")),
        "case_id": _prompt_case_id(_required_str(case, "case_id")),
        "decision_contract": {
            "allowed_decisions": ["accept_vlm", "accept_dsg", "reject_both", "uncertain"],
            "scope": "Use only the two supplied candidate answers and public case context.",
        },
        "dsg_candidate": _candidate_payload(graph_prediction),
        "episode_id": _optional_string(case.get("episode_id")),
        "question_text": _optional_string(case.get("question_text"))
        or _optional_string(case.get("question")),
        "question_type": _optional_string(case.get("question_type")),
        "scene_id": _optional_string(case.get("scene_id")),
        "situation": _situation_payload(case.get("situation")),
        "target": _target_payload(case.get("target")),
        "vlm_candidate": _candidate_payload(vlm_prediction),
    }


def _candidate_payload(prediction: QAPrediction) -> dict[str, Any]:
    answer = dict(prediction.answer)
    location = _current_location(answer)
    return {
        "answer_text": _optional_string(answer.get("answer_text"))
        or _optional_string(answer.get("text")),
        "confidence": float(prediction.confidence),
        "current_location": location,
        "error": prediction.error or _optional_string(answer.get("error")),
        "evidence_edge_count": len(prediction.evidence_edges),
        "evidence_node_count": len(prediction.evidence_nodes),
        "observability": _optional_string(answer.get("observability")),
        "reasoning_summary": _optional_string(answer.get("reasoning_summary")),
        "source": _optional_string(answer.get("source")),
    }


def _target_payload(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    label = _optional_string(value.get("label"))
    return {"label": label} if label is not None else None


def _situation_payload(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    payload: dict[str, Any] = {}
    step = value.get("step")
    if isinstance(step, int) and not isinstance(step, bool):
        payload["step"] = step
    pose = value.get("agent_pose")
    if isinstance(pose, Mapping):
        payload["agent_pose"] = {
            key: item
            for key, item in pose.items()
            if key in {"x", "y", "z", "yaw"}
            and isinstance(item, (int, float))
            and not isinstance(item, bool)
        }
    return payload


def _answer_options(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    options: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        option_id = _optional_string(item.get("option_id"))
        relation = _optional_string(item.get("relation"))
        dst_label = _optional_string(item.get("destination_label"))
        if option_id is None or relation is None or dst_label is None:
            continue
        options.append(
            {
                "destination_label": dst_label,
                "option_id": option_id,
                "relation": relation,
            }
        )
    return options


def _send_chat_completion(
    payload: dict[str, Any],
    *,
    api_key: str,
    base_url: str,
) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, Mapping):
        raise SpatialQAError("Adjudication response must be a JSON object")
    return cast(dict[str, Any], parsed)


def _extract_structured_response(raw_response: Mapping[str, Any]) -> dict[str, Any]:
    choices = raw_response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
        raise SpatialQAError("Adjudication response missing choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise SpatialQAError("Adjudication response choice must be an object")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise SpatialQAError("Adjudication response choice missing message")
    content = message.get("content")
    if isinstance(content, Mapping):
        return cast(dict[str, Any], dict(content))
    if not isinstance(content, str):
        raise SpatialQAError("Adjudication response content must be JSON")
    parsed = json.loads(_strip_json_fence(content))
    if not isinstance(parsed, Mapping):
        raise SpatialQAError("Structured adjudication response must be an object")
    return cast(dict[str, Any], parsed)


def _structured_responses_by_prompt_id(response: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    predictions = response.get("predictions")
    if isinstance(predictions, Sequence) and not isinstance(predictions, str):
        items = predictions
    else:
        items = [response]
    result: dict[str, Mapping[str, Any]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise SpatialQAError("Batched adjudication predictions must be objects")
        case_id = _required_str(item, "case_id")
        result[case_id] = item
    return result


def _prediction_from_response(
    case: Mapping[str, Any],
    response: Mapping[str, Any],
    vlm_prediction: QAPrediction,
    graph_prediction: QAPrediction,
) -> QAPrediction:
    case_id = _required_str(case, "case_id")
    response_case_id = _required_str(response, "case_id")
    prompt_case_id = _prompt_case_id(case_id)
    if response_case_id not in {case_id, prompt_case_id}:
        raise SpatialQAError(
            "Structured adjudication response case_id mismatch: "
            f"{response_case_id} != {prompt_case_id}"
        )
    answer_payload = response.get("answer")
    if not isinstance(answer_payload, Mapping):
        raise SpatialQAError("Structured adjudication answer must be an object")
    decision = _required_decision(answer_payload)
    selected = _selected_prediction(decision, vlm_prediction, graph_prediction)
    selected_location = _selected_location(answer_payload, selected)
    answer = {
        "adjudication_source": "external_vlm_graph_adjudicator",
        "confidence": _float_or_zero(answer_payload.get("confidence") or response.get("confidence")),
        "current_location": selected_location,
        "decision": decision,
        "evidence_summary": _required_non_empty_str(answer_payload, "evidence_summary"),
        "reasoning_summary": _optional_string(answer_payload.get("reasoning_summary")),
        "selected_candidate": _selected_candidate_name(decision),
    }
    if selected is not None:
        answer["selected_candidate_confidence"] = selected.confidence
        answer["selected_candidate_error"] = selected.error
    return QAPrediction(
        id=case_id,
        answer=answer,
        evidence_nodes=selected.evidence_nodes if selected is not None else (),
        evidence_edges=selected.evidence_edges if selected is not None else (),
        confidence=_float_or_zero(response.get("confidence") or answer_payload.get("confidence")),
        error=_optional_error(response.get("error")),
    )


def _selected_prediction(
    decision: str,
    vlm_prediction: QAPrediction,
    graph_prediction: QAPrediction,
) -> QAPrediction | None:
    if decision == "accept_vlm":
        return vlm_prediction
    if decision == "accept_dsg":
        return graph_prediction
    return None


def _selected_candidate_name(decision: str) -> str:
    if decision == "accept_vlm":
        return "vlm"
    if decision == "accept_dsg":
        return "graph_tool_dsg"
    return "none"


def _selected_location(
    answer_payload: Mapping[str, Any],
    selected: QAPrediction | None,
) -> dict[str, Any]:
    if selected is not None:
        location = _current_location(selected.answer)
        if _known_location(location):
            return location
    supplied = _current_location(answer_payload)
    if _known_location(supplied):
        return supplied
    return {"relation": "UNKNOWN"}


def _current_location(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = value.get("current_location")
    if not isinstance(raw, Mapping):
        raw = value
    relation = _optional_string(raw.get("relation"))
    if relation is None:
        return {"relation": "UNKNOWN"}
    location: dict[str, Any] = {"relation": relation}
    for key in ("dst", "dst_label", "option_id"):
        item = _optional_string(raw.get(key))
        if item is not None:
            location[key] = item
    step = raw.get("step")
    if isinstance(step, int) and not isinstance(step, bool):
        location["step"] = step
    return location


def _known_location(location: Mapping[str, Any]) -> bool:
    relation = location.get("relation")
    if relation == "UNKNOWN":
        return False
    return isinstance(relation, str) and bool(location.get("dst") or location.get("dst_label"))


def _required_decision(answer: Mapping[str, Any]) -> str:
    decision = answer.get("decision")
    if decision not in {"accept_vlm", "accept_dsg", "reject_both", "uncertain"}:
        raise SpatialQAError("Adjudication decision is invalid")
    return str(decision)


def _required_non_empty_str(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SpatialQAError(f"Adjudication response missing {key}")
    return value


def _load_resume_predictions(path: Path, case_ids: Sequence[str], resume: bool) -> list[QAPrediction]:
    if not resume or not path.exists():
        return []
    allowed = set(case_ids)
    return [prediction for prediction in load_qa_predictions(path) if prediction.id in allowed]


def _load_resume_traces(path: Path, resume: bool) -> list[dict[str, Any]]:
    if not resume or not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _save_trace_records(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _trace_record(
    case: Mapping[str, Any],
    vlm_prediction: QAPrediction,
    graph_prediction: QAPrediction,
    *,
    request_payload: Mapping[str, Any],
    raw_response: Mapping[str, Any],
    structured_response: Mapping[str, Any],
    prediction: QAPrediction,
    model: str,
) -> dict[str, Any]:
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "case_id": prediction.id,
        "model": model,
        "prediction": {
            "answer": dict(prediction.answer),
            "confidence": prediction.confidence,
            "error": prediction.error,
            "evidence_edges": list(prediction.evidence_edges),
            "evidence_nodes": list(prediction.evidence_nodes),
            "id": prediction.id,
        },
        "request": {
            "case_id": case.get("case_id"),
            "model": request_payload.get("model"),
            "prompt_payload": _adjudication_prompt_payload(
                case,
                vlm_prediction,
                graph_prediction,
            ),
            "question_text": case.get("question_text"),
        },
        "raw_response": dict(raw_response),
        "structured_response": dict(structured_response),
    }


def _failure_trace(
    case: Mapping[str, Any],
    vlm_prediction: QAPrediction,
    graph_prediction: QAPrediction,
    *,
    request_payload: Mapping[str, Any],
    model: str,
    error: str,
) -> dict[str, Any]:
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "case_id": _required_str(case, "case_id"),
        "diagnostics": {"failure_reasons": ["external_call_failed"], "error": error},
        "model": model,
        "prediction": None,
        "request": {
            "case_id": case.get("case_id"),
            "model": request_payload.get("model"),
            "prompt_payload": _adjudication_prompt_payload(
                case,
                vlm_prediction,
                graph_prediction,
            ),
            "question_text": case.get("question_text"),
        },
        "raw_response": None,
        "structured_response": None,
    }


def _mapping_sequence(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError("Expected a list of objects")
    result: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise SpatialQAError("Expected a list of objects")
        result.append(item)
    return result


def _required_str(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Missing required string field: {key}")
    return value


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _float_or_zero(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _optional_error(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    return None


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _prompt_case_id(case_id: str) -> str:
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:16]
    return f"adjudication_case_{digest}"


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
