from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import struct
from typing import Any, cast
import urllib.error
import urllib.request
import zlib

from dsg_spatialqa_lab import (
    QAPrediction,
    SpatialQAError,
    load_offline_control_prediction_request_bundle,
    load_qa_predictions,
    qa_predictions_digest,
    save_qa_predictions,
)


TRACE_SCHEMA_VERSION = "dsg-spatialqa-lab.vlm-control-run-trace.v1"
DEFAULT_API_KEY_ENV = "DSG_SPATIALQA_DASHSCOPE_API_KEY"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-plus"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Run explicit offline-control VLM requests through an OpenAI-compatible "
            "chat/completions endpoint. This is opt-in and never runs during tests "
            "without a fake sender."
        ),
    )
    parser.add_argument("--request-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--source-kind", default="vlm", choices=("vlm", "multi_frame_vlm"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--normalization-frame-index",
        type=Path,
        help=(
            "Optional local frame-index JSONL used only after the model response "
            "to map visible destination labels such as countertop into stable "
            "object ids. This file is never sent to the VLM."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help=(
            "Optional maximum image count for multi_frame_vlm requests. The "
            "selected window always includes the primary frame when available."
        ),
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Required before any external VLM request is sent.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from existing output predictions and trace files, skipping "
            "case ids that have already been written."
        ),
    )
    parser.add_argument(
        "--replay-trace",
        type=Path,
        help=(
            "Rebuild predictions from a local VLM trace JSONL raw_response field "
            "without sending network requests."
        ),
    )
    args = parser.parse_args(argv)

    if args.replay_trace is not None:
        return _run_replay_mode(args)

    if not args.allow_network:
        _emit_json(
            {
                "action": "run_vlm_controls",
                "ready": False,
                "error": "network_not_allowed",
                "message": "Pass --allow-network to run external VLM requests.",
            }
        )
        return 1
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        _emit_json(
            {
                "action": "run_vlm_controls",
                "ready": False,
                "api_key_env": args.api_key_env,
                "error": "api_key_env_unset",
                "message": f"Set {args.api_key_env} without modifying system keys.",
            }
        )
        return 1

    try:
        bundle = load_offline_control_prediction_request_bundle(args.request_bundle)
        normalization_frame_index = _load_normalization_frame_index(
            args.normalization_frame_index
        )
        cases = _limited_cases(_mapping_sequence(bundle.get("case_inputs")), args.limit)
        case_ids = [_required_str(case, "case_id") for case in cases]
        resumed_predictions = _load_resume_predictions(args.output, case_ids, args.resume)
        resumed_prediction_ids = {prediction.id for prediction in resumed_predictions}
        predictions: list[QAPrediction] = []
        predictions.extend(resumed_predictions)
        traces: list[dict[str, Any]] = _load_resume_traces(args.trace_output, args.resume)
        for case in cases:
            case_id = _required_str(case, "case_id")
            if case_id in resumed_prediction_ids:
                continue
            request_payload, image_refs = _chat_completion_payload(
                case,
                source_kind=args.source_kind,
                max_frames=args.max_frames,
                model=args.model,
            )
            try:
                raw_response = _send_chat_completion(
                    request_payload,
                    api_key=api_key,
                    base_url=args.base_url,
                )
            except (
                OSError,
                SpatialQAError,
                ValueError,
                json.JSONDecodeError,
                urllib.error.URLError,
            ) as exc:
                traces.append(
                    _failure_trace_record(
                        case,
                        source_kind=args.source_kind,
                        model=args.model,
                        request_payload=request_payload,
                        image_refs=image_refs,
                        error=str(exc),
                    )
                )
                if predictions:
                    save_qa_predictions(predictions, args.output)
                _save_trace_records(traces, args.trace_output)
                _emit_json(
                    {
                        "action": "run_vlm_controls",
                        "checkpoint_prediction_count": len(predictions),
                        "checkpoint_trace_count": len(traces),
                        "ready": False,
                        "error": str(exc),
                        "failed_case_id": _required_str(case, "case_id"),
                        "resumed_prediction_count": len(resumed_predictions),
                        "trace_count": len(traces),
                        "trace_output": str(args.trace_output),
                    }
                )
                return 1
            structured_response = _extract_structured_response(raw_response, case=case)
            normalized_response = _normalize_structured_response(
                case,
                structured_response,
                frame_index=normalization_frame_index,
            )
            prediction = _prediction_from_structured_response(
                case,
                normalized_response,
            )
            predictions.append(prediction)
            traces.append(
                _trace_record(
                    case,
                    source_kind=args.source_kind,
                    model=args.model,
                    request_payload=request_payload,
                    image_refs=image_refs,
                    raw_response=raw_response,
                    structured_response=structured_response,
                    normalized_structured_response=normalized_response,
                    prediction=prediction,
                )
            )
            save_qa_predictions(predictions, args.output)
            _save_trace_records(traces, args.trace_output)
        save_qa_predictions(predictions, args.output)
        _save_trace_records(traces, args.trace_output)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
        _emit_json(
            {
                "action": "run_vlm_controls",
                "ready": False,
                "error": str(exc),
            }
        )
        return 1

    _emit_json(
        {
            "action": "run_vlm_controls",
            "base_url": args.base_url,
            "model": args.model,
            "output": str(args.output),
            "prediction_count": len(predictions),
            "prediction_digest": qa_predictions_digest(predictions),
            "ready": True,
            "request_bundle": str(args.request_bundle),
            "resumed_prediction_count": len(resumed_predictions),
            "skipped_case_count": len(resumed_prediction_ids),
            "source_kind": args.source_kind,
            "trace_count": len(traces),
            "trace_output": str(args.trace_output),
        }
    )
    return 0


def _run_replay_mode(args: argparse.Namespace) -> int:
    try:
        bundle = load_offline_control_prediction_request_bundle(args.request_bundle)
        normalization_frame_index = _load_normalization_frame_index(
            args.normalization_frame_index
        )
        cases = _limited_cases(_mapping_sequence(bundle.get("case_inputs")), args.limit)
        source_traces = _load_trace_records(args.replay_trace)
        source_by_case_id = {
            str(record["case_id"]): record
            for record in source_traces
            if isinstance(record.get("case_id"), str)
        }
        predictions: list[QAPrediction] = []
        replay_traces: list[dict[str, Any]] = []
        missing_case_ids: list[str] = []
        missing_raw_response_case_ids: list[str] = []
        for case in cases:
            case_id = _required_str(case, "case_id")
            source_record = source_by_case_id.get(case_id)
            if source_record is None:
                missing_case_ids.append(case_id)
                continue
            raw_response = source_record.get("raw_response")
            if not isinstance(raw_response, Mapping):
                missing_raw_response_case_ids.append(case_id)
                continue
            image_refs = _selected_image_refs(
                case,
                args.source_kind,
                max_frames=args.max_frames,
            )
            structured_response = _extract_structured_response(
                raw_response,
                case=case,
            )
            normalized_response = _normalize_structured_response(
                case,
                structured_response,
                frame_index=normalization_frame_index,
            )
            prediction = _prediction_from_structured_response(
                case,
                normalized_response,
            )
            predictions.append(prediction)
            trace = _trace_record(
                case,
                source_kind=args.source_kind,
                model=args.model,
                request_payload={"model": args.model},
                image_refs=image_refs,
                raw_response=raw_response,
                structured_response=structured_response,
                normalized_structured_response=normalized_response,
                prediction=prediction,
            )
            trace["replay_source_trace"] = str(args.replay_trace)
            replay_traces.append(trace)
        if predictions:
            save_qa_predictions(predictions, args.output)
        _save_trace_records(replay_traces, args.trace_output)
        ready = not missing_case_ids and not missing_raw_response_case_ids
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(
            {
                "action": "replay_vlm_controls",
                "ready": False,
                "error": str(exc),
            }
        )
        return 1

    _emit_json(
        {
            "action": "replay_vlm_controls",
            "missing_case_count": len(missing_case_ids),
            "missing_case_ids": missing_case_ids,
            "missing_raw_response_case_count": len(missing_raw_response_case_ids),
            "missing_raw_response_case_ids": missing_raw_response_case_ids,
            "output": str(args.output),
            "prediction_count": len(predictions),
            "prediction_digest": (
                qa_predictions_digest(predictions) if predictions else None
            ),
            "ready": ready,
            "request_bundle": str(args.request_bundle),
            "replay_trace": str(args.replay_trace),
            "trace_count": len(replay_traces),
            "trace_output": str(args.trace_output),
        }
    )
    return 0 if ready else 1


def _chat_completion_payload(
    case: Mapping[str, Any],
    *,
    source_kind: str,
    max_frames: int | None,
    model: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image_refs = _selected_image_refs(case, source_kind, max_frames=max_frames)
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": json.dumps(
                _visual_prompt_payload(case, source_kind=source_kind),
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
    ]
    content.extend(_image_content_items(image_refs, case))
    return (
        {
            "messages": [
                {
                    "role": "system",
                    "content": _system_prompt(source_kind),
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            "model": model,
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "top_p": 1,
        },
        image_refs,
    )


def _system_prompt(source_kind: str) -> str:
    shared = (
        "Return exactly one JSON object. Do not wrap it in Markdown. "
        "The object must contain case_id, answer, answer_text, confidence, "
        "evidence, observability, reasoning_summary, and error. "
        "For object_location answers, answer must include object_id, label, "
        "current_location, last_seen_step, state_step, visible, pose, and "
        "confidence when evidence supports them. current_location must be null "
        "or an object with relation, dst, and step. Use stable errors such as "
        "target_not_observed, relation_not_observed, state_not_observed, or "
        "ambiguous_evidence when evidence is insufficient. Do not use gold answers, "
        "gold evidence, hidden evaluator fields, or destination ids that are not "
        "grounded in the provided visual evidence. The case_id is an opaque response "
        "identifier; do not treat it as an object id. If a visual object id is not "
        "available, include answer.object_id as null and rely on case_id alignment. "
        "Choose answer.current_location from answer_options when answer_options is "
        "non-empty; choose exactly one allowed answer_option_id when the target "
        "is visible, and copy its relation and destination_label into "
        "current_location. Follow visual_decision_checklist before answering. "
        "If answer.current_location is "
        "null, or answer.visible is false, error must be target_not_observed or "
        "relation_not_observed; never return unknown with error null."
    )
    if source_kind == "vlm":
        return (
            f"{shared} Use the primary RGB frame and question_text. "
            "If the target or relation is not visible, return a structured error "
            "such as target_not_observed or relation_not_observed. Do not guess "
            "hidden object locations."
        )
    return (
        f"{shared} Use the supplied RGB frames and question_text. "
        "If evidence is insufficient, return a structured uncertainty error "
        "instead of guessing."
    )


def _image_content_items(
    image_refs: Sequence[Mapping[str, Any]],
    case: Mapping[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    primary_frame = case.get("primary_frame")
    target_label = _optional_string(_mapping_or_empty(case.get("target")).get("label"))
    for index, frame in enumerate(image_refs, start=1):
        role = _image_role(frame, primary_frame)
        items.append(
            {
                "type": "text",
                "text": _image_role_text(
                    index,
                    frame,
                    role=role,
                    target_label=target_label,
                ),
            }
        )
        items.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(_required_str(frame, "rgb_path"))},
            }
        )
    return items


def _image_role_text(
    index: int,
    frame: Mapping[str, Any],
    *,
    role: str,
    target_label: str | None,
) -> str:
    step = _int_or_none(frame.get("step"))
    suffix = f" step {step}" if step is not None else ""
    if role == "target_crop":
        target = f" for the {target_label}" if target_label is not None else ""
        return (
            f"Image {index}: target crop{target}. Use this crop to confirm the "
            "target object, then use the primary RGB frame to choose the visible "
            "support/place from answer_options."
        )
    if role == "primary_frame":
        return (
            f"Image {index}: primary RGB frame{suffix}. Use this image as the "
            "main evidence for the target location."
        )
    return (
        f"Image {index}: context RGB frame{suffix}. Use it only as supporting "
        "visual evidence for this case."
    )


def _image_role(frame: Mapping[str, Any], primary_frame: object) -> str:
    crop_role = _optional_string(frame.get("crop_role"))
    if crop_role == "target_crop":
        return "target_crop"
    if isinstance(primary_frame, Mapping) and _frame_identity(frame) == _frame_identity(
        primary_frame
    ):
        return "primary_frame"
    return "context_frame"


def _image_roles(
    image_refs: Sequence[Mapping[str, Any]],
    case: Mapping[str, Any],
) -> list[str]:
    primary_frame = case.get("primary_frame")
    return [_image_role(frame, primary_frame) for frame in image_refs]


def _selected_image_refs(
    case: Mapping[str, Any],
    source_kind: str,
    *,
    max_frames: int | None,
) -> list[dict[str, Any]]:
    if source_kind == "vlm":
        primary = case.get("primary_frame")
        if not isinstance(primary, Mapping):
            raise SpatialQAError("VLM request case is missing primary_frame")
        return _with_target_crop_refs(
            [cast(dict[str, Any], dict(primary))],
            case,
        )
    frames = _mapping_sequence(case.get("frames"))
    if not frames:
        raise SpatialQAError("Multi-frame VLM request case is missing frames")
    if max_frames is None:
        return _with_target_crop_refs([dict(frame) for frame in frames], case)
    if max_frames <= 0:
        raise SpatialQAError("--max-frames must be greater than zero")
    return _with_target_crop_refs(
        _limited_frame_window(
        [dict(frame) for frame in frames],
        case.get("primary_frame"),
        max_frames=max_frames,
        ),
        case,
    )


def _with_target_crop_refs(
    image_refs: list[dict[str, Any]],
    case: Mapping[str, Any],
) -> list[dict[str, Any]]:
    crop_ref = _target_crop_image_ref(case.get("target_crop"))
    if crop_ref is None:
        return image_refs
    return [*image_refs, crop_ref]


def _target_crop_image_ref(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    rgb_path = _optional_string(value.get("rgb_path")) or _optional_string(
        value.get("crop_path")
    )
    if rgb_path is None:
        return None
    ref: dict[str, Any] = {
        "crop_role": "target_crop",
        "rgb_path": rgb_path,
    }
    for key in ("bbox_2d_xyxy", "source_frame_id"):
        item = value.get(key)
        if isinstance(item, (str, int, float, bool, list, tuple)) or item is None:
            ref[key] = list(item) if isinstance(item, tuple) else item
    return ref


def _limited_frame_window(
    frames: Sequence[Mapping[str, Any]],
    primary_frame: object,
    *,
    max_frames: int,
) -> list[dict[str, Any]]:
    if len(frames) <= max_frames:
        return [dict(frame) for frame in frames]
    selected: list[dict[str, Any]] = []
    primary_key = (
        _frame_identity(primary_frame) if isinstance(primary_frame, Mapping) else None
    )
    if primary_key is not None:
        for frame in frames:
            if _frame_identity(frame) == primary_key:
                selected.append(dict(frame))
                break
        if not selected and isinstance(primary_frame, Mapping):
            selected.append(dict(primary_frame))
    for frame in reversed(frames):
        if len(selected) >= max_frames:
            break
        if _frame_identity(frame) in {_frame_identity(item) for item in selected}:
            continue
        selected.append(dict(frame))
    return sorted(selected, key=lambda frame: (_int_or_none(frame.get("step")) or 0, _required_str(frame, "rgb_path")))


def _frame_identity(frame: Mapping[str, Any]) -> tuple[str | None, str | None, int | None]:
    return (
        _optional_string(frame.get("frame_id")) or _optional_string(frame.get("rgb_path")),
        _optional_string(frame.get("scene_id")),
        _int_or_none(frame.get("step")),
    )


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
        raise SpatialQAError("VLM response must be a JSON object")
    return cast(dict[str, Any], parsed)


def _extract_structured_response(
    raw_response: Mapping[str, Any],
    *,
    case: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    choices = raw_response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
        raise SpatialQAError("VLM response missing choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise SpatialQAError("VLM response choice must be an object")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise SpatialQAError("VLM response choice missing message")
    content = message.get("content")
    if isinstance(content, Mapping):
        return cast(dict[str, Any], dict(content))
    if not isinstance(content, str):
        raise SpatialQAError("VLM response content must be a JSON string or object")
    try:
        parsed = json.loads(_strip_json_fence(content))
    except json.JSONDecodeError as exc:
        if case is not None:
            recovered = _plain_text_answer_option_response(case, content)
            if recovered is not None:
                return recovered
        raise exc
    if not isinstance(parsed, Mapping):
        if isinstance(parsed, str) and case is not None:
            recovered = _plain_text_answer_option_response(case, parsed)
            if recovered is not None:
                return recovered
        raise SpatialQAError("Structured VLM response must be a JSON object")
    return cast(dict[str, Any], parsed)


def _plain_text_answer_option_response(
    case: Mapping[str, Any],
    content: str,
) -> dict[str, Any] | None:
    option_id = _unique_allowed_answer_option_id_in_text(case, content)
    if option_id is None:
        return None
    target = _mapping_or_empty(case.get("target"))
    target_label = _canonical_label(_optional_string(target.get("label")))
    return {
        "case_id": _prompt_case_id(_required_str(case, "case_id")),
        "answer": {
            "answer_option_id": option_id,
            "confidence": 0.0,
            "current_location": None,
            "label": target_label,
            "object_id": None,
            "visible": True,
        },
        "answer_text": content,
        "confidence": 0.0,
        "evidence": [],
        "error": None,
        "recovered_from": "plain_text_answer_option",
    }


def _unique_allowed_answer_option_id_in_text(
    case: Mapping[str, Any],
    content: str,
) -> str | None:
    options = _visual_answer_options(case.get("answer_options"))
    allowed_ids = {
        str(option["option_id"])
        for option in options
        if isinstance(option.get("option_id"), str)
    }
    if not allowed_ids:
        return None
    matches = [
        option_id
        for option_id in sorted(allowed_ids)
        if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(option_id)}(?![A-Za-z0-9_])",
            content,
        )
    ]
    if len(matches) != 1:
        ordinal_option_id = _answer_option_id_from_ordinal_text(options, content)
        return ordinal_option_id
    return matches[0]


def _answer_option_id_from_ordinal_text(
    options: Sequence[Mapping[str, Any]],
    content: str,
) -> str | None:
    indexes: set[int] = set()
    for pattern in (
        r"\b(?:option|choice)\s*(?P<number>\d+)\b",
        r"^\s*(?P<number>\d+)\s*$",
    ):
        for match in re.finditer(pattern, content, flags=re.IGNORECASE):
            indexes.add(int(match.group("number")) - 1)
    for pattern in (
        r"\b(?:option|choice)\s*(?P<letter>[A-Z])\b",
        r"^\s*(?P<letter>[A-Z])\s*$",
    ):
        for match in re.finditer(pattern, content, flags=re.IGNORECASE):
            indexes.add(ord(match.group("letter").upper()) - ord("A"))
    valid_indexes = {
        index
        for index in indexes
        if 0 <= index < len(options)
        and isinstance(options[index].get("option_id"), str)
    }
    if len(valid_indexes) != 1:
        return None
    index = next(iter(valid_indexes))
    return str(options[index]["option_id"])


def _prediction_from_structured_response(
    case: Mapping[str, Any],
    response: Mapping[str, Any],
) -> QAPrediction:
    case_id = _required_str(case, "case_id")
    response_case_id = _required_str(response, "case_id")
    prompt_case_id = _prompt_case_id(case_id)
    if response_case_id not in {case_id, prompt_case_id}:
        raise SpatialQAError(
            "Structured VLM response case_id mismatch: "
            f"{response_case_id} != {prompt_case_id}"
        )
    answer = _prediction_answer_mapping(response)
    diagnostics = _structured_response_diagnostics(case, response)
    evidence_nodes: list[str] = []
    evidence_edges: list[str] = []
    evidence = response.get("evidence", [])
    if isinstance(evidence, Sequence) and not isinstance(evidence, str):
        for item in evidence:
            if isinstance(item, Mapping):
                evidence_nodes.extend(_string_items(item.get("object_ids")))
                evidence_edges.extend(_string_items(item.get("relation_ids")))
    error = _optional_error(response.get("error"))
    if error is None:
        error = _implicit_observability_error(case, {**diagnostics, "answer": answer})
    if error is None and diagnostics["answer_schema_valid"] is not True:
        error = "answer_schema_mismatch"
    return QAPrediction(
        id=case_id,
        answer=answer,
        evidence_nodes=tuple(_unique_strings(evidence_nodes)),
        evidence_edges=tuple(_unique_strings(evidence_edges)),
        confidence=_float_or_zero(response.get("confidence")),
        error=error,
    )


def _visual_prompt_payload(case: Mapping[str, Any], *, source_kind: str) -> dict[str, Any]:
    return {
        "answer_schema_hint": case.get("answer_schema_hint"),
        "answer_option_response_schema": _visual_answer_option_response_schema(
            case.get("answer_option_response_schema")
        ),
        "answer_type": case.get("answer_type"),
        "answer_options": _visual_answer_options(case.get("answer_options")),
        "case_id": _prompt_case_id(_required_str(case, "case_id")),
        "choices": case.get("choices", []),
        "question": _visual_question_payload(case.get("question")),
        "question_text": case.get("question_text"),
        "question_type": case.get("question_type"),
        "source_kind": source_kind,
        "support_candidates": _visual_support_candidates(
            case.get("support_candidates")
        ),
        "target": _visual_target_payload(case.get("target")),
        "target_crop": _visual_target_crop_payload(case.get("target_crop")),
        "visual_answer_option_strategy": _visual_answer_option_strategy(
            case.get("answer_options")
        ),
        "visual_decision_checklist": _visual_decision_checklist(),
        "visual_location_contract": _visual_location_contract(),
    }


def _visual_question_payload(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    payload: dict[str, Any] = {}
    for key in ("type", "relation", "attribute"):
        item = value.get(key)
        if isinstance(item, (str, int, float, bool)) or item is None:
            payload[key] = item
    return payload


def _visual_target_payload(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    label = value.get("label")
    if not isinstance(label, str) or label == "":
        return None
    return {"label": label}


def _visual_target_crop_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"available": False}
    payload: dict[str, Any] = {"available": True}
    bbox = value.get("bbox_2d_xyxy")
    if (
        isinstance(bbox, Sequence)
        and not isinstance(bbox, str)
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in bbox)
    ):
        payload["bbox_2d_xyxy"] = [float(item) if isinstance(item, float) else item for item in bbox]
    source_frame_id = _optional_string(value.get("source_frame_id"))
    if source_frame_id is not None:
        payload["source_frame_id"] = source_frame_id
    return payload


def _visual_support_candidates(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    candidates: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            label = _canonical_label(item)
            if label is not None:
                candidates.append({"label": label})
            continue
        if not isinstance(item, Mapping):
            continue
        label = _canonical_label(_optional_string(item.get("label")))
        if label is None:
            continue
        candidate: dict[str, Any] = {"label": label}
        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            candidate["confidence"] = float(confidence)
        relation_hint = _normalize_relation(item.get("relation_hint"))
        if relation_hint is not None:
            candidate["relation_hint"] = relation_hint
        candidates.append(candidate)
    return candidates


def _visual_answer_options(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    options: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        option_id = _optional_string(item.get("option_id"))
        destination_label = _canonical_label(_optional_string(item.get("destination_label")))
        relation = _normalize_relation(item.get("relation"))
        if option_id is None or destination_label is None or relation is None:
            continue
        options.append(
            {
                "destination_label": destination_label,
                "option_id": option_id,
                "relation": relation,
            }
        )
    return options


def _visual_answer_option_response_schema(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    allowed_ids = _string_items(value.get("allowed_answer_option_ids"))
    if not allowed_ids:
        return None
    return {
        "answer_current_location_rule": _optional_string(
            value.get("answer_current_location_rule")
        ),
        "answer_option_id_field": _optional_string(
            value.get("answer_option_id_field")
        )
        or "answer.answer_option_id",
        "allowed_answer_option_ids": list(allowed_ids),
        "required_when_answer_options_present": (
            value.get("required_when_answer_options_present") is True
        ),
    }


def _visual_answer_option_strategy(value: object) -> dict[str, str] | None:
    if not _visual_answer_options(value):
        return None
    return {
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


def _visual_decision_checklist() -> list[dict[str, str]]:
    return [
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


def _visual_location_contract() -> dict[str, Any]:
    return {
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


def _prompt_case_id(case_id: str) -> str:
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:16]
    return f"vlm_case_{digest}"


def _trace_record(
    case: Mapping[str, Any],
    *,
    source_kind: str,
    model: str,
    request_payload: Mapping[str, Any],
    image_refs: Sequence[Mapping[str, Any]],
    raw_response: Mapping[str, Any],
    structured_response: Mapping[str, Any],
    normalized_structured_response: Mapping[str, Any],
    prediction: QAPrediction,
) -> dict[str, Any]:
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "case_id": prediction.id,
        "image_refs": [_trace_image_ref(frame) for frame in image_refs],
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
            "frame_count": len(image_refs),
            "image_roles": _image_roles(image_refs, case),
            "model": request_payload.get("model"),
            "question_text": case.get("question_text"),
            "source_kind": source_kind,
            "target": case.get("target"),
            "visual_prompt_payload": _visual_prompt_payload(
                case,
                source_kind=source_kind,
            ),
        },
        "raw_response": dict(raw_response),
        "source_kind": source_kind,
        "structured_response": dict(structured_response),
        "normalized_structured_response": dict(normalized_structured_response),
        "diagnostics": _structured_response_diagnostics(
            case,
            normalized_structured_response,
        ),
    }


def _failure_trace_record(
    case: Mapping[str, Any],
    *,
    source_kind: str,
    model: str,
    request_payload: Mapping[str, Any],
    image_refs: Sequence[Mapping[str, Any]],
    error: str,
) -> dict[str, Any]:
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "case_id": _required_str(case, "case_id"),
        "diagnostics": {
            "answer_schema_valid": False,
            "evidence_present": False,
            "failure_reasons": ["external_call_failed"],
            "missing_answer_fields": _required_answer_fields(case),
            "parse_valid": False,
            "response_error": error,
        },
        "image_refs": [_trace_image_ref(frame) for frame in image_refs],
        "model": model,
        "prediction": None,
        "request": {
            "case_id": case.get("case_id"),
            "frame_count": len(image_refs),
            "image_roles": _image_roles(image_refs, case),
            "model": request_payload.get("model"),
            "question_text": case.get("question_text"),
            "source_kind": source_kind,
            "target": case.get("target"),
            "visual_prompt_payload": _visual_prompt_payload(
                case,
                source_kind=source_kind,
            ),
        },
        "raw_response": None,
        "source_kind": source_kind,
        "structured_response": None,
    }


def _trace_image_ref(frame: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: frame.get(key)
        for key in (
            "depth_path",
            "bbox_2d_xyxy",
            "crop_role",
            "episode_id",
            "frame_id",
            "rgb_digest",
            "rgb_path",
            "scene_id",
            "segmentation_path",
            "source_frame_id",
            "step",
        )
        if key in frame
    }


def _image_data_url(path: str) -> str:
    image_path = Path(path)
    payload = image_path.read_bytes()
    suffix = image_path.suffix.lower()
    if suffix == ".ppm":
        payload = _ppm_to_png(payload)
        mime = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    else:
        mime = "application/octet-stream"
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _ppm_to_png(payload: bytes) -> bytes:
    tokens, data_start = _ppm_header_tokens(payload)
    if len(tokens) < 4 or tokens[0] not in {b"P3", b"P6"}:
        raise SpatialQAError("Unsupported PPM image")
    width = int(tokens[1])
    height = int(tokens[2])
    max_value = int(tokens[3])
    if width <= 0 or height <= 0 or max_value <= 0 or max_value > 255:
        raise SpatialQAError("Unsupported PPM dimensions or max value")
    if tokens[0] == b"P6":
        rgb = payload[data_start : data_start + width * height * 3]
        if len(rgb) != width * height * 3:
            raise SpatialQAError("PPM binary payload has unexpected length")
    else:
        values = [int(token) for token in payload[data_start:].split()]
        if len(values) != width * height * 3:
            raise SpatialQAError("PPM text payload has unexpected length")
        rgb = bytes(values)
    scanlines = b"".join(
        b"\x00" + rgb[row * width * 3 : (row + 1) * width * 3]
        for row in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(scanlines))
        + _png_chunk(b"IEND", b"")
    )


def _ppm_header_tokens(payload: bytes) -> tuple[list[bytes], int]:
    tokens: list[bytes] = []
    index = 0
    while len(tokens) < 4 and index < len(payload):
        while index < len(payload) and payload[index:index + 1].isspace():
            index += 1
        if index < len(payload) and payload[index:index + 1] == b"#":
            while index < len(payload) and payload[index:index + 1] not in {b"\n", b"\r"}:
                index += 1
            continue
        start = index
        while index < len(payload) and not payload[index:index + 1].isspace():
            index += 1
        if start != index:
            tokens.append(payload[start:index])
    while index < len(payload) and payload[index:index + 1].isspace():
        index += 1
    return tokens, index


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _save_trace_records(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _load_resume_predictions(
    path: Path,
    case_ids: Sequence[str],
    resume: bool,
) -> list[QAPrediction]:
    if not resume or not path.exists():
        return []
    requested_ids = set(case_ids)
    predictions = load_qa_predictions(path)
    return [prediction for prediction in predictions if prediction.id in requested_ids]


def _load_resume_traces(path: Path, resume: bool) -> list[dict[str, Any]]:
    if not resume or not path.exists():
        return []
    return _load_trace_records(path)


def _load_trace_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line == "":
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise SpatialQAError(f"VLM trace line {line_number} must be an object")
        records.append(cast(dict[str, Any], dict(payload)))
    return records


def _load_normalization_frame_index(
    path: Path | None,
) -> dict[tuple[str, str, int], tuple[str, ...]]:
    if path is None:
        return {}
    rows: dict[tuple[str, str, int], tuple[str, ...]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line == "":
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise SpatialQAError(f"VLM normalization frame-index line {line_number} must be an object")
        episode_id = _required_str(payload, "episode_id")
        scene_id = _required_str(payload, "scene_id")
        step = _required_int(payload.get("step"), "step")
        rows[(episode_id, scene_id, step)] = _string_items(
            payload.get("visible_object_ids")
        )
    return rows


def _normalize_structured_response(
    case: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    frame_index: Mapping[tuple[str, str, int], Sequence[str]],
) -> dict[str, Any]:
    normalized = json.loads(
        json.dumps(response, separators=(",", ":"), sort_keys=True)
    )
    if not isinstance(normalized, Mapping):
        raise SpatialQAError("Structured VLM response must be a JSON object")
    result = cast(dict[str, Any], normalized)
    normalization: dict[str, Any] = {
        "applied": False,
        "changes": [],
        "frame_index_used": bool(frame_index),
        "warnings": [],
    }
    if case.get("question_type") != "object_location":
        result["normalization"] = normalization
        return result
    answer = dict(_mapping_or_empty(result.get("answer")))
    target = _mapping_or_empty(case.get("target"))
    primary_frame = _mapping_or_empty(case.get("primary_frame"))
    target_id = _optional_string(target.get("object_id"))
    target_label = _canonical_label(_optional_string(target.get("label")))
    answer_label = _canonical_label(_optional_string(answer.get("label")))
    if (
        answer.get("visible") is True
        and target_id is not None
        and _optional_string(answer.get("object_id")) is None
        and (answer_label is None or answer_label == target_label)
    ):
        answer["object_id"] = target_id
        _normalization_change(normalization, "filled_target_object_id")
    if target_label is not None and _optional_string(answer.get("label")) is None:
        answer["label"] = target_label
        _normalization_change(normalization, "filled_target_label")

    location = dict(_mapping_or_empty(answer.get("current_location")))
    explicit_option_id = _explicit_answer_option_id(result, answer, location)
    selected_option = _selected_answer_option(case, result, answer, location)
    if selected_option is not None:
        option_id = _required_str(selected_option, "option_id")
        answer["answer_option_id"] = option_id
        location["option_id"] = option_id
        location["relation"] = _required_str(selected_option, "relation")
        location["dst_label"] = _required_str(selected_option, "destination_label")
        if explicit_option_id is None:
            _normalization_change(normalization, "extracted_answer_option_id_from_text")
        else:
            _normalization_change(normalization, "applied_answer_option_id")
    text_relation, text_destination = _parse_location_text(_answer_text(result, answer))
    relation = _normalize_relation(
        location.get("relation")
        or answer.get("relation")
        or answer.get("relation_label")
        or text_relation
    )
    destination_label = _canonical_label(
        _optional_string(location.get("dst_label"))
        or _optional_string(location.get("destination_label"))
        or _optional_string(answer.get("destination_label"))
        or _optional_string(answer.get("location_label"))
        or _optional_string(answer.get("place"))
        or _generic_destination_label(location.get("dst"), frame_index, primary_frame)
        or text_destination
    )
    if relation is not None:
        location["relation"] = relation
        _normalization_change(normalization, "normalized_relation")
    if destination_label is not None:
        location["dst_label"] = destination_label
        _normalization_change(normalization, "normalized_destination_label")
    destination_id = _optional_string(location.get("dst"))
    if destination_label is not None and (
        destination_id is None
        or not _visible_object_id_exists(frame_index, primary_frame, destination_id)
    ):
        candidates = _visible_candidates_for_label(
            frame_index,
            primary_frame,
            destination_label,
        )
        if len(candidates) == 1:
            location["dst"] = candidates[0]
            _normalization_change(normalization, "filled_destination_object_id")
        elif len(candidates) > 1:
            normalization["warnings"].append(
                {
                    "name": "ambiguous_destination_label",
                    "destination_label": destination_label,
                    "candidate_count": len(candidates),
                }
            )
        elif relation in {"IN_ROOM", "INSIDE"} and destination_label in {
            "bathroom",
            "bedroom",
            "kitchen",
            "livingroom",
            "room",
            "scene",
        }:
            location["dst"] = "ai2thor_room"
            location["dst_label"] = "room"
            location["relation"] = "IN_ROOM"
            _normalization_change(normalization, "filled_room_destination")
    evaluation_step = _int_or_none(case.get("step"))
    frame_step = _int_or_none(primary_frame.get("step"))
    answer_step = evaluation_step if evaluation_step is not None else frame_step
    location_step = _int_or_none(location.get("step"))
    if (
        location
        and answer_step is not None
        and (
            location_step is None
            or (location_step <= 0 and answer_step > 0)
        )
    ):
        location["step"] = answer_step
        _normalization_change(normalization, "filled_location_step")
    if location:
        answer["current_location"] = location
    if answer.get("visible") is True and answer_step is not None:
        if _int_or_none(answer.get("last_seen_step")) is None:
            answer["last_seen_step"] = answer_step
            _normalization_change(normalization, "filled_last_seen_step")
        if _int_or_none(answer.get("state_step")) is None:
            answer["state_step"] = answer_step
            _normalization_change(normalization, "filled_state_step")
    if "pose" not in answer:
        answer["pose"] = None
        _normalization_change(normalization, "filled_null_pose")
    if "confidence" not in answer:
        confidence = result.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            answer["confidence"] = float(confidence)
            _normalization_change(normalization, "filled_answer_confidence")
    result["answer"] = answer
    _append_normalization_evidence(result, answer, primary_frame, normalization)
    result["normalization"] = normalization
    return result


def _selected_answer_option(
    case: Mapping[str, Any],
    response: Mapping[str, Any],
    answer: Mapping[str, Any],
    location: Mapping[str, Any],
) -> dict[str, Any] | None:
    option_id = _explicit_answer_option_id(response, answer, location)
    if option_id is None:
        option_id = _answer_option_id_from_text(response, answer, case)
    if option_id is None:
        return None
    for option in _visual_answer_options(case.get("answer_options")):
        if option.get("option_id") == option_id:
            return option
    return None


def _explicit_answer_option_id(
    response: Mapping[str, Any],
    answer: Mapping[str, Any],
    location: Mapping[str, Any],
) -> str | None:
    return (
        _optional_string(answer.get("answer_option_id"))
        or _optional_string(location.get("option_id"))
        or _optional_string(answer.get("option_id"))
        or _optional_string(response.get("answer_option_id"))
    )


def _answer_option_id_from_text(
    response: Mapping[str, Any],
    answer: Mapping[str, Any],
    case: Mapping[str, Any],
) -> str | None:
    text = _answer_text(response, answer)
    if text is None:
        return None
    allowed_ids = {
        str(option["option_id"])
        for option in _visual_answer_options(case.get("answer_options"))
        if isinstance(option.get("option_id"), str)
    }
    if not allowed_ids:
        return None
    for option_id in sorted(allowed_ids):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(option_id)}(?![A-Za-z0-9_])", text):
            answer["answer_option_id"] = option_id
            return option_id
    return None


def _normalization_change(normalization: dict[str, Any], name: str) -> None:
    normalization["applied"] = True
    changes = normalization["changes"]
    if isinstance(changes, list) and name not in changes:
        changes.append(name)


def _append_normalization_evidence(
    response: dict[str, Any],
    answer: Mapping[str, Any],
    primary_frame: Mapping[str, Any],
    normalization: dict[str, Any],
) -> None:
    node_ids: list[str] = []
    object_id = _optional_string(answer.get("object_id"))
    if object_id is not None and answer.get("visible") is True:
        node_ids.append(object_id)
    location = _mapping_or_empty(answer.get("current_location"))
    destination_id = _optional_string(location.get("dst"))
    if destination_id is not None:
        node_ids.append(destination_id)
    if not node_ids:
        return
    evidence = response.get("evidence")
    evidence_rows = list(evidence) if isinstance(evidence, Sequence) and not isinstance(evidence, str) else []
    evidence_rows.append(
        {
            "object_ids": _unique_strings(node_ids),
            "relation_ids": [],
            "source_id": _optional_string(primary_frame.get("frame_id")),
            "source_kind": "vlm_normalization_frame_index",
        }
    )
    response["evidence"] = evidence_rows
    _normalization_change(normalization, "added_normalization_evidence")


def _visible_candidates_for_label(
    frame_index: Mapping[tuple[str, str, int], Sequence[str]],
    primary_frame: Mapping[str, Any],
    destination_label: str,
) -> tuple[str, ...]:
    key = _frame_key(primary_frame)
    if key is None:
        return ()
    candidates = [
        object_id
        for object_id in frame_index.get(key, ())
        if _object_id_label(object_id) == destination_label
    ]
    return tuple(sorted(candidates))


def _generic_destination_label(
    value: object,
    frame_index: Mapping[tuple[str, str, int], Sequence[str]],
    primary_frame: Mapping[str, Any],
) -> str | None:
    destination = _optional_string(value)
    if destination is None:
        return None
    if _visible_object_id_exists(frame_index, primary_frame, destination):
        return None
    return _canonical_label(destination)


def _visible_object_id_exists(
    frame_index: Mapping[tuple[str, str, int], Sequence[str]],
    primary_frame: Mapping[str, Any],
    object_id: str,
) -> bool:
    key = _frame_key(primary_frame)
    return key is not None and object_id in set(frame_index.get(key, ()))


def _frame_key(frame: Mapping[str, Any]) -> tuple[str, str, int] | None:
    episode_id = _optional_string(frame.get("episode_id"))
    scene_id = _optional_string(frame.get("scene_id"))
    step = _int_or_none(frame.get("step"))
    if episode_id is None or scene_id is None or step is None:
        return None
    return (episode_id, scene_id, step)


def _object_id_label(object_id: str) -> str | None:
    raw_label = object_id.split("_", 1)[0]
    return _canonical_label(raw_label)


def _parse_location_text(value: str | None) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    normalized = _normalize_text(value)
    patterns = (
        ("ON", r"\bon top of\b\s+(?P<dst>.+)$"),
        ("ON", r"\batop\b\s+(?P<dst>.+)$"),
        ("ON", r"\bon\b\s+(?P<dst>.+)$"),
        ("INSIDE", r"\binside\b\s+(?P<dst>.+)$"),
        ("INSIDE", r"\bin\b\s+(?P<dst>.+)$"),
        ("IN_ROOM", r"\bvisible in\b\s+(?P<dst>.+)$"),
    )
    for relation, pattern in patterns:
        match = re.search(pattern, normalized)
        if match is None:
            continue
        return relation, _canonical_label(match.group("dst"))
    return None, _canonical_label(normalized)


def _answer_text(response: Mapping[str, Any], answer: Mapping[str, Any]) -> str | None:
    for key in ("answer_text", "text", "location_text"):
        value = response.get(key) if key == "answer_text" else answer.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _canonical_label(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _normalize_text(value)
    normalized = re.sub(r"\b(the|a|an)\b", " ", normalized)
    normalized = re.sub(r"\b(on|in|inside|top|of|at|near|under)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    aliases = {
        "bath room": "bathroom",
        "bath tub": "bathtub",
        "bathtub basin": "bathtub",
        "butter knife": "butterknife",
        "coffee machine": "coffeemachine",
        "counter": "countertop",
        "counter top": "countertop",
        "counter surface": "countertop",
        "desk lamp": "desklamp",
        "floor lamp": "floorlamp",
        "garbage can": "garbagecan",
        "hand towel": "handtowel",
        "kitchen counter": "countertop",
        "kitchen counter surface": "countertop",
        "living room": "livingroom",
        "paper towel roll": "papertowelroll",
        "side table": "sidetable",
        "spray bottle": "spraybottle",
        "stove burner": "stoveburner",
        "table top": "table",
        "teddy bear": "teddybear",
        "toilet paper": "toiletpaper",
        "tub": "bathtub",
        "watering can": "wateringcan",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == "":
        return None
    return normalized.replace(" ", "")


def _normalize_text(value: str) -> str:
    lowered = value.lower().replace("_", " ").replace("-", " ")
    return re.sub(r"[^a-z0-9 ]+", " ", lowered)


def _normalize_relation(value: object) -> str | None:
    if not isinstance(value, str) or value == "":
        return None
    normalized = value.strip().upper().replace(" ", "_")
    aliases = {
        "IN": "INSIDE",
        "IN_THE_ROOM": "IN_ROOM",
        "ON_TOP_OF": "ON",
        "ROOM": "IN_ROOM",
    }
    return aliases.get(normalized, normalized)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpatialQAError(f"Missing required integer field: {field}")
    return value


def _limited_cases(
    cases: Sequence[Mapping[str, Any]],
    limit: int | None,
) -> tuple[Mapping[str, Any], ...]:
    if limit is None:
        return tuple(cases)
    if limit < 0:
        raise SpatialQAError("--limit must be non-negative")
    return tuple(cases[:limit])


def _strip_json_fence(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _mapping_sequence(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError("Expected a sequence of objects")
    rows: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise SpatialQAError("Expected a sequence of objects")
        rows.append(cast(Mapping[str, Any], item))
    return tuple(rows)


def _mapping_or_empty(value: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def _structured_response_diagnostics(
    case: Mapping[str, Any],
    response: Mapping[str, Any],
) -> dict[str, Any]:
    answer = _prediction_answer_mapping(response)
    evidence = response.get("evidence")
    required_fields = _required_answer_fields(case)
    missing_answer_fields = [
        field for field in required_fields if field not in answer
    ]
    answer_schema_valid = not missing_answer_fields
    evidence_present = (
        isinstance(evidence, Sequence)
        and not isinstance(evidence, str)
        and any(isinstance(item, Mapping) for item in evidence)
    )
    failure_reasons: list[str] = []
    if not answer_schema_valid:
        failure_reasons.append("answer_schema_mismatch")
    if not evidence_present:
        failure_reasons.append("evidence_absent")
    response_error = _optional_error(response.get("error"))
    if response_error is None:
        response_error = _implicit_observability_error(
            case,
            {
                "answer": answer,
                "answer_schema_valid": answer_schema_valid,
            },
        )
    if response_error is not None:
        failure_reasons.append(response_error)
    diagnostics: dict[str, Any] = {
        "answer_schema_valid": answer_schema_valid,
        "evidence_present": evidence_present,
        "failure_reasons": failure_reasons,
        "missing_answer_fields": missing_answer_fields,
        "parse_valid": True,
        "response_error": response_error,
    }
    normalization = _mapping_or_empty(response.get("normalization"))
    if normalization:
        diagnostics["normalization"] = dict(normalization)
    return diagnostics


def _implicit_observability_error(
    case: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> str | None:
    if case.get("question_type") != "object_location":
        return None
    if diagnostics.get("answer_schema_valid") is not True:
        return None
    answer = _mapping_or_empty(diagnostics.get("answer"))
    if answer.get("visible") is False:
        return "target_not_observed"
    location = answer.get("current_location")
    if not isinstance(location, Mapping):
        return "relation_not_observed"
    if _optional_string(location.get("relation")) is None:
        return "relation_not_observed"
    if (
        _optional_string(location.get("dst")) is None
        and _optional_string(location.get("dst_label")) is None
        and _optional_string(location.get("destination_label")) is None
    ):
        return "relation_not_observed"
    return None


def _prediction_answer_mapping(response: Mapping[str, Any]) -> dict[str, Any]:
    answer = dict(_mapping_or_empty(response.get("answer")))
    for key in ("answer_text", "observability", "reasoning_summary"):
        value = response.get(key)
        if key not in answer and isinstance(value, (str, Mapping)):
            answer[key] = dict(value) if isinstance(value, Mapping) else value
    if "confidence" not in answer:
        confidence = response.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            answer["confidence"] = float(confidence)
    return answer


def _required_answer_fields(case: Mapping[str, Any]) -> tuple[str, ...]:
    hint = case.get("answer_schema_hint")
    if isinstance(hint, Mapping):
        fields = _string_items(hint.get("required_answer_fields"))
        if fields:
            return fields
    answer_type = case.get("answer_type")
    if answer_type == "object_location":
        return (
            "object_id",
            "label",
            "current_location",
            "last_seen_step",
            "state_step",
            "visible",
            "pose",
            "confidence",
        )
    return ()


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Missing required string field: {key}")
    return value


def _string_items(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item != "")


def _unique_strings(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _float_or_zero(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _optional_error(value: object) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) and value != "" else "invalid_response"


def _emit_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
