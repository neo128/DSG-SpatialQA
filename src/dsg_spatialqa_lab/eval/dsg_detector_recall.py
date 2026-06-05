from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab.qa import DETECTOR_SUPPORT_FALLBACK_SUPPORTED_LABELS


DSG_DETECTOR_RECALL_HANDOFF_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.dsg-detector-recall-handoff.v1"
)
DSG_DETECTOR_RECALL_FORBIDDEN_FIELDS = frozenset(
    {
        "answer",
        "evidence_edges",
        "evidence_nodes",
        "gold",
        "gold_answer",
        "gold_evidence",
        "gold_evidence_edges",
        "gold_evidence_nodes",
        "gold_location",
        "gold_support_label",
        "required_edges",
        "required_nodes",
    }
)
DEFAULT_QUERY_DIAGNOSTIC_RECALL_STATUSES = (
    "query_error",
    "support_fallback_missing",
)


def dsg_detector_recall_handoff(
    gap_report: Mapping[str, Any],
    frame_index: Sequence[Mapping[str, Any]],
    *,
    support_label_vocabulary: Sequence[str] = tuple(
        sorted(DETECTOR_SUPPORT_FALLBACK_SUPPORTED_LABELS)
    ),
) -> dict[str, Any]:
    support_labels = {_canonical_label(label) for label in support_label_vocabulary}
    support_labels.discard("")
    rows_by_key = _frame_index_rows_by_key(frame_index)
    frames: dict[tuple[str, str, int], dict[str, Any]] = {}
    missing_cases: list[dict[str, Any]] = []

    for case in _gap_cases(gap_report):
        episode_id = _optional_str(case.get("episode_id"))
        case_id = _optional_str(case.get("case_id"))
        target_label = _canonical_label(case.get("target_label"))
        step = case.get("step")
        if episode_id is None or case_id is None or not isinstance(step, int):
            continue
        row = _matching_frame_index_row(rows_by_key, episode_id=episode_id, step=step)
        if row is None:
            missing_cases.append(
                {
                    "case_id": case_id,
                    "episode_id": episode_id,
                    "step": step,
                    **({"target_label": target_label} if target_label else {}),
                }
            )
            continue
        scene_id = _required_str(row.get("scene_id"), "scene_id")
        frame_step = _required_int(row.get("step"), "step")
        frame_key = (episode_id, scene_id, frame_step)
        frame = frames.setdefault(
            frame_key,
            {
                "case_ids": [],
                "depth_path": _frame_path(row, "depth"),
                "episode_id": episode_id,
                "frame_step": frame_step,
                "original_case_steps": [],
                "requested_detection_labels": [],
                "rgb_path": _frame_path(row, "rgb"),
                "scene_id": scene_id,
                "segmentation_path": _frame_path(row, "segmentation"),
                "support_labels": [],
                "target_labels": [],
            },
        )
        _append_unique(frame["case_ids"], case_id)
        _append_unique(frame["original_case_steps"], step)
        if target_label:
            _append_unique(frame["target_labels"], target_label)
        for label in _visible_support_labels(row, support_labels):
            _append_unique(frame["support_labels"], label)

    required_frames = []
    for frame in sorted(frames.values(), key=_frame_sort_key):
        frame["case_ids"] = sorted(frame["case_ids"])
        frame["original_case_steps"] = sorted(frame["original_case_steps"])
        frame["support_labels"] = sorted(frame["support_labels"])
        frame["target_labels"] = sorted(frame["target_labels"])
        frame["requested_detection_labels"] = sorted(
            set(frame["support_labels"]) | set(frame["target_labels"])
        )
        required_frames.append(frame)

    handoff: dict[str, Any] = {
        "schema_version": DSG_DETECTOR_RECALL_HANDOFF_SCHEMA_VERSION,
        "gap_report_digest": gap_report.get("report_digest"),
        "label_sources": {
            "support_labels": "frame_index_visible_object_labels",
            "target_labels": "qa_target_labels",
        },
        "missing_frame_cases": sorted(missing_cases, key=_case_sort_key),
        "required_frames": required_frames,
        "summary": _handoff_summary(required_frames, missing_cases),
    }
    handoff["handoff_digest"] = dsg_detector_recall_handoff_digest(handoff)
    return handoff


def dsg_detector_recall_handoff_from_query_diagnostics(
    query_diagnostic_report: Mapping[str, Any],
    frame_index: Sequence[Mapping[str, Any]],
    *,
    included_statuses: Sequence[str] = DEFAULT_QUERY_DIAGNOSTIC_RECALL_STATUSES,
    support_label_vocabulary: Sequence[str] = tuple(
        sorted(DETECTOR_SUPPORT_FALLBACK_SUPPORTED_LABELS)
    ),
) -> dict[str, Any]:
    status_set = {status for status in included_statuses if status}
    cases: list[dict[str, Any]] = []
    for row in _query_diagnostic_cases(query_diagnostic_report):
        if row.get("semantic_match") is not False:
            continue
        status = _optional_str(row.get("location_evidence_status"))
        if status not in status_set:
            continue
        case_id = _optional_str(row.get("case_id"))
        object_id = _optional_str(row.get("object_id"))
        if case_id is None or object_id is None:
            continue
        episode_id = _optional_str(row.get("episode_id"))
        step = row.get("step")
        if episode_id is None or not isinstance(step, int):
            parsed = _parse_query_case_id(case_id)
            if parsed is None:
                continue
            episode_id, step = parsed
        target_label = _target_label_from_object_id(object_id)
        if not target_label:
            continue
        cases.append(
            {
                "case_id": case_id,
                "episode_id": episode_id,
                "step": step,
                "target_label": target_label,
            }
        )
    gap_report = {
        "report_digest": query_diagnostic_report.get("report_digest"),
        "on_to_none_cases": cases,
    }
    handoff = dsg_detector_recall_handoff(
        gap_report,
        frame_index,
        support_label_vocabulary=support_label_vocabulary,
    )
    handoff["query_diagnostic_source"] = {
        "included_statuses": sorted(status_set),
        "report_digest": query_diagnostic_report.get("report_digest"),
    }
    handoff["handoff_digest"] = dsg_detector_recall_handoff_digest(handoff)
    return handoff


def dsg_detector_recall_handoff_digest(handoff: Mapping[str, Any]) -> str:
    payload = {
        key: value for key, value in handoff.items() if key != "handoff_digest"
    }
    text = json.dumps(_json_value(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dsg_detector_recall_handoff_json(handoff: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(handoff), indent=2, sort_keys=True) + "\n"


def save_dsg_detector_recall_handoff(
    handoff: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dsg_detector_recall_handoff_json(handoff), encoding="utf-8")
    return output_path


def load_dsg_detector_recall_handoff(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DSG detector recall handoff must be a JSON object")
    return payload


def validate_dsg_detector_recall_handoff(
    handoff: Mapping[str, Any],
) -> dict[str, Any]:
    expected_digest = dsg_detector_recall_handoff_digest(handoff)
    checks = [
        {
            "name": "schema_version",
            "passed": handoff.get("schema_version")
            == DSG_DETECTOR_RECALL_HANDOFF_SCHEMA_VERSION,
            "expected": DSG_DETECTOR_RECALL_HANDOFF_SCHEMA_VERSION,
            "actual": handoff.get("schema_version"),
        },
        {
            "name": "required_frames",
            "passed": isinstance(handoff.get("required_frames"), list),
            "expected": "list",
            "actual": type(handoff.get("required_frames")).__name__,
        },
        {
            "name": "forbidden_evaluator_only_fields_absent",
            "passed": not _contains_forbidden_field(handoff),
            "expected": sorted(DSG_DETECTOR_RECALL_FORBIDDEN_FIELDS),
            "actual": _forbidden_field_paths(handoff),
        },
        {
            "name": "handoff_digest",
            "passed": handoff.get("handoff_digest") == expected_digest
            or handoff.get("handoff_digest") is None,
            "expected": expected_digest,
            "actual": handoff.get("handoff_digest"),
        },
    ]
    return {
        "action": "validate_dsg_detector_recall_handoff",
        "valid": all(check["passed"] for check in checks),
        "checks": checks,
    }


def _gap_cases(gap_report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cases: list[Mapping[str, Any]] = []
    for key in ("on_to_in_room_cases", "on_to_none_cases"):
        value = gap_report.get(key)
        if isinstance(value, list):
            cases.extend(item for item in value if isinstance(item, Mapping))
    return cases


def _query_diagnostic_cases(
    query_diagnostic_report: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    value = query_diagnostic_report.get("cases")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _parse_query_case_id(case_id: str) -> tuple[str, int] | None:
    parts = case_id.split(":")
    if len(parts) < 2:
        return None
    try:
        step = int(parts[-1])
    except ValueError:
        return None
    episode_id = parts[0]
    if not episode_id:
        return None
    return episode_id, step


def _target_label_from_object_id(object_id: str) -> str:
    parts: list[str] = []
    for part in object_id.split("_"):
        if any(char.isdigit() for char in part):
            break
        parts.append(part)
    return _canonical_label("_".join(parts))


def _frame_index_rows_by_key(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        episode_id = _optional_str(row.get("episode_id"))
        step = row.get("step")
        if episode_id is None or not isinstance(step, int):
            continue
        indexed[(episode_id, step)] = row
    return indexed


def _matching_frame_index_row(
    rows_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    episode_id: str,
    step: int,
) -> Mapping[str, Any] | None:
    for candidate_step in (step, step % 100000):
        row = rows_by_key.get((episode_id, candidate_step))
        if row is not None:
            return row
    return None


def _visible_support_labels(
    row: Mapping[str, Any],
    support_labels: set[str],
) -> list[str]:
    labels = row.get("visible_object_labels")
    if not isinstance(labels, list):
        return []
    return sorted(
        {
            normalized
            for label in labels
            if (normalized := _canonical_label(label)) in support_labels
        }
    )


def _frame_path(row: Mapping[str, Any], kind: str) -> str | None:
    direct = _optional_str(row.get(f"detector_{kind}_path"))
    if direct is not None:
        return direct
    asset_paths = row.get("asset_paths")
    if isinstance(asset_paths, Mapping):
        return _optional_str(asset_paths.get(kind))
    return None


def _handoff_summary(
    frames: Sequence[Mapping[str, Any]],
    missing_cases: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    support_label_count = sum(len(_list_value(frame.get("support_labels"))) for frame in frames)
    target_label_count = sum(len(_list_value(frame.get("target_labels"))) for frame in frames)
    requested_label_count = sum(
        len(_list_value(frame.get("requested_detection_labels"))) for frame in frames
    )
    case_ids = {
        case_id
        for frame in frames
        for case_id in _list_value(frame.get("case_ids"))
        if isinstance(case_id, str)
    } | {
        case_id
        for case in missing_cases
        if isinstance((case_id := case.get("case_id")), str)
    }
    return {
        "case_count": len(case_ids),
        "frame_count": len(frames),
        "frames_with_support_labels": sum(
            1 for frame in frames if _list_value(frame.get("support_labels"))
        ),
        "missing_frame_case_count": len(missing_cases),
        "requested_detection_label_count": requested_label_count,
        "support_label_count": support_label_count,
        "target_label_count": target_label_count,
    }


def _append_unique(items: list[Any], value: Any) -> None:
    if value not in items:
        items.append(value)


def _frame_sort_key(frame: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(frame.get("episode_id") or ""),
        str(frame.get("scene_id") or ""),
        int(frame.get("frame_step") or 0),
    )


def _case_sort_key(case: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(case.get("episode_id") or ""),
        int(case.get("step") or 0),
        str(case.get("case_id") or ""),
    )


def _canonical_label(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("_", "").replace(" ", "").lower()


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _required_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _required_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(value[key]) for key in sorted(value)}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value


def _contains_forbidden_field(value: Any) -> bool:
    return bool(_forbidden_field_paths(value))


def _forbidden_field_paths(value: Any, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{prefix}.{key_text}"
            if key_text in DSG_DETECTOR_RECALL_FORBIDDEN_FIELDS:
                paths.append(item_path)
            paths.extend(_forbidden_field_paths(item, item_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_forbidden_field_paths(item, f"{prefix}[{index}]"))
    return paths
