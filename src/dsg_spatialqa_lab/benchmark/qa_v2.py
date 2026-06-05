from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.benchmark.qa_generator import QACase, qa_dataset_digest
from dsg_spatialqa_lab.episodes import EpisodeFrame
from dsg_spatialqa_lab.schema import SpatialQAError


QA_V2_CASE_SCHEMA_VERSION = "dsg-spatialqa-lab.qa-case.v2"
QA_V2_SPLIT_REPORT_SCHEMA_VERSION = "dsg-spatialqa-lab.qa-v2-split-report.v1"
QA_V2_SPLIT_NAMES = (
    "full_oracle",
    "observation_aware",
    "situated",
    "temporal",
    "anti_shortcut",
)
TEMPORAL_QUESTION_TYPES = frozenset(
    {
        "agent_history",
        "agent_timeline",
        "object_history",
        "object_timeline",
        "recent_events",
        "relation_timeline",
        "reobserve_targets",
        "scene_delta",
        "state_change",
        "temporal_last_seen",
    }
)
SITUATED_QUESTION_TYPES = frozenset(
    {
        "egocentric_view",
        "relative_direction",
        "relative_relation",
    }
)


def qa_v2_splits(
    cases: Sequence[QACase],
    *,
    observability_report: Mapping[str, Any] | None = None,
    episode_frames: Sequence[EpisodeFrame] = (),
) -> dict[str, list[dict[str, Any]]]:
    obs_splits = _observability_splits(observability_report)
    obs_statuses = _observability_statuses(observability_report)
    frame_index = {
        (frame.episode_id, frame.step): frame
        for frame in episode_frames
    }
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in QA_V2_SPLIT_NAMES}
    for case in cases:
        split_names = _recommended_splits(case, obs_splits)
        for split_name in split_names:
            record = qa_v2_case_record(
                case,
                split=split_name,
                observability_status=obs_statuses.get(case.id),
                obs_splits=obs_splits,
                episode_frame=frame_index.get((case.episode_id, case.step)),
            )
            splits[split_name].append(record)
    return splits


def qa_v2_case_record(
    case: QACase,
    *,
    split: str,
    observability_status: str | None = None,
    obs_splits: Mapping[str, set[str]] | None = None,
    episode_frame: EpisodeFrame | None = None,
) -> dict[str, Any]:
    if split not in QA_V2_SPLIT_NAMES:
        raise SpatialQAError(f"Unsupported QA v2 split: {split}")
    obs_splits = obs_splits or {}
    target = _target(case)
    answer = _answer(case)
    required_nodes = _required_nodes(case, answer)
    record = {
        "schema_version": QA_V2_CASE_SCHEMA_VERSION,
        "id": f"{case.id}:{split}",
        "source_case_id": case.id,
        "scene_id": case.scene_id,
        "episode_id": case.episode_id,
        "split": split,
        "question_type": case.question_type,
        "question_text": _question_text(case, target),
        "situation": _situation(case, episode_frame),
        "target": target,
        "answer": answer,
        "answer_options": _answer_options(answer),
        "required_evidence": {
            "nodes": required_nodes,
            "edges": list(case.required_edges),
            "states": [
                node_id for node_id in required_nodes if node_id.startswith("state:")
            ],
            "frames": [case.step],
        },
        "observability": _observability(case.id, observability_status, obs_splits),
        "anti_shortcut": _anti_shortcut(case, answer),
        "source_v1": {
            "graph_digest": case.graph_digest,
            "answer_type": case.answer_type,
            "tags": list(case.tags),
        },
    }
    return record


def qa_v2_splits_jsonl(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, str]:
    return {
        split_name: "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        )
        for split_name, rows in splits.items()
    }


def qa_v2_records_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        for record in records
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_qa_v2_splits(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for split_name in QA_V2_SPLIT_NAMES:
        rows = splits.get(split_name, ())
        path = root / f"qa-{split_name.replace('_', '-')}.jsonl"
        path.write_text(
            "".join(
                json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        paths[split_name] = str(path)
    return paths


def qa_v2_split_report(
    cases: Sequence[QACase],
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    qa_path: str | Path | None = None,
    episode_path: str | Path | None = None,
    observability_report_path: str | Path | None = None,
    split_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    split_counts = {
        split_name: len(splits.get(split_name, ()))
        for split_name in QA_V2_SPLIT_NAMES
    }
    report: dict[str, Any] = {
        "schema_version": QA_V2_SPLIT_REPORT_SCHEMA_VERSION,
        "qa_path": str(qa_path) if qa_path is not None else None,
        "episode_path": str(episode_path) if episode_path is not None else None,
        "observability_report_path": (
            str(observability_report_path)
            if observability_report_path is not None
            else None
        ),
        "qa_digest": qa_dataset_digest(cases),
        "split_paths": dict(split_paths or {}),
        "summary": {
            "source_case_count": len(cases),
            "split_counts": split_counts,
        },
        "split_digests": {
            split_name: qa_v2_records_digest(splits.get(split_name, ()))
            for split_name in QA_V2_SPLIT_NAMES
        },
    }
    report["report_digest"] = qa_v2_split_report_digest(report)
    return report


def qa_v2_split_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def qa_v2_split_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_qa_v2_split_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(qa_v2_split_report_json(report), encoding="utf-8")
    return output_path


def load_qa_v2_split_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("QA v2 split report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_qa_v2_splits(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    checks = [
        {
            "name": "required_split_names",
            "passed": set(splits) == set(QA_V2_SPLIT_NAMES),
            "expected": list(QA_V2_SPLIT_NAMES),
            "actual": sorted(splits),
        },
        {
            "name": "record_shapes",
            "passed": all(_qa_v2_record_shape_ok(row) for rows in splits.values() for row in rows),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "checks": checks,
    }


def validate_qa_v2_split_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    report_digest = report.get("report_digest")
    expected_digest = qa_v2_split_report_digest(report)
    summary = report.get("summary")
    split_digests = report.get("split_digests")
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == QA_V2_SPLIT_REPORT_SCHEMA_VERSION,
            "expected": QA_V2_SPLIT_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_digest,
            "expected": expected_digest,
            "actual": report_digest,
        },
        {
            "name": "summary_shape",
            "passed": isinstance(summary, Mapping)
            and isinstance(summary.get("split_counts"), Mapping),
        },
        {
            "name": "split_digests_shape",
            "passed": isinstance(split_digests, Mapping)
            and all(isinstance(split_digests.get(split), str) for split in QA_V2_SPLIT_NAMES),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks,
    }


def _qa_v2_record_shape_ok(row: Mapping[str, Any]) -> bool:
    return (
        row.get("schema_version") == QA_V2_CASE_SCHEMA_VERSION
        and isinstance(row.get("id"), str)
        and isinstance(row.get("question_text"), str)
        and row.get("split") in QA_V2_SPLIT_NAMES
        and isinstance(row.get("situation"), Mapping)
        and isinstance(row.get("required_evidence"), Mapping)
        and isinstance(row.get("observability"), Mapping)
        and isinstance(row.get("anti_shortcut"), Mapping)
    )


def _observability_splits(
    observability_report: Mapping[str, Any] | None,
) -> dict[str, set[str]]:
    if observability_report is None:
        return {}
    splits = observability_report.get("splits")
    if not isinstance(splits, Mapping):
        return {}
    result: dict[str, set[str]] = {}
    for split_name, values in splits.items():
        if isinstance(values, Sequence) and not isinstance(values, str):
            result[str(split_name)] = {str(value) for value in values}
    return result


def _observability_statuses(
    observability_report: Mapping[str, Any] | None,
) -> dict[str, str]:
    if observability_report is None:
        return {}
    rows = observability_report.get("cases")
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        return {}
    statuses: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        case_id = row.get("case_id")
        status = row.get("observability_status")
        if isinstance(case_id, str) and isinstance(status, str):
            statuses[case_id] = status
    return statuses


def _recommended_splits(case: QACase, obs_splits: Mapping[str, set[str]]) -> tuple[str, ...]:
    split_names = ["full_oracle"]
    if case.id in obs_splits.get("evidence_observable", set()):
        split_names.append("observation_aware")
    if case.question_type in SITUATED_QUESTION_TYPES or case.reference_frame == "agent_egocentric":
        split_names.append("situated")
    if case.question_type in TEMPORAL_QUESTION_TYPES:
        split_names.append("temporal")
    if _language_prior_risk(case, _answer(case)) != "high":
        split_names.append("anti_shortcut")
    return tuple(split_names)


def _target(case: QACase) -> dict[str, str | None]:
    object_id = case.question.get("object_id")
    if not isinstance(object_id, str):
        object_id = case.question.get("src")
    if not isinstance(object_id, str):
        object_id = case.answer.get("object_id")
    if not isinstance(object_id, str):
        return {"object_id": None, "label": None}
    return {"object_id": object_id, "label": _label_from_object_id(object_id)}


def _answer(case: QACase) -> dict[str, Any]:
    current_location = case.answer.get("current_location")
    if isinstance(current_location, Mapping):
        relation = current_location.get("relation")
        dst = current_location.get("dst")
        step = current_location.get("step", case.step)
        return {
            "relation": relation if isinstance(relation, str) else None,
            "dst": dst if isinstance(dst, str) else None,
            "dst_label": _label_from_object_id(dst) if isinstance(dst, str) else None,
            "step": step if isinstance(step, int) else case.step,
        }
    dst = case.question.get("dst")
    relation = case.question.get("relation")
    return {
        "relation": relation if isinstance(relation, str) else None,
        "dst": dst if isinstance(dst, str) else None,
        "dst_label": _label_from_object_id(dst) if isinstance(dst, str) else None,
        "step": case.step,
    }


def _question_text(case: QACase, target: Mapping[str, str | None]) -> str:
    label = target.get("label") or "target object"
    if case.question_type == "object_location":
        return f"Where is the {label}?"
    if case.question_type == "relative_relation":
        relation = str(case.question.get("relation", "related to")).replace("_", " ").lower()
        dst = case.question.get("dst")
        dst_label = _label_from_object_id(dst) if isinstance(dst, str) else "reference object"
        return f"Is the {label} {relation} the {dst_label}?"
    if case.question_type == "relation_timeline":
        relation = str(case.question.get("relation", "relation")).replace("_", " ")
        dst = case.question.get("dst")
        dst_label = _label_from_object_id(dst) if isinstance(dst, str) else "reference object"
        return f"When was the {label} observed with relation {relation} to the {dst_label}?"
    if case.question_type == "reobserve_targets":
        return "Which targets should the robot reobserve?"
    return f"Answer the {case.question_type.replace('_', ' ')} question for the {label}."


def _situation(case: QACase, episode_frame: EpisodeFrame | None) -> dict[str, Any]:
    reference_frame = case.reference_frame or str(case.question.get("reference_frame", "world"))
    if episode_frame is None:
        return {
            "step": case.step,
            "reference_frame": reference_frame,
            "agent_pose": None,
            "view_frame": None,
            "source": "missing_episode_frame",
        }
    return {
        "step": case.step,
        "reference_frame": reference_frame,
        "agent_pose": episode_frame.agent_pose.to_dict(),
        "view_frame": episode_frame.rgb_path,
        "source": "episode_frame",
    }


def _answer_options(answer: Mapping[str, Any]) -> list[dict[str, str | None]]:
    relation = answer.get("relation") if isinstance(answer.get("relation"), str) else None
    dst_label = answer.get("dst_label") if isinstance(answer.get("dst_label"), str) else None
    options = [{"relation": relation, "dst_label": dst_label}]
    if relation != "IN_ROOM":
        options.append({"relation": "IN_ROOM", "dst_label": "room"})
    options.append({"relation": "UNKNOWN", "dst_label": None})
    return options


def _required_nodes(case: QACase, answer: Mapping[str, Any]) -> list[str]:
    nodes = list(case.required_nodes)
    dst = answer.get("dst")
    if isinstance(dst, str) and dst not in nodes:
        nodes.append(dst)
    return nodes


def _observability(
    case_id: str,
    status: str | None,
    obs_splits: Mapping[str, set[str]],
) -> dict[str, Any]:
    return {
        "observability_status": status,
        "target_observed": case_id in obs_splits.get("target_observable", set()),
        "target_missing": case_id in obs_splits.get("target_missing", set()),
        "evidence_observable": case_id in obs_splits.get("evidence_observable", set()),
        "missing_evidence": case_id in obs_splits.get("missing_evidence", set()),
        "answerable_from_dsg": case_id in obs_splits.get("evidence_observable", set()),
    }


def _anti_shortcut(case: QACase, answer: Mapping[str, Any]) -> dict[str, Any]:
    risk = _language_prior_risk(case, answer)
    return {
        "language_prior_risk": risk,
        "requires_3d_evidence": bool(case.required_edges),
        "has_distractor_same_category": False,
        "has_distractor_same_support": False,
    }


def _language_prior_risk(case: QACase, answer: Mapping[str, Any]) -> str:
    if answer.get("relation") == "IN_ROOM":
        return "high"
    if case.question_type == "object_location":
        return "medium"
    return "low"


def _label_from_object_id(object_id: object) -> str | None:
    if not isinstance(object_id, str) or object_id == "":
        return None
    return object_id.split("_")[0]
