from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.benchmark import (
    QACase,
    qa_dataset_digest,
    save_qa_dataset,
)
from dsg_spatialqa_lab.observations import (
    ObjectObservation,
    SceneObservation,
    load_scene_observation_sequence,
    scene_observation_sequence_digest,
)
from dsg_spatialqa_lab.schema import SpatialQAError


OBSERVATION_AWARE_QA_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.observation-aware-qa-report.v1"
)
REQUIRED_OBSERVATION_AWARE_EVIDENCE_KINDS = frozenset(("depth", "detector", "rgb"))


def observation_aware_qa_cases(
    cases: Sequence[QACase],
    observations: Sequence[SceneObservation],
    *,
    qa_path: str | Path | None = None,
    observation_sequence_path: str | Path | None = None,
    target_case_count: int | None = None,
) -> tuple[tuple[QACase, ...], dict[str, Any]]:
    if target_case_count is not None and target_case_count < 0:
        raise SpatialQAError("target_case_count must be non-negative")
    sequence_digest = scene_observation_sequence_digest(observations)
    observed_by_object = _visible_detector_observations_by_object(observations)
    generated: list[QACase] = []
    skipped_reasons: dict[str, int] = {}
    object_location_count = 0
    rows: list[dict[str, Any]] = []
    for case in cases:
        if case.question_type != "object_location":
            _count(skipped_reasons, "unsupported_question_type")
            continue
        object_location_count += 1
        object_id = _target_object_id(case)
        if object_id is None:
            _count(skipped_reasons, "missing_target_object_id")
            rows.append(_skip_row(case, "missing_target_object_id"))
            continue
        observed = observed_by_object.get(object_id)
        if observed is None:
            _count(skipped_reasons, "missing_visible_detector_observation")
            rows.append(_skip_row(case, "missing_visible_detector_observation"))
            continue
        observation, obj = observed
        generated_case = _observation_aware_case(
            case,
            observation,
            obj,
            graph_digest=sequence_digest,
        )
        generated.append(generated_case)
        rows.append(
            {
                "base_case_id": case.id,
                "generated_case_id": generated_case.id,
                "object_id": object_id,
                "reason": None,
                "step": observation.step,
            }
        )
    supplemental_count = 0
    if target_case_count is not None and len(generated) < target_case_count:
        supplemental_cases = _supplemental_observation_aware_cases(
            observations,
            graph_digest=sequence_digest,
            existing_object_ids=_generated_object_ids(generated),
            limit=target_case_count - len(generated),
        )
        for supplemental_case in supplemental_cases:
            supplemental_count += 1
            generated.append(supplemental_case)
            rows.append(
                {
                    "base_case_id": None,
                    "generated_case_id": supplemental_case.id,
                    "object_id": supplemental_case.answer["object_id"],
                    "reason": None,
                    "step": supplemental_case.step,
                    "supplemental": True,
                }
            )
    report = observation_aware_qa_report(
        cases,
        generated,
        rows=rows,
        skipped_reasons=skipped_reasons,
        object_location_count=object_location_count,
        observation_sequence_digest_value=sequence_digest,
        qa_path=qa_path,
        observation_sequence_path=observation_sequence_path,
        supplemental_case_count=supplemental_count,
        target_case_count=target_case_count,
    )
    return tuple(generated), report


def observation_aware_qa_report(
    base_cases: Sequence[QACase],
    generated_cases: Sequence[QACase],
    *,
    rows: Sequence[Mapping[str, Any]] = (),
    skipped_reasons: Mapping[str, int] | None = None,
    object_location_count: int | None = None,
    observation_sequence_digest_value: str,
    qa_path: str | Path | None = None,
    observation_sequence_path: str | Path | None = None,
    supplemental_case_count: int = 0,
    target_case_count: int | None = None,
) -> dict[str, Any]:
    reasons = dict(sorted((skipped_reasons or {}).items()))
    summary = {
        "base_case_count": len(base_cases),
        "generated_case_count": len(generated_cases),
        "object_location_case_count": (
            object_location_count
            if object_location_count is not None
            else sum(1 for case in base_cases if case.question_type == "object_location")
        ),
        "skipped_case_count": sum(reasons.values()),
        "skipped_reasons": reasons,
    }
    if target_case_count is not None:
        summary["supplemental_case_count"] = supplemental_case_count
        summary["target_case_count"] = target_case_count
    report: dict[str, Any] = {
        "schema_version": OBSERVATION_AWARE_QA_REPORT_SCHEMA_VERSION,
        "base_qa_digest": qa_dataset_digest(base_cases),
        "base_qa_path": str(qa_path) if qa_path is not None else None,
        "generated_qa_digest": qa_dataset_digest(generated_cases),
        "observation_sequence_digest": observation_sequence_digest_value,
        "observation_sequence_path": (
            str(observation_sequence_path)
            if observation_sequence_path is not None
            else None
        ),
        "summary": summary,
        "cases": [dict(row) for row in rows],
    }
    report["report_digest"] = observation_aware_qa_report_digest(report)
    return report


def observation_aware_qa_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def observation_aware_qa_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_observation_aware_qa_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(observation_aware_qa_report_json(report), encoding="utf-8")
    return output_path


def load_observation_aware_qa_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Observation-aware QA report must be a JSON object")
    return cast(dict[str, Any], payload)


def validate_observation_aware_qa_report(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = report.get("summary")
    cases = report.get("cases")
    expected_digest = observation_aware_qa_report_digest(report)
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == OBSERVATION_AWARE_QA_REPORT_SCHEMA_VERSION,
            "expected": OBSERVATION_AWARE_QA_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "report_digest",
            "passed": report.get("report_digest") == expected_digest,
            "expected": expected_digest,
            "actual": report.get("report_digest"),
        },
        {
            "name": "summary_shape",
            "passed": isinstance(summary, Mapping),
        },
        {
            "name": "cases_shape",
            "passed": isinstance(cases, Sequence) and not isinstance(cases, str),
        },
    ]
    return {
        "action": "validate_observation_aware_qa_report",
        "valid": all(check["passed"] is True for check in checks),
        "checks": checks,
    }


def save_observation_aware_qa_outputs(
    cases: Sequence[QACase],
    report: Mapping[str, Any],
    *,
    output_qa_path: str | Path,
    report_path: str | Path,
) -> None:
    save_qa_dataset(cases, output_qa_path)
    save_observation_aware_qa_report(report, report_path)


def load_observation_aware_inputs(
    *,
    qa_path: str | Path,
    observation_sequence_path: str | Path,
) -> tuple[list[QACase], tuple[SceneObservation, ...]]:
    from dsg_spatialqa_lab.benchmark import load_qa_dataset

    return (
        load_qa_dataset(qa_path),
        load_scene_observation_sequence(observation_sequence_path),
    )


def _visible_detector_observations_by_object(
    observations: Sequence[SceneObservation],
) -> dict[str, tuple[SceneObservation, ObjectObservation]]:
    result: dict[str, tuple[SceneObservation, ObjectObservation]] = {}
    for observation in sorted(observations, key=lambda item: item.step):
        for obj in observation.objects:
            if not _usable_detector_observation(obj):
                continue
            for object_id in _object_aliases(obj):
                previous = result.get(object_id)
                if previous is None or observation.step >= previous[0].step:
                    result[object_id] = (observation, obj)
    return result


def _supplemental_observation_aware_cases(
    observations: Sequence[SceneObservation],
    *,
    graph_digest: str,
    existing_object_ids: set[str],
    limit: int,
) -> tuple[QACase, ...]:
    if limit <= 0:
        return ()
    latest_by_object: dict[str, tuple[SceneObservation, ObjectObservation]] = {}
    for observation in sorted(observations, key=lambda item: item.step):
        for obj in observation.objects:
            if not _usable_detector_observation(obj):
                continue
            if obj.object_id in existing_object_ids:
                continue
            if not _supplemental_scene_episode_ready(obj):
                continue
            previous = latest_by_object.get(obj.object_id)
            if previous is None or observation.step >= previous[0].step:
                latest_by_object[obj.object_id] = (observation, obj)
    selected = sorted(
        latest_by_object.values(),
        key=_supplemental_sort_key,
    )[:limit]
    return tuple(
        _supplemental_observation_aware_case(
            index=index,
            observation=observation,
            obj=obj,
            graph_digest=graph_digest,
        )
        for index, (observation, obj) in enumerate(selected, start=1)
    )


def _supplemental_sort_key(
    item: tuple[SceneObservation, ObjectObservation],
) -> tuple[int, int, str, str]:
    observation, obj = item
    relation = str(obj.attributes.get("current_location_relation", ""))
    relation_priority = {
        "ON": 0,
        "INSIDE": 1,
        "IN_REGION": 2,
        "IN_ROOM": 3,
    }.get(relation, 4)
    semantic_priority = 1 if obj.label in _SUPPLEMENTAL_LOW_VALUE_LABELS else 0
    return (relation_priority, semantic_priority, obj.label, obj.object_id)


_SUPPLEMENTAL_LOW_VALUE_LABELS = frozenset(
    {
        "ceiling",
        "floor",
        "room",
        "wall",
        "window",
    }
)


def _supplemental_scene_episode_ready(obj: ObjectObservation) -> bool:
    return _non_empty_string(obj.attributes.get("episode_id")) and _non_empty_string(
        obj.attributes.get("scene_id")
    )


def _generated_object_ids(cases: Sequence[QACase]) -> set[str]:
    ids: set[str] = set()
    for case in cases:
        object_id = case.answer.get("object_id")
        if isinstance(object_id, str) and object_id != "":
            ids.add(object_id)
    return ids


def _usable_detector_observation(obj: ObjectObservation) -> bool:
    attributes = obj.attributes
    if obj.visible is not True:
        return False
    if attributes.get("source_kind") != "detector":
        return False
    if not _non_empty_string(attributes.get("current_location_id")):
        return False
    if not _non_empty_string(attributes.get("current_location_relation")):
        return False
    if not isinstance(attributes.get("states"), Mapping) or not attributes["states"]:
        return False
    evidence_kinds = {
        item
        for item in attributes.get("evidence_kinds", ())
        if isinstance(item, str)
    }
    return REQUIRED_OBSERVATION_AWARE_EVIDENCE_KINDS.issubset(evidence_kinds)


def _object_aliases(obj: ObjectObservation) -> tuple[str, ...]:
    aliases = [obj.object_id]
    for key in (
        "coverage_target_object_id",
        "collection_target_object_id",
        "target_object_id",
    ):
        value = obj.attributes.get(key)
        if isinstance(value, str) and value != "":
            aliases.append(value)
    return tuple(dict.fromkeys(aliases))


def _target_object_id(case: QACase) -> str | None:
    question_id = case.question.get("object_id")
    if isinstance(question_id, str) and question_id != "":
        return question_id
    answer_id = case.answer.get("object_id")
    if isinstance(answer_id, str) and answer_id != "":
        return answer_id
    return None


def _observation_aware_case(
    base_case: QACase,
    observation: SceneObservation,
    obj: ObjectObservation,
    *,
    graph_digest: str,
) -> QACase:
    attributes = obj.attributes
    location_id = cast(str, attributes["current_location_id"])
    relation = cast(str, attributes["current_location_relation"])
    step = observation.step
    state_node = f"state:{obj.object_id}:{step}"
    return QACase(
        id=f"{base_case.id}:observation_aware:{step}",
        scene_id=base_case.scene_id,
        episode_id=base_case.episode_id,
        graph_digest=graph_digest,
        step=step,
        question=dict(base_case.question),
        question_type=base_case.question_type,
        answer={
            "confidence": float(obj.confidence),
            "current_location": {
                "dst": location_id,
                "relation": relation,
                "step": step,
            },
            "label": obj.label,
            "last_seen_step": step,
            "object_id": obj.object_id,
            "pose": _pose_mapping(obj.pose),
            "state_step": step,
            "visible": True,
        },
        answer_type=base_case.answer_type,
        choices=base_case.choices,
        reference_frame=base_case.reference_frame,
        required_nodes=(obj.object_id, state_node, location_id),
        required_edges=(
            f"{obj.object_id}-{relation}-{location_id}-{step}",
            f"{obj.object_id}-STATE_CHANGED-{state_node}-{step}",
        ),
        difficulty=base_case.difficulty,
        tags=tuple(
            dict.fromkeys(
                (
                    *base_case.tags,
                    "observation_aware",
                    "detector_visible",
                )
            )
        ),
    )


def _supplemental_observation_aware_case(
    *,
    index: int,
    observation: SceneObservation,
    obj: ObjectObservation,
    graph_digest: str,
) -> QACase:
    episode_id = cast(str, obj.attributes["episode_id"])
    scene_id = cast(str, obj.attributes["scene_id"])
    base_case = QACase(
        id=(
            f"{episode_id}:{scene_id}:supplemental_object_location:"
            f"{index:04d}:{obj.object_id}"
        ),
        scene_id=scene_id,
        episode_id=episode_id,
        graph_digest=graph_digest,
        step=observation.step,
        question={"type": "object_location", "object_id": obj.object_id},
        question_type="object_location",
        answer={},
        answer_type="object_location",
        difficulty="easy",
        tags=(
            "generated",
            "benchmark",
            "real",
            "qa",
            "object_location",
            "observation_aware_supplemental",
        ),
    )
    return _observation_aware_case(
        base_case,
        observation,
        obj,
        graph_digest=graph_digest,
    )


def _pose_mapping(pose: Any) -> dict[str, float]:
    return {
        "x": float(pose.x),
        "y": float(pose.y),
        "yaw": float(pose.yaw),
        "z": float(pose.z),
    }


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _skip_row(case: QACase, reason: str) -> dict[str, Any]:
    return {
        "base_case_id": case.id,
        "generated_case_id": None,
        "object_id": _target_object_id(case),
        "reason": reason,
        "step": None,
    }
