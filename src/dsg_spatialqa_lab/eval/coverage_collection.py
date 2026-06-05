from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.episodes import (
    EpisodeFrame,
    episode_sequence_digest,
    load_episode_sequence,
)
from dsg_spatialqa_lab.eval.qa_observability import (
    load_qa_observability_report,
    qa_observability_report_digest,
)
from dsg_spatialqa_lab.observations import (
    SceneObservation,
    scene_observation_sequence_digest,
)
from dsg_spatialqa_lab.schema import SpatialQAError


COVERAGE_COLLECTION_PLAN_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.coverage-collection-plan.v1"
)
COVERAGE_COLLECTION_REQUEST_BUNDLE_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.coverage-collection-request-bundle.v1"
)
COVERAGE_COLLECTION_TARGET_TASK_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.coverage-collection-target-task.v1"
)
COVERAGE_COLLECTION_TOP_BATCH_RETURN_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.coverage-collection-top-batch-return-report.v1"
)
COVERAGE_COLLECTION_ACCEPTANCE_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.coverage-collection-acceptance-report.v1"
)
DEFAULT_COVERAGE_ACCEPTANCE_EVIDENCE_KINDS = ("depth", "detector", "rgb")
CURRENT_LOCATION_RELATIONS = ("IN_REGION", "IN_ROOM", "INSIDE", "ON")
HANDOFF_TARGET_OBJECT_ID_FIELDS = (
    "collection_target_object_id",
    "coverage_target_object_id",
    "target_object_id",
)


def coverage_collection_plan(
    qa_observability_report: Mapping[str, Any],
    frames: Sequence[EpisodeFrame],
    *,
    qa_observability_report_path: str | Path | None = None,
    episode_paths: Sequence[str | Path] = (),
    target_evidence_observable_count: int = 30,
    target_node_recall_floor: float = 0.5,
    max_viewpoint_hints: int = 3,
) -> dict[str, Any]:
    summary = _mapping(qa_observability_report.get("summary"), "summary")
    missing_target_nodes = sorted(
        set(_string_sequence(summary.get("missing_target_nodes")))
    )
    observations = _object_observations(frames)
    case_index = _case_index(qa_observability_report)
    targets = [
        _collection_target(
            node_id,
            observations[node_id],
            case_index,
            viewpoint_hints=_viewpoint_hints(
                frames,
                observations[node_id],
                max_viewpoint_hints=max_viewpoint_hints,
            ),
        )
        for node_id in missing_target_nodes
        if node_id in observations
    ]
    unresolved = [
        {"object_id": node_id, "missing_reasons": ["target_missing"]}
        for node_id in missing_target_nodes
        if node_id not in observations
    ]
    relation_targets = _relation_evidence_targets(qa_observability_report)
    state_targets = _state_evidence_targets(qa_observability_report)
    report: dict[str, Any] = {
        "schema_version": COVERAGE_COLLECTION_PLAN_SCHEMA_VERSION,
        "action": "coverage_collection_plan",
        "qa_observability_report_path": (
            str(qa_observability_report_path)
            if qa_observability_report_path is not None
            else None
        ),
        "qa_observability_report_digest": qa_observability_report_digest(
            qa_observability_report
        ),
        "episode_paths": [str(path) for path in episode_paths],
        "episode_sequence_digest": episode_sequence_digest(frames),
        "source_use_policy": {
            "episode_metadata_used_for": "collection_planning_only",
            "not_predicted_graph_evidence": True,
            "requires_visible_detector_evidence_before_claim": True,
        },
        "summary": {
            "case_count": _int_value(summary.get("case_count")),
            "current_evidence_observable_count": _int_value(
                summary.get("evidence_observable_count")
            ),
            "current_target_node_recall": _float_value(
                summary.get("target_node_recall")
            ),
            "missing_target_node_count": len(missing_target_nodes),
            "planned_target_count": len(targets),
            "relation_evidence_target_count": len(relation_targets),
            "state_evidence_target_count": len(state_targets),
            "target_evidence_observable_count": target_evidence_observable_count,
            "target_node_recall_floor": target_node_recall_floor,
            "unresolved_target_count": len(unresolved),
        },
        "collection_targets": sorted(
            targets,
            key=lambda item: (
                str(item["episode_id"]),
                str(item["scene_id"]),
                str(item["object_id"]),
            ),
        ),
        "relation_evidence_targets": relation_targets,
        "state_evidence_targets": state_targets,
        "unresolved_targets": unresolved,
    }
    report["digest"] = coverage_collection_plan_digest(report)
    return report


def coverage_collection_plan_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def coverage_collection_plan_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_coverage_collection_plan(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(coverage_collection_plan_json(report), encoding="utf-8")
    return output_path


def load_coverage_collection_plan(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError("Coverage collection plan JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_coverage_collection_plan(report: Mapping[str, Any]) -> dict[str, Any]:
    expected_digest = coverage_collection_plan_digest(report)
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == COVERAGE_COLLECTION_PLAN_SCHEMA_VERSION,
            "expected": COVERAGE_COLLECTION_PLAN_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "action",
            "passed": report.get("action") == "coverage_collection_plan",
            "expected": "coverage_collection_plan",
            "actual": report.get("action"),
        },
        {
            "name": "digest",
            "passed": report.get("digest") == expected_digest,
            "expected": expected_digest,
            "actual": report.get("digest"),
        },
        {
            "name": "metadata_not_prediction_evidence",
            "passed": _source_policy_valid(report.get("source_use_policy")),
            "expected": True,
            "actual": report.get("source_use_policy"),
        },
        {
            "name": "collection_targets",
            "passed": isinstance(report.get("collection_targets"), list),
            "expected": "list",
            "actual": type(report.get("collection_targets")).__name__,
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "checks": checks,
        "digest": expected_digest,
    }


def compare_coverage_collection_plan(report: Mapping[str, Any]) -> dict[str, Any]:
    episode_paths = [Path(path) for path in _string_sequence(report.get("episode_paths"))]
    observability_path = report.get("qa_observability_report_path")
    if not isinstance(observability_path, str) or observability_path == "":
        raise SpatialQAError(
            "Coverage collection plan requires qa_observability_report_path for compare"
        )
    frames = tuple(
        frame for episode_path in episode_paths for frame in load_episode_sequence(episode_path)
    )
    current = coverage_collection_plan(
        load_qa_observability_report(observability_path),
        frames,
        qa_observability_report_path=observability_path,
        episode_paths=episode_paths,
        target_evidence_observable_count=_int_value(
            _mapping(report.get("summary"), "summary").get(
                "target_evidence_observable_count"
            )
        ),
        target_node_recall_floor=_float_value(
            _mapping(report.get("summary"), "summary").get("target_node_recall_floor")
        ),
    )
    validation = validate_coverage_collection_plan(report)
    return {
        "matches": report.get("digest") == current.get("digest")
        and validation["valid"] is True,
        "saved_digest": report.get("digest"),
        "current_digest": current.get("digest"),
        "validation": validation,
    }


def coverage_collection_request_bundle(
    plan: Mapping[str, Any],
    *,
    detector_jsonl_output_path: str | Path,
    observation_sequence_output_path: str | Path,
    acceptance_report_output_path: str | Path,
    required_evidence_kinds: Sequence[str] = DEFAULT_COVERAGE_ACCEPTANCE_EVIDENCE_KINDS,
) -> dict[str, Any]:
    required = tuple(sorted(set(required_evidence_kinds)))
    targets = [
        _request_target(_mapping(item, "collection target"), required)
        for item in _sequence(plan.get("collection_targets"))
    ]
    viewpoint_batches = _viewpoint_batches(targets)
    plan_summary = _mapping(plan.get("summary"), "coverage plan summary")
    target_case_count_gap = max(
        0,
        _int_value(plan_summary.get("target_evidence_observable_count"))
        - _int_value(plan_summary.get("current_evidence_observable_count")),
    )
    bundle: dict[str, Any] = {
        "schema_version": COVERAGE_COLLECTION_REQUEST_BUNDLE_SCHEMA_VERSION,
        "action": "coverage_collection_request_bundle",
        "coverage_collection_plan_digest": _required_text(plan, "digest"),
        "source_use_policy": {
            "episode_metadata_used_for": "collection_planning_only",
            "not_predicted_graph_evidence": True,
            "requires_visible_detector_evidence_before_claim": True,
        },
        "required_evidence_kinds": list(required),
        "planned_outputs": {
            "acceptance_report_output_path": str(acceptance_report_output_path),
            "detector_jsonl_output_path": str(detector_jsonl_output_path),
            "observation_sequence_output_path": str(observation_sequence_output_path),
        },
        "producer_instructions": [
            "Collect or import visible RGB-D frames for every target.",
            "Include visible per-step object state evidence in detection attributes.states.",
            "Return explicit local detector JSONL records only.",
            "Do not write hidden simulator metadata as predicted graph evidence.",
            "Run coverage acceptance before rebuilding the predicted DSG.",
        ],
        "summary": {
            "highest_yield_batch_related_case_count": (
                _int_value(viewpoint_batches[0].get("related_case_count"))
                if viewpoint_batches
                else 0
            ),
            "highest_yield_batch_target_count": (
                _int_value(viewpoint_batches[0].get("target_count"))
                if viewpoint_batches
                else 0
            ),
            "priority_batches_to_reach_target_case_count": (
                _priority_batches_to_reach_case_count(
                    viewpoint_batches,
                    target_case_count_gap,
                )
            ),
            "target_count": len(targets),
            "target_case_count_gap": target_case_count_gap,
            "relation_evidence_target_count": len(
                _sequence(plan.get("relation_evidence_targets"))
            ),
            "state_evidence_target_count": len(
                _sequence(plan.get("state_evidence_targets"))
            ),
            "viewpoint_batch_count": len(viewpoint_batches),
        },
        "targets": sorted(
            targets,
            key=lambda item: (
                str(item["episode_id"]),
                str(item["scene_id"]),
                str(item["object_id"]),
            ),
        ),
        "relation_evidence_targets": list(
            _sequence(plan.get("relation_evidence_targets"))
        ),
        "state_evidence_targets": list(_sequence(plan.get("state_evidence_targets"))),
        "viewpoint_batches": viewpoint_batches,
    }
    bundle["digest"] = coverage_collection_request_bundle_digest(bundle)
    return bundle


def coverage_collection_request_bundle_digest(bundle: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in bundle.items() if key != "digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def coverage_collection_request_bundle_json(bundle: Mapping[str, Any]) -> str:
    return json.dumps(bundle, indent=2, sort_keys=True) + "\n"


def save_coverage_collection_request_bundle(
    bundle: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        coverage_collection_request_bundle_json(bundle),
        encoding="utf-8",
    )
    return output_path


def load_coverage_collection_request_bundle(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError("Coverage collection request bundle JSON must be an object")
    return cast(dict[str, Any], payload)


def coverage_collection_top_batch_handoff_tasks(
    bundle: Mapping[str, Any],
    *,
    max_priority_batches: int = 5,
) -> list[dict[str, Any]]:
    if max_priority_batches < 0:
        raise SpatialQAError("max_priority_batches must be non-negative")
    request_digest = _required_text(bundle, "digest")
    source_use_policy = _mapping(bundle.get("source_use_policy"), "source_use_policy")
    planned_outputs = dict(_mapping(bundle.get("planned_outputs"), "planned_outputs"))
    target_index = {
        _required_text(_mapping(item, "request target"), "object_id"): _mapping(
            item,
            "request target",
        )
        for item in _sequence(bundle.get("targets"))
    }
    state_case_ids = _state_case_ids_by_object_id(
        _sequence(bundle.get("state_evidence_targets"))
    )
    tasks: list[dict[str, Any]] = []
    batches = sorted(
        (
            _mapping(item, "viewpoint batch")
            for item in _sequence(bundle.get("viewpoint_batches"))
        ),
        key=lambda item: _int_value(item.get("priority_rank")),
    )[:max_priority_batches]
    for batch in batches:
        priority_rank = _int_value(batch.get("priority_rank"))
        batch_id = _required_text(batch, "batch_id")
        batch_targets = sorted(
            (_mapping(item, "viewpoint batch target") for item in _sequence(batch.get("targets"))),
            key=lambda item: _required_text(item, "object_id"),
        )
        for batch_target in batch_targets:
            object_id = _required_text(batch_target, "object_id")
            target = target_index.get(object_id)
            if target is None:
                raise SpatialQAError(
                    f"Viewpoint batch target missing request target: {object_id}"
                )
            task: dict[str, Any] = {
                "schema_version": COVERAGE_COLLECTION_TARGET_TASK_SCHEMA_VERSION,
                "action": "coverage_collection_target_task",
                "coverage_collection_request_bundle_digest": request_digest,
                "source_use_policy": dict(source_use_policy),
                "task_id": f"{priority_rank:03d}:{batch_id}:{object_id}",
                "priority_rank": priority_rank,
                "batch_id": batch_id,
                "collection_action": _required_text(batch, "collection_action"),
                "episode_id": _required_text(batch, "episode_id"),
                "scene_id": _required_text(batch, "scene_id"),
                "step": _int_value(batch.get("step")),
                "agent_pose": dict(_mapping(batch.get("agent_pose"), "agent_pose")),
                "batch_execution_plan": dict(
                    _mapping(batch.get("execution_plan"), "execution_plan")
                ),
                "object_id": object_id,
                "ai2thor_object_id": target.get("ai2thor_object_id"),
                "label": _required_text(batch_target, "label"),
                "pose": target.get("pose"),
                "distance_to_target": _float_value(
                    batch_target.get("distance_to_target")
                ),
                "suggested_yaw_to_target": _float_value(
                    batch_target.get("suggested_yaw_to_target")
                ),
                "related_case_ids": sorted(
                    _string_sequence(batch_target.get("related_case_ids"))
                ),
                "planned_outputs": planned_outputs,
                "required_detection_contract": dict(
                    _mapping(
                        target.get("required_detection_contract"),
                        "required_detection_contract",
                    )
                ),
                "state_evidence_required": object_id in state_case_ids,
                "state_evidence_case_ids": sorted(state_case_ids.get(object_id, ())),
            }
            task["task_digest"] = coverage_collection_target_task_digest(task)
            tasks.append(task)
    return tasks


def coverage_collection_target_task_digest(task: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in task.items() if key != "task_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def coverage_collection_top_batch_handoff_tasks_digest(
    tasks: Sequence[Mapping[str, Any]],
) -> str:
    return hashlib.sha256(
        json.dumps(
            list(tasks),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def coverage_collection_top_batch_handoff_tasks_jsonl(
    tasks: Sequence[Mapping[str, Any]],
) -> str:
    if not tasks:
        return ""
    return (
        "\n".join(json.dumps(task, sort_keys=True) for task in tasks)
        + "\n"
    )


def save_coverage_collection_top_batch_handoff_tasks(
    tasks: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        coverage_collection_top_batch_handoff_tasks_jsonl(tasks),
        encoding="utf-8",
    )
    return output_path


def load_coverage_collection_top_batch_handoff_tasks(
    path: str | Path,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise SpatialQAError(
                f"Coverage collection top-batch handoff line {line_number} must be an object"
            )
        tasks.append(cast(dict[str, Any], payload))
    return tasks


def validate_coverage_collection_top_batch_handoff_tasks(
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    task_ids = [_required_text(task, "task_id") for task in tasks]
    batch_ids = sorted({_required_text(task, "batch_id") for task in tasks})
    related_case_ids: set[str] = set()
    for task in tasks:
        related_case_ids.update(_string_sequence(task.get("related_case_ids")))
    checks = [
        {
            "name": "schema_version",
            "passed": all(
                task.get("schema_version")
                == COVERAGE_COLLECTION_TARGET_TASK_SCHEMA_VERSION
                for task in tasks
            ),
            "expected": COVERAGE_COLLECTION_TARGET_TASK_SCHEMA_VERSION,
            "actual": sorted({str(task.get("schema_version")) for task in tasks}),
        },
        {
            "name": "action",
            "passed": all(
                task.get("action") == "coverage_collection_target_task"
                for task in tasks
            ),
            "expected": "coverage_collection_target_task",
            "actual": sorted({str(task.get("action")) for task in tasks}),
        },
        {
            "name": "task_digests",
            "passed": all(
                task.get("task_digest")
                == coverage_collection_target_task_digest(task)
                for task in tasks
            ),
            "expected": "matching per-task digests",
            "actual": "checked",
        },
        {
            "name": "unique_task_ids",
            "passed": len(set(task_ids)) == len(task_ids),
            "expected": len(task_ids),
            "actual": len(set(task_ids)),
        },
        {
            "name": "metadata_not_prediction_evidence",
            "passed": all(
                _source_policy_valid(task.get("source_use_policy")) for task in tasks
            ),
            "expected": True,
            "actual": [
                task.get("source_use_policy")
                for task in tasks
                if not _source_policy_valid(task.get("source_use_policy"))
            ],
        },
        {
            "name": "batch_execution_plans",
            "passed": all(
                _task_batch_execution_plan_valid(task.get("batch_execution_plan"))
                for task in tasks
            ),
            "expected": True,
            "actual": [
                task.get("batch_execution_plan")
                for task in tasks
                if not _task_batch_execution_plan_valid(
                    task.get("batch_execution_plan")
                )
            ],
        },
        {
            "name": "no_gold_answers_or_evidence",
            "passed": not _contains_forbidden_gold_field(list(tasks)),
            "expected": False,
            "actual": _contains_forbidden_gold_field(list(tasks)),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "checks": checks,
        "digest": coverage_collection_top_batch_handoff_tasks_digest(tasks),
        "task_count": len(tasks),
        "batch_count": len(batch_ids),
        "related_case_count": len(related_case_ids),
    }


def coverage_collection_top_batch_return_report(
    tasks: Sequence[Mapping[str, Any]],
    observations: Sequence[SceneObservation],
    *,
    observation_sequence_path: str | Path | None = None,
    required_evidence_kinds: Sequence[str] = DEFAULT_COVERAGE_ACCEPTANCE_EVIDENCE_KINDS,
) -> dict[str, Any]:
    required = tuple(sorted(set(required_evidence_kinds)))
    task_index = _top_batch_task_index(tasks)
    accepted_by_task: dict[str, dict[str, Any]] = {}
    accepted_location_by_task: dict[str, dict[str, Any]] = {}
    accepted_state_by_task: dict[str, dict[str, Any]] = {}
    for observation in observations:
        for obj in observation.objects:
            for task_id, matched_by in _matched_top_batch_tasks(
                obj.object_id,
                obj.attributes,
                task_index,
            ):
                task = task_index["by_task_id"][task_id]
                evidence_kinds = _object_evidence_kinds(obj.attributes)
                if not _is_accepted_visible_detector_target(
                    obj.visible,
                    obj.attributes,
                    evidence_kinds,
                    required,
                ):
                    continue
                accepted_by_task.setdefault(
                    task_id,
                    {
                        "evidence_kinds": [],
                        "matched_by": [],
                        "object_id": _required_text(task, "object_id"),
                        "observed_object_ids": [],
                        "observation_steps": [],
                        "task_id": task_id,
                    },
                )
                accepted_by_task[task_id]["evidence_kinds"].extend(
                    sorted(evidence_kinds)
                )
                accepted_by_task[task_id]["matched_by"].append(matched_by)
                accepted_by_task[task_id]["observed_object_ids"].append(obj.object_id)
                accepted_by_task[task_id]["observation_steps"].append(observation.step)
                current_location = _current_location_evidence(obj.attributes)
                if current_location is not None:
                    accepted_location_by_task.setdefault(
                        task_id,
                        {
                            **current_location,
                            "object_id": _required_text(task, "object_id"),
                            "task_id": task_id,
                        },
                    )
                if task.get("state_evidence_required") is True and _has_state_evidence(
                    obj.attributes
                ):
                    accepted_state_by_task.setdefault(
                        task_id,
                        {
                            "object_id": _required_text(task, "object_id"),
                            "state_attribute_keys": [],
                            "state_evidence_case_ids": sorted(
                                _string_sequence(task.get("state_evidence_case_ids"))
                            ),
                            "task_id": task_id,
                        },
                    )
                    accepted_state_by_task[task_id]["state_attribute_keys"].extend(
                        _state_attribute_keys(obj.attributes)
                    )
    accepted_target_tasks = [
        {
            **item,
            "evidence_kinds": sorted(set(_string_sequence(item["evidence_kinds"]))),
            "matched_by": sorted(set(_string_sequence(item["matched_by"]))),
            "observed_object_ids": sorted(
                set(_string_sequence(item["observed_object_ids"]))
            ),
            "observation_steps": sorted(set(_int_sequence(item["observation_steps"]))),
        }
        for item in accepted_by_task.values()
    ]
    accepted_location_tasks = list(accepted_location_by_task.values())
    accepted_state_tasks = [
        {
            **item,
            "state_attribute_keys": sorted(
                set(_string_sequence(item["state_attribute_keys"]))
            ),
        }
        for item in accepted_state_by_task.values()
    ]
    accepted_task_ids = set(accepted_by_task)
    accepted_location_task_ids = set(accepted_location_by_task)
    accepted_state_task_ids = set(accepted_state_by_task)
    task_ids = sorted(task_index["by_task_id"])
    location_required_task_ids = [
        task_id
        for task_id in task_ids
        if _task_requires_current_location(task_index["by_task_id"][task_id])
    ]
    state_required_task_ids = [
        task_id
        for task_id in task_ids
        if task_index["by_task_id"][task_id].get("state_evidence_required") is True
    ]
    unaccepted_target_tasks = [
        {
            "object_id": _required_text(task_index["by_task_id"][task_id], "object_id"),
            "reason": "missing_visible_detector_evidence",
            "task_id": task_id,
        }
        for task_id in task_ids
        if task_id not in accepted_task_ids
    ]
    unaccepted_location_tasks = [
        {
            "object_id": _required_text(task_index["by_task_id"][task_id], "object_id"),
            "reason": "missing_current_location_evidence",
            "task_id": task_id,
        }
        for task_id in location_required_task_ids
        if task_id not in accepted_location_task_ids
    ]
    unaccepted_state_tasks = [
        {
            "object_id": _required_text(task_index["by_task_id"][task_id], "object_id"),
            "reason": "missing_state_evidence",
            "state_evidence_case_ids": sorted(
                _string_sequence(
                    task_index["by_task_id"][task_id].get("state_evidence_case_ids")
                )
            ),
            "task_id": task_id,
        }
        for task_id in state_required_task_ids
        if task_id not in accepted_state_task_ids
    ]
    batch_ids = sorted(
        {_required_text(task_index["by_task_id"][task_id], "batch_id") for task_id in task_ids}
    )
    related_case_ids: set[str] = set()
    for task_id in task_ids:
        related_case_ids.update(
            _string_sequence(task_index["by_task_id"][task_id].get("related_case_ids"))
        )
    summary = {
        "accepted_location_task_count": len(accepted_location_tasks),
        "accepted_state_task_count": len(accepted_state_tasks),
        "accepted_target_task_count": len(accepted_target_tasks),
        "batch_count": len(batch_ids),
        "location_evidence_ready": len(accepted_location_tasks)
        == len(location_required_task_ids),
        "location_required_task_count": len(location_required_task_ids),
        "related_case_count": len(related_case_ids),
        "return_ready": (
            len(accepted_target_tasks) == len(task_ids)
            and len(accepted_location_tasks) == len(location_required_task_ids)
            and len(accepted_state_tasks) == len(state_required_task_ids)
        ),
        "state_evidence_ready": len(accepted_state_tasks)
        == len(state_required_task_ids),
        "state_required_task_count": len(state_required_task_ids),
        "target_acceptance_rate": _rate(len(accepted_target_tasks), len(task_ids)),
        "target_evidence_ready": len(accepted_target_tasks) == len(task_ids),
        "task_count": len(task_ids),
        "unaccepted_location_task_count": len(unaccepted_location_tasks),
        "unaccepted_state_task_count": len(unaccepted_state_tasks),
        "unaccepted_target_task_count": len(unaccepted_target_tasks),
    }
    report: dict[str, Any] = {
        "schema_version": COVERAGE_COLLECTION_TOP_BATCH_RETURN_REPORT_SCHEMA_VERSION,
        "action": "coverage_collection_top_batch_return_report",
        "top_batch_handoff_digest": coverage_collection_top_batch_handoff_tasks_digest(
            tasks
        ),
        "observation_sequence_path": (
            str(observation_sequence_path)
            if observation_sequence_path is not None
            else None
        ),
        "observation_sequence_digest": scene_observation_sequence_digest(observations),
        "required_evidence_kinds": list(required),
        "summary": summary,
        "accepted_target_tasks": sorted(
            accepted_target_tasks,
            key=lambda item: str(item["object_id"]),
        ),
        "accepted_location_tasks": sorted(
            accepted_location_tasks,
            key=lambda item: str(item["object_id"]),
        ),
        "accepted_state_tasks": sorted(
            accepted_state_tasks,
            key=lambda item: str(item["object_id"]),
        ),
        "unaccepted_target_tasks": unaccepted_target_tasks,
        "unaccepted_location_tasks": unaccepted_location_tasks,
        "unaccepted_state_tasks": unaccepted_state_tasks,
    }
    report["digest"] = coverage_collection_top_batch_return_report_digest(report)
    return report


def coverage_collection_top_batch_return_report_digest(
    report: Mapping[str, Any],
) -> str:
    payload = {key: value for key, value in report.items() if key != "digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def coverage_collection_top_batch_return_report_json(
    report: Mapping[str, Any],
) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_coverage_collection_top_batch_return_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        coverage_collection_top_batch_return_report_json(report),
        encoding="utf-8",
    )
    return output_path


def load_coverage_collection_top_batch_return_report(
    path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError("Coverage collection top-batch return report must be an object")
    return cast(dict[str, Any], payload)


def validate_coverage_collection_top_batch_return_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    expected_digest = coverage_collection_top_batch_return_report_digest(report)
    summary = report.get("summary")
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == COVERAGE_COLLECTION_TOP_BATCH_RETURN_REPORT_SCHEMA_VERSION,
            "expected": COVERAGE_COLLECTION_TOP_BATCH_RETURN_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "action",
            "passed": report.get("action")
            == "coverage_collection_top_batch_return_report",
            "expected": "coverage_collection_top_batch_return_report",
            "actual": report.get("action"),
        },
        {
            "name": "digest",
            "passed": report.get("digest") == expected_digest,
            "expected": expected_digest,
            "actual": report.get("digest"),
        },
        {
            "name": "summary_counts",
            "passed": _top_batch_return_summary_counts_valid(summary),
            "expected": True,
            "actual": summary,
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "checks": checks,
        "digest": expected_digest,
    }


def validate_coverage_collection_request_bundle(
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    expected_digest = coverage_collection_request_bundle_digest(bundle)
    checks = [
        {
            "name": "schema_version",
            "passed": bundle.get("schema_version")
            == COVERAGE_COLLECTION_REQUEST_BUNDLE_SCHEMA_VERSION,
            "expected": COVERAGE_COLLECTION_REQUEST_BUNDLE_SCHEMA_VERSION,
            "actual": bundle.get("schema_version"),
        },
        {
            "name": "action",
            "passed": bundle.get("action") == "coverage_collection_request_bundle",
            "expected": "coverage_collection_request_bundle",
            "actual": bundle.get("action"),
        },
        {
            "name": "digest",
            "passed": bundle.get("digest") == expected_digest,
            "expected": expected_digest,
            "actual": bundle.get("digest"),
        },
        {
            "name": "metadata_not_prediction_evidence",
            "passed": _source_policy_valid(bundle.get("source_use_policy")),
            "expected": True,
            "actual": bundle.get("source_use_policy"),
        },
        {
            "name": "targets",
            "passed": isinstance(bundle.get("targets"), list),
            "expected": "list",
            "actual": type(bundle.get("targets")).__name__,
        },
        {
            "name": "viewpoint_batch_execution_plans",
            "passed": _viewpoint_batch_execution_plans_valid(
                bundle.get("viewpoint_batches")
            ),
            "expected": True,
            "actual": bundle.get("viewpoint_batches"),
        },
        {
            "name": "no_gold_answers_or_evidence",
            "passed": not _contains_forbidden_gold_field(bundle),
            "expected": False,
            "actual": _contains_forbidden_gold_field(bundle),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "checks": checks,
        "digest": expected_digest,
    }


def coverage_collection_acceptance_report(
    plan: Mapping[str, Any],
    observations: Sequence[SceneObservation],
    *,
    observation_sequence_path: str | Path | None = None,
    required_evidence_kinds: Sequence[str] = DEFAULT_COVERAGE_ACCEPTANCE_EVIDENCE_KINDS,
) -> dict[str, Any]:
    required = tuple(sorted(set(required_evidence_kinds)))
    target_index = _planned_target_index(plan)
    planned_targets = set(target_index["by_id"])
    state_targets = _planned_state_evidence_targets(plan)
    state_targets_by_object_id = _state_targets_by_object_id(state_targets)
    accepted_by_target: dict[str, dict[str, Any]] = {}
    accepted_location_by_target: dict[str, dict[str, Any]] = {}
    accepted_state_by_case: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    for observation in observations:
        for obj in observation.objects:
            match = _planned_target_match(obj.object_id, obj.attributes, target_index)
            if match is None:
                continue
            target_id, matched_by = match
            evidence_kinds = _object_evidence_kinds(obj.attributes)
            if _is_accepted_visible_detector_target(
                obj.visible,
                obj.attributes,
                evidence_kinds,
                required,
            ):
                accepted_by_target.setdefault(
                    target_id,
                    {
                        "evidence_kinds": sorted(evidence_kinds),
                        "matched_by": [],
                        "object_id": target_id,
                        "observation_steps": [],
                        "observed_object_ids": [],
                        "source_kind": _source_kind(obj.attributes),
                        "source_name": _source_name(obj.attributes),
                    },
                )["observation_steps"].append(observation.step)
                accepted_by_target[target_id]["observed_object_ids"].append(obj.object_id)
                accepted_by_target[target_id]["matched_by"].append(matched_by)
                current_location = _current_location_evidence(obj.attributes)
                if current_location is not None:
                    accepted_location_by_target.setdefault(
                        target_id,
                        {
                            **current_location,
                            "matched_by": [],
                            "object_id": target_id,
                            "observation_steps": [],
                            "observed_object_ids": [],
                        },
                    )["observation_steps"].append(observation.step)
                    accepted_location_by_target[target_id][
                        "observed_object_ids"
                    ].append(obj.object_id)
                    accepted_location_by_target[target_id]["matched_by"].append(
                        matched_by
                    )
                if _has_state_evidence(obj.attributes):
                    for state_target in state_targets_by_object_id.get(target_id, ()):
                        case_id = _required_text(state_target, "case_id")
                        accepted_state_by_case.setdefault(
                            case_id,
                            {
                                "case_id": case_id,
                                "matched_by": [],
                                "missing_state_edges": sorted(
                                    _string_sequence(
                                        state_target.get("missing_state_edges")
                                    )
                                ),
                                "missing_state_nodes": sorted(
                                    _string_sequence(
                                        state_target.get("missing_state_nodes")
                                    )
                                ),
                                "object_ids": sorted(
                                    _string_sequence(state_target.get("object_ids"))
                                ),
                                "observation_steps": [],
                                "observed_object_ids": [],
                                "state_attribute_keys": [],
                            },
                        )["observation_steps"].append(observation.step)
                        accepted_state_by_case[case_id]["observed_object_ids"].append(
                            obj.object_id
                        )
                        accepted_state_by_case[case_id]["matched_by"].append(matched_by)
                        accepted_state_by_case[case_id][
                            "state_attribute_keys"
                        ].extend(_state_attribute_keys(obj.attributes))
            else:
                rejected.append(
                    {
                        "object_id": target_id,
                        "reason": "not_visible_detector_evidence",
                        "step": observation.step,
                    }
                )
    accepted_targets = [
        {
            **target,
            "matched_by": sorted(set(_string_sequence(target["matched_by"]))),
            "observation_steps": sorted(set(_int_sequence(target["observation_steps"]))),
            "observed_object_ids": sorted(
                set(_string_sequence(target["observed_object_ids"]))
            ),
        }
        for target in accepted_by_target.values()
    ]
    accepted_location_evidence_targets = [
        {
            **target,
            "matched_by": sorted(set(_string_sequence(target["matched_by"]))),
            "observation_steps": sorted(set(_int_sequence(target["observation_steps"]))),
            "observed_object_ids": sorted(
                set(_string_sequence(target["observed_object_ids"]))
            ),
        }
        for target in accepted_location_by_target.values()
    ]
    accepted_state_evidence_targets = [
        {
            **target,
            "matched_by": sorted(set(_string_sequence(target["matched_by"]))),
            "observation_steps": sorted(set(_int_sequence(target["observation_steps"]))),
            "observed_object_ids": sorted(
                set(_string_sequence(target["observed_object_ids"]))
            ),
            "state_attribute_keys": sorted(
                set(_string_sequence(target["state_attribute_keys"]))
            ),
        }
        for target in accepted_state_by_case.values()
    ]
    accepted_ids = {target["object_id"] for target in accepted_targets}
    accepted_location_ids = {
        target["object_id"] for target in accepted_location_evidence_targets
    }
    accepted_state_case_ids = {
        target["case_id"] for target in accepted_state_evidence_targets
    }
    unaccepted_targets = [
        {"object_id": object_id}
        for object_id in sorted(planned_targets)
        if object_id not in accepted_ids
    ]
    unaccepted_location_evidence_targets = [
        {"object_id": object_id}
        for object_id in sorted(planned_targets)
        if object_id not in accepted_location_ids
    ]
    unaccepted_state_evidence_targets = [
        {
            "case_id": _required_text(target, "case_id"),
            "missing_state_edges": sorted(
                _string_sequence(target.get("missing_state_edges"))
            ),
            "missing_state_nodes": sorted(
                _string_sequence(target.get("missing_state_nodes"))
            ),
            "object_ids": sorted(_string_sequence(target.get("object_ids"))),
        }
        for target in state_targets
        if _required_text(target, "case_id") not in accepted_state_case_ids
    ]
    report: dict[str, Any] = {
        "schema_version": COVERAGE_COLLECTION_ACCEPTANCE_REPORT_SCHEMA_VERSION,
        "action": "coverage_collection_acceptance_report",
        "coverage_collection_plan_digest": _required_text(plan, "digest"),
        "observation_sequence_path": (
            str(observation_sequence_path)
            if observation_sequence_path is not None
            else None
        ),
        "observation_sequence_digest": scene_observation_sequence_digest(observations),
        "required_evidence_kinds": list(required),
        "summary": {
            "accepted_location_evidence_target_count": len(
                accepted_location_evidence_targets
            ),
            "accepted_target_count": len(accepted_targets),
            "accepted_state_evidence_target_count": len(
                accepted_state_evidence_targets
            ),
            "location_evidence_acceptance_rate": _rate(
                len(accepted_location_evidence_targets),
                len(planned_targets),
            ),
            "location_evidence_ready": len(accepted_location_evidence_targets)
            == len(planned_targets),
            "location_evidence_target_count": len(planned_targets),
            "planned_target_count": len(planned_targets),
            "rejected_target_count": len(rejected),
            "state_evidence_acceptance_rate": _rate(
                len(accepted_state_evidence_targets),
                len(state_targets),
            ),
            "state_evidence_ready": len(accepted_state_evidence_targets)
            == len(state_targets),
            "state_evidence_target_count": len(state_targets),
            "target_acceptance_rate": _rate(len(accepted_targets), len(planned_targets)),
            "target_evidence_ready": len(accepted_targets) == len(planned_targets),
            "unaccepted_state_evidence_target_count": len(
                unaccepted_state_evidence_targets
            ),
            "unaccepted_location_evidence_target_count": len(
                unaccepted_location_evidence_targets
            ),
            "unaccepted_target_count": len(unaccepted_targets),
        },
        "accepted_targets": sorted(
            accepted_targets,
            key=lambda item: str(item["object_id"]),
        ),
        "accepted_state_evidence_targets": sorted(
            accepted_state_evidence_targets,
            key=lambda item: str(item["case_id"]),
        ),
        "accepted_location_evidence_targets": sorted(
            accepted_location_evidence_targets,
            key=lambda item: str(item["object_id"]),
        ),
        "rejected_observations": sorted(
            rejected,
            key=lambda item: (str(item["object_id"]), int(item["step"])),
        ),
        "unaccepted_state_evidence_targets": sorted(
            unaccepted_state_evidence_targets,
            key=lambda item: str(item["case_id"]),
        ),
        "unaccepted_location_evidence_targets": unaccepted_location_evidence_targets,
        "unaccepted_targets": unaccepted_targets,
    }
    report["digest"] = coverage_collection_acceptance_report_digest(report)
    return report


def coverage_collection_acceptance_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def coverage_collection_acceptance_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_coverage_collection_acceptance_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        coverage_collection_acceptance_report_json(report),
        encoding="utf-8",
    )
    return output_path


def load_coverage_collection_acceptance_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError("Coverage collection acceptance report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_coverage_collection_acceptance_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    expected_digest = coverage_collection_acceptance_report_digest(report)
    summary = report.get("summary")
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == COVERAGE_COLLECTION_ACCEPTANCE_REPORT_SCHEMA_VERSION,
            "expected": COVERAGE_COLLECTION_ACCEPTANCE_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "action",
            "passed": report.get("action")
            == "coverage_collection_acceptance_report",
            "expected": "coverage_collection_acceptance_report",
            "actual": report.get("action"),
        },
        {
            "name": "digest",
            "passed": report.get("digest") == expected_digest,
            "expected": expected_digest,
            "actual": report.get("digest"),
        },
        {
            "name": "accepted_targets",
            "passed": isinstance(report.get("accepted_targets"), list),
            "expected": "list",
            "actual": type(report.get("accepted_targets")).__name__,
        },
        {
            "name": "summary_counts",
            "passed": _acceptance_summary_counts_valid(summary),
            "expected": True,
            "actual": summary,
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "checks": checks,
        "digest": expected_digest,
    }


def _object_observations(frames: Sequence[EpisodeFrame]) -> dict[str, dict[str, Any]]:
    observations: dict[str, dict[str, Any]] = {}
    for frame in sorted(frames, key=lambda item: (item.episode_id, item.step)):
        metadata = _mapping(frame.metadata, "frame.metadata")
        for item in _sequence(metadata.get("objects")):
            obj = _mapping(item, "metadata object")
            object_id = _required_str(obj, "object_id")
            record = observations.setdefault(
                object_id,
                {
                    "object_id": object_id,
                    "episode_id": frame.episode_id,
                    "scene_id": frame.scene_id,
                    "label": _required_str(obj, "label"),
                    "first_metadata_step": frame.step,
                    "last_metadata_step": frame.step,
                    "metadata_observation_count": 0,
                    "current_visible_observation_count": 0,
                    "pose": _pose(obj.get("pose")),
                    "ai2thor_object_id": _ai2thor_object_id(obj),
                },
            )
            record["last_metadata_step"] = frame.step
            record["metadata_observation_count"] += 1
            if obj.get("visible") is True or object_id in frame.visible_object_ids:
                record["current_visible_observation_count"] += 1
    return observations


def _collection_target(
    object_id: str,
    observation: Mapping[str, Any],
    case_index: Mapping[str, list[str]],
    *,
    viewpoint_hints: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    related_case_ids = sorted(set(case_index.get(object_id, [])))
    return {
        "ai2thor_object_id": observation.get("ai2thor_object_id"),
        "current_visible_observation_count": _int_value(
            observation.get("current_visible_observation_count")
        ),
        "episode_id": _required_text(observation, "episode_id"),
        "first_metadata_step": _int_value(observation.get("first_metadata_step")),
        "label": _required_text(observation, "label"),
        "last_metadata_step": _int_value(observation.get("last_metadata_step")),
        "missing_reasons": ["target_missing"],
        "object_id": object_id,
        "pose": observation.get("pose"),
        "related_case_ids": related_case_ids,
        "scene_id": _required_text(observation, "scene_id"),
        "suggested_action": "collect_visible_rgbd_detection",
        "viewpoint_hints": list(viewpoint_hints),
    }


def _case_index(report: Mapping[str, Any]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for case in _sequence(report.get("cases")):
        payload = _mapping(case, "observability case")
        case_id = _required_str(payload, "case_id")
        for node_id in _string_sequence(payload.get("target_nodes")):
            index.setdefault(node_id, []).append(case_id)
        for node_id in _string_sequence(payload.get("missing_target_nodes")):
            index.setdefault(node_id, []).append(case_id)
    return index


def _relation_evidence_targets(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for case in _sequence(report.get("cases")):
        payload = _mapping(case, "observability case")
        if payload.get("observability_status") != "target_observable_relation_missing":
            continue
        targets.append(
            {
                "case_id": _required_str(payload, "case_id"),
                "missing_required_edge_relations": sorted(
                    set(_string_sequence(payload.get("missing_required_edge_relations")))
                ),
                "missing_required_edges": sorted(
                    _string_sequence(payload.get("missing_required_edges"))
                ),
                "target_nodes": sorted(_string_sequence(payload.get("target_nodes"))),
            }
        )
    return sorted(targets, key=lambda item: str(item["case_id"]))


def _state_evidence_targets(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for case in _sequence(report.get("cases")):
        payload = _mapping(case, "observability case")
        missing_relations = set(
            _string_sequence(payload.get("missing_required_edge_relations"))
        )
        if "STATE_CHANGED" not in missing_relations:
            continue
        missing_state_edges = sorted(
            edge
            for edge in _string_sequence(payload.get("missing_required_edges"))
            if "-STATE_CHANGED-" in edge
        )
        missing_state_nodes = sorted(
            node
            for node in _string_sequence(payload.get("missing_required_nodes"))
            if node.startswith("state:")
        )
        object_ids = sorted(
            {
                edge.split("-STATE_CHANGED-", 1)[0]
                for edge in missing_state_edges
            }
        )
        if not object_ids:
            object_ids = sorted(_string_sequence(payload.get("target_nodes")))
        targets.append(
            {
                "case_id": _required_str(payload, "case_id"),
                "missing_state_edges": missing_state_edges,
                "missing_state_nodes": missing_state_nodes,
                "object_ids": object_ids,
                "required_state_contract": {
                    "evidence_kinds": list(DEFAULT_COVERAGE_ACCEPTANCE_EVIDENCE_KINDS),
                    "state_attributes_field": "attributes.states",
                    "visible": True,
                },
                "target_nodes": sorted(_string_sequence(payload.get("target_nodes"))),
            }
        )
    return sorted(targets, key=lambda item: str(item["case_id"]))


def _request_target(
    target: Mapping[str, Any],
    required_evidence_kinds: Sequence[str],
) -> dict[str, Any]:
    object_id = _required_text(target, "object_id")
    return {
        "ai2thor_object_id": target.get("ai2thor_object_id"),
        "episode_id": _required_text(target, "episode_id"),
        "label": _required_text(target, "label"),
        "object_id": object_id,
        "pose": target.get("pose"),
        "related_case_ids": sorted(_string_sequence(target.get("related_case_ids"))),
        "required_detection_contract": {
            "current_location_fields": [
                "attributes.current_location_id",
                "attributes.current_location_relation",
            ],
            "evidence_kinds": list(required_evidence_kinds),
            "handoff_target_id_fields": [
                f"attributes.{field_name}"
                for field_name in HANDOFF_TARGET_OBJECT_ID_FIELDS
            ],
            "object_id": object_id,
            "source_kind": "detector",
            "supported_current_location_relations": [
                "IN_REGION",
                "IN_ROOM",
                "INSIDE",
                "ON",
            ],
            "visible": True,
        },
        "scene_id": _required_text(target, "scene_id"),
        "suggested_action": _required_text(target, "suggested_action"),
        "viewpoint_hints": [
            dict(_mapping(item, "viewpoint hint"))
            for item in _sequence(target.get("viewpoint_hints"))
        ],
    }


def _viewpoint_batches(targets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    batches: dict[tuple[str, str, int], dict[str, Any]] = {}
    for target in targets:
        hints = _sequence(target.get("viewpoint_hints"))
        if not hints:
            continue
        primary_hint = _mapping(hints[0], "viewpoint hint")
        episode_id = _required_text(primary_hint, "episode_id")
        scene_id = _required_text(primary_hint, "scene_id")
        step = _int_value(primary_hint.get("step"))
        key = (episode_id, scene_id, step)
        batch = batches.setdefault(
            key,
            {
                "agent_pose": dict(_mapping(primary_hint.get("agent_pose"), "agent_pose")),
                "batch_id": f"{episode_id}:{scene_id}:{step:06d}",
                "collection_action": "collect_visible_rgbd_detection",
                "episode_id": episode_id,
                "scene_id": scene_id,
                "step": step,
                "related_case_ids": [],
                "target_ids": [],
                "targets": [],
            },
        )
        object_id = _required_text(target, "object_id")
        related_case_ids = sorted(_string_sequence(target.get("related_case_ids")))
        batch["related_case_ids"].extend(related_case_ids)
        batch["target_ids"].append(object_id)
        batch["targets"].append(
            {
                "distance_to_target": _float_value(
                    primary_hint.get("distance_to_target")
                ),
                "label": _required_text(target, "label"),
                "object_id": object_id,
                "related_case_ids": related_case_ids,
                "suggested_yaw_to_target": _float_value(
                    primary_hint.get("suggested_yaw_to_target")
                ),
            }
        )
    normalized_batches: list[dict[str, Any]] = []
    for batch in batches.values():
        targets_list = sorted(
            batch["targets"],
            key=lambda item: str(_mapping(item, "batch target").get("object_id")),
        )
        target_ids = sorted(str(item["object_id"]) for item in targets_list)
        related_case_ids = sorted(set(_string_sequence(batch.get("related_case_ids"))))
        normalized_batches.append(
            {
                **batch,
                "execution_plan": _viewpoint_batch_execution_plan(
                    batch,
                    targets_list,
                ),
                "related_case_count": len(related_case_ids),
                "related_case_ids": related_case_ids,
                "target_count": len(targets_list),
                "target_ids": target_ids,
                "targets": targets_list,
            }
        )
    sorted_batches = sorted(
        normalized_batches,
        key=lambda item: (
            -_int_value(item.get("related_case_count")),
            -_int_value(item.get("target_count")),
            str(item.get("episode_id")),
            str(item.get("scene_id")),
            _int_value(item.get("step")),
        ),
    )
    seen_case_ids: set[str] = set()
    cumulative_target_count = 0
    ranked_batches: list[dict[str, Any]] = []
    for index, batch in enumerate(sorted_batches, start=1):
        seen_case_ids.update(_string_sequence(batch.get("related_case_ids")))
        cumulative_target_count += _int_value(batch.get("target_count"))
        ranked_batches.append(
            {
                **batch,
                "cumulative_related_case_count": len(seen_case_ids),
                "cumulative_target_count": cumulative_target_count,
                "priority_rank": index,
            }
        )
    return ranked_batches


def _viewpoint_batch_execution_plan(
    batch: Mapping[str, Any],
    targets_list: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    agent_pose = _mapping(batch.get("agent_pose"), "agent_pose")
    target_ids = sorted(
        _required_text(_mapping(target, "viewpoint batch target"), "object_id")
        for target in targets_list
    )
    target_yaws = sorted(
        {
            _float_value(
                _mapping(target, "viewpoint batch target").get(
                    "suggested_yaw_to_target"
                )
            )
            for target in targets_list
        }
    )
    return {
        "ai2thor_actions": [
            {
                "action": "TeleportFull",
                "position": {
                    "x": _float_value(agent_pose.get("x")),
                    "y": _float_value(agent_pose.get("y")),
                    "z": _float_value(agent_pose.get("z")),
                },
                "rotation": {
                    "x": 0.0,
                    "y": _float_value(agent_pose.get("yaw")),
                    "z": 0.0,
                },
            },
            {
                "action": "RotateToTargetYaw",
                "target_yaws": target_yaws,
            },
            {
                "action": "CaptureVisibleRgbdDetection",
                "required_evidence_kinds": list(
                    DEFAULT_COVERAGE_ACCEPTANCE_EVIDENCE_KINDS
                ),
                "target_ids": target_ids,
            },
        ],
        "acceptance_command_hint": (
            "python scripts/plan_coverage_collection.py "
            "--top-batch-return-tasks <handoff-jsonl> "
            "--detector-jsonl <detector-jsonl> "
            "--top-batch-return-report <report-json>"
        ),
        "not_predicted_graph_evidence": True,
    }


def _priority_batches_to_reach_case_count(
    batches: Sequence[Mapping[str, Any]],
    target_case_count: int,
) -> int | None:
    if target_case_count <= 0:
        return 0
    for batch in batches:
        cumulative = _int_value(batch.get("cumulative_related_case_count"))
        if cumulative >= target_case_count:
            return _int_value(batch.get("priority_rank"))
    return None


def _viewpoint_hints(
    frames: Sequence[EpisodeFrame],
    observation: Mapping[str, Any],
    *,
    max_viewpoint_hints: int,
) -> list[dict[str, Any]]:
    if max_viewpoint_hints <= 0:
        return []
    target_pose = observation.get("pose")
    if not isinstance(target_pose, Mapping):
        return []
    target_x = _float_value(target_pose.get("x"))
    target_z = _float_value(target_pose.get("z"))
    episode_id = _required_text(observation, "episode_id")
    scene_id = _required_text(observation, "scene_id")
    hints: list[dict[str, Any]] = []
    for frame in frames:
        if frame.episode_id != episode_id or frame.scene_id != scene_id:
            continue
        dx = target_x - frame.agent_pose.x
        dz = target_z - frame.agent_pose.z
        distance = math.sqrt(dx * dx + dz * dz)
        yaw = (math.degrees(math.atan2(dx, dz)) + 360.0) % 360.0
        hints.append(
            {
                "agent_pose": _pose_dict(frame.agent_pose),
                "distance_to_target": _stable_float(distance),
                "episode_id": frame.episode_id,
                "scene_id": frame.scene_id,
                "step": frame.step,
                "suggested_yaw_to_target": _stable_float(yaw),
            }
        )
    return sorted(
        hints,
        key=lambda item: (
            float(item["distance_to_target"]),
            int(item["step"]),
        ),
    )[:max_viewpoint_hints]


def _contains_forbidden_gold_field(value: object) -> bool:
    forbidden = {
        "answer",
        "gold_answer",
        "gold_evidence",
        "gold_evidence_edges",
        "gold_evidence_nodes",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key in forbidden:
                return True
            if _contains_forbidden_gold_field(item):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, str):
        return any(_contains_forbidden_gold_field(item) for item in value)
    return False


def _source_policy_valid(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        value.get("episode_metadata_used_for") == "collection_planning_only"
        and value.get("not_predicted_graph_evidence") is True
        and value.get("requires_visible_detector_evidence_before_claim") is True
    )


def _viewpoint_batch_execution_plans_valid(value: object) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return False
    for item in value:
        if not isinstance(item, Mapping):
            return False
        plan = item.get("execution_plan")
        if not isinstance(plan, Mapping):
            return False
        actions = plan.get("ai2thor_actions")
        if not isinstance(actions, Sequence) or isinstance(actions, str):
            return False
        action_names = [
            action.get("action")
            for action in actions
            if isinstance(action, Mapping)
        ]
        if action_names != [
            "TeleportFull",
            "RotateToTargetYaw",
            "CaptureVisibleRgbdDetection",
        ]:
            return False
        if plan.get("not_predicted_graph_evidence") is not True:
            return False
        if not isinstance(plan.get("acceptance_command_hint"), str):
            return False
    return True


def _task_batch_execution_plan_valid(value: object) -> bool:
    return _viewpoint_batch_execution_plans_valid(
        (
            {
                "execution_plan": value,
            },
        )
    )


def _planned_target_ids(plan: Mapping[str, Any]) -> set[str]:
    ids: set[str] = set()
    for item in _sequence(plan.get("collection_targets")):
        target = _mapping(item, "collection target")
        ids.add(_required_str(target, "object_id"))
    return ids


def _planned_target_index(plan: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    by_id: dict[str, str] = {}
    by_ai2thor_object_id: dict[str, str] = {}
    by_handoff_target_object_id: dict[str, str] = {}
    for item in _sequence(plan.get("collection_targets")):
        target = _mapping(item, "collection target")
        object_id = _required_str(target, "object_id")
        by_id[object_id] = object_id
        by_handoff_target_object_id[object_id] = object_id
        ai2thor_object_id = target.get("ai2thor_object_id")
        if isinstance(ai2thor_object_id, str) and ai2thor_object_id:
            by_ai2thor_object_id[ai2thor_object_id] = object_id
    return {
        "by_ai2thor_object_id": by_ai2thor_object_id,
        "by_handoff_target_object_id": by_handoff_target_object_id,
        "by_id": by_id,
    }


def _planned_state_evidence_targets(plan: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        _mapping(item, "state evidence target")
        for item in _sequence(plan.get("state_evidence_targets"))
    )


def _state_targets_by_object_id(
    state_targets: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for target in state_targets:
        for object_id in _string_sequence(target.get("object_ids")):
            grouped.setdefault(object_id, []).append(target)
    return {
        object_id: tuple(sorted(targets, key=lambda item: _required_text(item, "case_id")))
        for object_id, targets in grouped.items()
    }


def _state_case_ids_by_object_id(
    state_targets: Sequence[Any],
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, set[str]] = {}
    for item in state_targets:
        target = _mapping(item, "state evidence target")
        case_id = _required_text(target, "case_id")
        for object_id in _string_sequence(target.get("object_ids")):
            grouped.setdefault(object_id, set()).add(case_id)
    return {
        object_id: tuple(sorted(case_ids))
        for object_id, case_ids in grouped.items()
    }


def _top_batch_task_index(
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_task_id: dict[str, Mapping[str, Any]] = {}
    by_object_id: dict[str, list[str]] = {}
    by_ai2thor_object_id: dict[str, list[str]] = {}
    by_handoff_target_object_id: dict[str, list[str]] = {}
    for task in tasks:
        task_id = _required_text(task, "task_id")
        object_id = _required_text(task, "object_id")
        by_task_id[task_id] = task
        by_object_id.setdefault(object_id, []).append(task_id)
        by_handoff_target_object_id.setdefault(object_id, []).append(task_id)
        ai2thor_object_id = task.get("ai2thor_object_id")
        if isinstance(ai2thor_object_id, str) and ai2thor_object_id:
            by_ai2thor_object_id.setdefault(ai2thor_object_id, []).append(task_id)
    return {
        "by_ai2thor_object_id": {
            key: sorted(value) for key, value in by_ai2thor_object_id.items()
        },
        "by_handoff_target_object_id": {
            key: sorted(value) for key, value in by_handoff_target_object_id.items()
        },
        "by_object_id": {key: sorted(value) for key, value in by_object_id.items()},
        "by_task_id": dict(by_task_id),
    }


def _matched_top_batch_tasks(
    object_id: str,
    attributes: Mapping[str, Any],
    task_index: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, str], ...]:
    matches: list[tuple[str, str]] = []
    by_object_id = task_index.get("by_object_id", {})
    object_task_ids = by_object_id.get(object_id)
    if isinstance(object_task_ids, Sequence) and not isinstance(object_task_ids, str):
        matches.extend((str(task_id), "object_id") for task_id in object_task_ids)
    ai2thor_object_id = attributes.get("ai2thor_object_id")
    by_ai2thor_object_id = task_index.get("by_ai2thor_object_id", {})
    if isinstance(ai2thor_object_id, str) and ai2thor_object_id:
        alias_task_ids = by_ai2thor_object_id.get(ai2thor_object_id)
        if isinstance(alias_task_ids, Sequence) and not isinstance(alias_task_ids, str):
            matches.extend(
                (str(task_id), "ai2thor_object_id") for task_id in alias_task_ids
            )
    by_handoff_target_object_id = task_index.get("by_handoff_target_object_id", {})
    for field_name, alias_object_id in _handoff_target_object_ids(attributes):
        alias_task_ids = by_handoff_target_object_id.get(alias_object_id)
        if isinstance(alias_task_ids, Sequence) and not isinstance(alias_task_ids, str):
            matches.extend((str(task_id), field_name) for task_id in alias_task_ids)
    deduped: dict[str, str] = {}
    for task_id, matched_by in matches:
        deduped.setdefault(task_id, matched_by)
    return tuple(sorted(deduped.items(), key=lambda item: item[0]))


def _task_requires_current_location(task: Mapping[str, Any]) -> bool:
    contract = task.get("required_detection_contract")
    if not isinstance(contract, Mapping):
        return False
    return bool(_string_sequence(contract.get("current_location_fields")))


def _planned_target_match(
    object_id: str,
    attributes: Mapping[str, Any],
    target_index: Mapping[str, Mapping[str, str]],
) -> tuple[str, str] | None:
    by_id = target_index.get("by_id", {})
    if object_id in by_id:
        return by_id[object_id], "object_id"
    ai2thor_object_id = attributes.get("ai2thor_object_id")
    by_ai2thor_object_id = target_index.get("by_ai2thor_object_id", {})
    if (
        isinstance(ai2thor_object_id, str)
        and ai2thor_object_id
        and ai2thor_object_id in by_ai2thor_object_id
    ):
        return by_ai2thor_object_id[ai2thor_object_id], "ai2thor_object_id"
    by_handoff_target_object_id = target_index.get("by_handoff_target_object_id", {})
    for field_name, alias_object_id in _handoff_target_object_ids(attributes):
        if alias_object_id in by_handoff_target_object_id:
            return by_handoff_target_object_id[alias_object_id], field_name
    return None


def _handoff_target_object_ids(
    attributes: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    aliases: list[tuple[str, str]] = []
    for field_name in HANDOFF_TARGET_OBJECT_ID_FIELDS:
        value = attributes.get(field_name)
        if isinstance(value, str) and value:
            aliases.append((field_name, value))
    return tuple(aliases)


def _is_accepted_visible_detector_target(
    visible: bool,
    attributes: Mapping[str, Any],
    evidence_kinds: set[str],
    required_evidence_kinds: Sequence[str],
) -> bool:
    if visible is not True:
        return False
    if attributes.get("coverage_source") == "episode_metadata":
        return False
    if attributes.get("source_kind") == "ai2thor_metadata_coverage":
        return False
    if attributes.get("source_kind") != "detector":
        return False
    return set(required_evidence_kinds).issubset(evidence_kinds)


def _object_evidence_kinds(attributes: Mapping[str, Any]) -> set[str]:
    evidence: set[str] = set()
    raw_evidence = attributes.get("evidence_kinds")
    if isinstance(raw_evidence, Sequence) and not isinstance(raw_evidence, str):
        evidence.update(item for item in raw_evidence if isinstance(item, str))
    if isinstance(attributes.get("rgb_path"), str):
        evidence.add("rgb")
    if isinstance(attributes.get("depth_path"), str):
        evidence.add("depth")
    if any(
        isinstance(attributes.get(field), str)
        for field in ("detector", "source", "source_name")
    ):
        evidence.add("detector")
    return evidence


def _source_kind(attributes: Mapping[str, Any]) -> str | None:
    value = attributes.get("source_kind")
    return value if isinstance(value, str) and value else None


def _source_name(attributes: Mapping[str, Any]) -> str | None:
    for field in ("source_name", "detector", "source"):
        value = attributes.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _current_location_evidence(attributes: Mapping[str, Any]) -> dict[str, Any] | None:
    location_id = attributes.get("current_location_id")
    relation = attributes.get("current_location_relation")
    if not isinstance(location_id, str) or not location_id:
        return None
    if not isinstance(relation, str) or not relation:
        return None
    normalized_relation = relation.upper()
    if normalized_relation not in CURRENT_LOCATION_RELATIONS:
        return None
    return {
        "current_location_id": location_id,
        "current_location_relation": normalized_relation,
    }


def _has_state_evidence(attributes: Mapping[str, Any]) -> bool:
    states = attributes.get("states")
    return isinstance(states, Mapping) and bool(states)


def _state_attribute_keys(attributes: Mapping[str, Any]) -> tuple[str, ...]:
    states = attributes.get("states")
    if not isinstance(states, Mapping):
        return ()
    return tuple(sorted(key for key in states if isinstance(key, str)))


def _acceptance_summary_counts_valid(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    accepted = value.get("accepted_target_count")
    planned = value.get("planned_target_count")
    unaccepted = value.get("unaccepted_target_count")
    accepted_state = value.get("accepted_state_evidence_target_count")
    planned_state = value.get("state_evidence_target_count")
    unaccepted_state = value.get("unaccepted_state_evidence_target_count")
    accepted_location = value.get("accepted_location_evidence_target_count")
    planned_location = value.get("location_evidence_target_count")
    unaccepted_location = value.get("unaccepted_location_evidence_target_count")
    if (
        not isinstance(accepted, int)
        or isinstance(accepted, bool)
        or not isinstance(planned, int)
        or isinstance(planned, bool)
        or not isinstance(unaccepted, int)
        or isinstance(unaccepted, bool)
        or not isinstance(accepted_state, int)
        or isinstance(accepted_state, bool)
        or not isinstance(planned_state, int)
        or isinstance(planned_state, bool)
        or not isinstance(unaccepted_state, int)
        or isinstance(unaccepted_state, bool)
        or not isinstance(accepted_location, int)
        or isinstance(accepted_location, bool)
        or not isinstance(planned_location, int)
        or isinstance(planned_location, bool)
        or not isinstance(unaccepted_location, int)
        or isinstance(unaccepted_location, bool)
    ):
        return False
    return (
        accepted + unaccepted == planned
        and accepted_state + unaccepted_state == planned_state
        and accepted_location + unaccepted_location == planned_location
    )


def _top_batch_return_summary_counts_valid(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    keys = (
        "accepted_target_task_count",
        "task_count",
        "unaccepted_target_task_count",
        "accepted_state_task_count",
        "state_required_task_count",
        "unaccepted_state_task_count",
        "accepted_location_task_count",
        "location_required_task_count",
        "unaccepted_location_task_count",
    )
    counts: dict[str, int] = {}
    for key in keys:
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int):
            return False
        counts[key] = item
    return (
        counts["accepted_target_task_count"]
        + counts["unaccepted_target_task_count"]
        == counts["task_count"]
        and counts["accepted_state_task_count"]
        + counts["unaccepted_state_task_count"]
        == counts["state_required_task_count"]
        and counts["accepted_location_task_count"]
        + counts["unaccepted_location_task_count"]
        == counts["location_required_task_count"]
    )


def _ai2thor_object_id(obj: Mapping[str, Any]) -> str | None:
    attributes = obj.get("attributes")
    if not isinstance(attributes, Mapping):
        return None
    value = attributes.get("ai2thor_object_id")
    return value if isinstance(value, str) and value else None


def _pose(value: object) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        "x": _float_value(value.get("x")),
        "y": _float_value(value.get("y")),
        "z": _float_value(value.get("z")),
        "yaw": _float_value(value.get("yaw")),
    }


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"{field_name} must be an object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object) -> tuple[Any, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpatialQAError("Expected a sequence")
    return tuple(value)


def _string_sequence(value: object) -> tuple[str, ...]:
    items: list[str] = []
    for item in _sequence(value):
        if not isinstance(item, str):
            raise SpatialQAError("Expected a string sequence")
        items.append(item)
    return tuple(items)


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Required string field missing: {key}")
    return value


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    return _required_str(payload, key)


def _int_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpatialQAError("Expected integer value")
    return value


def _int_sequence(value: object) -> tuple[int, ...]:
    items: list[int] = []
    for item in _sequence(value):
        items.append(_int_value(item))
    return tuple(items)


def _float_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SpatialQAError("Expected numeric value")
    return float(value)


def _pose_dict(pose: Any) -> dict[str, float]:
    return {
        "x": _stable_float(pose.x),
        "y": _stable_float(pose.y),
        "z": _stable_float(pose.z),
        "yaw": _stable_float(pose.yaw),
    }


def _stable_float(value: float) -> float:
    stable = round(float(value), 6)
    if stable == 0.0:
        return 0.0
    return stable


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total, 6)
