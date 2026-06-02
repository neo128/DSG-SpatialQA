from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.scene_io import graph_json_digest
from dsg_spatialqa_lab.schema import SpatialQAError


ACTIVE_EQA_TASK_SCHEMA_VERSION = "dsg-spatialqa-lab.active-eqa-task.v1"


@dataclass(frozen=True)
class ActiveEQATask:
    id: str
    scene_id: str
    episode_id: str
    initial_step: int
    question: Mapping[str, Any]
    gold_answer: Mapping[str, Any]
    success_conditions: Mapping[str, Any] = field(default_factory=dict)
    max_actions: int = 0
    required_evidence: Mapping[str, Sequence[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.id, "id")
        _validate_non_empty_str(self.scene_id, "scene_id")
        _validate_non_empty_str(self.episode_id, "episode_id")
        if isinstance(self.initial_step, bool) or self.initial_step < 0:
            raise SpatialQAError("Active EQA task initial_step must be a non-negative integer")
        if isinstance(self.max_actions, bool) or self.max_actions < 0:
            raise SpatialQAError("Active EQA task max_actions must be a non-negative integer")
        object.__setattr__(self, "question", _json_mapping(self.question))
        object.__setattr__(self, "gold_answer", _json_mapping(self.gold_answer))
        object.__setattr__(
            self,
            "success_conditions",
            _json_mapping(self.success_conditions),
        )
        object.__setattr__(
            self,
            "required_evidence",
            _evidence_mapping(self.required_evidence),
        )


@dataclass(frozen=True)
class ActiveAction:
    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.name, "name")
        object.__setattr__(self, "parameters", _json_mapping(self.parameters))


@dataclass(frozen=True)
class ActiveObservation:
    step: int
    graph_digest: str
    action_count: int


class MockActiveEnvironment:
    def __init__(self, graphs_by_step: Mapping[int, DynamicSceneGraph]) -> None:
        if not graphs_by_step:
            raise SpatialQAError("Mock active environment requires at least one graph")
        normalized: dict[int, DynamicSceneGraph] = {}
        for step, graph in graphs_by_step.items():
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise SpatialQAError("Mock active environment step must be a non-negative integer")
            normalized[step] = graph
        self._graphs_by_step = dict(sorted(normalized.items()))
        self._task: ActiveEQATask | None = None
        self._current_step = min(self._graphs_by_step)
        self._action_count = 0

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def action_count(self) -> int:
        return self._action_count

    def reset(self, task: ActiveEQATask) -> ActiveObservation:
        self._task = task
        self._current_step = task.initial_step
        self._action_count = 0
        return self.observe()

    def observe(self) -> ActiveObservation:
        return ActiveObservation(
            step=self._current_step,
            graph_digest=graph_json_digest(self.current_graph()),
            action_count=self._action_count,
        )

    def step(self, action: ActiveAction) -> ActiveObservation:
        if self._task is None:
            raise SpatialQAError("Mock active environment must be reset before step")
        if self._action_count >= self._task.max_actions:
            raise SpatialQAError("max_actions_exceeded")
        self._action_count += 1
        self._current_step = self._next_step()
        return self.observe()

    def current_graph(self) -> DynamicSceneGraph:
        if self._current_step in self._graphs_by_step:
            return self._graphs_by_step[self._current_step]
        prior_steps = [step for step in self._graphs_by_step if step <= self._current_step]
        if prior_steps:
            return self._graphs_by_step[max(prior_steps)]
        return self._graphs_by_step[min(self._graphs_by_step)]

    def done(self) -> bool:
        if self._task is None:
            return False
        return self._action_count >= self._task.max_actions

    def _next_step(self) -> int:
        future_steps = [step for step in self._graphs_by_step if step > self._current_step]
        if future_steps:
            return min(future_steps)
        return self._current_step + 1


def active_eqa_task_to_dict(task: ActiveEQATask) -> dict[str, Any]:
    return {
        "schema_version": ACTIVE_EQA_TASK_SCHEMA_VERSION,
        "id": task.id,
        "scene_id": task.scene_id,
        "episode_id": task.episode_id,
        "initial_step": task.initial_step,
        "question": _json_mapping(task.question),
        "gold_answer": _json_mapping(task.gold_answer),
        "success_conditions": _json_mapping(task.success_conditions),
        "max_actions": task.max_actions,
        "required_evidence": {
            key: list(values) for key, values in sorted(task.required_evidence.items())
        },
    }


def active_eqa_task_from_dict(payload: Mapping[str, Any]) -> ActiveEQATask:
    schema_version = _required_str(payload, "schema_version")
    if schema_version != ACTIVE_EQA_TASK_SCHEMA_VERSION:
        raise SpatialQAError(f"Unsupported Active EQA task schema version: {schema_version}")
    return ActiveEQATask(
        id=_required_str(payload, "id"),
        scene_id=_required_str(payload, "scene_id"),
        episode_id=_required_str(payload, "episode_id"),
        initial_step=_required_int(payload, "initial_step"),
        question=_required_mapping(payload, "question"),
        gold_answer=_required_mapping(payload, "gold_answer"),
        success_conditions=_required_mapping(payload, "success_conditions"),
        max_actions=_required_int(payload, "max_actions"),
        required_evidence=_required_evidence(payload, "required_evidence"),
    )


def active_eqa_tasks_jsonl(tasks: Sequence[ActiveEQATask]) -> str:
    return "".join(
        json.dumps(active_eqa_task_to_dict(task), separators=(",", ":"), sort_keys=True) + "\n"
        for task in tasks
    )


def active_eqa_tasks_from_jsonl(payload: str) -> list[ActiveEQATask]:
    tasks: list[ActiveEQATask] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if line == "":
            continue
        item = json.loads(line)
        if not isinstance(item, Mapping):
            raise SpatialQAError(f"Active EQA task line {line_number} must be an object")
        tasks.append(active_eqa_task_from_dict(cast(Mapping[str, Any], item)))
    return tasks


def active_eqa_tasks_digest(tasks: Sequence[ActiveEQATask]) -> str:
    return hashlib.sha256(active_eqa_tasks_jsonl(tasks).encode("utf-8")).hexdigest()


def save_active_eqa_tasks(tasks: Sequence[ActiveEQATask], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(active_eqa_tasks_jsonl(tasks), encoding="utf-8")
    return output_path


def load_active_eqa_tasks(path: str | Path) -> list[ActiveEQATask]:
    return active_eqa_tasks_from_jsonl(Path(path).read_text(encoding="utf-8"))


def validate_active_eqa_tasks(tasks: Sequence[ActiveEQATask]) -> dict[str, Any]:
    ids = [task.id for task in tasks]
    duplicate_ids = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
    max_action_violations = [task.id for task in tasks if task.max_actions < 0]
    checks = [
        {
            "name": "task_count",
            "passed": len(tasks) > 0,
            "expected": "one or more tasks",
            "actual": len(tasks),
        },
        {
            "name": "duplicate_ids",
            "passed": not duplicate_ids,
            "expected": [],
            "actual": duplicate_ids,
        },
        {
            "name": "max_actions",
            "passed": not max_action_violations,
            "expected": [],
            "actual": max_action_violations,
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": ACTIVE_EQA_TASK_SCHEMA_VERSION,
        "task_count": len(tasks),
        "digest": active_eqa_tasks_digest(tasks),
        "checks": checks,
    }


def _required_evidence(
    payload: Mapping[str, Any],
    key: str,
) -> dict[str, tuple[str, ...]]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"Active EQA task field must be an object: {key}")
    return _evidence_mapping(cast(Mapping[str, Sequence[str]], value))


def _evidence_mapping(value: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    evidence: dict[str, tuple[str, ...]] = {}
    for key, items in sorted(value.items(), key=lambda pair: str(pair[0])):
        if not isinstance(key, str):
            raise SpatialQAError("Active EQA task evidence keys must be strings")
        if isinstance(items, str) or not isinstance(items, Sequence):
            raise SpatialQAError(f"Active EQA task evidence field must be a string sequence: {key}")
        normalized: list[str] = []
        for item in items:
            if not isinstance(item, str):
                raise SpatialQAError(
                    f"Active EQA task evidence field must be a string sequence: {key}"
                )
            normalized.append(item)
        evidence[key] = tuple(normalized)
    return evidence


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Active EQA task field must be a non-empty string: {key}")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpatialQAError(f"Active EQA task field must be an integer: {key}")
    return value


def _required_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"Active EQA task field must be an object: {key}")
    return _json_mapping(cast(Mapping[str, Any], value))


def _validate_non_empty_str(value: object, key: str) -> None:
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Active EQA task field must be a non-empty string: {key}")


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], _json_value(value))


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_json_value(item) for item in value]
    return value
