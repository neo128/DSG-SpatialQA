from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.graph_tool import GraphTool
from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.qa import SpatialQAEngine
from dsg_spatialqa_lab.scene_io import graph_json_digest
from dsg_spatialqa_lab.schema import Edge, ObjectState, SpatialQAError


QA_CASE_SCHEMA_VERSION = "dsg-spatialqa-lab.qa-case.v1"
QA_DATASET_SUMMARY_SCHEMA_VERSION = "dsg-spatialqa-lab.qa-dataset-summary.v1"
DEFAULT_NEAREST_MARGIN = 0.1
STRUCTURAL_OBJECT_LABELS = frozenset(
    {
        "cabinet",
        "counter",
        "drawer",
        "fridge",
        "oven",
        "refrigerator",
        "shelf",
        "sink",
        "stove",
        "table",
    }
)
QUESTION_RELATION_EXCLUDES = frozenset(
    {
        "ACTION_CAUSED",
        "IN_REGION",
        "IN_ROOM",
        "MOVED_FROM",
        "MOVED_TO",
        "STATE_CHANGED",
    }
)


@dataclass
class QACase:
    id: str
    scene_id: str
    episode_id: str
    graph_digest: str
    step: int
    question: dict[str, Any]
    question_type: str
    answer: dict[str, Any]
    answer_type: str
    choices: tuple[str, ...] = field(default_factory=tuple)
    reference_frame: str | None = None
    required_nodes: tuple[str, ...] = field(default_factory=tuple)
    required_edges: tuple[str, ...] = field(default_factory=tuple)
    difficulty: str = "easy"
    tags: tuple[str, ...] = field(default_factory=tuple)


def generate_qa_cases(
    graph: DynamicSceneGraph,
    *,
    scene_id: str,
    episode_id: str,
    max_cases: int | None = None,
    tags: Sequence[str] = (),
    nearest_margin: float = DEFAULT_NEAREST_MARGIN,
) -> list[QACase]:
    _validate_non_empty_str(scene_id, "scene_id")
    _validate_non_empty_str(episode_id, "episode_id")
    if max_cases is not None and max_cases <= 0:
        raise SpatialQAError("max_cases must be positive")
    if nearest_margin < 0.0:
        raise SpatialQAError("nearest_margin must be non-negative")

    tool = GraphTool(graph)
    qa = SpatialQAEngine(tool)
    digest = graph_json_digest(graph)
    extra_tags = _unique_strings(tags)
    cases: list[QACase] = []

    def is_full() -> bool:
        return max_cases is not None and len(cases) >= max_cases

    def add_case(
        question: Mapping[str, Any],
        *,
        question_type: str,
        slug: str,
        step: int | None = None,
        choices: Sequence[str] = (),
        reference_frame: str | None = None,
        answer_type: str | None = None,
        difficulty: str = "easy",
    ) -> None:
        if is_full():
            return
        response = qa.answer(question)
        if response.error is not None:
            raise SpatialQAError(response.error)
        index = len(cases) + 1
        case_tags = _unique_strings(("generated", *extra_tags, "qa", question_type))
        cases.append(
            QACase(
                id=f"{episode_id}:{scene_id}:{index:04d}:{question_type}:{slug}",
                scene_id=scene_id,
                episode_id=episode_id,
                graph_digest=digest,
                step=step if step is not None else _max_graph_step(graph),
                question=_json_mapping(question),
                question_type=question_type,
                answer=_json_mapping(response.answer),
                answer_type=answer_type or question_type,
                choices=tuple(choices),
                reference_frame=reference_frame,
                required_nodes=tuple(response.evidence_nodes),
                required_edges=tuple(response.evidence_edges),
                difficulty=difficulty,
                tags=case_tags,
            )
        )

    for state in tool.find_objects():
        add_case(
            {"type": "object_location", "object_id": state.object_id},
            question_type="object_location",
            slug=state.object_id,
            step=state.step,
        )
    if is_full():
        return cases

    if _has_place_nodes(graph):
        for state in _room_question_objects(tool):
            add_case(
                {"type": "object_room", "object_id": state.object_id},
                question_type="object_room",
                slug=state.object_id,
                step=state.step,
            )
            if is_full():
                return cases

    for edge in _question_relation_edges(graph):
        add_case(
            {
                "type": "relative_relation",
                "src": edge.src,
                "relation": edge.relation,
                "dst": edge.dst,
                "reference_frame": edge.reference_frame,
            },
            question_type="relative_relation",
            slug=f"{edge.src}_{edge.relation}_{edge.dst}",
            step=edge.step,
            reference_frame=edge.reference_frame,
            answer_type="boolean",
        )
        if is_full():
            return cases

    for state in tool.find_objects():
        choices = tuple(
            object_id for object_id in sorted(graph.object_states) if object_id != state.object_id
        )
        if len(choices) < 2:
            continue
        distances = tool.nearest_distances(state.object_id, candidates=choices)
        if len(distances) < 2:
            continue
        margin = float(distances[1]["distance"]) - float(distances[0]["distance"])
        if margin < nearest_margin:
            continue
        add_case(
            {"type": "nearest_object", "src": state.object_id, "candidates": list(choices)},
            question_type="nearest_object",
            slug=state.object_id,
            step=state.step,
            choices=choices,
        )
        break
    if is_full():
        return cases

    timeline_edge = next(iter(_question_relation_edges(graph)), None)
    if timeline_edge is not None:
        add_case(
            {
                "type": "relation_timeline",
                "src": timeline_edge.src,
                "relation": timeline_edge.relation,
                "dst": timeline_edge.dst,
                "reference_frame": timeline_edge.reference_frame,
            },
            question_type="relation_timeline",
            slug=f"{timeline_edge.src}_{timeline_edge.relation}_{timeline_edge.dst}",
            step=timeline_edge.step,
            reference_frame=timeline_edge.reference_frame,
        )
    if is_full():
        return cases

    add_case(
        {"type": "reobserve_targets"},
        question_type="reobserve_targets",
        slug="all",
        step=_max_graph_step(graph),
    )
    if is_full():
        return cases

    action_state = next((state for state in tool.find_objects() if state.visible), None)
    if action_state is not None:
        preconditions: list[dict[str, Any]] = []
        if action_state.last_seen_step is not None:
            preconditions.append({"type": "last_seen_step", "value": action_state.last_seen_step})
        add_case(
            {
                "type": "next_action_validity",
                "action": {
                    "skill": "pick",
                    "target_object": action_state.object_id,
                    "target_pose": action_state.pose.to_dict(),
                    "preconditions": preconditions,
                },
            },
            question_type="next_action_validity",
            slug=action_state.object_id,
            step=action_state.step,
        )
    if is_full():
        return cases

    steps = _graph_steps(graph)
    if steps and min(steps) < max(steps):
        add_case(
            {"type": "scene_delta", "from_step": min(steps), "to_step": max(steps)},
            question_type="scene_delta",
            slug=f"{min(steps)}_{max(steps)}",
            step=max(steps),
        )

    return cases


def qa_case_to_dict(case: QACase) -> dict[str, Any]:
    return {
        "schema_version": QA_CASE_SCHEMA_VERSION,
        "id": case.id,
        "scene_id": case.scene_id,
        "episode_id": case.episode_id,
        "graph_digest": case.graph_digest,
        "step": case.step,
        "question": _json_mapping(case.question),
        "question_type": case.question_type,
        "answer": _json_mapping(case.answer),
        "answer_type": case.answer_type,
        "choices": list(case.choices),
        "reference_frame": case.reference_frame,
        "required_nodes": list(case.required_nodes),
        "required_edges": list(case.required_edges),
        "difficulty": case.difficulty,
        "tags": list(case.tags),
    }


def qa_case_from_dict(payload: Mapping[str, Any]) -> QACase:
    schema_version = _required_str(payload, "schema_version")
    if schema_version != QA_CASE_SCHEMA_VERSION:
        raise SpatialQAError(f"Unsupported QA case schema version: {schema_version}")
    return QACase(
        id=_required_str(payload, "id"),
        scene_id=_required_str(payload, "scene_id"),
        episode_id=_required_str(payload, "episode_id"),
        graph_digest=_required_str(payload, "graph_digest"),
        step=_required_int(payload, "step"),
        question=_required_mapping(payload, "question"),
        question_type=_required_str(payload, "question_type"),
        answer=_required_mapping(payload, "answer"),
        answer_type=_required_str(payload, "answer_type"),
        choices=_string_tuple(payload, "choices"),
        reference_frame=_optional_str(payload, "reference_frame"),
        required_nodes=_string_tuple(payload, "required_nodes"),
        required_edges=_string_tuple(payload, "required_edges"),
        difficulty=_required_str(payload, "difficulty"),
        tags=_string_tuple(payload, "tags"),
    )


def qa_dataset_jsonl(cases: Sequence[QACase]) -> str:
    return "".join(
        json.dumps(qa_case_to_dict(case), separators=(",", ":"), sort_keys=True) + "\n"
        for case in cases
    )


def qa_dataset_digest(cases: Sequence[QACase]) -> str:
    return hashlib.sha256(qa_dataset_jsonl(cases).encode("utf-8")).hexdigest()


def save_qa_dataset(cases: Sequence[QACase], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(qa_dataset_jsonl(cases), encoding="utf-8")
    return output_path


def load_qa_dataset(path: str | Path) -> list[QACase]:
    cases: list[QACase] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise SpatialQAError(f"QA dataset line {line_number} must be an object")
        cases.append(qa_case_from_dict(cast(Mapping[str, Any], payload)))
    return cases


def qa_dataset_summary(cases: Sequence[QACase]) -> dict[str, Any]:
    return {
        "schema_version": QA_DATASET_SUMMARY_SCHEMA_VERSION,
        "case_count": len(cases),
        "scene_ids": sorted({case.scene_id for case in cases}),
        "episode_ids": sorted({case.episode_id for case in cases}),
        "question_types": _sorted_counts(case.question_type for case in cases),
        "tags": _sorted_counts(tag for case in cases for tag in case.tags),
    }


def validate_qa_dataset(cases: Sequence[QACase]) -> dict[str, Any]:
    case_ids = [case.id for case in cases]
    checks = [
        {
            "name": "case_count_positive",
            "passed": len(cases) > 0,
            "expected": "at least 1",
            "actual": len(cases),
        },
        {
            "name": "unique_case_ids",
            "passed": len(set(case_ids)) == len(case_ids),
            "expected": len(case_ids),
            "actual": len(set(case_ids)),
        },
        {
            "name": "case_schema_versions",
            "passed": True,
            "expected": QA_CASE_SCHEMA_VERSION,
            "actual": [QA_CASE_SCHEMA_VERSION],
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "digest": qa_dataset_digest(cases),
        "summary": qa_dataset_summary(cases),
        "checks": checks,
    }


def compare_qa_dataset(cases: Sequence[QACase], graph: DynamicSceneGraph) -> dict[str, Any]:
    validation = validate_qa_dataset(cases)
    graph_digest = graph_json_digest(graph)
    graph_digest_mismatches = [
        {
            "case_id": case.id,
            "expected": case.graph_digest,
            "actual": graph_digest,
        }
        for case in cases
        if case.graph_digest != graph_digest
    ]
    answer_differences = _answer_differences(cases, graph)
    graph_digest_check: dict[str, Any] = {
        "name": "graph_digest_matches_current",
        "passed": not graph_digest_mismatches,
        "expected": "each case graph_digest matches current graph",
        "actual": len(graph_digest_mismatches),
    }
    if graph_digest_mismatches:
        graph_digest_check["differences"] = graph_digest_mismatches
    answers_check: dict[str, Any] = {
        "name": "answers_match_graph",
        "passed": not answer_differences,
        "expected": "stored answers and evidence match current graph",
        "actual": len(answer_differences),
    }
    if answer_differences:
        answers_check["differences"] = answer_differences
    checks = [
        {
            "name": "dataset_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        graph_digest_check,
        answers_check,
    ]
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": qa_dataset_digest(cases),
        "current_graph_digest": graph_digest,
        "summary": qa_dataset_summary(cases),
        "validation": validation,
        "checks": checks,
    }


def _answer_differences(
    cases: Sequence[QACase],
    graph: DynamicSceneGraph,
) -> list[dict[str, Any]]:
    qa = SpatialQAEngine(GraphTool(graph))
    differences: list[dict[str, Any]] = []
    for case in cases:
        response = qa.answer(case.question)
        if response.error is not None:
            differences.append(
                {
                    "case_id": case.id,
                    "path": "question",
                    "expected": "answerable question",
                    "actual": response.error,
                }
            )
            continue
        if response.answer != case.answer:
            differences.append(
                {
                    "case_id": case.id,
                    "path": "answer",
                    "expected": case.answer,
                    "actual": response.answer,
                }
            )
        if tuple(response.evidence_nodes) != case.required_nodes:
            differences.append(
                {
                    "case_id": case.id,
                    "path": "required_nodes",
                    "expected": list(case.required_nodes),
                    "actual": response.evidence_nodes,
                }
            )
        if tuple(response.evidence_edges) != case.required_edges:
            differences.append(
                {
                    "case_id": case.id,
                    "path": "required_edges",
                    "expected": list(case.required_edges),
                    "actual": response.evidence_edges,
                }
            )
    return differences


def _question_relation_edges(graph: DynamicSceneGraph) -> tuple[Edge, ...]:
    edges = [
        edge
        for edge in graph.edges
        if edge.src in graph.object_states
        and edge.dst in graph.object_states
        and edge.relation not in QUESTION_RELATION_EXCLUDES
    ]
    return tuple(sorted(edges, key=_edge_sort_key))


def _room_question_objects(tool: GraphTool) -> tuple[ObjectState, ...]:
    return tuple(
        state
        for state in tool.find_objects()
        if state.label.casefold() not in STRUCTURAL_OBJECT_LABELS
    )


def _has_place_nodes(graph: DynamicSceneGraph) -> bool:
    return any(node.type == "region" for node in graph.nodes.values())


def _graph_steps(graph: DynamicSceneGraph) -> tuple[int, ...]:
    steps: set[int] = {edge.step for edge in graph.edges}
    for object_states in graph.object_state_history.values():
        steps.update(state.step for state in object_states)
    for agent_states in graph.agent_pose_history.values():
        steps.update(state.step for state in agent_states)
    return tuple(sorted(steps))


def _max_graph_step(graph: DynamicSceneGraph) -> int:
    steps = _graph_steps(graph)
    return steps[-1] if steps else 0


def _edge_sort_key(edge: Edge) -> tuple[int, str, str, str, str]:
    return (edge.step, edge.src, edge.relation, edge.dst, edge.reference_frame)


def _validate_non_empty_str(value: str, key: str) -> None:
    if not isinstance(value, str) or not value:
        raise SpatialQAError(f"{key} must be a non-empty string")


def _unique_strings(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise SpatialQAError("tags must be strings")
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


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


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise SpatialQAError(f"QA case field must be a string: {key}")
    return value


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SpatialQAError(f"QA case field must be a string: {key}")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpatialQAError(f"QA case field must be an integer: {key}")
    return value


def _required_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"QA case field must be an object: {key}")
    return _json_mapping(cast(Mapping[str, Any], value))


def _string_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key, [])
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SpatialQAError(f"QA case field must be a string sequence: {key}")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise SpatialQAError(f"QA case field must be a string sequence: {key}")
        items.append(item)
    return tuple(items)
