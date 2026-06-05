from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


TRAJECTORY_AUDIT_SCHEMA_VERSION = "dsg-spatialqa-lab.trajectory-audit.v1"
TRAJECTORY_PROTOCOL_COMPARISON_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.trajectory-protocol-comparison.v1"
)
TRAJECTORY_ARTIFACT_SELF_CHECK_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.trajectory-artifact-self-check.v1"
)
FORBIDDEN_TRAJECTORY_FIELDS = frozenset(
    {"gold_answer", "gold_evidence", "required_edges", "required_nodes"}
)
FORMAL_PROTOCOL_GATE_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.reachable-nbv-formal-protocol-gate.v1"
)


def diagnostic_protocol_metadata() -> dict[str, Any]:
    return {
        "collection_kind": "coverage_diagnostic",
        "not_autonomous_exploration": True,
        "metadata_used_for": "planning_only",
        "not_predicted_graph_evidence": True,
        "navigation_validated": False,
    }


def filter_cases_for_trajectory(
    cases: Sequence[Any],
    trajectory: Mapping[str, Any],
) -> list[Any]:
    episode_id = trajectory.get("episode_id")
    if not isinstance(episode_id, str) or episode_id == "":
        return list(cases)
    return [
        case
        for case in cases
        if getattr(case, "episode_id", None) == episode_id
    ]


def observed_node_ids_from_observations(observations: Sequence[Any]) -> set[str]:
    node_ids: set[str] = set()
    for observation in observations:
        for room in _sequence(getattr(observation, "rooms", ())):
            node_id = getattr(room, "node_id", None)
            if isinstance(node_id, str):
                node_ids.add(node_id)
        for region in _sequence(getattr(observation, "regions", ())):
            node_id = getattr(region, "node_id", None)
            if isinstance(node_id, str):
                node_ids.add(node_id)
        for obj in _sequence(getattr(observation, "objects", ())):
            object_id = getattr(obj, "object_id", None)
            if isinstance(object_id, str):
                node_ids.add(object_id)
    return node_ids


def trajectory_coverage_audit(
    trajectory: Mapping[str, Any],
    *,
    qa_case_count: int = 0,
    graph_tool_strict_exact_count: int = 0,
    graph_tool_semantic_match_count: int = 0,
    target_object_ids: set[str] | frozenset[str] | None = None,
    support_object_ids: set[str] | frozenset[str] | None = None,
    observed_object_ids: set[str] | frozenset[str] | None = None,
    observed_support_ids: set[str] | frozenset[str] | None = None,
    same_frame_relation_count: int = 0,
    current_location_edge_count: int = 0,
    on_relation_observable_count: int = 0,
    state_evidence_observable_count: int = 0,
    relation_recall: float = 0.0,
    relation_precision: float = 0.0,
) -> dict[str, Any]:
    steps = _steps(trajectory)
    selected = [_mapping(step.get("selected_viewpoint")) for step in steps]
    visited_positions = {
        (round(_float(view.get("x")), 3), round(_float(view.get("z")), 3))
        for view in selected
        if "x" in view and "z" in view
    }
    yaw_values = {
        round(_float(view.get("yaw")), 3)
        for view in selected
        if "yaw" in view
    }
    pitch_values = {
        round(_float(view.get("pitch")), 3)
        for view in selected
        if "pitch" in view
    }
    action_count = sum(
        len(_sequence(step.get("executed_actions")))
        + len(_sequence(step.get("capture_actions")))
        for step in steps
    )
    observed_objects = observed_object_ids or set()
    observed_supports = observed_support_ids or set()
    targets = target_object_ids or set()
    supports = support_object_ids or set()
    evidence_observable = min(
        qa_case_count,
        same_frame_relation_count + current_location_edge_count,
    )
    relation_f1 = (
        0.0
        if relation_recall + relation_precision == 0.0
        else 2 * relation_recall * relation_precision / (relation_recall + relation_precision)
    )
    state_count = min(state_evidence_observable_count, qa_case_count)
    audit = {
        "schema_version": TRAJECTORY_AUDIT_SCHEMA_VERSION,
        "protocol": str(trajectory.get("collection_kind", "unknown")),
        "trajectory_id": trajectory.get("trajectory_id"),
        "navigation_validated": trajectory.get("navigation_validated") is True,
        "runtime_kind": trajectory.get("runtime_kind"),
        "real_ai2thor_runtime": trajectory.get("real_ai2thor_runtime"),
        "trajectory_length": len(steps),
        "action_count": action_count,
        "visited_reachable_position_count": len(visited_positions),
        "visited_grid_ratio": _ratio(len(visited_positions), max(1, len(steps) + 3)),
        "convex_hull_area": _bbox_area(visited_positions),
        "convex_hull_area_ratio": _ratio(_bbox_area(visited_positions), 16.0),
        "yaw_coverage_degrees": float(len(yaw_values) * 90),
        "pitch_coverage_degrees": float(len(pitch_values) * 30),
        "viewpoint_count": len(selected),
        "capture_frame_count": sum(len(_sequence(step.get("capture_actions"))) for step in steps),
        "target_visible_count": len(set(observed_objects) & set(targets)),
        "target_visible_rate": _ratio(len(set(observed_objects) & set(targets)), len(targets)),
        "support_visible_count": len(set(observed_supports) & set(supports)),
        "support_visible_rate": _ratio(len(set(observed_supports) & set(supports)), len(supports)),
        "target_support_same_frame_count": same_frame_relation_count,
        "target_support_same_frame_rate": _ratio(same_frame_relation_count, qa_case_count),
        "current_location_edge_count": current_location_edge_count,
        "current_location_edge_acceptance_rate": _ratio(current_location_edge_count, qa_case_count),
        "ON_relation_observable_count": on_relation_observable_count,
        "ON_relation_observable_rate": _ratio(on_relation_observable_count, qa_case_count),
        "state_evidence_observable_count": state_count,
        "state_evidence_observable_rate": _ratio(state_count, qa_case_count),
        "object_recall": _ratio(len(set(observed_objects) & set(targets)), len(targets)),
        "relation_recall": relation_recall,
        "relation_precision": relation_precision,
        "relation_f1": relation_f1,
        "evidence_observable_qa_count": evidence_observable,
        "evidence_observable_qa_rate": _ratio(evidence_observable, qa_case_count),
        "GraphTool_strict_exact": graph_tool_strict_exact_count,
        "GraphTool_semantic_match": graph_tool_semantic_match_count,
        "missing_object_count": max(0, len(targets - set(observed_objects))),
        "missing_support_count": max(0, len(supports - set(observed_supports))),
        "missing_relation_count": max(0, qa_case_count - same_frame_relation_count),
        "missing_state_count": max(0, qa_case_count - state_evidence_observable_count),
        "unlocated_object_count": max(0, len(set(observed_objects)) - current_location_edge_count),
    }
    return audit


def compare_trajectory_protocols(
    fixed_audit: Mapping[str, Any],
    diagnostic_audit: Mapping[str, Any],
    nbv_audit: Mapping[str, Any],
) -> dict[str, Any]:
    fixed_evidence = _number(fixed_audit.get("evidence_observable_qa_count"))
    diagnostic_evidence = _number(diagnostic_audit.get("evidence_observable_qa_count"))
    nbv_evidence = _number(nbv_audit.get("evidence_observable_qa_count"))
    fixed_semantic = _number(fixed_audit.get("GraphTool_semantic_match"))
    nbv_semantic = _number(nbv_audit.get("GraphTool_semantic_match"))
    missing_support_reduced = _number(nbv_audit.get("missing_support_count")) < _number(
        fixed_audit.get("missing_support_count")
    )
    missing_relation_reduced = _number(nbv_audit.get("missing_relation_count")) < _number(
        fixed_audit.get("missing_relation_count")
    )
    judgement = {
        "fixed_coverage_insufficient": fixed_audit.get("navigation_validated") is not True
        or fixed_evidence < diagnostic_evidence,
        "diagnostic_improves_evidence_coverage": diagnostic_evidence > fixed_evidence,
        "reachable_nbv_navigation_validated": nbv_audit.get("navigation_validated") is True,
        "reachable_nbv_reduces_missing_support": missing_support_reduced,
        "reachable_nbv_reduces_missing_relation": missing_relation_reduced,
        "reachable_nbv_improves_graphtool_semantic_match": nbv_semantic > fixed_semantic,
        "reachable_nbv_can_be_formal_protocol": (
            nbv_audit.get("navigation_validated") is True
            and nbv_evidence > fixed_evidence
            and missing_support_reduced
            and missing_relation_reduced
            and _number(nbv_audit.get("target_support_same_frame_rate"))
            > _number(fixed_audit.get("target_support_same_frame_rate"))
        ),
    }
    return {
        "schema_version": TRAJECTORY_PROTOCOL_COMPARISON_SCHEMA_VERSION,
        "fixed": dict(fixed_audit),
        "diagnostic": dict(diagnostic_audit),
        "reachable_nbv": dict(nbv_audit),
        "judgement": judgement,
    }


def reachable_nbv_formal_protocol_gate(
    *,
    episode_id: str,
    trajectory: Mapping[str, Any],
    fixed_audit: Mapping[str, Any],
    nbv_audit: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    checks = [
        {
            "name": "real_ai2thor_runtime",
            "passed": trajectory.get("real_ai2thor_runtime") is True,
        },
        {
            "name": "navigation_validated",
            "passed": trajectory.get("navigation_validated") is True
            and nbv_audit.get("navigation_validated") is True,
        },
        {
            "name": "teleport_used_false",
            "passed": trajectory.get("teleport_used") is False
            and not _trajectory_uses_action(trajectory, "TeleportFull"),
        },
        {
            "name": "closed_loop_memory_update",
            "passed": trajectory.get("closed_loop_memory_update") is True
            and all(
                isinstance(_mapping(decision).get("memory_before"), Mapping)
                and isinstance(_mapping(decision).get("memory_after"), Mapping)
                for decision in decisions
            ),
        },
        {
            "name": "target_support_same_frame_rate_gt_fixed",
            "passed": _number(nbv_audit.get("target_support_same_frame_rate"))
            > _number(fixed_audit.get("target_support_same_frame_rate")),
        },
        {
            "name": "evidence_observable_qa_count_gte_fixed",
            "passed": _number(nbv_audit.get("evidence_observable_qa_count"))
            >= _number(fixed_audit.get("evidence_observable_qa_count")),
        },
        {
            "name": "missing_support_count_lte_fixed",
            "passed": _number(nbv_audit.get("missing_support_count"))
            <= _number(fixed_audit.get("missing_support_count")),
        },
        {
            "name": "missing_relation_count_lt_fixed",
            "passed": _number(nbv_audit.get("missing_relation_count"))
            < _number(fixed_audit.get("missing_relation_count")),
        },
        {
            "name": "graphtool_semantic_match_gte_fixed",
            "passed": _number(nbv_audit.get("GraphTool_semantic_match"))
            >= _number(fixed_audit.get("GraphTool_semantic_match")),
        },
        {
            "name": "no_gold_leakage",
            "passed": trajectory.get("uses_gold_answer") is False
            and trajectory.get("uses_gold_evidence") is False
            and trajectory.get("uses_required_edges") is False
            and trajectory.get("uses_required_nodes") is False
            and not _forbidden_paths(trajectory)
            and not any(_forbidden_paths(decision) for decision in decisions),
        },
    ]
    failed = [check["name"] for check in checks if check["passed"] is not True]
    return {
        "schema_version": FORMAL_PROTOCOL_GATE_SCHEMA_VERSION,
        "episode_id": episode_id,
        "formal_protocol_ready": not failed,
        "failed_checks": failed,
        "checks": checks,
    }


def trajectory_artifact_self_check(
    *,
    trajectory: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    artifact_paths: Mapping[str, str],
) -> dict[str, Any]:
    executed_actions = [
        action.get("action")
        for step in _steps(trajectory)
        for action in (
            *_sequence(_mapping(step).get("executed_actions")),
            *_sequence(_mapping(step).get("executed_capture_actions")),
        )
        if isinstance(action, Mapping)
    ]
    checks = [
        {
            "name": "trajectory_exists",
            "passed": _path_exists(artifact_paths.get("trajectory")),
        },
        {
            "name": "decision_trace_non_empty",
            "passed": bool(decisions),
        },
        {
            "name": "observation_sequence_exists",
            "passed": _path_exists(artifact_paths.get("observation_sequence")),
        },
        {
            "name": "predicted_graph_exists",
            "passed": _path_exists(artifact_paths.get("predicted_graph")),
        },
        {
            "name": "navigation_validated",
            "passed": trajectory.get("navigation_validated") is True,
        },
        {
            "name": "collection_kind",
            "passed": trajectory.get("collection_kind")
            == "reachable_relation_centric_nbv",
        },
        {
            "name": "autonomous_exploration",
            "passed": trajectory.get("autonomous_exploration") is True,
        },
        {
            "name": "closed_loop_memory_update",
            "passed": trajectory.get("closed_loop_memory_update") is True,
        },
        {
            "name": "no_forbidden_gold_fields",
            "passed": not _forbidden_paths(trajectory)
            and not any(_forbidden_paths(decision) for decision in decisions),
        },
        {
            "name": "no_teleportfull_in_formal_nbv",
            "passed": "TeleportFull" not in executed_actions,
        },
        {
            "name": "each_selected_viewpoint_has_path",
            "passed": all(
                isinstance(_mapping(step).get("path_to_reach"), Mapping)
                for step in _steps(trajectory)
            ),
        },
        {
            "name": "each_iteration_has_memory_and_score",
            "passed": all(
                isinstance(decision.get("memory_before"), Mapping)
                and isinstance(decision.get("memory_after"), Mapping)
                and isinstance(decision.get("score_terms"), Mapping)
                for decision in decisions
            ),
        },
    ]
    return {
        "schema_version": TRAJECTORY_ARTIFACT_SELF_CHECK_SCHEMA_VERSION,
        "valid": all(check["passed"] is True for check in checks),
        "runtime_kind": trajectory.get("runtime_kind"),
        "real_ai2thor_runtime": trajectory.get("real_ai2thor_runtime"),
        "artifact_paths": dict(artifact_paths),
        "checks": checks,
    }


def comparison_markdown(report: Mapping[str, Any]) -> str:
    judgement = _mapping(report.get("judgement"))
    lines = [
        "# Episode001 三轨迹协议对比",
        "",
        "## 结论",
        f"- fixed trajectory 覆盖不足: {judgement.get('fixed_coverage_insufficient')}",
        f"- diagnostic 补采提升 evidence coverage: {judgement.get('diagnostic_improves_evidence_coverage')}",
        f"- reachable NBV navigation validated: {judgement.get('reachable_nbv_navigation_validated')}",
        f"- reachable NBV 减少 missing support: {judgement.get('reachable_nbv_reduces_missing_support')}",
        f"- reachable NBV 减少 missing relation: {judgement.get('reachable_nbv_reduces_missing_relation')}",
        f"- reachable NBV 提升 GraphTool semantic match: {judgement.get('reachable_nbv_improves_graphtool_semantic_match')}",
        f"- reachable NBV 可作为正式探索协议: {judgement.get('reachable_nbv_can_be_formal_protocol')}",
        "",
        "## 边界",
        "- 若 runtime_kind=fake_controller，则该结果只验证机制，不是实时 AI2-THOR 真实 rollout 结论。",
    ]
    return "\n".join(lines) + "\n"


def save_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _steps(trajectory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    steps = trajectory.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, str):
        return []
    return [_mapping(step) for step in steps]


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _float(value: object) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _number(value: object) -> float:
    return _float(value)


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else round(float(numerator) / float(denominator), 6)


def _bbox_area(points: set[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    xs = [point[0] for point in points]
    zs = [point[1] for point in points]
    return round((max(xs) - min(xs)) * (max(zs) - min(zs)), 6)


def _path_exists(value: object) -> bool:
    return isinstance(value, str) and Path(value).exists()


def _forbidden_paths(value: object, *, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_prefix = f"{prefix}.{key_text}"
            if key_text in FORBIDDEN_TRAJECTORY_FIELDS:
                paths.append(child_prefix)
            paths.extend(_forbidden_paths(child, prefix=child_prefix))
    elif isinstance(value, Sequence) and not isinstance(value, str):
        for index, child in enumerate(value):
            paths.extend(_forbidden_paths(child, prefix=f"{prefix}[{index}]"))
    return paths


def _trajectory_uses_action(trajectory: Mapping[str, Any], action_name: str) -> bool:
    return any(
        action.get("action") == action_name
        for step in _steps(trajectory)
        for action in (
            *_sequence(step.get("executed_actions")),
            *_sequence(step.get("executed_capture_actions")),
        )
        if isinstance(action, Mapping)
    )
