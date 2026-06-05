from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from dsg_spatialqa_lab.benchmark import load_qa_dataset
from dsg_spatialqa_lab.navigation.trajectory_audit import (
    canonical_id_aliases_for_qa,
    canonicalize_id,
    filter_cases_for_trajectory,
    observed_node_ids_from_observations,
    save_json,
    trajectory_coverage_audit,
)
from dsg_spatialqa_lab.observations import load_scene_observation_sequence
from dsg_spatialqa_lab.scene_io import load_graph_json
from dsg_spatialqa_lab.schema import SpatialQAError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Audit relation-centric trajectory coverage from local artifacts.",
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--observation-sequence", type=Path, required=True)
    parser.add_argument("--predicted-graph", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
        if not isinstance(trajectory, dict):
            raise SpatialQAError("trajectory must be a JSON object")
        observations = load_scene_observation_sequence(args.observation_sequence)
        graph = load_graph_json(args.predicted_graph)
        qa_cases = filter_cases_for_trajectory(load_qa_dataset(args.qa), trajectory)
        targets, supports = _qa_target_support_ids(qa_cases)
        qa_ids = targets | supports
        observed_objects = {
            obj.object_id
            for observation in observations
            for obj in observation.objects
        }
        observed_nodes = observed_node_ids_from_observations(observations)
        aliases = canonical_id_aliases_for_qa(
            observed_ids=observed_nodes,
            qa_ids=qa_ids,
        )
        canonical_observed_objects = {
            canonicalize_id(object_id, aliases) for object_id in observed_objects
        }
        canonical_observed_nodes = {
            canonicalize_id(object_id, aliases) for object_id in observed_nodes
        }
        observed_supports = canonical_observed_nodes & supports
        current_edges = {
            (
                f"{canonicalize_id(edge.src, aliases)}-"
                f"{edge.relation}-"
                f"{canonicalize_id(edge.dst, aliases)}"
            )
            for edge in graph.edges
            if edge.relation in {"ON", "INSIDE", "IN_ROOM", "IN_REGION"}
        }
        same_frame_count = _same_frame_count(observations, qa_cases, aliases)
        current_location_count = _current_location_count(current_edges, qa_cases)
        audit = trajectory_coverage_audit(
            trajectory,
            qa_case_count=len(qa_cases),
            graph_tool_strict_exact_count=current_location_count,
            graph_tool_semantic_match_count=current_location_count,
            target_object_ids=targets,
            support_object_ids=supports,
            observed_object_ids=canonical_observed_objects,
            observed_support_ids=observed_supports,
            same_frame_relation_count=same_frame_count,
            current_location_edge_count=current_location_count,
            on_relation_observable_count=current_location_count,
            state_evidence_observable_count=len(canonical_observed_objects),
            relation_recall=(
                0.0 if not qa_cases else current_location_count / float(len(qa_cases))
            ),
            relation_precision=(
                0.0 if not current_edges else current_location_count / float(len(current_edges))
            ),
        )
        if aliases:
            audit["canonical_id_alias_count"] = len(aliases)
            audit["canonical_id_aliases"] = aliases
        save_json(audit, args.output)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit({"valid": False, "error": str(exc)})
        return 1
    _emit({"valid": True, "path": str(args.output), "summary": audit})
    return 0


def _same_frame_count(
    observations: Any,
    qa_cases: Any,
    aliases: Mapping[str, str] | None = None,
) -> int:
    count = 0
    alias_map = aliases or {}
    object_sets = [
        {
            canonicalize_id(object_id, alias_map)
            for object_id in observed_node_ids_from_observations((observation,))
        }
        for observation in observations
    ]
    for case in qa_cases:
        object_id = case.question.get("object_id")
        location = case.answer.get("current_location")
        dst = location.get("dst") if isinstance(location, dict) else None
        if isinstance(object_id, str) and isinstance(dst, str):
            if any({object_id, dst}.issubset(objects) for objects in object_sets):
                count += 1
    return count


def _current_location_count(edges: set[str], qa_cases: Any) -> int:
    count = 0
    for case in qa_cases:
        object_id = case.question.get("object_id")
        location = case.answer.get("current_location")
        if not isinstance(object_id, str) or not isinstance(location, dict):
            continue
        relation = location.get("relation")
        dst = location.get("dst")
        if isinstance(relation, str) and isinstance(dst, str):
            if f"{object_id}-{relation}-{dst}" in edges:
                count += 1
    return count


def _qa_target_support_ids(cases: Any) -> tuple[set[str], set[str]]:
    targets: set[str] = set()
    supports: set[str] = set()
    for case in cases:
        object_id = case.question.get("object_id")
        if isinstance(object_id, str):
            targets.add(object_id)
        location = case.answer.get("current_location")
        if isinstance(location, dict):
            dst = location.get("dst")
            if isinstance(dst, str):
                supports.add(dst)
    return targets, supports


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
