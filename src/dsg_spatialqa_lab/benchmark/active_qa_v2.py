from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab.memory import CONTAINMENT_RELATIONS, DynamicSceneGraph
from dsg_spatialqa_lab.observations import ObjectObservation, SceneObservation
from dsg_spatialqa_lab.scene_io import graph_json_digest


ACTIVE_QA_V2_SCHEMA_VERSION = "dsg-spatialqa-lab.active-qa-case.v2"
ACTIVE_QA_V2_QUALITY_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.active-qa-v2-quality-report.v1"
)
ACTIVE_QA_V2_REQUEST_BUNDLE_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.active-qa-v2-vlm-request-bundle.v1"
)
ACTIVE_QA_V2_SPLIT_NAMES = (
    "full_oracle",
    "observation_aware",
    "situated",
    "temporal",
    "anti_shortcut",
    "relation_centric",
)
SUPPORT_RELATIONS = frozenset({"ON", "INSIDE"})
RELATIVE_RELATIONS = frozenset({"BEHIND", "FRONT_OF", "LEFT_OF", "RIGHT_OF"})
FORBIDDEN_REQUEST_KEYS = frozenset(
    {
        "gold_answer",
        "gold_evidence",
        "required_edges",
        "required_nodes",
        "visible_object_ids",
        "visible_object_labels",
    }
)


def build_active_qa_v2_splits(
    *,
    episode_id: str,
    scene_id: str,
    trajectory: Mapping[str, Any],
    observations: Sequence[SceneObservation],
    graph: DynamicSceneGraph,
    max_records_per_split: int = 60,
) -> dict[str, list[dict[str, Any]]]:
    graph_digest = graph_json_digest(graph)
    index = _observation_index(observations)
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in ACTIVE_QA_V2_SPLIT_NAMES}

    relation_edges = [
        edge
        for edge in sorted(
            graph.edges,
            key=lambda item: (item.step, item.src, item.relation, item.dst),
        )
        if edge.relation in CONTAINMENT_RELATIONS
        and edge.src in graph.nodes
        and edge.dst in graph.nodes
    ]
    relation_edges = relation_edges[:max_records_per_split]

    for edge in relation_edges:
        src_node = graph.nodes[edge.src]
        dst_node = graph.nodes[edge.dst]
        evidence_frames = _same_frame_steps(index["object_steps"], edge.src, edge.dst)
        if not evidence_frames:
            evidence_frames = [edge.step]
        base = _record(
            episode_id=episode_id,
            scene_id=scene_id,
            split="relation_centric",
            question_type=(
                "support_relation"
                if edge.relation in SUPPORT_RELATIONS
                else "object_location"
            ),
            question_text=(
                f"What support surface is the {src_node.label or _label(edge.src)} on?"
                if edge.relation == "ON"
                else f"Where is the {src_node.label or _label(edge.src)}?"
            ),
            target_id=edge.src,
            target_label=src_node.label or _label(edge.src),
            relation=edge.relation,
            dst_id=edge.dst,
            dst_label=dst_node.label or _label(edge.dst),
            step=edge.step,
            situation=_situation(
                index["step_pose"].get(evidence_frames[0]),
                evidence_frames[0],
                index["step_frame"].get(evidence_frames[0]),
            ),
            required_edges=[f"{edge.src}-{edge.relation}-{edge.dst}"],
            evidence_frames=evidence_frames,
            trajectory=trajectory,
            source_graph_digest=graph_digest,
            evidence_observable=bool(evidence_frames),
            anti_shortcut=_anti_shortcut(
                src_label=src_node.label or _label(edge.src),
                dst_label=dst_node.label or _label(edge.dst),
                distractor_count=index["label_counts"].get(src_node.label or _label(edge.src), 0) - 1,
                relation=edge.relation,
            ),
        )
        _append(records, "relation_centric", base, max_records_per_split)
        _append(records, "observation_aware", {**base, "split": "observation_aware"}, max_records_per_split)
        if base["anti_shortcut"]["language_prior_risk"] != "high" or base["anti_shortcut"]["has_distractor_same_category"]:
            _append(records, "anti_shortcut", {**base, "split": "anti_shortcut"}, max_records_per_split)
        if base["question_type"] != "object_location":
            _append(records, "full_oracle", {**base, "split": "full_oracle"}, max_records_per_split)

    relative_edges = [
        edge
        for edge in sorted(
            graph.edges,
            key=lambda item: (item.step, item.src, item.relation, item.dst),
        )
        if edge.relation in RELATIVE_RELATIONS
        and edge.src in graph.nodes
        and edge.dst in graph.nodes
    ][:max_records_per_split]
    for edge in relative_edges:
        src_node = graph.nodes[edge.src]
        dst_node = graph.nodes[edge.dst]
        evidence_frames = _same_frame_steps(index["object_steps"], edge.src, edge.dst) or [edge.step]
        relative = _record(
            episode_id=episode_id,
            scene_id=scene_id,
            split="situated",
            question_type="relative_relation",
            question_text=(
                f"From the robot viewpoint, is the "
                f"{src_node.label or _label(edge.src)} {edge.relation.lower()} "
                f"the {dst_node.label or _label(edge.dst)}?"
            ),
            target_id=edge.src,
            target_label=src_node.label or _label(edge.src),
            relation=edge.relation,
            dst_id=edge.dst,
            dst_label=dst_node.label or _label(edge.dst),
            step=edge.step,
            situation=_situation(
                index["step_pose"].get(evidence_frames[0]),
                evidence_frames[0],
                index["step_frame"].get(evidence_frames[0]),
            ),
            required_edges=[f"{edge.src}-{edge.relation}-{edge.dst}"],
            evidence_frames=evidence_frames,
            trajectory=trajectory,
            source_graph_digest=graph_digest,
            evidence_observable=True,
            anti_shortcut={
                "language_prior_risk": "low",
                "requires_3d_evidence": True,
                "has_distractor_same_category": index["label_counts"].get(src_node.label or _label(edge.src), 0) > 1,
                "has_distractor_same_support": False,
            },
        )
        _append(records, "situated", relative, max_records_per_split)
        _append(records, "relation_centric", {**relative, "split": "relation_centric"}, max_records_per_split)

    for relative in _relative_relation_records_from_observations(
        episode_id=episode_id,
        scene_id=scene_id,
        trajectory=trajectory,
        index=index,
        source_graph_digest=graph_digest,
        max_records=max_records_per_split,
    ):
        _append(records, "situated", relative, max_records_per_split)
        _append(records, "relation_centric", {**relative, "split": "relation_centric"}, max_records_per_split)

    for nearest in _nearest_object_records(
        episode_id=episode_id,
        scene_id=scene_id,
        trajectory=trajectory,
        index=index,
        source_graph_digest=graph_digest,
        max_records=max_records_per_split,
    ):
        _append(records, "relation_centric", nearest, max_records_per_split)

    for multi_hop in _multi_hop_records(
        episode_id=episode_id,
        scene_id=scene_id,
        trajectory=trajectory,
        graph=graph,
        index=index,
        relation_edges=relation_edges,
        source_graph_digest=graph_digest,
        max_records=max_records_per_split,
    ):
        _append(records, "relation_centric", multi_hop, max_records_per_split)
        _append(records, "anti_shortcut", {**multi_hop, "split": "anti_shortcut"}, max_records_per_split)

    for state_change in _state_change_records(
        episode_id=episode_id,
        scene_id=scene_id,
        trajectory=trajectory,
        index=index,
        source_graph_digest=graph_digest,
        max_records=max_records_per_split,
    ):
        _append(records, "temporal", state_change, max_records_per_split)

    for obj_id, steps in sorted(index["object_steps"].items())[:max_records_per_split]:
        node = graph.nodes.get(obj_id)
        label = (node.label if node else None) or _label(obj_id)
        last_step = max(steps)
        last_pose = index["step_pose"].get(last_step)
        temporal = _record(
            episode_id=episode_id,
            scene_id=scene_id,
            split="temporal",
            question_type="temporal_last_seen",
            question_text=f"Where was the {label} last seen during this trajectory?",
            target_id=obj_id,
            target_label=label,
            relation="VISIBLE_FROM",
            dst_id="agent",
            dst_label="agent",
            step=last_step,
            situation=_situation(
                last_pose,
                last_step,
                index["step_frame"].get(last_step),
            ),
            required_edges=[],
            evidence_frames=[last_step],
            trajectory=trajectory,
            source_graph_digest=graph_digest,
            evidence_observable=True,
            anti_shortcut={
                "language_prior_risk": "low",
                "requires_3d_evidence": True,
                "has_distractor_same_category": index["label_counts"].get(label, 0) > 1,
                "has_distractor_same_support": False,
            },
        )
        _append(records, "temporal", temporal, max_records_per_split)
        situated = {
            **temporal,
            "id": temporal["id"].replace(":temporal_last_seen:", ":situated_egocentric:"),
            "split": "situated",
            "question_type": "situated_egocentric",
            "question_text": f"From the robot pose at step {last_step}, where is the {label} relative to the agent?",
            "answer": {
                "relation": "VISIBLE_FROM",
                "dst": "agent",
                "dst_label": "agent",
                "step": last_step,
            },
        }
        _append(records, "situated", situated, max_records_per_split)

    for split_name, rows in list(records.items()):
        records[split_name] = _dedupe_records(rows)[:max_records_per_split]
    return records


def active_qa_v2_quality_report(
    *,
    episode_id: str,
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    all_records = [row for rows in splits.values() for row in rows]
    unique_records = {str(row.get("id")): row for row in all_records}.values()
    question_type_counts = Counter(str(row.get("question_type")) for row in unique_records)
    non_full_records = [
        row for row in unique_records if row.get("split") != "full_oracle"
    ]
    object_location_count = sum(
        1 for row in non_full_records if row.get("question_type") == "object_location"
    )
    situated_temporal_relation = sum(
        len(splits.get(name, ()))
        for name in ("situated", "temporal", "relation_centric")
    )
    observation_aware_aligned = all(
        _mapping(row.get("observability")).get("evidence_observable") is True
        for row in splits.get("observation_aware", ())
    )
    anti_shortcut_ok = all(
        _mapping(row.get("anti_shortcut")).get("language_prior_risk") != "high"
        or _mapping(row.get("anti_shortcut")).get("has_distractor_same_category") is True
        or _mapping(row.get("anti_shortcut")).get("has_distractor_same_support") is True
        for row in splits.get("anti_shortcut", ())
    )
    required_fields_ok = all(
        all(key in row for key in ("question_text", "situation", "observability", "anti_shortcut"))
        for row in unique_records
    )
    object_location_rate = _ratio(object_location_count, len(non_full_records))
    checks = [
        {"name": "object_location_lte_60_percent", "passed": object_location_rate <= 0.6},
        {"name": "at_least_three_question_types", "passed": len(question_type_counts) >= 3},
        {"name": "at_least_eight_question_types", "passed": len(question_type_counts) >= 8},
        {"name": "situated_temporal_relation_nonzero", "passed": situated_temporal_relation > 0},
        {"name": "observation_aware_aligned", "passed": observation_aware_aligned},
        {"name": "anti_shortcut_cases_valid", "passed": anti_shortcut_ok},
        {"name": "required_quality_fields_present", "passed": required_fields_ok},
    ]
    report: dict[str, Any] = {
        "schema_version": ACTIVE_QA_V2_QUALITY_REPORT_SCHEMA_VERSION,
        "episode_id": episode_id,
        "valid": all(check["passed"] is True for check in checks),
        "checks": checks,
        "summary": {
            "split_counts": {name: len(splits.get(name, ())) for name in ACTIVE_QA_V2_SPLIT_NAMES},
            "question_type_counts": dict(sorted(question_type_counts.items())),
            "question_type_count": len(question_type_counts),
            "p53_required_question_types": [
                "multi_hop",
                "nearest_object",
                "relative_relation",
                "state_change",
            ],
            "object_location_rate": object_location_rate,
            "situated_temporal_relation_count": situated_temporal_relation,
            "observation_aware_count": len(splits.get("observation_aware", ())),
        },
    }
    report["report_digest"] = _digest(report)
    return report


def build_active_qa_v2_vlm_request_bundle(
    *,
    episode_id: str,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prediction_cases = _dedupe_prediction_cases([
        {
            "case_id": row.get("id"),
            "episode_id": row.get("episode_id"),
            "scene_id": row.get("scene_id"),
            "question": row.get("question_text"),
            "question_text": row.get("question_text"),
            "question_type": row.get("question_type"),
            "answer_options": _request_answer_options(row.get("answer_options")),
            "answer_option_response_schema": _answer_option_response_schema(
                row.get("answer_options")
            ),
            "primary_frame": _mapping(row.get("situation")).get("view_frame"),
            "frames": [_mapping(row.get("situation")).get("view_frame")]
            if isinstance(_mapping(row.get("situation")).get("view_frame"), Mapping)
            else [],
            "situation": _public_situation(_mapping(row.get("situation"))),
            "target": _public_target(_mapping(row.get("target"))),
            "required_output_schema": {
                "answer": "string or structured current_location JSON",
                "evidence_summary": "brief text",
                "confidence": "number in [0,1]",
            },
        }
        for row in records
    ])
    bundle: dict[str, Any] = {
        "schema_version": ACTIVE_QA_V2_REQUEST_BUNDLE_SCHEMA_VERSION,
        "episode_id": episode_id,
        "request_count": len(prediction_cases),
        "prediction_cases": prediction_cases,
    }
    leak_paths = _forbidden_paths(bundle)
    bundle["leak_free"] = not leak_paths
    bundle["leak_paths"] = leak_paths
    bundle["request_bundle_digest"] = _digest(bundle)
    return bundle


def save_active_qa_v2_splits(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split_name in ACTIVE_QA_V2_SPLIT_NAMES:
        path = root / f"qa-{split_name.replace('_', '-')}.jsonl"
        path.write_text(
            "".join(
                json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
                for row in splits.get(split_name, ())
            ),
            encoding="utf-8",
        )
        paths[split_name] = str(path)
    return paths


def load_active_qa_v2_records(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _record(
    *,
    episode_id: str,
    scene_id: str,
    split: str,
    question_type: str,
    question_text: str,
    target_id: str,
    target_label: str,
    relation: str,
    dst_id: str,
    dst_label: str,
    step: int,
    situation: Mapping[str, Any],
    required_edges: Sequence[str],
    evidence_frames: Sequence[int],
    trajectory: Mapping[str, Any],
    source_graph_digest: str,
    evidence_observable: bool,
    anti_shortcut: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": ACTIVE_QA_V2_SCHEMA_VERSION,
        "id": f"{episode_id}:{scene_id}:{step:06d}:{question_type}:{target_id}:{split}",
        "scene_id": scene_id,
        "episode_id": episode_id,
        "split": split,
        "question_type": question_type,
        "question_text": question_text,
        "situation": dict(situation),
        "target": {"object_id": target_id, "label": target_label},
        "answer": {
            "relation": relation,
            "dst": dst_id,
            "dst_label": dst_label,
            "step": step,
        },
        "answer_options": _answer_options(relation, dst_label),
        "required_evidence": {
            "nodes": [target_id, dst_id],
            "edges": list(required_edges),
            "states": [],
            "frames": list(evidence_frames),
        },
        "observability": {
            "target_visible": evidence_observable,
            "support_visible": evidence_observable and dst_id != "agent",
            "evidence_observable": evidence_observable,
            "answerable_from_single_frame_vlm": evidence_observable,
            "answerable_from_dsg": evidence_observable,
        },
        "anti_shortcut": dict(anti_shortcut),
        "trajectory_context": {
            "collection_kind": trajectory.get("collection_kind"),
            "real_ai2thor_runtime": trajectory.get("real_ai2thor_runtime"),
            "navigation_validated": trajectory.get("navigation_validated"),
        },
        "evidence_frames": list(evidence_frames),
        "source_graph_digest": source_graph_digest,
    }


def _observation_index(observations: Sequence[SceneObservation]) -> dict[str, Any]:
    object_steps: dict[str, set[int]] = defaultdict(set)
    object_by_step: dict[int, dict[str, ObjectObservation]] = defaultdict(dict)
    object_pose_by_step: dict[tuple[str, int], Any] = {}
    object_state_by_step: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    label_counts: Counter[str] = Counter()
    step_pose: dict[int, Any] = {}
    step_frame: dict[int, dict[str, Any]] = {}
    seen_labels: set[tuple[str, str]] = set()
    for observation in observations:
        if observation.agent_pose is not None:
            step_pose[observation.step] = observation.agent_pose
        for obj in observation.objects:
            if observation.step not in step_frame:
                rgb_path = obj.attributes.get("rgb_path")
                if isinstance(rgb_path, str) and rgb_path:
                    step_frame[observation.step] = {
                        "frame_id": f"{observation.step:06d}",
                        "rgb_path": rgb_path,
                        "scene_id": "",
                        "step": observation.step,
                    }
            object_steps[obj.object_id].add(observation.step)
            object_by_step[observation.step][obj.object_id] = obj
            object_pose_by_step[(obj.object_id, observation.step)] = obj.pose
            state = obj.attributes.get("state")
            if isinstance(state, Mapping):
                object_state_by_step[obj.object_id].append((observation.step, state))
            key = (obj.object_id, obj.label)
            if key not in seen_labels:
                label_counts[obj.label] += 1
                seen_labels.add(key)
    return {
        "object_steps": {key: sorted(value) for key, value in object_steps.items()},
        "object_by_step": {key: dict(value) for key, value in object_by_step.items()},
        "object_pose_by_step": object_pose_by_step,
        "object_state_by_step": {
            key: sorted(value, key=lambda item: item[0])
            for key, value in object_state_by_step.items()
        },
        "label_counts": label_counts,
        "step_pose": step_pose,
        "step_frame": step_frame,
    }


def _same_frame_steps(
    object_steps: Mapping[str, Sequence[int]],
    src: str,
    dst: str,
) -> list[int]:
    return sorted(set(object_steps.get(src, ())) & set(object_steps.get(dst, ())))


def _nearest_object_records(
    *,
    episode_id: str,
    scene_id: str,
    trajectory: Mapping[str, Any],
    index: Mapping[str, Any],
    source_graph_digest: str,
    max_records: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    object_by_step = _step_object_mapping(index.get("object_by_step"))
    step_pose = _int_key_mapping(index.get("step_pose"))
    step_frame = _int_key_mapping(index.get("step_frame"))
    for step, objects in sorted(object_by_step.items()):
        if len(objects) < 2:
            continue
        for src_id, src in sorted(objects.items()):
            nearest = _nearest_other(src, objects)
            if nearest is None:
                continue
            dst_id, dst = nearest
            rows.append(
                _record(
                    episode_id=episode_id,
                    scene_id=scene_id,
                    split="relation_centric",
                    question_type="nearest_object",
                    question_text=(
                        f"Which observed object is nearest to the {src.label} at step {step}?"
                    ),
                    target_id=src_id,
                    target_label=src.label,
                    relation="NEAR",
                    dst_id=dst_id,
                    dst_label=dst.label,
                    step=step,
                    situation=_situation(
                        step_pose.get(step),
                        step,
                        step_frame.get(step),
                    ),
                    required_edges=[f"{src_id}-NEAR-{dst_id}"],
                    evidence_frames=[step],
                    trajectory=trajectory,
                    source_graph_digest=source_graph_digest,
                    evidence_observable=True,
                    anti_shortcut={
                        "language_prior_risk": "low",
                        "requires_3d_evidence": True,
                        "has_distractor_same_category": False,
                        "has_distractor_same_support": False,
                    },
                )
            )
            if len(rows) >= max_records:
                return rows
    return rows


def _multi_hop_records(
    *,
    episode_id: str,
    scene_id: str,
    trajectory: Mapping[str, Any],
    graph: DynamicSceneGraph,
    index: Mapping[str, Any],
    relation_edges: Sequence[Any],
    source_graph_digest: str,
    max_records: int,
) -> list[dict[str, Any]]:
    by_support: dict[str, list[Any]] = defaultdict(list)
    for edge in relation_edges:
        if edge.relation in SUPPORT_RELATIONS:
            by_support[edge.dst].append(edge)
    rows: list[dict[str, Any]] = []
    step_pose = _int_key_mapping(index.get("step_pose"))
    step_frame = _int_key_mapping(index.get("step_frame"))
    for support_id, edges in sorted(by_support.items()):
        if len(edges) < 2:
            continue
        support_node = graph.nodes.get(support_id)
        if support_node is None:
            continue
        first, second = sorted(edges, key=lambda edge: (edge.step, edge.src))[:2]
        first_node = graph.nodes.get(first.src)
        second_node = graph.nodes.get(second.src)
        if first_node is None or second_node is None:
            continue
        frames = sorted(
            set(_same_frame_steps(_mapping(index.get("object_steps")), first.src, support_id))
            & set(_same_frame_steps(_mapping(index.get("object_steps")), second.src, support_id))
        )
        if not frames:
            frames = [first.step]
        rows.append(
            _record(
                episode_id=episode_id,
                scene_id=scene_id,
                split="relation_centric",
                question_type="multi_hop",
                question_text=(
                    f"Which support surface links the {first_node.label or _label(first.src)} "
                    f"and the {second_node.label or _label(second.src)}?"
                ),
                target_id=first.src,
                target_label=first_node.label or _label(first.src),
                relation=first.relation,
                dst_id=support_id,
                dst_label=support_node.label or _label(support_id),
                step=first.step,
                situation=_situation(
                    step_pose.get(frames[0]),
                    frames[0],
                    step_frame.get(frames[0]),
                ),
                required_edges=[
                    f"{first.src}-{first.relation}-{support_id}",
                    f"{second.src}-{second.relation}-{support_id}",
                ],
                evidence_frames=frames,
                trajectory=trajectory,
                source_graph_digest=source_graph_digest,
                evidence_observable=True,
                anti_shortcut={
                    "language_prior_risk": "low",
                    "requires_3d_evidence": True,
                    "has_distractor_same_category": False,
                    "has_distractor_same_support": True,
                },
            )
        )
        if len(rows) >= max_records:
            break
    return rows


def _relative_relation_records_from_observations(
    *,
    episode_id: str,
    scene_id: str,
    trajectory: Mapping[str, Any],
    index: Mapping[str, Any],
    source_graph_digest: str,
    max_records: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    object_by_step = _step_object_mapping(index.get("object_by_step"))
    step_pose = _int_key_mapping(index.get("step_pose"))
    step_frame = _int_key_mapping(index.get("step_frame"))
    for step, objects in sorted(object_by_step.items()):
        object_items = sorted(objects.items())
        if len(object_items) < 2:
            continue
        for src_id, src in object_items:
            for dst_id, dst in object_items:
                if src_id == dst_id:
                    continue
                relation = _egocentric_relation(
                    src.pose,
                    dst.pose,
                    step_pose.get(step),
                )
                if relation is None:
                    continue
                rows.append(
                    _record(
                        episode_id=episode_id,
                        scene_id=scene_id,
                        split="situated",
                        question_type="relative_relation",
                        question_text=(
                            f"From the robot viewpoint at step {step}, is the "
                            f"{src.label} {relation.lower()} the {dst.label}?"
                        ),
                        target_id=src_id,
                        target_label=src.label,
                        relation=relation,
                        dst_id=dst_id,
                        dst_label=dst.label,
                        step=step,
                        situation=_situation(
                            step_pose.get(step),
                            step,
                            step_frame.get(step),
                        ),
                        required_edges=[f"{src_id}-{relation}-{dst_id}"],
                        evidence_frames=[step],
                        trajectory=trajectory,
                        source_graph_digest=source_graph_digest,
                        evidence_observable=True,
                        anti_shortcut={
                            "language_prior_risk": "low",
                            "requires_3d_evidence": True,
                            "has_distractor_same_category": False,
                            "has_distractor_same_support": False,
                        },
                    )
                )
                if len(rows) >= max_records:
                    return rows
    return rows


def _state_change_records(
    *,
    episode_id: str,
    scene_id: str,
    trajectory: Mapping[str, Any],
    index: Mapping[str, Any],
    source_graph_digest: str,
    max_records: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    object_by_step = _step_object_mapping(index.get("object_by_step"))
    object_steps = _mapping(index.get("object_steps"))
    state_by_object = _mapping(index.get("object_state_by_step"))
    step_pose = _int_key_mapping(index.get("step_pose"))
    step_frame = _int_key_mapping(index.get("step_frame"))
    for obj_id, steps_value in sorted(object_steps.items()):
        steps = [step for step in steps_value if isinstance(step, int)]
        if len(steps) < 2:
            continue
        first_step, last_step = min(steps), max(steps)
        first_obj = object_by_step.get(first_step, {}).get(obj_id)
        last_obj = object_by_step.get(last_step, {}).get(obj_id)
        if first_obj is None or last_obj is None:
            continue
        changed = _state_changed(state_by_object.get(obj_id)) or _pose_changed(first_obj, last_obj)
        row = _record(
            episode_id=episode_id,
            scene_id=scene_id,
            split="temporal",
            question_type="state_change",
            question_text=(
                f"Did the {last_obj.label} change observed state between step "
                f"{first_step} and step {last_step}?"
            ),
            target_id=obj_id,
            target_label=last_obj.label,
            relation="STATE_CHANGED",
            dst_id=obj_id,
            dst_label="changed" if changed else "unchanged",
            step=last_step,
            situation=_situation(
                step_pose.get(last_step),
                last_step,
                step_frame.get(last_step),
            ),
            required_edges=[],
            evidence_frames=[first_step, last_step],
            trajectory=trajectory,
            source_graph_digest=source_graph_digest,
            evidence_observable=True,
            anti_shortcut={
                "language_prior_risk": "low",
                "requires_3d_evidence": True,
                "has_distractor_same_category": False,
                "has_distractor_same_support": False,
            },
        )
        row["required_evidence"]["states"] = [f"{obj_id}:{first_step}", f"{obj_id}:{last_step}"]
        rows.append(row)
        if len(rows) >= max_records:
            break
    return rows


def _step_object_mapping(value: object) -> dict[int, dict[str, ObjectObservation]]:
    if not isinstance(value, Mapping):
        return {}
    return {
        step: {
            object_id: obj
            for object_id, obj in objects.items()
            if isinstance(object_id, str) and isinstance(obj, ObjectObservation)
        }
        for step, objects in value.items()
        if isinstance(step, int) and isinstance(objects, Mapping)
    }


def _int_key_mapping(value: object) -> dict[int, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, int)
    }


def _nearest_other(
    src: ObjectObservation,
    objects: Mapping[str, ObjectObservation],
) -> tuple[str, ObjectObservation] | None:
    candidates = [
        (_pose_distance(src.pose, other.pose), object_id, other)
        for object_id, other in objects.items()
        if object_id != src.object_id
    ]
    if not candidates:
        return None
    _, object_id, obj = min(candidates, key=lambda item: (item[0], item[1]))
    return object_id, obj


def _pose_distance(a: Any, b: Any) -> float:
    return math.sqrt(
        (float(getattr(a, "x", 0.0)) - float(getattr(b, "x", 0.0))) ** 2
        + (float(getattr(a, "y", 0.0)) - float(getattr(b, "y", 0.0))) ** 2
        + (float(getattr(a, "z", 0.0)) - float(getattr(b, "z", 0.0))) ** 2
    )


def _egocentric_relation(src_pose: Any, dst_pose: Any, agent_pose: Any) -> str | None:
    dx = float(getattr(src_pose, "x", 0.0)) - float(getattr(dst_pose, "x", 0.0))
    dz = float(getattr(src_pose, "z", 0.0)) - float(getattr(dst_pose, "z", 0.0))
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return None
    yaw_degrees = float(getattr(agent_pose, "yaw", 0.0)) if agent_pose is not None else 0.0
    yaw = math.radians(-yaw_degrees)
    ego_x = dx * math.cos(yaw) - dz * math.sin(yaw)
    ego_z = dx * math.sin(yaw) + dz * math.cos(yaw)
    if abs(ego_x) >= abs(ego_z):
        return "LEFT_OF" if ego_x < 0 else "RIGHT_OF"
    return "FRONT_OF" if ego_z > 0 else "BEHIND"


def _state_changed(value: object) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return False
    states = [
        dict(state)
        for _, state in value
        if isinstance(state, Mapping)
    ]
    return len({json.dumps(state, sort_keys=True) for state in states}) > 1


def _pose_changed(first: ObjectObservation, last: ObjectObservation) -> bool:
    return _pose_distance(first.pose, last.pose) > 0.01


def _situation(
    pose: Any,
    step: int,
    view_frame: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "step": step,
        "reference_frame": "agent_egocentric",
        "agent_pose": pose.to_dict() if hasattr(pose, "to_dict") else None,
        "view_frame": dict(view_frame) if view_frame is not None else None,
        "source": "reachable_nbv_observation",
    }


def _anti_shortcut(
    *,
    src_label: str,
    dst_label: str,
    distractor_count: int,
    relation: str,
) -> dict[str, Any]:
    common_pairs = {("pillow", "bed"), ("towel", "bathroom"), ("book", "shelf")}
    high_prior = (src_label, dst_label) in common_pairs or relation == "IN_ROOM"
    return {
        "language_prior_risk": "high" if high_prior else "medium",
        "requires_3d_evidence": True,
        "has_distractor_same_category": distractor_count > 0,
        "has_distractor_same_support": relation in SUPPORT_RELATIONS,
    }


def _answer_options(relation: str, dst_label: str) -> list[dict[str, str]]:
    options = [{"relation": relation, "dst_label": dst_label}]
    for candidate in ("countertop", "table", "chair", "cabinet", "floor", "room"):
        if candidate != dst_label:
            options.append({"relation": relation, "dst_label": candidate})
        if len(options) == 4:
            break
    return options


def _append(
    records: dict[str, list[dict[str, Any]]],
    split: str,
    row: dict[str, Any],
    limit: int,
) -> None:
    normalized = {**row, "split": split}
    row_id = normalized.get("id")
    existing_ids = {
        existing.get("id")
        for existing in records[split]
        if isinstance(existing.get("id"), str)
    }
    if isinstance(row_id, str):
        if row_id in existing_ids:
            return
        if len(existing_ids) < limit:
            records[split].append(normalized)
            return
        row_type = normalized.get("question_type")
        existing_types = [
            str(existing.get("question_type"))
            for existing in records[split]
            if isinstance(existing.get("question_type"), str)
        ]
        if isinstance(row_type, str) and row_type not in existing_types:
            type_counts = Counter(existing_types)
            replace_type = max(
                sorted(type_counts),
                key=lambda item: (type_counts[item], item),
            )
            replace_indices = [
                index
                for index, existing in enumerate(records[split])
                if existing.get("question_type") == replace_type
            ]
            if replace_indices:
                records[split][replace_indices[-1]] = normalized
        return
    if len(records[split]) < limit:
        records[split].append(normalized)


def _dedupe_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["id"]: row for row in rows}
    return [by_id[key] for key in sorted(by_id)]


def _public_situation(situation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "step": situation.get("step"),
        "reference_frame": situation.get("reference_frame"),
        "agent_pose": situation.get("agent_pose"),
        "view_frame": situation.get("view_frame"),
    }


def _public_target(target: Mapping[str, Any]) -> dict[str, Any] | None:
    label = target.get("label")
    if not isinstance(label, str) or label == "":
        return None
    return {"label": label}


def _request_answer_options(value: object) -> list[dict[str, Any]]:
    options = []
    for index, item in enumerate(_mapping_sequence(value), start=1):
        relation = item.get("relation")
        dst_label = item.get("dst_label") or item.get("destination_label")
        if not isinstance(relation, str) or not isinstance(dst_label, str):
            continue
        options.append(
            {
                "option_id": f"option_{index}",
                "relation": relation,
                "destination_label": dst_label,
            }
        )
    return options


def _answer_option_response_schema(value: object) -> dict[str, Any] | None:
    option_ids = [
        option["option_id"]
        for option in _request_answer_options(value)
        if isinstance(option.get("option_id"), str)
    ]
    if not option_ids:
        return None
    return {
        "allowed_answer_option_ids": option_ids,
        "answer_option_id_field": "answer.answer_option_id",
        "answer_current_location_rule": (
            "Copy relation and destination_label from the selected answer option."
        ),
        "required_when_answer_options_present": True,
    }


def _label(object_id: str) -> str:
    return object_id.split("_", 1)[0]


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_sequence(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _dedupe_prediction_cases(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {
        str(case.get("case_id")): dict(case)
        for case in cases
        if isinstance(case.get("case_id"), str)
    }
    return [by_id[case_id] for case_id in sorted(by_id)]


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(float(numerator) / float(denominator), 6)


def _digest(payload: Mapping[str, Any]) -> str:
    normalized = {key: value for key, value in payload.items() if not key.endswith("_digest")}
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _forbidden_paths(value: object, *, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_prefix = f"{prefix}.{key_text}"
            if key_text in FORBIDDEN_REQUEST_KEYS:
                paths.append(child_prefix)
            paths.extend(_forbidden_paths(child, prefix=child_prefix))
    elif isinstance(value, Sequence) and not isinstance(value, str):
        for index, child in enumerate(value):
            paths.extend(_forbidden_paths(child, prefix=f"{prefix}[{index}]"))
    return paths
