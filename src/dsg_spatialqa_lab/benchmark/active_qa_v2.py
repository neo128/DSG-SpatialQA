from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab.memory import CONTAINMENT_RELATIONS, DynamicSceneGraph
from dsg_spatialqa_lab.observations import SceneObservation
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
            situation=_situation(index["step_pose"].get(evidence_frames[0]), evidence_frames[0]),
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
            situation=_situation(last_pose, last_step),
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
    prediction_cases = [
        {
            "case_id": row.get("id"),
            "episode_id": row.get("episode_id"),
            "scene_id": row.get("scene_id"),
            "question": row.get("question_text"),
            "question_type": row.get("question_type"),
            "answer_options": row.get("answer_options"),
            "situation": _public_situation(_mapping(row.get("situation"))),
            "required_output_schema": {
                "answer": "string or structured current_location JSON",
                "evidence_summary": "brief text",
                "confidence": "number in [0,1]",
            },
        }
        for row in records
    ]
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
    label_counts: Counter[str] = Counter()
    step_pose: dict[int, Any] = {}
    seen_labels: set[tuple[str, str]] = set()
    for observation in observations:
        if observation.agent_pose is not None:
            step_pose[observation.step] = observation.agent_pose
        for obj in observation.objects:
            object_steps[obj.object_id].add(observation.step)
            key = (obj.object_id, obj.label)
            if key not in seen_labels:
                label_counts[obj.label] += 1
                seen_labels.add(key)
    return {
        "object_steps": {key: sorted(value) for key, value in object_steps.items()},
        "label_counts": label_counts,
        "step_pose": step_pose,
    }


def _same_frame_steps(
    object_steps: Mapping[str, Sequence[int]],
    src: str,
    dst: str,
) -> list[int]:
    return sorted(set(object_steps.get(src, ())) & set(object_steps.get(dst, ())))


def _situation(pose: Any, step: int) -> dict[str, Any]:
    return {
        "step": step,
        "reference_frame": "agent_egocentric",
        "agent_pose": pose.to_dict() if hasattr(pose, "to_dict") else None,
        "view_frame": None,
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
    if len(records[split]) < limit:
        records[split].append({**row, "split": split})


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


def _label(object_id: str) -> str:
    return object_id.split("_", 1)[0]


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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
