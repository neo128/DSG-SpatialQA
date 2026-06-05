from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Literal

from dsg_spatialqa_lab.schema import Pose3D, SpatialQAError


ScoringMode = Literal["predicted_memory_only", "diagnostic_oracle_assisted"]
FORBIDDEN_FORMAL_FIELDS = frozenset(
    {"gold_answer", "gold_evidence", "required_edges", "required_nodes"}
)


@dataclass(frozen=True)
class CoverageMemory:
    observed_object_ids: frozenset[str] = frozenset()
    observed_support_ids: frozenset[str] = frozenset()
    same_frame_relation_keys: frozenset[str] = frozenset()
    current_location_edge_keys: frozenset[str] = frozenset()
    state_evidence_object_ids: frozenset[str] = frozenset()
    visited_view_keys: frozenset[str] = frozenset()
    visited_position_keys: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_object_count": len(self.observed_object_ids),
            "observed_support_count": len(self.observed_support_ids),
            "same_frame_relation_count": len(self.same_frame_relation_keys),
            "current_location_edge_count": len(self.current_location_edge_keys),
            "state_evidence_count": len(self.state_evidence_object_ids),
            "visited_view_count": len(self.visited_view_keys),
            "visited_position_count": len(self.visited_position_keys),
        }

    def update(
        self,
        *,
        object_ids: Sequence[str] = (),
        support_ids: Sequence[str] = (),
        same_frame_relations: Sequence[str] = (),
        current_location_edges: Sequence[str] = (),
        state_object_ids: Sequence[str] = (),
        view_key: str | None = None,
    ) -> CoverageMemory:
        visited = set(self.visited_view_keys)
        visited_positions = set(self.visited_position_keys)
        if view_key is not None:
            visited.add(view_key)
            visited_positions.add(":".join(view_key.split(":")[:2]))
        return CoverageMemory(
            observed_object_ids=self.observed_object_ids | frozenset(object_ids),
            observed_support_ids=self.observed_support_ids | frozenset(support_ids),
            same_frame_relation_keys=(
                self.same_frame_relation_keys | frozenset(same_frame_relations)
            ),
            current_location_edge_keys=(
                self.current_location_edge_keys | frozenset(current_location_edges)
            ),
            state_evidence_object_ids=(
                self.state_evidence_object_ids | frozenset(state_object_ids)
            ),
            visited_view_keys=frozenset(visited),
            visited_position_keys=frozenset(visited_positions),
        )


@dataclass(frozen=True)
class ViewpointCandidate:
    candidate_id: str
    pose: Pose3D
    pitch: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def position_key(self) -> str:
        return f"{self.pose.x:.2f}:{self.pose.z:.2f}"

    @property
    def view_key(self) -> str:
        return f"{self.pose.x:.2f}:{self.pose.z:.2f}:{self.pose.yaw:.0f}:{self.pitch:.0f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "pose": self.pose.to_dict(),
            "pitch": self.pitch,
            "view_key": self.view_key,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ViewpointScore:
    candidate_id: str
    score: float
    terms: dict[str, float]
    why_selected: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "score": self.score,
            "score_terms": self.terms,
            "why_selected": self.why_selected,
        }


def viewpoint_score(
    candidate: ViewpointCandidate,
    memory: CoverageMemory,
    *,
    expected: Mapping[str, Any],
    scoring_mode: ScoringMode = "predicted_memory_only",
    weights: Mapping[str, float] | None = None,
) -> ViewpointScore:
    if scoring_mode == "predicted_memory_only":
        forbidden = _forbidden_paths(expected)
        if forbidden:
            raise SpatialQAError(f"formal NBV expected payload contains forbidden gold field: {forbidden[0]}")
    if scoring_mode not in {"predicted_memory_only", "diagnostic_oracle_assisted"}:
        raise SpatialQAError(f"Unsupported scoring mode: {scoring_mode}")
    merged_weights = _default_weights()
    merged_weights.update(dict(weights or {}))
    terms = _score_terms(candidate, memory, expected)
    score = (
        merged_weights["new_object_gain"] * terms["new_object_gain"]
        + merged_weights["target_recall_gain"] * terms["target_recall_gain"]
        + merged_weights["support_recall_gain"] * terms["support_recall_gain"]
        + merged_weights["target_support_same_frame_gain"]
        * terms["target_support_same_frame_gain"]
        + merged_weights["current_location_edge_gain"]
        * terms["current_location_edge_gain"]
        + merged_weights["state_evidence_gain"] * terms["state_evidence_gain"]
        + merged_weights["unseen_region_gain"] * terms["unseen_region_gain"]
        + merged_weights["bbox_depth_quality"] * terms["bbox_depth_quality"]
        + merged_weights["support_surface_gap_gain"]
        * terms["support_surface_gap_gain"]
        - merged_weights["travel_cost"] * terms["travel_cost"]
        - merged_weights["repeated_view_penalty"] * terms["repeated_view_penalty"]
        - merged_weights["position_revisit_penalty"]
        * terms["position_revisit_penalty"]
        - merged_weights["occlusion_risk"] * terms["occlusion_risk"]
        - merged_weights["oracle_target_prior_penalty"]
        * terms["oracle_target_prior_penalty"]
    )
    return ViewpointScore(
        candidate_id=candidate.candidate_id,
        score=round(score, 6),
        terms={key: round(value, 6) for key, value in sorted(terms.items())},
        why_selected="highest relation-centric gain per travel cost",
    )


def _score_terms(
    candidate: ViewpointCandidate,
    memory: CoverageMemory,
    expected: Mapping[str, Any],
) -> dict[str, float]:
    new_object_ids = _string_set(expected.get("new_object_ids"))
    target_object_ids = _string_set(expected.get("target_object_ids"))
    support_object_ids = _string_set(expected.get("support_object_ids"))
    same_frame_relations = _string_set(expected.get("same_frame_relations"))
    current_location_edges = _string_set(expected.get("current_location_edges"))
    state_object_ids = _string_set(expected.get("state_object_ids"))
    unseen_region_ids = _string_set(expected.get("unseen_region_ids"))
    return {
        "new_object_gain": _gain(new_object_ids, memory.observed_object_ids),
        "target_recall_gain": _gain(target_object_ids, memory.observed_object_ids),
        "support_recall_gain": _gain(support_object_ids, memory.observed_support_ids),
        "target_support_same_frame_gain": _gain(
            same_frame_relations,
            memory.same_frame_relation_keys,
        ),
        "current_location_edge_gain": _gain(
            current_location_edges,
            memory.current_location_edge_keys,
        ),
        "state_evidence_gain": _gain(
            state_object_ids,
            memory.state_evidence_object_ids,
        ),
        "unseen_region_gain": min(1.0, float(len(unseen_region_ids)) / 3.0),
        "bbox_depth_quality": _float_or_default(expected.get("bbox_depth_quality"), 0.0),
        "travel_cost": _float_or_default(
            expected.get("travel_cost"),
            math.sqrt(candidate.pose.x**2 + candidate.pose.z**2) / 10.0,
        ),
        "repeated_view_penalty": 1.0 if candidate.view_key in memory.visited_view_keys else 0.0,
        "occlusion_risk": _float_or_default(expected.get("occlusion_risk"), 0.0),
        "oracle_target_prior_penalty": _float_or_default(
            expected.get("oracle_target_prior_penalty"),
            0.0,
        ),
        "support_surface_gap_gain": _gain(
            _string_set(expected.get("support_surface_gap_ids")),
            memory.observed_support_ids,
        ),
        "position_revisit_penalty": 1.0
        if candidate.position_key in memory.visited_position_keys
        else 0.0,
    }


def _default_weights() -> dict[str, float]:
    return {
        "new_object_gain": 0.4,
        "target_recall_gain": 0.6,
        "support_recall_gain": 1.0,
        "target_support_same_frame_gain": 1.8,
        "current_location_edge_gain": 1.5,
        "state_evidence_gain": 0.5,
        "unseen_region_gain": 0.3,
        "bbox_depth_quality": 0.3,
        "travel_cost": 0.25,
        "repeated_view_penalty": 0.8,
        "occlusion_risk": 0.4,
        "oracle_target_prior_penalty": 10.0,
        "support_surface_gap_gain": 2.2,
        "position_revisit_penalty": 1.1,
    }


def _gain(items: frozenset[str], seen: frozenset[str]) -> float:
    if not items:
        return 0.0
    unseen = items - seen
    return float(len(unseen)) / float(len(items))


def _string_set(value: object) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset({value})
    if not isinstance(value, Sequence):
        return frozenset()
    return frozenset(str(item) for item in value if isinstance(item, str))


def _float_or_default(value: object, default: float) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return default


def _forbidden_paths(value: object, *, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_prefix = f"{prefix}.{key_text}"
            if key_text in FORBIDDEN_FORMAL_FIELDS:
                paths.append(child_prefix)
            paths.extend(_forbidden_paths(child, prefix=child_prefix))
    elif isinstance(value, Sequence) and not isinstance(value, str):
        for index, child in enumerate(value):
            paths.extend(_forbidden_paths(child, prefix=f"{prefix}[{index}]"))
    return paths
