from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.navigation.action_planner import (
    ReachablePosition,
    plan_reachable_path,
    reject_unreachable_candidates,
)
from dsg_spatialqa_lab.navigation.viewpoint_scoring import (
    CoverageMemory,
    ViewpointCandidate,
    viewpoint_score,
)
from dsg_spatialqa_lab.observations import (
    NodeObservation,
    ObjectObservation,
    SceneObservation,
)
from dsg_spatialqa_lab.predicted import build_predicted_graph_from_observations
from dsg_spatialqa_lab.schema import BBox3D, Pose3D, SpatialQAError


REACHABLE_NBV_TRAJECTORY_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.reachable-nbv-trajectory.v1"
)


@dataclass(frozen=True)
class CandidateObservation:
    object_ids: frozenset[str] = frozenset()
    support_ids: frozenset[str] = frozenset()
    same_frame_relations: frozenset[str] = frozenset()
    current_location_edges: frozenset[str] = frozenset()
    state_object_ids: frozenset[str] = frozenset()
    unseen_region_ids: frozenset[str] = frozenset()
    object_labels: Mapping[str, str] | None = None
    object_locations: Mapping[str, str] | None = None

    def expected_payload(self, *, travel_cost: float = 0.0) -> dict[str, Any]:
        object_ids = sorted(self.object_ids)
        return {
            "new_object_ids": object_ids,
            "target_object_ids": object_ids,
            "support_object_ids": sorted(self.support_ids),
            "support_surface_gap_ids": sorted(self.support_ids),
            "same_frame_relations": sorted(self.same_frame_relations),
            "current_location_edges": sorted(self.current_location_edges),
            "state_object_ids": sorted(self.state_object_ids),
            "unseen_region_ids": sorted(self.unseen_region_ids),
            "bbox_depth_quality": 1.0 if object_ids else 0.0,
            "travel_cost": travel_cost,
        }


@dataclass(frozen=True)
class ExecutedViewpoint:
    executed_actions: tuple[Mapping[str, Any], ...]
    executed_capture_actions: tuple[Mapping[str, Any], ...] = ()
    observations: tuple[SceneObservation, ...] = ()
    observed_candidate: CandidateObservation = field(default_factory=CandidateObservation)
    agent_pose_after: Pose3D | None = None


@dataclass(frozen=True)
class ReachableNBVResult:
    trajectory: dict[str, Any]
    decisions: tuple[dict[str, Any], ...]
    observations: tuple[SceneObservation, ...]
    graph: DynamicSceneGraph

    @property
    def observation_count(self) -> int:
        return len(self.observations)


def reachable_relation_centric_nbv(
    *,
    scene_id: str,
    episode_id: str,
    reachable_positions: Sequence[ReachablePosition],
    start_pose: Pose3D,
    candidate_observations: Mapping[str, CandidateObservation],
    max_iterations: int,
    yaw_sweep_degrees: Sequence[float],
    pitch_sweep_degrees: Sequence[float],
    scoring_mode: str = "predicted_memory_only",
    runtime_kind: str = "fake_controller",
    real_ai2thor_runtime: bool = False,
    grid_step: float = 0.25,
    execute_viewpoint: Callable[
        [
            tuple[str, ...],
            ViewpointCandidate,
            Sequence[float],
            Sequence[float],
            int,
        ],
        ExecutedViewpoint,
    ]
    | None = None,
) -> ReachableNBVResult:
    if max_iterations <= 0:
        raise SpatialQAError("max_iterations must be positive")
    memory = CoverageMemory()
    current_pose = start_pose
    trajectory_steps: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    observations: list[SceneObservation] = []
    rejected_candidates: list[dict[str, str]] = []

    for iteration in range(max_iterations):
        candidates = _candidate_viewpoints(
            reachable_positions,
            yaw_sweep_degrees=yaw_sweep_degrees,
            pitch_sweep_degrees=pitch_sweep_degrees,
        )
        candidates, rejected = reject_unreachable_candidates(candidates, reachable_positions)
        rejected_candidates.extend(rejected)
        scored = []
        for candidate in candidates:
            observation = candidate_observations.get(candidate.position_key)
            if observation is None:
                continue
            path = plan_reachable_path(
                reachable_positions,
                current_pose,
                candidate.pose,
                grid_step=grid_step,
            )
            if not path.reachable:
                rejected_candidates.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "reason": path.rejected_reason or "unreachable",
                    }
                )
                continue
            score = viewpoint_score(
                candidate,
                memory,
                expected=observation.expected_payload(
                    travel_cost=max(0.0, float(len(path.actions)) / 20.0)
                ),
                scoring_mode=scoring_mode,  # type: ignore[arg-type]
            )
            scored.append((score.score, score, candidate, observation, path))
        if not scored:
            break
        _, selected_score, selected, selected_observation, selected_path = sorted(
            scored,
            key=lambda item: (-item[0], item[2].candidate_id),
        )[0]
        memory_before = memory
        capture_actions = _capture_actions(yaw_sweep_degrees, pitch_sweep_degrees)
        step_index = len(trajectory_steps)
        if execute_viewpoint is None:
            executed = ExecutedViewpoint(
                executed_actions=tuple(
                    {"action": action, "success": True}
                    for action in selected_path.actions
                ),
                observations=_capture_observations(
                    scene_id=scene_id,
                    episode_id=episode_id,
                    agent_pose=selected.pose,
                    selected=selected_observation,
                    step_base=1000 + step_index * 100,
                    yaw_sweep_degrees=yaw_sweep_degrees,
                    pitch_sweep_degrees=pitch_sweep_degrees,
                ),
                observed_candidate=selected_observation,
                agent_pose_after=selected.pose,
            )
        else:
            executed = execute_viewpoint(
                selected_path.actions,
                selected,
                yaw_sweep_degrees,
                pitch_sweep_degrees,
                step_index,
            )
        observations.extend(executed.observations)
        observed_candidate = executed.observed_candidate
        memory = memory.update(
            object_ids=sorted(observed_candidate.object_ids),
            support_ids=sorted(observed_candidate.support_ids),
            same_frame_relations=sorted(observed_candidate.same_frame_relations),
            current_location_edges=sorted(observed_candidate.current_location_edges),
            state_object_ids=sorted(observed_candidate.state_object_ids),
            view_key=selected.view_key,
        )
        executed_actions = [dict(action) for action in executed.executed_actions]
        executed_capture_actions = [
            dict(action) for action in executed.executed_capture_actions
        ]
        agent_pose_after = executed.agent_pose_after or selected.pose
        navigation_success = all(
            action.get("success") is True
            for action in (*executed_actions, *executed_capture_actions)
        )
        trajectory_steps.append(
            {
                "step_index": step_index,
                "agent_pose_before": current_pose.to_dict(),
                "selected_viewpoint": {
                    "candidate_id": selected.candidate_id,
                    "x": selected.pose.x,
                    "y": selected.pose.y,
                    "z": selected.pose.z,
                    "yaw": selected.pose.yaw,
                    "pitch": selected.pitch,
                },
                "path_to_reach": selected_path.to_dict(),
                "planned_actions": list(selected_path.actions),
                "executed_actions": executed_actions,
                "capture_actions": capture_actions,
                "executed_capture_actions": executed_capture_actions,
                "agent_pose_after": agent_pose_after.to_dict(),
                "navigation_success": navigation_success,
            }
        )
        decisions.append(
            {
                "iteration": iteration,
                "candidate_count": len(candidates),
                "selected_candidate_id": selected.candidate_id,
                "score_terms": selected_score.terms,
                "selected_score": selected_score.score,
                "why_selected": selected_score.why_selected,
                "memory_before": memory_before.to_dict(),
                "memory_after": memory.to_dict(),
            }
        )
        current_pose = agent_pose_after

    graph = build_predicted_graph_from_observations(
        observations,
        source_path="reachable_relation_centric_nbv",
        infer_relations=("LEFT_OF", "RIGHT_OF", "NEAR"),
        reference_frames=("world",),
        infer_containment=True,
        containment_axis="y",
        relation_top_k=2,
        require_detector_state_evidence=False,
    )
    all_executed_actions = [
        action
        for step in trajectory_steps
        for action in (
            *step["executed_actions"],
            *step["executed_capture_actions"],
        )
    ]
    all_actions_success_checked = all(
        action.get("lastActionSuccess") in {True, False}
        for action in all_executed_actions
    )
    if runtime_kind == "fake_controller":
        all_actions_success_checked = all(
            action.get("success") in {True, False}
            for action in all_executed_actions
        )
    trajectory = {
        "schema_version": REACHABLE_NBV_TRAJECTORY_SCHEMA_VERSION,
        "trajectory_id": f"reachable_nbv_{episode_id}_v1",
        "scene_id": scene_id,
        "episode_id": episode_id,
        "collection_kind": "reachable_relation_centric_nbv",
        "autonomous_exploration": True,
        "navigation_validated": bool(trajectory_steps)
        and all(step["navigation_success"] is True for step in trajectory_steps),
        "closed_loop_memory_update": True,
        "runtime_kind": runtime_kind,
        "real_ai2thor_runtime": real_ai2thor_runtime,
        "scoring_mode": scoring_mode,
        "teleport_used": any(
            action.get("action") == "TeleportFull"
            for action in all_executed_actions
        ),
        "uses_gold_answer": False,
        "uses_gold_evidence": False,
        "uses_required_edges": False,
        "uses_required_nodes": False,
        "all_actions_last_action_success_checked": all_actions_success_checked,
        "stations": [
            {
                "step_index": step["step_index"],
                "selected_viewpoint": step["selected_viewpoint"],
                "agent_pose_after": step["agent_pose_after"],
                "navigation_success": step["navigation_success"],
            }
            for step in trajectory_steps
        ],
        "rejected_candidates": rejected_candidates,
        "steps": trajectory_steps,
    }
    return ReachableNBVResult(
        trajectory=trajectory,
        decisions=tuple(decisions),
        observations=tuple(observations),
        graph=graph,
    )


def _candidate_viewpoints(
    reachable_positions: Sequence[ReachablePosition],
    *,
    yaw_sweep_degrees: Sequence[float],
    pitch_sweep_degrees: Sequence[float],
) -> tuple[ViewpointCandidate, ...]:
    candidates: list[ViewpointCandidate] = []
    for position in sorted(reachable_positions, key=lambda item: item.key):
        for yaw in yaw_sweep_degrees:
            for pitch in pitch_sweep_degrees:
                pose = position.to_pose(yaw=float(yaw))
                candidates.append(
                    ViewpointCandidate(
                        f"vp_{position.x:.2f}_{position.z:.2f}_{float(yaw):.0f}_{float(pitch):.0f}",
                        pose,
                        pitch=float(pitch),
                    )
                )
    return tuple(candidates)


def _capture_actions(
    yaw_sweep_degrees: Sequence[float],
    pitch_sweep_degrees: Sequence[float],
) -> list[str]:
    actions: list[str] = []
    if len(yaw_sweep_degrees) > 1:
        actions.extend(["RotateRight"] * (len(yaw_sweep_degrees) - 1))
    if any(pitch < 0 for pitch in pitch_sweep_degrees):
        actions.append("LookDown")
    if any(pitch > 0 for pitch in pitch_sweep_degrees):
        actions.append("LookUp")
    return actions


def _capture_observations(
    *,
    scene_id: str,
    episode_id: str,
    agent_pose: Pose3D,
    selected: CandidateObservation,
    step_base: int,
    yaw_sweep_degrees: Sequence[float],
    pitch_sweep_degrees: Sequence[float],
) -> tuple[SceneObservation, ...]:
    observations: list[SceneObservation] = []
    labels = dict(selected.object_labels or {})
    object_locations = dict(selected.object_locations or {})
    for yaw_index, yaw in enumerate(yaw_sweep_degrees):
        for pitch_index, _pitch in enumerate(pitch_sweep_degrees):
            step = step_base + yaw_index * 10 + pitch_index
            pose = Pose3D(agent_pose.x, agent_pose.y, agent_pose.z, yaw=float(yaw))
            objects = tuple(
                _object_observation(
                    object_id=object_id,
                    label=labels.get(object_id, _label_from_object_id(object_id)),
                    support_id=object_locations.get(object_id),
                    step=step,
                    pose=pose,
                    is_support=object_id in selected.support_ids,
                )
                for object_id in sorted(selected.object_ids | selected.support_ids)
            )
            observations.append(
                SceneObservation(
                    step=step,
                    agent_pose=pose,
                    rooms=(
                        NodeObservation(
                            "ai2thor_room",
                            "kitchen",
                            {"source_kind": "detector", "source_name": "reachable_nbv_fake"},
                        ),
                    ),
                    regions=(
                        NodeObservation(
                            "visible_region",
                            "visible frame",
                            {
                                "room_id": "ai2thor_room",
                                "source_kind": "detector",
                                "source_name": "reachable_nbv_fake",
                            },
                        ),
                    ),
                    objects=objects,
                )
            )
    return tuple(observations)


def _object_observation(
    *,
    object_id: str,
    label: str,
    support_id: str | None,
    step: int,
    pose: Pose3D,
    is_support: bool,
) -> ObjectObservation:
    offset = float((sum(ord(ch) for ch in object_id) % 7) - 3) * 0.03
    object_pose = Pose3D(pose.x + offset, pose.y + (0.0 if is_support else 0.2), pose.z - offset)
    size = (0.7, 0.12, 0.7) if is_support else (0.12, 0.12, 0.12)
    attributes: dict[str, Any] = {
        "source_kind": "detector",
        "source_name": "reachable_nbv_fake",
        "detector": "reachable_nbv_fake",
        "evidence_kinds": ["depth", "detector", "rgb"],
        "rgb_path": f"fake/reachable-nbv/rgb/{step:04d}.ppm",
        "depth_path": f"fake/reachable-nbv/depth/{step:04d}.npy",
        "mask_path": f"fake/reachable-nbv/masks/{step:04d}_{object_id}.ppm",
        "states": {"visible": True},
    }
    if support_id is not None:
        attributes.update(
            {
                "current_location_id": support_id,
                "current_location_label": _label_from_object_id(support_id),
                "current_location_relation": "ON",
            }
        )
    return ObjectObservation(
        object_id=object_id,
        label=label,
        pose=object_pose,
        bbox=BBox3D(object_pose, size),
        confidence=0.9,
        visible=True,
        attributes=attributes,
    )


def _label_from_object_id(object_id: str) -> str:
    return object_id.split("_", 1)[0]
