from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from dsg_spatialqa_lab.schema import (
    AgentPoseState,
    BBox3D,
    Edge,
    Node,
    NodeType,
    ObjectState,
    Pose3D,
    SpatialQAError,
)


VALID_NODE_TYPES: set[str] = {"agent", "object", "region", "room", "state", "action", "event"}
VALID_RELATIONS: set[str] = {
    "IN_ROOM",
    "IN_REGION",
    "ON",
    "INSIDE",
    "SUPPORTS",
    "LEFT_OF",
    "RIGHT_OF",
    "FRONT_OF",
    "BEHIND",
    "NEAR",
    "DISTANCE",
    "DISTANCE_LT",
    "ABOVE",
    "VISIBLE_FROM",
    "REACHABLE_FROM",
    "OCCLUDES",
    "MOVED_FROM",
    "MOVED_TO",
    "STATE_CHANGED",
    "ACTION_CAUSED",
}
CONTAINMENT_RELATIONS: set[str] = {"IN_ROOM", "IN_REGION", "ON", "INSIDE"}


class DynamicSceneGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.object_states: dict[str, ObjectState] = {}
        self.agent_poses: dict[str, Pose3D] = {}
        self.agent_pose_history: dict[str, list[AgentPoseState]] = defaultdict(list)
        self.object_state_history: dict[str, list[ObjectState]] = defaultdict(list)

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        label: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Node:
        if node_type not in VALID_NODE_TYPES:
            raise SpatialQAError(f"Unsupported node type: {node_type}")
        node = Node(
            id=node_id,
            type=node_type,
            label=label,
            attributes=dict(attributes or {}),
        )
        self.nodes[node_id] = node
        return node

    def add_room(
        self,
        room_id: str,
        label: str,
        *,
        step: int,
        attributes: dict[str, Any] | None = None,
    ) -> Node:
        merged = {"step": step}
        merged.update(attributes or {})
        return self.add_node(room_id, "room", label, merged)

    def add_region(
        self,
        region_id: str,
        label: str,
        *,
        step: int,
        attributes: dict[str, Any] | None = None,
    ) -> Node:
        merged = {"step": step}
        merged.update(attributes or {})
        return self.add_node(region_id, "region", label, merged)

    def set_agent_pose(self, agent_id: str, pose: Pose3D, *, step: int) -> AgentPoseState:
        state = AgentPoseState(agent_id=agent_id, pose=pose, step=step)
        self.agent_poses[agent_id] = pose
        self.agent_pose_history[agent_id].append(state)
        self.add_node(
            agent_id,
            "agent",
            label=agent_id,
            attributes={"pose": pose.to_dict(), "step": step},
        )
        self._record_agent_pose_change(state)
        return state

    def get_agent_pose(self, agent_id: str = "agent") -> Pose3D:
        pose = self.agent_poses.get(agent_id)
        if pose is None:
            raise SpatialQAError(f"Agent pose not found: {agent_id}")
        return pose

    def get_agent_pose_history(self, agent_id: str = "agent") -> list[AgentPoseState]:
        history = self.agent_pose_history.get(agent_id)
        if not history:
            raise SpatialQAError(f"Agent pose history not found: {agent_id}")
        return sorted(history, key=lambda state: (state.step, state.agent_id))

    def upsert_object(
        self,
        object_id: str,
        label: str,
        pose: Pose3D,
        bbox: BBox3D,
        *,
        confidence: float,
        visible: bool,
        step: int,
        attributes: dict[str, Any] | None = None,
    ) -> ObjectState:
        previous = self.object_states.get(object_id)
        last_seen_step = step if visible else (previous.last_seen_step if previous else None)
        last_seen_pose = pose if visible else (previous.last_seen_pose if previous else None)
        state = ObjectState(
            object_id=object_id,
            label=label,
            pose=pose,
            bbox=bbox.with_center(pose),
            confidence=confidence,
            visible=visible,
            step=step,
            last_seen_step=last_seen_step,
            last_seen_pose=last_seen_pose,
        )
        self.object_states[object_id] = state
        self.object_state_history[object_id].append(state)

        node_attributes: dict[str, Any] = {
            "pose": pose.to_dict(),
            "confidence": confidence,
            "visible": visible,
            "step": step,
        }
        node_attributes.update(attributes or {})
        self.add_node(object_id, "object", label=label, attributes=node_attributes)
        self._record_state_change(state)
        return state

    def get_object_state(self, object_id: str) -> ObjectState:
        state = self.object_states.get(object_id)
        if state is None:
            raise SpatialQAError(f"Object not found: {object_id}")
        return state

    def add_edge(
        self,
        src: str,
        relation: str,
        dst: str,
        reference_frame: str,
        confidence: float,
        *,
        step: int,
        evidence: Iterable[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Edge:
        normalized = relation.upper()
        if normalized not in VALID_RELATIONS:
            raise SpatialQAError(f"Unsupported relation: {relation}")
        edge = Edge(
            src=src,
            relation=normalized,
            dst=dst,
            reference_frame=reference_frame,
            confidence=confidence,
            step=step,
            evidence=list(evidence or []),
            attributes=dict(attributes or {}),
        )
        self.edges.append(edge)
        return edge

    def find_edges(
        self,
        src: str | None = None,
        relation: str | None = None,
        dst: str | None = None,
        reference_frame: str | None = None,
    ) -> list[Edge]:
        normalized = relation.upper() if relation else None
        matches = [
            edge
            for edge in self.edges
            if (src is None or edge.src == src)
            and (normalized is None or edge.relation == normalized)
            and (dst is None or edge.dst == dst)
            and (reference_frame is None or edge.reference_frame == reference_frame)
        ]
        return sorted(matches, key=self._edge_sort_key)

    def history(self, object_id: str) -> list[Edge]:
        self.get_object_state(object_id)
        matches = [
            edge
            for edge in self.edges
            if edge.src == object_id
            or edge.dst == object_id
            or edge.dst.startswith(f"state:{object_id}:")
        ]
        return sorted(matches, key=self._edge_sort_key)

    def move_object(
        self,
        object_id: str,
        *,
        new_pose: Pose3D,
        new_bbox: BBox3D,
        destination_id: str,
        destination_relation: str,
        step: int,
        confidence: float | None = None,
        visible: bool = True,
        action_id: str | None = None,
        event_id: str | None = None,
        evidence: Iterable[str] | None = None,
    ) -> ObjectState:
        previous = self.get_object_state(object_id)
        previous_parent = self._latest_parent_edge(object_id)
        effective_confidence = previous.confidence if confidence is None else confidence
        effective_event_id = event_id or f"event:{object_id}:move:{step}"
        effective_evidence = list(evidence or []) + [effective_event_id]

        if action_id is not None:
            self.add_node(
                action_id,
                "action",
                label="move",
                attributes={"step": step, "object_id": object_id},
            )
        self.add_node(
            effective_event_id,
            "event",
            label="move_object",
            attributes={"step": step, "object_id": object_id},
        )
        if action_id is not None:
            self.add_edge(
                action_id,
                "ACTION_CAUSED",
                effective_event_id,
                "world",
                1.0,
                step=step,
                evidence=effective_evidence,
            )

        state = self.upsert_object(
            object_id,
            previous.label,
            new_pose,
            new_bbox,
            confidence=effective_confidence,
            visible=visible,
            step=step,
        )
        if previous_parent is not None:
            self.add_edge(
                object_id,
                "MOVED_FROM",
                previous_parent.dst,
                previous_parent.reference_frame,
                previous_parent.confidence,
                step=step,
                evidence=effective_evidence,
                attributes={"previous_relation": previous_parent.relation},
            )
        self.add_edge(
            object_id,
            "MOVED_TO",
            destination_id,
            "world",
            effective_confidence,
            step=step,
            evidence=effective_evidence,
        )
        self.add_edge(
            object_id,
            destination_relation,
            destination_id,
            "world",
            effective_confidence,
            step=step,
            evidence=effective_evidence,
        )
        return state

    def _record_state_change(self, state: ObjectState) -> None:
        state_id = f"state:{state.object_id}:{state.step}"
        self.add_node(
            state_id,
            "state",
            label=f"{state.object_id}@{state.step}",
            attributes={
                "object_id": state.object_id,
                "pose": state.pose.to_dict(),
                "visible": state.visible,
                "confidence": state.confidence,
                "last_seen_step": state.last_seen_step,
                "step": state.step,
            },
        )
        self.add_edge(
            state.object_id,
            "STATE_CHANGED",
            state_id,
            "world",
            state.confidence,
            step=state.step,
            evidence=[],
        )

    def _record_agent_pose_change(self, state: AgentPoseState) -> None:
        state_id = f"state:{state.agent_id}:{state.step}"
        self.add_node(
            state_id,
            "state",
            label=f"{state.agent_id}@{state.step}",
            attributes={
                "agent_id": state.agent_id,
                "pose": state.pose.to_dict(),
                "step": state.step,
            },
        )
        self.add_edge(
            state.agent_id,
            "STATE_CHANGED",
            state_id,
            "world",
            1.0,
            step=state.step,
            evidence=[],
        )

    def _latest_parent_edge(self, object_id: str) -> Edge | None:
        candidates = [
            edge
            for edge in self.edges
            if edge.src == object_id and edge.relation in CONTAINMENT_RELATIONS
        ]
        if not candidates:
            return None
        return sorted(candidates, key=self._edge_sort_key)[-1]

    @staticmethod
    def _edge_sort_key(edge: Edge) -> tuple[int, str, str, str, str]:
        return (edge.step, edge.src, edge.relation, edge.dst, edge.reference_frame)
