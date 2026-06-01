from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any, cast

from dsg_spatialqa_lab.memory import CONTAINMENT_RELATIONS, DynamicSceneGraph
from dsg_spatialqa_lab.relations import RelationEngine
from dsg_spatialqa_lab.schema import (
    ActionValidation,
    AgentPoseState,
    Edge,
    GraphQuery,
    Node,
    NodeType,
    ObjectState,
    Pose3D,
    SkillCommand,
    SpatialQAError,
    Subgraph,
)


class GraphTool:
    _AGENT_RELATIONS = frozenset({"LEFT_OF", "RIGHT_OF", "FRONT_OF", "BEHIND", "NEAR"})
    _WORLD_RELATIONS = frozenset({"ABOVE", "ON", "NEAR"})

    def __init__(
        self,
        graph: DynamicSceneGraph,
        *,
        reobserve_confidence_threshold: float = 0.5,
        relation_engine: RelationEngine | None = None,
    ) -> None:
        self.graph = graph
        self.reobserve_confidence_threshold = reobserve_confidence_threshold
        self.relation_engine = relation_engine or RelationEngine()

    def find_objects(self, label: str | None = None, visible: bool | None = None) -> list[ObjectState]:
        objects = list(self.graph.object_states.values())
        if label is not None:
            objects = [state for state in objects if state.label == label]
        if visible is not None:
            objects = [state for state in objects if state.visible is visible]
        return sorted(objects, key=lambda state: state.object_id)

    def get_object(self, object_id: str) -> ObjectState:
        return self.graph.get_object_state(object_id)

    def get_agent_pose(self, agent_id: str = "agent") -> Pose3D:
        return self.graph.get_agent_pose(agent_id)

    def get_agent_pose_history(self, agent_id: str = "agent") -> list[AgentPoseState]:
        return self.graph.get_agent_pose_history(agent_id)

    def agent_timeline(self, agent_id: str = "agent") -> list[dict[str, Any]]:
        timeline: list[dict[str, Any]] = []
        for state in self.get_agent_pose_history(agent_id):
            state_edge = self._state_changed_edge(agent_id, state.step)
            timeline.append(
                {
                    "agent_id": state.agent_id,
                    "step": state.step,
                    "pose": state.pose.to_dict(),
                    "evidence_edges": [state_edge.id] if state_edge is not None else [],
                }
            )
        return timeline

    def get_object_pose(self, object_id: str) -> Pose3D:
        return self.get_object(object_id).pose

    def get_relation(
        self,
        src: str,
        relation: str,
        dst: str | None = None,
        reference_frame: str | None = None,
    ) -> list[Edge]:
        return self.graph.find_edges(src=src, relation=relation, dst=dst, reference_frame=reference_frame)

    def relation_timeline(
        self,
        *,
        src: str | None = None,
        relation: str | None = None,
        dst: str | None = None,
        reference_frame: str | None = None,
        step_min: int | None = None,
        step_max: int | None = None,
    ) -> list[dict[str, Any]]:
        if step_min is not None and (not isinstance(step_min, int) or isinstance(step_min, bool)):
            raise SpatialQAError("step_min must be an integer")
        if step_max is not None and (not isinstance(step_max, int) or isinstance(step_max, bool)):
            raise SpatialQAError("step_max must be an integer")
        if step_min is not None and step_max is not None and step_min > step_max:
            raise SpatialQAError("step_min cannot be greater than step_max")

        normalized_relation = relation.upper() if relation is not None else None
        edges = [
            edge
            for edge in self.graph.edges
            if (src is None or edge.src == src)
            and (normalized_relation is None or edge.relation == normalized_relation)
            and (dst is None or edge.dst == dst)
            and (reference_frame is None or edge.reference_frame == reference_frame)
            and (step_min is None or edge.step >= step_min)
            and (step_max is None or edge.step <= step_max)
        ]
        return [self._edge_to_dict(edge) for edge in sorted(edges, key=self._edge_sort_key)]

    def update_spatial_relations(
        self,
        *,
        step: int,
        object_ids: Sequence[str] | None = None,
        relations: Sequence[str] | None = None,
        reference_frames: Sequence[str] = ("world", "agent"),
        agent_id: str = "agent",
        confidence: float = 1.0,
        evidence: Sequence[str] | None = None,
    ) -> list[Edge]:
        if not isinstance(step, int) or isinstance(step, bool):
            raise SpatialQAError("step must be an integer")

        ids = self._unique_strings(object_ids or sorted(self.graph.object_states), "object_ids")
        relation_names = self._unique_upper_strings(
            relations or ("LEFT_OF", "RIGHT_OF", "FRONT_OF", "BEHIND", "NEAR", "ABOVE", "ON"),
            "relations",
        )
        frames = self._unique_strings(reference_frames, "reference_frames")
        states = {object_id: self.get_object(object_id) for object_id in ids}

        agent_pose = None
        if "agent" in frames:
            agent_pose = self.get_agent_pose(agent_id)

        added: list[Edge] = []
        evidence_list = list(evidence or [])
        for reference_frame in frames:
            allowed_relations = self._relations_for_frame(reference_frame)
            for src_id in ids:
                for dst_id in ids:
                    if src_id == dst_id:
                        continue
                    for relation in relation_names:
                        if relation not in allowed_relations:
                            continue
                        if not self.relation_engine.evaluate(
                            states[src_id].bbox,
                            states[dst_id].bbox,
                            relation,
                            reference_frame=reference_frame,
                            agent_pose=agent_pose if reference_frame == "agent" else None,
                        ):
                            continue
                        if self._has_edge(src_id, relation, dst_id, reference_frame, step):
                            continue
                        added.append(
                            self.graph.add_edge(
                                src_id,
                                relation,
                                dst_id,
                                reference_frame,
                                confidence,
                                step=step,
                                evidence=evidence_list,
                                attributes={"inferred": True},
                            )
                        )
        return sorted(added, key=self._edge_sort_key)

    def nearest(self, src: str, candidates: Sequence[str] | None = None) -> ObjectState:
        return self._nearest_candidate_states(src, candidates)[0][0]

    def nearest_distances(
        self,
        src: str,
        candidates: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "object_id": state.object_id,
                "label": state.label,
                "distance": round(distance, 6),
                "visible": state.visible,
                "confidence": state.confidence,
                "needs_reobserve": self.needs_reobserve(state.object_id),
            }
            for state, distance in self._nearest_candidate_states(src, candidates)
        ]

    def _nearest_candidate_states(
        self,
        src: str,
        candidates: Sequence[str] | None = None,
    ) -> list[tuple[ObjectState, float]]:
        source_pose = self._pose_for_node(src)
        candidate_ids = (
            list(candidates) if candidates is not None else sorted(self.graph.object_states)
        )
        candidate_ids = [candidate_id for candidate_id in candidate_ids if candidate_id != src]
        if not candidate_ids:
            raise SpatialQAError("No candidate objects found")
        candidate_states = [self.get_object(candidate_id) for candidate_id in candidate_ids]
        return sorted(
            ((state, source_pose.distance_to(state.pose)) for state in candidate_states),
            key=lambda item: (item[1], item[0].object_id),
        )

    def history(self, object_id: str) -> list[Edge]:
        return self.graph.history(object_id)

    def object_timeline(self, object_id: str) -> list[dict[str, Any]]:
        self.get_object(object_id)
        timeline: list[dict[str, Any]] = []
        for state in sorted(
            self.graph.object_state_history.get(object_id, []),
            key=lambda item: item.step,
        ):
            location = self._latest_location_at_or_before(object_id, state.step)
            state_edge = self._state_changed_edge(object_id, state.step)
            evidence_edges = []
            if location is not None:
                evidence_edges.append(location.id)
            if state_edge is not None:
                evidence_edges.append(state_edge.id)
            timeline.append(
                {
                    "object_id": state.object_id,
                    "label": state.label,
                    "step": state.step,
                    "pose": state.pose.to_dict(),
                    "visible": state.visible,
                    "confidence": state.confidence,
                    "last_seen_step": state.last_seen_step,
                    "last_seen_pose": (
                        state.last_seen_pose.to_dict()
                        if state.last_seen_pose is not None
                        else None
                    ),
                    "current_location": (
                        self._location_to_dict(location) if location is not None else None
                    ),
                    "evidence_edges": evidence_edges,
                }
            )
        return timeline

    def scene_snapshot(
        self,
        *,
        step: int,
        agent_id: str = "agent",
        visible: bool | None = None,
    ) -> dict[str, Any]:
        if not isinstance(step, int) or isinstance(step, bool):
            raise SpatialQAError("step must be an integer")
        if visible is not None and not isinstance(visible, bool):
            raise SpatialQAError("visible must be a boolean")

        agent_state = self._agent_pose_state_at_or_before(agent_id, step)
        evidence_nodes = [agent_id, f"state:{agent_id}:{agent_state.step}"]
        evidence_edges: list[str] = []
        agent_edge = self._state_changed_edge(agent_id, agent_state.step)
        if agent_edge is not None:
            evidence_edges.append(agent_edge.id)

        object_answers: list[dict[str, Any]] = []
        for object_id in sorted(self.graph.object_state_history):
            state = self._object_state_at_or_before(object_id, step)
            if state is None:
                continue
            if visible is not None and state.visible is not visible:
                continue

            evidence_nodes.extend([object_id, f"state:{object_id}:{state.step}"])
            location = self._latest_location_at_or_before(object_id, step)
            state_edge = self._state_changed_edge(object_id, state.step)
            if location is not None:
                evidence_edges.append(location.id)
            if state_edge is not None:
                evidence_edges.append(state_edge.id)

            object_answers.append(
                {
                    "object_id": state.object_id,
                    "label": state.label,
                    "pose": state.pose.to_dict(),
                    "visible": state.visible,
                    "confidence": state.confidence,
                    "last_seen_step": state.last_seen_step,
                    "state_step": state.step,
                    "current_location": (
                        {
                            "relation": location.relation,
                            "dst": location.dst,
                            "step": location.step,
                        }
                        if location is not None
                        else None
                    ),
                }
            )

        return {
            "answer": {
                "step": step,
                "agent": {
                    "agent_id": agent_id,
                    "pose": agent_state.pose.to_dict(),
                    "state_step": agent_state.step,
                },
                "objects": object_answers,
            },
            "evidence_nodes": list(dict.fromkeys(evidence_nodes)),
            "evidence_edges": list(dict.fromkeys(evidence_edges)),
        }

    def scene_delta(
        self,
        *,
        from_step: int,
        to_step: int,
        agent_id: str = "agent",
        visible: bool | None = None,
    ) -> dict[str, Any]:
        if not isinstance(from_step, int) or isinstance(from_step, bool):
            raise SpatialQAError("from_step must be an integer")
        if not isinstance(to_step, int) or isinstance(to_step, bool):
            raise SpatialQAError("to_step must be an integer")
        if from_step > to_step:
            raise SpatialQAError("from_step cannot be greater than to_step")

        before = self.scene_snapshot(step=from_step, agent_id=agent_id, visible=visible)
        after = self.scene_snapshot(step=to_step, agent_id=agent_id, visible=visible)
        before_answer = cast(dict[str, Any], before["answer"])
        after_answer = cast(dict[str, Any], after["answer"])
        before_agent = cast(dict[str, Any], before_answer["agent"])
        after_agent = cast(dict[str, Any], after_answer["agent"])
        before_objects = self._snapshot_objects_by_id(before_answer)
        after_objects = self._snapshot_objects_by_id(after_answer)

        changed_objects = [
            self._object_delta(object_id, before_objects.get(object_id), after_objects.get(object_id))
            for object_id in sorted(set(before_objects) | set(after_objects))
        ]
        changed_objects = [item for item in changed_objects if item is not None]

        evidence_nodes = list(
            dict.fromkeys(
                cast(list[str], before["evidence_nodes"])
                + cast(list[str], after["evidence_nodes"])
            )
        )
        evidence_edges = list(
            dict.fromkeys(
                cast(list[str], before["evidence_edges"])
                + cast(list[str], after["evidence_edges"])
                + [edge.id for edge in self._edges_between_steps(from_step, to_step)]
            )
        )
        return {
            "answer": {
                "from_step": from_step,
                "to_step": to_step,
                "agent": {
                    "agent_id": agent_id,
                    "changed": before_agent["pose"] != after_agent["pose"],
                    "from_pose": before_agent["pose"],
                    "to_pose": after_agent["pose"],
                    "from_state_step": before_agent["state_step"],
                    "to_state_step": after_agent["state_step"],
                },
                "objects": changed_objects,
            },
            "evidence_nodes": evidence_nodes,
            "evidence_edges": evidence_edges,
        }

    def world_state(
        self,
        *,
        agent_id: str = "agent",
        visible: bool | None = None,
    ) -> dict[str, Any]:
        if visible is not None and not isinstance(visible, bool):
            raise SpatialQAError("visible must be a boolean")

        objects = self.find_objects(visible=visible)
        object_answers: list[dict[str, Any]] = []
        evidence_nodes = [agent_id] if agent_id in self.graph.nodes else []
        evidence_edges: list[str] = []
        max_object_step = max((state.step for state in objects), default=0)

        for state in objects:
            evidence_nodes.append(state.object_id)
            current_location = self._latest_location(state.object_id)
            latest_state_edge = self._state_changed_edge(state.object_id, state.step)
            if current_location is not None:
                evidence_edges.append(current_location.id)
            if latest_state_edge is not None and latest_state_edge.step == max_object_step:
                evidence_edges.append(latest_state_edge.id)
            object_answers.append(
                {
                    "object_id": state.object_id,
                    "label": state.label,
                    "pose": state.pose.to_dict(),
                    "visible": state.visible,
                    "confidence": state.confidence,
                    "last_seen_step": state.last_seen_step,
                    "current_location": (
                        self._location_to_dict(current_location)
                        if current_location is not None
                        else None
                    ),
                }
            )

        return {
            "answer": {
                "agent_pose": self.get_agent_pose(agent_id).to_dict(),
                "objects": object_answers,
            },
            "evidence_nodes": list(dict.fromkeys(evidence_nodes)),
            "evidence_edges": sorted(dict.fromkeys(evidence_edges)),
        }

    def recent_events(
        self,
        *,
        since_step: int | None = None,
        until_step: int | None = None,
    ) -> dict[str, Any]:
        if since_step is not None:
            since_step = self._int_value(since_step, "since_step")
        if until_step is not None:
            until_step = self._int_value(until_step, "until_step")
        if since_step is not None and until_step is not None and since_step > until_step:
            raise SpatialQAError("since_step cannot be greater than until_step")

        event_nodes = [
            node
            for node in self.graph.nodes.values()
            if node.type in {"action", "event"}
            and self._node_step_in_window(node.attributes.get("step"), since_step, until_step)
        ]
        change_edges = [
            edge
            for edge in self.graph.edges
            if self._edge_step_in_window(edge, since_step, until_step)
        ]
        sorted_event_nodes = sorted(event_nodes, key=lambda node: (int(node.attributes["step"]), node.id))
        sorted_change_edges = sorted(change_edges, key=self._edge_sort_key)
        return {
            "answer": {
                "events": [
                    {
                        "id": node.id,
                        "type": node.type,
                        "label": node.label,
                        "step": node.attributes["step"],
                    }
                    for node in sorted_event_nodes
                ],
                "changes": [
                    {
                        "src": edge.src,
                        "relation": edge.relation,
                        "dst": edge.dst,
                        "step": edge.step,
                    }
                    for edge in sorted_change_edges
                ],
            },
            "evidence_nodes": [node.id for node in sorted_event_nodes],
            "evidence_edges": [edge.id for edge in sorted_change_edges],
        }

    def query_graph(self, query: GraphQuery | Mapping[str, Any]) -> Subgraph:
        graph_query = self._coerce_graph_query(query)
        self._validate_graph_query(graph_query)

        edges = self._query_edges(graph_query)
        relation_scoped = bool(
            graph_query.relations
            or graph_query.src
            or graph_query.dst
            or graph_query.reference_frame
        )
        edge_candidates = edges[: graph_query.max_edges] if relation_scoped else edges
        if relation_scoped:
            endpoint_ids = {edge.src for edge in edge_candidates} | {edge.dst for edge in edge_candidates}
            nodes = [
                node
                for node in self.graph.nodes.values()
                if node.id in endpoint_ids and self._matches_node_query(node, graph_query)
            ]
        else:
            nodes = [
                node
                for node in self.graph.nodes.values()
                if self._matches_node_query(node, graph_query)
            ]

        limited_nodes = sorted(nodes, key=self._node_sort_key)[: graph_query.max_nodes]
        limited_node_ids = {node.id for node in limited_nodes}
        limited_edges = [
            edge
            for edge in edge_candidates
            if edge.src in limited_node_ids and edge.dst in limited_node_ids
        ][: graph_query.max_edges]
        return {"nodes": limited_nodes, "edges": limited_edges}

    def retrieve_subgraph(self, query: str, max_nodes: int = 20, hops: int = 2) -> Subgraph:
        if max_nodes <= 0 or hops < 0:
            return {"nodes": [], "edges": []}

        normalized_query = query.casefold()
        seeds = [
            node
            for node in self.graph.nodes.values()
            if self._node_matches_query(node, normalized_query)
        ]
        if not seeds:
            return {"nodes": [], "edges": []}

        selected_ids: set[str] = set()
        selected_edges: list[Edge] = []
        queue: deque[tuple[str, int]] = deque(
            (node.id, 0) for node in sorted(seeds, key=self._node_sort_key)
        )
        traversal_limit = max_nodes * max(2, hops + 1)

        while queue and len(selected_ids) < traversal_limit:
            node_id, depth = queue.popleft()
            if node_id in selected_ids:
                continue
            selected_ids.add(node_id)
            if depth >= hops:
                continue
            adjacent = [
                edge
                for edge in self.graph.edges
                if edge.src == node_id or edge.dst == node_id
            ]
            for edge in sorted(adjacent, key=self._edge_sort_key):
                if edge not in selected_edges:
                    selected_edges.append(edge)
                other_id = edge.dst if edge.src == node_id else edge.src
                if other_id in self.graph.nodes and other_id not in selected_ids:
                    queue.append((other_id, depth + 1))

        nodes = sorted(
            (self.graph.nodes[node_id] for node_id in selected_ids),
            key=self._node_sort_key,
        )
        limited_node_ids = {node.id for node in nodes[:max_nodes]}
        edges = [
            edge
            for edge in sorted(selected_edges, key=self._edge_sort_key)
            if edge.src in limited_node_ids and edge.dst in limited_node_ids
        ][:max_nodes]
        return {"nodes": nodes[:max_nodes], "edges": edges}

    def needs_reobserve(self, object_id: str) -> bool:
        state = self.get_object(object_id)
        return (not state.visible) and state.confidence < self.reobserve_confidence_threshold

    def reobserve_targets(self, label: str | None = None) -> list[ObjectState]:
        return [
            state
            for state in self.find_objects(label=label)
            if (not state.visible) and state.confidence < self.reobserve_confidence_threshold
        ]

    def validate_next_action(self, action: SkillCommand | Mapping[str, Any]) -> ActionValidation:
        target_object = self._target_object_from_action(action)
        state = self.get_object(target_object)
        evidence_edges = [edge.id for edge in self.history(target_object)]
        details = self._action_validation_details(action, target_object, evidence_edges)

        if self.needs_reobserve(target_object):
            return ActionValidation(False, True, "needs_reobserve", evidence_edges, details)
        if not state.visible:
            return ActionValidation(False, True, "target_not_visible", evidence_edges, details)

        expected_last_seen_step = self._precondition_value(action, "last_seen_step")
        if expected_last_seen_step is not None and expected_last_seen_step != state.last_seen_step:
            return ActionValidation(False, True, "stale_object_state", evidence_edges, details)

        expected_pose = self._target_pose_from_action(action)
        if expected_pose is not None and not expected_pose.almost_equals(state.pose):
            return ActionValidation(False, True, "stale_object_state", evidence_edges, details)

        return ActionValidation(True, False, "valid", evidence_edges, details)

    def _pose_for_node(self, node_id: str) -> Pose3D:
        if node_id in self.graph.object_states:
            return self.graph.object_states[node_id].pose
        if node_id in self.graph.agent_poses:
            return self.graph.agent_poses[node_id]
        raise SpatialQAError(f"Node pose not found: {node_id}")

    def _action_validation_details(
        self,
        action: SkillCommand | Mapping[str, Any],
        target_object: str,
        evidence_edges: list[str],
    ) -> dict[str, Any]:
        state = self.get_object(target_object)
        expected_pose = self._target_pose_from_action(action)
        expected_last_seen_step = self._precondition_value(action, "last_seen_step")
        current_location = self._latest_location(target_object)
        return {
            "target_object": target_object,
            "expected_pose": expected_pose.to_dict() if expected_pose is not None else None,
            "current_pose": state.pose.to_dict(),
            "expected_last_seen_step": expected_last_seen_step,
            "current_last_seen_step": state.last_seen_step,
            "changed_at_step": state.step,
            "current_location": (
                {
                    "relation": current_location.relation,
                    "dst": current_location.dst,
                    "step": current_location.step,
                }
                if current_location is not None
                else None
            ),
            "evidence_edges": evidence_edges,
        }

    def _latest_location(self, object_id: str) -> Edge | None:
        candidates = [
            edge
            for edge in self.graph.edges
            if edge.src == object_id and edge.relation in CONTAINMENT_RELATIONS
            ]
        if not candidates:
            return None
        return sorted(candidates, key=self._edge_sort_key)[-1]

    def _latest_location_at_or_before(self, object_id: str, step: int) -> Edge | None:
        candidates = [
            edge
            for edge in self.graph.edges
            if edge.src == object_id
            and edge.relation in CONTAINMENT_RELATIONS
            and edge.step <= step
        ]
        if not candidates:
            return None
        return sorted(candidates, key=self._edge_sort_key)[-1]

    def _state_changed_edge(self, src: str, step: int) -> Edge | None:
        state_id = f"state:{src}:{step}"
        matches = self.graph.find_edges(src=src, relation="STATE_CHANGED", dst=state_id)
        return matches[-1] if matches else None

    @staticmethod
    def _location_to_dict(edge: Edge) -> dict[str, Any]:
        return {
            "relation": edge.relation,
            "dst": edge.dst,
            "step": edge.step,
        }

    def _agent_pose_state_at_or_before(self, agent_id: str, step: int) -> AgentPoseState:
        candidates = [
            state
            for state in self.graph.agent_pose_history.get(agent_id, [])
            if state.step <= step
        ]
        if not candidates:
            raise SpatialQAError(f"Agent pose history not found at or before step: {agent_id}@{step}")
        return sorted(candidates, key=lambda state: (state.step, state.agent_id))[-1]

    def _object_state_at_or_before(self, object_id: str, step: int) -> ObjectState | None:
        candidates = [
            state
            for state in self.graph.object_state_history.get(object_id, [])
            if state.step <= step
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda state: (state.step, state.object_id))[-1]

    @staticmethod
    def _snapshot_objects_by_id(snapshot_answer: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        objects = cast(list[dict[str, Any]], snapshot_answer["objects"])
        return {str(item["object_id"]): item for item in objects}

    @staticmethod
    def _object_delta(
        object_id: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if before is None and after is None:
            return None
        if before is None:
            changes = ["appeared"]
        elif after is None:
            changes = ["disappeared"]
        else:
            changes = []
            comparisons = (
                ("pose", "pose"),
                ("visible", "visible"),
                ("confidence", "confidence"),
                ("last_seen_step", "last_seen_step"),
                ("location", "current_location"),
            )
            for change_name, field_name in comparisons:
                if before[field_name] != after[field_name]:
                    changes.append(change_name)
        if not changes:
            return None

        return {
            "object_id": object_id,
            "label": (after or before or {})["label"],
            "changes": changes,
            "from_pose": before["pose"] if before is not None else None,
            "to_pose": after["pose"] if after is not None else None,
            "from_visible": before["visible"] if before is not None else None,
            "to_visible": after["visible"] if after is not None else None,
            "from_confidence": before["confidence"] if before is not None else None,
            "to_confidence": after["confidence"] if after is not None else None,
            "from_last_seen_step": before["last_seen_step"] if before is not None else None,
            "to_last_seen_step": after["last_seen_step"] if after is not None else None,
            "from_location": before["current_location"] if before is not None else None,
            "to_location": after["current_location"] if after is not None else None,
            "from_state_step": before["state_step"] if before is not None else None,
            "to_state_step": after["state_step"] if after is not None else None,
        }

    def _edges_between_steps(self, from_step: int, to_step: int) -> list[Edge]:
        edges = [edge for edge in self.graph.edges if from_step < edge.step <= to_step]
        return sorted(edges, key=self._edge_sort_key)

    def _query_edges(self, query: GraphQuery) -> list[Edge]:
        relations = {relation.upper() for relation in query.relations}
        edges = [
            edge
            for edge in self.graph.edges
            if (not relations or edge.relation in relations)
            and (query.src is None or edge.src == query.src)
            and (query.dst is None or edge.dst == query.dst)
            and (query.reference_frame is None or edge.reference_frame == query.reference_frame)
            and (query.step_min is None or edge.step >= query.step_min)
            and (query.step_max is None or edge.step <= query.step_max)
        ]
        return sorted(edges, key=self._edge_sort_key)

    def _matches_node_query(self, node: Node, query: GraphQuery) -> bool:
        if query.node_ids and node.id not in query.node_ids:
            return False
        if query.node_types and node.type not in query.node_types:
            return False
        if query.labels and node.label not in query.labels:
            return False
        if query.visible is not None:
            state = self.graph.object_states.get(node.id)
            if state is None or state.visible is not query.visible:
                return False
        if query.text is not None and not self._node_matches_query(node, query.text.casefold()):
            return False
        return True

    def _coerce_graph_query(self, query: GraphQuery | Mapping[str, Any]) -> GraphQuery:
        if isinstance(query, GraphQuery):
            return query
        return GraphQuery(
            node_ids=self._string_tuple(query, "node_ids"),
            node_types=self._node_type_tuple(query, "node_types"),
            labels=self._string_tuple(query, "labels"),
            visible=self._optional_bool(query, "visible"),
            text=self._optional_str(query, "text"),
            relations=self._string_tuple(query, "relations"),
            src=self._optional_str(query, "src"),
            dst=self._optional_str(query, "dst"),
            reference_frame=self._optional_str(query, "reference_frame"),
            step_min=self._optional_int(query, "step_min"),
            step_max=self._optional_int(query, "step_max"),
            max_nodes=self._int_value(query.get("max_nodes", 20), "max_nodes"),
            max_edges=self._int_value(query.get("max_edges", 20), "max_edges"),
        )

    @staticmethod
    def _validate_graph_query(query: GraphQuery) -> None:
        if query.max_nodes <= 0:
            raise SpatialQAError("max_nodes must be positive")
        if query.max_edges <= 0:
            raise SpatialQAError("max_edges must be positive")
        if query.step_min is not None and query.step_max is not None and query.step_min > query.step_max:
            raise SpatialQAError("step_min cannot be greater than step_max")

    @staticmethod
    def _string_tuple(query: Mapping[str, Any], key: str) -> tuple[str, ...]:
        value = query.get(key, ())
        if isinstance(value, str):
            raise SpatialQAError(f"{key} must be a sequence of strings")
        if not isinstance(value, Sequence):
            raise SpatialQAError(f"{key} must be a sequence of strings")
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise SpatialQAError(f"{key} must be a sequence of strings")
            items.append(item)
        return tuple(items)

    @staticmethod
    def _node_type_tuple(query: Mapping[str, Any], key: str) -> tuple[NodeType, ...]:
        allowed = {"agent", "object", "region", "room", "state", "action", "event"}
        values = GraphTool._string_tuple(query, key)
        for value in values:
            if value not in allowed:
                raise SpatialQAError(f"Unsupported node type in query: {value}")
        return cast(tuple[NodeType, ...], values)

    @staticmethod
    def _optional_bool(query: Mapping[str, Any], key: str) -> bool | None:
        value = query.get(key)
        if value is None:
            return None
        if not isinstance(value, bool):
            raise SpatialQAError(f"{key} must be a boolean")
        return value

    @staticmethod
    def _optional_str(query: Mapping[str, Any], key: str) -> str | None:
        value = query.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise SpatialQAError(f"{key} must be a string")
        return value

    @staticmethod
    def _optional_int(query: Mapping[str, Any], key: str) -> int | None:
        value = query.get(key)
        if value is None:
            return None
        return GraphTool._int_value(value, key)

    @staticmethod
    def _int_value(value: Any, key: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise SpatialQAError(f"{key} must be an integer")
        return value

    @staticmethod
    def _node_matches_query(node: Node, normalized_query: str) -> bool:
        searchable = " ".join(
            [
                node.id,
                node.type,
                node.label or "",
                " ".join(str(value) for value in node.attributes.values()),
            ]
        ).casefold()
        return normalized_query in searchable

    @staticmethod
    def _node_step_in_window(value: object, since_step: int | None, until_step: int | None) -> bool:
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        return (since_step is None or value >= since_step) and (
            until_step is None or value <= until_step
        )

    @staticmethod
    def _edge_step_in_window(edge: Edge, since_step: int | None, until_step: int | None) -> bool:
        return (since_step is None or edge.step >= since_step) and (
            until_step is None or edge.step <= until_step
        )

    @classmethod
    def _relations_for_frame(cls, reference_frame: str) -> frozenset[str]:
        if reference_frame == "agent":
            return cls._AGENT_RELATIONS
        if reference_frame == "world":
            return cls._WORLD_RELATIONS
        raise SpatialQAError(f"Unsupported reference frame: {reference_frame}")

    def _has_edge(
        self,
        src: str,
        relation: str,
        dst: str,
        reference_frame: str,
        step: int,
    ) -> bool:
        return any(
            edge.step == step
            for edge in self.graph.find_edges(
                src=src,
                relation=relation,
                dst=dst,
                reference_frame=reference_frame,
            )
        )

    @staticmethod
    def _edge_to_dict(edge: Edge) -> dict[str, Any]:
        return {
            "id": edge.id,
            "src": edge.src,
            "relation": edge.relation,
            "dst": edge.dst,
            "reference_frame": edge.reference_frame,
            "confidence": edge.confidence,
            "step": edge.step,
            "evidence": list(edge.evidence),
            "attributes": dict(edge.attributes),
        }

    @staticmethod
    def _unique_strings(values: Sequence[str], key: str) -> tuple[str, ...]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if not isinstance(value, str):
                raise SpatialQAError(f"{key} must be a sequence of strings")
            if value not in seen:
                seen.add(value)
                result.append(value)
        return tuple(result)

    @staticmethod
    def _unique_upper_strings(values: Sequence[str], key: str) -> tuple[str, ...]:
        return tuple(value.upper() for value in GraphTool._unique_strings(values, key))

    @staticmethod
    def _edge_sort_key(edge: Edge) -> tuple[int, str, str, str, str]:
        return (edge.step, edge.src, edge.relation, edge.dst, edge.reference_frame)

    @staticmethod
    def _node_sort_key(node: Node) -> tuple[int, str]:
        priority = {
            "object": 0,
            "agent": 1,
            "room": 2,
            "region": 2,
            "state": 3,
            "action": 4,
            "event": 4,
        }
        return (priority[node.type], node.id)

    @staticmethod
    def _target_object_from_action(action: SkillCommand | Mapping[str, Any]) -> str:
        if isinstance(action, SkillCommand):
            return action.target_object
        target = action.get("target_object")
        if not isinstance(target, str):
            raise SpatialQAError("Action missing target_object")
        return target

    @staticmethod
    def _target_pose_from_action(action: SkillCommand | Mapping[str, Any]) -> Pose3D | None:
        if isinstance(action, SkillCommand):
            return action.target_pose
        pose = action.get("target_pose")
        if pose is None:
            return None
        if isinstance(pose, Pose3D):
            return pose
        raise SpatialQAError("Action target_pose must be Pose3D")

    @staticmethod
    def _precondition_value(action: SkillCommand | Mapping[str, Any], precondition_type: str) -> Any | None:
        preconditions: Sequence[Mapping[str, Any]]
        if isinstance(action, SkillCommand):
            preconditions = action.preconditions
        else:
            raw = action.get("preconditions", [])
            if not isinstance(raw, Sequence):
                raise SpatialQAError("Action preconditions must be a sequence")
            preconditions = cast(Sequence[Mapping[str, Any]], raw)

        for precondition in preconditions:
            if precondition.get("type") == precondition_type:
                return precondition.get("value")
        return None
