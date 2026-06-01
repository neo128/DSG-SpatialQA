from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from dsg_spatialqa_lab.graph_tool import GraphTool
from dsg_spatialqa_lab.memory import CONTAINMENT_RELATIONS
from dsg_spatialqa_lab.relations import RelationEngine
from dsg_spatialqa_lab.schema import Edge, Node, QAResponse, SpatialQAError


class SpatialQAEngine:
    def __init__(self, graph_tool: GraphTool, relation_engine: RelationEngine | None = None) -> None:
        self.graph_tool = graph_tool
        self.relation_engine = relation_engine or RelationEngine()

    def answer(self, question: Mapping[str, Any]) -> QAResponse:
        try:
            question_type = self._required_str(question, "type")
            if question_type == "agent_location":
                return self._answer_agent_location(question)
            if question_type == "agent_history":
                return self._answer_agent_history(question)
            if question_type == "agent_timeline":
                return self._answer_agent_timeline(question)
            if question_type == "object_location":
                return self._answer_object_location(question)
            if question_type == "object_room":
                return self._answer_object_room(question)
            if question_type == "object_status":
                return self._answer_object_status(question)
            if question_type == "relative_relation":
                return self._answer_relative_relation(question)
            if question_type == "nearest_object":
                return self._answer_nearest_object(question)
            if question_type == "label_candidates":
                return self._answer_label_candidates(question)
            if question_type == "object_history":
                return self._answer_object_history(question)
            if question_type == "object_timeline":
                return self._answer_object_timeline(question)
            if question_type == "relation_timeline":
                return self._answer_relation_timeline(question)
            if question_type == "reobserve_targets":
                return self._answer_reobserve_targets(question)
            if question_type == "next_action_validity":
                return self._answer_next_action_validity(question)
            if question_type == "scene_snapshot":
                return self._answer_scene_snapshot(question)
            if question_type == "scene_delta":
                return self._answer_scene_delta(question)
            if question_type == "world_state":
                return self._answer_world_state(question)
            if question_type == "recent_events":
                return self._answer_recent_events(question)
            if question_type == "graph_query":
                return self._answer_graph_query(question)
            if question_type == "retrieve_subgraph":
                return self._answer_retrieve_subgraph(question)
            return QAResponse({}, [], [], 0.0, False, error=f"Unsupported question type: {question_type}")
        except SpatialQAError as exc:
            return QAResponse({}, [], [], 0.0, False, error=str(exc))

    def _answer_agent_location(self, question: Mapping[str, Any]) -> QAResponse:
        agent_id = self._optional_str(question, "agent_id") or "agent"
        pose = self.graph_tool.get_agent_pose(agent_id)
        return QAResponse(
            answer={"agent_id": agent_id, "pose": pose.to_dict()},
            evidence_nodes=[agent_id],
            evidence_edges=[],
            confidence=1.0,
            needs_reobserve=False,
        )

    def _answer_agent_history(self, question: Mapping[str, Any]) -> QAResponse:
        agent_id = self._optional_str(question, "agent_id") or "agent"
        history = self.graph_tool.get_agent_pose_history(agent_id)
        evidence_edges = self.graph_tool.get_relation(agent_id, "STATE_CHANGED")
        return QAResponse(
            answer={
                "agent_id": agent_id,
                "poses": [state.pose.to_dict() for state in history],
                "steps": [state.step for state in history],
            },
            evidence_nodes=[agent_id] + [f"state:{agent_id}:{state.step}" for state in history],
            evidence_edges=[edge.id for edge in evidence_edges],
            confidence=1.0,
            needs_reobserve=False,
        )

    def _answer_agent_timeline(self, question: Mapping[str, Any]) -> QAResponse:
        agent_id = self._optional_str(question, "agent_id") or "agent"
        timeline = self.graph_tool.agent_timeline(agent_id)
        evidence_nodes = [agent_id] + [
            f"state:{agent_id}:{int(entry['step'])}" for entry in timeline
        ]
        evidence_edges: list[str] = []
        for entry in timeline:
            for edge_id in cast(list[str], entry["evidence_edges"]):
                evidence_edges.append(edge_id)
        return QAResponse(
            answer={"agent_id": agent_id, "timeline": timeline},
            evidence_nodes=list(dict.fromkeys(evidence_nodes)),
            evidence_edges=list(dict.fromkeys(evidence_edges)),
            confidence=1.0,
            needs_reobserve=False,
        )

    def _answer_object_location(self, question: Mapping[str, Any]) -> QAResponse:
        object_id = self._required_str(question, "object_id")
        state = self.graph_tool.get_object(object_id)
        latest_location_edge = self._latest_location_edge(object_id)
        latest_state_edge = self._latest_state_edge(object_id)
        evidence_edges: list[str] = []
        if latest_location_edge is not None:
            evidence_edges.append(latest_location_edge.id)
        if latest_state_edge is not None:
            evidence_edges.append(latest_state_edge.id)
        return QAResponse(
            answer={
                "object_id": object_id,
                "label": state.label,
                "pose": state.pose.to_dict(),
                "visible": state.visible,
                "confidence": state.confidence,
                "last_seen_step": state.last_seen_step,
                "state_step": state.step,
                "current_location": (
                    {
                        "relation": latest_location_edge.relation,
                        "dst": latest_location_edge.dst,
                        "step": latest_location_edge.step,
                    }
                    if latest_location_edge is not None
                    else None
                ),
            },
            evidence_nodes=[object_id, f"state:{object_id}:{state.step}"],
            evidence_edges=evidence_edges,
            confidence=state.confidence,
            needs_reobserve=self.graph_tool.needs_reobserve(object_id),
        )

    def _answer_object_room(self, question: Mapping[str, Any]) -> QAResponse:
        object_id = self._required_str(question, "object_id")
        state = self.graph_tool.get_object(object_id)
        room = self.graph_tool.current_room(object_id)
        needs_reobserve = self.graph_tool.needs_reobserve(object_id)
        if room is None:
            return QAResponse(
                answer={"object_id": object_id, "room": None},
                evidence_nodes=[object_id],
                evidence_edges=[],
                confidence=0.0,
                needs_reobserve=needs_reobserve,
            )

        path = cast(list[dict[str, Any]], room["path"])
        evidence_nodes = [object_id]
        for edge in path:
            dst = str(edge["dst"])
            if dst not in evidence_nodes:
                evidence_nodes.append(dst)
        return QAResponse(
            answer={
                "object_id": object_id,
                "room_id": str(room["room_id"]),
                "room_label": str(room["room_label"]),
                "path": path,
            },
            evidence_nodes=evidence_nodes,
            evidence_edges=cast(list[str], room["evidence_edges"]),
            confidence=state.confidence,
            needs_reobserve=needs_reobserve,
        )

    def _answer_object_status(self, question: Mapping[str, Any]) -> QAResponse:
        object_id = self._required_str(question, "object_id")
        state = self.graph_tool.get_object(object_id)
        latest_state_edge = self._latest_state_edge(object_id)
        needs_reobserve = self.graph_tool.needs_reobserve(object_id)
        return QAResponse(
            answer={
                "object_id": object_id,
                "label": state.label,
                "visible": state.visible,
                "confidence": state.confidence,
                "last_seen_step": state.last_seen_step,
                "last_seen_pose": (
                    state.last_seen_pose.to_dict() if state.last_seen_pose is not None else None
                ),
                "needs_reobserve": needs_reobserve,
            },
            evidence_nodes=[object_id],
            evidence_edges=[latest_state_edge.id] if latest_state_edge is not None else [],
            confidence=state.confidence,
            needs_reobserve=needs_reobserve,
        )

    def _answer_relative_relation(self, question: Mapping[str, Any]) -> QAResponse:
        src = self._required_str(question, "src")
        dst = self._required_str(question, "dst")
        relation = self._required_str(question, "relation").upper()
        reference_frame = str(question.get("reference_frame", "world"))
        edges = self.graph_tool.get_relation(src, relation, dst, reference_frame)

        if edges:
            confidence = edges[-1].confidence
            holds = True
        else:
            src_state = self.graph_tool.get_object(src)
            dst_state = self.graph_tool.get_object(dst)
            agent_pose = (
                self.graph_tool.get_agent_pose("agent") if reference_frame == "agent" else None
            )
            holds = self.relation_engine.evaluate(
                src_state.bbox,
                dst_state.bbox,
                relation,
                reference_frame=reference_frame,
                agent_pose=agent_pose,
            )
            confidence = min(src_state.confidence, dst_state.confidence) if holds else 0.0

        return QAResponse(
            answer={"holds": holds, "relation": relation, "src": src, "dst": dst},
            evidence_nodes=[src, dst],
            evidence_edges=[edge.id for edge in edges],
            confidence=confidence,
            needs_reobserve=self.graph_tool.needs_reobserve(src)
            or self.graph_tool.needs_reobserve(dst),
        )

    def _answer_nearest_object(self, question: Mapping[str, Any]) -> QAResponse:
        src = self._required_str(question, "src")
        candidates = self._optional_str_list(question, "candidates")
        candidate_distances = self.graph_tool.nearest_distances(src, candidates=candidates)
        nearest = candidate_distances[0]
        nearest_object = str(nearest["object_id"])
        answer: dict[str, Any] = {
            "src": src,
            "nearest_object": nearest_object,
            "distance": nearest["distance"],
            "candidate_distances": candidate_distances,
        }
        if candidates is not None:
            answer["candidates"] = candidates
        return QAResponse(
            answer=answer,
            evidence_nodes=list(
                dict.fromkeys(
                    [src] + [str(candidate["object_id"]) for candidate in candidate_distances]
                )
            ),
            evidence_edges=[],
            confidence=float(nearest["confidence"]),
            needs_reobserve=bool(nearest["needs_reobserve"]),
        )

    def _answer_label_candidates(self, question: Mapping[str, Any]) -> QAResponse:
        label = self._required_str(question, "label")
        visible = self._optional_bool(question, "visible")
        candidates = self.graph_tool.find_objects(label=label, visible=visible)

        evidence_nodes: list[str] = []
        evidence_edges: list[str] = []
        object_answers: list[dict[str, Any]] = []
        needs_reobserve = False
        for state in candidates:
            object_needs_reobserve = self.graph_tool.needs_reobserve(state.object_id)
            needs_reobserve = needs_reobserve or object_needs_reobserve
            evidence_nodes.extend([state.object_id, f"state:{state.object_id}:{state.step}"])
            state_edge = self._latest_state_edge(state.object_id)
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
                    "last_seen_pose": (
                        state.last_seen_pose.to_dict()
                        if state.last_seen_pose is not None
                        else None
                    ),
                    "state_step": state.step,
                    "needs_reobserve": object_needs_reobserve,
                }
            )

        return QAResponse(
            answer={
                "label": label,
                "visible": visible,
                "count": len(object_answers),
                "ambiguous": len(object_answers) > 1,
                "objects": object_answers,
            },
            evidence_nodes=list(dict.fromkeys(evidence_nodes)),
            evidence_edges=list(dict.fromkeys(evidence_edges)),
            confidence=min((state.confidence for state in candidates), default=0.0),
            needs_reobserve=needs_reobserve,
        )

    def _answer_object_history(self, question: Mapping[str, Any]) -> QAResponse:
        object_id = self._required_str(question, "object_id")
        edges = self.graph_tool.history(object_id)
        return QAResponse(
            answer={
                "object_id": object_id,
                "relations": [edge.relation for edge in edges],
                "steps": [edge.step for edge in edges],
            },
            evidence_nodes=[object_id],
            evidence_edges=[edge.id for edge in edges],
            confidence=1.0 if edges else 0.0,
            needs_reobserve=self.graph_tool.needs_reobserve(object_id),
        )

    def _answer_object_timeline(self, question: Mapping[str, Any]) -> QAResponse:
        object_id = self._required_str(question, "object_id")
        state = self.graph_tool.get_object(object_id)
        timeline = self.graph_tool.object_timeline(object_id)
        evidence_nodes = [object_id] + [
            f"state:{object_id}:{int(entry['step'])}" for entry in timeline
        ]
        evidence_edges: list[str] = []
        for entry in timeline:
            for edge_id in cast(list[str], entry["evidence_edges"]):
                evidence_edges.append(edge_id)
        return QAResponse(
            answer={"object_id": object_id, "timeline": timeline},
            evidence_nodes=list(dict.fromkeys(evidence_nodes)),
            evidence_edges=list(dict.fromkeys(evidence_edges)),
            confidence=state.confidence,
            needs_reobserve=self.graph_tool.needs_reobserve(object_id),
        )

    def _answer_relation_timeline(self, question: Mapping[str, Any]) -> QAResponse:
        timeline = self.graph_tool.relation_timeline(
            src=self._optional_str(question, "src"),
            relation=self._optional_str(question, "relation"),
            dst=self._optional_str(question, "dst"),
            reference_frame=self._optional_str(question, "reference_frame"),
            step_min=self._optional_int(question, "step_min"),
            step_max=self._optional_int(question, "step_max"),
        )
        evidence_nodes: list[str] = []
        evidence_edges: list[str] = []
        for entry in timeline:
            evidence_nodes.extend([str(entry["src"]), str(entry["dst"])])
            evidence_edges.append(str(entry["id"]))
        return QAResponse(
            answer={"timeline": timeline},
            evidence_nodes=list(dict.fromkeys(evidence_nodes)),
            evidence_edges=evidence_edges,
            confidence=min((float(entry["confidence"]) for entry in timeline), default=0.0),
            needs_reobserve=self._relation_timeline_needs_reobserve(timeline),
        )

    def _answer_reobserve_targets(self, question: Mapping[str, Any]) -> QAResponse:
        label = self._optional_str(question, "label")
        targets = self.graph_tool.reobserve_targets(label=label)
        evidence_nodes: list[str] = []
        evidence_edges: list[str] = []
        object_answers: list[dict[str, Any]] = []
        for state in targets:
            evidence_nodes.extend([state.object_id, f"state:{state.object_id}:{state.step}"])
            state_edge = self._latest_state_edge(state.object_id)
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
                    "last_seen_pose": (
                        state.last_seen_pose.to_dict()
                        if state.last_seen_pose is not None
                        else None
                    ),
                    "state_step": state.step,
                }
            )
        return QAResponse(
            answer={"count": len(object_answers), "objects": object_answers},
            evidence_nodes=list(dict.fromkeys(evidence_nodes)),
            evidence_edges=list(dict.fromkeys(evidence_edges)),
            confidence=min((state.confidence for state in targets), default=1.0),
            needs_reobserve=bool(targets),
        )

    def _answer_next_action_validity(self, question: Mapping[str, Any]) -> QAResponse:
        action = question.get("action")
        if action is None:
            raise SpatialQAError("Question missing action")
        validation = self.graph_tool.validate_next_action(cast(Any, action))
        return QAResponse(
            answer={
                "valid": validation.valid,
                "needs_replan": validation.needs_replan,
                "reason": validation.reason,
            },
            evidence_nodes=[],
            evidence_edges=validation.evidence_edges,
            confidence=1.0,
            needs_reobserve=validation.reason == "needs_reobserve",
        )

    def _answer_scene_snapshot(self, question: Mapping[str, Any]) -> QAResponse:
        step = self._required_int(question, "step")
        visible = self._optional_bool(question, "visible")
        agent_id = self._optional_str(question, "agent_id") or "agent"
        snapshot = self.graph_tool.scene_snapshot(step=step, agent_id=agent_id, visible=visible)
        answer = cast(dict[str, Any], snapshot["answer"])
        evidence_nodes = cast(list[str], snapshot["evidence_nodes"])
        evidence_edges = cast(list[str], snapshot["evidence_edges"])
        objects = cast(list[dict[str, Any]], answer["objects"])
        confidences = [float(item["confidence"]) for item in objects]
        needs_reobserve = any(
            item["visible"] is False
            and float(item["confidence"]) < self.graph_tool.reobserve_confidence_threshold
            for item in objects
        )
        return QAResponse(
            answer=answer,
            evidence_nodes=evidence_nodes,
            evidence_edges=evidence_edges,
            confidence=min(confidences) if confidences else 1.0,
            needs_reobserve=needs_reobserve,
        )

    def _answer_scene_delta(self, question: Mapping[str, Any]) -> QAResponse:
        from_step = self._required_int(question, "from_step")
        to_step = self._required_int(question, "to_step")
        visible = self._optional_bool(question, "visible")
        agent_id = self._optional_str(question, "agent_id") or "agent"
        delta = self.graph_tool.scene_delta(
            from_step=from_step,
            to_step=to_step,
            agent_id=agent_id,
            visible=visible,
        )
        answer = cast(dict[str, Any], delta["answer"])
        evidence_nodes = cast(list[str], delta["evidence_nodes"])
        evidence_edges = cast(list[str], delta["evidence_edges"])
        objects = cast(list[dict[str, Any]], answer["objects"])
        confidences = [
            float(item["to_confidence"])
            for item in objects
            if item["to_confidence"] is not None
        ]
        needs_reobserve = any(
            item["to_visible"] is False
            and item["to_confidence"] is not None
            and float(item["to_confidence"]) < self.graph_tool.reobserve_confidence_threshold
            for item in objects
        )
        return QAResponse(
            answer=answer,
            evidence_nodes=evidence_nodes,
            evidence_edges=evidence_edges,
            confidence=min(confidences) if confidences else 1.0,
            needs_reobserve=needs_reobserve,
        )

    def _answer_world_state(self, question: Mapping[str, Any]) -> QAResponse:
        visible = self._optional_bool(question, "visible")
        agent_id = self._optional_str(question, "agent_id") or "agent"
        world_state = self.graph_tool.world_state(agent_id=agent_id, visible=visible)
        answer = cast(dict[str, Any], world_state["answer"])
        evidence_nodes = cast(list[str], world_state["evidence_nodes"])
        evidence_edges = cast(list[str], world_state["evidence_edges"])
        objects = self.graph_tool.find_objects(visible=visible)
        return QAResponse(
            answer=answer,
            evidence_nodes=evidence_nodes,
            evidence_edges=evidence_edges,
            confidence=min((state.confidence for state in objects), default=0.0),
            needs_reobserve=any(self.graph_tool.needs_reobserve(state.object_id) for state in objects),
        )

    def _answer_recent_events(self, question: Mapping[str, Any]) -> QAResponse:
        recent = self.graph_tool.recent_events(
            since_step=self._optional_int(question, "since_step"),
            until_step=self._optional_int(question, "until_step"),
        )
        answer = cast(dict[str, Any], recent["answer"])
        evidence_nodes = cast(list[str], recent["evidence_nodes"])
        evidence_edges = cast(list[str], recent["evidence_edges"])
        events = cast(list[dict[str, Any]], answer["events"])
        changes = cast(list[dict[str, Any]], answer["changes"])
        return QAResponse(
            answer=answer,
            evidence_nodes=evidence_nodes,
            evidence_edges=evidence_edges,
            confidence=1.0 if events or changes else 0.0,
            needs_reobserve=False,
        )

    def _answer_graph_query(self, question: Mapping[str, Any]) -> QAResponse:
        query = self._required_mapping(question, "query")
        subgraph = self.graph_tool.query_graph(query)
        nodes = subgraph["nodes"]
        edges = subgraph["edges"]
        return QAResponse(
            answer={
                "nodes": [self._node_to_dict(node) for node in nodes],
                "edges": [self._edge_to_dict(edge) for edge in edges],
            },
            evidence_nodes=[node.id for node in nodes],
            evidence_edges=[edge.id for edge in edges],
            confidence=self._graph_query_confidence(nodes, edges),
            needs_reobserve=any(
                node.id in self.graph_tool.graph.object_states
                and self.graph_tool.needs_reobserve(node.id)
                for node in nodes
            ),
        )

    def _answer_retrieve_subgraph(self, question: Mapping[str, Any]) -> QAResponse:
        query = self._required_str(question, "query")
        max_nodes = self._optional_int(question, "max_nodes") or 20
        hops = self._optional_int(question, "hops") or 2
        subgraph = self.graph_tool.retrieve_subgraph(query, max_nodes=max_nodes, hops=hops)
        nodes = subgraph["nodes"]
        edges = subgraph["edges"]
        return QAResponse(
            answer={
                "nodes": [self._node_to_dict(node) for node in nodes],
                "edges": [self._edge_to_dict(edge) for edge in edges],
            },
            evidence_nodes=[node.id for node in nodes],
            evidence_edges=[edge.id for edge in edges],
            confidence=self._graph_query_confidence(nodes, edges),
            needs_reobserve=any(
                node.id in self.graph_tool.graph.object_states
                and self.graph_tool.needs_reobserve(node.id)
                for node in nodes
            ),
        )

    def _latest_location_edge(self, object_id: str) -> Edge | None:
        edges = [
            edge
            for edge in self.graph_tool.graph.edges
            if edge.src == object_id and edge.relation in CONTAINMENT_RELATIONS
        ]
        if not edges:
            return None
        return sorted(edges, key=self._edge_sort_key)[-1]

    def _latest_state_edge(self, object_id: str) -> Edge | None:
        edges = [
            edge
            for edge in self.graph_tool.graph.edges
            if edge.src == object_id and edge.relation == "STATE_CHANGED"
        ]
        if not edges:
            return None
        return sorted(edges, key=self._edge_sort_key)[-1]

    @staticmethod
    def _node_to_dict(node: Node) -> dict[str, Any]:
        return {
            "id": node.id,
            "type": node.type,
            "label": node.label,
            "attributes": dict(node.attributes),
        }

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
    def _graph_query_confidence(nodes: list[Node], edges: list[Edge]) -> float:
        if edges:
            return min(edge.confidence for edge in edges)
        return 1.0 if nodes else 0.0

    def _relation_timeline_needs_reobserve(self, timeline: list[dict[str, Any]]) -> bool:
        object_ids = {
            node_id
            for entry in timeline
            for node_id in (str(entry["src"]), str(entry["dst"]))
            if node_id in self.graph_tool.graph.object_states
        }
        return any(self.graph_tool.needs_reobserve(object_id) for object_id in object_ids)

    @staticmethod
    def _edge_sort_key(edge: Edge) -> tuple[int, str, str, str, str]:
        return (edge.step, edge.src, edge.relation, edge.dst, edge.reference_frame)

    @staticmethod
    def _optional_bool(question: Mapping[str, Any], key: str) -> bool | None:
        value = question.get(key)
        if value is None:
            return None
        if not isinstance(value, bool):
            raise SpatialQAError(f"Question field must be boolean: {key}")
        return value

    @staticmethod
    def _optional_int(question: Mapping[str, Any], key: str) -> int | None:
        value = question.get(key)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool):
            raise SpatialQAError(f"Question field must be integer: {key}")
        return value

    @staticmethod
    def _required_int(question: Mapping[str, Any], key: str) -> int:
        value = question.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise SpatialQAError(f"Question field must be integer: {key}")
        return value

    @staticmethod
    def _required_mapping(question: Mapping[str, Any], key: str) -> Mapping[str, Any]:
        value = question.get(key)
        if not isinstance(value, Mapping):
            raise SpatialQAError(f"Question field must be mapping: {key}")
        return value

    @staticmethod
    def _optional_str(question: Mapping[str, Any], key: str) -> str | None:
        value = question.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise SpatialQAError(f"Question field must be string: {key}")
        return value

    @staticmethod
    def _optional_str_list(question: Mapping[str, Any], key: str) -> list[str] | None:
        value = question.get(key)
        if value is None:
            return None
        if isinstance(value, str) or not isinstance(value, list):
            raise SpatialQAError(f"Question field must be list of strings: {key}")
        if not all(isinstance(item, str) for item in value):
            raise SpatialQAError(f"Question field must be list of strings: {key}")
        return list(value)

    @staticmethod
    def _required_str(question: Mapping[str, Any], key: str) -> str:
        value = question.get(key)
        if not isinstance(value, str):
            raise SpatialQAError(f"Question missing string field: {key}")
        return value
