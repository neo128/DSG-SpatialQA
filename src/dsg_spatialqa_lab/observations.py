from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from dsg_spatialqa_lab.graph_tool import GraphTool
from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.schema import BBox3D, Pose3D, SpatialQAError


@dataclass(frozen=True)
class NodeObservation:
    node_id: str
    label: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectObservation:
    object_id: str
    label: str
    pose: Pose3D
    bbox: BBox3D
    confidence: float
    visible: bool
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneObservation:
    step: int
    agent_pose: Pose3D | None = None
    agent_id: str = "agent"
    rooms: tuple[NodeObservation, ...] = field(default_factory=tuple)
    regions: tuple[NodeObservation, ...] = field(default_factory=tuple)
    objects: tuple[ObjectObservation, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class IngestResult:
    step: int
    node_ids: tuple[str, ...]
    object_ids: tuple[str, ...]
    state_edge_ids: tuple[str, ...]
    inferred_edge_ids: tuple[str, ...]


class ObservationIngestor:
    def __init__(self, graph: DynamicSceneGraph, *, graph_tool: GraphTool | None = None) -> None:
        self.graph = graph
        self.graph_tool = graph_tool or GraphTool(graph)

    def ingest(
        self,
        observation: SceneObservation,
        *,
        infer_relations: Sequence[str] = (),
        reference_frames: Sequence[str] = ("world", "agent"),
        relation_confidence: float = 1.0,
        relation_evidence: Sequence[str] | None = None,
    ) -> IngestResult:
        self._validate_step(observation.step)
        self._ensure_unique((room.node_id for room in observation.rooms), "rooms")
        self._ensure_unique((region.node_id for region in observation.regions), "regions")
        self._ensure_unique((obj.object_id for obj in observation.objects), "objects")

        node_ids: set[str] = set()
        if observation.agent_pose is not None:
            self.graph.set_agent_pose(
                observation.agent_id,
                observation.agent_pose,
                step=observation.step,
            )
            node_ids.add(observation.agent_id)

        for room in sorted(observation.rooms, key=lambda item: item.node_id):
            self.graph.add_room(
                room.node_id,
                room.label,
                step=observation.step,
                attributes=dict(room.attributes),
            )
            node_ids.add(room.node_id)

        for region in sorted(observation.regions, key=lambda item: item.node_id):
            self.graph.add_region(
                region.node_id,
                region.label,
                step=observation.step,
                attributes=dict(region.attributes),
            )
            node_ids.add(region.node_id)

        object_ids: list[str] = []
        state_edge_ids: list[str] = []
        for obj in sorted(observation.objects, key=lambda item: item.object_id):
            self.graph.upsert_object(
                obj.object_id,
                obj.label,
                obj.pose,
                obj.bbox,
                confidence=obj.confidence,
                visible=obj.visible,
                step=observation.step,
                attributes=dict(obj.attributes),
            )
            object_ids.append(obj.object_id)
            node_ids.add(obj.object_id)
            state_edge_ids.append(self._state_edge_id(obj.object_id, observation.step))

        inferred_edge_ids: tuple[str, ...] = ()
        if infer_relations and object_ids:
            inferred_edges = self.graph_tool.update_spatial_relations(
                step=observation.step,
                object_ids=tuple(object_ids),
                relations=infer_relations,
                reference_frames=reference_frames,
                agent_id=observation.agent_id,
                confidence=relation_confidence,
                evidence=relation_evidence,
            )
            inferred_edge_ids = tuple(edge.id for edge in inferred_edges)

        return IngestResult(
            step=observation.step,
            node_ids=tuple(sorted(node_ids)),
            object_ids=tuple(object_ids),
            state_edge_ids=tuple(state_edge_ids),
            inferred_edge_ids=inferred_edge_ids,
        )

    @staticmethod
    def _validate_step(step: int) -> None:
        if not isinstance(step, int) or isinstance(step, bool):
            raise SpatialQAError("observation step must be an integer")

    @staticmethod
    def _ensure_unique(ids: Iterable[str], label: str) -> None:
        seen: set[str] = set()
        for item_id in ids:
            if item_id in seen:
                raise SpatialQAError(f"Duplicate {label} observation id: {item_id}")
            seen.add(item_id)

    @staticmethod
    def _state_edge_id(object_id: str, step: int) -> str:
        return f"{object_id}-STATE_CHANGED-state:{object_id}:{step}-{step}"
