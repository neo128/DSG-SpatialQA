from __future__ import annotations

from collections.abc import Sequence

from dsg_spatialqa_lab.episodes import EpisodeFrame
from dsg_spatialqa_lab.graph_tool import GraphTool
from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.perception.depth_projector import Instance3D


class SimpleObjectFusion:
    def __init__(self, *, hidden_confidence: float = 0.2) -> None:
        self.hidden_confidence = hidden_confidence

    def ingest_frame(
        self,
        graph: DynamicSceneGraph,
        frame: EpisodeFrame,
        instances: Sequence[Instance3D],
        *,
        infer_relations: bool = True,
    ) -> None:
        graph.set_agent_pose(frame.agent_id, frame.agent_pose, step=frame.step)
        known_before = set(graph.object_states)
        visible_instance_ids: set[str] = set()
        for instance in sorted(instances, key=lambda item: item.instance_id):
            visible_instance_ids.add(instance.instance_id)
            graph.upsert_object(
                instance.instance_id,
                instance.label,
                instance.pose,
                instance.bbox,
                confidence=instance.confidence,
                visible=instance.visible,
                step=frame.step,
                attributes=_instance_source_attributes(instance),
            )

        for object_id in sorted(known_before - visible_instance_ids):
            previous = graph.get_object_state(object_id)
            previous_node = graph.nodes.get(object_id)
            previous_source = _source_value(
                previous_node.attributes if previous_node is not None else {}
            )
            hidden_attributes = {
                "hidden_reason": "missing_detection",
                "source": "missing_detection",
            }
            if previous_source is not None:
                hidden_attributes["previous_source"] = previous_source
            graph.upsert_object(
                object_id,
                previous.label,
                previous.pose,
                previous.bbox,
                confidence=min(previous.confidence, self.hidden_confidence),
                visible=False,
                step=frame.step,
                attributes=hidden_attributes,
            )

        if infer_relations and len(visible_instance_ids) > 1:
            GraphTool(graph).update_spatial_relations(
                step=frame.step,
                object_ids=tuple(sorted(visible_instance_ids)),
                relations=("LEFT_OF", "RIGHT_OF", "NEAR"),
                reference_frames=("agent",),
                agent_id=frame.agent_id,
                evidence=(f"episode:{frame.episode_id}:{frame.step}",),
                attributes={"source": "geometry_inference"},
            )


def _instance_source_attributes(instance: Instance3D) -> dict[str, object]:
    attributes: dict[str, object] = dict(instance.attributes)
    attributes["source_detection_id"] = instance.source_detection_id
    if _source_value(attributes) is None:
        attributes["source"] = "mock_perception"
    else:
        attributes["source"] = _source_value(attributes)
    return attributes


def _source_value(attributes: object) -> str | None:
    if not isinstance(attributes, dict):
        return None
    for key in ("source", "source_name", "source_kind"):
        value = attributes.get(key)
        if isinstance(value, str) and value:
            return value
    return None
