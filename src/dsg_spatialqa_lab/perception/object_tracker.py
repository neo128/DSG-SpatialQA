from __future__ import annotations

from collections.abc import Sequence

from dsg_spatialqa_lab.perception.depth_projector import Instance3D


class SimpleObjectTracker:
    def __init__(self) -> None:
        self._known_instance_ids: set[str] = set()

    @property
    def known_instance_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._known_instance_ids))

    def track(self, instances: Sequence[Instance3D]) -> tuple[Instance3D, ...]:
        tracked = tuple(
            sorted(instances, key=lambda item: (item.instance_id, item.source_detection_id))
        )
        for instance in tracked:
            self._known_instance_ids.add(instance.instance_id)
        return tracked
