from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

from dsg_spatialqa_lab.schema import BBox3D, Pose3D, SpatialQAError


@dataclass(frozen=True)
class RelationConfig:
    margin: float = 0.05
    near_threshold: float = 1.0
    on_vertical_margin: float = 0.08
    support_overlap_ratio: float = 0.25


class RelationEngine:
    """Deterministic geometric relation engine.

    The agent frame uses local x as right and local y as forward. With yaw=0,
    world x is right and world y is forward.
    """

    _EGOCENTRIC_RELATIONS = {"LEFT_OF", "RIGHT_OF", "FRONT_OF", "BEHIND", "NEAR"}
    _WORLD_RELATIONS = {"ABOVE", "ON", "INSIDE", "SUPPORTS", "NEAR"}

    def __init__(self, config: RelationConfig | None = None) -> None:
        self.config = config or RelationConfig()

    def evaluate(
        self,
        src: BBox3D,
        dst: BBox3D,
        relation: str,
        *,
        reference_frame: str = "world",
        agent_pose: Pose3D | None = None,
    ) -> bool:
        normalized = relation.upper()
        if normalized == "NEAR":
            return src.surface_distance_to(dst) <= self.config.near_threshold

        if reference_frame == "agent":
            if agent_pose is None:
                raise SpatialQAError("Agent pose required for agent-frame relation")
            if normalized not in self._EGOCENTRIC_RELATIONS:
                raise SpatialQAError(f"Unsupported agent-frame relation: {relation}")
            return self._evaluate_agent_frame(src, dst, normalized, agent_pose)

        if reference_frame == "world":
            if normalized not in self._WORLD_RELATIONS:
                raise SpatialQAError(f"Unsupported world-frame relation: {relation}")
            return self._evaluate_world_frame(src, dst, normalized)

        raise SpatialQAError(f"Unsupported reference frame: {reference_frame}")

    def _evaluate_agent_frame(
        self,
        src: BBox3D,
        dst: BBox3D,
        relation: str,
        agent_pose: Pose3D,
    ) -> bool:
        src_x, src_y = self._to_agent_xy(src.center, agent_pose)
        dst_x, dst_y = self._to_agent_xy(dst.center, agent_pose)
        margin = self.config.margin

        if relation == "LEFT_OF":
            return src_x < dst_x - margin
        if relation == "RIGHT_OF":
            return src_x > dst_x + margin
        if relation == "FRONT_OF":
            return src_y > dst_y + margin
        if relation == "BEHIND":
            return src_y < dst_y - margin
        raise SpatialQAError(f"Unsupported agent-frame relation: {relation}")

    def _evaluate_world_frame(self, src: BBox3D, dst: BBox3D, relation: str) -> bool:
        if relation == "ABOVE":
            return src.min_z >= dst.max_z - self.config.margin
        if relation == "ON":
            return self._is_on(src, dst)
        if relation == "INSIDE":
            return self._is_inside(src, dst)
        if relation == "SUPPORTS":
            return self._is_on(dst, src)
        raise SpatialQAError(f"Unsupported world-frame relation: {relation}")

    def _is_on(self, src: BBox3D, dst: BBox3D) -> bool:
        vertical_touch = abs(src.min_z - dst.max_z) <= self.config.on_vertical_margin
        support_area = src.xy_overlap_area(dst, margin=self.config.margin)
        required_area = src.xy_area() * self.config.support_overlap_ratio
        return vertical_touch and support_area >= required_area

    def _is_inside(self, src: BBox3D, dst: BBox3D) -> bool:
        center = src.center
        margin = self.config.margin
        return (
            dst.min_x - margin <= center.x <= dst.max_x + margin
            and dst.min_y - margin <= center.y <= dst.max_y + margin
            and dst.min_z - margin <= center.z <= dst.max_z + margin
        )

    @staticmethod
    def _to_agent_xy(pose: Pose3D, agent_pose: Pose3D) -> tuple[float, float]:
        dx = pose.x - agent_pose.x
        dy = pose.y - agent_pose.y
        angle = -agent_pose.yaw
        local_x = dx * cos(angle) - dy * sin(angle)
        local_y = dx * sin(angle) + dy * cos(angle)
        return local_x, local_y
