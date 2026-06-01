from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Literal, TypedDict


NodeType = Literal["agent", "object", "region", "room", "state", "action", "event"]


class SpatialQAError(ValueError):
    """Raised when a graph query cannot be answered safely."""


@dataclass(frozen=True)
class Pose3D:
    x: float
    y: float
    z: float
    yaw: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "yaw": self.yaw}

    def distance_to(self, other: Pose3D) -> float:
        return sqrt(
            (self.x - other.x) * (self.x - other.x)
            + (self.y - other.y) * (self.y - other.y)
            + (self.z - other.z) * (self.z - other.z)
        )

    def almost_equals(self, other: Pose3D, tolerance: float = 1e-9) -> bool:
        return (
            abs(self.x - other.x) <= tolerance
            and abs(self.y - other.y) <= tolerance
            and abs(self.z - other.z) <= tolerance
            and abs(self.yaw - other.yaw) <= tolerance
        )


@dataclass(frozen=True)
class BBox3D:
    center: Pose3D
    size: tuple[float, float, float]

    @property
    def half_x(self) -> float:
        return self.size[0] / 2.0

    @property
    def half_y(self) -> float:
        return self.size[1] / 2.0

    @property
    def half_z(self) -> float:
        return self.size[2] / 2.0

    @property
    def min_x(self) -> float:
        return self.center.x - self.half_x

    @property
    def max_x(self) -> float:
        return self.center.x + self.half_x

    @property
    def min_y(self) -> float:
        return self.center.y - self.half_y

    @property
    def max_y(self) -> float:
        return self.center.y + self.half_y

    @property
    def min_z(self) -> float:
        return self.center.z - self.half_z

    @property
    def max_z(self) -> float:
        return self.center.z + self.half_z

    def with_center(self, center: Pose3D) -> BBox3D:
        return BBox3D(center=center, size=self.size)

    def surface_distance_to(self, other: BBox3D) -> float:
        dx = max(other.min_x - self.max_x, self.min_x - other.max_x, 0.0)
        dy = max(other.min_y - self.max_y, self.min_y - other.max_y, 0.0)
        dz = max(other.min_z - self.max_z, self.min_z - other.max_z, 0.0)
        return sqrt(dx * dx + dy * dy + dz * dz)

    def xy_overlap_area(self, other: BBox3D, margin: float = 0.0) -> float:
        overlap_x = max(
            0.0,
            min(self.max_x, other.max_x + margin) - max(self.min_x, other.min_x - margin),
        )
        overlap_y = max(
            0.0,
            min(self.max_y, other.max_y + margin) - max(self.min_y, other.min_y - margin),
        )
        return overlap_x * overlap_y

    def xy_area(self) -> float:
        return self.size[0] * self.size[1]


@dataclass
class Node:
    id: str
    type: NodeType
    label: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    relation: str
    dst: str
    reference_frame: str
    confidence: float
    step: int
    evidence: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.src}-{self.relation}-{self.dst}-{self.step}"


@dataclass
class AgentPoseState:
    agent_id: str
    pose: Pose3D
    step: int


@dataclass
class ObjectState:
    object_id: str
    label: str
    pose: Pose3D
    bbox: BBox3D
    confidence: float
    visible: bool
    step: int
    last_seen_step: int | None
    last_seen_pose: Pose3D | None


@dataclass
class SkillCommand:
    skill: str
    target_object: str
    target_pose: Pose3D
    reference_object: str | None = None
    preconditions: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerResult:
    status: str
    command: SkillCommand | None = None
    error: str | None = None
    needs_reobserve: bool = False
    needs_replan: bool = False
    ambiguous_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionValidation:
    valid: bool
    needs_replan: bool
    reason: str
    evidence_edges: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAResponse:
    answer: dict[str, Any]
    evidence_nodes: list[str]
    evidence_edges: list[str]
    confidence: float
    needs_reobserve: bool
    error: str | None = None


@dataclass(frozen=True)
class GraphQuery:
    node_ids: tuple[str, ...] = field(default_factory=tuple)
    node_types: tuple[NodeType, ...] = field(default_factory=tuple)
    labels: tuple[str, ...] = field(default_factory=tuple)
    visible: bool | None = None
    text: str | None = None
    relations: tuple[str, ...] = field(default_factory=tuple)
    src: str | None = None
    dst: str | None = None
    reference_frame: str | None = None
    step_min: int | None = None
    step_max: int | None = None
    max_nodes: int = 20
    max_edges: int = 20


class Subgraph(TypedDict):
    nodes: list[Node]
    edges: list[Edge]
