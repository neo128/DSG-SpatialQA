from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.relations import RelationConfig, RelationEngine
from dsg_spatialqa_lab.schema import BBox3D, Pose3D, SpatialQAError

SCENE_FIXTURE_MANIFEST_SCHEMA_VERSION = "dsg-spatialqa-lab.scene-fixture-manifest.v1"


@dataclass(frozen=True)
class SceneFixture:
    name: str
    description: str
    tags: tuple[str, ...]


def build_tabletop_scene() -> DynamicSceneGraph:
    """Build the canonical deterministic scene used by examples and tests."""

    graph = DynamicSceneGraph()
    graph.set_agent_pose("agent", Pose3D(0.0, 0.0, 0.0, yaw=0.0), step=1)
    graph.add_room("kitchen", "Kitchen", step=1)
    graph.upsert_object(
        "mug_1",
        "mug",
        Pose3D(-0.4, 1.0, 0.78),
        BBox3D(center=Pose3D(-0.4, 1.0, 0.78), size=(0.12, 0.12, 0.16)),
        confidence=0.95,
        visible=True,
        step=1,
    )
    graph.upsert_object(
        "plate_1",
        "plate",
        Pose3D(0.35, 1.0, 0.72),
        BBox3D(center=Pose3D(0.35, 1.0, 0.72), size=(0.26, 0.26, 0.04)),
        confidence=0.9,
        visible=True,
        step=1,
    )
    graph.upsert_object(
        "table_1",
        "table",
        Pose3D(0.0, 1.0, 0.35),
        BBox3D(center=Pose3D(0.0, 1.0, 0.35), size=(1.2, 0.8, 0.7)),
        confidence=1.0,
        visible=True,
        step=1,
    )

    engine = RelationEngine(RelationConfig(near_threshold=0.9, margin=0.05))
    if engine.evaluate(
        graph.get_object_state("mug_1").bbox,
        graph.get_object_state("plate_1").bbox,
        "LEFT_OF",
        reference_frame="agent",
        agent_pose=graph.get_agent_pose("agent"),
    ):
        graph.add_edge("mug_1", "LEFT_OF", "plate_1", "agent", 1.0, step=1)
        graph.add_edge("plate_1", "RIGHT_OF", "mug_1", "agent", 1.0, step=1)
    if engine.evaluate(
        graph.get_object_state("mug_1").bbox,
        graph.get_object_state("plate_1").bbox,
        "NEAR",
        reference_frame="agent",
        agent_pose=graph.get_agent_pose("agent"),
    ):
        graph.add_edge("mug_1", "NEAR", "plate_1", "agent", 1.0, step=1)
    if engine.evaluate(
        graph.get_object_state("mug_1").bbox,
        graph.get_object_state("table_1").bbox,
        "ON",
        reference_frame="world",
    ):
        graph.add_edge("mug_1", "ON", "table_1", "world", 1.0, step=1)
    return graph


def build_moved_mug_scene() -> DynamicSceneGraph:
    graph = build_tabletop_scene()
    graph.add_region("sink_region", "Sink region", step=1)
    graph.move_object(
        "mug_1",
        new_pose=Pose3D(1.2, 0.2, 0.5),
        new_bbox=BBox3D(center=Pose3D(1.2, 0.2, 0.5), size=(0.12, 0.12, 0.16)),
        destination_id="sink_region",
        destination_relation="IN_REGION",
        step=2,
        action_id="action_move_mug",
        event_id="event_move_mug",
    )
    return graph


def build_relation_shift_scene() -> DynamicSceneGraph:
    graph = build_tabletop_scene()
    graph.move_object(
        "mug_1",
        new_pose=Pose3D(0.64, 1.0, 0.78),
        new_bbox=BBox3D(center=Pose3D(0.64, 1.0, 0.78), size=(0.12, 0.12, 0.16)),
        destination_id="table_1",
        destination_relation="ON",
        step=2,
        action_id="action_shift_mug_relation",
        event_id="event_shift_mug_relation",
    )
    relation_evidence = ("event_shift_mug_relation",)
    graph.add_edge(
        "mug_1",
        "NEAR",
        "plate_1",
        "agent",
        1.0,
        step=2,
        evidence=relation_evidence,
        attributes={"inferred": True},
    )
    graph.add_edge(
        "mug_1",
        "RIGHT_OF",
        "plate_1",
        "agent",
        1.0,
        step=2,
        evidence=relation_evidence,
        attributes={"inferred": True},
    )
    graph.add_edge(
        "plate_1",
        "LEFT_OF",
        "mug_1",
        "agent",
        1.0,
        step=2,
        evidence=relation_evidence,
        attributes={"inferred": True},
    )
    return graph


def build_multi_room_rearrangement_scene() -> DynamicSceneGraph:
    graph = DynamicSceneGraph()
    graph.set_agent_pose("agent", Pose3D(0.0, 0.0, 0.0, yaw=0.0), step=1)
    graph.add_room("kitchen", "Kitchen", step=1)
    graph.add_room("pantry", "Pantry", step=1)
    graph.add_region("prep_counter", "Prep counter", step=1)
    graph.add_region("pantry_shelf", "Pantry shelf", step=1)
    graph.add_edge("prep_counter", "IN_ROOM", "kitchen", "world", 1.0, step=1)
    graph.add_edge("pantry_shelf", "IN_ROOM", "pantry", "world", 1.0, step=1)
    graph.upsert_object(
        "cereal_box_1",
        "cereal_box",
        Pose3D(0.2, 1.0, 0.9),
        BBox3D(center=Pose3D(0.2, 1.0, 0.9), size=(0.18, 0.08, 0.32)),
        confidence=0.92,
        visible=True,
        step=1,
    )
    graph.upsert_object(
        "milk_1",
        "milk",
        Pose3D(0.42, 1.0, 0.88),
        BBox3D(center=Pose3D(0.42, 1.0, 0.88), size=(0.12, 0.12, 0.28)),
        confidence=0.9,
        visible=True,
        step=1,
    )
    graph.upsert_object(
        "fork_1",
        "fork",
        Pose3D(-0.25, 0.95, 0.82),
        BBox3D(center=Pose3D(-0.25, 0.95, 0.82), size=(0.2, 0.04, 0.02)),
        confidence=0.88,
        visible=True,
        step=1,
    )
    graph.add_edge("cereal_box_1", "IN_REGION", "prep_counter", "world", 0.92, step=1)
    graph.add_edge("milk_1", "IN_REGION", "prep_counter", "world", 0.9, step=1)
    graph.add_edge("fork_1", "IN_REGION", "prep_counter", "world", 0.88, step=1)
    graph.add_edge("cereal_box_1", "NEAR", "milk_1", "world", 0.9, step=1)
    graph.set_agent_pose("agent", Pose3D(2.8, 0.2, 0.0, yaw=0.0), step=2)
    graph.move_object(
        "cereal_box_1",
        new_pose=Pose3D(3.2, 0.4, 1.2),
        new_bbox=BBox3D(center=Pose3D(3.2, 0.4, 1.2), size=(0.18, 0.08, 0.32)),
        destination_id="pantry_shelf",
        destination_relation="IN_REGION",
        step=2,
        action_id="action_move_cereal_box",
        event_id="event_move_cereal_box",
    )
    graph.upsert_object(
        "fork_1",
        "fork",
        Pose3D(-0.25, 0.95, 0.82),
        BBox3D(center=Pose3D(-0.25, 0.95, 0.82), size=(0.2, 0.04, 0.02)),
        confidence=0.2,
        visible=False,
        step=3,
    )
    return graph


def build_needs_reobserve_scene() -> DynamicSceneGraph:
    graph = build_tabletop_scene()
    graph.upsert_object(
        "spoon_1",
        "spoon",
        Pose3D(0.2, 0.8, 0.75),
        BBox3D(center=Pose3D(0.2, 0.8, 0.75), size=(0.2, 0.04, 0.02)),
        confidence=0.25,
        visible=False,
        step=2,
    )
    graph.upsert_object(
        "bowl_1",
        "bowl",
        Pose3D(0.6, 0.8, 0.75),
        BBox3D(center=Pose3D(0.6, 0.8, 0.75), size=(0.3, 0.3, 0.12)),
        confidence=0.75,
        visible=False,
        step=2,
    )
    graph.upsert_object(
        "cup_1",
        "cup",
        Pose3D(0.8, 0.75, 0.78),
        BBox3D(center=Pose3D(0.8, 0.75, 0.78), size=(0.1, 0.1, 0.14)),
        confidence=0.2,
        visible=True,
        step=2,
    )
    return graph


def build_ambiguous_mugs_scene() -> DynamicSceneGraph:
    graph = DynamicSceneGraph()
    graph.set_agent_pose("agent", Pose3D(0.0, 0.0, 0.0), step=1)
    graph.upsert_object(
        "mug_1",
        "mug",
        Pose3D(0.0, 1.0, 0.7),
        BBox3D(center=Pose3D(0.0, 1.0, 0.7), size=(0.12, 0.12, 0.16)),
        confidence=0.9,
        visible=True,
        step=1,
    )
    graph.upsert_object(
        "mug_2",
        "mug",
        Pose3D(0.3, 1.0, 0.7),
        BBox3D(center=Pose3D(0.3, 1.0, 0.7), size=(0.12, 0.12, 0.16)),
        confidence=0.88,
        visible=True,
        step=1,
    )
    return graph


def build_ambiguous_plates_scene() -> DynamicSceneGraph:
    graph = DynamicSceneGraph()
    graph.set_agent_pose("agent", Pose3D(0.0, 0.0, 0.0), step=1)
    graph.upsert_object(
        "mug_1",
        "mug",
        Pose3D(0.0, 1.0, 0.78),
        BBox3D(center=Pose3D(0.0, 1.0, 0.78), size=(0.12, 0.12, 0.16)),
        confidence=0.9,
        visible=True,
        step=1,
    )
    graph.upsert_object(
        "plate_1",
        "plate",
        Pose3D(0.35, 1.0, 0.72),
        BBox3D(center=Pose3D(0.35, 1.0, 0.72), size=(0.26, 0.26, 0.04)),
        confidence=0.9,
        visible=True,
        step=1,
    )
    graph.upsert_object(
        "plate_2",
        "plate",
        Pose3D(0.7, 1.0, 0.72),
        BBox3D(center=Pose3D(0.7, 1.0, 0.72), size=(0.26, 0.26, 0.04)),
        confidence=0.88,
        visible=True,
        step=1,
    )
    return graph


_SCENE_FIXTURES: dict[str, tuple[SceneFixture, Callable[[], DynamicSceneGraph]]] = {
    "ambiguous_mugs": (
        SceneFixture(
            name="ambiguous_mugs",
            description="Static tabletop scene with two visible mugs sharing one label.",
            tags=("static", "tabletop", "ambiguity"),
        ),
        build_ambiguous_mugs_scene,
    ),
    "ambiguous_plates": (
        SceneFixture(
            name="ambiguous_plates",
            description="Static tabletop scene with two visible plates sharing one label.",
            tags=("static", "tabletop", "ambiguity"),
        ),
        build_ambiguous_plates_scene,
    ),
    "moved_mug": (
        SceneFixture(
            name="moved_mug",
            description="Dynamic tabletop scene where mug_1 moves from table_1 to sink_region.",
            tags=("dynamic", "tabletop", "move"),
        ),
        build_moved_mug_scene,
    ),
    "multi_room_rearrangement": (
        SceneFixture(
            name="multi_room_rearrangement",
            description=(
                "Dynamic kitchen-to-pantry scene with relocated cereal and an occluded fork."
            ),
            tags=("dynamic", "multi_room", "move", "occlusion", "reobserve"),
        ),
        build_multi_room_rearrangement_scene,
    ),
    "needs_reobserve": (
        SceneFixture(
            name="needs_reobserve",
            description=(
                "Tabletop scene with invisible and low-confidence objects for re-observation checks."
            ),
            tags=("static", "tabletop", "reobserve"),
        ),
        build_needs_reobserve_scene,
    ),
    "relation_shift": (
        SceneFixture(
            name="relation_shift",
            description=(
                "Dynamic tabletop scene where mug_1 moves from left of plate_1 to right of it."
            ),
            tags=("dynamic", "tabletop", "relations", "move"),
        ),
        build_relation_shift_scene,
    ),
    "tabletop": (
        SceneFixture(
            name="tabletop",
            description="Static tabletop scene with mug, plate, table, room, and agent.",
            tags=("static", "tabletop"),
        ),
        build_tabletop_scene,
    ),
}


def list_scene_fixtures() -> tuple[str, ...]:
    return tuple(sorted(_SCENE_FIXTURES))


def list_scene_fixture_metadata(tags: Sequence[str] | None = None) -> tuple[dict[str, Any], ...]:
    tag_filter = set(tags or ())
    metadata: list[dict[str, Any]] = []
    for name in list_scene_fixtures():
        fixture = _SCENE_FIXTURES[name][0]
        if tag_filter and not tag_filter.issubset(set(fixture.tags)):
            continue
        metadata.append(
            {
                "name": fixture.name,
                "description": fixture.description,
                "tags": list(fixture.tags),
            }
        )
    return tuple(metadata)


def scene_fixture_manifest(tags: Sequence[str] | None = None) -> dict[str, Any]:
    tag_filter = tuple(tags or ())
    fixtures = list_scene_fixture_metadata(tags=tag_filter)
    manifest: dict[str, Any] = {
        "schema_version": SCENE_FIXTURE_MANIFEST_SCHEMA_VERSION,
        "filters": {
            "tags": list(tag_filter),
        },
        "fixture_count": len(fixtures),
        "scene_fixtures": list(fixtures),
    }
    manifest["digest"] = _scene_fixture_manifest_digest(manifest)
    return manifest


def scene_fixture_manifest_json(manifest: Mapping[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def save_scene_fixture_manifest(path: str | Path, tags: Sequence[str] | None = None) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        scene_fixture_manifest_json(scene_fixture_manifest(tags=tags)),
        encoding="utf-8",
    )
    return output_path


def load_scene_fixture_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Scene fixture manifest JSON must be an object")
    return payload


def validate_scene_fixture_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = manifest.get("schema_version")
    manifest_digest = manifest.get("digest")
    expected_digest = _scene_fixture_manifest_digest(manifest)
    scene_fixtures = manifest.get("scene_fixtures")
    fixture_count = manifest.get("fixture_count")
    actual_fixture_count = len(scene_fixtures) if isinstance(scene_fixtures, Sequence) else None
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == SCENE_FIXTURE_MANIFEST_SCHEMA_VERSION,
            "expected": SCENE_FIXTURE_MANIFEST_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "manifest_digest",
            "passed": manifest_digest == expected_digest,
            "expected": expected_digest,
            "actual": manifest_digest,
        },
        {
            "name": "fixture_count_matches_manifest",
            "passed": fixture_count == actual_fixture_count,
            "expected": actual_fixture_count,
            "actual": fixture_count,
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "digest": manifest_digest,
        "checks": checks,
    }


def compare_scene_fixture_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    filters = _scene_fixture_manifest_filters(manifest)
    current_manifest = scene_fixture_manifest(tags=filters)
    saved_digest = manifest.get("digest")
    current_digest = current_manifest["digest"]
    saved_fixtures = _scene_fixture_manifest_entries(manifest)
    current_fixtures = _scene_fixture_manifest_entries(current_manifest)
    fixture_differences = _scene_fixture_manifest_entry_differences(
        saved_fixtures,
        current_fixtures,
    )
    checks: list[dict[str, Any]] = [
        {
            "name": "saved_manifest_valid",
            "passed": validate_scene_fixture_manifest(manifest)["valid"] is True,
        },
        {
            "name": "manifest_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        {
            "name": "fixture_manifest_matches_current",
            "passed": fixture_differences == [],
            "expected": _scene_fixture_manifest_names(saved_fixtures),
            "actual": _scene_fixture_manifest_names(current_fixtures),
        },
    ]
    if fixture_differences:
        checks[-1]["differences"] = fixture_differences
    return {
        "matches": all(check["passed"] is True for check in checks),
        "filters": {
            "tags": list(filters),
        },
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "checks": checks,
    }


def _scene_fixture_manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _scene_fixture_manifest_filters(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    filters = manifest.get("filters", {})
    if not isinstance(filters, Mapping):
        return ()
    tags = filters.get("tags", ())
    if not isinstance(tags, Sequence) or isinstance(tags, str):
        return ()
    return tuple(str(tag) for tag in tags)


def _scene_fixture_manifest_entries(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    scene_fixtures = manifest.get("scene_fixtures", ())
    if not isinstance(scene_fixtures, Sequence) or isinstance(scene_fixtures, str):
        return []
    return [fixture for fixture in scene_fixtures if isinstance(fixture, Mapping)]


def _scene_fixture_manifest_names(scene_fixtures: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(fixture.get("name")) for fixture in scene_fixtures]


def _scene_fixture_manifest_entry_differences(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected_by_name = {str(fixture.get("name")): fixture for fixture in expected}
    actual_by_name = {str(fixture.get("name")): fixture for fixture in actual}
    differences: list[dict[str, Any]] = []
    for name in sorted(set(expected_by_name) | set(actual_by_name)):
        expected_fixture = expected_by_name.get(name)
        actual_fixture = actual_by_name.get(name)
        if expected_fixture is None or actual_fixture is None:
            differences.append(
                {"path": name, "expected": expected_fixture, "actual": actual_fixture}
            )
            continue
        for key in ("description", "tags"):
            expected_value = expected_fixture.get(key)
            actual_value = actual_fixture.get(key)
            if expected_value != actual_value:
                differences.append(
                    {
                        "path": f"{name}.{key}",
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )
    return differences


def get_scene_fixture(name: str) -> SceneFixture:
    fixture = _SCENE_FIXTURES.get(name)
    if fixture is None:
        raise SpatialQAError(f"Unknown scene fixture: {name}")
    return fixture[0]


def load_scene_fixture(name: str) -> DynamicSceneGraph:
    fixture = _SCENE_FIXTURES.get(name)
    if fixture is None:
        raise SpatialQAError(f"Unknown scene fixture: {name}")
    return fixture[1]()
