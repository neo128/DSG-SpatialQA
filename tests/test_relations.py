from math import pi

from dsg_spatialqa_lab import BBox3D, Pose3D, RelationConfig, RelationEngine


def box(center: tuple[float, float, float], size: tuple[float, float, float]) -> BBox3D:
    return BBox3D(center=Pose3D(*center), size=size)


def test_near_uses_bbox_surface_distance_not_center_distance() -> None:
    engine = RelationEngine(RelationConfig(near_threshold=0.2, margin=0.0))
    large_table = box((0.0, 0.0, 0.5), (4.0, 1.0, 1.0))
    mug_near_edge = box((2.15, 0.0, 0.5), (0.2, 0.2, 0.2))
    mug_far_from_edge = box((2.45, 0.0, 0.5), (0.2, 0.2, 0.2))

    assert engine.evaluate(mug_near_edge, large_table, "NEAR", reference_frame="world") is True
    assert engine.evaluate(mug_far_from_edge, large_table, "NEAR", reference_frame="world") is False


def test_on_requires_vertical_contact_and_support_overlap_ratio() -> None:
    engine = RelationEngine(
        RelationConfig(margin=0.0, on_vertical_margin=0.05, support_overlap_ratio=0.5)
    )
    table = box((0.0, 0.0, 0.35), (1.0, 1.0, 0.7))
    centered_mug = box((0.0, 0.0, 0.78), (0.2, 0.2, 0.16))
    barely_supported_mug = box((0.58, 0.0, 0.78), (0.2, 0.2, 0.16))
    floating_mug = box((0.0, 0.0, 1.0), (0.2, 0.2, 0.16))

    assert engine.evaluate(centered_mug, table, "ON", reference_frame="world") is True
    assert engine.evaluate(barely_supported_mug, table, "ON", reference_frame="world") is False
    assert engine.evaluate(floating_mug, table, "ON", reference_frame="world") is False


def test_agent_frame_relations_respect_agent_yaw() -> None:
    engine = RelationEngine(RelationConfig(margin=0.01))
    agent = Pose3D(0.0, 0.0, 0.0, yaw=pi / 2.0)
    forward_object = box((0.0, 1.0, 0.0), (0.2, 0.2, 0.2))
    right_object = box((1.0, 0.0, 0.0), (0.2, 0.2, 0.2))

    assert (
        engine.evaluate(
            forward_object,
            right_object,
            "RIGHT_OF",
            reference_frame="agent",
            agent_pose=agent,
        )
        is True
    )
    assert (
        engine.evaluate(
            right_object,
            forward_object,
            "BEHIND",
            reference_frame="agent",
            agent_pose=agent,
        )
        is True
    )
