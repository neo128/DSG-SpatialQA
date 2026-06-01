from dsg_spatialqa_lab import BBox3D, DynamicSceneGraph, GraphTool, Pose3D, VLAAnchorPlanner


def build_pick_scene() -> DynamicSceneGraph:
    graph = DynamicSceneGraph()
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
    return graph


def test_pick_visible_object_returns_deterministic_skill_command() -> None:
    planner = VLAAnchorPlanner(GraphTool(build_pick_scene()))

    result = planner.plan_pick(target_object="mug_1")

    assert result.status == "ok"
    assert result.command is not None
    assert result.command.skill == "pick"
    assert result.command.target_object == "mug_1"
    assert result.command.target_pose == Pose3D(-0.4, 1.0, 0.78)
    assert result.command.preconditions == [
        {"type": "visible", "object_id": "mug_1", "value": True},
        {"type": "min_confidence", "object_id": "mug_1", "value": 0.5},
        {"type": "last_seen_step", "object_id": "mug_1", "value": 1},
    ]
    assert result.command.evidence == ["mug_1"]


def test_place_relative_returns_anchor_pose_right_of_reference() -> None:
    planner = VLAAnchorPlanner(GraphTool(build_pick_scene()), place_margin=0.1)

    result = planner.plan_place_relative("mug_1", "plate_1", "RIGHT_OF")

    assert result.status == "ok"
    assert result.command is not None
    assert result.command.skill == "place_relative"
    assert result.command.target_object == "mug_1"
    assert result.command.reference_object == "plate_1"
    assert result.command.target_pose == Pose3D(0.64, 1.0, 0.82)
    assert result.command.parameters == {"relation": "RIGHT_OF"}


def test_place_relative_can_resolve_unique_labels() -> None:
    planner = VLAAnchorPlanner(GraphTool(build_pick_scene()), place_margin=0.1)

    result = planner.plan_place_relative(
        None,
        None,
        "RIGHT_OF",
        target_label="mug",
        reference_label="plate",
    )

    assert result.status == "ok"
    assert result.command is not None
    assert result.command.skill == "place_relative"
    assert result.command.target_object == "mug_1"
    assert result.command.reference_object == "plate_1"
    assert result.command.target_pose == Pose3D(0.64, 1.0, 0.82)


def test_place_relative_ambiguous_reference_label_does_not_choose_arbitrarily() -> None:
    graph = build_pick_scene()
    graph.upsert_object(
        "plate_2",
        "plate",
        Pose3D(0.7, 1.0, 0.72),
        BBox3D(center=Pose3D(0.7, 1.0, 0.72), size=(0.26, 0.26, 0.04)),
        confidence=0.88,
        visible=True,
        step=1,
    )
    planner = VLAAnchorPlanner(GraphTool(graph), place_margin=0.1)

    result = planner.plan_place_relative(
        "mug_1",
        None,
        "RIGHT_OF",
        reference_label="plate",
    )

    assert result.status == "ambiguous"
    assert result.command is None
    assert result.error == "Ambiguous label: plate"
    assert result.ambiguous_ids == ["plate_1", "plate_2"]
    assert result.details == {
        "label": "plate",
        "candidate_count": 2,
        "candidates": [
            {
                "object_id": "plate_1",
                "label": "plate",
                "pose": {"x": 0.35, "y": 1.0, "z": 0.72, "yaw": 0.0},
                "visible": True,
                "confidence": 0.9,
                "last_seen_step": 1,
                "needs_reobserve": False,
            },
            {
                "object_id": "plate_2",
                "label": "plate",
                "pose": {"x": 0.7, "y": 1.0, "z": 0.72, "yaw": 0.0},
                "visible": True,
                "confidence": 0.88,
                "last_seen_step": 1,
                "needs_reobserve": False,
            },
        ],
    }


def test_place_relative_validation_accepts_current_anchor_plan() -> None:
    planner = VLAAnchorPlanner(GraphTool(build_pick_scene()), place_margin=0.1)
    result = planner.plan_place_relative("mug_1", "plate_1", "RIGHT_OF")
    assert result.command is not None

    validation = planner.validate(result.command)

    assert validation.status == "ok"
    assert validation.command == result.command


def test_place_relative_validation_replans_when_reference_moves() -> None:
    graph = build_pick_scene()
    planner = VLAAnchorPlanner(GraphTool(graph), place_margin=0.1)
    result = planner.plan_place_relative("mug_1", "plate_1", "RIGHT_OF")
    assert result.command is not None

    graph.move_object(
        "plate_1",
        new_pose=Pose3D(1.0, 1.4, 0.72),
        new_bbox=BBox3D(center=Pose3D(1.0, 1.4, 0.72), size=(0.26, 0.26, 0.04)),
        destination_id="counter_region",
        destination_relation="IN_REGION",
        step=2,
    )

    validation = planner.validate(result.command)

    assert validation.status == "needs_replan"
    assert validation.command is None
    assert validation.needs_replan is True
    assert validation.error == "stale_reference_state"
    assert validation.details == {
        "target_object": "mug_1",
        "reference_object": "plate_1",
        "relation": "RIGHT_OF",
        "expected_anchor_pose": {"x": 0.64, "y": 1.0, "z": 0.82, "yaw": 0.0},
        "current_anchor_pose": {"x": 1.29, "y": 1.4, "z": 0.82, "yaw": 0.0},
        "expected_reference_last_seen_step": 1,
        "current_reference_last_seen_step": 2,
    }


def test_invisible_low_confidence_target_returns_needs_reobserve() -> None:
    graph = build_pick_scene()
    graph.upsert_object(
        "mug_1",
        "mug",
        Pose3D(-0.4, 1.0, 0.78),
        BBox3D(center=Pose3D(-0.4, 1.0, 0.78), size=(0.12, 0.12, 0.16)),
        confidence=0.2,
        visible=False,
        step=2,
    )
    planner = VLAAnchorPlanner(GraphTool(graph, reobserve_confidence_threshold=0.5))

    result = planner.plan_pick(target_object="mug_1")

    assert result.status == "needs_reobserve"
    assert result.command is None
    assert result.needs_reobserve is True


def test_ambiguous_label_does_not_choose_arbitrarily() -> None:
    graph = build_pick_scene()
    graph.upsert_object(
        "mug_2",
        "mug",
        Pose3D(0.1, 1.2, 0.78),
        BBox3D(center=Pose3D(0.1, 1.2, 0.78), size=(0.12, 0.12, 0.16)),
        confidence=0.88,
        visible=True,
        step=1,
    )
    planner = VLAAnchorPlanner(GraphTool(graph))

    result = planner.plan_pick(label="mug")

    assert result.status == "ambiguous"
    assert result.command is None
    assert result.ambiguous_ids == ["mug_1", "mug_2"]
    assert result.details == {
        "label": "mug",
        "candidate_count": 2,
        "candidates": [
            {
                "object_id": "mug_1",
                "label": "mug",
                "pose": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
                "visible": True,
                "confidence": 0.95,
                "last_seen_step": 1,
                "needs_reobserve": False,
            },
            {
                "object_id": "mug_2",
                "label": "mug",
                "pose": {"x": 0.1, "y": 1.2, "z": 0.78, "yaw": 0.0},
                "visible": True,
                "confidence": 0.88,
                "last_seen_step": 1,
                "needs_reobserve": False,
            },
        ],
    }


def test_stale_pick_action_returns_needs_replan() -> None:
    graph = build_pick_scene()
    planner = VLAAnchorPlanner(GraphTool(graph))
    result = planner.plan_pick(target_object="mug_1")
    assert result.command is not None

    graph.move_object(
        "mug_1",
        new_pose=Pose3D(1.2, 0.2, 0.5),
        new_bbox=BBox3D(center=Pose3D(1.2, 0.2, 0.5), size=(0.12, 0.12, 0.16)),
        destination_id="sink_region",
        destination_relation="IN_REGION",
        step=2,
    )

    validation = planner.validate(result.command)

    assert validation.status == "needs_replan"
    assert validation.command is None
    assert validation.needs_replan is True
    assert validation.error == "stale_object_state"
    assert validation.details == {
        "target_object": "mug_1",
        "expected_pose": {"x": -0.4, "y": 1.0, "z": 0.78, "yaw": 0.0},
        "current_pose": {"x": 1.2, "y": 0.2, "z": 0.5, "yaw": 0.0},
        "expected_last_seen_step": 1,
        "current_last_seen_step": 2,
        "changed_at_step": 2,
        "current_location": {"relation": "IN_REGION", "dst": "sink_region", "step": 2},
        "evidence_edges": [
            "mug_1-STATE_CHANGED-state:mug_1:1-1",
            "mug_1-IN_REGION-sink_region-2",
            "mug_1-MOVED_TO-sink_region-2",
            "mug_1-STATE_CHANGED-state:mug_1:2-2",
        ],
    }


def test_needs_reobserve_pick_result_includes_precondition_details() -> None:
    graph = build_pick_scene()
    graph.upsert_object(
        "mug_1",
        "mug",
        Pose3D(-0.4, 1.0, 0.78),
        BBox3D(center=Pose3D(-0.4, 1.0, 0.78), size=(0.12, 0.12, 0.16)),
        confidence=0.2,
        visible=False,
        step=2,
    )
    planner = VLAAnchorPlanner(GraphTool(graph, reobserve_confidence_threshold=0.5))

    result = planner.plan_pick(target_object="mug_1")

    assert result.status == "needs_reobserve"
    assert result.details == {
        "target_object": "mug_1",
        "visible": False,
        "confidence": 0.2,
        "min_confidence": 0.5,
        "last_seen_step": 1,
        "current_step": 2,
    }
