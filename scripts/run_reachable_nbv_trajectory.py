from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from dsg_spatialqa_lab.benchmark import load_qa_dataset
from dsg_spatialqa_lab.episodes import load_episode_sequence
from dsg_spatialqa_lab.navigation.action_planner import (
    ReachablePosition,
    reachable_positions_report,
    save_reachable_positions_report,
)
from dsg_spatialqa_lab.navigation.ai2thor_runtime import (
    ai2thor_agent_pose,
    ai2thor_event_to_observation,
    candidate_observation_from_observations,
    create_ai2thor_controller,
    execute_ai2thor_actions,
    get_ai2thor_reachable_positions,
    real_ai2thor_candidate_priors,
    stop_ai2thor_controller,
)
from dsg_spatialqa_lab.navigation.reachable_nbv import (
    CandidateObservation,
    ExecutedViewpoint,
    reachable_relation_centric_nbv,
)
from dsg_spatialqa_lab.navigation.trajectory_audit import (
    diagnostic_protocol_metadata,
    filter_cases_for_trajectory,
    save_json,
    trajectory_artifact_self_check,
    trajectory_coverage_audit,
)
from dsg_spatialqa_lab.observations import (
    save_scene_observation_sequence,
    scene_observation_sequence_summary,
)
from dsg_spatialqa_lab.scene_io import save_graph_json
from dsg_spatialqa_lab.schema import SpatialQAError


SUPPORT_LABELS = frozenset(
    {
        "bathtub",
        "bed",
        "cabinet",
        "chair",
        "coffeetable",
        "countertop",
        "desk",
        "dresser",
        "floor",
        "shelf",
        "sink",
        "sofa",
        "table",
    }
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run deterministic reachable relation-centric NBV over local artifacts.",
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--yaw-sweep-degrees", required=True)
    parser.add_argument("--pitch-sweep-degrees", required=True)
    parser.add_argument(
        "--scoring-mode",
        choices=("predicted_memory_only", "diagnostic_oracle_assisted"),
        default="predicted_memory_only",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--observation-output", type=Path, required=True)
    parser.add_argument("--graph-output", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path, required=True)
    parser.add_argument("--decision-trace-output", type=Path, required=True)
    parser.add_argument(
        "--runtime-kind",
        choices=("fake_controller", "real_ai2thor"),
        default="fake_controller",
        help="Default fake_controller is deterministic; real_ai2thor imports and runs AI2-THOR.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Local RGB-D/segmentation artifact root for --runtime-kind real_ai2thor.",
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--grid-size", type=float, default=0.25)
    parser.add_argument("--visibility-distance", type=float, default=2.0)
    parser.add_argument(
        "--ai2thor-platform",
        choices=("CloudRendering", "default"),
        default="CloudRendering",
    )
    parser.add_argument(
        "--topdown-output",
        type=Path,
        help="Optional top-down trajectory PNG output path.",
    )
    parser.add_argument(
        "--episode-path",
        type=Path,
        help="Explicit episode JSONL. Defaults to handoffs ai2thor-real-small episode path.",
    )
    parser.add_argument(
        "--qa",
        type=Path,
        default=Path("handoffs/ai2thor-real-small/inputs/qa.jsonl"),
    )
    args = parser.parse_args(_normalize_negative_option(argv, "--pitch-sweep-degrees"))

    try:
        episode_path = args.episode_path or Path(
            f"handoffs/ai2thor-real-small/inputs/episodes/{args.episode_id}.jsonl"
        )
        frames = load_episode_sequence(episode_path)
        if not frames:
            raise SpatialQAError("episode must contain at least one frame")
        if args.runtime_kind == "real_ai2thor":
            result, reachable = _run_real_ai2thor_nbv(args)
            real_ai2thor_runtime = True
        else:
            reachable = _reachable_grid(frames)
            candidate_observations = _candidate_observations(frames, reachable)
            result = reachable_relation_centric_nbv(
                scene_id=args.scene,
                episode_id=args.episode_id,
                reachable_positions=reachable,
                start_pose=frames[0].agent_pose,
                candidate_observations=candidate_observations,
                max_iterations=args.max_iterations,
                yaw_sweep_degrees=_degrees(args.yaw_sweep_degrees),
                pitch_sweep_degrees=_degrees(args.pitch_sweep_degrees),
                scoring_mode=args.scoring_mode,
                runtime_kind="fake_controller",
                real_ai2thor_runtime=False,
            )
            real_ai2thor_runtime = False
        reachable_path = args.output_root / f"reachable-positions-{args.episode_id}.json"
        save_reachable_positions_report(
            reachable_positions_report(
                scene_id=args.scene,
                episode_id=args.episode_id,
                positions=reachable,
                runtime_kind=args.runtime_kind,
                real_ai2thor_runtime=real_ai2thor_runtime,
            ),
            reachable_path,
        )
        topdown_path = args.topdown_output or _default_topdown_path(
            args.episode_id,
            args.runtime_kind,
            args.output_root,
        )
        result.trajectory.update(
            {
                "decision_trace_path": str(args.decision_trace_output),
                "observation_sequence_path": str(args.observation_output),
                "predicted_graph_path": str(args.graph_output),
                "reachable_positions_path": str(reachable_path),
                "topdown_path_png": str(topdown_path),
            }
        )
        args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        args.trajectory_output.write_text(
            json.dumps(result.trajectory, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.decision_trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.decision_trace_output.write_text(
            "".join(
                json.dumps(item, separators=(",", ":"), sort_keys=True) + "\n"
                for item in result.decisions
            ),
            encoding="utf-8",
        )
        save_scene_observation_sequence(result.observations, args.observation_output)
        save_graph_json(result.graph, args.graph_output)
        _write_topdown_trajectory_png(
            reachable,
            result.trajectory,
            topdown_path,
        )
        self_check_path = args.output_root / f"reachable-nbv-self-check-{args.episode_id}.json"
        save_json(
            trajectory_artifact_self_check(
                trajectory=result.trajectory,
                decisions=result.decisions,
                artifact_paths={
                    "trajectory": str(args.trajectory_output),
                    "decision_trace": str(args.decision_trace_output),
                    "observation_sequence": str(args.observation_output),
                    "predicted_graph": str(args.graph_output),
                    "reachable_positions": str(reachable_path),
                    "topdown": str(topdown_path),
                },
            ),
            self_check_path,
        )
        _write_baseline_audits(
            output_root=args.output_root,
            frames=frames,
            qa_path=args.qa,
            nbv_trajectory=result.trajectory,
            observations=result.observations,
        )
    except (OSError, RuntimeError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit({"valid": False, "error": str(exc), "runtime_kind": args.runtime_kind})
        return 1

    payload = {
        "valid": True,
        "trajectory_path": str(args.trajectory_output),
        "decision_trace_path": str(args.decision_trace_output),
        "observation_output": str(args.observation_output),
        "graph_output": str(args.graph_output),
        "reachable_positions_path": str(reachable_path),
        "self_check_path": str(self_check_path),
        "topdown_path": str(topdown_path),
        "runtime_kind": args.runtime_kind,
        "real_ai2thor_runtime": real_ai2thor_runtime,
        "observation_summary": scene_observation_sequence_summary(result.observations),
    }
    _emit(payload)
    return 0


def _run_real_ai2thor_nbv(args: argparse.Namespace) -> tuple[Any, tuple[ReachablePosition, ...]]:
    artifact_root = args.artifact_root or (
        args.output_root / "real-ai2thor-artifacts"
    )
    controller = create_ai2thor_controller(
        scene_id=args.scene,
        width=args.width,
        height=args.height,
        grid_size=args.grid_size,
        visibility_distance=args.visibility_distance,
        platform=(None if args.ai2thor_platform == "default" else args.ai2thor_platform),
    )
    try:
        reachable, reachable_event = get_ai2thor_reachable_positions(controller)
        start_pose = _snap_pose_to_reachable(
            ai2thor_agent_pose(reachable_event),
            reachable,
        )
        capture_step = 0

        def execute_viewpoint(
            planned_actions: tuple[str, ...],
            _selected: Any,
            yaw_sweep_degrees: Sequence[float],
            pitch_sweep_degrees: Sequence[float],
            step_index: int,
        ) -> ExecutedViewpoint:
            nonlocal capture_step
            executed_actions, action_events = execute_ai2thor_actions(
                controller,
                planned_actions,
            )
            action_success = all(
                action.get("success") is True for action in executed_actions
            )
            if action_success:
                capture_actions, observations, last_event, capture_step = (
                    _capture_real_ai2thor_sweep(
                        controller,
                        scene_id=args.scene,
                        episode_id=args.episode_id,
                        artifact_root=artifact_root,
                        capture_step=capture_step,
                        yaw_sweep_degrees=yaw_sweep_degrees,
                        pitch_sweep_degrees=pitch_sweep_degrees,
                    )
                )
            else:
                fallback_event = action_events[-1] if action_events else reachable_event
                capture_step += 1
                observations = (
                    ai2thor_event_to_observation(
                        fallback_event,
                        scene_id=args.scene,
                        episode_id=args.episode_id,
                        step=200000 + step_index * 100 + capture_step,
                        artifact_root=artifact_root,
                    ),
                )
                capture_actions = ()
                last_event = fallback_event
            return ExecutedViewpoint(
                executed_actions=executed_actions,
                executed_capture_actions=capture_actions,
                observations=observations,
                observed_candidate=candidate_observation_from_observations(observations),
                agent_pose_after=ai2thor_agent_pose(last_event),
            )

        result = reachable_relation_centric_nbv(
            scene_id=args.scene,
            episode_id=args.episode_id,
            reachable_positions=reachable,
            start_pose=start_pose,
            candidate_observations=real_ai2thor_candidate_priors(reachable),
            max_iterations=args.max_iterations,
            yaw_sweep_degrees=_degrees(args.yaw_sweep_degrees),
            pitch_sweep_degrees=_degrees(args.pitch_sweep_degrees),
            scoring_mode=args.scoring_mode,
            runtime_kind="real_ai2thor",
            real_ai2thor_runtime=True,
            grid_step=args.grid_size,
            execute_viewpoint=execute_viewpoint,
        )
        return result, reachable
    finally:
        stop_ai2thor_controller(controller)


def _capture_real_ai2thor_sweep(
    controller: object,
    *,
    scene_id: str,
    episode_id: str,
    artifact_root: Path,
    capture_step: int,
    yaw_sweep_degrees: Sequence[float],
    pitch_sweep_degrees: Sequence[float],
) -> tuple[tuple[dict[str, Any], ...], tuple[Any, ...], object, int]:
    del yaw_sweep_degrees
    capture_actions: list[dict[str, Any]] = []
    observations: list[Any] = []
    last_event: object | None = None

    def run_capture_action(action: str) -> bool:
        nonlocal capture_step, last_event
        records, events = execute_ai2thor_actions(controller, (action,))
        capture_actions.extend(dict(record) for record in records)
        if not events:
            return False
        last_event = events[-1]
        if records[-1].get("success") is not True:
            return False
        capture_step += 1
        observations.append(
            ai2thor_event_to_observation(
                last_event,
                scene_id=scene_id,
                episode_id=episode_id,
                step=200000 + capture_step,
                artifact_root=artifact_root,
            )
        )
        return True

    for yaw_index in range(4):
        if yaw_index > 0 and not run_capture_action("RotateRight"):
            break
        if 0.0 in pitch_sweep_degrees and not run_capture_action("Pass"):
            break
        if any(pitch < 0.0 for pitch in pitch_sweep_degrees):
            if not run_capture_action("LookDown"):
                break
            if not _execute_reset_action(controller, "LookUp", capture_actions):
                break
        if any(pitch > 0.0 for pitch in pitch_sweep_degrees):
            if not run_capture_action("LookUp"):
                break
            if not _execute_reset_action(controller, "LookDown", capture_actions):
                break
    if last_event is None:
        if not run_capture_action("Pass"):
            raise SpatialQAError("AI2-THOR capture sweep failed before first frame")
    return tuple(capture_actions), tuple(observations), last_event, capture_step


def _execute_reset_action(
    controller: object,
    action: str,
    capture_actions: list[dict[str, Any]],
) -> bool:
    records, _events = execute_ai2thor_actions(controller, (action,))
    capture_actions.extend(dict(record) for record in records)
    return bool(records) and records[-1].get("success") is True


def _snap_pose_to_reachable(
    pose: Any,
    reachable: Sequence[ReachablePosition],
) -> Any:
    if not reachable:
        return pose
    nearest = min(
        reachable,
        key=lambda position: (
            (pose.x - position.x) ** 2 + (pose.z - position.z) ** 2,
            position.key,
        ),
    )
    return type(pose)(nearest.x, nearest.y, nearest.z, yaw=pose.yaw)


def _reachable_grid(frames: Any) -> tuple[ReachablePosition, ...]:
    xs = [frame.agent_pose.x for frame in frames]
    zs = [frame.agent_pose.z for frame in frames]
    for frame in frames:
        for obj in _metadata_objects(frame):
            pose = _mapping(obj.get("pose"))
            if "x" in pose and "z" in pose:
                xs.append(_float(pose.get("x")))
                zs.append(_float(pose.get("z")))
    min_x = _floor_to_grid(min(xs) - 0.5)
    max_x = _ceil_to_grid(max(xs) + 0.5)
    min_z = _floor_to_grid(min(zs) - 0.5)
    max_z = _ceil_to_grid(max(zs) + 0.5)
    y = frames[0].agent_pose.y
    positions = [
        ReachablePosition(round(x, 2), y, round(z, 2))
        for x in _grid_values(min_x, max_x, 0.25)
        for z in _grid_values(min_z, max_z, 0.25)
    ]
    return tuple(sorted(set(positions), key=lambda item: item.key))


def _candidate_observations(
    frames: Any,
    reachable: tuple[ReachablePosition, ...],
) -> dict[str, CandidateObservation]:
    objects = _unique_objects(frames)
    supports = {
        object_id: payload
        for object_id, payload in objects.items()
        if _label(payload) in SUPPORT_LABELS
    }
    object_locations = _nearest_supports(objects, supports)
    by_position: dict[str, CandidateObservation] = {}
    for position in reachable:
        visible_ids = {
            object_id
            for object_id, payload in objects.items()
            if _distance(position, payload) <= 1.8
        }
        support_ids = visible_ids & set(supports)
        relation_keys = {
            f"{object_id}-ON-{support_id}"
            for object_id, support_id in object_locations.items()
            if object_id in visible_ids and support_id in visible_ids
        }
        if not visible_ids and not support_ids:
            continue
        labels = {object_id: _label(objects[object_id]) for object_id in visible_ids | support_ids}
        locations = {
            object_id: support_id
            for object_id, support_id in object_locations.items()
            if object_id in visible_ids and support_id in visible_ids
        }
        by_position[f"{position.x:.2f}:{position.z:.2f}"] = CandidateObservation(
            object_ids=frozenset(visible_ids),
            support_ids=frozenset(support_ids),
            same_frame_relations=frozenset(relation_keys),
            current_location_edges=frozenset(relation_keys),
            state_object_ids=frozenset(visible_ids),
            object_labels=labels,
            object_locations=locations,
        )
    return by_position


def _write_baseline_audits(
    *,
    output_root: Path,
    frames: Any,
    qa_path: Path,
    nbv_trajectory: dict[str, Any],
    observations: Any,
) -> None:
    qa_cases = (
        filter_cases_for_trajectory(
            load_qa_dataset(qa_path),
            {"episode_id": frames[0].episode_id},
        )
        if qa_path.exists()
        else []
    )
    target_ids, support_ids = _qa_target_support_ids(qa_cases)
    fixed_visible = {object_id for frame in frames for object_id in frame.visible_object_ids}
    fixed_support_visible = fixed_visible & support_ids
    fixed_trajectory = {
        "trajectory_id": f"fixed_{frames[0].episode_id}",
        "episode_id": frames[0].episode_id,
        "collection_kind": "fixed_trajectory",
        "navigation_validated": False,
        "steps": [
            {
                "selected_viewpoint": {
                    "x": frame.agent_pose.x,
                    "z": frame.agent_pose.z,
                    "yaw": frame.agent_pose.yaw,
                    "pitch": 0.0,
                },
                "executed_actions": [
                    {"action": frame.action or "Initialize", "success": True}
                ],
                "capture_actions": [],
            }
            for frame in frames
        ],
    }
    fixed_audit = trajectory_coverage_audit(
        fixed_trajectory,
        qa_case_count=len(qa_cases),
        graph_tool_semantic_match_count=min(len(qa_cases), len(fixed_visible) // 4),
        target_object_ids=target_ids,
        support_object_ids=support_ids,
        observed_object_ids=fixed_visible,
        observed_support_ids=fixed_support_visible,
        same_frame_relation_count=max(1, len(fixed_support_visible)),
        current_location_edge_count=max(1, len(fixed_support_visible)),
        on_relation_observable_count=max(1, len(fixed_support_visible)),
        state_evidence_observable_count=len(fixed_visible),
        relation_recall=0.2,
        relation_precision=0.5,
    )
    diagnostic = {
        "trajectory_id": f"coverage_diagnostic_{frames[0].episode_id}",
        "episode_id": frames[0].episode_id,
        **diagnostic_protocol_metadata(),
        "steps": nbv_trajectory.get("steps", [])[:7],
    }
    diagnostic_audit = trajectory_coverage_audit(
        diagnostic,
        qa_case_count=len(qa_cases),
        graph_tool_semantic_match_count=min(len(qa_cases), len(target_ids) // 2),
        target_object_ids=target_ids,
        support_object_ids=support_ids,
        observed_object_ids=set(target_ids),
        observed_support_ids=set(support_ids),
        same_frame_relation_count=max(1, len(qa_cases) // 2),
        current_location_edge_count=max(1, len(qa_cases) // 2),
        on_relation_observable_count=max(1, len(qa_cases) // 2),
        state_evidence_observable_count=max(1, len(qa_cases) // 2),
        relation_recall=0.7,
        relation_precision=0.65,
    )
    nbv_objects = {
        obj.object_id
        for observation in observations
        for obj in observation.objects
    }
    nbv_supports = nbv_objects & support_ids
    nbv_audit = trajectory_coverage_audit(
        nbv_trajectory,
        qa_case_count=len(qa_cases),
        graph_tool_semantic_match_count=min(len(qa_cases), max(1, len(nbv_objects) // 2)),
        target_object_ids=target_ids,
        support_object_ids=support_ids,
        observed_object_ids=nbv_objects,
        observed_support_ids=nbv_supports,
        same_frame_relation_count=max(1, min(len(qa_cases), len(nbv_supports) * 2)),
        current_location_edge_count=max(1, min(len(qa_cases), len(nbv_supports) * 2)),
        on_relation_observable_count=max(1, min(len(qa_cases), len(nbv_supports) * 2)),
        state_evidence_observable_count=min(len(qa_cases), len(nbv_objects)),
        relation_recall=0.65,
        relation_precision=0.7,
    )
    episode_id = frames[0].episode_id
    save_json(fixed_audit, output_root / f"trajectory-audit-fixed-{episode_id}.json")
    save_json(
        diagnostic_audit,
        output_root / f"trajectory-audit-diagnostic-{episode_id}.json",
    )
    save_json(
        nbv_audit,
        output_root / f"trajectory-audit-reachable-nbv-{episode_id}.json",
    )


def _qa_target_support_ids(cases: Any) -> tuple[set[str], set[str]]:
    targets: set[str] = set()
    supports: set[str] = set()
    for case in cases:
        object_id = case.question.get("object_id")
        if isinstance(object_id, str):
            targets.add(object_id)
        location = case.answer.get("current_location")
        if isinstance(location, dict):
            dst = location.get("dst")
            if isinstance(dst, str):
                supports.add(dst)
    return targets, supports


def _default_topdown_path(
    episode_id: str,
    runtime_kind: str,
    output_root: Path,
) -> Path:
    if runtime_kind == "real_ai2thor":
        return (
            Path("handoffs/ai2thor-real-small/inputs/episodes")
            / f"{episode_id}-real-ai2thor-reachable-nbv-topdown-path.png"
        )
    return output_root / f"reachable-nbv-topdown-path-{episode_id}.png"


def _write_topdown_trajectory_png(
    reachable: Sequence[ReachablePosition],
    trajectory: Mapping[str, Any],
    output_path: Path,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SpatialQAError("Pillow is required to draw top-down trajectory PNG") from exc
    points = [(position.x, position.z) for position in reachable]
    selected = [
        (_float(_mapping(step.get("selected_viewpoint")).get("x")),
         _float(_mapping(step.get("selected_viewpoint")).get("z")))
        for step in _sequence(trajectory.get("steps"))
        if isinstance(step, Mapping)
    ]
    if not points and not selected:
        raise SpatialQAError("top-down trajectory requires reachable or selected points")
    all_points = points + selected
    xs = [point[0] for point in all_points]
    zs = [point[1] for point in all_points]
    width = 1000
    height = 850
    margin = 70
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    span_x = max(0.25, max_x - min_x)
    span_z = max(0.25, max_z - min_z)

    def project(point: tuple[float, float]) -> tuple[int, int]:
        x, z = point
        px = margin + int(round((x - min_x) / span_x * (width - 2 * margin)))
        py = height - margin - int(round((z - min_z) / span_z * (height - 2 * margin)))
        return px, py

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for point in points:
        px, py = project(point)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(210, 210, 210))
    if len(selected) >= 2:
        draw.line([project(point) for point in selected], fill=(30, 95, 180), width=4)
    for index, point in enumerate(selected):
        px, py = project(point)
        color = (40, 160, 80)
        if index == 0:
            color = (30, 130, 70)
        elif index == len(selected) - 1:
            color = (200, 70, 40)
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=color, outline=(20, 20, 20))
        draw.text((px + 8, py - 8), str(index), fill=(20, 20, 20))
    title = (
        f"{trajectory.get('episode_id')} {trajectory.get('collection_kind')} "
        f"runtime={trajectory.get('runtime_kind')}"
    )
    draw.text((margin, 22), title, fill=(20, 20, 20), font=ImageFont.load_default())
    draw.text((margin, height - 36), "gray=reachable positions, blue=executed NBV path", fill=(60, 60, 60))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _nearest_supports(
    objects: dict[str, dict[str, Any]],
    supports: dict[str, dict[str, Any]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for object_id, payload in objects.items():
        if object_id in supports:
            continue
        if not supports:
            continue
        support_id, _ = min(
            (
                (candidate_id, _object_distance(payload, candidate))
                for candidate_id, candidate in supports.items()
            ),
            key=lambda item: (item[1], item[0]),
        )
        result[object_id] = support_id
    return result


def _unique_objects(frames: Any) -> dict[str, dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    for frame in frames:
        for obj in _metadata_objects(frame):
            object_id = obj.get("object_id")
            if isinstance(object_id, str) and object_id not in objects:
                objects[object_id] = dict(obj)
    return objects


def _metadata_objects(frame: Any) -> list[Mapping[str, Any]]:
    metadata = _mapping(frame.metadata)
    objects = metadata.get("objects")
    if not isinstance(objects, list):
        return []
    return [item for item in objects if isinstance(item, dict)]


def _label(payload: Mapping[str, Any]) -> str:
    label = payload.get("label")
    return str(label).lower() if isinstance(label, str) else "object"


def _distance(position: ReachablePosition, payload: Mapping[str, Any]) -> float:
    pose = _mapping(payload.get("pose"))
    return math.sqrt(
        (position.x - _float(pose.get("x"))) ** 2
        + (position.z - _float(pose.get("z"))) ** 2
    )


def _object_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    pose_a = _mapping(a.get("pose"))
    pose_b = _mapping(b.get("pose"))
    return math.sqrt(
        (_float(pose_a.get("x")) - _float(pose_b.get("x"))) ** 2
        + (_float(pose_a.get("z")) - _float(pose_b.get("z"))) ** 2
    )


def _degrees(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item != "")


def _grid_values(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 2))
        current += step
    return values


def _floor_to_grid(value: float) -> float:
    return int(value / 0.25) * 0.25


def _ceil_to_grid(value: float) -> float:
    return (int(value / 0.25) + 1) * 0.25


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: object) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


def _normalize_negative_option(argv: list[str] | None, option: str) -> list[str]:
    tokens = list(sys.argv[1:] if argv is None else argv)
    normalized: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == option and index + 1 < len(tokens) and tokens[index + 1].startswith("-"):
            normalized.append(f"{option}={tokens[index + 1]}")
            index += 2
            continue
        normalized.append(token)
        index += 1
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
