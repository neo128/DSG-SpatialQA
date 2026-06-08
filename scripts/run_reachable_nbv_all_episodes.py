#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import audit_trajectory_coverage  # noqa: E402
import run_reachable_nbv_trajectory  # noqa: E402


DEFAULT_EPISODES: tuple[tuple[str, str, str], ...] = (
    ("episode001", "ai2thor-real-small-episode-001", "FloorPlan1"),
    ("episode002", "ai2thor-real-small-episode-002", "FloorPlan201"),
    ("episode003", "ai2thor-real-small-episode-003", "FloorPlan301"),
    ("episode004", "ai2thor-real-small-episode-004", "FloorPlan401"),
    ("episode005", "ai2thor-real-small-episode-005", "FloorPlan2"),
)
EPISODES = DEFAULT_EPISODES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run real AI2-THOR reachable NBV over the real-small episode set.",
    )
    parser.add_argument("--episode", action="append")
    parser.add_argument(
        "--episode-plan",
        type=Path,
        help="JSON plan with episodes [{short_id, episode_id, scene_id}].",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only materialize the planned paths/report; do not launch AI2-THOR.",
    )
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--yaw-sweep-degrees", default="0,90,180,270")
    parser.add_argument("--pitch-sweep-degrees", default="-30,0,30")
    parser.add_argument("--scoring-mode", default="predicted_memory_only")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("handoffs/ai2thor-real-small/outputs/navigation"),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("handoffs/ai2thor-real-small/inputs"),
    )
    parser.add_argument(
        "--predicted-root",
        type=Path,
        default=Path("handoffs/ai2thor-real-small/outputs/predicted-dsg"),
    )
    parser.add_argument(
        "--qa",
        type=Path,
        default=Path("handoffs/ai2thor-real-small/inputs/qa.jsonl"),
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
        "--report",
        type=Path,
        default=Path("handoffs/ai2thor-real-small/outputs/navigation/reachable-nbv-real-ai2thor-all-episodes-run-report.json"),
    )
    args = parser.parse_args(argv)

    episodes = load_episode_plan(args.episode_plan)
    selected = {item for item in args.episode or [episode[0] for episode in episodes]}
    unknown = sorted(selected - {episode[0] for episode in episodes})
    if unknown:
        _emit(
            {
                "action": "run_reachable_nbv_all_episodes",
                "valid": False,
                "blockers": ["unknown_episode_short_id"],
                "unknown_episode_short_ids": unknown,
            }
        )
        return 1
    episode_rows: list[dict[str, Any]] = []
    for short_id, full_id, scene_id in episodes:
        if short_id not in selected:
            continue
        paths = _episode_paths(args, short_id, full_id)
        result = (
            _dry_run_episode(short_id, full_id, scene_id, paths)
            if args.dry_run
            else _run_episode(args, short_id, full_id, scene_id, paths)
        )
        episode_rows.append(result)

    report = {
        "schema_version": "dsg-spatialqa-lab.reachable-nbv-all-episodes-run-report.v1",
        "runtime_kind": "dry_run" if args.dry_run else "real_ai2thor",
        "episode_plan_path": str(args.episode_plan) if args.episode_plan is not None else None,
        "episode_count": len(episode_rows),
        "valid": all(row.get("valid") is True for row in episode_rows),
        "episodes": episode_rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _emit({"action": "run_reachable_nbv_all_episodes", "report": str(args.report), **report})
    return 0 if report["valid"] is True else 1


def load_episode_plan(path: Path | None = None) -> tuple[tuple[str, str, str], ...]:
    if path is None:
        return DEFAULT_EPISODES
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    rows = payload.get("episodes")
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain an episodes list")
    episodes = []
    seen_short_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path} episodes[{index}] must be a JSON object")
        short_id = row.get("short_id")
        episode_id = row.get("episode_id")
        scene_id = row.get("scene_id")
        if (
            not isinstance(short_id, str)
            or not short_id
            or not isinstance(episode_id, str)
            or not episode_id
            or not isinstance(scene_id, str)
            or not scene_id
        ):
            raise ValueError(
                f"{path} episodes[{index}] must include non-empty short_id, episode_id, scene_id"
            )
        if short_id in seen_short_ids:
            raise ValueError(f"{path} duplicate short_id: {short_id}")
        seen_short_ids.add(short_id)
        episodes.append((short_id, episode_id, scene_id))
    return tuple(episodes)


def _dry_run_episode(
    short_id: str,
    full_id: str,
    scene_id: str,
    paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "episode_id": full_id,
        "short_id": short_id,
        "scene_id": scene_id,
        "valid": True,
        "dry_run": True,
        "blockers": [],
        "paths": {key: str(value) for key, value in paths.items()},
    }


def _episode_paths(args: argparse.Namespace, short_id: str, full_id: str) -> dict[str, Path]:
    episodes_root = args.input_root / "episodes"
    return {
        "episode": episodes_root / f"{full_id}.jsonl",
        "trajectory": args.output_root / f"reachable-nbv-real-ai2thor-trajectory-{short_id}.json",
        "decision_trace": args.output_root / f"reachable-nbv-real-ai2thor-decision-trace-{short_id}.jsonl",
        "observation": args.predicted_root / f"detector-observations-real-ai2thor-reachable-nbv-{short_id}.json",
        "graph": args.predicted_root / f"predicted-graph-real-ai2thor-reachable-nbv-{short_id}.json",
        "topdown": episodes_root / f"{full_id}-real-ai2thor-reachable-nbv-topdown-path.png",
        "fixed_vs_overlay_png": episodes_root / f"{full_id}-fixed-vs-real-ai2thor-reachable-nbv-overlay.png",
        "fixed_vs_overlay_json": episodes_root / f"{full_id}-fixed-vs-real-ai2thor-reachable-nbv-overlay.json",
        "fixed_episode_topdown": episodes_root / f"{full_id}-topdown-path.png",
        "audit": args.output_root / f"trajectory-audit-real-ai2thor-reachable-nbv-{short_id}.json",
    }


def _run_episode(
    args: argparse.Namespace,
    short_id: str,
    full_id: str,
    scene_id: str,
    paths: dict[str, Path],
) -> dict[str, Any]:
    command = [
        "--scene",
        scene_id,
        "--episode-id",
        full_id,
        "--episode-path",
        str(paths["episode"]),
        "--max-iterations",
        str(args.max_iterations),
        "--yaw-sweep-degrees",
        args.yaw_sweep_degrees,
        "--pitch-sweep-degrees",
        args.pitch_sweep_degrees,
        "--scoring-mode",
        args.scoring_mode,
        "--runtime-kind",
        "real_ai2thor",
        "--output-root",
        str(args.output_root),
        "--artifact-root",
        str(args.output_root / "real-ai2thor-frame-assets"),
        "--observation-output",
        str(paths["observation"]),
        "--graph-output",
        str(paths["graph"]),
        "--trajectory-output",
        str(paths["trajectory"]),
        "--decision-trace-output",
        str(paths["decision_trace"]),
        "--topdown-output",
        str(paths["topdown"]),
        "--qa",
        str(args.qa),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--grid-size",
        str(args.grid_size),
        "--visibility-distance",
        str(args.visibility_distance),
        "--ai2thor-platform",
        args.ai2thor_platform,
    ]
    run_code = run_reachable_nbv_trajectory.main(command)
    if run_code != 0:
        return {
            "episode_id": full_id,
            "short_id": short_id,
            "scene_id": scene_id,
            "valid": False,
            "blockers": ["reachable_nbv_run_failed"],
            "paths": {key: str(value) for key, value in paths.items()},
        }
    audit_code = audit_trajectory_coverage.main(
        [
            "--trajectory",
            str(paths["trajectory"]),
            "--observation-sequence",
            str(paths["observation"]),
            "--predicted-graph",
            str(paths["graph"]),
            "--qa",
            str(args.qa),
            "--output",
            str(paths["audit"]),
        ]
    )
    overlay_valid = _write_fixed_vs_nbv_overlay(paths, full_id, short_id, scene_id)
    trajectory = _load_json(paths["trajectory"])
    trajectory.update(
        {
            "fixed_vs_nbv_overlay_png": str(paths["fixed_vs_overlay_png"]),
            "topdown_path_png": str(paths["topdown"]),
        }
    )
    paths["trajectory"].write_text(json.dumps(trajectory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    valid = audit_code == 0 and overlay_valid
    return {
        "episode_id": full_id,
        "short_id": short_id,
        "scene_id": scene_id,
        "valid": valid,
        "blockers": [] if valid else ["trajectory_audit_or_overlay_failed"],
        "paths": {key: str(value) for key, value in paths.items()},
    }


def _write_fixed_vs_nbv_overlay(
    paths: Mapping[str, Path],
    episode_id: str,
    short_id: str,
    scene_id: str,
) -> bool:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False
    if not paths["fixed_episode_topdown"].exists():
        return False
    fixed_points = _fixed_points(paths["episode"])
    trajectory = _load_json(paths["trajectory"])
    nbv_points = [
        (
            float(step["selected_viewpoint"]["x"]),
            float(step["selected_viewpoint"]["z"]),
        )
        for step in trajectory.get("steps", [])
        if isinstance(step, dict)
        and isinstance(step.get("selected_viewpoint"), dict)
        and "x" in step["selected_viewpoint"]
        and "z" in step["selected_viewpoint"]
    ]
    points = fixed_points + nbv_points
    if not points:
        return False
    img = Image.open(paths["fixed_episode_topdown"]).convert("RGBA")
    draw = ImageDraw.Draw(img)
    font = _font(16)
    min_x, max_x, min_z, max_z = _bounds(points)

    def pixel(point: tuple[float, float]) -> tuple[int, int]:
        x, z = point
        left, right, top, bottom = 70, img.width - 260, 90, img.height - 80
        px = left + (x - min_x) / max(0.001, max_x - min_x) * (right - left)
        py = bottom - (z - min_z) / max(0.001, max_z - min_z) * (bottom - top)
        return int(round(px)), int(round(py))

    if len(fixed_points) > 1:
        draw.line([pixel(point) for point in fixed_points], fill=(220, 60, 60, 230), width=5)
    if len(nbv_points) > 1:
        draw.line([pixel(point) for point in nbv_points], fill=(36, 116, 210, 245), width=6)
    for index, point in enumerate(nbv_points):
        x, y = pixel(point)
        color = (48, 160, 88, 255) if index == 0 else (150, 43, 43, 255) if index == len(nbv_points) - 1 else (36, 116, 210, 245)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=(255, 255, 255, 255), width=2)
        if index not in {0, len(nbv_points) - 1}:
            draw.text((x + 8, y - 8), str(index), fill=(36, 116, 210, 255), font=font)
    for index, point in enumerate(fixed_points):
        x, y = pixel(point)
        draw.rectangle((x - 5, y - 5, x + 5, y + 5), fill=(220, 60, 60, 230))
    draw.rounded_rectangle((28, 28, 620, 118), radius=8, fill=(255, 255, 255, 230), outline=(40, 40, 40, 220), width=2)
    draw.text((44, 44), f"{scene_id} fixed vs real reachable NBV", fill=(30, 36, 45, 255), font=_font(22))
    draw.text((44, 78), "red=fixed trajectory, blue=real AI2-THOR reachable NBV", fill=(30, 36, 45, 255), font=font)
    paths["fixed_vs_overlay_png"].parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(paths["fixed_vs_overlay_png"])
    meta = {
        "schema_version": "dsg-spatialqa-lab.fixed-vs-reachable-nbv-overlay.v1",
        "episode_id": episode_id,
        "short_id": short_id,
        "scene_id": scene_id,
        "fixed_point_count": len(fixed_points),
        "nbv_point_count": len(nbv_points),
        "overlay_png_path": str(paths["fixed_vs_overlay_png"]),
        "trajectory_path": str(paths["trajectory"]),
        "base_topdown_path": str(paths["fixed_episode_topdown"]),
        "mapping_source": "local_bounds_from_fixed_and_nbv_points",
    }
    paths["fixed_vs_overlay_json"].write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True


def _fixed_points(path: Path) -> list[tuple[float, float]]:
    points = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pose = row.get("agent_pose")
        if isinstance(pose, dict) and "x" in pose and "z" in pose:
            points.append((float(pose["x"]), float(pose["z"])))
    return _dedupe(points)


def _dedupe(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for point in points:
        if not out or math.dist(out[-1], point) > 1e-9:
            out.append(point)
    return out


def _bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    zs = [point[1] for point in points]
    return min(xs) - 0.25, max(xs) + 0.25, min(zs) - 0.25, max(zs) + 0.25


def _font(size: int) -> Any:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            from PIL import ImageFont
            return ImageFont.truetype(str(path), size)
    from PIL import ImageFont
    return ImageFont.load_default()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
