from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    AI2ThorAdapterConfig,
    AI2ThorEpisodeCollector,
    SpatialQAError,
    build_mock_ai2thor_episode,
    episode_sequence_digest,
    episode_sequence_summary,
    save_episode_sequence,
    validate_episode_sequence,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Collect deterministic AI2-THOR adapter episode JSONL artifacts.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the deterministic local mock adapter instead of a simulator.",
    )
    parser.add_argument("--scene", required=True, help="Explicit scene id.")
    parser.add_argument("--episode-id", required=True, help="Explicit episode id.")
    parser.add_argument(
        "--step",
        action="append",
        type=int,
        required=True,
        dest="steps",
        help="Explicit episode step. May be repeated.",
    )
    parser.add_argument(
        "--action",
        action="append",
        dest="actions",
        help="Optional action name for each explicit step. May be repeated.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="MVP adapter accepts exactly one explicit episode.",
    )
    parser.add_argument(
        "--artifact-root",
        help="Optional local artifact root used to fill mock RGB/depth/segmentation paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Explicit local episode JSONL output path.",
    )
    args = parser.parse_args(argv)

    if args.episodes != 1:
        parser.error("AI2-THOR adapter MVP accepts --episodes 1 only")

    config = AI2ThorAdapterConfig(
        scene_id=args.scene,
        episode_id=args.episode_id,
        steps=tuple(args.steps),
        actions=tuple(args.actions or ()),
        artifact_root=args.artifact_root,
    )

    try:
        if args.mock:
            frames = build_mock_ai2thor_episode(config)
            save_episode_sequence(frames, args.output)
            validation = validate_episode_sequence(frames)
            payload = {
                "action": "collect_ai2thor_mock",
                "path": str(args.output),
                "valid": validation["valid"],
                "digest": episode_sequence_digest(frames),
                "summary": episode_sequence_summary(frames),
            }
            _emit_json(payload)
            return 0 if validation["valid"] is True else 1

        frames = AI2ThorEpisodeCollector(config).collect_episode()
        save_episode_sequence(frames, args.output)
        validation = validate_episode_sequence(frames)
        payload = {
            "action": "collect_ai2thor_episode",
            "path": str(args.output),
            "valid": validation["valid"],
            "digest": episode_sequence_digest(frames),
            "summary": episode_sequence_summary(frames),
        }
        _emit_json(payload)
        return 0 if validation["valid"] is True else 1
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        action = "collect_ai2thor_mock" if args.mock else "collect_ai2thor_episode"
        _emit_json(_error_payload(action, args.output, exc))
        return 1


def _error_payload(action: str, path: Path, error: Exception) -> dict[str, Any]:
    return {
        "action": action,
        "path": str(path),
        "valid": False,
        "error": str(error),
    }


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
