from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    compare_episode_sequence,
    episode_sequence_digest,
    episode_sequence_summary,
    load_episode_sequence,
    validate_episode_sequence,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Validate, compare, or summarize deterministic episode JSONL files.",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        help="Explicit local episode JSONL file to validate.",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="Explicit local episode JSONL file to compare against canonical round trip.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Explicit local episode JSONL file to summarize.",
    )
    args = parser.parse_args(argv)

    if args.summary is not None:
        try:
            frames = load_episode_sequence(args.summary)
            validation = validate_episode_sequence(frames)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(_error_payload("summary_episode_sequence", args.summary, exc))
            return 1
        payload = {
            "action": "summary_episode_sequence",
            "path": str(args.summary),
            "valid": validation["valid"],
            "digest": episode_sequence_digest(frames),
            "summary": episode_sequence_summary(frames),
        }
        _emit_json(payload)
        return 0 if validation["valid"] is True else 1

    if args.validate is not None:
        try:
            validation = validate_episode_sequence(load_episode_sequence(args.validate))
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(_error_payload("validate_episode_sequence", args.validate, exc))
            return 1
        payload = {
            "action": "validate_episode_sequence",
            "path": str(args.validate),
            **validation,
        }
        _emit_json(payload)
        return 0 if validation["valid"] is True else 1

    if args.compare is not None:
        try:
            comparison = compare_episode_sequence(load_episode_sequence(args.compare))
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    **_error_payload("compare_episode_sequence", args.compare, exc),
                    "matches": False,
                }
            )
            return 1
        payload = {
            "action": "compare_episode_sequence",
            "path": str(args.compare),
            **comparison,
        }
        _emit_json(payload)
        return 0 if comparison["matches"] is True else 1

    parser.error("episodes requires --validate, --compare, or --summary")


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
