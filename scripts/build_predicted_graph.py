from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    build_predicted_graph_from_episode,
    compare_predicted_graph_report,
    episode_sequence_digest,
    load_episode_sequence,
    load_predicted_graph_report,
    predicted_graph_report,
    save_graph_json,
    save_predicted_graph_report,
    validate_predicted_graph_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Build or validate deterministic predicted DSGs from mock episode perception.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use EpisodeFrame.metadata['mock_detections'] as the perception source.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Explicit local episode JSONL file to build into a predicted DSG.",
    )
    parser.add_argument(
        "--output-graph",
        type=Path,
        help="Explicit local path where the predicted graph JSON is written.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Explicit local path where the predicted graph report JSON is written.",
    )
    parser.add_argument(
        "--validate-report",
        type=Path,
        help="Validate an explicit local predicted graph report JSON file.",
    )
    parser.add_argument(
        "--compare-report",
        type=Path,
        help="Compare an explicit local predicted graph report with current episode/graph files.",
    )
    args = parser.parse_args(argv)

    if args.validate_report is not None:
        try:
            validation = validate_predicted_graph_report(
                load_predicted_graph_report(args.validate_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload("validate_predicted_graph_report", args.validate_report, exc)
            )
            return 1
        payload = {
            "action": "validate_predicted_graph_report",
            "path": str(args.validate_report),
            **validation,
        }
        _emit_json(payload)
        return 0 if validation["valid"] is True else 1

    if args.compare_report is not None:
        try:
            comparison = compare_predicted_graph_report(
                load_predicted_graph_report(args.compare_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    **_error_payload(
                        "compare_predicted_graph_report",
                        args.compare_report,
                        exc,
                    ),
                    "matches": False,
                }
            )
            return 1
        payload = {
            "action": "compare_predicted_graph_report",
            "path": str(args.compare_report),
            **comparison,
        }
        _emit_json(payload)
        return 0 if comparison["matches"] is True else 1

    if args.input is None or args.output_graph is None:
        parser.error("build requires --input and --output-graph")
    if not args.mock:
        parser.error("build requires --mock")

    try:
        frames = load_episode_sequence(args.input)
        graph = build_predicted_graph_from_episode(frames)
        args.output_graph.parent.mkdir(parents=True, exist_ok=True)
        save_graph_json(graph, args.output_graph)
        report = predicted_graph_report(
            input_path=args.input,
            graph_path=args.output_graph,
            graph=graph,
            frames=frames,
        )
        if args.report is not None:
            save_predicted_graph_report(report, args.report)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("build_predicted_graph", args.input, exc))
        return 1

    payload = {
        **report,
        "episode_digest": episode_sequence_digest(frames),
    }
    _emit_json(payload)
    return 0


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
