from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    build_predicted_graph_from_episode,
    build_predicted_graph_from_observations,
    compare_predicted_graph_report,
    load_episode_sequence,
    load_predicted_graph_report,
    load_scene_observation_sequence,
    predicted_graph_report,
    predicted_graph_report_from_observations,
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
        "--input-kind",
        choices=("episode", "observation_sequence"),
        default="episode",
        help=(
            "Input artifact kind. Use episode with --mock for metadata mock detections, "
            "or observation_sequence for explicit detector/RGB-D observation artifacts."
        ),
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
    parser.add_argument(
        "--infer-relation",
        action="append",
        dest="infer_relations",
        help=(
            "Spatial relation name to infer for observation_sequence input. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--reference-frame",
        action="append",
        dest="reference_frames",
        help=(
            "Reference frame for inferred observation_sequence relations. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--infer-containment",
        action="store_true",
        help=(
            "Infer observation_sequence containment relations "
            "(IN_REGION, IN_ROOM, ON)."
        ),
    )
    parser.add_argument(
        "--containment-axis",
        choices=("z", "y"),
        default="z",
        help=(
            "Vertical axis for ON containment inference. Use y for AI2-THOR "
            "metadata coordinates and z for the default local graph frame."
        ),
    )
    parser.add_argument(
        "--relation-top-k",
        type=int,
        help=(
            "Optional per-object top-k cap for NEAR relation inference on "
            "observation_sequence input."
        ),
    )
    parser.add_argument(
        "--require-detector-state-evidence",
        action="store_true",
        help=(
            "Require any object attributes.states used by observation_sequence "
            "input to come from visible detector RGB-D evidence."
        ),
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
    if args.input_kind == "episode" and not args.mock:
        parser.error("episode build requires --mock")

    try:
        if args.input_kind == "observation_sequence":
            observations = load_scene_observation_sequence(args.input)
            infer_relations = tuple(
                args.infer_relations or ("LEFT_OF", "RIGHT_OF", "NEAR")
            )
            reference_frames = tuple(args.reference_frames or ("world",))
            graph = build_predicted_graph_from_observations(
                observations,
                source_path=args.input,
                infer_relations=infer_relations,
                reference_frames=reference_frames,
                infer_containment=args.infer_containment,
                containment_axis=args.containment_axis,
                relation_top_k=args.relation_top_k,
                require_detector_state_evidence=(
                    args.require_detector_state_evidence
                ),
            )
            report = predicted_graph_report_from_observations(
                input_path=args.input,
                graph_path=args.output_graph,
                graph=graph,
                observations=observations,
                infer_relations=infer_relations,
                reference_frames=reference_frames,
                infer_containment=args.infer_containment,
                containment_axis=args.containment_axis,
                relation_top_k=args.relation_top_k,
                require_detector_state_evidence=(
                    args.require_detector_state_evidence
                ),
            )
        else:
            frames = load_episode_sequence(args.input)
            graph = build_predicted_graph_from_episode(frames)
            report = predicted_graph_report(
                input_path=args.input,
                graph_path=args.output_graph,
                graph=graph,
                frames=frames,
            )
        args.output_graph.parent.mkdir(parents=True, exist_ok=True)
        save_graph_json(graph, args.output_graph)
        if args.report is not None:
            save_predicted_graph_report(report, args.report)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("build_predicted_graph", args.input, exc))
        return 1

    _emit_json(dict(report))
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
