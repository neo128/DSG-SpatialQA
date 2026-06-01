from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    compare_observation_ingest_report,
    ingest_scene_observation_sequence,
    load_observation_ingest_report,
    load_scene_observation_sequence,
    observation_ingest_report,
    observation_ingest_report_json,
    save_graph_json,
    scene_observation_sequence_digest,
    validate_observation_ingest_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest deterministic DSG-SpatialQA SceneObservation sequence JSON "
            "from an explicit local file and export graph JSON."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Explicit local SceneObservation sequence JSON file to ingest.",
    )
    parser.add_argument(
        "--output-graph",
        type=Path,
        help="Explicit local path where the ingested scene graph JSON is written.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional explicit local path where the structured ingest report is written.",
    )
    parser.add_argument(
        "--infer-relation",
        action="append",
        dest="infer_relations",
        help="Spatial relation name to infer after each observation. May be repeated.",
    )
    parser.add_argument(
        "--reference-frame",
        action="append",
        dest="reference_frames",
        help=(
            "Reference frame for inferred relations. May be repeated. "
            "Defaults to world for CLI ingestion."
        ),
    )
    parser.add_argument(
        "--validate-report",
        type=Path,
        help="Validate an explicit local observation ingest report JSON file.",
    )
    parser.add_argument(
        "--compare-report",
        type=Path,
        help="Compare an explicit local observation ingest report with a current local re-ingest.",
    )
    args = parser.parse_args(argv)

    if args.validate_report is not None:
        try:
            validation = validate_observation_ingest_report(
                load_observation_ingest_report(args.validate_report)
            )
        except (OSError, SpatialQAError, ValueError) as exc:
            payload = _error_report(
                args.validate_report,
                None,
                exc,
                action="validate_observation_ingest_report",
            )
            _emit_json_payload(payload, args.report)
            return 1
        payload = {
            "action": "validate_observation_ingest_report",
            "path": str(args.validate_report),
            **validation,
        }
        _emit_json_payload(payload, args.report)
        return 0 if validation["valid"] is True else 1

    if args.compare_report is not None:
        try:
            comparison = compare_observation_ingest_report(
                load_observation_ingest_report(args.compare_report)
            )
        except (OSError, SpatialQAError, ValueError) as exc:
            payload = _error_report(
                args.compare_report,
                None,
                exc,
                action="compare_observation_ingest_report",
                matches=False,
            )
            _emit_json_payload(payload, args.report)
            return 1
        payload = {
            "action": "compare_observation_ingest_report",
            "path": str(args.compare_report),
            **comparison,
        }
        _emit_json_payload(payload, args.report)
        return 0 if comparison["matches"] is True else 1

    if args.input is None or args.output_graph is None:
        parser.error("ingest requires --input and --output-graph")

    try:
        observations = load_scene_observation_sequence(args.input)
        sequence_digest = scene_observation_sequence_digest(observations)
        infer_relations = tuple(args.infer_relations or ())
        reference_frames = tuple(args.reference_frames or ("world",))
        graph, ingest_results = ingest_scene_observation_sequence(
            observations,
            source_path=args.input,
            infer_relations=infer_relations,
            reference_frames=reference_frames,
        )
        args.output_graph.parent.mkdir(parents=True, exist_ok=True)
        save_graph_json(graph, args.output_graph)
    except (OSError, SpatialQAError, ValueError) as exc:
        payload = _error_report(args.input, args.output_graph, exc)
        _emit_json_payload(payload, args.report)
        return 1

    payload = observation_ingest_report(
        input_path=args.input,
        graph_path=args.output_graph,
        graph=graph,
        ingest_results=ingest_results,
        sequence_digest=sequence_digest,
        infer_relations=infer_relations,
        reference_frames=reference_frames,
    )
    _emit_json_payload(payload, args.report)
    return 0


def _error_report(
    input_path: Path,
    graph_path: Path | None,
    error: Exception,
    *,
    action: str = "ingest_observation_sequence",
    matches: bool | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "action": action,
        "path": str(input_path),
        "valid": False,
        "error": str(error),
    }
    if graph_path is not None:
        report["graph_path"] = str(graph_path)
    if matches is not None:
        report["matches"] = matches
    return report


def _emit_json_payload(payload: dict[str, Any], report_path: Path | None = None) -> None:
    payload_json = observation_ingest_report_json(payload)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload_json, encoding="utf-8")
    print(payload_json, end="")


if __name__ == "__main__":
    raise SystemExit(main())
