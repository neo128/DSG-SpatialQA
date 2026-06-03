from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    compare_real_experiment_readiness_report,
    load_benchmark_manifest,
    load_real_experiment_readiness_report,
    real_experiment_readiness_report,
    save_real_experiment_readiness_report,
    validate_real_experiment_readiness_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Check whether an explicit benchmark manifest has enough local "
            "artifacts for a real DSG-vs-control experiment."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Explicit benchmark manifest JSON to check.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Explicit local readiness report output path.",
    )
    parser.add_argument(
        "--data-source-kind",
        default="unspecified",
        help="Declared data source kind for this experiment package, e.g. real.",
    )
    parser.add_argument(
        "--min-episode-count",
        type=int,
        default=3,
        help="Minimum required episode count.",
    )
    parser.add_argument(
        "--min-scene-count",
        type=int,
        default=1,
        help="Minimum required scene count.",
    )
    parser.add_argument(
        "--min-qa-count",
        type=int,
        default=30,
        help="Minimum required QA case count.",
    )
    parser.add_argument(
        "--required-control-kind",
        action="append",
        dest="required_control_kinds",
        help="Offline control source kind that must be present. May be repeated.",
    )
    parser.add_argument(
        "--required-predicted-input-kind",
        action="append",
        dest="required_predicted_input_kinds",
        help="Predicted graph input kind that must be present. May be repeated.",
    )
    parser.add_argument(
        "--validate-report",
        type=Path,
        help="Validate an explicit real experiment readiness report JSON file.",
    )
    parser.add_argument(
        "--compare-report",
        type=Path,
        help="Compare a readiness report with the current manifest and artifacts.",
    )
    args = parser.parse_args(argv)

    if args.validate_report is not None:
        try:
            validation = validate_real_experiment_readiness_report(
                load_real_experiment_readiness_report(args.validate_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(_error_payload("validate_real_experiment_readiness_report", args.validate_report, exc))
            return 1
        _emit_json(
            {
                "action": "validate_real_experiment_readiness_report",
                "path": str(args.validate_report),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.compare_report is not None:
        try:
            comparison = compare_real_experiment_readiness_report(
                load_real_experiment_readiness_report(args.compare_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    **_error_payload(
                        "compare_real_experiment_readiness_report",
                        args.compare_report,
                        exc,
                    ),
                    "matches": False,
                }
            )
            return 1
        _emit_json(
            {
                "action": "compare_real_experiment_readiness_report",
                "path": str(args.compare_report),
                **comparison,
            }
        )
        return 0 if comparison["matches"] is True else 1

    if args.manifest is None:
        parser.error("--manifest is required")
    if args.report is None:
        parser.error("--report is required")

    try:
        report = real_experiment_readiness_report(
            load_benchmark_manifest(args.manifest),
            manifest_path=args.manifest,
            declared_data_source_kind=args.data_source_kind,
            min_episode_count=args.min_episode_count,
            min_scene_count=args.min_scene_count,
            min_qa_count=args.min_qa_count,
            required_control_kinds=tuple(
                args.required_control_kinds
                or ("caption_memory", "graph_text", "multi_frame_vlm", "vlm")
            ),
            required_predicted_input_kinds=tuple(
                args.required_predicted_input_kinds or ("observation_sequence",)
            ),
        )
        save_real_experiment_readiness_report(report, args.report)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("real_experiment_readiness", args.report, exc))
        return 1

    _emit_json(
        {
            "action": "real_experiment_readiness",
            "path": str(args.report),
            "ready": report["readiness"]["ready"],
            "report_digest": report["report_digest"],
            "readiness": report["readiness"],
        }
    )
    return 0 if report["readiness"]["ready"] is True else 1


def _error_payload(action: str, path: Path, error: Exception) -> dict[str, Any]:
    return {
        "action": action,
        "path": str(path),
        "valid": False,
        "error": str(error),
    }


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
