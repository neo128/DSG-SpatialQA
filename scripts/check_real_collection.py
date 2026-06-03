from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    compare_real_collection_request_bundle,
    compare_real_collection_report,
    load_real_collection_request_bundle,
    load_real_collection_report,
    real_collection_request_bundle,
    real_collection_report,
    save_real_collection_request_bundle,
    save_real_collection_report,
    validate_real_collection_request_bundle,
    validate_real_collection_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Check whether explicit local episode files provide real "
            "AI2-THOR/Habitat collection evidence."
        ),
    )
    parser.add_argument("--dataset-name", default="real_collection")
    parser.add_argument(
        "--episode",
        "--episodes",
        action="append",
        type=Path,
        dest="episodes",
    )
    parser.add_argument("--source-kind", default="unspecified")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--min-episode-count", type=int, default=3)
    parser.add_argument("--min-scene-count", type=int, default=1)
    parser.add_argument("--min-frame-count", type=int, default=30)
    parser.add_argument(
        "--required-frame-evidence",
        action="append",
        dest="required_frame_evidence",
    )
    parser.add_argument("--validate-report", type=Path)
    parser.add_argument("--compare-report", type=Path)
    parser.add_argument("--request-bundle", type=Path)
    parser.add_argument("--validate-request-bundle", type=Path)
    parser.add_argument("--compare-request-bundle", type=Path)
    args = parser.parse_args(argv)

    if args.validate_request_bundle is not None:
        try:
            validation = validate_real_collection_request_bundle(
                load_real_collection_request_bundle(args.validate_request_bundle)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_real_collection_request_bundle",
                    args.validate_request_bundle,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "validate_real_collection_request_bundle",
                "path": str(args.validate_request_bundle),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.compare_request_bundle is not None:
        try:
            comparison = compare_real_collection_request_bundle(
                load_real_collection_request_bundle(args.compare_request_bundle)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    **_error_payload(
                        "compare_real_collection_request_bundle",
                        args.compare_request_bundle,
                        exc,
                    ),
                    "matches": False,
                }
            )
            return 1
        _emit_json(
            {
                "action": "compare_real_collection_request_bundle",
                "path": str(args.compare_request_bundle),
                **comparison,
            }
        )
        return 0 if comparison["matches"] is True else 1

    if args.validate_report is not None:
        try:
            validation = validate_real_collection_report(
                load_real_collection_report(args.validate_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload("validate_real_collection_report", args.validate_report, exc)
            )
            return 1
        _emit_json(
            {
                "action": "validate_real_collection_report",
                "path": str(args.validate_report),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.compare_report is not None:
        try:
            comparison = compare_real_collection_report(
                load_real_collection_report(args.compare_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    **_error_payload(
                        "compare_real_collection_report",
                        args.compare_report,
                        exc,
                    ),
                    "matches": False,
                }
            )
            return 1
        _emit_json(
            {
                "action": "compare_real_collection_report",
                "path": str(args.compare_report),
                **comparison,
            }
        )
        return 0 if comparison["matches"] is True else 1

    if args.request_bundle is not None:
        if not args.episodes:
            _emit_json(_missing_argument_payload("--episode", args.request_bundle))
            return 1
        if args.report is None:
            _emit_json(_missing_argument_payload("--report", args.request_bundle))
            return 1
        try:
            bundle = real_collection_request_bundle(
                dataset_name=args.dataset_name,
                episode_paths=tuple(args.episodes),
                source_kind=args.source_kind,
                report_path=args.report,
                min_episode_count=args.min_episode_count,
                min_scene_count=args.min_scene_count,
                min_frame_count=args.min_frame_count,
                required_frame_evidence=tuple(
                    args.required_frame_evidence or ("depth", "rgb", "segmentation")
                ),
            )
            save_real_collection_request_bundle(bundle, args.request_bundle)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "real_collection_request_bundle",
                    args.request_bundle,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "real_collection_request_bundle",
                "request_bundle_path": str(args.request_bundle),
                "bundle": bundle,
            }
        )
        return 0

    if not args.episodes:
        _emit_json(_missing_argument_payload("--episode", args.report))
        return 1
    if args.report is None:
        _emit_json(_missing_argument_payload("--report", None))
        return 1

    try:
        report = real_collection_report(
            dataset_name=args.dataset_name,
            episode_paths=tuple(args.episodes),
            source_kind=args.source_kind,
            min_episode_count=args.min_episode_count,
            min_scene_count=args.min_scene_count,
            min_frame_count=args.min_frame_count,
            required_frame_evidence=tuple(
                args.required_frame_evidence or ("depth", "rgb", "segmentation")
            ),
        )
        save_real_collection_report(report, args.report)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("real_collection", args.report, exc))
        return 1

    _emit_json(
        {
            "action": "real_collection",
            "path": str(args.report),
            "ready": report["readiness"]["ready"],
            "report_digest": report["report_digest"],
            "readiness": report["readiness"],
            "summary": report["collection_summary"],
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


def _missing_argument_payload(argument: str, path: Path | None) -> dict[str, Any]:
    return {
        "action": "real_collection",
        "path": str(path) if path is not None else "",
        "valid": False,
        "error": f"{argument} is required",
    }


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
