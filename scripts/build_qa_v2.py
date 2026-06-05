from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    load_episode_sequence,
    load_qa_dataset,
    load_qa_observability_report,
    load_qa_v2_split_report,
    qa_v2_split_report,
    qa_v2_split_report_digest,
    qa_v2_splits,
    save_qa_v2_split_report,
    save_qa_v2_splits,
    validate_qa_v2_split_report,
    validate_qa_v2_splits,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Build QA v2 split JSONL files from deterministic local QA artifacts.",
    )
    parser.add_argument("--qa", type=Path, help="Explicit local v1 QA JSONL path.")
    parser.add_argument(
        "--episode",
        type=Path,
        action="append",
        default=[],
        help="Optional local episode JSONL path used to fill situation fields.",
    )
    parser.add_argument(
        "--observability-report",
        type=Path,
        help="Optional QA observability report used for split assignment.",
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory for QA v2 split JSONL.")
    parser.add_argument("--report", type=Path, help="QA v2 split report output path.")
    parser.add_argument(
        "--validate-report",
        type=Path,
        help="Validate an explicit QA v2 split report.",
    )
    args = parser.parse_args(argv)

    if args.validate_report is not None:
        try:
            validation = validate_qa_v2_split_report(
                load_qa_v2_split_report(args.validate_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(_error_payload("validate_qa_v2_split_report", args.validate_report, exc))
            return 1
        _emit_json(
            {
                "action": "validate_qa_v2_split_report",
                "path": str(args.validate_report),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.qa is None or args.output_dir is None or args.report is None:
        parser.error("QA v2 build requires --qa, --output-dir, and --report")

    try:
        cases = load_qa_dataset(args.qa)
        episode_frames = tuple(
            frame for episode_path in args.episode for frame in load_episode_sequence(episode_path)
        )
        observability_report = (
            load_qa_observability_report(args.observability_report)
            if args.observability_report is not None
            else None
        )
        splits = qa_v2_splits(
            cases,
            observability_report=observability_report,
            episode_frames=episode_frames,
        )
        split_validation = validate_qa_v2_splits(splits)
        split_paths = save_qa_v2_splits(splits, args.output_dir)
        report = qa_v2_split_report(
            cases,
            splits,
            qa_path=args.qa,
            episode_path=";".join(str(path) for path in args.episode) or None,
            observability_report_path=args.observability_report,
            split_paths=split_paths,
        )
        save_qa_v2_split_report(report, args.report)
        validation = validate_qa_v2_split_report(report)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("build_qa_v2_splits", args.report, exc))
        return 1

    valid = split_validation["valid"] is True and validation["valid"] is True
    _emit_json(
        {
            "action": "build_qa_v2_splits",
            "path": str(args.report),
            "valid": valid,
            "digest": qa_v2_split_report_digest(report),
            "summary": report["summary"],
        }
    )
    return 0 if valid else 1


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
