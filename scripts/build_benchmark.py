from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    benchmark_manifest_digest,
    build_benchmark_artifacts,
    compare_benchmark_manifest,
    load_benchmark_manifest,
    save_benchmark_manifest,
    validate_benchmark_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Build deterministic benchmark graph/QA artifacts and manifests.",
    )
    parser.add_argument(
        "--episodes",
        action="append",
        type=Path,
        help="Explicit episode JSONL path. May be repeated.",
    )
    parser.add_argument("--dataset-name", default="benchmark", help="Manifest dataset name.")
    parser.add_argument("--output-dir", type=Path, help="Explicit benchmark artifact output dir.")
    parser.add_argument("--max-qa-per-episode", type=int, help="QA case limit per episode.")
    parser.add_argument("--tag", action="append", dest="tags", help="Optional QA generation tag.")
    parser.add_argument(
        "--qa-eval-report",
        action="append",
        type=Path,
        dest="qa_eval_reports",
        help="Optional explicit QA eval report path to record in the manifest.",
    )
    parser.add_argument(
        "--qa-eval-delta-report",
        action="append",
        type=Path,
        dest="qa_eval_delta_reports",
        help="Optional explicit QA eval delta report path to record in the manifest.",
    )
    parser.add_argument(
        "--active-task-report",
        action="append",
        type=Path,
        dest="active_task_reports",
        help="Optional explicit active task report path to record in the manifest.",
    )
    parser.add_argument(
        "--active-task-delta-report",
        action="append",
        type=Path,
        dest="active_task_delta_reports",
        help="Optional explicit active task delta report path to record in the manifest.",
    )
    parser.add_argument(
        "--dashboard-bundle",
        action="append",
        type=Path,
        dest="dashboard_bundles",
        help="Optional explicit dashboard bundle JSON path to record in the manifest.",
    )
    parser.add_argument(
        "--error-attribution-report",
        action="append",
        type=Path,
        dest="error_attribution_reports",
        help="Optional explicit error attribution report path to record in the manifest.",
    )
    parser.add_argument(
        "--graph-eval-report",
        action="append",
        type=Path,
        dest="graph_eval_reports",
        help="Optional explicit graph eval report path to record in the manifest.",
    )
    parser.add_argument(
        "--predicted-graph-report",
        action="append",
        type=Path,
        dest="predicted_graph_reports",
        help="Optional explicit predicted graph report path to record in the manifest.",
    )
    parser.add_argument("--manifest", type=Path, help="Explicit manifest JSON output path.")
    parser.add_argument("--validate-manifest", type=Path, help="Validate a benchmark manifest.")
    parser.add_argument("--compare-manifest", type=Path, help="Compare a benchmark manifest.")
    args = parser.parse_args(argv)

    if args.validate_manifest is not None:
        try:
            validation = validate_benchmark_manifest(
                load_benchmark_manifest(args.validate_manifest)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(_error_payload("validate_benchmark_manifest", args.validate_manifest, exc))
            return 1
        _emit_json(
            {
                "action": "validate_benchmark_manifest",
                "path": str(args.validate_manifest),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.compare_manifest is not None:
        try:
            comparison = compare_benchmark_manifest(
                load_benchmark_manifest(args.compare_manifest)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    **_error_payload("compare_benchmark_manifest", args.compare_manifest, exc),
                    "matches": False,
                }
            )
            return 1
        _emit_json(
            {
                "action": "compare_benchmark_manifest",
                "path": str(args.compare_manifest),
                **comparison,
            }
        )
        return 0 if comparison["matches"] is True else 1

    if not args.episodes:
        parser.error("--episodes is required")
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if args.manifest is None:
        parser.error("--manifest is required")

    try:
        manifest = build_benchmark_artifacts(
            dataset_name=args.dataset_name,
            episode_paths=tuple(args.episodes),
            output_dir=args.output_dir,
            max_qa_per_episode=args.max_qa_per_episode,
            tags=tuple(args.tags or ("benchmark", "oracle")),
            qa_eval_report_paths=tuple(args.qa_eval_reports or ()),
            qa_eval_delta_report_paths=tuple(args.qa_eval_delta_reports or ()),
            active_task_report_paths=tuple(args.active_task_reports or ()),
            active_task_delta_report_paths=tuple(args.active_task_delta_reports or ()),
            dashboard_bundle_paths=tuple(args.dashboard_bundles or ()),
            error_attribution_report_paths=tuple(args.error_attribution_reports or ()),
            graph_eval_report_paths=tuple(args.graph_eval_reports or ()),
            predicted_graph_report_paths=tuple(args.predicted_graph_reports or ()),
        )
        save_benchmark_manifest(manifest, args.manifest)
        validation = validate_benchmark_manifest(manifest)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("build_benchmark", args.manifest, exc))
        return 1

    _emit_json(
        {
            "action": "build_benchmark",
            "path": str(args.manifest),
            "valid": validation["valid"],
            "digest": benchmark_manifest_digest(manifest),
            "summary": manifest["summary"],
        }
    )
    return 0 if validation["valid"] is True else 1


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
