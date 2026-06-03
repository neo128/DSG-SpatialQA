from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import SpatialQAError, assemble_real_experiment_package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Assemble explicit local real-experiment inputs into a benchmark "
            "manifest and readiness report."
        ),
    )
    parser.add_argument(
        "--episode",
        "--episodes",
        action="append",
        type=Path,
        dest="episodes",
        help="Explicit episode JSONL path. May be repeated.",
    )
    parser.add_argument("--dataset-name", default="real_experiment")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--readiness-report", type=Path)
    parser.add_argument("--max-qa-per-episode", type=int)
    parser.add_argument("--tag", action="append", dest="tags")
    parser.add_argument("--data-source-kind", default="real")
    parser.add_argument("--min-episode-count", type=int, default=3)
    parser.add_argument("--min-scene-count", type=int, default=1)
    parser.add_argument("--min-qa-count", type=int, default=30)
    parser.add_argument(
        "--required-control-kind",
        action="append",
        dest="required_control_kinds",
    )
    parser.add_argument(
        "--required-predicted-input-kind",
        action="append",
        dest="required_predicted_input_kinds",
    )
    parser.add_argument(
        "--qa-eval-report",
        action="append",
        type=Path,
        dest="qa_eval_reports",
    )
    parser.add_argument(
        "--qa-eval-delta-report",
        action="append",
        type=Path,
        dest="qa_eval_delta_reports",
    )
    parser.add_argument(
        "--active-task-report",
        action="append",
        type=Path,
        dest="active_task_reports",
    )
    parser.add_argument(
        "--active-task-delta-report",
        action="append",
        type=Path,
        dest="active_task_delta_reports",
    )
    parser.add_argument(
        "--dashboard-bundle",
        action="append",
        type=Path,
        dest="dashboard_bundles",
    )
    parser.add_argument(
        "--error-attribution-report",
        action="append",
        type=Path,
        dest="error_attribution_reports",
    )
    parser.add_argument(
        "--graph-eval-report",
        action="append",
        type=Path,
        dest="graph_eval_reports",
    )
    parser.add_argument(
        "--offline-prediction-import-report",
        action="append",
        type=Path,
        dest="offline_prediction_import_reports",
    )
    parser.add_argument(
        "--offline-control-matrix-report",
        action="append",
        type=Path,
        dest="offline_control_matrix_reports",
    )
    parser.add_argument(
        "--offline-control-result-report",
        action="append",
        type=Path,
        dest="offline_control_result_reports",
    )
    parser.add_argument(
        "--predicted-graph-report",
        action="append",
        type=Path,
        dest="predicted_graph_reports",
    )
    parser.add_argument(
        "--predicted-dsg-evidence-report",
        action="append",
        type=Path,
        dest="predicted_dsg_evidence_reports",
    )
    parser.add_argument(
        "--real-collection-report",
        action="append",
        type=Path,
        dest="real_collection_reports",
    )
    args = parser.parse_args(argv)

    if not args.episodes:
        parser.error("--episode is required")
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if args.manifest is None:
        parser.error("--manifest is required")
    if args.readiness_report is None:
        parser.error("--readiness-report is required")

    try:
        result = assemble_real_experiment_package(
            dataset_name=args.dataset_name,
            episode_paths=tuple(args.episodes),
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            readiness_report_path=args.readiness_report,
            max_qa_per_episode=args.max_qa_per_episode,
            tags=tuple(args.tags or ("benchmark", "real")),
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
            qa_eval_report_paths=tuple(args.qa_eval_reports or ()),
            qa_eval_delta_report_paths=tuple(args.qa_eval_delta_reports or ()),
            active_task_report_paths=tuple(args.active_task_reports or ()),
            active_task_delta_report_paths=tuple(args.active_task_delta_reports or ()),
            dashboard_bundle_paths=tuple(args.dashboard_bundles or ()),
            error_attribution_report_paths=tuple(args.error_attribution_reports or ()),
            graph_eval_report_paths=tuple(args.graph_eval_reports or ()),
            offline_control_matrix_report_paths=tuple(
                args.offline_control_matrix_reports or ()
            ),
            offline_control_result_report_paths=tuple(
                args.offline_control_result_reports or ()
            ),
            offline_prediction_import_report_paths=tuple(
                args.offline_prediction_import_reports or ()
            ),
            predicted_dsg_evidence_report_paths=tuple(
                args.predicted_dsg_evidence_reports or ()
            ),
            predicted_graph_report_paths=tuple(args.predicted_graph_reports or ()),
            real_collection_report_paths=tuple(args.real_collection_reports or ()),
        )
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("assemble_real_experiment_package", args.manifest, exc))
        return 1

    _emit_json(result)
    return 0 if result["ready"] is True else 1


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
