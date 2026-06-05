from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from dsg_spatialqa_lab import (
    SpatialQAError,
    compare_coverage_collection_plan,
    coverage_collection_acceptance_report,
    coverage_collection_acceptance_report_digest,
    coverage_collection_plan,
    coverage_collection_plan_digest,
    coverage_collection_request_bundle,
    coverage_collection_request_bundle_digest,
    coverage_collection_top_batch_handoff_tasks,
    coverage_collection_top_batch_handoff_tasks_digest,
    coverage_collection_top_batch_return_report,
    coverage_collection_top_batch_return_report_digest,
    detector_observation_sequence_from_jsonl,
    load_coverage_collection_acceptance_report,
    load_coverage_collection_plan,
    load_coverage_collection_request_bundle,
    load_coverage_collection_top_batch_handoff_tasks,
    load_coverage_collection_top_batch_return_report,
    load_episode_sequence,
    load_qa_observability_report,
    load_scene_observation_sequence,
    save_coverage_collection_acceptance_report,
    save_coverage_collection_plan,
    save_coverage_collection_request_bundle,
    save_coverage_collection_top_batch_handoff_tasks,
    save_coverage_collection_top_batch_return_report,
    validate_coverage_collection_acceptance_report,
    validate_coverage_collection_plan,
    validate_coverage_collection_request_bundle,
    validate_coverage_collection_top_batch_handoff_tasks,
    validate_coverage_collection_top_batch_return_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Build a deterministic coverage-driven collection plan from QA "
            "observability gaps and explicit local real episode JSONL files."
        ),
    )
    parser.add_argument("--qa-observability-report", type=Path)
    parser.add_argument(
        "--episode",
        action="append",
        dest="episodes",
        type=Path,
        help="Explicit local episode JSONL path. May be repeated.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--target-evidence-observable-count",
        type=int,
        default=30,
    )
    parser.add_argument("--target-node-recall-floor", type=float, default=0.5)
    parser.add_argument("--validate-plan", type=Path)
    parser.add_argument("--compare-plan", type=Path)
    parser.add_argument("--acceptance-plan", type=Path)
    parser.add_argument("--detector-jsonl", type=Path)
    parser.add_argument("--observation-sequence", type=Path)
    parser.add_argument("--acceptance-report", type=Path)
    parser.add_argument("--validate-acceptance-report", type=Path)
    parser.add_argument("--request-plan", type=Path)
    parser.add_argument("--request-bundle", type=Path)
    parser.add_argument("--detector-jsonl-output", type=Path)
    parser.add_argument("--observation-sequence-output", type=Path)
    parser.add_argument("--acceptance-report-output", type=Path)
    parser.add_argument("--validate-request-bundle", type=Path)
    parser.add_argument("--top-batch-handoff-request-bundle", type=Path)
    parser.add_argument("--top-batch-handoff-jsonl", type=Path)
    parser.add_argument("--max-priority-batches", type=int, default=5)
    parser.add_argument("--validate-top-batch-handoff", type=Path)
    parser.add_argument("--top-batch-return-tasks", type=Path)
    parser.add_argument("--top-batch-return-report", type=Path)
    parser.add_argument("--validate-top-batch-return-report", type=Path)
    args = parser.parse_args(argv)

    if args.validate_top_batch_return_report is not None:
        try:
            report = load_coverage_collection_top_batch_return_report(
                args.validate_top_batch_return_report
            )
            validation = validate_coverage_collection_top_batch_return_report(report)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_coverage_collection_top_batch_return_report",
                    args.validate_top_batch_return_report,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "validate_coverage_collection_top_batch_return_report",
                "path": str(args.validate_top_batch_return_report),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.validate_top_batch_handoff is not None:
        try:
            tasks = load_coverage_collection_top_batch_handoff_tasks(
                args.validate_top_batch_handoff
            )
            validation = validate_coverage_collection_top_batch_handoff_tasks(tasks)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_coverage_collection_top_batch_handoff",
                    args.validate_top_batch_handoff,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "validate_coverage_collection_top_batch_handoff",
                "path": str(args.validate_top_batch_handoff),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.validate_request_bundle is not None:
        try:
            validation = validate_coverage_collection_request_bundle(
                load_coverage_collection_request_bundle(args.validate_request_bundle)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_coverage_collection_request_bundle",
                    args.validate_request_bundle,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "validate_coverage_collection_request_bundle",
                "path": str(args.validate_request_bundle),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.validate_acceptance_report is not None:
        try:
            validation = validate_coverage_collection_acceptance_report(
                load_coverage_collection_acceptance_report(
                    args.validate_acceptance_report
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_coverage_collection_acceptance_report",
                    args.validate_acceptance_report,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "validate_coverage_collection_acceptance_report",
                "path": str(args.validate_acceptance_report),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.validate_plan is not None:
        try:
            validation = validate_coverage_collection_plan(
                load_coverage_collection_plan(args.validate_plan)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(_error_payload("validate_coverage_collection_plan", args.validate_plan, exc))
            return 1
        _emit_json(
            {
                "action": "validate_coverage_collection_plan",
                "path": str(args.validate_plan),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.compare_plan is not None:
        try:
            comparison = compare_coverage_collection_plan(
                load_coverage_collection_plan(args.compare_plan)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    **_error_payload(
                        "compare_coverage_collection_plan",
                        args.compare_plan,
                        exc,
                    ),
                    "matches": False,
                }
            )
            return 1
        _emit_json(
            {
                "action": "compare_coverage_collection_plan",
                "path": str(args.compare_plan),
                **comparison,
            }
        )
        return 0 if comparison["matches"] is True else 1

    if args.top_batch_return_tasks is not None:
        if args.top_batch_return_report is None:
            parser.error("--top-batch-return-tasks requires --top-batch-return-report")
        if (args.detector_jsonl is None) == (args.observation_sequence is None):
            parser.error(
                "--top-batch-return-tasks requires exactly one of --detector-jsonl "
                "or --observation-sequence"
            )
        try:
            tasks = load_coverage_collection_top_batch_handoff_tasks(
                args.top_batch_return_tasks
            )
            if args.detector_jsonl is not None:
                observations = detector_observation_sequence_from_jsonl(
                    args.detector_jsonl.read_text(encoding="utf-8")
                )
                observation_path = args.detector_jsonl
            else:
                assert args.observation_sequence is not None
                observations = load_scene_observation_sequence(args.observation_sequence)
                observation_path = args.observation_sequence
            report = coverage_collection_top_batch_return_report(
                tasks,
                observations,
                observation_sequence_path=observation_path,
            )
            save_coverage_collection_top_batch_return_report(
                report,
                args.top_batch_return_report,
            )
            validation = validate_coverage_collection_top_batch_return_report(report)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "coverage_collection_top_batch_return_report",
                    args.top_batch_return_report,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "coverage_collection_top_batch_return_report",
                "path": str(args.top_batch_return_report),
                "valid": validation["valid"],
                "digest": coverage_collection_top_batch_return_report_digest(report),
                "summary": report["summary"],
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.acceptance_plan is not None:
        if args.acceptance_report is None:
            parser.error("--acceptance-plan requires --acceptance-report")
        if (args.detector_jsonl is None) == (args.observation_sequence is None):
            parser.error(
                "--acceptance-plan requires exactly one of --detector-jsonl "
                "or --observation-sequence"
            )
        try:
            plan = load_coverage_collection_plan(args.acceptance_plan)
            if args.detector_jsonl is not None:
                observations = detector_observation_sequence_from_jsonl(
                    args.detector_jsonl.read_text(encoding="utf-8")
                )
                observation_path = args.detector_jsonl
            else:
                assert args.observation_sequence is not None
                observations = load_scene_observation_sequence(args.observation_sequence)
                observation_path = args.observation_sequence
            report = coverage_collection_acceptance_report(
                plan,
                observations,
                observation_sequence_path=observation_path,
            )
            save_coverage_collection_acceptance_report(
                report,
                args.acceptance_report,
            )
            validation = validate_coverage_collection_acceptance_report(report)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "coverage_collection_acceptance_report",
                    args.acceptance_report,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "coverage_collection_acceptance_report",
                "path": str(args.acceptance_report),
                "valid": validation["valid"],
                "digest": coverage_collection_acceptance_report_digest(report),
                "summary": report["summary"],
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.top_batch_handoff_request_bundle is not None:
        if args.top_batch_handoff_jsonl is None:
            parser.error(
                "--top-batch-handoff-request-bundle requires "
                "--top-batch-handoff-jsonl"
            )
        try:
            tasks = coverage_collection_top_batch_handoff_tasks(
                load_coverage_collection_request_bundle(
                    args.top_batch_handoff_request_bundle
                ),
                max_priority_batches=args.max_priority_batches,
            )
            save_coverage_collection_top_batch_handoff_tasks(
                tasks,
                args.top_batch_handoff_jsonl,
            )
            validation = validate_coverage_collection_top_batch_handoff_tasks(tasks)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "coverage_collection_top_batch_handoff",
                    args.top_batch_handoff_jsonl,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "coverage_collection_top_batch_handoff",
                "path": str(args.top_batch_handoff_jsonl),
                "valid": validation["valid"],
                "digest": coverage_collection_top_batch_handoff_tasks_digest(tasks),
                "task_count": validation["task_count"],
                "batch_count": validation["batch_count"],
                "related_case_count": validation["related_case_count"],
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.request_plan is not None:
        if args.request_bundle is None:
            parser.error("--request-plan requires --request-bundle")
        missing_outputs = [
            name
            for name, value in (
                ("--detector-jsonl-output", args.detector_jsonl_output),
                ("--observation-sequence-output", args.observation_sequence_output),
                ("--acceptance-report-output", args.acceptance_report_output),
            )
            if value is None
        ]
        if missing_outputs:
            parser.error("--request-plan requires " + ", ".join(missing_outputs))
        try:
            assert args.detector_jsonl_output is not None
            assert args.observation_sequence_output is not None
            assert args.acceptance_report_output is not None
            bundle = coverage_collection_request_bundle(
                load_coverage_collection_plan(args.request_plan),
                detector_jsonl_output_path=args.detector_jsonl_output,
                observation_sequence_output_path=args.observation_sequence_output,
                acceptance_report_output_path=args.acceptance_report_output,
            )
            save_coverage_collection_request_bundle(bundle, args.request_bundle)
            validation = validate_coverage_collection_request_bundle(bundle)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "coverage_collection_request_bundle",
                    args.request_bundle,
                    exc,
                )
            )
            return 1
        _emit_json(
            {
                "action": "coverage_collection_request_bundle",
                "path": str(args.request_bundle),
                "valid": validation["valid"],
                "digest": coverage_collection_request_bundle_digest(bundle),
                "summary": bundle["summary"],
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.qa_observability_report is None or args.output is None:
        parser.error("generation requires --qa-observability-report and --output")
    if not args.episodes:
        parser.error("generation requires at least one --episode")

    try:
        frames = tuple(
            frame
            for episode_path in args.episodes
            for frame in load_episode_sequence(episode_path)
        )
        plan = coverage_collection_plan(
            load_qa_observability_report(args.qa_observability_report),
            frames,
            qa_observability_report_path=args.qa_observability_report,
            episode_paths=tuple(args.episodes),
            target_evidence_observable_count=args.target_evidence_observable_count,
            target_node_recall_floor=args.target_node_recall_floor,
        )
        save_coverage_collection_plan(plan, args.output)
        validation = validate_coverage_collection_plan(plan)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("coverage_collection_plan", args.output, exc))
        return 1

    _emit_json(
        {
            "action": "coverage_collection_plan",
            "path": str(args.output),
            "valid": validation["valid"],
            "digest": coverage_collection_plan_digest(plan),
            "summary": plan["summary"],
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


def _emit_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
