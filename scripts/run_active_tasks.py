from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    ActiveGraphAgent,
    MockActiveEnvironment,
    active_task_delta_report,
    active_task_delta_report_digest,
    active_task_report,
    compare_active_task_delta_report,
    compare_active_task_report,
    load_active_eqa_tasks,
    load_active_task_delta_report,
    load_active_task_report,
    load_graph_json,
    save_active_task_delta_report,
    save_active_task_report,
    validate_active_task_delta_report,
    validate_active_task_report,
)
from dsg_spatialqa_lab.schema import SpatialQAError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run deterministic mock active EQA policies over explicit task JSONL files.",
    )
    parser.add_argument("--tasks", type=Path, help="Explicit local active task JSONL path.")
    parser.add_argument("--graph", type=Path, help="Explicit local graph JSON path.")
    parser.add_argument("--policy", default="direct_answer", help="Active policy name.")
    parser.add_argument("--report", type=Path, help="Active task report JSON output path.")
    parser.add_argument("--validate-report", type=Path, help="Validate an active task report JSON.")
    parser.add_argument(
        "--compare-report",
        type=Path,
        help="Compare an active task report with current task and graph files.",
    )
    parser.add_argument("--candidate-report", type=Path, help="Candidate active task report path.")
    parser.add_argument("--baseline-report", type=Path, help="Baseline active task report path.")
    parser.add_argument("--candidate-name", default="candidate", help="Candidate policy name.")
    parser.add_argument("--baseline-name", default="baseline", help="Baseline policy name.")
    parser.add_argument("--delta-report", type=Path, help="Active task delta report output path.")
    parser.add_argument("--validate-delta-report", type=Path, help="Validate an active task delta report.")
    parser.add_argument(
        "--compare-delta-report",
        type=Path,
        help="Compare an active task delta report with current candidate and baseline reports.",
    )
    args = parser.parse_args(argv)

    if args.validate_delta_report is not None:
        try:
            validation = validate_active_task_delta_report(
                load_active_task_delta_report(args.validate_delta_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    "action": "validate_active_task_delta_report",
                    "path": str(args.validate_delta_report),
                    "valid": False,
                    "error": str(exc),
                }
            )
            return 1
        _emit_json(
            {
                "action": "validate_active_task_delta_report",
                "path": str(args.validate_delta_report),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.compare_delta_report is not None:
        try:
            comparison = compare_active_task_delta_report(
                load_active_task_delta_report(args.compare_delta_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    "action": "compare_active_task_delta_report",
                    "path": str(args.compare_delta_report),
                    "valid": False,
                    "matches": False,
                    "error": str(exc),
                }
            )
            return 1
        _emit_json(
            {
                "action": "compare_active_task_delta_report",
                "path": str(args.compare_delta_report),
                **comparison,
            }
        )
        return 0 if comparison["matches"] is True else 1

    if (
        args.candidate_report is not None
        or args.baseline_report is not None
        or args.delta_report is not None
    ):
        if args.candidate_report is None:
            parser.error("--candidate-report is required when generating a delta report")
        if args.baseline_report is None:
            parser.error("--baseline-report is required when generating a delta report")
        if args.delta_report is None:
            parser.error("--delta-report is required when generating a delta report")
        try:
            delta_report = active_task_delta_report(
                load_active_task_report(args.candidate_report),
                load_active_task_report(args.baseline_report),
                candidate_name=args.candidate_name,
                baseline_name=args.baseline_name,
                candidate_report_path=args.candidate_report,
                baseline_report_path=args.baseline_report,
            )
            save_active_task_delta_report(delta_report, args.delta_report)
            validation = validate_active_task_delta_report(delta_report)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    "action": "active_task_delta_report",
                    "path": str(args.delta_report),
                    "valid": False,
                    "error": str(exc),
                }
            )
            return 1
        _emit_json(
            {
                "action": "active_task_delta_report",
                "path": str(args.delta_report),
                "valid": validation["valid"],
                "digest": active_task_delta_report_digest(delta_report),
                "summary_delta": delta_report["summary_delta"],
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.validate_report is not None:
        try:
            validation = validate_active_task_report(load_active_task_report(args.validate_report))
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    "action": "validate_active_task_report",
                    "path": str(args.validate_report),
                    "valid": False,
                    "error": str(exc),
                }
            )
            return 1
        _emit_json(
            {
                "action": "validate_active_task_report",
                "path": str(args.validate_report),
                **validation,
            }
        )
        return 0

    if args.compare_report is not None:
        try:
            comparison = compare_active_task_report(load_active_task_report(args.compare_report))
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    "action": "compare_active_task_report",
                    "path": str(args.compare_report),
                    "valid": False,
                    "matches": False,
                    "error": str(exc),
                }
            )
            return 1
        _emit_json(
            {
                "action": "compare_active_task_report",
                "path": str(args.compare_report),
                **comparison,
            }
        )
        return 0 if comparison["matches"] is True else 1

    if args.tasks is None:
        parser.error("--tasks is required")
    if args.graph is None:
        parser.error("--graph is required")
    if args.report is None:
        parser.error("--report is required")

    try:
        tasks = load_active_eqa_tasks(args.tasks)
        graph = load_graph_json(args.graph)
        agent = ActiveGraphAgent(policy=args.policy)
        results = [
            agent.run(task, MockActiveEnvironment({task.initial_step: graph})) for task in tasks
        ]
        report = active_task_report(
            tasks,
            results,
            task_path=args.tasks,
            graph_path=args.graph,
            policy=args.policy,
        )
        save_active_task_report(report, args.report)
        validation = validate_active_task_report(report)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(
            {
                "action": "active_task_report",
                "path": str(args.report),
                "valid": False,
                "error": str(exc),
            }
        )
        return 1

    _emit_json(
        {
            "action": "active_task_report",
            "path": str(args.report),
            "valid": validation["valid"],
            "digest": report["report_digest"],
            "summary": report["summary"],
            "metrics": report["metrics"],
        }
    )
    return 0


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
