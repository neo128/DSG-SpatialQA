from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    ActiveGraphAgent,
    MockActiveEnvironment,
    active_task_report,
    load_active_eqa_tasks,
    load_active_task_report,
    load_graph_json,
    save_active_task_report,
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
    args = parser.parse_args(argv)

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
