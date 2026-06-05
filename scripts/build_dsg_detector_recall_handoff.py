from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from dsg_spatialqa_lab.eval.dsg_detector_recall import (
    dsg_detector_recall_handoff,
    dsg_detector_recall_handoff_from_query_diagnostics,
    load_dsg_detector_recall_handoff,
    save_dsg_detector_recall_handoff,
    validate_dsg_detector_recall_handoff,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Build a gold-free detector recall handoff from DSG support/target "
            "gap diagnostics and explicit local frame-index rows."
        ),
    )
    parser.add_argument("--gap-report", type=Path)
    parser.add_argument("--query-diagnostic-report", type=Path)
    parser.add_argument("--frame-index-jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-report", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.validate_report is not None:
            handoff = load_dsg_detector_recall_handoff(args.validate_report)
            validation = validate_dsg_detector_recall_handoff(handoff)
            _emit_json(
                {
                    "action": "validate_dsg_detector_recall_handoff",
                    "path": str(args.validate_report),
                    **validation,
                }
            )
            return 0 if validation["valid"] is True else 1

        if args.gap_report is None and args.query_diagnostic_report is None:
            _emit_json(_missing_argument_payload("--gap-report or --query-diagnostic-report"))
            return 1
        if args.gap_report is not None and args.query_diagnostic_report is not None:
            _emit_json(
                {
                    "action": "build_dsg_detector_recall_handoff",
                    "error": (
                        "--gap-report and --query-diagnostic-report are mutually "
                        "exclusive"
                    ),
                    "valid": False,
                }
            )
            return 1
        if args.frame_index_jsonl is None:
            _emit_json(_missing_argument_payload("--frame-index-jsonl"))
            return 1
        if args.output is None:
            _emit_json(_missing_argument_payload("--output"))
            return 1

        frame_index = _load_jsonl_objects(args.frame_index_jsonl)
        if args.query_diagnostic_report is not None:
            query_diagnostic_report = _load_json_object(args.query_diagnostic_report)
            handoff = dsg_detector_recall_handoff_from_query_diagnostics(
                query_diagnostic_report,
                frame_index,
            )
        else:
            gap_report = _load_json_object(args.gap_report)
            handoff = dsg_detector_recall_handoff(gap_report, frame_index)
        save_dsg_detector_recall_handoff(handoff, args.output)
        _emit_json(
            {
                "action": "build_dsg_detector_recall_handoff",
                "handoff_digest": handoff["handoff_digest"],
                "output": str(args.output),
                "summary": handoff["summary"],
            }
        )
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(
            {
                "action": "build_dsg_detector_recall_handoff",
                "error": str(exc),
                "valid": False,
            }
        )
        return 1


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(payload)
    return rows


def _missing_argument_payload(argument: str) -> dict[str, Any]:
    return {
        "action": "build_dsg_detector_recall_handoff",
        "error": f"{argument} is required",
        "valid": False,
    }


def _emit_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
