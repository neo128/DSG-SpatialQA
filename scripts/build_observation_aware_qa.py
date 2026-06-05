from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    load_observation_aware_inputs,
    observation_aware_qa_cases,
    save_observation_aware_qa_outputs,
    validate_observation_aware_qa_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Build an observation-aware object-location QA slice from explicit "
            "detector/RGB-D observations."
        ),
    )
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--observation-sequence", type=Path, required=True)
    parser.add_argument("--output-qa", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--target-case-count",
        type=int,
        help=(
            "Optional deterministic target count. When set, the builder keeps "
            "base-QA-aligned object-location cases first and supplements from "
            "visible detector/RGB-D observations until this count is reached."
        ),
    )
    args = parser.parse_args(argv)

    try:
        cases, observations = load_observation_aware_inputs(
            qa_path=args.qa,
            observation_sequence_path=args.observation_sequence,
        )
        generated_cases, report = observation_aware_qa_cases(
            cases,
            observations,
            qa_path=args.qa,
            observation_sequence_path=args.observation_sequence,
            target_case_count=args.target_case_count,
        )
        save_observation_aware_qa_outputs(
            generated_cases,
            report,
            output_qa_path=args.output_qa,
            report_path=args.report,
        )
        validation = validate_observation_aware_qa_report(report)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(
            {
                "action": "build_observation_aware_qa",
                "ready": False,
                "error": str(exc),
            }
        )
        return 1

    _emit_json(
        {
            "action": "build_observation_aware_qa",
            "output_qa": str(args.output_qa),
            "ready": validation["valid"] is True,
            "report": str(args.report),
            "report_digest": report["report_digest"],
            "summary": report["summary"],
            "valid": validation["valid"],
        }
    )
    return 0 if validation["valid"] is True else 1


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
