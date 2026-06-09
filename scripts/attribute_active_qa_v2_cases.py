#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab.benchmark.active_qa_v2 import load_active_qa_v2_records
from dsg_spatialqa_lab.eval.active_qa_v2_analysis import (
    active_qa_v2_case_attribution_markdown,
    active_qa_v2_case_attribution_report,
    save_json,
)


COMPARISON_SPLITS = {
    "qa-observation-aware.jsonl",
    "qa-relation-centric.jsonl",
    "qa-situated.jsonl",
    "qa-temporal.jsonl",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Attribute active QA v2 VLM+DSG adjudication wins and failures case by case.",
    )
    parser.add_argument(
        "--qa-root",
        type=Path,
        action="append",
        default=None,
        help="Active QA v2 root. May be supplied multiple times.",
    )
    parser.add_argument("--vlm-predictions", type=Path, required=True)
    parser.add_argument("--graph-predictions", type=Path, required=True)
    parser.add_argument("--trusted-predictions", type=Path, required=True)
    parser.add_argument("--adjudicated-predictions", type=Path, required=True)
    parser.add_argument(
        "--match-mode",
        choices=("p50_comparison", "structured_text"),
        default="p50_comparison",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "handoffs/ai2thor-real-small/outputs/diagnostics/"
            "p51-active-qa-v2-case-attribution.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(
            "handoffs/ai2thor-real-small/outputs/diagnostics/"
            "p51-active-qa-v2-case-attribution.zh.md"
        ),
    )
    args = parser.parse_args(argv)

    qa_roots = args.qa_root or [Path("handoffs/ai2thor-real-small/inputs/qa-v2-active")]
    records = _load_comparison_records(qa_roots)
    if not records:
        payload = _not_ready("missing_active_qa_v2_records", args.output)
        _emit(payload)
        return 1
    missing_inputs = [
        str(path)
        for path in (
            args.vlm_predictions,
            args.graph_predictions,
            args.trusted_predictions,
            args.adjudicated_predictions,
        )
        if not path.exists()
    ]
    if missing_inputs:
        payload = _not_ready("missing_prediction_inputs", args.output, missing_inputs=missing_inputs)
        _emit(payload)
        return 1
    predictions = {
        "vlm": _load_predictions(args.vlm_predictions),
        "graph": _load_predictions(args.graph_predictions),
        "trusted": _load_predictions(args.trusted_predictions),
        "adjudicated": _load_predictions(args.adjudicated_predictions),
    }
    report = active_qa_v2_case_attribution_report(
        records,
        predictions["vlm"],
        predictions["graph"],
        predictions["trusted"],
        predictions["adjudicated"],
        match_mode=args.match_mode,
        source_paths={
            "adjudicated_predictions": str(args.adjudicated_predictions),
            "graph_predictions": str(args.graph_predictions),
            "qa_roots": [str(root) for root in qa_roots],
            "trusted_predictions": str(args.trusted_predictions),
            "vlm_predictions": str(args.vlm_predictions),
        },
    )
    save_json(args.output, report)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(active_qa_v2_case_attribution_markdown(report), encoding="utf-8")
    _emit(
        {
            "action": "attribute_active_qa_v2_cases",
            "case_count": report["case_count"],
            "match_mode": report["match_mode"],
            "output": str(args.output),
            "ready": True,
            "report_digest": report["report_digest"],
            "summary": report["summary"],
        }
    )
    return 0


def _load_comparison_records(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        for path in sorted(root.glob("*/qa-*.jsonl")):
            if path.name not in COMPARISON_SPLITS:
                continue
            for row in load_active_qa_v2_records(path):
                case_id = str(row.get("id"))
                if case_id in seen:
                    continue
                seen.add(case_id)
                rows.append(row)
    return rows


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        str(row["id"]): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }


def _not_ready(
    blocker: str,
    output: Path,
    *,
    missing_inputs: list[str] | None = None,
) -> dict[str, Any]:
    payload = {
        "action": "attribute_active_qa_v2_cases",
        "blockers": [blocker],
        "missing_inputs": missing_inputs or [],
        "output": str(output),
        "ready": False,
    }
    save_json(output, payload)
    return payload


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
