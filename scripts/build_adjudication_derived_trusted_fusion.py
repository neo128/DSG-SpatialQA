#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab.benchmark.active_qa_v2 import load_active_qa_v2_records
from dsg_spatialqa_lab.eval.active_qa_v2_analysis import (
    adjudication_derived_fusion_markdown,
    adjudication_derived_fusion_predictions,
    save_json,
    save_jsonl,
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
        description=(
            "Build a deterministic same-dataset calibrated VLM+DSG fusion policy from "
            "P50 adjudication experience."
        ),
    )
    parser.add_argument(
        "--qa-root",
        type=Path,
        default=Path("handoffs/ai2thor-real-small/inputs/qa-v2-active"),
    )
    parser.add_argument("--vlm-predictions", type=Path, required=True)
    parser.add_argument("--graph-predictions", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "handoffs/ai2thor-real-small/outputs/offline-controls/active-qa-v2/"
            "vlm-dsg-adjudication-derived-trusted-active-qa-v2-all-episodes.jsonl"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "handoffs/ai2thor-real-small/outputs/diagnostics/"
            "p52-adjudication-derived-trusted-fusion-report.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(
            "handoffs/ai2thor-real-small/outputs/diagnostics/"
            "p52-adjudication-derived-trusted-fusion-report.zh.md"
        ),
    )
    parser.add_argument("--vlm-confidence-threshold", type=float, default=0.55)
    args = parser.parse_args(argv)

    records = _load_comparison_records(args.qa_root)
    missing_inputs = [
        str(path)
        for path in (args.vlm_predictions, args.graph_predictions)
        if not path.exists()
    ]
    if not records or missing_inputs:
        payload = {
            "action": "build_adjudication_derived_trusted_fusion",
            "blockers": (
                ["missing_active_qa_v2_records"] if not records else []
            )
            + (["missing_prediction_inputs"] if missing_inputs else []),
            "missing_inputs": missing_inputs,
            "ready": False,
        }
        save_json(args.report, payload)
        _emit(payload)
        return 1
    predictions, report = adjudication_derived_fusion_predictions(
        records,
        _load_predictions(args.vlm_predictions),
        _load_predictions(args.graph_predictions),
        vlm_confidence_threshold=args.vlm_confidence_threshold,
    )
    report["source_paths"] = {
        "graph_predictions": str(args.graph_predictions),
        "qa_root": str(args.qa_root),
        "vlm_predictions": str(args.vlm_predictions),
    }
    save_jsonl(args.output, predictions)
    report["prediction_path"] = str(args.output)
    save_json(args.report, report)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(adjudication_derived_fusion_markdown(report), encoding="utf-8")
    _emit(
        {
            "action": "build_adjudication_derived_trusted_fusion",
            "output": str(args.output),
            "ready": True,
            "report": str(args.report),
            "summary": report["summary"],
        }
    )
    return 0


def _load_comparison_records(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
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


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
