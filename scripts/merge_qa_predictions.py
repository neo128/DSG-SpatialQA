#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import dsg_spatialqa_lab as lab
from dsg_spatialqa_lab.benchmark.active_qa_v2 import load_active_qa_v2_records


COMPARISON_SPLITS = {
    "qa-observation-aware.jsonl",
    "qa-relation-centric.jsonl",
    "qa-situated.jsonl",
    "qa-temporal.jsonl",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Merge QA prediction JSONL files by case id and validate optional QA coverage.",
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--expected-qa-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        predictions_by_id: dict[str, lab.QAPrediction] = {}
        input_counts: dict[str, int] = {}
        for input_path in args.input:
            predictions = lab.load_qa_predictions(input_path)
            input_counts[str(input_path)] = len(predictions)
            for prediction in predictions:
                predictions_by_id[prediction.id] = prediction
        merged_predictions = [predictions_by_id[key] for key in sorted(predictions_by_id)]
        lab.save_qa_predictions(merged_predictions, args.output)
        expected_ids = _expected_case_ids(args.expected_qa_root)
        coverage = _coverage(expected_ids, set(predictions_by_id))
        report: dict[str, Any] = {
            "schema_version": "dsg-spatialqa-lab.qa-prediction-merge-report.v1",
            "ready": coverage["missing_case_count"] == 0,
            "coverage": coverage,
            "expected_qa_roots": [str(root) for root in args.expected_qa_root],
            "input_counts": input_counts,
            "input_paths": [str(path) for path in args.input],
            "merged_prediction_count": len(merged_predictions),
            "merged_prediction_digest": lab.qa_predictions_digest(merged_predictions),
            "merged_prediction_path": str(args.output),
        }
        report["report_digest"] = _stable_digest_without(report, "report_digest")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError, lab.SpatialQAError) as exc:
        _emit({"action": "merge_qa_predictions", "error": str(exc), "ready": False})
        return 1

    _emit(
        {
            "action": "merge_qa_predictions",
            "merged_prediction_count": len(merged_predictions),
            "output": str(args.output),
            "ready": report["ready"],
            "report": str(args.report),
        }
    )
    return 0 if report["ready"] is True else 1


def _expected_case_ids(roots: list[Path]) -> list[str]:
    ids: set[str] = set()
    for root in roots:
        for path in sorted(root.glob("*/qa-*.jsonl")):
            if path.name not in COMPARISON_SPLITS:
                continue
            for row in load_active_qa_v2_records(path):
                case_id = row.get("id")
                if isinstance(case_id, str):
                    ids.add(case_id)
    return sorted(ids)


def _coverage(expected_ids: list[str], prediction_ids: set[str]) -> dict[str, Any]:
    missing = sorted(set(expected_ids) - prediction_ids)
    unexpected = sorted(prediction_ids - set(expected_ids)) if expected_ids else []
    present = len(expected_ids) - len(missing)
    return {
        "expected_case_count": len(expected_ids),
        "missing_case_count": len(missing),
        "missing_case_ids": missing,
        "prediction_coverage_rate": _ratio(present, len(expected_ids)),
        "unexpected_case_count": len(unexpected),
        "unexpected_case_ids": unexpected,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(numerator / denominator, 6)


def _stable_digest_without(payload: dict[str, Any], key_to_omit: str) -> str:
    normalized = {key: value for key, value in payload.items() if key != key_to_omit}
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
