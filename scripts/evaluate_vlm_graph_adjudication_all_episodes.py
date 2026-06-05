#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab.benchmark.active_qa_v2 import load_active_qa_v2_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Check/evaluate all-episode active QA v2 VLM+DSG adjudication inputs.",
    )
    parser.add_argument(
        "--qa-root",
        type=Path,
        default=Path("handoffs/ai2thor-real-small/inputs/qa-v2-active"),
    )
    parser.add_argument("--adjudicated-predictions", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("handoffs/ai2thor-real-small/outputs/diagnostics/vlm-graph-adjudication-active-qa-v2-all-episodes-readiness.json"),
    )
    args = parser.parse_args(argv)

    if not args.adjudicated_predictions.exists():
        payload = _not_ready(
            "missing_adjudicated_predictions",
            args.report,
            extra={"missing_path": str(args.adjudicated_predictions)},
        )
        _emit(payload)
        return 1
    try:
        predictions = _load_jsonl(args.adjudicated_predictions)
        invalid = [
            row.get("id")
            for row in predictions
            if not _valid_adjudicated_prediction(row)
        ]
        if invalid:
            payload = _not_ready(
                "invalid_adjudicated_prediction_schema",
                args.report,
                extra={"invalid_prediction_ids": invalid},
            )
            _emit(payload)
            return 1
        records = _load_active_records(args.qa_root)
        record_ids = {row["id"] for row in records if isinstance(row.get("id"), str)}
        prediction_ids = {str(row.get("id")) for row in predictions}
        missing = sorted(record_ids - prediction_ids)
        payload = {
            "schema_version": "dsg-spatialqa-lab.vlm-graph-adjudication-all-episodes-readiness.v1",
            "ready": not missing,
            "research_ready": False,
            "final_record_written": False,
            "blockers": [] if not missing else ["missing_adjudicated_active_qa_cases"],
            "active_qa_record_count": len(record_ids),
            "adjudicated_prediction_count": len(prediction_ids),
            "missing_case_count": len(missing),
            "missing_case_ids": missing[:100],
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        payload = _not_ready("invalid_adjudicated_prediction_schema", args.report, extra={"error": str(exc)})
        _emit(payload)
        return 1

    _save_json(args.report, payload)
    _emit(payload)
    return 0 if payload["ready"] is True else 1


def _not_ready(blocker: str, report_path: Path, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "schema_version": "dsg-spatialqa-lab.vlm-graph-adjudication-all-episodes-readiness.v1",
        "ready": False,
        "research_ready": False,
        "blockers": [blocker],
        "final_record_written": False,
    }
    payload.update(extra or {})
    _save_json(report_path, payload)
    return payload


def _load_active_records(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/qa-*.jsonl")):
        rows.extend(load_active_qa_v2_records(path))
    return rows


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("adjudicated prediction JSONL rows must be objects")
    return rows


def _valid_adjudicated_prediction(row: dict[str, Any]) -> bool:
    answer = row.get("answer")
    if not isinstance(row.get("id"), str) or not isinstance(answer, dict):
        return False
    if answer.get("decision") not in {"accept_vlm", "accept_dsg", "reject_both", "uncertain"}:
        return False
    if not isinstance(answer.get("evidence_summary"), str) or not answer["evidence_summary"].strip():
        return False
    location = answer.get("current_location")
    if not isinstance(location, dict):
        return False
    relation = location.get("relation")
    if relation not in {"ON", "INSIDE", "IN_REGION", "IN_ROOM", "UNKNOWN", "VISIBLE_FROM"}:
        return False
    if relation != "UNKNOWN" and not isinstance(location.get("dst") or location.get("dst_label"), str):
        return False
    return True


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
