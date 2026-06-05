#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import dsg_spatialqa_lab as lab


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build VLM+DSG evidence score, conflict, and request-bundle reports."
    )
    parser.add_argument("--qa", required=True, help="QA JSONL path.")
    parser.add_argument("--vlm-predictions", required=True, help="VLM-only prediction JSONL.")
    parser.add_argument("--graph-predictions", required=True, help="GraphTool prediction JSONL.")
    parser.add_argument(
        "--vlm-semantic-report",
        required=True,
        help="Semantic eval report for the VLM-only predictions.",
    )
    parser.add_argument(
        "--graph-semantic-report",
        required=True,
        help="Semantic eval report for the GraphTool predictions.",
    )
    parser.add_argument(
        "--detector-jsonl",
        action="append",
        dest="detector_jsonls",
        required=True,
        help=(
            "External detector/RGB-D observation JSONL used as DSG evidence. "
            "May be repeated; all explicit local records are merged."
        ),
    )
    parser.add_argument("--score-report", required=True, help="Output evidence score report JSON.")
    parser.add_argument("--conflict-report", required=True, help="Output conflict report JSON.")
    parser.add_argument(
        "--request-bundle",
        required=True,
        help="Output VLM+DSG evidence request bundle JSON.",
    )
    parser.add_argument(
        "--max-request-cases",
        type=int,
        default=None,
        help="Optional maximum number of adjudication cases in the request bundle.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    qa_cases = lab.load_qa_dataset(args.qa)
    vlm_predictions = lab.load_qa_predictions(args.vlm_predictions)
    graph_predictions = lab.load_qa_predictions(args.graph_predictions)
    vlm_semantic_report = lab.load_vlm_semantic_eval_report(args.vlm_semantic_report)
    graph_semantic_report = lab.load_vlm_semantic_eval_report(args.graph_semantic_report)
    detector_records = [
        record
        for detector_jsonl in args.detector_jsonls
        for record in _load_jsonl_mappings(detector_jsonl)
    ]

    score_report = lab.vlm_graph_evidence_score_report(
        qa_cases,
        graph_predictions,
        detector_records=detector_records,
        graph_prediction_path=args.graph_predictions,
        detector_observation_path=args.detector_jsonls[0],
    )
    lab.save_vlm_graph_evidence_score_report(score_report, args.score_report)

    conflict_report = lab.vlm_graph_conflict_report(
        qa_cases,
        vlm_predictions,
        graph_predictions,
        vlm_semantic_report,
        graph_semantic_report,
        score_report,
        vlm_prediction_path=args.vlm_predictions,
        graph_prediction_path=args.graph_predictions,
        vlm_semantic_report_path=args.vlm_semantic_report,
        graph_semantic_report_path=args.graph_semantic_report,
        evidence_score_report_path=args.score_report,
    )
    lab.save_vlm_graph_conflict_report(conflict_report, args.conflict_report)

    request_bundle = lab.vlm_graph_evidence_request_bundle(
        qa_cases,
        vlm_predictions,
        graph_predictions,
        score_report,
        conflict_report,
        detector_records=detector_records,
        max_cases=args.max_request_cases,
    )
    lab.save_vlm_graph_evidence_request_bundle(request_bundle, args.request_bundle)

    payload = {
        "schema_version": "dsg-spatialqa-lab.vlm-graph-evidence-build-run.v1",
        "conflict_report_digest": conflict_report["report_digest"],
        "conflict_report_path": str(args.conflict_report),
        "conflict_summary": conflict_report["summary"],
        "detector_record_count": len(detector_records),
        "detector_jsonl_paths": [str(path) for path in args.detector_jsonls],
        "request_bundle_digest": request_bundle["request_bundle_digest"],
        "request_bundle_path": str(args.request_bundle),
        "request_case_count": request_bundle["case_count"],
        "score_report_digest": score_report["report_digest"],
        "score_report_path": str(args.score_report),
        "score_summary": score_report["summary"],
        "validations": {
            "conflict_report": lab.validate_vlm_graph_conflict_report(conflict_report)["valid"],
            "request_bundle": lab.validate_vlm_graph_evidence_request_bundle(request_bundle)[
                "valid"
            ],
            "score_report": lab.validate_vlm_graph_evidence_score_report(score_report)["valid"],
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_jsonl_mappings(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise lab.SpatialQAError(f"Detector JSONL line {line_no} must be an object")
        records.append(payload)
    return records


if __name__ == "__main__":
    raise SystemExit(main())
