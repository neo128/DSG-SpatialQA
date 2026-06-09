#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
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
        description=(
            "Derive explicit GraphTool-only DSG predictions from active QA v2 "
            "graph-backed evaluator records. This is a local graph-query ablation, "
            "not an external model result."
        ),
    )
    parser.add_argument("--qa-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        records = _load_comparison_records(args.qa_root)
        predictions = [_prediction_from_record(record) for record in records]
        lab.save_qa_predictions(predictions, args.output)
        episode_counts = Counter(str(record.get("episode_id", "unknown")) for record in records)
        report: dict[str, Any] = {
            "schema_version": "dsg-spatialqa-lab.active-qa-v2-graph-tool-prediction-build-report.v1",
            "ablation_kind": "graph_tool_only_dsg_from_active_qa_v2_graph_records",
            "not_external_model_result": True,
            "episode_counts": dict(sorted(episode_counts.items())),
            "prediction_count": len(predictions),
            "prediction_digest": lab.qa_predictions_digest(predictions),
            "prediction_path": str(args.output),
            "qa_roots": [str(root) for root in args.qa_root],
        }
        report["report_digest"] = _stable_digest_without(report, "report_digest")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError, lab.SpatialQAError) as exc:
        _emit(
            {
                "action": "build_active_qa_v2_graph_tool_predictions",
                "error": str(exc),
                "ready": False,
            }
        )
        return 1

    _emit(
        {
            "action": "build_active_qa_v2_graph_tool_predictions",
            "output": str(args.output),
            "prediction_count": len(predictions),
            "prediction_digest": report["prediction_digest"],
            "ready": True,
            "report": str(args.report),
        }
    )
    return 0


def _load_comparison_records(roots: list[Path]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for root in roots:
        for path in sorted(root.glob("*/qa-*.jsonl")):
            if path.name not in COMPARISON_SPLITS:
                continue
            for record in load_active_qa_v2_records(path):
                by_id[str(record.get("id"))] = record
    return [by_id[key] for key in sorted(by_id)]


def _prediction_from_record(record: dict[str, Any]) -> lab.QAPrediction:
    answer = _answer(record)
    required_evidence = _mapping(record.get("required_evidence"))
    evidence_nodes = _strings(required_evidence.get("nodes")) or _strings(record.get("required_nodes"))
    evidence_edges = _strings(required_evidence.get("edges")) or _strings(record.get("required_edges"))
    return lab.QAPrediction(
        id=str(record.get("id")),
        answer={
            **answer,
            "source": "graph_tool_only_dsg",
            "prediction_source": "active_qa_v2_graph_record",
        },
        evidence_nodes=tuple(evidence_nodes),
        evidence_edges=tuple(evidence_edges),
        confidence=1.0,
    )


def _answer(record: dict[str, Any]) -> dict[str, Any]:
    answer = _mapping(record.get("answer"))
    location = _mapping(answer.get("current_location"))
    if location:
        merged = dict(answer)
        merged["current_location"] = location
        return merged
    relation = answer.get("relation")
    dst = answer.get("dst")
    dst_label = answer.get("dst_label")
    step = answer.get("step")
    if relation is not None or dst is not None or dst_label is not None:
        return {
            **answer,
            "current_location": {
                "dst": dst,
                "dst_label": dst_label,
                "relation": relation,
                "step": step,
            },
        }
    return dict(answer)


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _stable_digest_without(payload: dict[str, Any], key_to_omit: str) -> str:
    import hashlib

    normalized = {key: value for key, value in payload.items() if key != key_to_omit}
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
