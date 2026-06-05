#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import dsg_spatialqa_lab as lab


COMPARISON_SCHEMA_VERSION = "dsg-spatialqa-lab.vlm-graph-adjudication-comparison.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Evaluate local VLM+DSG adjudication predictions against VLM-only and "
            "GraphTool-only DSG semantic reports."
        ),
    )
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--request-bundle", type=Path, required=True)
    parser.add_argument("--vlm-predictions", type=Path, required=True)
    parser.add_argument("--adjudicated-predictions", type=Path, required=True)
    parser.add_argument("--vlm-semantic-report", type=Path, required=True)
    parser.add_argument("--graph-semantic-report", type=Path, required=True)
    parser.add_argument("--merged-predictions-output", type=Path)
    parser.add_argument("--adjudicated-semantic-report", type=Path)
    parser.add_argument("--delta-vs-vlm", type=Path)
    parser.add_argument("--delta-vs-graph", type=Path)
    parser.add_argument("--comparison-report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.adjudicated_predictions.exists():
        _emit_json(
            _not_ready_payload(
                "missing_adjudicated_predictions",
                [args.adjudicated_predictions],
            )
        )
        return 1
    try:
        cases = lab.load_qa_dataset(args.qa)
        request_bundle = lab.load_vlm_graph_evidence_request_bundle(args.request_bundle)
        bundle_validation = lab.validate_vlm_graph_evidence_request_bundle(request_bundle)
        if bundle_validation["valid"] is not True:
            raise lab.SpatialQAError("VLM+DSG evidence request bundle failed validation")
        vlm_predictions = lab.load_qa_predictions(args.vlm_predictions)
        adjudicated_predictions = lab.load_qa_predictions(args.adjudicated_predictions)
        vlm_semantic_report = lab.load_vlm_semantic_eval_report(args.vlm_semantic_report)
        graph_semantic_report = lab.load_vlm_semantic_eval_report(args.graph_semantic_report)

        merge = _merged_adjudicated_predictions(
            cases,
            request_bundle,
            vlm_predictions,
            adjudicated_predictions,
        )
        if merge["missing_adjudicated_case_ids"]:
            _emit_json(
                {
                    **_not_ready_payload("missing_adjudicated_request_cases", []),
                    "missing_adjudicated_case_ids": merge["missing_adjudicated_case_ids"],
                    "request_bundle_digest": request_bundle.get("request_bundle_digest"),
                }
            )
            return 1
        if merge["invalid_adjudicated_case_ids"]:
            _emit_json(
                {
                    **_not_ready_payload("invalid_adjudicated_prediction_schema", []),
                    "invalid_adjudicated_case_ids": merge["invalid_adjudicated_case_ids"],
                    "request_bundle_digest": request_bundle.get("request_bundle_digest"),
                }
            )
            return 1
        if merge["missing_vlm_fallback_case_ids"]:
            _emit_json(
                {
                    **_not_ready_payload("missing_vlm_fallback_predictions", []),
                    "missing_vlm_fallback_case_ids": merge["missing_vlm_fallback_case_ids"],
                }
            )
            return 1

        merged_predictions = merge["predictions"]
        if args.merged_predictions_output is not None:
            lab.save_qa_predictions(merged_predictions, args.merged_predictions_output)

        semantic_report = lab.vlm_semantic_eval_report(
            cases,
            merged_predictions,
            gold_path=args.qa,
            prediction_path=(
                args.merged_predictions_output
                if args.merged_predictions_output is not None
                else args.adjudicated_predictions
            ),
        )
        if args.adjudicated_semantic_report is not None:
            lab.save_vlm_semantic_eval_report(semantic_report, args.adjudicated_semantic_report)

        delta_vs_vlm = lab.vlm_semantic_eval_delta_report(
            semantic_report,
            vlm_semantic_report,
            candidate_name="vlm_graph_adjudicated",
            baseline_name="vlm_only",
        )
        delta_vs_graph = lab.vlm_semantic_eval_delta_report(
            semantic_report,
            graph_semantic_report,
            candidate_name="vlm_graph_adjudicated",
            baseline_name="graph_tool_only_dsg",
        )
        if args.delta_vs_vlm is not None:
            _save_json_text(lab.vlm_semantic_eval_delta_report_json(delta_vs_vlm), args.delta_vs_vlm)
        if args.delta_vs_graph is not None:
            _save_json_text(
                lab.vlm_semantic_eval_delta_report_json(delta_vs_graph),
                args.delta_vs_graph,
            )

        comparison = _comparison_report(
            semantic_report,
            vlm_semantic_report,
            graph_semantic_report,
            delta_vs_vlm,
            delta_vs_graph,
            merge_summary=merge["summary"],
            paths={
                "adjudicated_predictions": args.adjudicated_predictions,
                "adjudicated_semantic_report": args.adjudicated_semantic_report,
                "comparison_report": args.comparison_report,
                "delta_vs_graph": args.delta_vs_graph,
                "delta_vs_vlm": args.delta_vs_vlm,
                "merged_predictions": args.merged_predictions_output,
                "qa": args.qa,
                "request_bundle": args.request_bundle,
                "vlm_predictions": args.vlm_predictions,
            },
            request_bundle=request_bundle,
        )
        _save_json(args.comparison_report, comparison)
    except (OSError, lab.SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(
            {
                "action": "evaluate_vlm_graph_adjudication",
                "blockers": ["adjudication_evaluation_failed"],
                "error": str(exc),
                "final_record_written": False,
                "ready": False,
                "research_ready": False,
            }
        )
        return 1

    _emit_json(
        {
            "action": "evaluate_vlm_graph_adjudication",
            "comparison_report": str(args.comparison_report),
            "comparison_report_digest": comparison["report_digest"],
            "final_record_written": False,
            "ready": True,
            "research_ready": False,
            "summary": comparison["summary"],
        }
    )
    return 0


def _merged_adjudicated_predictions(
    cases: list[lab.QACase],
    request_bundle: dict[str, Any],
    vlm_predictions: list[lab.QAPrediction],
    adjudicated_predictions: list[lab.QAPrediction],
) -> dict[str, Any]:
    request_case_ids = _request_case_ids(request_bundle)
    vlm_by_key = {
        lab.prediction_alignment_key(prediction.id): prediction for prediction in vlm_predictions
    }
    adjudicated_by_key = {
        lab.prediction_alignment_key(prediction.id): prediction
        for prediction in adjudicated_predictions
    }
    merged: list[lab.QAPrediction] = []
    missing_adjudicated: list[str] = []
    invalid_adjudicated: list[str] = []
    missing_vlm: list[str] = []
    adjudicated_count = 0
    fallback_count = 0
    for case in cases:
        key = lab.prediction_alignment_key(case.id)
        if case.id in request_case_ids:
            adjudicated = adjudicated_by_key.get(key)
            if adjudicated is None:
                missing_adjudicated.append(case.id)
                continue
            if not _valid_adjudicated_answer(adjudicated.answer):
                invalid_adjudicated.append(case.id)
                continue
            merged.append(
                _with_adjudication_metadata(
                    adjudicated,
                    output_id=case.id,
                    source="vlm_graph_adjudication",
                    request_bundle_digest=request_bundle.get("request_bundle_digest"),
                )
            )
            adjudicated_count += 1
            continue
        vlm_prediction = vlm_by_key.get(key)
        if vlm_prediction is None:
            missing_vlm.append(case.id)
            continue
        merged.append(
            _with_adjudication_metadata(
                vlm_prediction,
                output_id=case.id,
                source="vlm_fallback",
                request_bundle_digest=request_bundle.get("request_bundle_digest"),
            )
        )
        fallback_count += 1
    extra_adjudicated_case_ids = sorted(
        {
            lab.prediction_alignment_key(prediction.id)
            for prediction in adjudicated_predictions
        }
        - request_case_ids
    )
    return {
        "missing_adjudicated_case_ids": missing_adjudicated,
        "invalid_adjudicated_case_ids": invalid_adjudicated,
        "missing_vlm_fallback_case_ids": missing_vlm,
        "predictions": merged,
        "summary": {
            "adjudicated_prediction_count": len(adjudicated_predictions),
            "adjudicated_request_case_count": adjudicated_count,
            "extra_adjudicated_case_ids": extra_adjudicated_case_ids,
            "fallback_vlm_case_count": fallback_count,
            "merged_prediction_count": len(merged),
            "missing_adjudicated_case_count": len(missing_adjudicated),
            "invalid_adjudicated_case_count": len(invalid_adjudicated),
            "missing_vlm_fallback_case_count": len(missing_vlm),
            "request_case_count": len(request_case_ids),
        },
    }


def _valid_adjudicated_answer(answer: dict[str, Any]) -> bool:
    decision = answer.get("decision")
    if decision not in {"accept_vlm", "accept_dsg", "reject_both", "uncertain"}:
        return False
    evidence_summary = answer.get("evidence_summary")
    if not isinstance(evidence_summary, str) or not evidence_summary.strip():
        return False
    location = answer.get("current_location")
    if not isinstance(location, dict):
        return False
    relation = location.get("relation")
    if relation not in {"ON", "INSIDE", "IN_REGION", "IN_ROOM", "UNKNOWN"}:
        return False
    if relation != "UNKNOWN":
        dst = location.get("dst") or location.get("dst_label")
        if not isinstance(dst, str) or not dst.strip():
            return False
    return True


def _with_adjudication_metadata(
    prediction: lab.QAPrediction,
    *,
    output_id: str,
    source: str,
    request_bundle_digest: object,
) -> lab.QAPrediction:
    answer = deepcopy(prediction.answer)
    answer["adjudication"] = {
        "adjudicated_prediction_id": prediction.id,
        "request_bundle_digest": request_bundle_digest,
        "source": source,
    }
    return lab.QAPrediction(
        id=output_id,
        answer=answer,
        evidence_nodes=prediction.evidence_nodes,
        evidence_edges=prediction.evidence_edges,
        confidence=prediction.confidence,
        error=prediction.error,
    )


def _comparison_report(
    adjudicated_semantic_report: dict[str, Any],
    vlm_semantic_report: dict[str, Any],
    graph_semantic_report: dict[str, Any],
    delta_vs_vlm: dict[str, Any],
    delta_vs_graph: dict[str, Any],
    *,
    merge_summary: dict[str, Any],
    paths: dict[str, Path | None],
    request_bundle: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        **merge_summary,
        "semantic_match_count": _summary_int(adjudicated_semantic_report, "semantic_match_count"),
        "semantic_match_rate": _summary_float(adjudicated_semantic_report, "semantic_match_rate"),
    }
    report: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "ready": True,
        "research_ready": False,
        "final_record_written": False,
        "claim_boundary": (
            "Stage comparison only: this does not write a final research claim record. "
            "A real conclusion requires expanded QA/scene coverage and real local artifacts."
        ),
        "request_bundle_digest": request_bundle.get("request_bundle_digest"),
        "paths": {key: str(path) if path is not None else None for key, path in paths.items()},
        "summary": summary,
        "groups": [
            _group("VLM-only", vlm_semantic_report),
            _group("GraphTool-only DSG", graph_semantic_report),
            _group("VLM+DSG adjudicated", adjudicated_semantic_report),
        ],
        "deltas": {
            "adjudicated_vlm_graph_vs_graph_tool_only": delta_vs_graph,
            "adjudicated_vlm_graph_vs_vlm_only": delta_vs_vlm,
        },
        "guardrails": {
            "final_claim_record_allowed": False,
            "requires_expanded_slice_for_research_conclusion": True,
        },
    }
    report["report_digest"] = _digest_without(report, "report_digest")
    return report


def _group(name: str, semantic_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "report_digest": semantic_report.get("report_digest"),
        "summary": semantic_report.get("summary"),
    }


def _request_case_ids(bundle: dict[str, Any]) -> set[str]:
    case_inputs = bundle.get("case_inputs")
    if not isinstance(case_inputs, list):
        raise lab.SpatialQAError("VLM+DSG evidence request bundle must contain case_inputs")
    case_ids: set[str] = set()
    for index, case_input in enumerate(case_inputs):
        if not isinstance(case_input, dict):
            raise lab.SpatialQAError(f"case_inputs[{index}] must be an object")
        case_id = case_input.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise lab.SpatialQAError(f"case_inputs[{index}] must contain case_id")
        case_ids.add(case_id)
    return case_ids


def _summary_int(report: dict[str, Any], key: str) -> int:
    summary = report.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get(key), int):
        return int(summary[key])
    return 0


def _summary_float(report: dict[str, Any], key: str) -> float:
    summary = report.get("summary")
    value = summary.get(key) if isinstance(summary, dict) else None
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _not_ready_payload(blocker: str, missing_paths: list[Path]) -> dict[str, Any]:
    return {
        "action": "evaluate_vlm_graph_adjudication",
        "blockers": [blocker],
        "final_record_written": False,
        "next_missing_artifacts": [str(path) for path in missing_paths],
        "ready": False,
        "research_ready": False,
    }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    _save_json_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", path)


def _save_json_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _digest_without(payload: dict[str, Any], digest_key: str) -> str:
    normalized = {key: value for key, value in payload.items() if key != digest_key}
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
