from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, cast

from dsg_spatialqa_lab.eval.qa_metrics import (
    QAPrediction,
    qa_predictions_digest,
    save_qa_predictions,
)
from dsg_spatialqa_lab.benchmark import QACase, qa_dataset_digest
from dsg_spatialqa_lab.schema import SpatialQAError


VLM_GRAPH_FUSION_REPORT_SCHEMA_VERSION = "dsg-spatialqa-lab.vlm-graph-fusion-report.v1"
VLM_GRAPH_EVIDENCE_SCORE_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-graph-evidence-score-report.v1"
)
VLM_GRAPH_CONFLICT_REPORT_SCHEMA_VERSION = "dsg-spatialqa-lab.vlm-graph-conflict-report.v1"
VLM_GRAPH_EVIDENCE_REQUEST_BUNDLE_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-graph-evidence-request-bundle.v1"
)
VLM_GRAPH_FUSION_POLICY = "explicit_graph_relation_or_vlm_fallback"
VLM_GRAPH_TRUSTED_FUSION_POLICY = "trusted_graph_relation_or_vlm_fallback"
VLM_GRAPH_TRUST_THRESHOLD = 0.75
_OBSERVATION_AWARE_SUFFIX_RE = re.compile(r":observation_aware:\d+$")
_REQUIRED_EVIDENCE_KINDS = frozenset(("depth", "detector", "rgb"))
_FORBIDDEN_REQUEST_KEYS = frozenset(
    (
        "gold",
        "gold_answer",
        "gold_evidence",
        "graph_digest",
        "required_edges",
        "required_nodes",
        "semantic_match",
        "strict_exact_match",
    )
)
_EXPLICIT_GRAPH_RELATIONS = frozenset(
    (
        "BEHIND",
        "CONTAINS",
        "IN",
        "IN_FRONT_OF",
        "IN_REGION",
        "INSIDE",
        "LEFT_OF",
        "NEAR",
        "ON",
        "RIGHT_OF",
        "UNDER",
    )
)
_PLAUSIBLE_LOCATION_SUPPORT_LABELS = frozenset(
    (
        "armchair",
        "bathtub",
        "bed",
        "bench",
        "cabinet",
        "chair",
        "coffeetable",
        "countertop",
        "desk",
        "diningtable",
        "dresser",
        "drawer",
        "floor",
        "fridge",
        "garbagecan",
        "handtowelholder",
        "microwave",
        "ottoman",
        "shelf",
        "sidetable",
        "sink",
        "sofa",
        "stoveburner",
        "table",
        "toilet",
        "tvstand",
    )
)
_PLAUSIBLE_ROOM_LEVEL_TARGET_LABELS = frozenset(
    (
        "armchair",
        "baseballbat",
        "basketball",
        "bathtub",
        "bed",
        "blinds",
        "boots",
        "box",
        "cabinet",
        "candle",
        "chair",
        "diningtable",
        "faucet",
        "floor",
        "garbagecan",
    )
)


def fuse_vlm_graph_predictions(
    vlm_predictions: Sequence[QAPrediction],
    graph_predictions: Sequence[QAPrediction],
    *,
    fusion_policy: str = VLM_GRAPH_FUSION_POLICY,
) -> list[QAPrediction]:
    _validate_fusion_policy(fusion_policy)
    vlm_by_key = _prediction_by_key(vlm_predictions)
    output: list[QAPrediction] = []
    emitted_keys: set[str] = set()
    for graph_prediction in graph_predictions:
        key = prediction_alignment_key(graph_prediction.id)
        vlm_prediction = vlm_by_key.get(key)
        output.append(
            _fused_prediction(
                vlm_prediction,
                graph_prediction,
                fusion_policy=fusion_policy,
            )
        )
        emitted_keys.add(key)
    for vlm_prediction in vlm_predictions:
        key = prediction_alignment_key(vlm_prediction.id)
        if key in emitted_keys:
            continue
        output.append(_fused_prediction(vlm_prediction, None, fusion_policy=fusion_policy))
    return output


def vlm_graph_fusion_report(
    vlm_predictions: Sequence[QAPrediction],
    graph_predictions: Sequence[QAPrediction],
    fused_predictions: Sequence[QAPrediction],
    *,
    vlm_prediction_path: str | Path | None = None,
    graph_prediction_path: str | Path | None = None,
    fused_prediction_path: str | Path | None = None,
    fusion_policy: str = VLM_GRAPH_FUSION_POLICY,
) -> dict[str, Any]:
    _validate_fusion_policy(fusion_policy)
    vlm_keys = {prediction_alignment_key(prediction.id) for prediction in vlm_predictions}
    graph_keys = {prediction_alignment_key(prediction.id) for prediction in graph_predictions}
    summary = {
        "fused_prediction_count": len(fused_predictions),
        "graph_prediction_count": len(graph_predictions),
        "graph_tool_source_count": _fusion_source_count(fused_predictions, "graph_tool"),
        "unmatched_graph_prediction_count": len(graph_keys - vlm_keys),
        "unmatched_vlm_prediction_count": len(vlm_keys - graph_keys),
        "vlm_prediction_count": len(vlm_predictions),
        "vlm_source_count": _fusion_source_count(fused_predictions, "vlm"),
    }
    report: dict[str, Any] = {
        "schema_version": VLM_GRAPH_FUSION_REPORT_SCHEMA_VERSION,
        "fusion_policy": fusion_policy,
        "fused_prediction_path": (
            str(fused_prediction_path) if fused_prediction_path is not None else None
        ),
        "graph_prediction_digest": qa_predictions_digest(graph_predictions),
        "graph_prediction_path": (
            str(graph_prediction_path) if graph_prediction_path is not None else None
        ),
        "prediction_digest": qa_predictions_digest(fused_predictions),
        "summary": summary,
        "vlm_prediction_digest": qa_predictions_digest(vlm_predictions),
        "vlm_prediction_path": str(vlm_prediction_path) if vlm_prediction_path is not None else None,
    }
    report["report_digest"] = vlm_graph_fusion_report_digest(report)
    return report


def vlm_graph_fusion_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def vlm_graph_fusion_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_vlm_graph_fusion_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(vlm_graph_fusion_report_json(report), encoding="utf-8")
    return output_path


def load_vlm_graph_fusion_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("VLM graph fusion report JSON must be an object")
    return cast(dict[str, Any], payload)


def load_vlm_graph_evidence_score_report(path: str | Path) -> dict[str, Any]:
    return _load_json_mapping(path, "VLM graph evidence score report")


def load_vlm_graph_conflict_report(path: str | Path) -> dict[str, Any]:
    return _load_json_mapping(path, "VLM graph conflict report")


def load_vlm_graph_evidence_request_bundle(path: str | Path) -> dict[str, Any]:
    return _load_json_mapping(path, "VLM graph evidence request bundle")


def validate_vlm_graph_fusion_report(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = report.get("summary")
    expected_digest = vlm_graph_fusion_report_digest(report)
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version") == VLM_GRAPH_FUSION_REPORT_SCHEMA_VERSION,
            "expected": VLM_GRAPH_FUSION_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "fusion_policy",
            "passed": report.get("fusion_policy")
            in {VLM_GRAPH_FUSION_POLICY, VLM_GRAPH_TRUSTED_FUSION_POLICY},
            "expected": [VLM_GRAPH_FUSION_POLICY, VLM_GRAPH_TRUSTED_FUSION_POLICY],
            "actual": report.get("fusion_policy"),
        },
        {
            "name": "report_digest",
            "passed": report.get("report_digest") == expected_digest,
            "expected": expected_digest,
            "actual": report.get("report_digest"),
        },
        {
            "name": "summary",
            "passed": isinstance(summary, Mapping),
            "expected": "mapping",
            "actual": type(summary).__name__,
        },
    ]
    return {
        "checks": checks,
        "valid": all(check["passed"] is True for check in checks),
    }


def validate_vlm_graph_evidence_score_report(report: Mapping[str, Any]) -> dict[str, Any]:
    expected_digest = vlm_graph_evidence_score_report_digest(report)
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == VLM_GRAPH_EVIDENCE_SCORE_REPORT_SCHEMA_VERSION,
            "expected": VLM_GRAPH_EVIDENCE_SCORE_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "report_digest",
            "passed": report.get("report_digest") == expected_digest,
            "expected": expected_digest,
            "actual": report.get("report_digest"),
        },
        {
            "name": "cases",
            "passed": isinstance(report.get("cases"), Sequence)
            and not isinstance(report.get("cases"), (str, bytes)),
            "expected": "sequence",
            "actual": type(report.get("cases")).__name__,
        },
    ]
    return {"checks": checks, "valid": all(check["passed"] is True for check in checks)}


def validate_vlm_graph_conflict_report(report: Mapping[str, Any]) -> dict[str, Any]:
    expected_digest = vlm_graph_conflict_report_digest(report)
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version") == VLM_GRAPH_CONFLICT_REPORT_SCHEMA_VERSION,
            "expected": VLM_GRAPH_CONFLICT_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "report_digest",
            "passed": report.get("report_digest") == expected_digest,
            "expected": expected_digest,
            "actual": report.get("report_digest"),
        },
        {
            "name": "cases",
            "passed": isinstance(report.get("cases"), Sequence)
            and not isinstance(report.get("cases"), (str, bytes)),
            "expected": "sequence",
            "actual": type(report.get("cases")).__name__,
        },
    ]
    return {"checks": checks, "valid": all(check["passed"] is True for check in checks)}


def validate_vlm_graph_evidence_request_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    expected_digest = vlm_graph_evidence_request_bundle_digest(bundle)
    forbidden_hits = _forbidden_request_key_hits(bundle)
    checks = [
        {
            "name": "schema_version",
            "passed": bundle.get("schema_version")
            == VLM_GRAPH_EVIDENCE_REQUEST_BUNDLE_SCHEMA_VERSION,
            "expected": VLM_GRAPH_EVIDENCE_REQUEST_BUNDLE_SCHEMA_VERSION,
            "actual": bundle.get("schema_version"),
        },
        {
            "name": "request_bundle_digest",
            "passed": bundle.get("request_bundle_digest") == expected_digest,
            "expected": expected_digest,
            "actual": bundle.get("request_bundle_digest"),
        },
        {
            "name": "forbidden_fields_absent",
            "passed": not forbidden_hits,
            "expected": [],
            "actual": forbidden_hits,
        },
        {
            "name": "case_inputs",
            "passed": isinstance(bundle.get("case_inputs"), Sequence)
            and not isinstance(bundle.get("case_inputs"), (str, bytes)),
            "expected": "sequence",
            "actual": type(bundle.get("case_inputs")).__name__,
        },
    ]
    return {"checks": checks, "valid": all(check["passed"] is True for check in checks)}


def save_vlm_graph_fusion_predictions(
    vlm_predictions: Sequence[QAPrediction],
    graph_predictions: Sequence[QAPrediction],
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
    vlm_prediction_path: str | Path | None = None,
    graph_prediction_path: str | Path | None = None,
    fusion_policy: str = VLM_GRAPH_FUSION_POLICY,
) -> dict[str, Any]:
    fused_predictions = fuse_vlm_graph_predictions(
        vlm_predictions,
        graph_predictions,
        fusion_policy=fusion_policy,
    )
    save_qa_predictions(fused_predictions, output_path)
    report = vlm_graph_fusion_report(
        vlm_predictions,
        graph_predictions,
        fused_predictions,
        vlm_prediction_path=vlm_prediction_path,
        graph_prediction_path=graph_prediction_path,
        fused_prediction_path=output_path,
        fusion_policy=fusion_policy,
    )
    if report_path is not None:
        save_vlm_graph_fusion_report(report, report_path)
    return report


def vlm_graph_evidence_score_report(
    cases: Sequence[QACase],
    graph_predictions: Sequence[QAPrediction],
    *,
    detector_records: Sequence[Mapping[str, Any]] = (),
    graph_prediction_path: str | Path | None = None,
    detector_observation_path: str | Path | None = None,
    trust_threshold: float = VLM_GRAPH_TRUST_THRESHOLD,
) -> dict[str, Any]:
    graph_by_key = _prediction_by_key(graph_predictions)
    detector_index = _detector_evidence_index(detector_records)
    rows = [
        _evidence_score_row(
            case,
            graph_by_key.get(prediction_alignment_key(case.id)),
            detector_index=detector_index,
            trust_threshold=trust_threshold,
        )
        for case in cases
    ]
    report: dict[str, Any] = {
        "schema_version": VLM_GRAPH_EVIDENCE_SCORE_REPORT_SCHEMA_VERSION,
        "case_count": len(cases),
        "detector_observation_path": (
            str(detector_observation_path) if detector_observation_path is not None else None
        ),
        "graph_prediction_digest": qa_predictions_digest(graph_predictions),
        "graph_prediction_path": (
            str(graph_prediction_path) if graph_prediction_path is not None else None
        ),
        "qa_digest": qa_dataset_digest(cases),
        "summary": _evidence_score_summary(rows),
        "trust_threshold": trust_threshold,
        "cases": rows,
    }
    report["report_digest"] = vlm_graph_evidence_score_report_digest(report)
    return report


def vlm_graph_conflict_report(
    cases: Sequence[QACase],
    vlm_predictions: Sequence[QAPrediction],
    graph_predictions: Sequence[QAPrediction],
    vlm_semantic_report: Mapping[str, Any],
    graph_semantic_report: Mapping[str, Any],
    evidence_score_report: Mapping[str, Any],
    *,
    vlm_prediction_path: str | Path | None = None,
    graph_prediction_path: str | Path | None = None,
    vlm_semantic_report_path: str | Path | None = None,
    graph_semantic_report_path: str | Path | None = None,
    evidence_score_report_path: str | Path | None = None,
) -> dict[str, Any]:
    vlm_by_key = _prediction_by_key(vlm_predictions)
    graph_by_key = _prediction_by_key(graph_predictions)
    vlm_matches = _semantic_match_map(vlm_semantic_report)
    graph_matches = _semantic_match_map(graph_semantic_report)
    score_rows = _score_rows_by_case_id(evidence_score_report)
    rows = [
        _conflict_row(
            case,
            vlm_by_key.get(prediction_alignment_key(case.id)),
            graph_by_key.get(prediction_alignment_key(case.id)),
            vlm_matches.get(case.id),
            graph_matches.get(case.id),
            score_rows.get(case.id, {}),
        )
        for case in cases
    ]
    opportunity_case_ids = sorted(
        {
            row["case_id"]
            for row in rows
            if row.get("conflict_reason")
            in {"vlm_unknown_graph_plausible", "vlm_wrong_graph_correct"}
        }
    )
    report: dict[str, Any] = {
        "schema_version": VLM_GRAPH_CONFLICT_REPORT_SCHEMA_VERSION,
        "case_count": len(cases),
        "evidence_score_report_digest": evidence_score_report.get("report_digest"),
        "evidence_score_report_path": (
            str(evidence_score_report_path) if evidence_score_report_path is not None else None
        ),
        "graph_prediction_digest": qa_predictions_digest(graph_predictions),
        "graph_prediction_path": (
            str(graph_prediction_path) if graph_prediction_path is not None else None
        ),
        "graph_semantic_report_digest": graph_semantic_report.get("report_digest"),
        "graph_semantic_report_path": (
            str(graph_semantic_report_path) if graph_semantic_report_path is not None else None
        ),
        "opportunity_case_ids": opportunity_case_ids,
        "qa_digest": qa_dataset_digest(cases),
        "summary": _conflict_summary(rows),
        "vlm_prediction_digest": qa_predictions_digest(vlm_predictions),
        "vlm_prediction_path": str(vlm_prediction_path) if vlm_prediction_path is not None else None,
        "vlm_semantic_report_digest": vlm_semantic_report.get("report_digest"),
        "vlm_semantic_report_path": (
            str(vlm_semantic_report_path) if vlm_semantic_report_path is not None else None
        ),
        "cases": rows,
    }
    report["report_digest"] = vlm_graph_conflict_report_digest(report)
    return report


def vlm_graph_evidence_request_bundle(
    cases: Sequence[QACase],
    vlm_predictions: Sequence[QAPrediction],
    graph_predictions: Sequence[QAPrediction],
    evidence_score_report: Mapping[str, Any],
    conflict_report: Mapping[str, Any],
    *,
    detector_records: Sequence[Mapping[str, Any]] = (),
    max_cases: int | None = None,
) -> dict[str, Any]:
    case_by_id = {case.id: case for case in cases}
    vlm_by_key = _prediction_by_key(vlm_predictions)
    graph_by_key = _prediction_by_key(graph_predictions)
    score_rows = _score_rows_by_case_id(evidence_score_report)
    conflict_rows = [
        row
        for row in _mapping_rows(conflict_report.get("cases"))
        if row.get("needs_vlm_dsg_adjudication") is True
    ]
    detector_index = _detector_evidence_index(detector_records)
    case_inputs: list[dict[str, Any]] = []
    for row in conflict_rows:
        case_id = _string_or_none(row.get("case_id"))
        if case_id is None or case_id not in case_by_id:
            continue
        case = case_by_id[case_id]
        key = prediction_alignment_key(case.id)
        graph_prediction = graph_by_key.get(key)
        if graph_prediction is None:
            continue
        score_row = score_rows.get(case.id, {})
        case_inputs.append(
            _request_case(
                case,
                vlm_by_key.get(key),
                graph_prediction,
                score_row,
                row,
                detector_index=detector_index,
            )
        )
        if max_cases is not None and len(case_inputs) >= max_cases:
            break
    bundle: dict[str, Any] = {
        "schema_version": VLM_GRAPH_EVIDENCE_REQUEST_BUNDLE_SCHEMA_VERSION,
        "case_count": len(case_inputs),
        "conflict_report_digest": conflict_report.get("report_digest"),
        "evidence_score_report_digest": evidence_score_report.get("report_digest"),
        "forbidden_fields_absent": True,
        "instructions": {
            "task": "Choose an answer using the image/frame evidence, the initial VLM answer, and the DSG candidate. Return structured JSON only.",
            "must_not_use": [
                "evaluator-only answers",
                "hidden evaluator nodes",
                "hidden evaluator edges",
                "semantic evaluator labels",
            ],
        },
        "required_output_schema": {
            "answer_text": "string",
            "current_location": {
                "dst": "string or null",
                "dst_label": "string or null",
                "relation": "ON|INSIDE|IN_REGION|IN_ROOM|UNKNOWN",
                "step": "integer or null",
            },
            "decision": "accept_vlm|accept_dsg|reject_both|uncertain",
            "evidence_summary": "string",
        },
        "case_inputs": case_inputs,
    }
    bundle["request_bundle_digest"] = vlm_graph_evidence_request_bundle_digest(bundle)
    return bundle


def vlm_graph_evidence_score_report_digest(report: Mapping[str, Any]) -> str:
    return _digest_without(report, "report_digest")


def vlm_graph_conflict_report_digest(report: Mapping[str, Any]) -> str:
    return _digest_without(report, "report_digest")


def vlm_graph_evidence_request_bundle_digest(bundle: Mapping[str, Any]) -> str:
    return _digest_without(bundle, "request_bundle_digest")


def vlm_graph_evidence_score_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def vlm_graph_conflict_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def vlm_graph_evidence_request_bundle_json(bundle: Mapping[str, Any]) -> str:
    return json.dumps(bundle, indent=2, sort_keys=True) + "\n"


def save_vlm_graph_evidence_score_report(report: Mapping[str, Any], path: str | Path) -> Path:
    return _save_json_text(vlm_graph_evidence_score_report_json(report), path)


def save_vlm_graph_conflict_report(report: Mapping[str, Any], path: str | Path) -> Path:
    return _save_json_text(vlm_graph_conflict_report_json(report), path)


def save_vlm_graph_evidence_request_bundle(bundle: Mapping[str, Any], path: str | Path) -> Path:
    return _save_json_text(vlm_graph_evidence_request_bundle_json(bundle), path)


def prediction_alignment_key(prediction_id: str) -> str:
    return _OBSERVATION_AWARE_SUFFIX_RE.sub("", prediction_id)


def _prediction_by_key(predictions: Sequence[QAPrediction]) -> dict[str, QAPrediction]:
    by_key: dict[str, QAPrediction] = {}
    for prediction in predictions:
        key = prediction_alignment_key(prediction.id)
        if key not in by_key:
            by_key[key] = prediction
    return by_key


def _fused_prediction(
    vlm_prediction: QAPrediction | None,
    graph_prediction: QAPrediction | None,
    *,
    fusion_policy: str,
) -> QAPrediction:
    graph_fallback_reason = _graph_fallback_reason(
        graph_prediction,
        vlm_prediction=vlm_prediction,
        fusion_policy=fusion_policy,
    )
    if graph_prediction is not None and graph_fallback_reason is None:
        answer = deepcopy(graph_prediction.answer)
        answer["fusion"] = _fusion_metadata(
            vlm_prediction,
            graph_prediction,
            "graph_tool",
            fusion_policy=fusion_policy,
        )
        return QAPrediction(
            id=graph_prediction.id,
            answer=answer,
            evidence_nodes=graph_prediction.evidence_nodes,
            evidence_edges=graph_prediction.evidence_edges,
            confidence=graph_prediction.confidence,
            error=graph_prediction.error,
        )
    if vlm_prediction is None:
        if graph_prediction is None:
            raise SpatialQAError("Cannot fuse predictions without a VLM or graph prediction")
        answer = deepcopy(graph_prediction.answer)
        answer["fusion"] = _fusion_metadata(
            None,
            graph_prediction,
            "graph_tool",
            fusion_policy=fusion_policy,
            graph_fallback_reason="missing_vlm_prediction",
        )
        return QAPrediction(
            id=graph_prediction.id,
            answer=answer,
            evidence_nodes=graph_prediction.evidence_nodes,
            evidence_edges=graph_prediction.evidence_edges,
            confidence=graph_prediction.confidence,
            error=graph_prediction.error,
        )
    answer = deepcopy(vlm_prediction.answer)
    answer["fusion"] = _fusion_metadata(
        vlm_prediction,
        graph_prediction,
        "vlm",
        fusion_policy=fusion_policy,
        graph_fallback_reason=graph_fallback_reason,
    )
    return QAPrediction(
        id=graph_prediction.id if graph_prediction is not None else vlm_prediction.id,
        answer=answer,
        evidence_nodes=vlm_prediction.evidence_nodes,
        evidence_edges=vlm_prediction.evidence_edges,
        confidence=vlm_prediction.confidence,
        error=vlm_prediction.error,
    )


def _fusion_metadata(
    vlm_prediction: QAPrediction | None,
    graph_prediction: QAPrediction | None,
    fusion_source: str,
    *,
    fusion_policy: str,
    graph_fallback_reason: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "fusion_policy": fusion_policy,
        "fusion_source": fusion_source,
        "graph_prediction_id": graph_prediction.id if graph_prediction is not None else None,
        "vlm_prediction_id": vlm_prediction.id if vlm_prediction is not None else None,
    }
    if graph_fallback_reason is not None:
        metadata["graph_fallback_reason"] = graph_fallback_reason
    return metadata


def _graph_fallback_reason(
    graph_prediction: QAPrediction | None,
    *,
    vlm_prediction: QAPrediction | None,
    fusion_policy: str,
) -> str | None:
    if graph_prediction is None:
        return "missing_graph_prediction"
    if graph_prediction.error is not None:
        return "graph_error"
    location = graph_prediction.answer.get("current_location")
    if not isinstance(location, Mapping):
        return "missing_graph_current_location"
    relation = location.get("relation")
    if not isinstance(relation, str) or not relation.strip():
        return "missing_graph_relation"
    normalized_relation = relation.strip().upper()
    if normalized_relation == "IN_ROOM":
        return "room_level_graph_location"
    if normalized_relation not in _EXPLICIT_GRAPH_RELATIONS:
        return "non_explicit_graph_relation"
    if fusion_policy == VLM_GRAPH_TRUSTED_FUSION_POLICY:
        return _trusted_graph_fallback_reason(vlm_prediction, graph_prediction)
    return None


def _fusion_source_count(predictions: Sequence[QAPrediction], source: str) -> int:
    count = 0
    for prediction in predictions:
        fusion = prediction.answer.get("fusion")
        if isinstance(fusion, Mapping) and fusion.get("fusion_source") == source:
            count += 1
    return count


def _trusted_graph_fallback_reason(
    vlm_prediction: QAPrediction | None,
    graph_prediction: QAPrediction,
) -> str | None:
    target_label = _target_label_from_prediction_id(graph_prediction.id)
    answer_label = _canonical_label(graph_prediction.answer.get("label"))
    if answer_label is not None and target_label is not None and answer_label != target_label:
        return "target_label_mismatch"
    location = graph_prediction.answer.get("current_location")
    if not isinstance(location, Mapping):
        return "missing_graph_current_location"
    relation = _string_upper(location.get("relation"))
    support_label = _canonical_label(_string_or_none(location.get("dst")))
    if support_label is None:
        return "missing_support_label"
    if relation == "ON":
        if target_label is not None and support_label == target_label:
            return "implausible_support_label"
        if support_label not in _PLAUSIBLE_LOCATION_SUPPORT_LABELS:
            return "implausible_support_label"
    return None


def _validate_fusion_policy(fusion_policy: str) -> None:
    if fusion_policy not in {VLM_GRAPH_FUSION_POLICY, VLM_GRAPH_TRUSTED_FUSION_POLICY}:
        raise SpatialQAError(f"Unsupported VLM graph fusion policy: {fusion_policy}")


def _target_label_from_prediction_id(prediction_id: str) -> str | None:
    target_id = _target_object_id_from_prediction_id(prediction_id)
    return _canonical_label(target_id)


def _target_object_id_from_prediction_id(prediction_id: str) -> str | None:
    parts = prediction_alignment_key(prediction_id).split(":")
    for index, part in enumerate(parts):
        if part == "object_location":
            if index + 1 < len(parts):
                return parts[index + 1]
        if part.endswith("object_location"):
            candidate_index = index + 1
            if candidate_index < len(parts) and parts[candidate_index].isdigit():
                candidate_index += 1
            if candidate_index < len(parts):
                return parts[candidate_index]
    return None


def _canonical_label(value: object) -> str | None:
    text = _string_or_none(value)
    if text is None:
        return None
    if text == "ai2thor_room":
        return "room"
    label_parts: list[str] = []
    for part in text.split("_"):
        if _looks_numeric(part):
            break
        if part:
            label_parts.append(part.lower())
    if label_parts:
        return "".join(label_parts)
    return re.sub(r"[^a-z0-9]+", "", text.lower()) or None


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _string_upper(value: object) -> str | None:
    text = _string_or_none(value)
    if text is None:
        return None
    return text.upper()


def _load_json_mapping(path: str | Path, label: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(f"{label} JSON must be an object")
    return cast(dict[str, Any], payload)


def _digest_without(payload: Mapping[str, Any], digest_key: str) -> str:
    normalized = {key: value for key, value in payload.items() if key != digest_key}
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _save_json_text(text: str, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _mapping_rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _score_rows_by_case_id(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for row in _mapping_rows(report.get("cases")):
        case_id = _string_or_none(row.get("case_id"))
        if case_id is not None:
            rows[case_id] = row
    return rows


def _semantic_match_map(report: Mapping[str, Any]) -> dict[str, bool]:
    matches: dict[str, bool] = {}
    for row in _mapping_rows(report.get("cases")):
        case_id = _string_or_none(row.get("case_id"))
        if case_id is not None:
            matches[case_id] = row.get("semantic_match") is True
    return matches


def _detector_evidence_index(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        metadata = _mapping_or_empty(record.get("metadata"))
        frame_step = _int_or_none(record.get("step"))
        frame_evidence = {
            "depth_path": _string_or_none(record.get("depth_path")),
            "episode_id": _string_or_none(record.get("episode_id")),
            "rgb_path": _string_or_none(record.get("rgb_path")),
            "scene_id": _string_or_none(record.get("scene_id")),
            "step": frame_step,
        }
        detector_name = (
            _string_or_none(record.get("detector_name"))
            or _string_or_none(record.get("detector"))
            or _string_or_none(metadata.get("source_name"))
            or _string_or_none(metadata.get("detector"))
        )
        detections = record.get("detections")
        if not isinstance(detections, Sequence) or isinstance(detections, (str, bytes)):
            continue
        for detection in detections:
            if not isinstance(detection, Mapping):
                continue
            attributes = _mapping_or_empty(detection.get("attributes"))
            object_id = (
                _string_or_none(detection.get("object_id"))
                or _string_or_none(detection.get("track_id"))
                or _string_or_none(detection.get("detection_id"))
            )
            if object_id is None:
                continue
            evidence_kinds = _string_set(detection.get("evidence_kinds"))
            evidence_kinds.update(_string_set(attributes.get("evidence_kinds")))
            evidence_kinds.update(_string_set(metadata.get("required_evidence_kinds")))
            rgb_path = (
                _string_or_none(detection.get("rgb_path"))
                or _string_or_none(attributes.get("rgb_path"))
                or frame_evidence["rgb_path"]
            )
            depth_path = (
                _string_or_none(detection.get("depth_path"))
                or _string_or_none(attributes.get("depth_path"))
                or frame_evidence["depth_path"]
            )
            source_kind = (
                _string_or_none(attributes.get("source_kind"))
                or _string_or_none(metadata.get("source_kind"))
            )
            source_name = (
                _string_or_none(attributes.get("source_name"))
                or _string_or_none(metadata.get("source_name"))
                or detector_name
            )
            if rgb_path is not None:
                evidence_kinds.add("rgb")
            if depth_path is not None:
                evidence_kinds.add("depth")
            if detector_name is not None or source_kind == "detector":
                evidence_kinds.add("detector")
            bbox = _mapping_or_empty(detection.get("bbox"))
            evidence = {
                "bbox_2d_xyxy": _json_safe_or_none(
                    detection.get("bbox_2d_xyxy") or attributes.get("bbox_2d_xyxy")
                ),
                "bbox_3d_center": _json_safe_or_none(
                    detection.get("bbox_3d_center") or bbox.get("center")
                ),
                "bbox_3d_size": _json_safe_or_none(
                    detection.get("bbox_3d_size") or bbox.get("size")
                ),
                "confidence": _float_or_none(detection.get("confidence")),
                "depth_path": depth_path,
                "detection_id": _string_or_none(detection.get("detection_id")),
                "detector_name": detector_name or source_name,
                "episode_id": frame_evidence["episode_id"],
                "evidence_kinds": sorted(evidence_kinds),
                "label": _string_or_none(detection.get("label")) or _canonical_label(object_id),
                "mask_path": _string_or_none(detection.get("mask_path"))
                or _string_or_none(attributes.get("mask_path")),
                "object_id": object_id,
                "rgb_path": rgb_path,
                "scene_id": frame_evidence["scene_id"],
                "source_kind": source_kind,
                "source_name": source_name,
                "step": frame_step,
                "visible": detection.get("visible") is not False,
            }
            index.setdefault(object_id, []).append(evidence)
    for rows in index.values():
        rows.sort(key=lambda row: (_int_or_none(row.get("step")) is None, row.get("step") or 0))
    return index


def _evidence_score_row(
    case: QACase,
    graph_prediction: QAPrediction | None,
    *,
    detector_index: Mapping[str, Sequence[Mapping[str, Any]]],
    trust_threshold: float,
) -> dict[str, Any]:
    case_target_id = _string_or_none(case.question.get("object_id")) or _target_object_id_from_prediction_id(
        case.id
    )
    if graph_prediction is None:
        return _empty_score_row(
            case,
            target_object_id=case_target_id,
            reject_reason="missing_graph_prediction",
        )
    fallback_reason = _graph_fallback_reason(
        graph_prediction,
        vlm_prediction=None,
        fusion_policy=VLM_GRAPH_TRUSTED_FUSION_POLICY,
    )
    location = graph_prediction.answer.get("current_location")
    if not isinstance(location, Mapping):
        return _empty_score_row(
            case,
            target_object_id=case_target_id,
            graph_prediction_id=graph_prediction.id,
            reject_reason="missing_graph_current_location",
        )
    graph_object_id = _string_or_none(graph_prediction.answer.get("object_id"))
    target_object_id = graph_object_id or case_target_id or _target_object_id_from_prediction_id(
        graph_prediction.id
    )
    support_object_id = _string_or_none(location.get("dst"))
    target_label = _canonical_label(target_object_id)
    answer_label = _canonical_label(graph_prediction.answer.get("label"))
    support_label = _canonical_label(support_object_id)
    relation = _string_upper(location.get("relation"))
    graph_step = _int_or_none(location.get("step")) or _int_or_none(
        graph_prediction.answer.get("last_seen_step")
    )
    target_evidence = _best_detector_evidence(detector_index, target_object_id, graph_step)
    support_evidence = _best_detector_evidence(detector_index, support_object_id, graph_step)
    evidence_kinds = set(_string_set(target_evidence.get("evidence_kinds") if target_evidence else ()))
    evidence_kinds.update(_string_set(support_evidence.get("evidence_kinds") if support_evidence else ()))
    target_evidence_kinds = _string_set(
        target_evidence.get("evidence_kinds") if target_evidence else ()
    )
    checks = {
        "bbox_available": _has_bbox(target_evidence) and _has_bbox(support_evidence),
        "confidence_pass": graph_prediction.confidence >= 0.5,
        "current_location_fresh": _steps_compatible(case.step, graph_step),
        "relation_explicit": relation in _EXPLICIT_GRAPH_RELATIONS and relation != "IN_ROOM",
        "required_evidence_kinds_present": _REQUIRED_EVIDENCE_KINDS.issubset(evidence_kinds),
        "step_compatible": _steps_compatible(case.step, graph_step),
        "support_frame_available": _has_frame_evidence(support_evidence),
        "support_label_plausible": _support_label_plausible(
            relation,
            target_label=target_label,
            support_label=support_label,
        ),
        "room_level_target_plausible": _room_level_target_plausible(
            relation,
            target_label=target_label,
        ),
        "target_bbox_available": _has_bbox(target_evidence),
        "target_required_evidence_kinds_present": _REQUIRED_EVIDENCE_KINDS.issubset(
            target_evidence_kinds
        ),
        "target_frame_available": _has_frame_evidence(target_evidence),
        "target_label_match": _labels_match(target_label, answer_label),
    }
    trust_scope = "explicit_relation"
    room_level_trusted = _room_level_target_trusted(checks)
    if fallback_reason == "room_level_graph_location" and room_level_trusted:
        fallback_reason = None
        trust_scope = "room_level_target"
    elif fallback_reason is None:
        fallback_reason = _score_reject_reason(checks)
    score = _room_level_graph_trust_score(checks) if room_level_trusted else _graph_trust_score(checks)
    if fallback_reason is not None:
        score = min(score, 0.49)
    trusted = fallback_reason is None and score >= trust_threshold
    return {
        "case_id": case.id,
        "checks": checks,
        "dsg_candidate": {
            "current_location": {
                "dst": support_object_id,
                "relation": relation,
                "step": graph_step,
            },
            "support_label": support_label,
            "target_label": target_label,
        },
        "graph_confidence": graph_prediction.confidence,
        "graph_prediction_id": graph_prediction.id,
        "graph_trust_score": round(score, 6),
        "reject_reason": None if trusted else fallback_reason,
        "relation": relation,
        "support_label": support_label,
        "support_object_evidence": support_evidence,
        "support_object_id": support_object_id,
        "target_label": target_label,
        "target_object_evidence": target_evidence,
        "target_object_id": target_object_id,
        "trust_scope": trust_scope if trusted else None,
        "trusted": trusted,
    }


def _empty_score_row(
    case: QACase,
    *,
    target_object_id: str | None,
    reject_reason: str,
    graph_prediction_id: str | None = None,
) -> dict[str, Any]:
    checks = {
        "bbox_available": False,
        "confidence_pass": False,
        "current_location_fresh": False,
        "relation_explicit": False,
        "required_evidence_kinds_present": False,
        "step_compatible": False,
        "support_frame_available": False,
        "support_label_plausible": False,
        "target_frame_available": False,
        "target_label_match": False,
        "room_level_target_plausible": False,
        "target_bbox_available": False,
        "target_required_evidence_kinds_present": False,
    }
    return {
        "case_id": case.id,
        "checks": checks,
        "dsg_candidate": None,
        "graph_confidence": None,
        "graph_prediction_id": graph_prediction_id,
        "graph_trust_score": 0.0,
        "reject_reason": reject_reason,
        "relation": None,
        "support_label": None,
        "support_object_evidence": None,
        "support_object_id": None,
        "target_label": _canonical_label(target_object_id),
        "target_object_evidence": None,
        "target_object_id": target_object_id,
        "trust_scope": None,
        "trusted": False,
    }


def _evidence_score_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reject_reasons: dict[str, int] = {}
    score_total = 0.0
    explicit_count = 0
    room_level_count = 0
    for row in rows:
        score_total += float(row.get("graph_trust_score") or 0.0)
        if row.get("trusted") is True and row.get("trust_scope") == "explicit_relation":
            explicit_count += 1
        if row.get("trusted") is True and row.get("trust_scope") == "room_level_target":
            room_level_count += 1
        reason = _string_or_none(row.get("reject_reason"))
        if reason is not None:
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
    case_count = len(rows)
    trusted_count = sum(1 for row in rows if row.get("trusted") is True)
    return {
        "average_graph_trust_score": round(score_total / case_count, 6) if case_count else 0.0,
        "case_count": case_count,
        "explicit_trusted_relation_count": explicit_count,
        "reject_reason_counts": dict(sorted(reject_reasons.items())),
        "rejected_case_count": case_count - trusted_count,
        "trusted_case_count": trusted_count,
        "trusted_case_rate": round(trusted_count / case_count, 6) if case_count else 0.0,
        "trusted_room_level_count": room_level_count,
    }


def _conflict_row(
    case: QACase,
    vlm_prediction: QAPrediction | None,
    graph_prediction: QAPrediction | None,
    vlm_match: bool | None,
    graph_match: bool | None,
    score_row: Mapping[str, Any],
) -> dict[str, Any]:
    vlm_correct = vlm_match is True
    graph_correct = graph_match is True
    if vlm_correct and graph_correct:
        status = "both_correct"
    elif vlm_correct and not graph_correct:
        status = "vlm_correct_graph_wrong"
    elif not vlm_correct and graph_correct:
        status = "vlm_wrong_graph_correct"
    else:
        status = "both_wrong"
    vlm_unknown_like = _prediction_unknown_like(vlm_prediction)
    graph_plausible = score_row.get("trusted") is True or float(
        score_row.get("graph_trust_score") or 0.0
    ) >= VLM_GRAPH_TRUST_THRESHOLD
    conflict_reason: str | None = None
    if vlm_unknown_like and graph_plausible:
        conflict_reason = "vlm_unknown_graph_plausible"
    elif status == "vlm_wrong_graph_correct":
        conflict_reason = "vlm_wrong_graph_correct"
    elif status == "vlm_correct_graph_wrong":
        conflict_reason = "vlm_correct_graph_wrong"
    elif status == "both_wrong" and graph_plausible:
        conflict_reason = "both_wrong_graph_plausible"
    observable_conflict_reason = _observable_conflict_reason(
        vlm_prediction,
        graph_prediction,
        vlm_unknown_like=vlm_unknown_like,
        graph_plausible=graph_plausible,
    )
    return {
        "case_id": case.id,
        "conflict_reason": conflict_reason,
        "conflict_status": status,
        "graph_prediction_id": graph_prediction.id if graph_prediction is not None else None,
        "graph_reject_reason": score_row.get("reject_reason"),
        "graph_semantic_match": graph_correct,
        "graph_trust_score": score_row.get("graph_trust_score"),
        "needs_vlm_dsg_adjudication": observable_conflict_reason is not None,
        "observable_conflict_reason": observable_conflict_reason,
        "question_type": case.question_type,
        "vlm_prediction_id": vlm_prediction.id if vlm_prediction is not None else None,
        "vlm_semantic_match": vlm_correct,
        "vlm_unknown_like": vlm_unknown_like,
    }


def _conflict_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = {
        "both_correct": 0,
        "both_wrong": 0,
        "vlm_correct_graph_wrong": 0,
        "vlm_wrong_graph_correct": 0,
    }
    reason_counts: dict[str, int] = {}
    for row in rows:
        status = _string_or_none(row.get("conflict_status"))
        if status in statuses:
            statuses[status] += 1
        reason = _string_or_none(row.get("conflict_reason"))
        if reason is not None:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "adjudication_case_count": sum(
            1 for row in rows if row.get("needs_vlm_dsg_adjudication") is True
        ),
        "both_correct_count": statuses["both_correct"],
        "both_wrong_count": statuses["both_wrong"],
        "case_count": len(rows),
        "conflict_reason_counts": dict(sorted(reason_counts.items())),
        "vlm_correct_graph_wrong_count": statuses["vlm_correct_graph_wrong"],
        "vlm_unknown_graph_plausible_count": reason_counts.get(
            "vlm_unknown_graph_plausible", 0
        ),
        "vlm_wrong_graph_correct_count": statuses["vlm_wrong_graph_correct"],
    }


def _request_case(
    case: QACase,
    vlm_prediction: QAPrediction | None,
    graph_prediction: QAPrediction,
    score_row: Mapping[str, Any],
    conflict_row: Mapping[str, Any],
    *,
    detector_index: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    dsg_candidate = _candidate_without_forbidden(score_row.get("dsg_candidate"))
    current_location = dsg_candidate.get("current_location")
    support_id = None
    if isinstance(current_location, Mapping):
        support_id = _string_or_none(current_location.get("dst"))
    target_id = _string_or_none(score_row.get("target_object_id")) or _string_or_none(
        case.question.get("object_id")
    )
    graph_step = (
        _int_or_none(current_location.get("step")) if isinstance(current_location, Mapping) else None
    )
    target_evidence = _candidate_without_forbidden(
        score_row.get("target_object_evidence")
        or _best_detector_evidence(detector_index, target_id, graph_step)
    )
    support_evidence = _candidate_without_forbidden(
        score_row.get("support_object_evidence")
        or _best_detector_evidence(detector_index, support_id, graph_step)
    )
    return {
        "answer_type": case.answer_type,
        "case_id": case.id,
        "choices": list(case.choices),
        "conflict_reason": conflict_row.get("observable_conflict_reason"),
        "crop_refs": _crop_refs(
            ("target", target_evidence),
            ("support", support_evidence),
        ),
        "dsg_candidate": dsg_candidate,
        "episode_id": case.episode_id,
        "frame_refs": _frame_refs(target_evidence, support_evidence),
        "graph_prediction_id": graph_prediction.id,
        "graph_reject_reason": score_row.get("reject_reason"),
        "graph_trust_score": score_row.get("graph_trust_score"),
        "question": deepcopy(case.question),
        "question_type": case.question_type,
        "scene_id": case.scene_id,
        "step": case.step,
        "support_object_evidence": support_evidence,
        "target_object_evidence": target_evidence,
        "vlm_confidence": vlm_prediction.confidence if vlm_prediction is not None else None,
        "vlm_initial_answer": deepcopy(vlm_prediction.answer) if vlm_prediction is not None else None,
        "vlm_prediction_id": vlm_prediction.id if vlm_prediction is not None else None,
    }


def _candidate_without_forbidden(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    output: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        if key_text in _FORBIDDEN_REQUEST_KEYS:
            continue
        if isinstance(item, Mapping):
            output[key_text] = _candidate_without_forbidden(item)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            output[key_text] = [
                _candidate_without_forbidden(member) if isinstance(member, Mapping) else member
                for member in item
            ]
        else:
            output[key_text] = item
    return output


def _frame_refs(*evidence_rows: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None, int | None]] = set()
    for evidence in evidence_rows:
        if not evidence:
            continue
        rgb_path = _string_or_none(evidence.get("rgb_path"))
        depth_path = _string_or_none(evidence.get("depth_path"))
        step = _int_or_none(evidence.get("step"))
        key = (rgb_path, depth_path, step)
        if key in seen or (rgb_path is None and depth_path is None):
            continue
        seen.add(key)
        refs.append({"depth_path": depth_path, "rgb_path": rgb_path, "step": step})
    return refs


def _crop_refs(*role_evidence_rows: tuple[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for role, evidence in role_evidence_rows:
        if not evidence:
            continue
        rgb_path = _string_or_none(evidence.get("rgb_path"))
        bbox_2d = evidence.get("bbox_2d_xyxy")
        object_id = _string_or_none(evidence.get("object_id"))
        if rgb_path is None or object_id is None or not isinstance(bbox_2d, Sequence):
            continue
        refs.append(
            {
                "bbox_2d_xyxy": list(bbox_2d),
                "object_id": object_id,
                "rgb_path": rgb_path,
                "role": role,
                "step": _int_or_none(evidence.get("step")),
            }
        )
    return refs


def _best_detector_evidence(
    detector_index: Mapping[str, Sequence[Mapping[str, Any]]],
    object_id: str | None,
    step: int | None,
) -> dict[str, Any] | None:
    if object_id is None:
        return None
    rows = detector_index.get(object_id)
    if not rows:
        return None
    compatible = [row for row in rows if _steps_compatible(step, _int_or_none(row.get("step")))]
    if compatible:
        return dict(compatible[-1])
    return dict(rows[-1])


def _steps_compatible(left: int | None, right: int | None) -> bool:
    if left is None or right is None:
        return False
    return left == right or left % 100000 == right % 100000


def _support_label_plausible(
    relation: str | None,
    *,
    target_label: str | None,
    support_label: str | None,
) -> bool:
    if relation is None or support_label is None:
        return False
    if relation == "ON":
        return support_label != target_label and support_label in _PLAUSIBLE_LOCATION_SUPPORT_LABELS
    if relation in _EXPLICIT_GRAPH_RELATIONS and relation != "IN_ROOM":
        return support_label != target_label
    return False


def _room_level_target_plausible(
    relation: str | None,
    *,
    target_label: str | None,
) -> bool:
    return relation == "IN_ROOM" and target_label in _PLAUSIBLE_ROOM_LEVEL_TARGET_LABELS


def _room_level_target_trusted(checks: Mapping[str, bool]) -> bool:
    return all(
        checks.get(name) is True
        for name in (
            "confidence_pass",
            "current_location_fresh",
            "room_level_target_plausible",
            "step_compatible",
            "target_bbox_available",
            "target_frame_available",
            "target_label_match",
            "target_required_evidence_kinds_present",
        )
    )


def _labels_match(target_label: str | None, answer_label: str | None) -> bool:
    return target_label is not None and (answer_label is None or target_label == answer_label)


def _score_reject_reason(checks: Mapping[str, bool]) -> str | None:
    order = (
        ("target_label_match", "target_label_mismatch"),
        ("support_label_plausible", "implausible_support_label"),
        ("relation_explicit", "non_explicit_graph_relation"),
        ("required_evidence_kinds_present", "missing_required_evidence_kinds"),
        ("step_compatible", "step_incompatible"),
        ("confidence_pass", "low_graph_confidence"),
        ("target_frame_available", "missing_target_frame_evidence"),
        ("support_frame_available", "missing_support_frame_evidence"),
        ("bbox_available", "missing_bbox_evidence"),
        ("current_location_fresh", "stale_current_location"),
    )
    for check_name, reason in order:
        if checks.get(check_name) is not True:
            return reason
    return None


def _graph_trust_score(checks: Mapping[str, bool]) -> float:
    weights = {
        "bbox_available": 0.05,
        "confidence_pass": 0.10,
        "current_location_fresh": 0.05,
        "relation_explicit": 0.15,
        "required_evidence_kinds_present": 0.15,
        "step_compatible": 0.10,
        "support_frame_available": 0.05,
        "support_label_plausible": 0.15,
        "target_frame_available": 0.05,
        "target_label_match": 0.15,
    }
    return sum(weight for check, weight in weights.items() if checks.get(check) is True)


def _room_level_graph_trust_score(checks: Mapping[str, bool]) -> float:
    weights = {
        "confidence_pass": 0.10,
        "current_location_fresh": 0.10,
        "room_level_target_plausible": 0.15,
        "step_compatible": 0.10,
        "target_bbox_available": 0.15,
        "target_frame_available": 0.10,
        "target_label_match": 0.10,
        "target_required_evidence_kinds_present": 0.20,
    }
    return sum(weight for check, weight in weights.items() if checks.get(check) is True)


def _has_frame_evidence(evidence: Mapping[str, Any] | None) -> bool:
    return bool(evidence and evidence.get("rgb_path") and evidence.get("depth_path"))


def _has_bbox(evidence: Mapping[str, Any] | None) -> bool:
    return bool(
        evidence
        and evidence.get("bbox_2d_xyxy") is not None
        and evidence.get("bbox_3d_center") is not None
    )


def _prediction_unknown_like(prediction: QAPrediction | None) -> bool:
    if prediction is None:
        return True
    text_parts: list[str] = []
    for key in ("text", "answer", "reasoning"):
        value = prediction.answer.get(key)
        if isinstance(value, str):
            text_parts.append(value.lower())
    text = " ".join(text_parts)
    if prediction.confidence <= 0.2:
        return True
    unknown_markers = ("unknown", "not visible", "cannot determine", "uncertain", "not sure")
    return any(marker in text for marker in unknown_markers)


def _observable_conflict_reason(
    vlm_prediction: QAPrediction | None,
    graph_prediction: QAPrediction | None,
    *,
    vlm_unknown_like: bool,
    graph_plausible: bool,
) -> str | None:
    if not graph_plausible:
        return None
    if vlm_unknown_like:
        return "vlm_unknown_graph_plausible"
    if graph_prediction is None or vlm_prediction is None:
        return None
    if _answer_signature(vlm_prediction.answer) != _answer_signature(graph_prediction.answer):
        return "vlm_graph_disagreement_graph_plausible"
    return None


def _answer_signature(answer: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    location = answer.get("current_location")
    if isinstance(location, Mapping):
        return (
            _string_upper(location.get("relation")),
            _canonical_label(location.get("dst")),
            _canonical_label(answer.get("object_id")),
        )
    text = _string_or_none(answer.get("text"))
    if text is not None:
        return (None, _canonical_label(text), None)
    return (None, None, None)


def _forbidden_request_key_hits(value: object, *, prefix: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            current_path = f"{prefix}.{key_text}"
            if key_text in _FORBIDDEN_REQUEST_KEYS:
                hits.append(current_path)
            hits.extend(_forbidden_request_key_hits(item, prefix=current_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            hits.extend(_forbidden_request_key_hits(item, prefix=f"{prefix}[{index}]"))
    return hits


def _string_set(value: object) -> set[str]:
    if isinstance(value, str):
        return {value.strip().lower()} if value.strip() else set()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    output: set[str] = set()
    for item in value:
        text = _string_or_none(item)
        if text is not None:
            output.add(text.lower())
    return output


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _json_safe_or_none(value: object) -> object:
    if value is None:
        return None
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return None
    return value


def _mapping_or_empty(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}
