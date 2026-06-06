from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


CASE_ATTRIBUTION_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.active-qa-v2-case-attribution-report.v1"
)
ADJUDICATION_DERIVED_FUSION_POLICY = (
    "adjudication_derived_trusted_graph_or_vlm_fallback"
)
ADJUDICATION_DERIVED_FUSION_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.adjudication-derived-fusion-report.v1"
)


def active_qa_v2_case_attribution_report(
    records: Sequence[Mapping[str, Any]],
    vlm_predictions: Mapping[str, Mapping[str, Any]],
    graph_predictions: Mapping[str, Mapping[str, Any]],
    trusted_predictions: Mapping[str, Mapping[str, Any]],
    adjudicated_predictions: Mapping[str, Mapping[str, Any]],
    *,
    match_mode: str = "p50_comparison",
    source_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    rows = [
        _case_attribution_row(
            record,
            vlm_predictions.get(str(record.get("id"))),
            graph_predictions.get(str(record.get("id"))),
            trusted_predictions.get(str(record.get("id"))),
            adjudicated_predictions.get(str(record.get("id"))),
            match_mode=match_mode,
        )
        for record in records
    ]
    report: dict[str, Any] = {
        "schema_version": CASE_ATTRIBUTION_SCHEMA_VERSION,
        "case_count": len(rows),
        "match_mode": match_mode,
        "known_limitations": [
            "p50_comparison mode reproduces the existing P50 comparison report counts.",
            "This evaluator-facing report may include gold-derived match labels and must not be sent to external VLMs.",
        ],
        "source_paths": dict(source_paths or {}),
        "summary": _case_attribution_summary(rows),
        "cases": rows,
    }
    report["report_digest"] = stable_digest_without(report, "report_digest")
    return report


def adjudication_derived_fusion_predictions(
    records: Sequence[Mapping[str, Any]],
    vlm_predictions: Mapping[str, Mapping[str, Any]],
    graph_predictions: Mapping[str, Mapping[str, Any]],
    *,
    vlm_confidence_threshold: float = 0.55,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for record in records:
        case_id = str(record.get("id"))
        vlm_prediction = vlm_predictions.get(case_id)
        graph_prediction = graph_predictions.get(case_id)
        use_graph, reason = _should_use_graph_for_adjudication_derived_fusion(
            record,
            vlm_prediction,
            graph_prediction,
            vlm_confidence_threshold=vlm_confidence_threshold,
        )
        selected = graph_prediction if use_graph else vlm_prediction
        fallback = vlm_prediction if use_graph else graph_prediction
        if selected is None:
            selected = fallback
        if selected is None:
            selected = {"id": case_id, "answer": {"current_location": {"relation": "UNKNOWN"}}}
        prediction = _prediction_with_fusion_metadata(
            case_id,
            selected,
            graph_prediction,
            vlm_prediction,
            fusion_source="graph_tool" if use_graph else "vlm",
            selection_reason=reason,
        )
        predictions.append(prediction)
    report: dict[str, Any] = {
        "schema_version": ADJUDICATION_DERIVED_FUSION_REPORT_SCHEMA_VERSION,
        "fusion_policy": ADJUDICATION_DERIVED_FUSION_POLICY,
        "calibration_kind": "same_dataset_adjudication_derived",
        "not_final_research_claim": True,
        "vlm_confidence_threshold": vlm_confidence_threshold,
        "summary": _fusion_summary(predictions),
        "prediction_digest": predictions_digest(predictions),
    }
    report["report_digest"] = stable_digest_without(report, "report_digest")
    return predictions, report


def active_qa_v2_case_attribution_markdown(report: Mapping[str, Any]) -> str:
    summary = _mapping_or_empty(report.get("summary"))
    lines = [
        "# P51 active QA v2 逐例归因",
        "",
        f"- case_count: {report.get('case_count', 0)}",
        f"- adjudicated wins: {summary.get('adjudicated_win_count', 0)}",
        f"- adjudicated failures: {summary.get('adjudicated_failure_count', 0)}",
        f"- tie correct: {summary.get('tie_correct_count', 0)}",
        f"- adjudicated losses: {summary.get('adjudicated_loss_count', 0)}",
        "",
        "## 归因分布",
        "",
        "| attribution | count |",
        "| --- | ---: |",
    ]
    for label, count in sorted(_mapping_or_empty(summary.get("primary_attribution_counts")).items()):
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- 本报告使用 evaluator gold answer 判断 win/failure，因此只能作为离线诊断。",
            "- 不得把本报告内容放入外部 VLM request bundle。",
        ]
    )
    return "\n".join(lines) + "\n"


def adjudication_derived_fusion_markdown(report: Mapping[str, Any]) -> str:
    summary = _mapping_or_empty(report.get("summary"))
    return (
        "# P52 adjudication-derived trusted fusion\n\n"
        f"- fusion_policy: `{report.get('fusion_policy')}`\n"
        f"- calibration_kind: `{report.get('calibration_kind')}`\n"
        f"- not_final_research_claim: `{report.get('not_final_research_claim')}`\n"
        f"- graph_source_count: {summary.get('graph_source_count', 0)}\n"
        f"- vlm_source_count: {summary.get('vlm_source_count', 0)}\n\n"
        "该策略把 P50 adjudication 的经验固化为 deterministic gate，"
        "但当前仍是 same-dataset calibration；需要在 P54/P55 的 held-out episode 上验证后，"
        "才能作为新的独立研究结论。\n"
    )


def save_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def save_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return output


def stable_digest_without(payload: Mapping[str, Any], omitted_key: str) -> str:
    normalized = {key: value for key, value in payload.items() if key != omitted_key}
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def predictions_digest(predictions: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            [dict(row) for row in predictions],
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _case_attribution_row(
    record: Mapping[str, Any],
    vlm_prediction: Mapping[str, Any] | None,
    graph_prediction: Mapping[str, Any] | None,
    trusted_prediction: Mapping[str, Any] | None,
    adjudicated_prediction: Mapping[str, Any] | None,
    *,
    match_mode: str,
) -> dict[str, Any]:
    vlm_match = _match(record, vlm_prediction, match_mode)
    graph_match = _match(record, graph_prediction, match_mode)
    trusted_match = _match(record, trusted_prediction, match_mode)
    adjudicated_match = _match(record, adjudicated_prediction, match_mode)
    outcome = _case_outcome(vlm_match, adjudicated_match)
    answer = _mapping_or_empty(adjudicated_prediction.get("answer") if adjudicated_prediction else {})
    selected_candidate = _string(answer.get("selected_candidate"))
    decision = _string(answer.get("decision"))
    row = {
        "case_id": str(record.get("id")),
        "episode_id": _string(record.get("episode_id")),
        "question_type": _string(record.get("question_type")),
        "split": _string(record.get("split")),
        "vlm_semantic_match": vlm_match,
        "graph_semantic_match": graph_match,
        "trusted_semantic_match": trusted_match,
        "adjudicated_semantic_match": adjudicated_match,
        "outcome": outcome,
        "adjudication_decision": decision,
        "selected_candidate": selected_candidate,
        "primary_attribution": _primary_attribution(
            outcome,
            _string(record.get("question_type")),
            selected_candidate,
            decision,
            graph_match=graph_match,
            trusted_match=trusted_match,
        ),
    }
    return row


def _case_outcome(vlm_match: bool, adjudicated_match: bool) -> str:
    if adjudicated_match and not vlm_match:
        return "adjudicated_win"
    if vlm_match and not adjudicated_match:
        return "adjudicated_loss"
    if adjudicated_match and vlm_match:
        return "tie_correct"
    return "adjudicated_failure"


def _primary_attribution(
    outcome: str,
    question_type: str | None,
    selected_candidate: str | None,
    decision: str | None,
    *,
    graph_match: bool,
    trusted_match: bool,
) -> str:
    if outcome == "adjudicated_win":
        if selected_candidate == "graph_tool_dsg":
            if question_type == "support_relation":
                return "dsg_support_relation_correction"
            if question_type == "temporal_last_seen":
                return "dsg_temporal_memory_correction"
            if question_type == "situated_egocentric":
                return "dsg_situated_evidence_correction"
            if question_type == "object_location":
                return "dsg_location_correction"
            return "dsg_evidence_correction"
        if selected_candidate == "vlm":
            return "vlm_answer_normalization_or_reformat"
        return "adjudicator_other_win"
    if outcome == "adjudicated_failure":
        if decision == "reject_both":
            return "adjudicator_rejected_both"
        if decision == "uncertain":
            return "adjudicator_uncertain"
        if selected_candidate == "vlm":
            return "accepted_vlm_but_wrong"
        if selected_candidate == "graph_tool_dsg":
            return "accepted_dsg_but_wrong"
        if graph_match and not trusted_match:
            return "trusted_gate_missed_correct_graph"
        return "both_wrong_or_unresolved"
    if outcome == "adjudicated_loss":
        return "adjudicator_regression"
    return "both_correct"


def _case_attribution_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes = Counter(_string(row.get("outcome")) for row in rows)
    by_question_type: dict[str, dict[str, int]] = {}
    for row in rows:
        qtype = _string(row.get("question_type")) or "unknown"
        outcome = _string(row.get("outcome")) or "unknown"
        by_question_type.setdefault(qtype, {})
        by_question_type[qtype][outcome] = by_question_type[qtype].get(outcome, 0) + 1
    return {
        "adjudicated_failure_count": outcomes.get("adjudicated_failure", 0),
        "adjudicated_loss_count": outcomes.get("adjudicated_loss", 0),
        "adjudicated_win_count": outcomes.get("adjudicated_win", 0),
        "case_count": len(rows),
        "outcome_counts": dict(sorted(outcomes.items())),
        "outcomes_by_question_type": {
            key: dict(sorted(value.items())) for key, value in sorted(by_question_type.items())
        },
        "primary_attribution_counts": dict(
            sorted(Counter(_string(row.get("primary_attribution")) for row in rows).items())
        ),
        "selected_candidate_counts": dict(
            sorted(Counter(_string(row.get("selected_candidate")) for row in rows).items())
        ),
        "tie_correct_count": outcomes.get("tie_correct", 0),
    }


def _should_use_graph_for_adjudication_derived_fusion(
    record: Mapping[str, Any],
    vlm_prediction: Mapping[str, Any] | None,
    graph_prediction: Mapping[str, Any] | None,
    *,
    vlm_confidence_threshold: float,
) -> tuple[bool, str]:
    if graph_prediction is None or not _has_structured_location(graph_prediction):
        return False, "missing_structured_graph_prediction"
    qtype = _string(record.get("question_type"))
    graph_relation = _relation(graph_prediction)
    if qtype in {"object_location", "support_relation"} and graph_relation not in {None, "UNKNOWN"}:
        return True, "relation_or_object_location_graph_prior"
    if _prediction_unknown_like(vlm_prediction) or _prediction_confidence(vlm_prediction) <= vlm_confidence_threshold:
        return True, "vlm_unknown_or_low_confidence"
    return False, "vlm_high_confidence_fallback"


def _prediction_with_fusion_metadata(
    case_id: str,
    selected: Mapping[str, Any],
    graph_prediction: Mapping[str, Any] | None,
    vlm_prediction: Mapping[str, Any] | None,
    *,
    fusion_source: str,
    selection_reason: str,
) -> dict[str, Any]:
    prediction = deepcopy(dict(selected))
    prediction["id"] = case_id
    answer = deepcopy(_mapping_or_empty(prediction.get("answer")))
    answer["fusion"] = {
        "calibration_kind": "same_dataset_adjudication_derived",
        "fusion_policy": ADJUDICATION_DERIVED_FUSION_POLICY,
        "fusion_source": fusion_source,
        "graph_prediction_id": graph_prediction.get("id") if graph_prediction else None,
        "not_final_research_claim": True,
        "selection_reason": selection_reason,
        "vlm_prediction_id": vlm_prediction.get("id") if vlm_prediction else None,
    }
    prediction["answer"] = answer
    prediction.setdefault("schema_version", "dsg-spatialqa-lab.qa-prediction.v1")
    return prediction


def _fusion_summary(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for prediction in predictions:
        fusion = _mapping_or_empty(_mapping_or_empty(prediction.get("answer")).get("fusion"))
        source_counts[_string(fusion.get("fusion_source")) or "unknown"] += 1
        reason_counts[_string(fusion.get("selection_reason")) or "unknown"] += 1
    return {
        "case_count": len(predictions),
        "graph_source_count": source_counts.get("graph_tool", 0),
        "selection_reason_counts": dict(sorted(reason_counts.items())),
        "vlm_source_count": source_counts.get("vlm", 0),
    }


def semantic_match(record: Mapping[str, Any], prediction: Mapping[str, Any] | None) -> bool:
    if prediction is None:
        return False
    gold = _answer(record.get("answer"))
    pred = _answer(prediction.get("answer"))
    if pred.get("relation") == gold.get("relation"):
        pred_dst = pred.get("dst")
        gold_dst = gold.get("dst")
        pred_label = pred.get("dst_label")
        gold_label = gold.get("dst_label")
        if pred_dst is not None and gold_dst is not None and pred_dst == gold_dst:
            return True
        if (
            pred_label is not None
            and gold_label is not None
            and str(pred_label).lower() == str(gold_label).lower()
        ):
            return True
    answer = _mapping_or_empty(prediction.get("answer"))
    if isinstance(answer.get("current_location"), Mapping):
        return False
    answer_text = " ".join(
        str(answer.get(key, "")).lower() for key in ("answer_text", "text", "reasoning_summary")
    )
    dst_label = gold.get("dst_label")
    return bool(dst_label) and str(dst_label).lower() in answer_text


def p50_comparison_semantic_match(
    record: Mapping[str, Any],
    prediction: Mapping[str, Any] | None,
) -> bool:
    if prediction is None:
        return False
    gold = _answer(record.get("answer"))
    pred = _answer(prediction.get("answer"))
    if pred.get("relation") == gold.get("relation") and (
        pred.get("dst") == gold.get("dst") or pred.get("dst_label") == gold.get("dst_label")
    ):
        return True
    text = str(_mapping_or_empty(prediction.get("answer")).get("text", "")).lower()
    return bool(gold.get("dst_label")) and str(gold["dst_label"]).lower() in text


def _match(
    record: Mapping[str, Any],
    prediction: Mapping[str, Any] | None,
    match_mode: str,
) -> bool:
    if match_mode == "p50_comparison":
        return p50_comparison_semantic_match(record, prediction)
    if match_mode == "structured_text":
        return semantic_match(record, prediction)
    raise ValueError(f"Unsupported active QA v2 attribution match mode: {match_mode}")


def _answer(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    location = value.get("current_location")
    if isinstance(location, Mapping):
        merged = dict(value)
        merged.update(location)
        return merged
    return dict(value)


def _has_structured_location(prediction: Mapping[str, Any]) -> bool:
    answer = _mapping_or_empty(prediction.get("answer"))
    return isinstance(answer.get("current_location"), Mapping) or "relation" in answer


def _relation(prediction: Mapping[str, Any]) -> str | None:
    relation = _answer(prediction.get("answer")).get("relation")
    text = _string(relation)
    return text.upper() if text is not None else None


def _prediction_unknown_like(prediction: Mapping[str, Any] | None) -> bool:
    if prediction is None:
        return True
    answer = json.dumps(prediction.get("answer", {}), sort_keys=True).lower()
    return any(token in answer for token in ("unknown", "not visible", "cannot", "unable"))


def _prediction_confidence(prediction: Mapping[str, Any] | None) -> float:
    if prediction is None:
        return 0.0
    confidence = prediction.get("confidence")
    if isinstance(confidence, int | float):
        return float(confidence)
    answer = _mapping_or_empty(prediction.get("answer"))
    answer_confidence = answer.get("confidence")
    if isinstance(answer_confidence, int | float):
        return float(answer_confidence)
    return 0.0


def _mapping_or_empty(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
