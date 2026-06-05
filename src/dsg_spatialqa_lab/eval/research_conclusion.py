from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from math import comb
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.eval.qa_metrics import (
    load_qa_eval_delta_report,
    load_qa_eval_report,
    qa_eval_report_digest,
    validate_qa_eval_report,
)
from dsg_spatialqa_lab.schema import SpatialQAError


RESEARCH_CONCLUSION_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.research-conclusion-report.v1"
)
DEFAULT_REQUIRED_CONCLUSION_SOURCE_KINDS = (
    "caption_memory",
    "graph_text",
    "multi_frame_vlm",
    "vlm",
)
DEFAULT_CONCLUSION_ALPHA = 0.05
DEFAULT_MIN_CANDIDATE_EXACT_MATCH_RATE = 0.2
DEFAULT_MIN_CANDIDATE_EXACT_MATCH_COUNT = 15
DEFAULT_MIN_EXACT_MATCH_RATE_DELTA = 0.05
DEFAULT_MIN_GRAPH_OBJECT_RECALL = 0.3
DEFAULT_MIN_OBSERVATION_AWARE_CASE_COUNT = 30
DEFAULT_MIN_CONCLUSION_QUESTION_TYPE_COUNT = 2
DEFAULT_MAX_UNLOCATED_OBJECT_COUNT = 0
CONCLUSION_EVALUATION_SCOPES = ("full_oracle", "observation_aware")


def research_conclusion_report(
    real_readiness_report: Mapping[str, Any],
    offline_control_result_report: Mapping[str, Any],
    *,
    predicted_dsg_evidence_report: Mapping[str, Any] | None = None,
    graph_eval_report: Mapping[str, Any] | None = None,
    error_attribution_report: Mapping[str, Any] | None = None,
    qa_observability_report: Mapping[str, Any] | None = None,
    real_readiness_report_path: str | Path | None = None,
    offline_control_result_report_path: str | Path | None = None,
    predicted_dsg_evidence_report_path: str | Path | None = None,
    graph_eval_report_path: str | Path | None = None,
    error_attribution_report_path: str | Path | None = None,
    qa_observability_report_path: str | Path | None = None,
    required_source_kinds: Sequence[str] = DEFAULT_REQUIRED_CONCLUSION_SOURCE_KINDS,
    evaluation_scope: str = "full_oracle",
    alpha: float = DEFAULT_CONCLUSION_ALPHA,
    min_candidate_exact_match_rate: float = DEFAULT_MIN_CANDIDATE_EXACT_MATCH_RATE,
    min_candidate_exact_match_count: int = DEFAULT_MIN_CANDIDATE_EXACT_MATCH_COUNT,
    min_exact_match_rate_delta: float = DEFAULT_MIN_EXACT_MATCH_RATE_DELTA,
    min_graph_object_recall: float = DEFAULT_MIN_GRAPH_OBJECT_RECALL,
    min_observation_aware_case_count: int = DEFAULT_MIN_OBSERVATION_AWARE_CASE_COUNT,
    min_question_type_count: int = DEFAULT_MIN_CONCLUSION_QUESTION_TYPE_COUNT,
    max_unlocated_object_count: int = DEFAULT_MAX_UNLOCATED_OBJECT_COUNT,
) -> dict[str, Any]:
    required = _unique_strings(required_source_kinds)
    scope = _evaluation_scope(evaluation_scope)
    thresholds = {
        "alpha": _round_float(alpha),
        "min_candidate_exact_match_rate": _round_float(
            min_candidate_exact_match_rate
        ),
        "min_candidate_exact_match_count": min_candidate_exact_match_count,
        "min_exact_match_rate_delta": _round_float(min_exact_match_rate_delta),
        "min_graph_object_recall": _round_float(min_graph_object_recall),
        "min_observation_aware_case_count": min_observation_aware_case_count,
        "min_question_type_count": min_question_type_count,
        "max_unlocated_object_count": max_unlocated_object_count,
    }
    comparisons = _control_comparisons(
        offline_control_result_report,
        required_source_kinds=required,
        alpha=alpha,
        min_candidate_exact_match_rate=min_candidate_exact_match_rate,
        min_exact_match_rate_delta=min_exact_match_rate_delta,
    )
    graph_quality = _graph_quality(
        graph_eval_report,
        min_graph_object_recall,
        max_unlocated_object_count,
    )
    readiness_summary = {
        "real_experiment_ready": _ready(real_readiness_report),
        "offline_controls_ready": _ready(offline_control_result_report),
        "predicted_dsg_ready": _ready(predicted_dsg_evidence_report),
    }
    candidate_summary = _candidate_summary(
        comparisons,
        min_question_type_count=min_question_type_count,
    )
    aggregate = _aggregate_comparison(comparisons)
    scope_summary = _evaluation_scope_summary(
        qa_observability_report,
        evaluation_scope=scope,
        min_observation_aware_case_count=min_observation_aware_case_count,
    )
    conclusion = _conclusion(
        readiness_summary=readiness_summary,
        comparisons=comparisons,
        graph_quality=graph_quality,
        candidate_summary=candidate_summary,
        evaluation_scope_summary=scope_summary,
        required_source_kinds=required,
        min_candidate_exact_match_rate=min_candidate_exact_match_rate,
        min_candidate_exact_match_count=min_candidate_exact_match_count,
        min_question_type_count=min_question_type_count,
    )
    report: dict[str, Any] = {
        "schema_version": RESEARCH_CONCLUSION_REPORT_SCHEMA_VERSION,
        "hypothesis": "dynamic_scene_graph_superiority_over_vlm_video_memory_and_text_controls",
        "input_paths": {
            "error_attribution_report": _path_or_none(error_attribution_report_path),
            "graph_eval_report": _path_or_none(graph_eval_report_path),
            "offline_control_result_report": _path_or_none(
                offline_control_result_report_path
            ),
            "predicted_dsg_evidence_report": _path_or_none(
                predicted_dsg_evidence_report_path
            ),
            "qa_observability_report": _path_or_none(qa_observability_report_path),
            "real_readiness_report": _path_or_none(real_readiness_report_path),
        },
        "input_digests": {
            "error_attribution_report": _digest_or_none(error_attribution_report),
            "graph_eval_report": _digest_or_none(graph_eval_report),
            "offline_control_result_report": _digest_or_none(
                offline_control_result_report
            ),
            "predicted_dsg_evidence_report": _digest_or_none(
                predicted_dsg_evidence_report
            ),
            "qa_observability_report": _digest_or_none(qa_observability_report),
            "real_readiness_report": _digest_or_none(real_readiness_report),
        },
        "required_source_kinds": list(required),
        "thresholds": thresholds,
        "evaluation_scope": scope_summary,
        "readiness_summary": readiness_summary,
        "candidate_summary": candidate_summary,
        "aggregate_comparison": aggregate,
        "control_comparisons": comparisons,
        "graph_quality": graph_quality,
        "error_attribution_summary": _error_attribution_summary(
            error_attribution_report
        ),
        "conclusion": conclusion,
    }
    report["report_digest"] = research_conclusion_report_digest(report)
    return report


def research_conclusion_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def research_conclusion_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def research_conclusion_markdown(report: Mapping[str, Any]) -> str:
    conclusion = _mapping(report.get("conclusion"), "conclusion")
    candidate = _mapping(report.get("candidate_summary"), "candidate_summary")
    aggregate = _mapping(report.get("aggregate_comparison"), "aggregate_comparison")
    evaluation_scope = _mapping(report.get("evaluation_scope"), "evaluation_scope")
    graph_quality = _mapping(report.get("graph_quality"), "graph_quality")
    reasons = _mapping_sequence(conclusion.get("reasons"))
    verdict = _string_or_none(conclusion.get("verdict")) or "unknown"
    title = {
        "dsg_superior": "结论：当前实验支持 DSG 优于 VLM / 视频记忆",
        "dsg_not_superior": "结论：当前实验不支持 DSG 优于 VLM / 视频记忆",
        "inconclusive_not_ready": "结论：实验包未 ready，不能形成优越性结论",
    }.get(verdict, "结论：无法识别的实验结论")
    lines = [
        "# DSG-vs-Control 研究结论报告",
        "",
        title,
        "",
        "## 判定依据",
        "",
        f"- candidate exact match: {candidate.get('exact_match_count')}/"
        f"{candidate.get('case_count')} = {candidate.get('exact_match_rate')}",
        f"- required controls passed: {aggregate.get('passed_control_count')}/"
        f"{aggregate.get('control_count')}",
        f"- minimum exact-match delta: {aggregate.get('minimum_exact_match_rate_delta')}",
        f"- minimum sign-test p-value: {aggregate.get('minimum_sign_test_p_value')}",
        f"- predicted graph object recall: {graph_quality.get('object_recall')}",
        (
            "- predicted graph unlocated objects: "
            f"{graph_quality.get('unlocated_object_count')}"
        ),
        f"- evaluation scope: {evaluation_scope.get('name')}",
        f"- evidence-observable QA: "
        f"{evaluation_scope.get('evidence_observable_case_count')}/"
        f"{evaluation_scope.get('full_case_count')}",
        "",
        "## 主要原因",
        "",
    ]
    if not reasons:
        lines.append("- all_required_controls_passed")
    else:
        for reason in reasons:
            lines.append(f"- {reason.get('code')}: {reason.get('detail')}")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            (
                "该报告只对输入 artifact 所代表的当前实验包负责；"
                "它不会把小规模结果推广成通用结论。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def save_research_conclusion_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(research_conclusion_report_json(report), encoding="utf-8")
    return output_path


def save_research_conclusion_markdown(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(research_conclusion_markdown(report), encoding="utf-8")
    return output_path


def load_research_conclusion_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Research conclusion report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_research_conclusion_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    report_digest = _string_or_none(report.get("report_digest"))
    expected_digest = research_conclusion_report_digest(report)
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == RESEARCH_CONCLUSION_REPORT_SCHEMA_VERSION,
            "expected": RESEARCH_CONCLUSION_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_digest,
            "expected": expected_digest,
            "actual": report_digest,
        },
        {
            "name": "conclusion_shape",
            "passed": isinstance(report.get("conclusion"), Mapping),
        },
        {
            "name": "control_comparisons_shape",
            "passed": isinstance(report.get("control_comparisons"), Sequence)
            and not isinstance(report.get("control_comparisons"), str),
        },
        {
            "name": "thresholds_shape",
            "passed": isinstance(report.get("thresholds"), Mapping),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks,
    }


def compare_research_conclusion_report(report: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_research_conclusion_report(report)
    paths = _mapping(report.get("input_paths"), "input_paths")
    if not _has_required_compare_paths(paths):
        checks = [
            {
                "name": "saved_report_valid",
                "passed": validation["valid"] is True,
                "expected": True,
                "actual": validation["valid"],
            },
            {
                "name": "compare_paths_available",
                "passed": False,
                "actual": sorted(
                    key for key, value in paths.items() if isinstance(value, str)
                ),
            },
        ]
        return {
            "matches": validation["valid"] is True,
            "saved_digest": _string_or_none(report.get("report_digest")),
            "current_digest": _string_or_none(report.get("report_digest")),
            "validation": validation,
            "checks": checks,
        }
    current = research_conclusion_report(
        _load_json(_required_path(paths, "real_readiness_report")),
        _load_json(_required_path(paths, "offline_control_result_report")),
        predicted_dsg_evidence_report=_load_optional_json(
            paths.get("predicted_dsg_evidence_report")
        ),
        graph_eval_report=_load_optional_json(paths.get("graph_eval_report")),
        error_attribution_report=_load_optional_json(
            paths.get("error_attribution_report")
        ),
        qa_observability_report=_load_optional_json(
            paths.get("qa_observability_report")
        ),
        real_readiness_report_path=_required_path(paths, "real_readiness_report"),
        offline_control_result_report_path=_required_path(
            paths,
            "offline_control_result_report",
        ),
        predicted_dsg_evidence_report_path=_optional_path(
            paths.get("predicted_dsg_evidence_report")
        ),
        graph_eval_report_path=_optional_path(paths.get("graph_eval_report")),
        error_attribution_report_path=_optional_path(
            paths.get("error_attribution_report")
        ),
        qa_observability_report_path=_optional_path(
            paths.get("qa_observability_report")
        ),
        required_source_kinds=_string_list(report.get("required_source_kinds")),
        evaluation_scope=_evaluation_scope_from_report(report),
        alpha=_threshold(report, "alpha", DEFAULT_CONCLUSION_ALPHA),
        min_candidate_exact_match_rate=_threshold(
            report,
            "min_candidate_exact_match_rate",
            DEFAULT_MIN_CANDIDATE_EXACT_MATCH_RATE,
        ),
        min_candidate_exact_match_count=_threshold_int(
            report,
            "min_candidate_exact_match_count",
            DEFAULT_MIN_CANDIDATE_EXACT_MATCH_COUNT,
        ),
        min_exact_match_rate_delta=_threshold(
            report,
            "min_exact_match_rate_delta",
            DEFAULT_MIN_EXACT_MATCH_RATE_DELTA,
        ),
        min_graph_object_recall=_threshold(
            report,
            "min_graph_object_recall",
            DEFAULT_MIN_GRAPH_OBJECT_RECALL,
        ),
        min_observation_aware_case_count=_threshold_int(
            report,
            "min_observation_aware_case_count",
            DEFAULT_MIN_OBSERVATION_AWARE_CASE_COUNT,
        ),
        min_question_type_count=_threshold_int(
            report,
            "min_question_type_count",
            DEFAULT_MIN_CONCLUSION_QUESTION_TYPE_COUNT,
        ),
        max_unlocated_object_count=_threshold_int(
            report,
            "max_unlocated_object_count",
            DEFAULT_MAX_UNLOCATED_OBJECT_COUNT,
        ),
    )
    saved_digest = _string_or_none(report.get("report_digest"))
    current_digest = _string_or_none(current.get("report_digest"))
    checks = [
        {
            "name": "saved_report_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        _equality_check(
            "conclusion_matches_current",
            current.get("conclusion"),
            report.get("conclusion"),
        ),
        _equality_check(
            "control_comparisons_match_current",
            current.get("control_comparisons"),
            report.get("control_comparisons"),
        ),
        _equality_check(
            "evaluation_scope_matches_current",
            current.get("evaluation_scope"),
            report.get("evaluation_scope"),
        ),
        _equality_check("report_digest_matches_current", current_digest, saved_digest),
    ]
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def _control_comparisons(
    offline_control_result_report: Mapping[str, Any],
    *,
    required_source_kinds: Sequence[str],
    alpha: float,
    min_candidate_exact_match_rate: float,
    min_exact_match_rate_delta: float,
) -> list[dict[str, Any]]:
    rows = _mapping_sequence(offline_control_result_report.get("source_result_matrix"))
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        source_kind = _required_str(row, "source_kind")
        if source_kind not in required_source_kinds:
            continue
        delta_path = _required_path(row, "qa_eval_delta_report_path")
        delta_report = load_qa_eval_delta_report(delta_path)
        candidate_path = _required_path(delta_report, "candidate_report_path")
        baseline_path = _required_path(delta_report, "baseline_report_path")
        candidate_report = load_qa_eval_report(candidate_path)
        baseline_report = load_qa_eval_report(baseline_path)
        _require_valid_qa_report(candidate_report, candidate_path)
        _require_valid_qa_report(baseline_report, baseline_path)
        paired = _paired_exact_match_counts(candidate_report, baseline_report)
        question_type_counts = _question_type_counts(candidate_report)
        summary_delta = _mapping(delta_report.get("summary_delta"), "summary_delta")
        candidate_rate = _float_or_zero(
            summary_delta.get("candidate_exact_match_rate")
        )
        baseline_rate = _float_or_zero(summary_delta.get("baseline_exact_match_rate"))
        rate_delta = _float_or_zero(summary_delta.get("exact_match_rate_delta"))
        p_value = _sign_test_p_value(paired["wins"], paired["losses"])
        passes_rate_delta = rate_delta >= min_exact_match_rate_delta
        passes_candidate_floor = candidate_rate >= min_candidate_exact_match_rate
        passes_statistical_test = p_value <= alpha and paired["wins"] > paired["losses"]
        decision = (
            "candidate_superior"
            if (
                passes_rate_delta
                and passes_candidate_floor
                and passes_statistical_test
                and summary_delta.get("case_count_match") is True
            )
            else "candidate_not_superior"
        )
        comparisons.append(
            {
                "baseline_exact_match_count": _int_or_zero(
                    summary_delta.get("baseline_exact_match_count")
                ),
                "baseline_exact_match_rate": _round_float(baseline_rate),
                "baseline_name": _string_or_none(delta_report.get("baseline_name")),
                "baseline_gold_digest": _string_or_none(
                    baseline_report.get("gold_digest")
                ),
                "baseline_qa_eval_report_digest": qa_eval_report_digest(
                    baseline_report
                ),
                "baseline_qa_eval_report_path": str(baseline_path),
                "candidate_exact_match_count": _int_or_zero(
                    summary_delta.get("candidate_exact_match_count")
                ),
                "candidate_exact_match_rate": _round_float(candidate_rate),
                "candidate_gold_digest": _string_or_none(
                    candidate_report.get("gold_digest")
                ),
                "candidate_name": _string_or_none(delta_report.get("candidate_name")),
                "candidate_qa_eval_report_digest": qa_eval_report_digest(
                    candidate_report
                ),
                "candidate_qa_eval_report_path": str(candidate_path),
                "case_count": _int_or_zero(summary_delta.get("candidate_case_count")),
                "case_count_match": summary_delta.get("case_count_match") is True,
                "decision": decision,
                "discordant_count": paired["wins"] + paired["losses"],
                "exact_match_count_delta": _int_or_zero(
                    summary_delta.get("exact_match_count_delta")
                ),
                "exact_match_rate_delta": _round_float(rate_delta),
                "paired_losses": paired["losses"],
                "paired_ties": paired["ties"],
                "paired_wins": paired["wins"],
                "passes_candidate_floor": passes_candidate_floor,
                "passes_exact_match_rate_delta": passes_rate_delta,
                "passes_statistical_test": passes_statistical_test,
                "qa_eval_delta_report_digest": _string_or_none(
                    delta_report.get("report_digest")
                ),
                "qa_eval_delta_report_path": str(delta_path),
                "question_type_count": len(question_type_counts),
                "question_type_counts": question_type_counts,
                "sign_test_p_value": _round_float(p_value, digits=12),
                "source_key": _required_str(row, "source_key"),
                "source_kind": source_kind,
                "source_name": _required_str(row, "source_name"),
            }
        )
    return sorted(comparisons, key=lambda item: str(item["source_kind"]))


def _paired_exact_match_counts(
    candidate_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
) -> dict[str, int]:
    baseline_by_id = {
        _required_str(case, "case_id"): case
        for case in _mapping_sequence(baseline_report.get("cases"))
    }
    wins = 0
    losses = 0
    ties = 0
    for candidate_case in _mapping_sequence(candidate_report.get("cases")):
        case_id = _required_str(candidate_case, "case_id")
        baseline_case = baseline_by_id.get(case_id)
        if baseline_case is None:
            continue
        candidate_correct = candidate_case.get("exact_match") is True
        baseline_correct = baseline_case.get("exact_match") is True
        if candidate_correct and not baseline_correct:
            wins += 1
        elif baseline_correct and not candidate_correct:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties}


def _question_type_counts(report: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in _mapping_sequence(report.get("cases")):
        question_type = _string_or_none(row.get("question_type")) or "unknown"
        counts[question_type] = counts.get(question_type, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _sign_test_p_value(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant <= 0 or wins <= losses:
        return 1.0
    tail = sum(comb(discordant, count) for count in range(wins, discordant + 1))
    return float(tail) / float(2**discordant)


def _candidate_summary(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    min_question_type_count: int,
) -> dict[str, Any]:
    if not comparisons:
        return {
            "case_count": 0,
            "exact_match_count": 0,
            "exact_match_rate": 0.0,
            "min_question_type_count": min_question_type_count,
            "passes_question_type_floor": False,
            "question_type_count": 0,
            "question_type_counts": {},
        }
    first = comparisons[0]
    question_type_counts = _int_mapping(first.get("question_type_counts"))
    question_type_count = len(question_type_counts)
    return {
        "case_count": _int_or_zero(first.get("case_count")),
        "exact_match_count": _int_or_zero(first.get("candidate_exact_match_count")),
        "exact_match_rate": _round_float(
            _float_or_zero(first.get("candidate_exact_match_rate"))
        ),
        "min_question_type_count": min_question_type_count,
        "passes_question_type_floor": question_type_count >= min_question_type_count,
        "question_type_count": question_type_count,
        "question_type_counts": question_type_counts,
    }


def _aggregate_comparison(comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passed_count = sum(
        1 for row in comparisons if row.get("decision") == "candidate_superior"
    )
    deltas = [_float_or_zero(row.get("exact_match_rate_delta")) for row in comparisons]
    p_values = [_float_or_zero(row.get("sign_test_p_value")) for row in comparisons]
    return {
        "control_count": len(comparisons),
        "passed_control_count": passed_count,
        "failed_control_count": len(comparisons) - passed_count,
        "minimum_exact_match_rate_delta": (
            _round_float(min(deltas)) if deltas else None
        ),
        "minimum_sign_test_p_value": _round_float(min(p_values), digits=12)
        if p_values
        else None,
    }


def _graph_quality(
    graph_eval_report: Mapping[str, Any] | None,
    min_graph_object_recall: float,
    max_unlocated_object_count: int,
) -> dict[str, Any]:
    object_recall = _metric_rate(graph_eval_report, "object_recall")
    relation_f1 = _metric_rate(graph_eval_report, "relation_f1")
    unlocated_count = _metric_count(graph_eval_report, "unlocated_object_count")
    return {
        "max_unlocated_object_count": max_unlocated_object_count,
        "object_recall": object_recall,
        "passes_object_recall_floor": object_recall is not None
        and object_recall >= min_graph_object_recall,
        "passes_unlocated_object_floor": unlocated_count is not None
        and unlocated_count <= max_unlocated_object_count,
        "relation_f1": relation_f1,
        "unlocated_object_count": unlocated_count,
    }


def _error_attribution_summary(
    error_attribution_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if error_attribution_report is None:
        return {}
    summary = _mapping(error_attribution_report.get("summary"), "summary")
    return {
        "answer_correct_count": _int_or_zero(summary.get("answer_correct_count")),
        "by_error_category": _int_mapping(summary.get("by_error_category")),
        "by_evidence_error_category": _int_mapping(
            summary.get("by_evidence_error_category")
        ),
        "case_count": _int_or_zero(summary.get("case_count")),
        "error_count": _int_or_zero(summary.get("error_count")),
    }


def _conclusion(
    *,
    readiness_summary: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
    graph_quality: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
    evaluation_scope_summary: Mapping[str, Any],
    required_source_kinds: Sequence[str],
    min_candidate_exact_match_rate: float,
    min_candidate_exact_match_count: int,
    min_question_type_count: int,
) -> dict[str, Any]:
    not_ready_reasons = _not_ready_reasons(readiness_summary)
    not_ready_reasons.extend(
        _evaluation_scope_not_ready_reasons(
            evaluation_scope_summary,
            candidate_summary,
            comparisons,
        )
    )
    if not_ready_reasons:
        return {
            "claim": "The package is not ready, so DSG superiority cannot be claimed.",
            "dsg_superiority_claim_allowed": False,
            "ready_to_claim_not_superior": False,
            "reasons": not_ready_reasons,
            "verdict": "inconclusive_not_ready",
        }
    passed_controls = [
        row for row in comparisons if row.get("decision") == "candidate_superior"
    ]
    present_kinds = {_required_str(row, "source_kind") for row in comparisons}
    missing_kinds = [kind for kind in required_source_kinds if kind not in present_kinds]
    graph_passed = (
        graph_quality.get("passes_object_recall_floor") is True
        and graph_quality.get("passes_unlocated_object_floor") is True
    )
    candidate_count_passed = (
        _int_or_zero(candidate_summary.get("exact_match_count"))
        >= min_candidate_exact_match_count
    )
    candidate_question_type_passed = (
        candidate_summary.get("passes_question_type_floor") is True
    )
    all_controls_passed = (
        len(missing_kinds) == 0 and len(passed_controls) == len(required_source_kinds)
    )
    if (
        graph_passed
        and candidate_count_passed
        and candidate_question_type_passed
        and all_controls_passed
    ):
        return {
            "claim": (
                "Within this ready experiment package, DSG is superior to all "
                "required VLM/video-memory/text controls under the configured "
                "paired QA criteria."
            ),
            "dsg_superiority_claim_allowed": True,
            "ready_to_claim_not_superior": False,
            "reasons": [],
            "verdict": "dsg_superior",
        }
    reasons: list[dict[str, Any]] = []
    if _float_or_zero(candidate_summary.get("exact_match_rate")) < (
        min_candidate_exact_match_rate
    ):
        reasons.append(
            {
                "code": "candidate_exact_match_rate_below_floor",
                "detail": (
                    "Candidate exact-match rate is below the practical "
                    "superiority floor."
                ),
            }
        )
    if not candidate_count_passed:
        reasons.append(
            {
                "code": "candidate_exact_match_count_below_floor",
                "detail": (
                    "Candidate exact-match count is below the configured "
                    "observation-aware conclusion floor."
                ),
            }
        )
    if not candidate_question_type_passed:
        reasons.append(
            {
                "code": "question_type_coverage_below_floor",
                "detail": (
                    "Final DSG superiority claims require an expanded QA slice "
                    "with multiple question types: "
                    f"{candidate_summary.get('question_type_count')}/"
                    f"{min_question_type_count}."
                ),
            }
        )
    if not graph_passed:
        if graph_quality.get("passes_object_recall_floor") is not True:
            reasons.append(
                {
                    "code": "graph_object_recall_below_floor",
                    "detail": "Predicted DSG object recall is below the evidence floor.",
                }
            )
        if graph_quality.get("unlocated_object_count") is None:
            reasons.append(
                {
                    "code": "graph_unlocated_object_metric_missing",
                    "detail": "Graph eval report does not include unlocated object count.",
                }
            )
        elif graph_quality.get("passes_unlocated_object_floor") is not True:
            reasons.append(
                {
                    "code": "graph_unlocated_objects_present",
                    "detail": (
                        "Predicted DSG contains objects without queryable "
                        "location evidence."
                    ),
                }
            )
    if missing_kinds:
        reasons.append(
            {
                "code": "missing_required_control_kind",
                "detail": ",".join(missing_kinds),
            }
        )
    elif not all_controls_passed:
        if len(passed_controls) == 0:
            reasons.append(
                {
                    "code": "no_control_passed_superiority",
                    "detail": "No required control comparison passed all superiority gates.",
                }
            )
        else:
            reasons.append(
                {
                    "code": "not_all_required_controls_passed_superiority",
                    "detail": (
                        f"{len(passed_controls)}/{len(required_source_kinds)} "
                        "required controls passed."
                    ),
                }
            )
    return {
        "claim": (
            "Within this ready experiment package, DSG is not superior to the "
            "required VLM/video-memory/text controls under the configured "
            "paired QA and graph-evidence criteria."
        ),
        "dsg_superiority_claim_allowed": False,
        "ready_to_claim_not_superior": True,
        "reasons": reasons,
        "verdict": "dsg_not_superior",
    }


def _evaluation_scope(value: str) -> str:
    if value not in CONCLUSION_EVALUATION_SCOPES:
        raise SpatialQAError(
            "Unsupported research conclusion evaluation scope: "
            f"{value}. Expected one of: {', '.join(CONCLUSION_EVALUATION_SCOPES)}"
        )
    return value


def _evaluation_scope_summary(
    qa_observability_report: Mapping[str, Any] | None,
    *,
    evaluation_scope: str,
    min_observation_aware_case_count: int,
) -> dict[str, Any]:
    if min_observation_aware_case_count < 0:
        raise SpatialQAError("min_observation_aware_case_count must be non-negative")
    summary = (
        _mapping(qa_observability_report.get("summary"), "qa_observability.summary")
        if qa_observability_report is not None
        else {}
    )
    full_count = _int_or_zero(summary.get("case_count"))
    evidence_count = _int_or_zero(summary.get("evidence_observable_count"))
    target_count = _int_or_zero(summary.get("target_observable_count"))
    missing_count = _int_or_zero(summary.get("missing_evidence_count"))
    observation_aware = evaluation_scope == "observation_aware"
    passes_floor = (
        not observation_aware
        or (
            qa_observability_report is not None
            and evidence_count >= min_observation_aware_case_count
        )
    )
    return {
        "name": evaluation_scope,
        "qa_observability_report_available": qa_observability_report is not None,
        "full_case_count": full_count,
        "evidence_observable_qa_digest": _evidence_observable_qa_digest(
            qa_observability_report
        ),
        "evidence_observable_case_count": evidence_count,
        "evidence_observable_rate": _round_float(
            evidence_count / full_count if full_count else 0.0
        ),
        "target_observable_case_count": target_count,
        "missing_evidence_case_count": missing_count,
        "min_observation_aware_case_count": min_observation_aware_case_count,
        "passes_observation_aware_case_floor": passes_floor,
    }


def _evaluation_scope_not_ready_reasons(
    evaluation_scope_summary: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if evaluation_scope_summary.get("name") != "observation_aware":
        return []
    if evaluation_scope_summary.get("qa_observability_report_available") is not True:
        return [
            {
                "code": "observation_aware_missing_qa_observability_report",
                "detail": (
                    "Observation-aware conclusion requires a QA observability report."
                ),
            }
        ]
    if evaluation_scope_summary.get("passes_observation_aware_case_floor") is not True:
        return [
            {
                "code": "observation_aware_case_count_below_floor",
                "detail": (
                    "Evidence-observable QA count is below the configured floor: "
                    f"{evaluation_scope_summary.get('evidence_observable_case_count')}/"
                    f"{evaluation_scope_summary.get('min_observation_aware_case_count')}"
                ),
            }
        ]
    evidence_count = _int_or_zero(
        evaluation_scope_summary.get("evidence_observable_case_count")
    )
    candidate_count = _int_or_zero(candidate_summary.get("case_count"))
    if candidate_count != evidence_count:
        return [
            {
                "code": "observation_aware_case_count_mismatch",
                "detail": (
                    "Observation-aware conclusion requires QA evals to be run "
                    "on the evidence-observable QA slice: "
                    f"candidate={candidate_count}, evidence_observable={evidence_count}"
                ),
            }
        ]
    expected_digest = _string_or_none(
        evaluation_scope_summary.get("evidence_observable_qa_digest")
    )
    if expected_digest is None:
        return [
            {
                "code": "observation_aware_missing_evidence_observable_qa_digest",
                "detail": (
                    "Observation-aware conclusion requires a QA observability "
                    "report with split_qa_digests.evidence_observable."
                ),
            }
        ]
    mismatched = [
        {
            "source_kind": _string_or_none(row.get("source_kind")),
            "candidate_qa_eval_report_path": _string_or_none(
                row.get("candidate_qa_eval_report_path")
            ),
            "baseline_qa_eval_report_path": _string_or_none(
                row.get("baseline_qa_eval_report_path")
            ),
            "candidate_gold_digest": _string_or_none(
                row.get("candidate_gold_digest")
            ),
            "baseline_gold_digest": _string_or_none(row.get("baseline_gold_digest")),
        }
        for row in comparisons
        if row.get("candidate_gold_digest") != expected_digest
        or row.get("baseline_gold_digest") != expected_digest
    ]
    if mismatched:
        return [
            {
                "code": "observation_aware_qa_digest_mismatch",
                "detail": (
                    "Observation-aware conclusion requires every candidate and "
                    "control QA eval to use the evidence-observable QA digest."
                ),
                "expected_gold_digest": expected_digest,
                "mismatched_reports": mismatched,
            }
        ]
    return []


def _evidence_observable_qa_digest(
    qa_observability_report: Mapping[str, Any] | None,
) -> str | None:
    if qa_observability_report is None:
        return None
    split_digests = qa_observability_report.get("split_qa_digests")
    if not isinstance(split_digests, Mapping):
        return None
    return _string_or_none(split_digests.get("evidence_observable"))


def _not_ready_reasons(readiness_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    mapping = (
        ("real_experiment_ready", "real_experiment_readiness_not_ready"),
        ("offline_controls_ready", "offline_controls_not_ready"),
        ("predicted_dsg_ready", "predicted_dsg_evidence_not_ready"),
    )
    return [
        {"code": reason, "detail": key}
        for key, reason in mapping
        if readiness_summary.get(key) is not True
    ]


def _require_valid_qa_report(report: Mapping[str, Any], path: Path) -> None:
    validation = validate_qa_eval_report(report)
    if validation["valid"] is not True:
        raise SpatialQAError(f"Invalid QA eval report for conclusion: {path}")


def _ready(report: Mapping[str, Any] | None) -> bool:
    if report is None:
        return False
    readiness = report.get("readiness")
    return isinstance(readiness, Mapping) and readiness.get("ready") is True


def _digest_or_none(report: Mapping[str, Any] | None) -> str | None:
    if report is None:
        return None
    return _string_or_none(report.get("report_digest"))


def _metric_rate(report: Mapping[str, Any] | None, metric_name: str) -> float | None:
    if report is None:
        return None
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    metric = metrics.get(metric_name)
    if not isinstance(metric, Mapping):
        return None
    value = metric.get("rate")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return _round_float(float(value))


def _metric_count(report: Mapping[str, Any] | None, metric_name: str) -> int | None:
    if report is None:
        return None
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    metric = metrics.get(metric_name)
    if not isinstance(metric, Mapping):
        return None
    value = metric.get("count")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _has_required_compare_paths(paths: Mapping[str, Any]) -> bool:
    return (
        _string_or_none(paths.get("real_readiness_report")) is not None
        and _string_or_none(paths.get("offline_control_result_report")) is not None
    )


def _threshold(report: Mapping[str, Any], name: str, default: float) -> float:
    thresholds = report.get("thresholds")
    if isinstance(thresholds, Mapping):
        value = thresholds.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return default


def _threshold_int(report: Mapping[str, Any], name: str, default: int) -> int:
    thresholds = report.get("thresholds")
    if isinstance(thresholds, Mapping):
        value = thresholds.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return default


def _evaluation_scope_from_report(report: Mapping[str, Any]) -> str:
    scope = report.get("evaluation_scope")
    if isinstance(scope, Mapping):
        value = scope.get("name")
        if isinstance(value, str):
            return _evaluation_scope(value)
    return "full_oracle"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(f"Conclusion input must be a JSON object: {path}")
    return cast(dict[str, Any], payload)


def _load_optional_json(value: object) -> dict[str, Any] | None:
    path = _optional_path(value)
    return _load_json(path) if path is not None else None


def _required_path(payload: Mapping[str, Any], key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Research conclusion path is required: {key}")
    return Path(value)


def _optional_path(value: object) -> Path | None:
    if isinstance(value, str) and value != "":
        return Path(value)
    return None


def _path_or_none(value: str | Path | None) -> str | None:
    return str(value) if value is not None else None


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"Research conclusion field must be an object: {field}")
    return cast(Mapping[str, Any], value)


def _mapping_sequence(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    return tuple(cast(Mapping[str, Any], item) for item in value if isinstance(item, Mapping))


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Research conclusion field is required: {key}")
    return value


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return list(DEFAULT_REQUIRED_CONCLUSION_SOURCE_KINDS)
    result = [item for item in value if isinstance(item, str) and item != ""]
    return list(_unique_strings(result))


def _unique_strings(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({value for value in values if value != ""}))


def _int_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item
        for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        if isinstance(item, int) and not isinstance(item, bool)
    }


def _int_or_zero(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _float_or_zero(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _round_float(value: float, *, digits: int = 6) -> float:
    return round(float(value), digits)


def _equality_check(name: str, expected: object, actual: object) -> dict[str, Any]:
    return {"name": name, "passed": expected == actual, "expected": expected, "actual": actual}
