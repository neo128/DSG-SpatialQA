from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.agents.active_graph_agent import ActiveGraphAgent, ActiveTaskResult
from dsg_spatialqa_lab.scene_io import load_graph_json
from dsg_spatialqa_lab.schema import SpatialQAError
from dsg_spatialqa_lab.tasks import (
    ActiveEQATask,
    MockActiveEnvironment,
    active_eqa_tasks_digest,
    active_eqa_task_to_dict,
    load_active_eqa_tasks,
)


ACTIVE_TASK_RESULT_SCHEMA_VERSION = "dsg-spatialqa-lab.active-task-result.v1"
ACTIVE_TASK_REPORT_SCHEMA_VERSION = "dsg-spatialqa-lab.active-task-report.v1"
ACTIVE_TASK_DELTA_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.active-task-delta-report.v1"
)


def active_task_result_to_dict(result: ActiveTaskResult) -> dict[str, Any]:
    return {
        "schema_version": ACTIVE_TASK_RESULT_SCHEMA_VERSION,
        "task_id": result.task_id,
        "policy": result.policy,
        "answer": _json_mapping(result.answer),
        "success": result.success,
        "action_count": result.action_count,
        "evidence_nodes": list(result.evidence_nodes),
        "evidence_edges": list(result.evidence_edges),
        "final_step": result.final_step,
        "confidence": result.confidence,
        "needs_reobserve": result.needs_reobserve,
        "error": result.error,
        "transcript": [_json_mapping(item) for item in result.transcript],
        "action_evidence_snapshots": [
            _json_mapping(item) for item in result.action_evidence_snapshots
        ],
    }


def active_task_result_from_dict(payload: Mapping[str, Any]) -> ActiveTaskResult:
    schema_version = _required_str(payload, "schema_version")
    if schema_version != ACTIVE_TASK_RESULT_SCHEMA_VERSION:
        raise SpatialQAError(f"Unsupported active task result schema version: {schema_version}")
    return ActiveTaskResult(
        task_id=_required_str(payload, "task_id"),
        policy=_required_str(payload, "policy"),
        answer=_required_mapping(payload, "answer"),
        success=_required_bool(payload, "success"),
        action_count=_required_int(payload, "action_count"),
        evidence_nodes=_string_tuple(payload, "evidence_nodes"),
        evidence_edges=_string_tuple(payload, "evidence_edges"),
        final_step=_required_int(payload, "final_step"),
        confidence=_required_float(payload, "confidence"),
        needs_reobserve=_required_bool(payload, "needs_reobserve"),
        error=_optional_str(payload, "error"),
        transcript=tuple(_mapping_tuple(payload, "transcript")),
        action_evidence_snapshots=tuple(
            _mapping_tuple(payload, "action_evidence_snapshots")
        ),
    )


def active_task_results_jsonl(results: Sequence[ActiveTaskResult]) -> str:
    return "".join(
        json.dumps(active_task_result_to_dict(result), separators=(",", ":"), sort_keys=True)
        + "\n"
        for result in results
    )


def active_task_results_digest(results: Sequence[ActiveTaskResult]) -> str:
    return hashlib.sha256(active_task_results_jsonl(results).encode("utf-8")).hexdigest()


def active_task_report(
    tasks: Sequence[ActiveEQATask],
    results: Sequence[ActiveTaskResult],
    *,
    task_path: str | Path | None = None,
    graph_path: str | Path | None = None,
    policy: str | None = None,
) -> dict[str, Any]:
    task_by_id = {task.id: task for task in tasks}
    case_results = [_case_result(task_by_id.get(result.task_id), result) for result in results]
    report: dict[str, Any] = {
        "schema_version": ACTIVE_TASK_REPORT_SCHEMA_VERSION,
        "task_path": str(task_path) if task_path is not None else None,
        "graph_path": str(graph_path) if graph_path is not None else None,
        "policy": policy,
        "task_digest": active_eqa_tasks_digest(tasks),
        "result_digest": active_task_results_digest(results),
        "summary": _summary(results),
        "metrics": _metrics(case_results),
        "budget_analysis": _budget_analysis(case_results),
        "tasks": [active_eqa_task_to_dict(task) for task in tasks],
        "results": [active_task_result_to_dict(result) for result in results],
        "cases": case_results,
    }
    report["report_digest"] = active_task_report_digest(report)
    return report


def active_task_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def active_task_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_active_task_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(active_task_report_json(report), encoding="utf-8")
    return output_path


def load_active_task_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Active task report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_active_task_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    report_digest = _string_or_none(report.get("report_digest"))
    expected_report_digest = active_task_report_digest(report)
    summary = report.get("summary")
    results = report.get("results")
    case_results = report.get("cases")
    case_items = (
        cast(Sequence[Mapping[str, Any]], case_results)
        if isinstance(case_results, Sequence) and not isinstance(case_results, str)
        else ()
    )
    expected_budget_analysis = _budget_analysis(case_items)
    budget_analysis = report.get("budget_analysis")
    budget_curve = (
        budget_analysis.get("budget_curve")
        if isinstance(budget_analysis, Mapping)
        else None
    )
    by_max_actions = (
        budget_analysis.get("by_max_actions")
        if isinstance(budget_analysis, Mapping)
        else None
    )
    budget_curve_count = (
        len(budget_curve)
        if isinstance(budget_curve, Sequence) and not isinstance(budget_curve, str)
        else None
    )
    by_max_actions_count = len(by_max_actions) if isinstance(by_max_actions, Mapping) else None
    result_count = len(results) if isinstance(results, Sequence) and not isinstance(results, str) else None
    success_count = _case_bool_count(case_items, "task_success")
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == ACTIVE_TASK_REPORT_SCHEMA_VERSION,
            "expected": ACTIVE_TASK_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_report_digest,
            "expected": expected_report_digest,
            "actual": report_digest,
        },
        {
            "name": "task_count",
            "passed": _summary_value(summary, "task_count") == result_count,
            "expected": result_count,
            "actual": _summary_value(summary, "task_count"),
        },
        {
            "name": "success_count",
            "passed": _summary_value(summary, "success_count") == success_count,
            "expected": success_count,
            "actual": _summary_value(summary, "success_count"),
        },
        {
            "name": "budget_analysis",
            "passed": budget_analysis == expected_budget_analysis,
            "expected": expected_budget_analysis,
            "actual": budget_analysis,
        },
        {
            "name": "budget_curve_count",
            "passed": budget_curve_count == by_max_actions_count,
            "expected": by_max_actions_count,
            "actual": budget_curve_count,
        },
        {
            "name": "budget_curve_keys",
            "passed": _budget_curve_keys(budget_curve) == _budget_mapping_keys(by_max_actions),
            "expected": _budget_mapping_keys(by_max_actions),
            "actual": _budget_curve_keys(budget_curve),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks,
    }


def compare_active_task_report(report: Mapping[str, Any]) -> dict[str, Any]:
    task_path = _required_report_path(report, "task_path")
    graph_path = _required_report_path(report, "graph_path")
    policy = _required_report_policy(report)
    tasks = load_active_eqa_tasks(task_path)
    graph = load_graph_json(graph_path)
    agent = ActiveGraphAgent(policy=policy)
    results = [
        agent.run(task, MockActiveEnvironment({task.initial_step: graph}))
        for task in tasks
    ]
    current_report = active_task_report(
        tasks,
        results,
        task_path=task_path,
        graph_path=graph_path,
        policy=policy,
    )
    validation = validate_active_task_report(report)
    saved_digest = _string_or_none(report.get("report_digest"))
    current_digest = _string_or_none(current_report.get("report_digest"))
    checks = [
        {
            "name": "report_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "report_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        _equality_check("summary_matches_current", report.get("summary"), current_report["summary"]),
        _equality_check("metrics_match_current", report.get("metrics"), current_report["metrics"]),
        _equality_check("budget_analysis_matches_current", report.get("budget_analysis"), current_report["budget_analysis"]),
        _equality_check("tasks_match_current", report.get("tasks"), current_report["tasks"]),
        _equality_check("results_match_current", report.get("results"), current_report["results"]),
        _equality_check("cases_match_current", report.get("cases"), current_report["cases"]),
    ]
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def active_task_delta_report(
    candidate_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    *,
    candidate_name: str = "candidate",
    baseline_name: str = "baseline",
    candidate_report_path: str | Path | None = None,
    baseline_report_path: str | Path | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": ACTIVE_TASK_DELTA_REPORT_SCHEMA_VERSION,
        "candidate_name": candidate_name,
        "baseline_name": baseline_name,
        "candidate_report_path": (
            str(candidate_report_path) if candidate_report_path is not None else None
        ),
        "baseline_report_path": (
            str(baseline_report_path) if baseline_report_path is not None else None
        ),
        "candidate_report_digest": _string_or_none(candidate_report.get("report_digest")),
        "baseline_report_digest": _string_or_none(baseline_report.get("report_digest")),
        "summary_delta": _active_summary_delta(
            candidate_report.get("summary"),
            baseline_report.get("summary"),
        ),
        "metrics_delta": _active_metrics_delta(
            candidate_report.get("metrics"),
            baseline_report.get("metrics"),
        ),
        "budget_delta": _active_budget_delta(
            candidate_report.get("budget_analysis"),
            baseline_report.get("budget_analysis"),
        ),
    }
    report["report_digest"] = active_task_delta_report_digest(report)
    return report


def active_task_delta_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def active_task_delta_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_active_task_delta_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(active_task_delta_report_json(report), encoding="utf-8")
    return output_path


def load_active_task_delta_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Active task delta report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_active_task_delta_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    report_digest = _string_or_none(report.get("report_digest"))
    expected_report_digest = active_task_delta_report_digest(report)
    summary_delta = report.get("summary_delta")
    metrics_delta = report.get("metrics_delta")
    budget_delta = report.get("budget_delta")
    expected_summary_delta = _active_summary_delta_from_entry(summary_delta)
    expected_metrics_delta = _active_metrics_delta_from_entry(metrics_delta)
    expected_budget_delta = _active_budget_delta_from_entry(budget_delta)
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == ACTIVE_TASK_DELTA_REPORT_SCHEMA_VERSION,
            "expected": ACTIVE_TASK_DELTA_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_report_digest,
            "expected": expected_report_digest,
            "actual": report_digest,
        },
        {
            "name": "summary_delta",
            "passed": summary_delta == expected_summary_delta,
            "expected": expected_summary_delta,
            "actual": summary_delta,
        },
        {
            "name": "metrics_delta",
            "passed": metrics_delta == expected_metrics_delta,
            "expected": expected_metrics_delta,
            "actual": metrics_delta,
        },
        {
            "name": "budget_delta",
            "passed": budget_delta == expected_budget_delta,
            "expected": expected_budget_delta,
            "actual": budget_delta,
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks,
    }


def compare_active_task_delta_report(report: Mapping[str, Any]) -> dict[str, Any]:
    candidate_report_path = _required_delta_report_path(report, "candidate_report_path")
    baseline_report_path = _required_delta_report_path(report, "baseline_report_path")
    current_report = active_task_delta_report(
        load_active_task_report(candidate_report_path),
        load_active_task_report(baseline_report_path),
        candidate_name=_delta_report_name(report, "candidate_name"),
        baseline_name=_delta_report_name(report, "baseline_name"),
        candidate_report_path=candidate_report_path,
        baseline_report_path=baseline_report_path,
    )
    validation = validate_active_task_delta_report(report)
    saved_digest = _string_or_none(report.get("report_digest"))
    current_digest = _string_or_none(current_report.get("report_digest"))
    checks = [
        {
            "name": "report_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "report_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        _equality_check(
            "summary_delta_matches_current",
            report.get("summary_delta"),
            current_report["summary_delta"],
        ),
        _equality_check(
            "metrics_delta_matches_current",
            report.get("metrics_delta"),
            current_report["metrics_delta"],
        ),
        _equality_check(
            "budget_delta_matches_current",
            report.get("budget_delta"),
            current_report["budget_delta"],
        ),
    ]
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def _case_result(task: ActiveEQATask | None, result: ActiveTaskResult) -> dict[str, Any]:
    required = task.required_evidence if task is not None else {}
    answer_accuracy = (
        result.error is None and task is not None and result.answer == task.gold_answer
    )
    evidence_coverage = _evidence_coverage(result.evidence_nodes, result.evidence_edges, required)
    return {
        "task_id": result.task_id,
        "policy": result.policy,
        "max_actions": task.max_actions if task is not None else None,
        "task_success": result.success,
        "answer_accuracy": answer_accuracy,
        "answer_graph_consistent": answer_accuracy and evidence_coverage == 1.0,
        "action_count": result.action_count,
        "evidence_coverage": evidence_coverage,
        "final_step": result.final_step,
        "confidence": result.confidence,
        "error": result.error,
        "action_evidence_snapshots": [
            _json_mapping(item) for item in result.action_evidence_snapshots
        ],
    }


def _budget_analysis(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[int, list[Mapping[str, Any]]] = {}
    for result in case_results:
        budget = result.get("max_actions")
        if isinstance(budget, bool) or not isinstance(budget, int):
            continue
        buckets.setdefault(budget, []).append(result)
    by_max_actions = {
        str(budget): _budget_bucket(buckets[budget])
        for budget in sorted(buckets)
    }
    return {
        "budget_curve": [
            {"max_actions": budget, **by_max_actions[str(budget)]}
            for budget in sorted(buckets)
        ],
        "by_max_actions": by_max_actions,
    }


def _budget_bucket(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    task_count = len(case_results)
    success_count = _case_bool_count(case_results, "task_success")
    return {
        "average_action_count": _average_float(
            result.get("action_count", 0.0) for result in case_results
        ),
        "average_evidence_coverage": _average_float(
            result.get("evidence_coverage", 0.0) for result in case_results
        ),
        "success_count": success_count,
        "success_rate": _rate(success_count, task_count),
        "task_count": task_count,
    }


def _summary(results: Sequence[ActiveTaskResult]) -> dict[str, Any]:
    success_count = sum(1 for result in results if result.success)
    by_policy: dict[str, int] = {}
    for result in results:
        by_policy[result.policy] = by_policy.get(result.policy, 0) + 1
    return {
        "task_count": len(results),
        "success_count": success_count,
        "failure_count": len(results) - success_count,
        "total_action_count": sum(result.action_count for result in results),
        "by_policy": {key: by_policy[key] for key in sorted(by_policy)},
    }


def _metrics(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    case_count = len(case_results)
    task_success_count = _case_bool_count(case_results, "task_success")
    answer_accuracy_count = _case_bool_count(case_results, "answer_accuracy")
    consistent_count = _case_bool_count(case_results, "answer_graph_consistent")
    return {
        "answer_accuracy": {
            "count": answer_accuracy_count,
            "rate": _rate(answer_accuracy_count, case_count),
            "total": case_count,
        },
        "answer_graph_consistency": {
            "count": consistent_count,
            "rate": _rate(consistent_count, case_count),
            "total": case_count,
        },
        "evidence_coverage": {
            "average": _average_float(result["evidence_coverage"] for result in case_results),
            "total": case_count,
        },
        "task_success": {
            "count": task_success_count,
            "rate": _rate(task_success_count, case_count),
            "total": case_count,
        },
        "action_count": {
            "average": _average_float(result["action_count"] for result in case_results),
            "total": case_count,
        },
    }


def _active_summary_delta(candidate_summary: object, baseline_summary: object) -> dict[str, Any]:
    candidate_task_count = _int_mapping_value(candidate_summary, "task_count")
    baseline_task_count = _int_mapping_value(baseline_summary, "task_count")
    candidate_success_count = _int_mapping_value(candidate_summary, "success_count")
    baseline_success_count = _int_mapping_value(baseline_summary, "success_count")
    candidate_failure_count = _int_mapping_value(candidate_summary, "failure_count")
    baseline_failure_count = _int_mapping_value(baseline_summary, "failure_count")
    candidate_total_action_count = _int_mapping_value(
        candidate_summary,
        "total_action_count",
    )
    baseline_total_action_count = _int_mapping_value(
        baseline_summary,
        "total_action_count",
    )
    return {
        "baseline_failure_count": baseline_failure_count,
        "baseline_success_count": baseline_success_count,
        "baseline_task_count": baseline_task_count,
        "baseline_total_action_count": baseline_total_action_count,
        "candidate_failure_count": candidate_failure_count,
        "candidate_success_count": candidate_success_count,
        "candidate_task_count": candidate_task_count,
        "candidate_total_action_count": candidate_total_action_count,
        "failure_count_delta": _int_delta(
            candidate_failure_count,
            baseline_failure_count,
        ),
        "success_count_delta": _int_delta(
            candidate_success_count,
            baseline_success_count,
        ),
        "task_count_delta": _int_delta(candidate_task_count, baseline_task_count),
        "task_count_match": candidate_task_count == baseline_task_count,
        "total_action_count_delta": _int_delta(
            candidate_total_action_count,
            baseline_total_action_count,
        ),
    }


def _active_summary_delta_from_entry(entry: object) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    return {
        "baseline_failure_count": _int_mapping_value(entry, "baseline_failure_count"),
        "baseline_success_count": _int_mapping_value(entry, "baseline_success_count"),
        "baseline_task_count": _int_mapping_value(entry, "baseline_task_count"),
        "baseline_total_action_count": _int_mapping_value(
            entry,
            "baseline_total_action_count",
        ),
        "candidate_failure_count": _int_mapping_value(entry, "candidate_failure_count"),
        "candidate_success_count": _int_mapping_value(entry, "candidate_success_count"),
        "candidate_task_count": _int_mapping_value(entry, "candidate_task_count"),
        "candidate_total_action_count": _int_mapping_value(
            entry,
            "candidate_total_action_count",
        ),
        "failure_count_delta": _int_delta(
            _int_mapping_value(entry, "candidate_failure_count"),
            _int_mapping_value(entry, "baseline_failure_count"),
        ),
        "success_count_delta": _int_delta(
            _int_mapping_value(entry, "candidate_success_count"),
            _int_mapping_value(entry, "baseline_success_count"),
        ),
        "task_count_delta": _int_delta(
            _int_mapping_value(entry, "candidate_task_count"),
            _int_mapping_value(entry, "baseline_task_count"),
        ),
        "task_count_match": (
            _int_mapping_value(entry, "candidate_task_count")
            == _int_mapping_value(entry, "baseline_task_count")
        ),
        "total_action_count_delta": _int_delta(
            _int_mapping_value(entry, "candidate_total_action_count"),
            _int_mapping_value(entry, "baseline_total_action_count"),
        ),
    }


def _active_metrics_delta(candidate_metrics: object, baseline_metrics: object) -> dict[str, Any]:
    return {
        "action_count": _average_metric_delta(
            candidate_metrics,
            baseline_metrics,
            "action_count",
        ),
        "answer_accuracy": _count_rate_metric_delta(
            candidate_metrics,
            baseline_metrics,
            "answer_accuracy",
        ),
        "answer_graph_consistency": _count_rate_metric_delta(
            candidate_metrics,
            baseline_metrics,
            "answer_graph_consistency",
        ),
        "evidence_coverage": _average_metric_delta(
            candidate_metrics,
            baseline_metrics,
            "evidence_coverage",
        ),
        "task_success": _count_rate_metric_delta(
            candidate_metrics,
            baseline_metrics,
            "task_success",
        ),
    }


def _active_metrics_delta_from_entry(entry: object) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    return {
        "action_count": _average_metric_delta_from_entry(entry.get("action_count")),
        "answer_accuracy": _count_rate_metric_delta_from_entry(
            entry.get("answer_accuracy")
        ),
        "answer_graph_consistency": _count_rate_metric_delta_from_entry(
            entry.get("answer_graph_consistency")
        ),
        "evidence_coverage": _average_metric_delta_from_entry(
            entry.get("evidence_coverage")
        ),
        "task_success": _count_rate_metric_delta_from_entry(entry.get("task_success")),
    }


def _count_rate_metric_delta(
    candidate_metrics: object,
    baseline_metrics: object,
    metric_name: str,
) -> dict[str, Any]:
    candidate_count = _metric_int_value(candidate_metrics, metric_name, "count")
    baseline_count = _metric_int_value(baseline_metrics, metric_name, "count")
    candidate_rate = _metric_float_value(candidate_metrics, metric_name, "rate")
    baseline_rate = _metric_float_value(baseline_metrics, metric_name, "rate")
    candidate_total = _metric_int_value(candidate_metrics, metric_name, "total")
    baseline_total = _metric_int_value(baseline_metrics, metric_name, "total")
    return {
        "baseline_count": baseline_count,
        "baseline_rate": baseline_rate,
        "candidate_count": candidate_count,
        "candidate_rate": candidate_rate,
        "count_delta": _int_delta(candidate_count, baseline_count),
        "rate_delta": _float_delta(candidate_rate, baseline_rate),
        "total_match": candidate_total == baseline_total,
    }


def _count_rate_metric_delta_from_entry(entry: object) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    return {
        "baseline_count": _int_mapping_value(entry, "baseline_count"),
        "baseline_rate": _float_mapping_value(entry, "baseline_rate"),
        "candidate_count": _int_mapping_value(entry, "candidate_count"),
        "candidate_rate": _float_mapping_value(entry, "candidate_rate"),
        "count_delta": _int_delta(
            _int_mapping_value(entry, "candidate_count"),
            _int_mapping_value(entry, "baseline_count"),
        ),
        "rate_delta": _float_delta(
            _float_mapping_value(entry, "candidate_rate"),
            _float_mapping_value(entry, "baseline_rate"),
        ),
        "total_match": _bool_mapping_value(entry, "total_match"),
    }


def _average_metric_delta(
    candidate_metrics: object,
    baseline_metrics: object,
    metric_name: str,
) -> dict[str, Any]:
    candidate_average = _metric_float_value(candidate_metrics, metric_name, "average")
    baseline_average = _metric_float_value(baseline_metrics, metric_name, "average")
    candidate_total = _metric_int_value(candidate_metrics, metric_name, "total")
    baseline_total = _metric_int_value(baseline_metrics, metric_name, "total")
    return {
        "average_delta": _float_delta(candidate_average, baseline_average),
        "baseline_average": baseline_average,
        "candidate_average": candidate_average,
        "total_match": candidate_total == baseline_total,
    }


def _average_metric_delta_from_entry(entry: object) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    return {
        "average_delta": _float_delta(
            _float_mapping_value(entry, "candidate_average"),
            _float_mapping_value(entry, "baseline_average"),
        ),
        "baseline_average": _float_mapping_value(entry, "baseline_average"),
        "candidate_average": _float_mapping_value(entry, "candidate_average"),
        "total_match": _bool_mapping_value(entry, "total_match"),
    }


def _active_budget_delta(
    candidate_budget: object,
    baseline_budget: object,
) -> dict[str, Any]:
    candidate_by_max_actions = _budget_analysis_mapping(candidate_budget)
    baseline_by_max_actions = _budget_analysis_mapping(baseline_budget)
    by_max_actions = {
        budget: _budget_bucket_delta(
            candidate_by_max_actions.get(budget),
            baseline_by_max_actions.get(budget),
        )
        for budget in _sorted_budget_keys(candidate_by_max_actions, baseline_by_max_actions)
    }
    return {
        "budget_curve": [
            {"max_actions": int(budget), **by_max_actions[budget]}
            for budget in _sorted_budget_keys(by_max_actions)
        ],
        "by_max_actions": by_max_actions,
    }


def _active_budget_delta_from_entry(entry: object) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    by_max_actions = entry.get("by_max_actions")
    if not isinstance(by_max_actions, Mapping):
        return None
    expected_by_max_actions = {
        str(budget): _budget_bucket_delta_from_entry(by_max_actions.get(str(budget)))
        for budget in _sorted_budget_keys(by_max_actions)
    }
    return {
        "budget_curve": [
            {"max_actions": int(budget), **expected_by_max_actions[budget]}
            for budget in _sorted_budget_keys(expected_by_max_actions)
        ],
        "by_max_actions": expected_by_max_actions,
    }


def _budget_bucket_delta(
    candidate_bucket: object,
    baseline_bucket: object,
) -> dict[str, Any]:
    candidate_average_action_count = _float_mapping_value(
        candidate_bucket,
        "average_action_count",
    )
    baseline_average_action_count = _float_mapping_value(
        baseline_bucket,
        "average_action_count",
    )
    candidate_average_evidence_coverage = _float_mapping_value(
        candidate_bucket,
        "average_evidence_coverage",
    )
    baseline_average_evidence_coverage = _float_mapping_value(
        baseline_bucket,
        "average_evidence_coverage",
    )
    candidate_success_count = _int_mapping_value(candidate_bucket, "success_count")
    baseline_success_count = _int_mapping_value(baseline_bucket, "success_count")
    candidate_success_rate = _float_mapping_value(candidate_bucket, "success_rate")
    baseline_success_rate = _float_mapping_value(baseline_bucket, "success_rate")
    candidate_task_count = _int_mapping_value(candidate_bucket, "task_count")
    baseline_task_count = _int_mapping_value(baseline_bucket, "task_count")
    return {
        "average_action_count_delta": _float_delta(
            candidate_average_action_count,
            baseline_average_action_count,
        ),
        "average_evidence_coverage_delta": _float_delta(
            candidate_average_evidence_coverage,
            baseline_average_evidence_coverage,
        ),
        "baseline_average_action_count": baseline_average_action_count,
        "baseline_average_evidence_coverage": baseline_average_evidence_coverage,
        "baseline_success_count": baseline_success_count,
        "baseline_success_rate": baseline_success_rate,
        "baseline_task_count": baseline_task_count,
        "candidate_average_action_count": candidate_average_action_count,
        "candidate_average_evidence_coverage": candidate_average_evidence_coverage,
        "candidate_success_count": candidate_success_count,
        "candidate_success_rate": candidate_success_rate,
        "candidate_task_count": candidate_task_count,
        "success_count_delta": _int_delta(
            candidate_success_count,
            baseline_success_count,
        ),
        "success_rate_delta": _float_delta(
            candidate_success_rate,
            baseline_success_rate,
        ),
        "task_count_delta": _int_delta(candidate_task_count, baseline_task_count),
        "task_count_match": candidate_task_count == baseline_task_count,
    }


def _budget_bucket_delta_from_entry(entry: object) -> dict[str, Any]:
    return {
        "average_action_count_delta": _float_delta(
            _float_mapping_value(entry, "candidate_average_action_count"),
            _float_mapping_value(entry, "baseline_average_action_count"),
        ),
        "average_evidence_coverage_delta": _float_delta(
            _float_mapping_value(entry, "candidate_average_evidence_coverage"),
            _float_mapping_value(entry, "baseline_average_evidence_coverage"),
        ),
        "baseline_average_action_count": _float_mapping_value(
            entry,
            "baseline_average_action_count",
        ),
        "baseline_average_evidence_coverage": _float_mapping_value(
            entry,
            "baseline_average_evidence_coverage",
        ),
        "baseline_success_count": _int_mapping_value(entry, "baseline_success_count"),
        "baseline_success_rate": _float_mapping_value(entry, "baseline_success_rate"),
        "baseline_task_count": _int_mapping_value(entry, "baseline_task_count"),
        "candidate_average_action_count": _float_mapping_value(
            entry,
            "candidate_average_action_count",
        ),
        "candidate_average_evidence_coverage": _float_mapping_value(
            entry,
            "candidate_average_evidence_coverage",
        ),
        "candidate_success_count": _int_mapping_value(entry, "candidate_success_count"),
        "candidate_success_rate": _float_mapping_value(entry, "candidate_success_rate"),
        "candidate_task_count": _int_mapping_value(entry, "candidate_task_count"),
        "success_count_delta": _int_delta(
            _int_mapping_value(entry, "candidate_success_count"),
            _int_mapping_value(entry, "baseline_success_count"),
        ),
        "success_rate_delta": _float_delta(
            _float_mapping_value(entry, "candidate_success_rate"),
            _float_mapping_value(entry, "baseline_success_rate"),
        ),
        "task_count_delta": _int_delta(
            _int_mapping_value(entry, "candidate_task_count"),
            _int_mapping_value(entry, "baseline_task_count"),
        ),
        "task_count_match": (
            _int_mapping_value(entry, "candidate_task_count")
            == _int_mapping_value(entry, "baseline_task_count")
        ),
    }


def _evidence_coverage(
    evidence_nodes: Sequence[str],
    evidence_edges: Sequence[str],
    required_evidence: Mapping[str, Sequence[str]],
) -> float:
    required_nodes = tuple(required_evidence.get("nodes", ()))
    required_edges = tuple(required_evidence.get("edges", ()))
    required_count = len(set(required_nodes)) + len(set(required_edges))
    if required_count == 0:
        return 1.0
    node_hits = len(set(evidence_nodes) & set(required_nodes))
    edge_hits = len(set(evidence_edges) & set(required_edges))
    return round((node_hits + edge_hits) / required_count, 6)


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, 6)


def _average_float(values: Iterable[Any]) -> float:
    numbers = [float(value) for value in values]
    if not numbers:
        return 0.0
    return round(sum(numbers) / len(numbers), 6)


def _case_bool_count(case_results: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(1 for result in case_results if result.get(key) is True)


def _summary_value(summary: object, key: str) -> object:
    if not isinstance(summary, Mapping):
        return None
    return summary.get(key)


def _required_report_path(report: Mapping[str, Any], key: str) -> Path:
    value = report.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Active task report missing required path: {key}")
    return Path(value)


def _required_report_policy(report: Mapping[str, Any]) -> str:
    value = report.get("policy")
    if not isinstance(value, str) or value == "":
        raise SpatialQAError("Active task report missing required policy")
    return value


def _equality_check(name: str, saved: object, current: object) -> dict[str, Any]:
    return {
        "name": name,
        "passed": saved == current,
        "expected": saved,
        "actual": current,
    }


def _budget_curve_keys(value: object) -> list[str] | None:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return None
    keys: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        budget = item.get("max_actions")
        if isinstance(budget, bool) or not isinstance(budget, int):
            return None
        keys.append(str(budget))
    return keys


def _budget_mapping_keys(value: object) -> list[str] | None:
    if not isinstance(value, Mapping):
        return None
    keys: list[int] = []
    for key in value:
        if not isinstance(key, str):
            return None
        try:
            keys.append(int(key))
        except ValueError:
            return None
    return [str(key) for key in sorted(keys)]


def _budget_analysis_mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    by_max_actions = value.get("by_max_actions")
    if not isinstance(by_max_actions, Mapping):
        return {}
    return cast(Mapping[str, Any], by_max_actions)


def _sorted_budget_keys(*values: Mapping[str, Any]) -> list[str]:
    keys: set[int] = set()
    for value in values:
        for key in value:
            if not isinstance(key, str):
                continue
            try:
                keys.add(int(key))
            except ValueError:
                continue
    return [str(key) for key in sorted(keys)]


def _metric_int_value(metrics: object, metric_name: str, key: str) -> int | None:
    return _int_mapping_value(_metric_mapping(metrics, metric_name), key)


def _metric_float_value(metrics: object, metric_name: str, key: str) -> float | None:
    return _float_mapping_value(_metric_mapping(metrics, metric_name), key)


def _metric_mapping(metrics: object, metric_name: str) -> Mapping[str, Any] | None:
    if not isinstance(metrics, Mapping):
        return None
    metric = metrics.get(metric_name)
    return cast(Mapping[str, Any], metric) if isinstance(metric, Mapping) else None


def _int_mapping_value(payload: object, key: str) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _float_mapping_value(payload: object, key: str) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _bool_mapping_value(payload: object, key: str) -> bool | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(key)
    return value if isinstance(value, bool) else None


def _int_delta(candidate: int | None, baseline: int | None) -> int | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def _float_delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return round(candidate - baseline, 6)


def _required_delta_report_path(report: Mapping[str, Any], key: str) -> Path:
    value = report.get(key)
    if not isinstance(value, str) or not value:
        raise SpatialQAError(f"Active task delta report missing required path: {key}")
    return Path(value)


def _delta_report_name(report: Mapping[str, Any], key: str) -> str:
    value = report.get(key)
    return value if isinstance(value, str) and value else key.replace("_name", "")


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise SpatialQAError(f"Active task result field must be a string: {key}")
    return value


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SpatialQAError(f"Active task result field must be a string: {key}")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpatialQAError(f"Active task result field must be an integer: {key}")
    return value


def _required_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SpatialQAError(f"Active task result field must be a number: {key}")
    return float(value)


def _required_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise SpatialQAError(f"Active task result field must be a boolean: {key}")
    return value


def _required_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"Active task result field must be an object: {key}")
    return _json_mapping(cast(Mapping[str, Any], value))


def _mapping_tuple(payload: Mapping[str, Any], key: str) -> tuple[dict[str, Any], ...]:
    value = payload.get(key, ())
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SpatialQAError(f"Active task result field must be an object sequence: {key}")
    items: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise SpatialQAError(f"Active task result field must be an object sequence: {key}")
        items.append(_json_mapping(cast(Mapping[str, Any], item)))
    return tuple(items)


def _string_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key, ())
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SpatialQAError(f"Active task result field must be a string sequence: {key}")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise SpatialQAError(f"Active task result field must be a string sequence: {key}")
        items.append(item)
    return tuple(items)


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], _json_value(value))


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_json_value(item) for item in value]
    return value
