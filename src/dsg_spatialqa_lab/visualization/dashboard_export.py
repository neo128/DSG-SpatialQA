from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
from html import escape
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.benchmark import QACase, qa_case_to_dict
from dsg_spatialqa_lab.eval import QAPrediction, qa_prediction_to_dict
from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.scene_io import graph_summary
from dsg_spatialqa_lab.schema import Edge, Node, SpatialQAError


DASHBOARD_BUNDLE_SCHEMA_VERSION = "dsg-spatialqa-lab.dashboard-bundle.v1"


def dashboard_bundle(
    gold_cases: Sequence[QACase],
    *,
    predictions: Sequence[QAPrediction],
    qa_eval_report: Mapping[str, Any],
    graph: DynamicSceneGraph,
    error_attribution_report: Mapping[str, Any] | None = None,
    active_task_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prediction_by_id = _prediction_mapping(predictions)
    eval_by_id = _case_mapping(qa_eval_report.get("cases", ()))
    attribution_by_id = _case_mapping(
        error_attribution_report.get("cases", ())
        if error_attribution_report is not None
        else (),
    )
    cases = [
        _dashboard_case(
            case,
            graph=graph,
            prediction=prediction_by_id.get(case.id),
            eval_result=eval_by_id.get(case.id),
            attribution=attribution_by_id.get(case.id),
        )
        for case in gold_cases
    ]
    bundle: dict[str, Any] = {
        "schema_version": DASHBOARD_BUNDLE_SCHEMA_VERSION,
        "summary": _summary(cases, prediction_count=len(predictions)),
        "graph_summary": graph_summary(graph),
        "cases": cases,
    }
    if active_task_report is not None:
        bundle["active_task_review"] = _active_task_review(active_task_report)
    bundle["bundle_digest"] = dashboard_bundle_digest(bundle)
    return bundle


def dashboard_bundle_digest(bundle: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in bundle.items() if key != "bundle_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def dashboard_bundle_json(bundle: Mapping[str, Any]) -> str:
    return json.dumps(bundle, indent=2, sort_keys=True) + "\n"


def save_dashboard_bundle(bundle: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dashboard_bundle_json(bundle), encoding="utf-8")
    return output_path


def load_dashboard_bundle(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Dashboard bundle JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_dashboard_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = bundle.get("schema_version")
    digest = bundle.get("bundle_digest")
    expected_digest = dashboard_bundle_digest(bundle)
    cases = bundle.get("cases")
    case_count = (
        len(cases)
        if isinstance(cases, Sequence) and not isinstance(cases, str)
        else None
    )
    summary = bundle.get("summary")
    summary_case_count = (
        summary.get("case_count") if isinstance(summary, Mapping) else None
    )
    graph_summary_value = bundle.get("graph_summary")
    active_task_review = bundle.get("active_task_review")
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == DASHBOARD_BUNDLE_SCHEMA_VERSION,
            "expected": DASHBOARD_BUNDLE_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "bundle_digest",
            "passed": digest == expected_digest,
            "expected": expected_digest,
            "actual": digest,
        },
        {
            "name": "case_count",
            "passed": case_count == summary_case_count,
            "expected": case_count,
            "actual": summary_case_count,
        },
        {
            "name": "graph_summary_present",
            "passed": isinstance(graph_summary_value, Mapping),
        },
    ]
    if active_task_review is not None:
        checks.append(
            {
                "name": "active_task_review",
                "passed": _validate_active_task_review(active_task_review),
            }
        )
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "bundle_digest": digest,
        "checks": checks,
    }


def dashboard_html(bundle: Mapping[str, Any]) -> str:
    payload = json.dumps(bundle, separators=(",", ":"), sort_keys=True)
    rows = "\n".join(_table_row(case) for case in _case_rows(bundle))
    active_task_section = _active_task_section(bundle)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DSG-SpatialQA Dashboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #17202a; }}
    header {{ margin-bottom: 20px; }}
    .filters {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }}
    label {{ display: grid; gap: 4px; font-size: 0.9rem; }}
    select {{ min-width: 180px; padding: 6px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d7dee8; padding: 8px; text-align: left; }}
    th {{ background: #eef3f8; }}
    code {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <header>
    <h1>DSG-SpatialQA Dashboard</h1>
    <p>Cases: {escape(str(_summary_value(bundle, "case_count")))} · Digest: {escape(str(bundle.get("bundle_digest", "")))}</p>
  </header>
  <section class="filters" aria-label="Filters">
    <label>Question type<select id="question-type-filter"><option value="all">All</option></select></label>
    <label>Tag<select id="tag-filter"><option value="all">All</option></select></label>
    <label>Correctness<select id="correctness-filter"><option value="all">All</option></select></label>
    <label>Error category<select id="error-category-filter"><option value="all">All</option></select></label>
  </section>
  <table>
    <thead>
      <tr><th>Case</th><th>Question Type</th><th>Correct</th><th>Error</th><th>Evidence Source</th><th>Answer</th></tr>
    </thead>
    <tbody>
{rows}
    </tbody>
  </table>
{active_task_section}
  <script id="dashboard-data" type="application/json">{escape(payload)}</script>
</body>
</html>
"""


def export_dashboard(bundle: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    bundle_path = output_path / "dashboard.json"
    index_path = output_path / "index.html"
    save_dashboard_bundle(bundle, bundle_path)
    index_path.write_text(dashboard_html(bundle), encoding="utf-8")
    return {
        "bundle_path": str(bundle_path),
        "index_path": str(index_path),
        "summary": bundle["summary"],
        "digest": bundle["bundle_digest"],
    }


def _dashboard_case(
    case: QACase,
    *,
    graph: DynamicSceneGraph,
    prediction: QAPrediction | None,
    eval_result: Mapping[str, Any] | None,
    attribution: Mapping[str, Any] | None,
) -> dict[str, Any]:
    attribution_payload = _json_mapping(attribution) if attribution is not None else None
    return {
        "case_id": case.id,
        "qa_case": qa_case_to_dict(case),
        "prediction": qa_prediction_to_dict(prediction) if prediction is not None else None,
        "eval_result": _json_mapping(eval_result) if eval_result is not None else None,
        "error_attribution": attribution_payload,
        "evidence_subgraph": _evidence_subgraph(graph, case, prediction),
        "frame_paths": _frame_paths(case),
        "predicted_evidence_sources": _predicted_evidence_sources(attribution_payload),
    }


def _evidence_subgraph(
    graph: DynamicSceneGraph,
    case: QACase,
    prediction: QAPrediction | None,
) -> dict[str, Any]:
    node_ids = set(case.required_nodes)
    edge_ids = set(case.required_edges)
    if prediction is not None:
        node_ids.update(prediction.evidence_nodes)
        edge_ids.update(prediction.evidence_edges)
    edges = [edge for edge in graph.edges if edge.id in edge_ids]
    for edge in edges:
        node_ids.add(edge.src)
        node_ids.add(edge.dst)
    nodes = [graph.nodes[node_id] for node_id in sorted(node_ids) if node_id in graph.nodes]
    return {
        "nodes": [_node_to_dict(node) for node in nodes],
        "edges": [_edge_to_dict(edge) for edge in sorted(edges, key=lambda item: item.id)],
    }


def _node_to_dict(node: Node) -> dict[str, Any]:
    return {
        "id": node.id,
        "type": node.type,
        "label": node.label,
        "attributes": _json_mapping(node.attributes),
    }


def _edge_to_dict(edge: Edge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "src": edge.src,
        "relation": edge.relation,
        "dst": edge.dst,
        "reference_frame": edge.reference_frame,
        "confidence": edge.confidence,
        "step": edge.step,
        "evidence": list(edge.evidence),
        "attributes": _json_mapping(edge.attributes),
    }


def _frame_paths(case: QACase) -> dict[str, str]:
    frame_paths: dict[str, str] = {}
    nested = case.question.get("frame_paths")
    if isinstance(nested, Mapping):
        for key, value in sorted(nested.items(), key=lambda item: str(item[0])):
            if isinstance(value, str) and value:
                frame_paths[str(key)] = value
    for key in ("rgb_path", "depth_path", "segmentation_path"):
        value = case.question.get(key)
        if isinstance(value, str) and value:
            frame_paths[key] = value
    return frame_paths


def _summary(cases: Sequence[Mapping[str, Any]], *, prediction_count: int) -> dict[str, Any]:
    eval_results = [case.get("eval_result") for case in cases if case.get("eval_result") is not None]
    attributions = [
        case.get("error_attribution")
        for case in cases
        if case.get("error_attribution") is not None
    ]
    return {
        "case_count": len(cases),
        "prediction_count": prediction_count,
        "eval_result_count": len(eval_results),
        "attribution_count": len(attributions),
        "exact_match_count": sum(
            1
            for result in eval_results
            if isinstance(result, Mapping) and result.get("exact_match") is True
        ),
        "by_question_type": _sorted_counts(
            str(_case_qa(case).get("question_type")) for case in cases
        ),
        "by_error_category": _sorted_counts(
            str(attribution.get("error_category"))
            for attribution in attributions
            if isinstance(attribution, Mapping)
        ),
        "by_predicted_evidence_source": _predicted_evidence_source_summary(cases),
    }


def _active_task_review(report: Mapping[str, Any]) -> dict[str, Any]:
    tasks = _mapping_items(report.get("tasks"))
    results = _mapping_items(report.get("results"))
    case_results = _mapping_items(report.get("cases"))
    task_by_id = _mapping_by_key(tasks, "id")
    result_by_task_id = _mapping_by_key(results, "task_id")
    panels = []
    for case_result in case_results:
        task_id = _string_value(case_result.get("task_id"))
        if task_id == "":
            continue
        task = task_by_id.get(task_id)
        result = result_by_task_id.get(task_id)
        panels.append(
            _active_task_panel(
                task_id,
                task=task,
                result=result,
                case_result=case_result,
            )
        )
    return {
        "schema_version": report.get("schema_version"),
        "report_digest": report.get("report_digest"),
        "policy": report.get("policy"),
        "paths": {
            "graph_path": report.get("graph_path"),
            "task_path": report.get("task_path"),
        },
        "summary": _json_mapping(report["summary"]) if isinstance(report.get("summary"), Mapping) else {},
        "metrics": _json_mapping(report["metrics"]) if isinstance(report.get("metrics"), Mapping) else {},
        "budget_analysis": (
            _json_mapping(report["budget_analysis"])
            if isinstance(report.get("budget_analysis"), Mapping)
            else {}
        ),
        "panels": panels,
    }


def _active_task_panel(
    task_id: str,
    *,
    task: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None,
    case_result: Mapping[str, Any],
) -> dict[str, Any]:
    task_payload = _json_mapping(task) if task is not None else None
    result_payload = _json_mapping(result) if result is not None else None
    required = task.get("required_evidence") if task is not None else {}
    required_nodes = _string_items(required.get("nodes") if isinstance(required, Mapping) else ())
    required_edges = _string_items(required.get("edges") if isinstance(required, Mapping) else ())
    observed_nodes = _string_items(result.get("evidence_nodes") if result is not None else ())
    observed_edges = _string_items(result.get("evidence_edges") if result is not None else ())
    transcript = result.get("transcript") if result is not None else ()
    action_evidence_snapshots = (
        result.get("action_evidence_snapshots") if result is not None else ()
    )
    return {
        "task_id": task_id,
        "task": task_payload,
        "result": result_payload,
        "case_result": _json_mapping(case_result),
        "transcript": _json_value(transcript),
        "action_evidence_snapshots": _json_value(action_evidence_snapshots),
        "evidence": {
            "observed_edges": observed_edges,
            "observed_nodes": observed_nodes,
            "required_edges": required_edges,
            "required_nodes": required_nodes,
            "missing_edges": sorted(set(required_edges) - set(observed_edges)),
            "missing_nodes": sorted(set(required_nodes) - set(observed_nodes)),
        },
    }


def _validate_active_task_review(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    panels = value.get("panels")
    summary = value.get("summary")
    if not isinstance(panels, Sequence) or isinstance(panels, str):
        return False
    if not isinstance(summary, Mapping):
        return False
    task_count = summary.get("task_count")
    if isinstance(task_count, bool) or not isinstance(task_count, int):
        return False
    return len(panels) == task_count


def _active_task_section(bundle: Mapping[str, Any]) -> str:
    rows = "\n".join(_active_task_row(panel) for panel in _active_task_rows(bundle))
    if rows == "":
        return ""
    return f"""
  <section aria-label="Active Tasks">
    <h2>Active Tasks</h2>
    <table>
      <thead>
        <tr><th>Task</th><th>Policy</th><th>Success</th><th>Actions</th><th>Evidence</th><th>Transcript</th></tr>
      </thead>
      <tbody>
{rows}
      </tbody>
    </table>
  </section>"""


def _active_task_row(panel: Mapping[str, Any]) -> str:
    result = panel.get("result")
    case_result = panel.get("case_result")
    transcript = panel.get("transcript")
    policy = result.get("policy") if isinstance(result, Mapping) else ""
    success = case_result.get("task_success") if isinstance(case_result, Mapping) else ""
    action_count = result.get("action_count") if isinstance(result, Mapping) else ""
    evidence_coverage = (
        case_result.get("evidence_coverage") if isinstance(case_result, Mapping) else ""
    )
    return (
        "        <tr>"
        f"<td>{escape(str(panel.get('task_id', '')))}</td>"
        f"<td>{escape(str(policy))}</td>"
        f"<td>{escape(str(success))}</td>"
        f"<td>{escape(str(action_count))}</td>"
        f"<td>{escape(str(evidence_coverage))}</td>"
        f"<td><code>{escape(json.dumps(transcript, sort_keys=True))}</code></td>"
        "</tr>"
    )


def _active_task_rows(bundle: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    review = bundle.get("active_task_review")
    if not isinstance(review, Mapping):
        return ()
    panels = review.get("panels", ())
    if not isinstance(panels, Sequence) or isinstance(panels, str):
        return ()
    return tuple(cast(Mapping[str, Any], panel) for panel in panels if isinstance(panel, Mapping))


def _table_row(case: Mapping[str, Any]) -> str:
    qa_case = _case_qa(case)
    eval_result = case.get("eval_result")
    attribution = case.get("error_attribution")
    exact_match = (
        eval_result.get("exact_match")
        if isinstance(eval_result, Mapping)
        else None
    )
    error_category = (
        attribution.get("error_category")
        if isinstance(attribution, Mapping)
        else "none"
    )
    evidence_sources = ", ".join(_predicted_evidence_sources(case))
    return (
        "      <tr>"
        f"<td>{escape(str(case.get('case_id', '')))}</td>"
        f"<td>{escape(str(qa_case.get('question_type', '')))}</td>"
        f"<td>{escape(str(exact_match))}</td>"
        f"<td>{escape(str(error_category))}</td>"
        f"<td>{escape(evidence_sources)}</td>"
        f"<td><code>{escape(json.dumps(qa_case.get('answer', {}), sort_keys=True))}</code></td>"
        "</tr>"
    )


def _case_rows(bundle: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    cases = bundle.get("cases", ())
    if not isinstance(cases, Sequence) or isinstance(cases, str):
        return ()
    return tuple(cast(Mapping[str, Any], case) for case in cases if isinstance(case, Mapping))


def _case_qa(case: Mapping[str, Any]) -> Mapping[str, Any]:
    qa_case = case.get("qa_case")
    return cast(Mapping[str, Any], qa_case) if isinstance(qa_case, Mapping) else {}


def _summary_value(bundle: Mapping[str, Any], key: str) -> object:
    summary = bundle.get("summary")
    return summary.get(key) if isinstance(summary, Mapping) else None


def _prediction_mapping(predictions: Sequence[QAPrediction]) -> dict[str, QAPrediction]:
    mapping: dict[str, QAPrediction] = {}
    for prediction in predictions:
        if prediction.id not in mapping:
            mapping[prediction.id] = prediction
    return mapping


def _case_mapping(value: object) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return {}
    mapping: dict[str, Mapping[str, Any]] = {}
    for item in value:
        if isinstance(item, Mapping) and isinstance(item.get("case_id"), str):
            case_id = str(item["case_id"])
            if case_id not in mapping:
                mapping[case_id] = cast(Mapping[str, Any], item)
    return mapping


def _mapping_items(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    return tuple(cast(Mapping[str, Any], item) for item in value if isinstance(item, Mapping))


def _mapping_by_key(
    items: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, Mapping[str, Any]]:
    mapping: dict[str, Mapping[str, Any]] = {}
    for item in items:
        item_id = _string_value(item.get(key))
        if item_id != "" and item_id not in mapping:
            mapping[item_id] = item
    return mapping


def _string_items(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, str)]


def _string_value(value: object) -> str:
    return value if isinstance(value, str) else ""


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _predicted_evidence_sources(payload: Mapping[str, Any] | None) -> list[str]:
    if payload is None:
        return []
    sources = payload.get("predicted_evidence_sources")
    if not isinstance(sources, Sequence) or isinstance(sources, str):
        return []
    return sorted(source for source in sources if isinstance(source, str) and source)


def _predicted_evidence_source_summary(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        attribution = case.get("error_attribution")
        if not isinstance(attribution, Mapping):
            continue
        sources = _predicted_evidence_sources(case)
        if not sources:
            sources = ["missing_predicted_evidence"]
        for source in sources:
            grouped.setdefault(source, []).append(attribution)
    return {
        source: {
            "by_error_category": _sorted_counts(
                str(attribution.get("error_category"))
                for attribution in attributions
            ),
            "by_evidence_error_category": _sorted_counts(
                str(attribution.get("evidence_error_category"))
                for attribution in attributions
            ),
            "case_count": len(attributions),
            "error_count": sum(
                1
                for attribution in attributions
                if attribution.get("error_category") != "correct"
            ),
        }
        for source, attributions in sorted(grouped.items())
    }


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
