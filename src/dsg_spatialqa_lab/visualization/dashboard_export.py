from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
from html import escape
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.benchmark import QACase, load_qa_dataset, qa_case_to_dict
from dsg_spatialqa_lab.eval import (
    QAPrediction,
    load_qa_eval_report,
    load_qa_predictions,
    qa_prediction_to_dict,
)
from dsg_spatialqa_lab.eval.error_attribution import load_error_attribution_report
from dsg_spatialqa_lab.eval.qa_metrics import (
    QA_RESEARCH_AXIS_NAMES,
    qa_research_axes_for_case,
)
from dsg_spatialqa_lab.eval.task_metrics import (
    load_active_task_delta_report,
    load_active_task_report,
)
from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.scene_io import graph_summary, load_graph_json
from dsg_spatialqa_lab.schema import Edge, Node, SpatialQAError


DASHBOARD_BUNDLE_SCHEMA_VERSION = "dsg-spatialqa-lab.dashboard-bundle.v1"
DASHBOARD_REQUIRED_SOURCE_PATH_KEYS = (
    "graph_path",
    "prediction_path",
    "qa_eval_report_path",
    "qa_path",
)


def dashboard_bundle(
    gold_cases: Sequence[QACase],
    *,
    predictions: Sequence[QAPrediction],
    qa_eval_report: Mapping[str, Any],
    graph: DynamicSceneGraph,
    error_attribution_report: Mapping[str, Any] | None = None,
    active_task_report: Mapping[str, Any] | None = None,
    active_task_delta_report: Mapping[str, Any] | None = None,
    experiment_summary_report: Mapping[str, Any] | None = None,
    qa_path: str | Path | None = None,
    prediction_path: str | Path | None = None,
    qa_eval_report_path: str | Path | None = None,
    graph_path: str | Path | None = None,
    error_attribution_report_path: str | Path | None = None,
    active_task_report_path: str | Path | None = None,
    active_task_delta_report_path: str | Path | None = None,
    experiment_summary_report_path: str | Path | None = None,
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
    if active_task_delta_report is not None:
        bundle["active_task_delta_review"] = _active_task_delta_review(
            active_task_delta_report
        )
    if experiment_summary_report is not None:
        bundle["experiment_summary_review"] = _experiment_summary_review(
            experiment_summary_report
        )
    source_paths = _source_paths(
        qa_path=qa_path,
        prediction_path=prediction_path,
        qa_eval_report_path=qa_eval_report_path,
        graph_path=graph_path,
        error_attribution_report_path=error_attribution_report_path,
        active_task_report_path=active_task_report_path,
        active_task_delta_report_path=active_task_delta_report_path,
        experiment_summary_report_path=experiment_summary_report_path,
    )
    if source_paths:
        bundle["source_paths"] = source_paths
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
    active_task_delta_review = bundle.get("active_task_delta_review")
    experiment_summary_review = bundle.get("experiment_summary_review")
    source_paths = bundle.get("source_paths")
    expected_source_summary = _predicted_evidence_source_summary(_case_rows(bundle))
    actual_source_summary = (
        summary.get("by_predicted_evidence_source")
        if isinstance(summary, Mapping)
        else None
    )
    expected_axis_summary = _research_axis_summary(_case_rows(bundle))
    actual_axis_summary = (
        summary.get("by_research_axis")
        if isinstance(summary, Mapping)
        else None
    )
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
        {
            "name": "predicted_evidence_source_summary",
            "passed": actual_source_summary == expected_source_summary,
            "expected": expected_source_summary,
            "actual": actual_source_summary,
        },
        {
            "name": "research_axis_summary",
            "passed": actual_axis_summary == expected_axis_summary,
            "expected": expected_axis_summary,
            "actual": actual_axis_summary,
        },
    ]
    if active_task_review is not None:
        checks.append(
            {
                "name": "active_task_review",
                "passed": _validate_active_task_review(active_task_review),
            }
        )
    if active_task_delta_review is not None:
        checks.append(
            {
                "name": "active_task_delta_review",
                "passed": _validate_active_task_delta_review(active_task_delta_review),
            }
        )
    if experiment_summary_review is not None:
        checks.append(
            {
                "name": "experiment_summary_review",
                "passed": _validate_experiment_summary_review(
                    experiment_summary_review
                ),
            }
        )
    if source_paths is not None:
        checks.append(
            {
                "name": "source_paths",
                "passed": _valid_source_paths(source_paths),
            }
        )
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "bundle_digest": digest,
        "checks": checks,
    }


def compare_dashboard_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_dashboard_bundle(bundle)
    source_paths = _dashboard_source_paths(bundle.get("source_paths"))
    missing_source_paths = [
        key for key in DASHBOARD_REQUIRED_SOURCE_PATH_KEYS if key not in source_paths
    ]
    saved_bundle_digest = _string_value(bundle.get("bundle_digest")) or None
    checks: list[dict[str, Any]] = [
        {
            "name": "bundle_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "required_source_paths_present",
            "passed": not missing_source_paths,
            "expected": [],
            "actual": missing_source_paths,
        },
    ]
    if missing_source_paths:
        return {
            "matches": False,
            "comparable": False,
            "saved_bundle_digest": saved_bundle_digest,
            "current_bundle_digest": None,
            "missing_source_paths": missing_source_paths,
            "validation": validation,
            "checks": checks,
        }
    try:
        current_bundle = _current_dashboard_bundle(source_paths)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        checks.append(
            {
                "name": "current_bundle_loadable",
                "passed": False,
                "error": str(exc),
            }
        )
        return {
            "matches": False,
            "comparable": False,
            "saved_bundle_digest": saved_bundle_digest,
            "current_bundle_digest": None,
            "missing_source_paths": [],
            "validation": validation,
            "checks": checks,
        }
    current_bundle_digest = _string_value(current_bundle.get("bundle_digest")) or None
    checks.extend(
        [
            {
                "name": "current_bundle_loadable",
                "passed": True,
            },
            _equality_check(
                "bundle_digest_matches_current",
                saved_bundle_digest,
                current_bundle_digest,
            ),
            _equality_check(
                "summary_matches_current",
                bundle.get("summary"),
                current_bundle["summary"],
            ),
            _equality_check(
                "graph_summary_matches_current",
                bundle.get("graph_summary"),
                current_bundle["graph_summary"],
            ),
            _equality_check(
                "cases_match_current",
                bundle.get("cases"),
                current_bundle["cases"],
            ),
            _equality_check(
                "active_task_review_matches_current",
                bundle.get("active_task_review"),
                current_bundle.get("active_task_review"),
            ),
            _equality_check(
                "active_task_delta_review_matches_current",
                bundle.get("active_task_delta_review"),
                current_bundle.get("active_task_delta_review"),
            ),
            _equality_check(
                "experiment_summary_review_matches_current",
                bundle.get("experiment_summary_review"),
                current_bundle.get("experiment_summary_review"),
            ),
        ]
    )
    return {
        "matches": all(check["passed"] is True for check in checks),
        "comparable": True,
        "saved_bundle_digest": saved_bundle_digest,
        "current_bundle_digest": current_bundle_digest,
        "missing_source_paths": [],
        "validation": validation,
        "checks": checks,
    }


def _current_dashboard_bundle(source_paths: Mapping[str, str]) -> dict[str, Any]:
    error_attribution_report = _optional_source_report(
        source_paths,
        "error_attribution_report_path",
        load_error_attribution_report,
    )
    active_task_report = _optional_source_report(
        source_paths,
        "active_task_report_path",
        load_active_task_report,
    )
    active_task_delta_report = _optional_source_report(
        source_paths,
        "active_task_delta_report_path",
        load_active_task_delta_report,
    )
    experiment_summary_report = _optional_json_source_report(
        source_paths,
        "experiment_summary_report_path",
    )
    return dashboard_bundle(
        load_qa_dataset(source_paths["qa_path"]),
        predictions=load_qa_predictions(source_paths["prediction_path"]),
        qa_eval_report=load_qa_eval_report(source_paths["qa_eval_report_path"]),
        graph=load_graph_json(source_paths["graph_path"]),
        error_attribution_report=error_attribution_report,
        active_task_report=active_task_report,
        active_task_delta_report=active_task_delta_report,
        experiment_summary_report=experiment_summary_report,
        qa_path=source_paths["qa_path"],
        prediction_path=source_paths["prediction_path"],
        qa_eval_report_path=source_paths["qa_eval_report_path"],
        graph_path=source_paths["graph_path"],
        error_attribution_report_path=source_paths.get(
            "error_attribution_report_path"
        ),
        active_task_report_path=source_paths.get("active_task_report_path"),
        active_task_delta_report_path=source_paths.get(
            "active_task_delta_report_path"
        ),
        experiment_summary_report_path=source_paths.get(
            "experiment_summary_report_path"
        ),
    )


def _optional_source_report(
    source_paths: Mapping[str, str],
    key: str,
    loader: Callable[[str | Path], dict[str, Any]],
) -> dict[str, Any] | None:
    path = source_paths.get(key)
    if path is None:
        return None
    return loader(path)


def _optional_json_source_report(
    source_paths: Mapping[str, str],
    key: str,
) -> dict[str, Any] | None:
    path = source_paths.get(key)
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Dashboard source report JSON must be an object")
    return cast(dict[str, Any], payload)


def _source_paths(
    *,
    qa_path: str | Path | None,
    prediction_path: str | Path | None,
    qa_eval_report_path: str | Path | None,
    graph_path: str | Path | None,
    error_attribution_report_path: str | Path | None,
    active_task_report_path: str | Path | None,
    active_task_delta_report_path: str | Path | None,
    experiment_summary_report_path: str | Path | None,
) -> dict[str, str]:
    rows = (
        ("active_task_delta_report_path", active_task_delta_report_path),
        ("active_task_report_path", active_task_report_path),
        ("error_attribution_report_path", error_attribution_report_path),
        ("experiment_summary_report_path", experiment_summary_report_path),
        ("graph_path", graph_path),
        ("prediction_path", prediction_path),
        ("qa_eval_report_path", qa_eval_report_path),
        ("qa_path", qa_path),
    )
    return {
        key: str(path)
        for key, path in rows
        if path is not None and str(path) != ""
    }


def _dashboard_source_paths(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, str) and item
    }


def _valid_source_paths(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
    )


def _equality_check(name: str, expected: object, actual: object) -> dict[str, Any]:
    return {
        "name": name,
        "passed": expected == actual,
        "expected": expected,
        "actual": actual,
    }


def dashboard_html(bundle: Mapping[str, Any]) -> str:
    payload = json.dumps(bundle, separators=(",", ":"), sort_keys=True)
    cases = _case_rows(bundle)
    rows = "\n".join(_table_row(case) for case in cases)
    experiment_summary_section = _experiment_summary_section(bundle)
    active_task_section = _active_task_section(bundle)
    active_task_delta_section = _active_task_delta_section(bundle)
    question_type_options = _filter_options(
        _string_value(_case_qa(case).get("question_type")) for case in cases
    )
    tag_options = _filter_options(tag for case in cases for tag in _case_tags(case))
    correctness_options = _filter_options(_case_exact_match_value(case) for case in cases)
    error_category_options = _filter_options(_case_error_category(case) for case in cases)
    research_axis_options = _filter_options(
        axis for case in cases for axis in _case_research_axes(case)
    )
    evidence_source_options = _filter_options(
        source for case in cases for source in _predicted_evidence_sources(case)
    )
    source_profile_options = _filter_options(_source_profile_keys(bundle))
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
    <label>Question type<select id="question-type-filter"><option value="all">All</option>{question_type_options}</select></label>
    <label>Tag<select id="tag-filter"><option value="all">All</option>{tag_options}</select></label>
    <label>Correctness<select id="correctness-filter"><option value="all">All</option>{correctness_options}</select></label>
    <label>Error category<select id="error-category-filter"><option value="all">All</option>{error_category_options}</select></label>
    <label>Research axis<select id="research-axis-filter"><option value="all">All</option>{research_axis_options}</select></label>
    <label>Evidence source<select id="evidence-source-filter"><option value="all">All</option>{evidence_source_options}</select></label>
    <label>Source profile<select id="source-profile-filter"><option value="all">All</option>{source_profile_options}</select></label>
  </section>
  <table>
    <thead>
      <tr><th>Case</th><th>Question Type</th><th>Correct</th><th>Error</th><th>Evidence Source</th><th>Answer</th></tr>
    </thead>
    <tbody>
{rows}
    </tbody>
  </table>
{experiment_summary_section}
{active_task_section}
{active_task_delta_section}
  <script id="dashboard-data" type="application/json">{escape(payload)}</script>
  <script>
    (function () {{
      const filters = {{
        questionType: document.getElementById("question-type-filter"),
        tag: document.getElementById("tag-filter"),
        correctness: document.getElementById("correctness-filter"),
        errorCategory: document.getElementById("error-category-filter"),
        researchAxis: document.getElementById("research-axis-filter"),
        evidenceSource: document.getElementById("evidence-source-filter"),
        sourceProfile: document.getElementById("source-profile-filter")
      }};
      const rows = Array.from(document.querySelectorAll("tbody tr[data-case-id]"));
      const sourceProfileRows = Array.from(document.querySelectorAll("tbody tr[data-source-profile-key]"));
      function matches(values, selected) {{
        return selected === "all" || values.split("|").includes(selected);
      }}
      function applyDashboardFilters() {{
        rows.forEach(function (row) {{
          const visible = (
            matches(row.dataset.questionType || "", filters.questionType.value) &&
            matches(row.dataset.tags || "", filters.tag.value) &&
            matches(row.dataset.exactMatch || "", filters.correctness.value) &&
            matches(row.dataset.errorCategory || "", filters.errorCategory.value) &&
            matches(row.dataset.researchAxes || "", filters.researchAxis.value) &&
            matches(row.dataset.evidenceSources || "", filters.evidenceSource.value)
          );
          row.hidden = !visible;
        }});
        sourceProfileRows.forEach(function (row) {{
          row.hidden = !matches(row.dataset.sourceProfileKey || "", filters.sourceProfile.value);
        }});
      }}
      Object.values(filters).forEach(function (filter) {{
        filter.addEventListener("change", applyDashboardFilters);
      }});
      applyDashboardFilters();
    }}());
  </script>
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
        "research_axes": list(qa_research_axes_for_case(qa_case_to_dict(case))),
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
        "by_research_axis": _research_axis_summary(cases),
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


def _active_task_delta_review(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report.get("schema_version"),
        "report_digest": report.get("report_digest"),
        "candidate_name": report.get("candidate_name"),
        "baseline_name": report.get("baseline_name"),
        "candidate_report_digest": report.get("candidate_report_digest"),
        "baseline_report_digest": report.get("baseline_report_digest"),
        "paths": {
            "candidate_report_path": report.get("candidate_report_path"),
            "baseline_report_path": report.get("baseline_report_path"),
        },
        "summary_delta": (
            _json_mapping(report["summary_delta"])
            if isinstance(report.get("summary_delta"), Mapping)
            else {}
        ),
        "metrics_delta": (
            _json_mapping(report["metrics_delta"])
            if isinstance(report.get("metrics_delta"), Mapping)
            else {}
        ),
        "budget_delta": (
            _json_mapping(report["budget_delta"])
            if isinstance(report.get("budget_delta"), Mapping)
            else {}
        ),
    }


def _experiment_summary_review(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report.get("schema_version"),
        "report_digest": report.get("report_digest"),
        "manifest_path": report.get("manifest_path"),
        "manifest_digest": report.get("manifest_digest"),
        "summary": (
            _json_mapping(report["summary"])
            if isinstance(report.get("summary"), Mapping)
            else {}
        ),
        "readiness": (
            _json_mapping(report["readiness"])
            if isinstance(report.get("readiness"), Mapping)
            else {}
        ),
        "source_artifact_digests": (
            _json_mapping(report["source_artifact_digests"])
            if isinstance(report.get("source_artifact_digests"), Mapping)
            else {}
        ),
        "research_questions": (
            _json_mapping(report["research_questions"])
            if isinstance(report.get("research_questions"), Mapping)
            else {}
        ),
        "source_profile_matrix": _source_profile_matrix(
            report.get("source_profile_matrix")
        ),
        "failure_linkage_review": _failure_linkage_review(
            report.get("failure_linkage_diagnostics")
        ),
        "research_question_matrix": _research_question_matrix(
            report.get("research_questions")
        ),
    }


def _failure_linkage_review(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for artifact_key, row in sorted(value.items(), key=lambda item: str(item[0])):
        if not isinstance(row, Mapping):
            continue
        error_key = row.get("error_attribution_artifact_key")
        rows.append(
            _json_mapping(
                {
                    "error_attribution_artifact_key": (
                        error_key if isinstance(error_key, str) else str(artifact_key)
                    ),
                    "graph_eval_artifact_key": row.get("graph_eval_artifact_key"),
                    "linked_by": row.get("linked_by"),
                    "graph_primary_metrics": (
                        row["graph_primary_metrics"]
                        if isinstance(row.get("graph_primary_metrics"), Mapping)
                        else {}
                    ),
                    "graph_diagnostics": (
                        row["graph_diagnostics"]
                        if isinstance(row.get("graph_diagnostics"), Mapping)
                        else {}
                    ),
                    "attribution_summary": (
                        row["attribution_summary"]
                        if isinstance(row.get("attribution_summary"), Mapping)
                        else {}
                    ),
                }
            )
        )
    return rows


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


def _validate_active_task_delta_review(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if not isinstance(value.get("candidate_name"), str):
        return False
    if not isinstance(value.get("baseline_name"), str):
        return False
    if not isinstance(value.get("report_digest"), str):
        return False
    if not isinstance(value.get("summary_delta"), Mapping):
        return False
    if not isinstance(value.get("metrics_delta"), Mapping):
        return False
    if not isinstance(value.get("budget_delta"), Mapping):
        return False
    paths = value.get("paths")
    return isinstance(paths, Mapping)


def _validate_experiment_summary_review(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if not isinstance(value.get("report_digest"), str):
        return False
    if not isinstance(value.get("manifest_digest"), str):
        return False
    summary = value.get("summary")
    readiness = value.get("readiness")
    research_questions = value.get("research_questions")
    research_question_matrix = value.get("research_question_matrix")
    failure_linkage_review = value.get("failure_linkage_review")
    source_profile_matrix = value.get("source_profile_matrix")
    if not isinstance(summary, Mapping):
        return False
    if not isinstance(readiness, Mapping):
        return False
    if not isinstance(readiness.get("status"), str):
        return False
    if not isinstance(research_questions, Mapping):
        return False
    if not isinstance(research_question_matrix, Sequence) or isinstance(
        research_question_matrix,
        str,
    ):
        return False
    if failure_linkage_review is not None and (
        not isinstance(failure_linkage_review, Sequence)
        or isinstance(failure_linkage_review, str)
        or not all(isinstance(item, Mapping) for item in failure_linkage_review)
    ):
        return False
    if not isinstance(source_profile_matrix, Sequence) or isinstance(
        source_profile_matrix,
        str,
    ):
        return False
    if not all(isinstance(item, Mapping) for item in source_profile_matrix):
        return False
    research_question_count = summary.get("research_question_count")
    if isinstance(research_question_count, bool) or not isinstance(
        research_question_count,
        int,
    ):
        return False
    source_profile_count = summary.get("source_profile_count", 0)
    if isinstance(source_profile_count, bool) or not isinstance(source_profile_count, int):
        return False
    return (
        len(research_questions) == research_question_count
        and len(research_question_matrix)
        == _experiment_summary_measurement_count(research_questions)
        and len(source_profile_matrix) == source_profile_count
    )


def _experiment_summary_section(bundle: Mapping[str, Any]) -> str:
    review = bundle.get("experiment_summary_review")
    if not isinstance(review, Mapping):
        return ""
    readiness = _mapping_value(review.get("readiness"))
    missing_questions = ", ".join(
        _string_items(readiness.get("missing_research_questions"))
    )
    if missing_questions == "":
        missing_questions = "none"
    rows = "\n".join(
        _experiment_summary_row(item)
        for item in _experiment_summary_research_question_rows(review)
    )
    matrix_rows = "\n".join(
        _experiment_summary_matrix_row(row)
        for row in _experiment_summary_matrix_rows(review)
    )
    matrix_table = ""
    if matrix_rows != "":
        matrix_table = f"""
    <h3>Measurement Matrix</h3>
    <table>
      <thead>
        <tr><th>Question</th><th>Candidate</th><th>Baseline</th><th>Verdict</th><th>Metric</th><th>Value</th><th>Artifact</th></tr>
      </thead>
      <tbody>
{matrix_rows}
      </tbody>
    </table>"""
    source_profile_rows = "\n".join(
        _source_profile_row(row) for row in _source_profile_rows(review)
    )
    source_profile_table = ""
    if source_profile_rows != "":
        source_profile_table = f"""
    <h3>Source Profiles</h3>
    <table>
      <thead>
        <tr><th>Source</th><th>Adapter</th><th>Model</th><th>Prompt</th><th>Capabilities</th><th>Imported</th><th>Missing</th><th>Artifact</th></tr>
      </thead>
      <tbody>
{source_profile_rows}
      </tbody>
    </table>"""
    linkage_rows = "\n".join(
        _failure_linkage_row(row)
        for row in _failure_linkage_rows(review)
    )
    linkage_table = ""
    if linkage_rows != "":
        linkage_table = f"""
    <h3>Failure Linkage</h3>
    <table>
      <thead>
        <tr><th>Attribution</th><th>Graph Eval</th><th>Linked By</th><th>Graph Metrics</th><th>Graph Diagnostics</th><th>Failures</th></tr>
      </thead>
      <tbody>
{linkage_rows}
      </tbody>
    </table>"""
    return f"""
  <section aria-label="Experiment Summary">
    <h2>Experiment Summary</h2>
    <p>Manifest: {escape(str(review.get("manifest_path", "")))} · Digest: {escape(str(review.get("report_digest", "")))} · Readiness: {escape(str(readiness.get("status", "")))} · Missing RQ: {escape(missing_questions)}</p>
    <table>
      <thead>
        <tr><th>Question</th><th>Status</th><th>Verdict</th><th>Source</th><th>Primary Metric</th><th>Value</th></tr>
      </thead>
      <tbody>
{rows}
      </tbody>
    </table>
{matrix_table}
{source_profile_table}
{linkage_table}
  </section>"""


def _experiment_summary_research_question_rows(
    review: Mapping[str, Any],
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    research_questions = review.get("research_questions")
    if not isinstance(research_questions, Mapping):
        return ()
    return tuple(
        (str(name), cast(Mapping[str, Any], value))
        for name, value in sorted(research_questions.items(), key=lambda item: str(item[0]))
        if isinstance(value, Mapping)
    )


def _experiment_summary_row(item: tuple[str, Mapping[str, Any]]) -> str:
    name, question = item
    primary_metric = question.get("primary_metric")
    metric_name = ""
    metric_value = ""
    if isinstance(primary_metric, Mapping):
        metric_name = str(primary_metric.get("name", ""))
        metric_value = str(primary_metric.get("value", ""))
    return (
        "        <tr>"
        f"<td>{escape(name)}</td>"
        f"<td>{escape(str(question.get('status', '')))}</td>"
        f"<td>{escape(str(question.get('verdict', '')))}</td>"
        f"<td>{escape(str(question.get('source_artifact_type', '')))}</td>"
        f"<td>{escape(metric_name)}</td>"
        f"<td>{escape(metric_value)}</td>"
        "</tr>"
    )


def _research_question_matrix(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for question_name, question_value in sorted(value.items(), key=lambda item: str(item[0])):
        if not isinstance(question_value, Mapping):
            continue
        source_artifact_type = question_value.get("source_artifact_type")
        question_verdict = question_value.get("verdict")
        label = question_value.get("label")
        measurements = question_value.get("measurements")
        if not isinstance(measurements, Sequence) or isinstance(measurements, str):
            continue
        for measurement in measurements:
            if not isinstance(measurement, Mapping):
                continue
            primary_metric = (
                _json_mapping(measurement["primary_metric"])
                if isinstance(measurement.get("primary_metric"), Mapping)
                else {}
            )
            row = {
                "artifact_key": measurement.get("artifact_key"),
                "baseline_name": measurement.get("baseline_name"),
                "candidate_name": measurement.get("candidate_name"),
                "case_count_match": measurement.get("case_count_match"),
                "label": label,
                "measurement_verdict": _measurement_verdict(
                    primary_metric,
                    measurement.get("case_count_match"),
                ),
                "primary_metric": primary_metric,
                "question_verdict": question_verdict,
                "research_question": str(question_name),
                "source_artifact_type": source_artifact_type,
                "status": question_value.get("status"),
                "supporting_metrics": (
                    _json_mapping(measurement["supporting_metrics"])
                    if isinstance(measurement.get("supporting_metrics"), Mapping)
                    else {}
                ),
            }
            rows.append(_json_mapping(row))
    return rows


def _source_profile_matrix(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [
        _json_mapping(row)
        for row in value
        if isinstance(row, Mapping)
    ]


def _measurement_verdict(primary_metric: Mapping[str, Any], case_count_match: object) -> str:
    if case_count_match is False:
        return "inconclusive"
    value = primary_metric.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "inconclusive"
    if value > 0:
        return "improved"
    if value < 0:
        return "regressed"
    return "unchanged"


def _experiment_summary_measurement_count(
    research_questions: Mapping[str, Any],
) -> int:
    count = 0
    for value in research_questions.values():
        if not isinstance(value, Mapping):
            continue
        measurements = value.get("measurements")
        if isinstance(measurements, Sequence) and not isinstance(measurements, str):
            count += sum(1 for item in measurements if isinstance(item, Mapping))
    return count


def _experiment_summary_matrix_rows(
    review: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    matrix = review.get("research_question_matrix")
    if not isinstance(matrix, Sequence) or isinstance(matrix, str):
        return ()
    return tuple(cast(Mapping[str, Any], row) for row in matrix if isinstance(row, Mapping))


def _experiment_summary_matrix_row(row: Mapping[str, Any]) -> str:
    primary_metric = row.get("primary_metric")
    metric_name = ""
    metric_value = ""
    if isinstance(primary_metric, Mapping):
        metric_name = str(primary_metric.get("name", ""))
        metric_value = str(primary_metric.get("value", ""))
    return (
        "        <tr>"
        f"<td>{escape(str(row.get('research_question', '')))}</td>"
        f"<td>{escape(str(row.get('candidate_name', '')))}</td>"
        f"<td>{escape(str(row.get('baseline_name', '')))}</td>"
        f"<td>{escape(str(row.get('measurement_verdict', '')))}</td>"
        f"<td>{escape(metric_name)}</td>"
        f"<td>{escape(metric_value)}</td>"
        f"<td>{escape(str(row.get('artifact_key', '')))}</td>"
        "</tr>"
    )


def _source_profile_rows(review: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    matrix = review.get("source_profile_matrix")
    if not isinstance(matrix, Sequence) or isinstance(matrix, str):
        return ()
    return tuple(cast(Mapping[str, Any], row) for row in matrix if isinstance(row, Mapping))


def _source_profile_row(row: Mapping[str, Any]) -> str:
    capability_axes = _string_items(row.get("capability_axes"))
    capabilities = ", ".join(capability_axes)
    capability_value = "|".join(capability_axes)
    return (
        "        <tr"
        f" data-source-profile-key=\"{escape(str(row.get('source_key', '')))}\""
        f" data-source-profile-capabilities=\"{escape(capability_value)}\""
        ">"
        f"<td>{escape(str(row.get('source_key', '')))}</td>"
        f"<td>{escape(str(row.get('adapter', '')))}</td>"
        f"<td>{escape(str(row.get('model_id', '')))}</td>"
        f"<td>{escape(str(row.get('prompt_id', '')))}</td>"
        f"<td>{escape(capabilities)}</td>"
        f"<td>{escape(str(row.get('imported_prediction_count', '')))}</td>"
        f"<td>{escape(str(row.get('missing_case_count', '')))}</td>"
        f"<td>{escape(str(row.get('artifact_key', '')))}</td>"
        "</tr>"
    )


def _failure_linkage_rows(review: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = review.get("failure_linkage_review")
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        return ()
    return tuple(cast(Mapping[str, Any], row) for row in rows if isinstance(row, Mapping))


def _failure_linkage_row(row: Mapping[str, Any]) -> str:
    metrics = row.get("graph_primary_metrics")
    diagnostics = row.get("graph_diagnostics")
    attribution_summary = row.get("attribution_summary")
    return (
        "        <tr>"
        f"<td>{escape(str(row.get('error_attribution_artifact_key', '')))}</td>"
        f"<td>{escape(str(row.get('graph_eval_artifact_key', '')))}</td>"
        f"<td>{escape(str(row.get('linked_by', '')))}</td>"
        f"<td><code>{escape(json.dumps(metrics if isinstance(metrics, Mapping) else {}, sort_keys=True))}</code></td>"
        f"<td><code>{escape(json.dumps(diagnostics if isinstance(diagnostics, Mapping) else {}, sort_keys=True))}</code></td>"
        f"<td><code>{escape(json.dumps(attribution_summary if isinstance(attribution_summary, Mapping) else {}, sort_keys=True))}</code></td>"
        "</tr>"
    )


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


def _active_task_delta_section(bundle: Mapping[str, Any]) -> str:
    review = bundle.get("active_task_delta_review")
    if not isinstance(review, Mapping):
        return ""
    metric_rows = "\n".join(_active_task_delta_metric_row(item) for item in _active_task_delta_metric_rows(review))
    budget_rows = "\n".join(_active_task_delta_budget_row(item) for item in _active_task_delta_budget_rows(review))
    return f"""
  <section aria-label="Active Task Delta">
    <h2>Active Task Delta</h2>
    <p>Candidate: {escape(str(review.get("candidate_name", "")))} · Baseline: {escape(str(review.get("baseline_name", "")))}</p>
    <table>
      <thead>
        <tr><th>Metric</th><th>Count Delta</th><th>Rate Delta</th><th>Average Delta</th></tr>
      </thead>
      <tbody>
{metric_rows}
      </tbody>
    </table>
    <table>
      <thead>
        <tr><th>Max Actions</th><th>Success Rate Delta</th><th>Evidence Delta</th><th>Action Delta</th></tr>
      </thead>
      <tbody>
{budget_rows}
      </tbody>
    </table>
  </section>"""


def _active_task_delta_metric_rows(review: Mapping[str, Any]) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    metrics = review.get("metrics_delta")
    if not isinstance(metrics, Mapping):
        return ()
    return tuple(
        (str(name), cast(Mapping[str, Any], value))
        for name, value in sorted(metrics.items(), key=lambda item: str(item[0]))
        if isinstance(value, Mapping)
    )


def _active_task_delta_metric_row(item: tuple[str, Mapping[str, Any]]) -> str:
    name, metric = item
    return (
        "        <tr>"
        f"<td>{escape(name)}</td>"
        f"<td>{escape(str(metric.get('count_delta', '')))}</td>"
        f"<td>{escape(str(metric.get('rate_delta', '')))}</td>"
        f"<td>{escape(str(metric.get('average_delta', '')))}</td>"
        "</tr>"
    )


def _active_task_delta_budget_rows(review: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    budget = review.get("budget_delta")
    if not isinstance(budget, Mapping):
        return ()
    curve = budget.get("budget_curve", ())
    if not isinstance(curve, Sequence) or isinstance(curve, str):
        return ()
    return tuple(cast(Mapping[str, Any], item) for item in curve if isinstance(item, Mapping))


def _active_task_delta_budget_row(item: Mapping[str, Any]) -> str:
    return (
        "        <tr>"
        f"<td>{escape(str(item.get('max_actions', '')))}</td>"
        f"<td>{escape(str(item.get('success_rate_delta', '')))}</td>"
        f"<td>{escape(str(item.get('average_evidence_coverage_delta', '')))}</td>"
        f"<td>{escape(str(item.get('average_action_count_delta', '')))}</td>"
        "</tr>"
    )


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
    exact_match = _case_exact_match_value(case)
    error_category = _case_error_category(case)
    research_axis_items = _case_research_axes(case)
    evidence_source_items = _predicted_evidence_sources(case)
    evidence_sources = ", ".join(evidence_source_items)
    tag_items = _case_tags(case)
    return (
        "      <tr"
        f" data-case-id=\"{escape(str(case.get('case_id', '')))}\""
        f" data-question-type=\"{escape(str(qa_case.get('question_type', '')))}\""
        f" data-tags=\"{escape('|'.join(tag_items))}\""
        f" data-exact-match=\"{escape(exact_match)}\""
        f" data-error-category=\"{escape(error_category)}\""
        f" data-research-axes=\"{escape('|'.join(research_axis_items))}\""
        f" data-evidence-sources=\"{escape('|'.join(evidence_source_items))}\""
        ">"
        f"<td>{escape(str(case.get('case_id', '')))}</td>"
        f"<td>{escape(str(qa_case.get('question_type', '')))}</td>"
        f"<td>{escape(exact_match)}</td>"
        f"<td>{escape(error_category)}</td>"
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


def _case_tags(case: Mapping[str, Any]) -> tuple[str, ...]:
    tags = _case_qa(case).get("tags", ())
    if not isinstance(tags, Sequence) or isinstance(tags, str):
        return ()
    return tuple(tag for tag in tags if isinstance(tag, str) and tag)


def _case_exact_match_value(case: Mapping[str, Any]) -> str:
    eval_result = case.get("eval_result")
    if not isinstance(eval_result, Mapping):
        return "None"
    return str(eval_result.get("exact_match"))


def _case_error_category(case: Mapping[str, Any]) -> str:
    attribution = case.get("error_attribution")
    if not isinstance(attribution, Mapping):
        return "none"
    return str(attribution.get("error_category"))


def _case_research_axes(case: Mapping[str, Any]) -> tuple[str, ...]:
    axes = case.get("research_axes")
    if not isinstance(axes, Sequence) or isinstance(axes, str):
        return ()
    return tuple(axis for axis in axes if isinstance(axis, str) and axis)


def _source_profile_keys(bundle: Mapping[str, Any]) -> tuple[str, ...]:
    review = bundle.get("experiment_summary_review")
    if not isinstance(review, Mapping):
        return ()
    return tuple(
        str(row["source_key"])
        for row in _source_profile_rows(review)
        if isinstance(row.get("source_key"), str) and row.get("source_key") != ""
    )


def _filter_options(values: Iterable[str]) -> str:
    return "".join(
        f'<option value="{escape(value)}">{escape(value)}</option>'
        for value in sorted({value for value in values if value})
    )


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


def _mapping_value(value: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


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


def _research_axis_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {
        axis: [] for axis in QA_RESEARCH_AXIS_NAMES
    }
    for case in cases:
        attribution = case.get("error_attribution")
        if not isinstance(attribution, Mapping):
            continue
        for axis in _case_research_axes(case):
            if axis in grouped:
                grouped[axis].append(attribution)
    return {axis: _attribution_summary(grouped[axis]) for axis in QA_RESEARCH_AXIS_NAMES}


def _attribution_summary(attributions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "answer_correct_count": _bool_count(attributions, "answer_correct"),
        "by_error_category": _sorted_counts(
            str(attribution.get("error_category")) for attribution in attributions
        ),
        "by_evidence_error_category": _sorted_counts(
            str(attribution.get("evidence_error_category")) for attribution in attributions
        ),
        "by_predicted_evidence_source": _attribution_source_summary(attributions),
        "case_count": len(attributions),
        "error_count": sum(
            1
            for attribution in attributions
            if attribution.get("error_category") != "correct"
        ),
        "oracle_graph_tool_correct_count": _bool_count(
            attributions,
            "oracle_graph_tool_correct",
        ),
        "predicted_graph_tool_correct_count": _bool_count(
            attributions,
            "predicted_graph_tool_correct",
        ),
    }


def _attribution_source_summary(
    attributions: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for attribution in attributions:
        sources = _predicted_evidence_sources(attribution)
        if not sources:
            sources = ["missing_predicted_evidence"]
        for source in sources:
            grouped.setdefault(source, []).append(attribution)
    return {
        source: {
            "by_error_category": _sorted_counts(
                str(attribution.get("error_category"))
                for attribution in source_attributions
            ),
            "by_evidence_error_category": _sorted_counts(
                str(attribution.get("evidence_error_category"))
                for attribution in source_attributions
            ),
            "case_count": len(source_attributions),
            "error_count": sum(
                1
                for attribution in source_attributions
                if attribution.get("error_category") != "correct"
            ),
        }
        for source, source_attributions in sorted(grouped.items())
    }


def _bool_count(attributions: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(1 for attribution in attributions if attribution.get(key) is True)


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
