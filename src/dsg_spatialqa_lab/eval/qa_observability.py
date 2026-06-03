from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.benchmark import QACase, load_qa_dataset, qa_dataset_digest
from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.scene_io import graph_json_digest, load_graph_json
from dsg_spatialqa_lab.schema import SpatialQAError


QA_OBSERVABILITY_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.qa-observability-report.v1"
)
QA_OBSERVABILITY_SPLITS = (
    "evidence_observable",
    "target_observable",
    "target_observable_relation_missing",
    "target_missing",
    "missing_evidence",
)


def qa_observability_report(
    cases: Sequence[QACase],
    graph: DynamicSceneGraph,
    *,
    qa_path: str | Path | None = None,
    graph_path: str | Path | None = None,
) -> dict[str, Any]:
    case_results = [_case_observability(case, graph) for case in cases]
    splits = {
        split: [
            result["case_id"]
            for result in case_results
            if _result_in_split(result, split)
        ]
        for split in QA_OBSERVABILITY_SPLITS
    }
    summary = {
        "case_count": len(case_results),
        "evidence_observable_count": len(splits["evidence_observable"]),
        "missing_evidence_count": len(splits["missing_evidence"]),
        "target_observable_count": len(splits["target_observable"]),
        "target_observable_relation_missing_count": len(
            splits["target_observable_relation_missing"]
        ),
        "target_missing_count": len(splits["target_missing"]),
    }
    report: dict[str, Any] = {
        "schema_version": QA_OBSERVABILITY_REPORT_SCHEMA_VERSION,
        "qa_path": str(qa_path) if qa_path is not None else None,
        "graph_path": str(graph_path) if graph_path is not None else None,
        "qa_digest": qa_dataset_digest(cases),
        "graph_digest": graph_json_digest(graph),
        "summary": summary,
        "splits": splits,
        "cases": case_results,
    }
    report["report_digest"] = qa_observability_report_digest(report)
    return report


def qa_observability_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def qa_observability_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_qa_observability_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(qa_observability_report_json(report), encoding="utf-8")
    return output_path


def load_qa_observability_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("QA observability report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_qa_observability_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    report_digest = report.get("report_digest")
    expected_digest = qa_observability_report_digest(report)
    cases_value = report.get("cases")
    cases = (
        cast(Sequence[Mapping[str, Any]], cases_value)
        if isinstance(cases_value, Sequence) and not isinstance(cases_value, str)
        else ()
    )
    splits_value = report.get("splits")
    splits = splits_value if isinstance(splits_value, Mapping) else {}
    summary = report.get("summary")
    expected_splits = {
        split: [
            str(result.get("case_id"))
            for result in cases
            if _result_in_split(result, split)
        ]
        for split in QA_OBSERVABILITY_SPLITS
    }
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == QA_OBSERVABILITY_REPORT_SCHEMA_VERSION,
            "expected": QA_OBSERVABILITY_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_digest,
            "expected": expected_digest,
            "actual": report_digest,
        },
        {
            "name": "cases_shape",
            "passed": bool(cases) or _summary_value(summary, "case_count") == 0,
        },
        {
            "name": "splits_match_cases",
            "passed": splits == expected_splits,
            "expected": expected_splits,
            "actual": splits,
        },
        {
            "name": "summary_matches_cases",
            "passed": _summary_matches_cases(summary, expected_splits, len(cases)),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks,
    }


def compare_qa_observability_report(report: Mapping[str, Any]) -> dict[str, Any]:
    qa_path = _required_report_path(report, "qa_path")
    graph_path = _required_report_path(report, "graph_path")
    current_report = qa_observability_report(
        load_qa_dataset(qa_path),
        load_graph_json(graph_path),
        qa_path=qa_path,
        graph_path=graph_path,
    )
    validation = validate_qa_observability_report(report)
    saved_digest = report.get("report_digest")
    current_digest = current_report["report_digest"]
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
        _equality_check("splits_match_current", report.get("splits"), current_report["splits"]),
        _equality_check("cases_match_current", report.get("cases"), current_report["cases"]),
    ]
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def qa_observability_split_ids(
    report: Mapping[str, Any],
    split_name: str,
) -> tuple[str, ...]:
    if split_name not in QA_OBSERVABILITY_SPLITS:
        raise SpatialQAError(f"Unsupported QA observability split: {split_name}")
    splits = report.get("splits")
    if not isinstance(splits, Mapping):
        raise SpatialQAError("QA observability report splits must be an object")
    split_value = splits.get(split_name)
    if not isinstance(split_value, Sequence) or isinstance(split_value, str):
        raise SpatialQAError(f"QA observability split must be a sequence: {split_name}")
    if not all(isinstance(item, str) for item in split_value):
        raise SpatialQAError(f"QA observability split ids must be strings: {split_name}")
    return tuple(cast(Sequence[str], split_value))


def filter_qa_cases_by_ids(
    cases: Sequence[QACase],
    case_ids: Sequence[str],
) -> tuple[QACase, ...]:
    selected = set(case_ids)
    return tuple(case for case in cases if case.id in selected)


def _case_observability(case: QACase, graph: DynamicSceneGraph) -> dict[str, Any]:
    node_ids = set(graph.nodes)
    edge_ids = {edge.id for edge in graph.edges}
    target_nodes = _target_nodes(case, graph)
    missing_target_nodes = [
        node_id for node_id in target_nodes if node_id not in node_ids
    ]
    missing_required_nodes = [
        node_id for node_id in case.required_nodes if node_id not in node_ids
    ]
    missing_required_edges = [
        edge_id for edge_id in case.required_edges if edge_id not in edge_ids
    ]
    target_observable = not missing_target_nodes
    required_nodes_present = not missing_required_nodes
    required_edges_present = not missing_required_edges
    evidence_observable = required_nodes_present and required_edges_present
    if evidence_observable:
        status = "evidence_observable"
    elif target_observable:
        status = "target_observable_relation_missing"
    else:
        status = "target_missing"
    return {
        "case_id": case.id,
        "question_type": case.question_type,
        "target_nodes": list(target_nodes),
        "target_observable": target_observable,
        "required_nodes_present": required_nodes_present,
        "required_edges_present": required_edges_present,
        "evidence_observable": evidence_observable,
        "observability_status": status,
        "missing_target_nodes": missing_target_nodes,
        "missing_required_nodes": missing_required_nodes,
        "missing_required_edges": missing_required_edges,
    }


def _target_nodes(case: QACase, graph: DynamicSceneGraph) -> tuple[str, ...]:
    explicit = list(_question_object_ids(case.question))
    if explicit:
        return tuple(dict.fromkeys(explicit))
    required_objects = [
        node_id for node_id in case.required_nodes if node_id in graph.object_states
    ]
    return tuple(dict.fromkeys(required_objects))


def _question_object_ids(value: object) -> tuple[str, ...]:
    ids: list[str] = []
    if isinstance(value, Mapping):
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if key in {"object_id", "src", "dst", "target_object"} and isinstance(item, str):
                ids.append(item)
                continue
            if key == "candidates" and isinstance(item, Sequence) and not isinstance(item, str):
                ids.extend(str(candidate) for candidate in item if isinstance(candidate, str))
                continue
            ids.extend(_question_object_ids(item))
    elif isinstance(value, Sequence) and not isinstance(value, str):
        for item in value:
            ids.extend(_question_object_ids(item))
    return tuple(ids)


def _result_in_split(result: Mapping[str, Any], split: str) -> bool:
    if split == "evidence_observable":
        return result.get("observability_status") == "evidence_observable"
    if split == "target_observable":
        return result.get("target_observable") is True
    if split == "target_observable_relation_missing":
        return (
            result.get("observability_status")
            == "target_observable_relation_missing"
        )
    if split == "target_missing":
        return result.get("observability_status") == "target_missing"
    if split == "missing_evidence":
        return result.get("evidence_observable") is not True
    raise SpatialQAError(f"Unsupported QA observability split: {split}")


def _summary_matches_cases(
    summary: object,
    splits: Mapping[str, Sequence[str]],
    case_count: int,
) -> bool:
    if not isinstance(summary, Mapping):
        return False
    expected = {
        "case_count": case_count,
        "evidence_observable_count": len(splits["evidence_observable"]),
        "missing_evidence_count": len(splits["missing_evidence"]),
        "target_observable_count": len(splits["target_observable"]),
        "target_observable_relation_missing_count": len(
            splits["target_observable_relation_missing"]
        ),
        "target_missing_count": len(splits["target_missing"]),
    }
    return all(summary.get(key) == value for key, value in expected.items())


def _summary_value(summary: object, key: str) -> int | None:
    if not isinstance(summary, Mapping):
        return None
    value = summary.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _required_report_path(report: Mapping[str, Any], field_name: str) -> str:
    value = report.get(field_name)
    if not isinstance(value, str) or not value:
        raise SpatialQAError(f"{field_name} must be a non-empty string")
    return value


def _equality_check(name: str, expected: object, actual: object) -> dict[str, Any]:
    check: dict[str, Any] = {
        "name": name,
        "passed": expected == actual,
        "expected": expected,
        "actual": actual,
    }
    return check
