from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.memory import CONTAINMENT_RELATIONS, DynamicSceneGraph
from dsg_spatialqa_lab.scene_io import graph_summary
from dsg_spatialqa_lab.schema import Edge, Node, SpatialQAError


DSG_VIEWER_PAYLOAD_SCHEMA_VERSION = "dsg-spatialqa-lab.dsg-viewer-payload.v1"


def dsg_viewer_payload(
    *,
    predicted_graph: DynamicSceneGraph,
    predicted_graph_path: str | Path | None = None,
    evidence_report: Mapping[str, Any] | None = None,
    evidence_report_path: str | Path | None = None,
) -> dict[str, Any]:
    summary = graph_summary(predicted_graph)
    readiness = _mapping_or_empty(
        evidence_report.get("readiness") if evidence_report is not None else None
    )
    payload: dict[str, Any] = {
        "schema_version": DSG_VIEWER_PAYLOAD_SCHEMA_VERSION,
        "artifacts": _artifact_paths(
            predicted_graph_path=predicted_graph_path,
            evidence_report_path=evidence_report_path,
        ),
        "graph": {
            "summary": summary,
            "nodes": [
                _node_row(node)
                for node in sorted(predicted_graph.nodes.values(), key=lambda item: item.id)
            ],
            "edges": [
                _edge_row(edge)
                for edge in sorted(predicted_graph.edges, key=_viewer_edge_sort_key)
            ],
        },
        "metrics": {
            "object_count": summary["object_count"],
            "edge_count": summary["edge_count"],
            "evidence_ready": readiness.get("ready"),
        },
        "diagnostics": {
            "failed_checks": _string_list(readiness.get("failed_checks")),
        },
        "evidence": {
            "report_digest": evidence_report.get("report_digest")
            if evidence_report is not None
            else None,
            "summary": evidence_report.get("evidence_summary")
            if evidence_report is not None
            else {},
        },
        "qa": {"cases": []},
        "oracle": {},
        "indexes": {"qa_case_ids_by_object_id": {}},
    }
    payload["payload_digest"] = dsg_viewer_payload_digest(payload)
    return payload


def dsg_viewer_payload_digest(payload: Mapping[str, Any]) -> str:
    digest_payload = {
        key: value for key, value in payload.items() if key != "payload_digest"
    }
    return hashlib.sha256(
        json.dumps(digest_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def dsg_viewer_payload_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def save_dsg_viewer_payload(payload: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dsg_viewer_payload_json(payload), encoding="utf-8")
    return output_path


def load_dsg_viewer_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("DSG viewer payload JSON must be an object")
    return cast(dict[str, Any], payload)


def _node_row(node: Node) -> dict[str, Any]:
    return {
        "id": node.id,
        "type": node.type,
        "label": node.label,
        "attributes": dict(node.attributes),
    }


def _edge_row(edge: Edge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "src": edge.src,
        "relation": edge.relation,
        "dst": edge.dst,
        "reference_frame": edge.reference_frame,
        "confidence": edge.confidence,
        "step": edge.step,
        "evidence": list(edge.evidence),
        "attributes": dict(edge.attributes),
    }


def _viewer_edge_sort_key(edge: Edge) -> tuple[int, int, str, str, str, str]:
    relation_rank = 0 if edge.relation in CONTAINMENT_RELATIONS else 1
    return (
        relation_rank,
        edge.step,
        edge.src,
        edge.relation,
        edge.dst,
        edge.reference_frame,
    )


def _artifact_paths(**paths: str | Path | None) -> dict[str, str]:
    return {
        key: str(path)
        for key, path in sorted(paths.items())
        if path is not None
    }


def _mapping_or_empty(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    return {}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, str)]
