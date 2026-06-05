from __future__ import annotations

import argparse
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab import (
    SpatialQAError,
    dsg_viewer_html,
    dsg_viewer_payload,
    dsg_viewer_payload_json,
    dsg_viewer_resolve_workspace_path,
    dsg_viewer_workspace_preset,
    load_graph_eval_report,
    load_graph_json,
    load_qa_dataset,
    load_qa_eval_report,
    save_dsg_viewer_payload,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Serve a read-only local DSG viewer from explicit artifact paths.",
    )
    parser.add_argument("--workspace", type=Path, help="Optional local workspace root.")
    parser.add_argument("--graph", type=Path, help="Predicted DSG graph JSON path.")
    parser.add_argument("--oracle-graph", type=Path, help="Optional oracle graph JSON path.")
    parser.add_argument("--qa", type=Path, help="Optional QA dataset JSONL path.")
    parser.add_argument(
        "--qa-eval-report",
        type=Path,
        help="Optional QA eval report JSON path.",
    )
    parser.add_argument(
        "--graph-eval-report",
        type=Path,
        help="Optional graph eval report JSON path.",
    )
    parser.add_argument(
        "--evidence-report",
        type=Path,
        help="Optional predicted DSG evidence report JSON path.",
    )
    parser.add_argument(
        "--write-payload",
        type=Path,
        help="Write payload JSON to this path and exit without starting a server.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    try:
        paths = _input_paths(args)
        graph_path = paths.get("predicted_graph_path")
        if graph_path is None:
            parser.error("--graph is required when no workspace preset graph exists")
        payload = dsg_viewer_payload(
            predicted_graph=load_graph_json(graph_path),
            oracle_graph=_load_optional_graph(paths.get("oracle_graph_path")),
            qa_cases=(
                load_qa_dataset(paths["qa_path"])
                if paths.get("qa_path") is not None
                else ()
            ),
            qa_eval_report=_load_optional_qa_eval_report(
                paths.get("qa_eval_report_path")
            ),
            graph_eval_report=_load_optional_graph_eval_report(
                paths.get("graph_eval_report_path")
            ),
            evidence_report=_load_optional_json_mapping(
                paths.get("evidence_report_path")
            ),
            predicted_graph_path=graph_path,
            evidence_report_path=paths.get("evidence_report_path"),
        )
        if args.write_payload is not None:
            save_dsg_viewer_payload(payload, args.write_payload)
            _emit_json(
                {
                    "action": "write_dsg_viewer_payload",
                    "payload_path": str(args.write_payload),
                    "payload_digest": payload["payload_digest"],
                }
            )
            return 0
        _serve(payload, host=args.host, port=args.port)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("serve_dsg_viewer", exc))
        return 1
    return 0


def _input_paths(args: argparse.Namespace) -> dict[str, Path]:
    preset_paths: dict[str, Path] = {}
    if args.workspace is not None:
        preset = dsg_viewer_workspace_preset(args.workspace)
        preset_mapping = _mapping_or_empty(preset.get("paths"))
        preset_paths = {
            key: Path(value)
            for key, value in preset_mapping.items()
            if isinstance(key, str) and isinstance(value, str)
        }
    explicit_paths = {
        "predicted_graph_path": args.graph,
        "oracle_graph_path": args.oracle_graph,
        "qa_path": args.qa,
        "qa_eval_report_path": args.qa_eval_report,
        "graph_eval_report_path": args.graph_eval_report,
        "evidence_report_path": args.evidence_report,
    }
    paths = dict(preset_paths)
    for key, path in explicit_paths.items():
        if path is not None:
            paths[key] = _resolve_optional_workspace_path(args.workspace, path)
    return paths


def _resolve_optional_workspace_path(
    workspace: Path | None,
    path: Path,
) -> Path:
    if workspace is None:
        return path
    return dsg_viewer_resolve_workspace_path(workspace, path)


def _load_optional_graph(path: Path | None) -> Any:
    if path is None:
        return None
    return load_graph_json(path)


def _load_optional_qa_eval_report(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    return load_qa_eval_report(path)


def _load_optional_graph_eval_report(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    return load_graph_eval_report(path)


def _load_optional_json_mapping(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(f"JSON artifact must be an object: {path}")
    return cast(Mapping[str, Any], payload)


def _serve(payload: Mapping[str, Any], *, host: str, port: int) -> None:
    payload_json = dsg_viewer_payload_json(payload).encode("utf-8")
    html = _viewer_html().encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                _write_response(self, html, content_type="text/html; charset=utf-8")
                return
            if self.path == "/payload.json":
                _write_response(
                    self,
                    payload_json,
                    content_type="application/json; charset=utf-8",
                )
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    _emit_json(
        {
            "action": "serve_dsg_viewer",
            "url": f"http://{host}:{server.server_port}/",
        }
    )
    server.serve_forever()


def _viewer_html() -> str:
    return dsg_viewer_html()


def _write_response(
    handler: BaseHTTPRequestHandler,
    body: bytes,
    *,
    content_type: str,
) -> None:
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _mapping_or_empty(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    return {}


def _error_payload(action: str, error: Exception) -> dict[str, Any]:
    return {
        "action": action,
        "valid": False,
        "error": str(error),
    }


def _emit_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
