from __future__ import annotations

import argparse
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import struct
from typing import Any, cast
import zlib

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
from dsg_spatialqa_lab.benchmark.active_qa_v2 import load_active_qa_v2_records


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
        "--trajectory",
        type=Path,
        help="Optional reachable NBV trajectory report JSON path.",
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
                _load_qa_cases(paths["qa_path"])
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
            trajectory_report=_load_optional_json_mapping(
                paths.get("trajectory_report_path")
            ),
            predicted_graph_path=graph_path,
            evidence_report_path=paths.get("evidence_report_path"),
            trajectory_report_path=paths.get("trajectory_report_path"),
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
        _serve(payload, host=args.host, port=args.port, workspace=args.workspace)
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
        "trajectory_report_path": args.trajectory,
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


def _load_qa_cases(path: Path) -> list[Any]:
    if path.is_dir():
        rows: list[Any] = []
        for split_path in sorted(path.glob("qa-*.jsonl")):
            rows.extend(load_active_qa_v2_records(split_path))
        return rows
    return list(load_qa_dataset(path))


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


def _serve(
    payload: Mapping[str, Any],
    *,
    host: str,
    port: int,
    workspace: Path | None = None,
) -> None:
    payload_json = dsg_viewer_payload_json(payload).encode("utf-8")
    html = _viewer_html().encode("utf-8")
    workspace_path = workspace.resolve() if workspace is not None else None

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
            if self.path.startswith("/asset?"):
                try:
                    body, content_type = _asset_response(self.path, workspace_path)
                except (OSError, SpatialQAError, ValueError) as exc:
                    self.send_error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                _write_response(self, body, content_type=content_type)
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


def _asset_response(request_path: str, workspace: Path | None) -> tuple[bytes, str]:
    if workspace is None:
        raise SpatialQAError("Asset serving requires --workspace")
    route, _, query = request_path.partition("?")
    if route != "/asset":
        raise SpatialQAError(f"Unsupported asset route: {route}")
    values = _query_values(query, "path")
    if not values:
        raise SpatialQAError("Asset route requires path")
    asset_path = _resolve_workspace_asset_path(workspace, values[0])
    suffix = asset_path.suffix.lower()
    if suffix == ".ppm":
        return _ppm_file_to_png(asset_path), "image/png"
    return asset_path.read_bytes(), _asset_content_type(asset_path)


def _resolve_workspace_asset_path(workspace: Path, path_text: str) -> Path:
    workspace_path = workspace.resolve()
    candidate = Path(path_text)
    candidates = (
        candidate if candidate.is_absolute() else workspace_path / candidate,
        candidate if candidate.is_absolute() else candidate.resolve(),
    )
    blocked = False
    for item in candidates:
        resolved = item.resolve()
        try:
            resolved.relative_to(workspace_path)
        except ValueError:
            blocked = True
            continue
        if resolved.exists() and resolved.is_file():
            return resolved
    if blocked:
        raise SpatialQAError(f"Asset path is outside workspace: {path_text}")
    raise SpatialQAError(f"Asset path does not exist: {path_text}")


def _query_values(query: str, key: str) -> tuple[str, ...]:
    values: list[str] = []
    for pair in query.split("&"):
        if pair == "":
            continue
        name, _, value = pair.partition("=")
        if _percent_decode(name) == key:
            values.append(_percent_decode(value))
    return tuple(values)


def _percent_decode(text: str) -> str:
    data = bytearray()
    index = 0
    while index < len(text):
        char = text[index]
        if (
            char == "%"
            and index + 2 < len(text)
            and _is_hex_pair(text[index + 1 : index + 3])
        ):
            data.append(int(text[index + 1 : index + 3], 16))
            index += 3
            continue
        if char == "+":
            data.append(ord(" "))
        else:
            data.extend(char.encode("utf-8"))
        index += 1
    return data.decode("utf-8")


def _is_hex_pair(text: str) -> bool:
    if len(text) != 2:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in text)


def _asset_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".gif":
        return "image/gif"
    return "application/octet-stream"


def _ppm_file_to_png(path: Path) -> bytes:
    width, height, rgb = _read_ppm_rgb(path.read_bytes())
    return _png_rgb(width, height, rgb)


def _read_ppm_rgb(data: bytes) -> tuple[int, int, bytes]:
    token, index = _ppm_token(data, 0)
    if token != b"P6":
        raise SpatialQAError("Only binary P6 PPM assets are supported")
    width_token, index = _ppm_token(data, index)
    height_token, index = _ppm_token(data, index)
    max_value_token, index = _ppm_token(data, index)
    width = int(width_token)
    height = int(height_token)
    max_value = int(max_value_token)
    if width <= 0 or height <= 0:
        raise SpatialQAError("PPM dimensions must be positive")
    if max_value != 255:
        raise SpatialQAError("Only 8-bit PPM assets are supported")
    if index < len(data) and data[index] in b" \t\r\n":
        index += 1
    byte_count = width * height * 3
    rgb = data[index:index + byte_count]
    if len(rgb) != byte_count:
        raise SpatialQAError("PPM asset ended before all RGB bytes were read")
    return width, height, rgb


def _ppm_token(data: bytes, start: int) -> tuple[bytes, int]:
    index = start
    while index < len(data):
        value = data[index]
        if value == ord("#"):
            while index < len(data) and data[index] not in b"\r\n":
                index += 1
            continue
        if value not in b" \t\r\n":
            break
        index += 1
    end = index
    while end < len(data) and data[end] not in b" \t\r\n":
        end += 1
    if end == index:
        raise SpatialQAError("PPM asset header is incomplete")
    return data[index:end], end


def _png_rgb(width: int, height: int, rgb: bytes) -> bytes:
    row_size = width * 3
    scanlines = b"".join(
        b"\x00" + rgb[row_start:row_start + row_size]
        for row_start in range(0, len(rgb), row_size)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(scanlines))
        + _png_chunk(b"IEND", b"")
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum)
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", checksum & 0xFFFFFFFF)
    )


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
