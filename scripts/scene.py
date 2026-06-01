from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    DynamicSceneGraph,
    SpatialQAError,
    compare_graph_report,
    compare_graph_report_to_file,
    compare_scene_fixture_manifest,
    compare_graph_file_to_fixture,
    graph_report,
    load_graph_report,
    load_graph_json,
    load_scene_fixture,
    load_scene_fixture_manifest,
    save_graph_json,
    scene_fixture_manifest,
    scene_fixture_manifest_json,
    validate_graph_report,
    validate_scene_fixture_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export or validate deterministic DSG-SpatialQA scene graph JSON."
    )
    parser.add_argument(
        "--fixture",
        help="Built-in scene fixture name to export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Explicit local path where exported scene graph JSON or fixture metadata JSON "
            "should be written."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Explicit local path where the structured JSON payload printed to stdout is written.",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        help="Explicit local scene graph JSON file to load and validate.",
    )
    parser.add_argument(
        "--validate-report",
        type=Path,
        help="Validate an explicit local scene graph report JSON file.",
    )
    parser.add_argument(
        "--compare-report",
        type=Path,
        help="Compare an explicit local scene graph report JSON file with current fixture data.",
    )
    parser.add_argument(
        "--compare-report-graph",
        type=Path,
        help="Compare an explicit local scene graph report JSON file with --input graph JSON.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Explicit local scene graph JSON file to compare against a built-in fixture.",
    )
    parser.add_argument(
        "--compare-fixture",
        help="Built-in scene fixture name to compare with --input.",
    )
    parser.add_argument(
        "--list-fixtures",
        action="store_true",
        help="Emit filtered built-in scene fixture metadata without loading graphs.",
    )
    parser.add_argument(
        "--validate-fixture-manifest",
        type=Path,
        help="Validate an explicit local scene fixture metadata manifest file.",
    )
    parser.add_argument(
        "--compare-fixture-manifest",
        type=Path,
        help="Compare an explicit local scene fixture metadata manifest with current metadata.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        help="Require a built-in scene fixture tag. May be repeated.",
    )
    args = parser.parse_args(argv)

    if args.list_fixtures:
        tags = tuple(args.tags or ())
        payload = scene_fixture_manifest(tags=tags)
        payload_json = scene_fixture_manifest_json(payload)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload_json, encoding="utf-8")
        _write_payload_json(payload_json, args.report)
        print(payload_json, end="")
        return 0

    if args.validate_fixture_manifest is not None:
        try:
            validation = validate_scene_fixture_manifest(
                load_scene_fixture_manifest(args.validate_fixture_manifest)
            )
        except (OSError, ValueError) as exc:
            payload = _scene_error_report(
                "validate_fixture_manifest",
                path=args.validate_fixture_manifest,
                error=exc,
            )
            _emit_json_payload(payload, args.report)
            return 1
        payload = {
            "action": "validate_fixture_manifest",
            "path": str(args.validate_fixture_manifest),
            **validation,
        }
        _emit_json_payload(payload, args.report)
        return 0 if validation["valid"] is True else 1

    if args.compare_fixture_manifest is not None:
        try:
            comparison = compare_scene_fixture_manifest(
                load_scene_fixture_manifest(args.compare_fixture_manifest)
            )
        except (OSError, ValueError) as exc:
            payload = _scene_error_report(
                "compare_fixture_manifest",
                path=args.compare_fixture_manifest,
                error=exc,
                matches=False,
            )
            _emit_json_payload(payload, args.report)
            return 1
        payload = {
            "action": "compare_fixture_manifest",
            "path": str(args.compare_fixture_manifest),
            **comparison,
        }
        _emit_json_payload(payload, args.report)
        return 0 if comparison["matches"] is True else 1

    if args.validate_report is not None:
        try:
            validation = validate_graph_report(load_graph_report(args.validate_report))
        except (OSError, ValueError) as exc:
            payload = _scene_error_report(
                "validate_report",
                path=args.validate_report,
                error=exc,
            )
            _emit_json_payload(payload, args.report)
            return 1
        payload = {
            "action": "validate_report",
            "path": str(args.validate_report),
            **validation,
        }
        _emit_json_payload(payload, args.report)
        return 0 if validation["valid"] is True else 1

    if args.compare_report is not None:
        try:
            comparison = compare_graph_report(load_graph_report(args.compare_report))
        except (OSError, SpatialQAError, ValueError) as exc:
            payload = _scene_error_report(
                "compare_report",
                path=args.compare_report,
                error=exc,
                matches=False,
            )
            _emit_json_payload(payload, args.report)
            return 1
        payload = {
            "action": "compare_report",
            "path": str(args.compare_report),
            **comparison,
        }
        _emit_json_payload(payload, args.report)
        return 0 if comparison["matches"] is True else 1

    if args.compare_report_graph is not None:
        if args.input is None:
            parser.error("report graph comparison requires --compare-report-graph and --input")
        try:
            comparison = compare_graph_report_to_file(
                load_graph_report(args.compare_report_graph),
                args.input,
            )
        except (OSError, ValueError) as exc:
            payload = _scene_error_report(
                "compare_report_graph",
                path=args.input,
                error=exc,
                matches=False,
            )
            payload["report_path"] = str(args.compare_report_graph)
            _emit_json_payload(payload, args.report)
            return 1
        payload = {
            "action": "compare_report_graph",
            "report_path": str(args.compare_report_graph),
            **comparison,
        }
        _emit_json_payload(payload, args.report)
        return 0 if comparison["matches"] is True else 1

    if args.compare_fixture is not None:
        if args.input is None:
            parser.error("compare requires --compare-fixture and --input")
        try:
            comparison = compare_graph_file_to_fixture(args.input, args.compare_fixture)
        except (OSError, ValueError) as exc:
            payload = _scene_error_report(
                "compare_fixture",
                path=args.input,
                error=exc,
                fixture=args.compare_fixture,
                matches=False,
            )
            _emit_json_payload(payload, args.report)
            return 1
        payload = {"action": "compare_fixture", "valid": True, **comparison}
        _emit_json_payload(payload, args.report)
        return 0 if comparison["matches"] is True else 1

    if args.validate is not None:
        try:
            graph = load_graph_json(args.validate)
        except (OSError, ValueError) as exc:
            payload = _scene_error_report("validate_graph", path=args.validate, error=exc)
            _emit_json_payload(payload, args.report)
            return 1
        payload = _scene_report("validate_graph", graph, path=args.validate)
        payload["valid"] = True
    else:
        if args.fixture is None or args.output is None:
            parser.error("export requires --fixture and --output")
        graph = load_scene_fixture(args.fixture)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_graph_json(graph, args.output)
        payload = _scene_report(
            "export_fixture",
            graph,
            path=args.output,
            fixture=args.fixture,
        )

    _emit_json_payload(payload, args.report)
    return 0


def _scene_report(
    action: str,
    graph: DynamicSceneGraph,
    *,
    path: Path,
    fixture: str | None = None,
) -> dict[str, Any]:
    return graph_report(graph, action=action, graph_path=path, fixture=fixture)


def _scene_error_report(
    action: str,
    *,
    path: Path,
    error: Exception,
    fixture: str | None = None,
    matches: bool | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "action": action,
        "path": str(path),
        "valid": False,
        "error": str(error),
    }
    if fixture is not None:
        report["fixture"] = fixture
    if matches is not None:
        report["matches"] = matches
    return report


def _emit_json_payload(payload: dict[str, Any], report_path: Path | None = None) -> None:
    payload_json = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _write_payload_json(payload_json, report_path)
    print(payload_json, end="")


def _write_payload_json(payload_json: str, report_path: Path | None = None) -> None:
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload_json, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
