from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    DynamicSceneGraph,
    compare_graph_file_to_fixture,
    graph_json_digest,
    graph_summary,
    load_graph_json,
    load_scene_fixture,
    save_graph_json,
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
        help="Explicit local path where exported scene graph JSON should be written.",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        help="Explicit local scene graph JSON file to load and validate.",
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
    args = parser.parse_args(argv)

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
            print(json.dumps(payload, indent=2, sort_keys=True), end="\n")
            return 1
        payload = {"action": "compare_fixture", "valid": True, **comparison}
        print(json.dumps(payload, indent=2, sort_keys=True), end="\n")
        return 0 if comparison["matches"] is True else 1

    if args.validate is not None:
        try:
            graph = load_graph_json(args.validate)
        except (OSError, ValueError) as exc:
            payload = _scene_error_report("validate_graph", path=args.validate, error=exc)
            print(json.dumps(payload, indent=2, sort_keys=True), end="\n")
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

    print(json.dumps(payload, indent=2, sort_keys=True), end="\n")
    return 0


def _scene_report(
    action: str,
    graph: DynamicSceneGraph,
    *,
    path: Path,
    fixture: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "action": action,
        "path": str(path),
        "digest": graph_json_digest(graph),
        "summary": graph_summary(graph),
    }
    if fixture is not None:
        report["fixture"] = fixture
    return report


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


if __name__ == "__main__":
    raise SystemExit(main())
