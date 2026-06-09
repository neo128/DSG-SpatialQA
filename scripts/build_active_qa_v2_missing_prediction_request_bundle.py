#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab.benchmark.active_qa_v2 import (
    build_active_qa_v2_vlm_request_bundle,
    load_active_qa_v2_records,
)


COMPARISON_SPLITS = {
    "qa-observation-aware.jsonl",
    "qa-relation-centric.jsonl",
    "qa-situated.jsonl",
    "qa-temporal.jsonl",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Build a leak-free active QA v2 request bundle for missing prediction cases.",
    )
    parser.add_argument(
        "--qa-root",
        type=Path,
        action="append",
        required=True,
        help="Active QA v2 root. May be supplied multiple times.",
    )
    parser.add_argument("--existing-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-method", default="vlm_only")
    args = parser.parse_args(argv)

    try:
        records = _load_comparison_records(args.qa_root)
        prediction_ids = _load_prediction_ids(args.existing_predictions)
        missing_records = [
            record
            for record in records
            if str(record.get("id")) not in prediction_ids
        ]
        bundle = build_active_qa_v2_vlm_request_bundle(
            episode_id="multiple",
            records=missing_records,
        )
        missing_by_episode = Counter(str(row.get("episode_id", "unknown")) for row in missing_records)
        bundle.update(
            {
                "bundle_kind": "missing_prediction_request_bundle",
                "existing_prediction_count": len(prediction_ids),
                "missing_by_episode": dict(sorted(missing_by_episode.items())),
                "missing_case_count": len(missing_records),
                "qa_case_count": len(records),
                "qa_roots": [str(root) for root in args.qa_root],
                "source_existing_predictions": str(args.existing_predictions),
                "target_method": args.target_method,
            }
        )
        leak_paths = _forbidden_paths(bundle)
        bundle["leak_free"] = not leak_paths
        bundle["leak_paths"] = leak_paths
        bundle["request_bundle_digest"] = _digest(bundle)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit(
            {
                "action": "build_active_qa_v2_missing_prediction_request_bundle",
                "error": str(exc),
                "ready": False,
            }
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = {
        "action": "build_active_qa_v2_missing_prediction_request_bundle",
        "leak_free": bundle["leak_free"],
        "missing_case_count": bundle["missing_case_count"],
        "output": str(args.output),
        "ready": bundle["leak_free"] is True,
        "request_count": bundle["request_count"],
        "target_method": args.target_method,
    }
    _emit(payload)
    return 0 if bundle["leak_free"] is True else 1


def _load_comparison_records(roots: list[Path]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for root in roots:
        for path in sorted(root.glob("*/qa-*.jsonl")):
            if path.name not in COMPARISON_SPLITS:
                continue
            for row in load_active_qa_v2_records(path):
                case_id = str(row.get("id"))
                rows[case_id] = row
    return [rows[key] for key in sorted(rows)]


def _load_prediction_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            ids.add(row["id"])
    return ids


def _forbidden_paths(value: object, *, path: str = "$") -> list[str]:
    forbidden = {
        "gold_answer",
        "gold_evidence",
        "required_edges",
        "required_nodes",
        "visible_object_ids",
        "visible_object_labels",
    }
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in forbidden:
                paths.append(child_path)
            paths.extend(_forbidden_paths(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_forbidden_paths(child, path=f"{path}[{index}]"))
    return paths


def _digest(payload: dict[str, Any]) -> str:
    normalized = {
        key: value
        for key, value in payload.items()
        if key != "request_bundle_digest"
    }
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
