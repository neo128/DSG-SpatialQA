#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab.benchmark.active_qa_v2 import (
    active_qa_v2_quality_report,
    build_active_qa_v2_splits,
    build_active_qa_v2_vlm_request_bundle,
    save_active_qa_v2_splits,
)
from dsg_spatialqa_lab.observations import load_scene_observation_sequence
from dsg_spatialqa_lab.scene_io import load_graph_json
from dsg_spatialqa_lab.schema import SpatialQAError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Build active-exploration QA v2 splits from reachable NBV artifacts.",
    )
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--observation-sequence", type=Path, required=True)
    parser.add_argument("--predicted-graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--vlm-request-bundle", type=Path)
    args = parser.parse_args(argv)

    try:
        trajectory = _load_json(args.trajectory)
        observations = load_scene_observation_sequence(args.observation_sequence)
        graph = load_graph_json(args.predicted_graph)
        splits = build_active_qa_v2_splits(
            episode_id=args.episode_id,
            scene_id=args.scene_id,
            trajectory=trajectory,
            observations=observations,
            graph=graph,
        )
        split_paths = save_active_qa_v2_splits(splits, args.output_dir)
        report = active_qa_v2_quality_report(
            episode_id=args.episode_id,
            splits=splits,
        )
        report["split_paths"] = split_paths
        report["source_paths"] = {
            "trajectory": str(args.trajectory),
            "observation_sequence": str(args.observation_sequence),
            "predicted_graph": str(args.predicted_graph),
        }
        _save_json(args.quality_report, report)
        request_bundle_path = None
        request_bundle = None
        if args.vlm_request_bundle is not None:
            request_records = (
                list(splits.get("observation_aware", ()))
                + list(splits.get("relation_centric", ()))
                + list(splits.get("situated", ()))
                + list(splits.get("temporal", ()))
            )
            request_bundle = build_active_qa_v2_vlm_request_bundle(
                episode_id=args.episode_id,
                records=request_records,
            )
            _save_json(args.vlm_request_bundle, request_bundle)
            request_bundle_path = str(args.vlm_request_bundle)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit(
            {
                "action": "build_active_exploration_qa_v2",
                "valid": False,
                "error": str(exc),
            }
        )
        return 1

    valid = report["valid"] is True and (
        request_bundle is None or request_bundle.get("leak_free") is True
    )
    _emit(
        {
            "action": "build_active_exploration_qa_v2",
            "episode_id": args.episode_id,
            "valid": valid,
            "split_paths": split_paths,
            "quality_report": str(args.quality_report),
            "vlm_request_bundle": request_bundle_path,
            "summary": report["summary"],
        }
    )
    return 0 if valid else 1


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError(f"{path} must contain a JSON object")
    return payload


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
