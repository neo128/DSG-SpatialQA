#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = "dsg-spatialqa-lab.observation-segmentation-color-recovery-report.v1"
RECOVERY_SOURCE = "episode_jsonl_segmentation_color_map"
SEGMENTATION_SOURCE = "ai2thor_instance_segmentation_frame"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Recover observation object segmentation colors from explicit local "
            "episode JSONL segmentation_color_map records."
        ),
    )
    parser.add_argument("--episode-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--observation-sequence", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        colors_by_episode = _load_episode_color_maps(args.episode_jsonl)
        report = _recover_sequences(
            observation_paths=args.observation_sequence,
            colors_by_episode=colors_by_episode,
            output_root=args.output_root,
            episode_jsonls=args.episode_jsonl,
        )
        _write_json(args.report, report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "blockers": ["segmentation_color_recovery_error"],
            "error": str(exc),
            "ready": False,
        }
        report["report_digest"] = _digest_without(report, "report_digest")
        _write_json(args.report, report)
        _emit(report)
        return 1

    _emit(
        {
            "action": "recover_observation_segmentation_colors",
            "blockers": report["blockers"],
            "ready": report["ready"],
            "recovered_object_count": report["summary"]["recovered_object_count"],
            "report": str(args.report),
        }
    )
    return 0


def _load_episode_color_maps(paths: list[Path]) -> dict[str, dict[str, list[int]]]:
    by_episode: dict[str, dict[str, list[int]]] = {}
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"episode JSONL row must be an object: {path}:{line_number}")
            episode_id = payload.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError(f"episode JSONL row missing episode_id: {path}:{line_number}")
            metadata = payload.get("metadata")
            color_map = metadata.get("segmentation_color_map") if isinstance(metadata, dict) else None
            if not isinstance(color_map, list):
                continue
            episode_colors = by_episode.setdefault(episode_id, {})
            for item in color_map:
                if not isinstance(item, dict):
                    continue
                object_id = item.get("object_id")
                rgb = _as_rgb(item.get("rgb"))
                if isinstance(object_id, str) and rgb is not None:
                    episode_colors.setdefault(object_id, rgb)
    return by_episode


def _recover_sequences(
    *,
    observation_paths: list[Path],
    colors_by_episode: dict[str, dict[str, list[int]]],
    output_root: Path,
    episode_jsonls: list[Path],
) -> dict[str, Any]:
    summary: Counter[str] = Counter()
    sequence_reports: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    for path in observation_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"observation sequence must be a JSON object: {path}")
        observations = payload.get("observations")
        if not isinstance(observations, list):
            raise ValueError(f"observation sequence missing observations list: {path}")
        episode_id = _episode_id_for_sequence(path, observations)
        episode_colors = _colors_for_episode(episode_id, colors_by_episode)
        sequence_counts: Counter[str] = Counter()
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            objects = observation.get("objects")
            if not isinstance(objects, list):
                continue
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                sequence_counts["object_count"] += 1
                _recover_object_color(obj, episode_colors, sequence_counts)
        output_path = output_root / path.name
        _write_json(output_path, payload)
        for key, value in sequence_counts.items():
            summary[key] += value
        sequence_reports.append(
            {
                "already_colored_object_count": sequence_counts[
                    "already_colored_object_count"
                ],
                "color_map_object_count": len(episode_colors),
                "episode_id": episode_id,
                "object_count": sequence_counts["object_count"],
                "output_path": str(output_path),
                "recovered_object_count": sequence_counts["recovered_object_count"],
                "source_path": str(path),
                "unmatched_object_count": sequence_counts["unmatched_object_count"],
            }
        )
    blockers: list[str] = []
    if summary["recovered_object_count"] == 0 and summary["object_count"] > 0:
        blockers.append("no_segmentation_colors_recovered")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "blockers": blockers,
        "enriched_observation_sequences": sequence_reports,
        "episode_jsonls": [str(path) for path in episode_jsonls],
        "ready": True,
        "summary": {
            "already_colored_object_count": summary["already_colored_object_count"],
            "object_count": summary["object_count"],
            "recovered_object_count": summary["recovered_object_count"],
            "sequence_count": len(observation_paths),
            "unmatched_object_count": summary["unmatched_object_count"],
        },
    }
    report["report_digest"] = _digest_without(report, "report_digest")
    return report


def _recover_object_color(
    obj: dict[str, Any],
    episode_colors: dict[str, list[int]],
    counts: Counter[str],
) -> None:
    attributes = obj.get("attributes")
    if not isinstance(attributes, dict):
        attributes = {}
        obj["attributes"] = attributes
    if _rgb(attributes.get("segmentation_color_rgb")):
        counts["already_colored_object_count"] += 1
        return
    raw_object_id = attributes.get("ai2thor_object_id")
    if not isinstance(raw_object_id, str) or not raw_object_id:
        counts["unmatched_object_count"] += 1
        return
    rgb = episode_colors.get(raw_object_id)
    if rgb is None:
        counts["unmatched_object_count"] += 1
        return
    attributes["segmentation_color_recovery_source"] = RECOVERY_SOURCE
    attributes["segmentation_color_rgb"] = rgb
    attributes["segmentation_source"] = SEGMENTATION_SOURCE
    counts["recovered_object_count"] += 1


def _episode_id_for_sequence(path: Path, observations: list[Any]) -> str:
    for observation in observations:
        if isinstance(observation, dict):
            value = observation.get("episode_id")
            if isinstance(value, str) and value:
                return value
    match = re.search(r"episode[-_]?(\d{3})", path.name)
    if match is not None:
        return f"ai2thor-real-small-episode-{match.group(1)}"
    return path.stem


def _colors_for_episode(
    episode_id: str,
    colors_by_episode: dict[str, dict[str, list[int]]],
) -> dict[str, list[int]]:
    if episode_id in colors_by_episode:
        return colors_by_episode[episode_id]
    match = re.search(r"(\d{3})$", episode_id)
    if match is not None:
        suffix = match.group(1)
        for candidate in (
            f"ai2thor-real-small-episode-{suffix}",
            f"episode-{suffix}",
            f"episode{suffix}",
        ):
            if candidate in colors_by_episode:
                return colors_by_episode[candidate]
    return {}


def _rgb(value: object) -> bool:
    return _as_rgb(value) is not None


def _as_rgb(value: object) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return None
    if not all(0 <= item <= 255 for item in value):
        return None
    return list(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _digest_without(payload: dict[str, Any], key_to_omit: str) -> str:
    normalized = {key: value for key, value in payload.items() if key != key_to_omit}
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
