#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from dsg_spatialqa_lab.navigation.trajectory_audit import (
    reachable_nbv_formal_protocol_gate,
    save_json,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_reachable_nbv_all_episodes import load_episode_plan  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run formal protocol gates for all real reachable NBV episode artifacts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("handoffs/ai2thor-real-small/outputs/navigation"),
    )
    parser.add_argument(
        "--episode-plan",
        type=Path,
        help="JSON plan with episodes [{short_id, episode_id, scene_id}].",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("handoffs/ai2thor-real-small/outputs/navigation/reachable-nbv-formal-gate-all-episodes.json"),
    )
    args = parser.parse_args(argv)

    episodes = load_episode_plan(args.episode_plan)
    rows = []
    for short_id, full_id, _scene_id in episodes:
        row = _audit_episode(args.output_root, short_id, full_id)
        rows.append(row)

    report = {
        "schema_version": "dsg-spatialqa-lab.reachable-nbv-formal-gate-all-episodes.v1",
        "episode_plan_path": str(args.episode_plan) if args.episode_plan is not None else None,
        "episode_count": len(rows),
        "formal_protocol_ready_episode_count": sum(
            1 for row in rows if row.get("formal_protocol_ready") is True
        ),
        "all_episodes_formal_protocol_ready": all(
            row.get("formal_protocol_ready") is True for row in rows
        ),
        "episodes": rows,
    }
    save_json(report, args.report)
    _emit({"action": "audit_reachable_nbv_all_episodes", "report": str(args.report), **report})
    return 0 if report["all_episodes_formal_protocol_ready"] is True else 1


def _audit_episode(output_root: Path, short_id: str, full_id: str) -> dict[str, Any]:
    paths = {
        "trajectory": output_root / f"reachable-nbv-real-ai2thor-trajectory-{short_id}.json",
        "decision_trace": output_root / f"reachable-nbv-real-ai2thor-decision-trace-{short_id}.jsonl",
        "fixed_audit": _first_existing(
            output_root / f"trajectory-audit-fixed-{full_id}.json",
            output_root / f"trajectory-audit-fixed-{short_id}.json",
        ),
        "nbv_audit": output_root / f"trajectory-audit-real-ai2thor-reachable-nbv-{short_id}.json",
        "gate": output_root / f"reachable-nbv-formal-gate-{short_id}.json",
    }
    missing = [name for name, path in paths.items() if name != "gate" and not path.exists()]
    if missing:
        row = {
            "episode_id": full_id,
            "short_id": short_id,
            "formal_protocol_ready": False,
            "failed_checks": [f"missing_{name}" for name in missing],
            "paths": {key: str(value) for key, value in paths.items()},
        }
        save_json(row, paths["gate"])
        return row
    trajectory = _load_json(paths["trajectory"])
    fixed_audit = _load_json(paths["fixed_audit"])
    nbv_audit = _load_json(paths["nbv_audit"])
    decisions = _load_jsonl(paths["decision_trace"])
    gate = reachable_nbv_formal_protocol_gate(
        episode_id=full_id,
        trajectory=trajectory,
        fixed_audit=fixed_audit,
        nbv_audit=nbv_audit,
        decisions=decisions,
    )
    gate["short_id"] = short_id
    gate["paths"] = {key: str(value) for key, value in paths.items()}
    save_json(gate, paths["gate"])
    return gate


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
