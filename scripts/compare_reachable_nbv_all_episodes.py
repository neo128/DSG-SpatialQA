#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from dsg_spatialqa_lab.navigation.trajectory_audit import save_json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_reachable_nbv_all_episodes import EPISODES  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Compare fixed vs real reachable NBV audits across real-small episodes.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("handoffs/ai2thor-real-small/outputs/navigation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("handoffs/ai2thor-real-small/outputs/navigation/reachable-nbv-all-episodes-comparison.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("handoffs/ai2thor-real-small/outputs/navigation/reachable-nbv-all-episodes-comparison.zh.md"),
    )
    args = parser.parse_args(argv)

    rows = [
        _episode_row(args.output_root, short_id, full_id, scene_id)
        for short_id, full_id, scene_id in EPISODES
    ]
    ready_count = sum(1 for row in rows if row.get("formal_protocol_ready") is True)
    report = {
        "schema_version": "dsg-spatialqa-lab.reachable-nbv-all-episodes-comparison.v1",
        "episode_count": len(rows),
        "formal_protocol_ready_episode_count": ready_count,
        "all_episodes_formal_protocol_ready": ready_count == len(rows),
        "fixed_trajectory_still_insufficient": any(
            _metric(row, "fixed", "evidence_observable_qa_count")
            < _metric(row, "nbv", "evidence_observable_qa_count")
            for row in rows
        ),
        "episodes": rows,
    }
    save_json(report, args.output)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_markdown(report), encoding="utf-8")
    _emit(
        {
            "action": "compare_reachable_nbv_all_episodes",
            "output": str(args.output),
            "markdown_output": str(args.markdown_output),
            "formal_protocol_ready_episode_count": ready_count,
            "all_episodes_formal_protocol_ready": report["all_episodes_formal_protocol_ready"],
        }
    )
    return 0 if report["all_episodes_formal_protocol_ready"] is True else 1


def _episode_row(output_root: Path, short_id: str, full_id: str, scene_id: str) -> dict[str, Any]:
    fixed_path = _first_existing(
        output_root / f"trajectory-audit-fixed-{full_id}.json",
        output_root / f"trajectory-audit-fixed-{short_id}.json",
    )
    nbv_path = output_root / f"trajectory-audit-real-ai2thor-reachable-nbv-{short_id}.json"
    gate_path = output_root / f"reachable-nbv-formal-gate-{short_id}.json"
    fixed = _load_json_if_exists(fixed_path)
    nbv = _load_json_if_exists(nbv_path)
    gate = _load_json_if_exists(gate_path)
    return {
        "episode_id": full_id,
        "short_id": short_id,
        "scene_id": scene_id,
        "formal_protocol_ready": gate.get("formal_protocol_ready", False),
        "failed_checks": gate.get("failed_checks", ["missing_gate_report"]),
        "fixed": _metrics(fixed),
        "nbv": _metrics(nbv),
        "deltas": {
            "target_support_same_frame_rate": _number(nbv.get("target_support_same_frame_rate")) - _number(fixed.get("target_support_same_frame_rate")),
            "evidence_observable_qa_count": _number(nbv.get("evidence_observable_qa_count")) - _number(fixed.get("evidence_observable_qa_count")),
            "missing_support_count": _number(nbv.get("missing_support_count")) - _number(fixed.get("missing_support_count")),
            "missing_relation_count": _number(nbv.get("missing_relation_count")) - _number(fixed.get("missing_relation_count")),
            "GraphTool_semantic_match": _number(nbv.get("GraphTool_semantic_match")) - _number(fixed.get("GraphTool_semantic_match")),
        },
    }


def _metrics(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "navigation_validated",
        "real_ai2thor_runtime",
        "target_support_same_frame_rate",
        "evidence_observable_qa_count",
        "missing_support_count",
        "missing_relation_count",
        "GraphTool_semantic_match",
        "visited_grid_ratio",
    )
    return {key: payload.get(key) for key in keys}


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# reachable NBV 多 episode 协议对比",
        "",
        "## 总结",
        f"- episode_count: {report['episode_count']}",
        f"- formal_protocol_ready_episode_count: {report['formal_protocol_ready_episode_count']}",
        f"- all_episodes_formal_protocol_ready: {report['all_episodes_formal_protocol_ready']}",
        "",
        "## Episode 表",
        "",
        "| episode | scene | ready | same_frame fixed→NBV | evidence fixed→NBV | missing_support fixed→NBV | missing_relation fixed→NBV | GraphTool semantic fixed→NBV | failed_checks |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["episodes"]:
        fixed = row["fixed"]
        nbv = row["nbv"]
        lines.append(
            "| {episode} | {scene} | {ready} | {sf}→{nsf} | {ev}→{nev} | {ms}→{nms} | {mr}→{nmr} | {gt}→{ngt} | {failed} |".format(
                episode=row["short_id"],
                scene=row["scene_id"],
                ready=row["formal_protocol_ready"],
                sf=fixed.get("target_support_same_frame_rate"),
                nsf=nbv.get("target_support_same_frame_rate"),
                ev=fixed.get("evidence_observable_qa_count"),
                nev=nbv.get("evidence_observable_qa_count"),
                ms=fixed.get("missing_support_count"),
                nms=nbv.get("missing_support_count"),
                mr=fixed.get("missing_relation_count"),
                nmr=nbv.get("missing_relation_count"),
                gt=fixed.get("GraphTool_semantic_match"),
                ngt=nbv.get("GraphTool_semantic_match"),
                failed=",".join(row.get("failed_checks", ())) or "-",
            )
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "- ready=false 的 episode 不能作为正式多 episode 探索结论。",
            "- coverage diagnostic 仍只作为上限诊断，不作为 predicted DSG evidence。",
        ]
    )
    return "\n".join(lines) + "\n"


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _metric(row: dict[str, Any], group: str, key: str) -> float:
    return _number(row.get(group, {}).get(key))


def _number(value: Any) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
