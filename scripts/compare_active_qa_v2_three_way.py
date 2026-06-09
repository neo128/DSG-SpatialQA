#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab.benchmark.active_qa_v2 import load_active_qa_v2_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Compare VLM-only, GraphTool-only DSG, and VLM+DSG trusted on active QA v2.",
    )
    parser.add_argument(
        "--qa-root",
        type=Path,
        action="append",
        default=None,
        help="Active QA v2 root. May be supplied multiple times.",
    )
    parser.add_argument("--vlm-predictions", type=Path)
    parser.add_argument("--graph-predictions", type=Path)
    parser.add_argument("--vlm-dsg-predictions", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("handoffs/ai2thor-real-small/outputs/diagnostics"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("handoffs/ai2thor-real-small/outputs/diagnostics/three-way-comparison-active-qa-v2-all-episodes.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("handoffs/ai2thor-real-small/outputs/diagnostics/three-way-comparison-active-qa-v2-all-episodes.zh.md"),
    )
    parser.add_argument(
        "--required-episode-count",
        type=int,
        default=5,
        help="Minimum episode count needed for the requested comparison scope.",
    )
    args = parser.parse_args(argv)

    qa_roots = args.qa_root or [Path("handoffs/ai2thor-real-small/inputs/qa-v2-active")]
    records_by_episode = _load_records_by_episode(qa_roots)
    blockers: list[str] = []
    if not records_by_episode:
        blockers.append("missing_active_qa_v2_records")
    vlm_predictions = _load_predictions_if_exists(args.vlm_predictions)
    vlm_dsg_predictions = _load_predictions_if_exists(args.vlm_dsg_predictions)
    if args.vlm_predictions is None or not args.vlm_predictions.exists():
        blockers.append("missing_active_vlm_only_predictions")
    if args.vlm_dsg_predictions is None or not args.vlm_dsg_predictions.exists():
        blockers.append("missing_active_vlm_dsg_trusted_predictions")
    graph_predictions = (
        _load_predictions_if_exists(args.graph_predictions)
        if args.graph_predictions is not None
        else _derive_graph_predictions(records_by_episode)
    )

    episode_reports = []
    for episode_id, records in sorted(records_by_episode.items()):
        report = _episode_report(
            episode_id,
            records,
            vlm_predictions,
            graph_predictions,
            vlm_dsg_predictions,
        )
        episode_reports.append(report)
        path = args.output_dir / f"three-way-comparison-active-qa-v2-{_short_episode(episode_id)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    coverage = _prediction_coverage(
        records_by_episode,
        {
            "vlm_only": vlm_predictions,
            "graph_tool_only_dsg": graph_predictions,
            "vlm_dsg_trusted": vlm_dsg_predictions,
        },
    )
    all_report = _all_episode_report(
        episode_reports,
        blockers,
        qa_roots=qa_roots,
        prediction_coverage=coverage,
        required_episode_count=args.required_episode_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_markdown(all_report), encoding="utf-8")
    _emit(
        {
            "action": "compare_active_qa_v2_three_way",
            "ready": all_report["ready"],
            "research_ready": all_report["research_ready"],
            "blockers": all_report["blockers"],
            "output": str(args.output),
            "markdown_output": str(args.markdown_output),
        }
    )
    return 0 if all_report["ready"] is True else 1


def _load_records_by_episode(roots: list[Path]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    comparison_splits = {
        "qa-observation-aware.jsonl",
        "qa-relation-centric.jsonl",
        "qa-situated.jsonl",
        "qa-temporal.jsonl",
    }
    for root in roots:
        for path in sorted(root.glob("*/qa-*.jsonl")):
            if path.name not in comparison_splits:
                continue
            for row in load_active_qa_v2_records(path):
                episode_id = str(row.get("episode_id", "unknown"))
                grouped[episode_id].append(row)
    return {key: _dedupe(rows) for key, rows in grouped.items()}


def _load_predictions_if_exists(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        str(row.get("id")): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }


def _derive_graph_predictions(
    records_by_episode: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    predictions = {}
    for records in records_by_episode.values():
        for row in records:
            predictions[str(row["id"])] = {
                "id": row["id"],
                "answer": {
                    **dict(row.get("answer", {})),
                    "source": "graph_tool_only_dsg",
                    "prediction_source": "graph_tool_from_active_qa_graph_records",
                },
                "confidence": 1.0,
            }
    return predictions


def _episode_report(
    episode_id: str,
    records: list[dict[str, Any]],
    vlm_predictions: dict[str, dict[str, Any]],
    graph_predictions: dict[str, dict[str, Any]],
    vlm_dsg_predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    methods = {
        "vlm_only": _method_rows(records, vlm_predictions),
        "graph_tool_only_dsg": _method_rows(records, graph_predictions),
        "vlm_dsg_trusted": _method_rows(records, vlm_dsg_predictions),
    }
    deltas = {
        "vlm_dsg_vs_vlm_only": _paired_delta(methods["vlm_dsg_trusted"], methods["vlm_only"]),
        "vlm_dsg_vs_graph_tool_only": _paired_delta(methods["vlm_dsg_trusted"], methods["graph_tool_only_dsg"]),
        "graph_tool_only_vs_vlm_only": _paired_delta(methods["graph_tool_only_dsg"], methods["vlm_only"]),
    }
    return {
        "schema_version": "dsg-spatialqa-lab.three-way-active-qa-v2-episode-comparison.v1",
        "episode_id": episode_id,
        "case_count": len(records),
        "question_type_groups": _question_type_groups(records, methods),
        "methods": {name: _method_summary(rows) for name, rows in methods.items()},
        "deltas": deltas,
    }


def _method_rows(records: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        prediction = predictions.get(str(record.get("id")))
        semantic = _semantic_match(record, prediction)
        strict = _strict_match(record, prediction)
        rows.append(
            {
                "case_id": record.get("id"),
                "episode_id": record.get("episode_id"),
                "question_type": record.get("question_type"),
                "semantic_match": semantic,
                "strict_exact": strict,
                "has_prediction": prediction is not None,
            }
        )
    return rows


def _semantic_match(record: dict[str, Any], prediction: dict[str, Any] | None) -> bool:
    if prediction is None:
        return False
    gold = _answer(record.get("answer"))
    pred = _answer(prediction.get("answer"))
    if pred.get("relation") == gold.get("relation") and (
        pred.get("dst") == gold.get("dst") or pred.get("dst_label") == gold.get("dst_label")
    ):
        return True
    text = str(prediction.get("answer", {}).get("text", "")).lower()
    return bool(gold.get("dst_label")) and str(gold["dst_label"]).lower() in text


def _strict_match(record: dict[str, Any], prediction: dict[str, Any] | None) -> bool:
    if prediction is None:
        return False
    gold = _answer(record.get("answer"))
    pred = _answer(prediction.get("answer"))
    return pred.get("relation") == gold.get("relation") and pred.get("dst") == gold.get("dst")


def _answer(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    location = value.get("current_location")
    if isinstance(location, dict):
        merged = dict(value)
        merged.update(location)
        return merged
    return value


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    semantic = sum(1 for row in rows if row["semantic_match"] is True)
    strict = sum(1 for row in rows if row["strict_exact"] is True)
    has_prediction = sum(1 for row in rows if row["has_prediction"] is True)
    return {
        "case_count": len(rows),
        "prediction_count": has_prediction,
        "semantic_match_count": semantic,
        "semantic_match_rate": _ratio(semantic, len(rows)),
        "strict_exact_count": strict,
        "strict_exact_rate": _ratio(strict, len(rows)),
        "unknown_or_missing_count": len(rows) - has_prediction,
    }


def _paired_delta(candidate: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> dict[str, Any]:
    wins = losses = ties = 0
    for cand, base in zip(candidate, baseline, strict=True):
        if cand["semantic_match"] and not base["semantic_match"]:
            wins += 1
        elif base["semantic_match"] and not cand["semantic_match"]:
            losses += 1
        else:
            ties += 1
    p_value = _sign_test_p_value(wins, losses)
    return {
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": ties,
        "sign_test_p_value": p_value,
        "decision": (
            "significant_candidate_advantage"
            if wins > losses and p_value < 0.05
            else "directional_not_significant"
            if wins > losses
            else "no_candidate_advantage"
        ),
    }


def _question_type_groups(
    records: list[dict[str, Any]],
    methods: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_type: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_type[str(record.get("question_type"))].append(index)
    groups = {}
    for qtype, indexes in sorted(by_type.items()):
        groups[qtype] = {
            name: _method_summary([rows[index] for index in indexes])
            for name, rows in methods.items()
        }
    return groups


def _all_episode_report(
    episode_reports: list[dict[str, Any]],
    blockers: list[str],
    *,
    qa_roots: list[Path],
    prediction_coverage: dict[str, Any],
    required_episode_count: int,
) -> dict[str, Any]:
    total_cases = sum(report["case_count"] for report in episode_reports)
    episode_count = len(episode_reports)
    question_types = sorted(
        {
            qtype
            for report in episode_reports
            for qtype in report["question_type_groups"]
        }
    )
    aggregate_methods = {
        method: _aggregate_method(episode_reports, method)
        for method in ("vlm_only", "graph_tool_only_dsg", "vlm_dsg_trusted")
    }
    aggregate_delta = _aggregate_delta(episode_reports, "vlm_dsg_vs_vlm_only")
    conclusion_blockers = list(blockers)
    for method, coverage in prediction_coverage.items():
        if coverage["missing_case_count"] > 0:
            conclusion_blockers.append(f"{method}_prediction_coverage_incomplete")
    if episode_count < required_episode_count:
        conclusion_blockers.append(f"episode_count_lt_{required_episode_count}")
    if len(question_types) < 3:
        conclusion_blockers.append("question_type_count_lt_3")
    if aggregate_methods["vlm_dsg_trusted"]["semantic_match_rate"] <= aggregate_methods["vlm_only"]["semantic_match_rate"]:
        conclusion_blockers.append("vlm_dsg_not_above_vlm_only")
    if aggregate_delta["paired_wins"] <= aggregate_delta["paired_losses"]:
        conclusion_blockers.append("paired_wins_not_above_losses")
    if aggregate_delta["sign_test_p_value"] >= 0.05:
        conclusion_blockers.append("directional_not_significant")
    return {
        "schema_version": "dsg-spatialqa-lab.three-way-active-qa-v2-all-episodes-comparison.v1",
        "ready": not conclusion_blockers and total_cases > 0,
        "research_ready": not conclusion_blockers,
        "blockers": sorted(set(conclusion_blockers)),
        "episode_count": episode_count,
        "case_count": total_cases,
        "question_type_count": len(question_types),
        "question_types": question_types,
        "qa_roots": [str(root) for root in qa_roots],
        "required_episode_count": required_episode_count,
        "prediction_coverage": prediction_coverage,
        "next_missing_predictions": _next_missing_predictions(prediction_coverage),
        "methods": aggregate_methods,
        "deltas": {
            "vlm_dsg_vs_vlm_only": aggregate_delta,
            "vlm_dsg_vs_graph_tool_only": _aggregate_delta(episode_reports, "vlm_dsg_vs_graph_tool_only"),
            "graph_tool_only_vs_vlm_only": _aggregate_delta(episode_reports, "graph_tool_only_vs_vlm_only"),
        },
        "episode_groups": episode_reports,
    }


def _prediction_coverage(
    records_by_episode: dict[str, list[dict[str, Any]]],
    predictions_by_method: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    coverage = {}
    for method, predictions in sorted(predictions_by_method.items()):
        missing_by_episode = {}
        present = 0
        total = 0
        for episode_id, records in sorted(records_by_episode.items()):
            missing_count = sum(1 for row in records if str(row.get("id")) not in predictions)
            total += len(records)
            present += len(records) - missing_count
            if missing_count:
                missing_by_episode[episode_id] = missing_count
        coverage[method] = {
            "case_count": total,
            "missing_by_episode": missing_by_episode,
            "missing_case_count": total - present,
            "prediction_case_count": present,
            "prediction_coverage_rate": _ratio(present, total),
        }
    return coverage


def _next_missing_predictions(prediction_coverage: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method, coverage in sorted(prediction_coverage.items()):
        for episode_id, missing_count in sorted(coverage["missing_by_episode"].items()):
            rows.append(
                {
                    "episode_id": episode_id,
                    "method": method,
                    "missing_count": missing_count,
                }
            )
    return rows


def _aggregate_method(episode_reports: list[dict[str, Any]], method: str) -> dict[str, Any]:
    case_count = sum(report["methods"][method]["case_count"] for report in episode_reports)
    semantic = sum(report["methods"][method]["semantic_match_count"] for report in episode_reports)
    strict = sum(report["methods"][method]["strict_exact_count"] for report in episode_reports)
    predictions = sum(report["methods"][method]["prediction_count"] for report in episode_reports)
    return {
        "case_count": case_count,
        "prediction_count": predictions,
        "semantic_match_count": semantic,
        "semantic_match_rate": _ratio(semantic, case_count),
        "strict_exact_count": strict,
        "strict_exact_rate": _ratio(strict, case_count),
    }


def _aggregate_delta(episode_reports: list[dict[str, Any]], delta_name: str) -> dict[str, Any]:
    wins = sum(report["deltas"][delta_name]["paired_wins"] for report in episode_reports)
    losses = sum(report["deltas"][delta_name]["paired_losses"] for report in episode_reports)
    ties = sum(report["deltas"][delta_name]["paired_ties"] for report in episode_reports)
    p_value = _sign_test_p_value(wins, losses)
    return {
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": ties,
        "sign_test_p_value": p_value,
    }


def _sign_test_p_value(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    probability = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(round(min(1.0, 2 * probability), 6))


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# active QA v2 三组对比",
        "",
        f"- ready: {report['ready']}",
        f"- research_ready: {report['research_ready']}",
        f"- blockers: {', '.join(report['blockers']) or '-'}",
        "",
        "| method | semantic | strict | predictions |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method, summary in report["methods"].items():
        lines.append(
            f"| {method} | {summary['semantic_match_count']}/{summary['case_count']} | {summary['strict_exact_count']}/{summary['case_count']} | {summary['prediction_count']} |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "- 缺少 active QA v2 对齐 VLM-only / VLM+DSG prediction 时，不能形成 superiority claim。",
            "- GraphTool-only 若由 active QA graph records 派生，只能作为图查询消融，不是外部模型结果。",
        ]
    )
    return "\n".join(lines) + "\n"


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row.get("id")): row for row in rows}
    return [by_id[key] for key in sorted(by_id)]


def _short_episode(episode_id: str) -> str:
    return episode_id.rsplit("-", 1)[-1] if "-" in episode_id else episode_id


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(float(numerator) / float(denominator), 6)


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
