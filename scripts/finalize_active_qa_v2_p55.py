#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import dsg_spatialqa_lab as lab
from dsg_spatialqa_lab.benchmark.active_qa_v2 import load_active_qa_v2_records


COMPARISON_SPLITS = {
    "qa-observation-aware.jsonl",
    "qa-relation-centric.jsonl",
    "qa-situated.jsonl",
    "qa-temporal.jsonl",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Finalize P55 active QA v2 evaluation from explicit local predictions: "
            "merge coverage, compare three methods, and attribute cases."
        ),
    )
    parser.add_argument("--qa-root", type=Path, action="append", default=None)
    parser.add_argument("--vlm-input", type=Path, action="append", required=True)
    parser.add_argument("--trusted-input", type=Path, action="append", required=True)
    parser.add_argument("--adjudicated-input", type=Path, action="append", default=None)
    parser.add_argument("--graph-predictions", type=Path, required=True)
    parser.add_argument("--required-episode-count", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    qa_roots = args.qa_root or [Path("handoffs/ai2thor-real-small/inputs/qa-v2-active")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": "dsg-spatialqa-lab.p55-active-qa-v2-finalize-report.v1",
        "qa_roots": [str(root) for root in qa_roots],
        "required_episode_count": args.required_episode_count,
        "input_paths": {
            "graph_predictions": str(args.graph_predictions),
            "trusted_inputs": [str(path) for path in args.trusted_input],
            "vlm_inputs": [str(path) for path in args.vlm_input],
        },
        "output_dir": str(args.output_dir),
    }
    blockers: list[str] = []
    try:
        expected_ids = _expected_case_ids(qa_roots)
        if not expected_ids:
            blockers.append("missing_active_qa_v2_records")
        vlm_output = args.output_dir / "p55-vlm-only-qwen37-active-qa-v2-20-episodes.jsonl"
        vlm_merge_report = _merge_predictions(
            args.vlm_input,
            expected_ids,
            vlm_output,
            method="vlm_only",
        )
        trusted_output = args.output_dir / "p55-vlm-dsg-trusted-qwen37-active-qa-v2-20-episodes.jsonl"
        trusted_merge_report = _merge_predictions(
            args.trusted_input,
            expected_ids,
            trusted_output,
            method="vlm_dsg_trusted",
        )
        report["merge_reports"] = {
            "vlm_only": vlm_merge_report,
            "vlm_dsg_trusted": trusted_merge_report,
        }
        if not vlm_merge_report["ready"]:
            blockers.append("vlm_only_prediction_coverage_incomplete")
        if not trusted_merge_report["ready"]:
            blockers.append("vlm_dsg_trusted_prediction_coverage_incomplete")
        if not args.graph_predictions.exists():
            blockers.append("missing_graph_tool_only_dsg_predictions")
        coverage_ready = not blockers
        report["coverage_ready"] = coverage_ready

        comparison_output = args.output_dir / "p55-three-way-comparison-active-qa-v2.json"
        comparison_markdown = args.output_dir / "p55-three-way-comparison-active-qa-v2.zh.md"
        attribution_output = args.output_dir / "p55-active-qa-v2-case-attribution.json"
        attribution_markdown = args.output_dir / "p55-active-qa-v2-case-attribution.zh.md"
        final_claim_output = args.output_dir / "p55-dsg-superiority-claim.json"
        report["outputs"] = {
            "attribution": str(attribution_output),
            "attribution_markdown": str(attribution_markdown),
            "comparison": str(comparison_output),
            "comparison_markdown": str(comparison_markdown),
            "final_claim": str(final_claim_output),
            "merged_trusted_predictions": str(trusted_output),
            "merged_vlm_predictions": str(vlm_output),
        }

        comparison_report: dict[str, Any] | None = None
        attribution_report: dict[str, Any] | None = None
        if coverage_ready:
            compare_module = _load_peer_script("compare_active_qa_v2_three_way.py")
            compare_exit = _run_peer_main(
                compare_module,
                [
                    *_repeat_args("--qa-root", qa_roots),
                    "--vlm-predictions",
                    str(vlm_output),
                    "--graph-predictions",
                    str(args.graph_predictions),
                    "--vlm-dsg-predictions",
                    str(trusted_output),
                    "--required-episode-count",
                    str(args.required_episode_count),
                    "--output",
                    str(comparison_output),
                    "--markdown-output",
                    str(comparison_markdown),
                ],
            )
            comparison_report = _load_json(comparison_output)
            report["comparison_ready"] = comparison_report.get("ready") is True
            report["comparison_exit_code"] = compare_exit
            blockers.extend(str(item) for item in comparison_report.get("blockers", []) if item)

            adjudicated_inputs = args.adjudicated_input or [trusted_output]
            adjudicated_output = (
                trusted_output
                if args.adjudicated_input is None
                else args.output_dir / "p55-adjudicated-merged-active-qa-v2-20-episodes.jsonl"
            )
            if args.adjudicated_input is not None:
                adjudicated_merge_report = _merge_predictions(
                    adjudicated_inputs,
                    expected_ids,
                    adjudicated_output,
                    method="adjudicated",
                )
                report["merge_reports"]["adjudicated"] = adjudicated_merge_report
                if not adjudicated_merge_report["ready"]:
                    blockers.append("adjudicated_prediction_coverage_incomplete")
            if "adjudicated_prediction_coverage_incomplete" not in blockers:
                attribution_module = _load_peer_script("attribute_active_qa_v2_cases.py")
                attribution_exit = _run_peer_main(
                    attribution_module,
                    [
                        *_repeat_args("--qa-root", qa_roots),
                        "--vlm-predictions",
                        str(vlm_output),
                        "--graph-predictions",
                        str(args.graph_predictions),
                        "--trusted-predictions",
                        str(trusted_output),
                        "--adjudicated-predictions",
                        str(adjudicated_output),
                        "--output",
                        str(attribution_output),
                        "--markdown-output",
                        str(attribution_markdown),
                    ],
                )
                attribution_report = _load_json(attribution_output)
                report["attribution_exit_code"] = attribution_exit
                report["attribution_ready"] = attribution_exit == 0
        else:
            report["comparison_ready"] = False
            report["attribution_ready"] = False

        unique_blockers = sorted(set(blockers))
        ready = coverage_ready and comparison_report is not None and comparison_report.get("ready") is True
        if ready:
            assert comparison_report is not None
            final_claim = _final_claim_record(
                comparison_report,
                attribution_report,
                merged_vlm_path=vlm_output,
                merged_trusted_path=trusted_output,
                graph_predictions_path=args.graph_predictions,
            )
            _write_json(final_claim_output, final_claim)
        report.update(
            {
                "attribution_summary": (
                    attribution_report.get("summary") if isinstance(attribution_report, dict) else {}
                ),
                "blockers": unique_blockers,
                "final_claim_written": ready,
                "ready": ready,
                "research_ready": ready,
            }
        )
        report["report_digest"] = _stable_digest_without(report, "report_digest")
        _write_json(args.report, report)
    except (OSError, ValueError, json.JSONDecodeError, lab.SpatialQAError) as exc:
        report.update(
            {
                "blockers": ["p55_finalize_error"],
                "error": str(exc),
                "final_claim_written": False,
                "ready": False,
                "research_ready": False,
            }
        )
        report["report_digest"] = _stable_digest_without(report, "report_digest")
        _write_json(args.report, report)
        _emit(report)
        return 1

    _emit(
        {
            "action": "finalize_active_qa_v2_p55",
            "blockers": report["blockers"],
            "final_claim_written": report["final_claim_written"],
            "ready": report["ready"],
            "report": str(args.report),
            "research_ready": report["research_ready"],
        }
    )
    return 0 if report["ready"] is True else 1


def _merge_predictions(
    input_paths: list[Path],
    expected_ids: list[str],
    output_path: Path,
    *,
    method: str,
) -> dict[str, Any]:
    predictions_by_id: dict[str, lab.QAPrediction] = {}
    input_counts: dict[str, int] = {}
    for input_path in input_paths:
        predictions = lab.load_qa_predictions(input_path)
        input_counts[str(input_path)] = len(predictions)
        for prediction in predictions:
            predictions_by_id[prediction.id] = prediction
    merged_predictions = [predictions_by_id[key] for key in sorted(predictions_by_id)]
    lab.save_qa_predictions(merged_predictions, output_path)
    coverage = _coverage(expected_ids, set(predictions_by_id))
    report: dict[str, Any] = {
        "coverage": coverage,
        "input_counts": input_counts,
        "input_paths": [str(path) for path in input_paths],
        "merged_prediction_count": len(merged_predictions),
        "merged_prediction_digest": lab.qa_predictions_digest(merged_predictions),
        "merged_prediction_path": str(output_path),
        "method": method,
        "ready": coverage["missing_case_count"] == 0,
        "schema_version": "dsg-spatialqa-lab.qa-prediction-merge-report.v1",
    }
    report["report_digest"] = _stable_digest_without(report, "report_digest")
    return report


def _expected_case_ids(roots: list[Path]) -> list[str]:
    ids: set[str] = set()
    for root in roots:
        for path in sorted(root.glob("*/qa-*.jsonl")):
            if path.name not in COMPARISON_SPLITS:
                continue
            for row in load_active_qa_v2_records(path):
                case_id = row.get("id")
                if isinstance(case_id, str):
                    ids.add(case_id)
    return sorted(ids)


def _coverage(expected_ids: list[str], prediction_ids: set[str]) -> dict[str, Any]:
    expected = set(expected_ids)
    missing = sorted(expected - prediction_ids)
    unexpected = sorted(prediction_ids - expected) if expected else []
    present = len(expected_ids) - len(missing)
    return {
        "expected_case_count": len(expected_ids),
        "missing_case_count": len(missing),
        "missing_case_ids": missing,
        "prediction_coverage_rate": _ratio(present, len(expected_ids)),
        "unexpected_case_count": len(unexpected),
        "unexpected_case_ids": unexpected,
    }


def _load_peer_script(filename: str) -> ModuleType:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(f"p55_finalize_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"failed to load peer script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_peer_main(module: ModuleType, argv: list[str]) -> int:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        return int(module.main(argv))


def _repeat_args(flag: str, values: list[Path]) -> list[str]:
    args: list[str] = []
    for value in values:
        args.extend([flag, str(value)])
    return args


def _final_claim_record(
    comparison_report: dict[str, Any],
    attribution_report: dict[str, Any] | None,
    *,
    merged_vlm_path: Path,
    merged_trusted_path: Path,
    graph_predictions_path: Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "dsg-spatialqa-lab.p55-dsg-superiority-claim.v1",
        "claim": "vlm_dsg_trusted_fusion_significantly_beats_vlm_only_on_active_qa_v2",
        "comparison_digest": _stable_digest_without(comparison_report, "report_digest"),
        "comparison_summary": {
            "case_count": comparison_report.get("case_count"),
            "deltas": comparison_report.get("deltas", {}),
            "episode_count": comparison_report.get("episode_count"),
            "methods": comparison_report.get("methods", {}),
            "question_type_count": comparison_report.get("question_type_count"),
        },
        "graph_predictions_path": str(graph_predictions_path),
        "merged_trusted_predictions_path": str(merged_trusted_path),
        "merged_vlm_predictions_path": str(merged_vlm_path),
        "ready": True,
    }
    if attribution_report is not None:
        record["attribution_digest"] = attribution_report.get("report_digest")
        record["attribution_summary"] = attribution_report.get("summary", {})
    record["record_digest"] = _stable_digest_without(record, "record_digest")
    return record


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stable_digest_without(payload: dict[str, Any], key_to_omit: str) -> str:
    normalized = {key: value for key, value in payload.items() if key != key_to_omit}
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(numerator / denominator, 6)


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
