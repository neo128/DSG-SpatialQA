#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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
            "Merge P55 base predictions with prediction shard outputs and validate "
            "both shard-local and expected active QA v2 coverage."
        ),
    )
    parser.add_argument("--base-input", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, action="append", required=True)
    parser.add_argument("--shard-output-dir", type=Path, action="append", required=True)
    parser.add_argument("--expected-qa-root", type=Path, action="append", default=[])
    parser.add_argument("--target-method", choices=("vlm_only", "vlm_dsg_trusted"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        shard_specs = _shard_specs_from_manifests(args.shard_manifest, args.shard_output_dir)
        shard_coverage = _shard_prediction_coverage(shard_specs)
        blockers = []
        if shard_coverage["missing_output_count"] > 0:
            blockers.append("missing_shard_outputs")
        if shard_coverage["missing_case_count"] > 0 and shard_coverage["missing_output_count"] == 0:
            blockers.append("incomplete_shard_predictions")
        predictions_by_id: dict[str, lab.QAPrediction] = {}
        input_counts: dict[str, int] = {}
        base_predictions = lab.load_qa_predictions(args.base_input)
        input_counts[str(args.base_input)] = len(base_predictions)
        for prediction in base_predictions:
            predictions_by_id[prediction.id] = prediction
        for spec in shard_specs:
            output_path = spec["output_path"]
            if not output_path.exists():
                continue
            predictions = lab.load_qa_predictions(output_path)
            input_counts[str(output_path)] = len(predictions)
            for prediction in predictions:
                predictions_by_id[prediction.id] = prediction
        merged_predictions = [predictions_by_id[key] for key in sorted(predictions_by_id)]
        lab.save_qa_predictions(merged_predictions, args.output)
        expected_ids = _expected_case_ids(args.expected_qa_root)
        expected_qa_coverage = _coverage(expected_ids, set(predictions_by_id))
        if expected_qa_coverage["missing_case_count"] > 0:
            blockers.append("expected_qa_prediction_coverage_incomplete")
        report: dict[str, Any] = {
            "schema_version": "dsg-spatialqa-lab.p55-prediction-shard-merge-report.v1",
            "base_input": str(args.base_input),
            "blockers": sorted(set(blockers)),
            "expected_qa_coverage": expected_qa_coverage,
            "expected_qa_roots": [str(root) for root in args.expected_qa_root],
            "input_counts": input_counts,
            "merged_prediction_count": len(merged_predictions),
            "merged_prediction_digest": lab.qa_predictions_digest(merged_predictions),
            "merged_prediction_path": str(args.output),
            "ready": not blockers,
            "shard_manifest_count": len(args.shard_manifest),
            "shard_manifests": [str(path) for path in args.shard_manifest],
            "shard_output_dirs": [str(path) for path in args.shard_output_dir],
            "shard_prediction_coverage": shard_coverage,
            "target_method": args.target_method,
        }
        report["report_digest"] = _stable_digest_without(report, "report_digest")
        _write_json(args.report, report)
    except (OSError, ValueError, json.JSONDecodeError, lab.SpatialQAError) as exc:
        report = {
            "schema_version": "dsg-spatialqa-lab.p55-prediction-shard-merge-report.v1",
            "blockers": ["p55_prediction_shard_merge_error"],
            "error": str(exc),
            "ready": False,
        }
        report["report_digest"] = _stable_digest_without(report, "report_digest")
        _write_json(args.report, report)
        _emit(report)
        return 1

    _emit(
        {
            "action": "merge_p55_prediction_shards",
            "blockers": report["blockers"],
            "merged_prediction_count": report["merged_prediction_count"],
            "output": str(args.output),
            "ready": report["ready"],
            "report": str(args.report),
            "target_method": args.target_method,
        }
    )
    return 0 if report["ready"] is True else 1


def _shard_specs_from_manifests(
    manifest_paths: list[Path],
    output_dirs: list[Path],
) -> list[dict[str, Any]]:
    if len(output_dirs) not in (1, len(manifest_paths)):
        raise ValueError(
            "--shard-output-dir must be supplied once or once per --shard-manifest "
            f"(got {len(output_dirs)} output dirs for {len(manifest_paths)} manifests)"
        )
    specs: list[dict[str, Any]] = []
    for index, manifest_path in enumerate(manifest_paths):
        output_dir = output_dirs[0] if len(output_dirs) == 1 else output_dirs[index]
        specs.extend(_shard_specs_from_manifest(manifest_path, output_dir))
    return specs


def _shard_specs_from_manifest(manifest_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    payload = _load_json(manifest_path)
    shards = payload.get("shards")
    if not isinstance(shards, list):
        raise ValueError(f"shard manifest missing shards list: {manifest_path}")
    specs = []
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("shard manifest contains non-object shard")
        shard_path = shard.get("path")
        if not isinstance(shard_path, str):
            raise ValueError("shard manifest shard missing path")
        request_bundle = Path(shard_path)
        specs.append(
            {
                "expected_case_ids": _case_ids_from_request_bundle(request_bundle),
                "manifest_path": manifest_path,
                "output_path": output_dir / (request_bundle.stem + ".jsonl"),
                "output_dir": output_dir,
                "request_bundle": request_bundle,
                "shard_index": shard.get("shard_index"),
            }
        )
    return specs


def _case_ids_from_request_bundle(path: Path) -> list[str]:
    payload = _load_json(path)
    rows = payload.get("prediction_cases")
    if not isinstance(rows, list):
        rows = payload.get("case_inputs")
    if not isinstance(rows, list):
        raise ValueError(f"shard request bundle missing prediction_cases/case_inputs: {path}")
    case_ids = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        case_id = row.get("case_id")
        if isinstance(case_id, str):
            case_ids.append(case_id)
    return sorted(set(case_ids))


def _shard_prediction_coverage(specs: list[dict[str, Any]]) -> dict[str, Any]:
    expected_total = 0
    present_total = 0
    missing_case_ids: list[str] = []
    missing_output_count = 0
    shard_reports = []
    for spec in specs:
        expected_ids = set(spec["expected_case_ids"])
        expected_total += len(expected_ids)
        output_path = spec["output_path"]
        if not output_path.exists():
            missing_output_count += 1
            present_ids: set[str] = set()
        else:
            present_ids = {prediction.id for prediction in lab.load_qa_predictions(output_path)}
        missing = sorted(expected_ids - present_ids)
        present_total += len(expected_ids) - len(missing)
        missing_case_ids.extend(missing)
        shard_reports.append(
            {
                "expected_case_count": len(expected_ids),
                "missing_case_count": len(missing),
                "missing_case_ids": missing,
                "output_path": str(output_path),
                "output_exists": output_path.exists(),
                "prediction_case_count": len(present_ids),
                "shard_manifest": str(spec["manifest_path"]),
                "shard_output_dir": str(spec["output_dir"]),
                "request_bundle": str(spec["request_bundle"]),
                "shard_index": spec.get("shard_index"),
            }
        )
    return {
        "expected_case_count": expected_total,
        "missing_case_count": len(missing_case_ids),
        "missing_case_ids": sorted(missing_case_ids),
        "missing_output_count": missing_output_count,
        "prediction_case_count": present_total,
        "prediction_coverage_rate": _ratio(present_total, expected_total),
        "shards": shard_reports,
    }


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
