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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Collect local P55 shard prediction outputs and invoke the P55 finalizer "
            "only when all expected shard JSONL files are present."
        ),
    )
    parser.add_argument("--qa-root", type=Path, action="append", default=None)
    parser.add_argument("--vlm-base-input", type=Path)
    parser.add_argument("--trusted-base-input", type=Path)
    parser.add_argument("--graph-predictions", type=Path)
    parser.add_argument("--required-episode-count", type=int, default=20)
    parser.add_argument("--vlm-shard-manifest", type=Path, action="append", required=True)
    parser.add_argument("--vlm-shard-output-dir", type=Path, action="append", required=True)
    parser.add_argument("--trusted-shard-manifest", type=Path, action="append", required=True)
    parser.add_argument("--trusted-shard-output-dir", type=Path, action="append", required=True)
    parser.add_argument("--finalize-output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        vlm_specs = _shard_specs_from_manifests(
            args.vlm_shard_manifest,
            args.vlm_shard_output_dir,
            method="vlm_only",
        )
        trusted_specs = _shard_specs_from_manifests(
            args.trusted_shard_manifest,
            args.trusted_shard_output_dir,
            method="vlm_dsg_trusted",
        )
        vlm_outputs = [spec["output_path"] for spec in vlm_specs]
        trusted_outputs = [spec["output_path"] for spec in trusted_specs]
        missing_vlm = [str(path) for path in vlm_outputs if not path.exists()]
        missing_trusted = [str(path) for path in trusted_outputs if not path.exists()]
        blockers: list[str] = []
        if missing_vlm:
            blockers.append("missing_vlm_only_shard_outputs")
        if missing_trusted:
            blockers.append("missing_vlm_dsg_trusted_shard_outputs")
        shard_prediction_coverage = {
            "vlm_dsg_trusted": _shard_prediction_coverage(trusted_specs),
            "vlm_only": _shard_prediction_coverage(vlm_specs),
        }
        if shard_prediction_coverage["vlm_only"]["missing_case_count"] > 0 and not missing_vlm:
            blockers.append("incomplete_vlm_only_shard_predictions")
        if (
            shard_prediction_coverage["vlm_dsg_trusted"]["missing_case_count"] > 0
            and not missing_trusted
        ):
            blockers.append("incomplete_vlm_dsg_trusted_shard_predictions")
        shard_coverage_ready = not blockers
        report: dict[str, Any] = {
            "schema_version": "dsg-spatialqa-lab.p55-shard-finalize-report.v1",
            "blockers": blockers,
            "expected_shard_outputs": {
                "vlm_dsg_trusted": [str(path) for path in trusted_outputs],
                "vlm_only": [str(path) for path in vlm_outputs],
            },
            "finalize_invoked": False,
            "finalize_output_dir": str(args.finalize_output_dir),
            "missing_shard_output_count": len(missing_vlm) + len(missing_trusted),
            "missing_shard_outputs": {
                "vlm_dsg_trusted": missing_trusted,
                "vlm_only": missing_vlm,
            },
            "ready": False,
            "research_ready": False,
            "shard_coverage_ready": shard_coverage_ready,
            "shard_manifest_counts": {
                "vlm_dsg_trusted": len(args.trusted_shard_manifest),
                "vlm_only": len(args.vlm_shard_manifest),
            },
            "shard_manifests": {
                "vlm_dsg_trusted": [str(path) for path in args.trusted_shard_manifest],
                "vlm_only": [str(path) for path in args.vlm_shard_manifest],
            },
            "shard_output_dirs": {
                "vlm_dsg_trusted": [str(path) for path in args.trusted_shard_output_dir],
                "vlm_only": [str(path) for path in args.vlm_shard_output_dir],
            },
            "shard_prediction_coverage": shard_prediction_coverage,
        }
        if shard_coverage_ready:
            _require_path(args.vlm_base_input, "--vlm-base-input")
            _require_path(args.trusted_base_input, "--trusted-base-input")
            _require_path(args.graph_predictions, "--graph-predictions")
            finalize_report_path = args.finalize_output_dir / "p55-active-qa-v2-finalize-report.json"
            finalize_module = _load_peer_script("finalize_active_qa_v2_p55.py")
            finalize_args = [
                *_repeat_args("--qa-root", args.qa_root or []),
                "--vlm-input",
                str(args.vlm_base_input),
                *_repeat_args("--vlm-input", vlm_outputs),
                "--trusted-input",
                str(args.trusted_base_input),
                *_repeat_args("--trusted-input", trusted_outputs),
                "--graph-predictions",
                str(args.graph_predictions),
                "--required-episode-count",
                str(args.required_episode_count),
                "--output-dir",
                str(args.finalize_output_dir),
                "--report",
                str(finalize_report_path),
            ]
            finalize_exit_code = _run_peer_main(finalize_module, finalize_args)
            finalize_report = _load_json(finalize_report_path)
            report.update(
                {
                    "blockers": sorted(set(finalize_report.get("blockers", []))),
                    "finalize_exit_code": finalize_exit_code,
                    "finalize_invoked": True,
                    "finalize_report": finalize_report,
                    "finalize_report_path": str(finalize_report_path),
                    "ready": finalize_report.get("ready") is True,
                    "research_ready": finalize_report.get("research_ready") is True,
                }
            )
        report["report_digest"] = _stable_digest_without(report, "report_digest")
        _write_json(args.report, report)
    except (OSError, ValueError, json.JSONDecodeError, lab.SpatialQAError) as exc:
        report = {
            "schema_version": "dsg-spatialqa-lab.p55-shard-finalize-report.v1",
            "blockers": ["p55_shard_finalize_error"],
            "error": str(exc),
            "finalize_invoked": False,
            "ready": False,
            "research_ready": False,
        }
        report["report_digest"] = _stable_digest_without(report, "report_digest")
        _write_json(args.report, report)
        _emit(report)
        return 1

    _emit(
        {
            "action": "finalize_p55_from_shards",
            "blockers": report["blockers"],
            "finalize_invoked": report["finalize_invoked"],
            "ready": report["ready"],
            "report": str(args.report),
            "research_ready": report["research_ready"],
            "shard_coverage_ready": report["shard_coverage_ready"],
        }
    )
    return 0 if report["ready"] is True else 1


def _shard_specs_from_manifests(
    manifest_paths: list[Path],
    output_dirs: list[Path],
    *,
    method: str,
) -> list[dict[str, Any]]:
    if len(output_dirs) not in (1, len(manifest_paths)):
        raise ValueError(
            f"{method} shard output dirs must be supplied once or once per shard manifest "
            f"(got {len(output_dirs)} output dirs for {len(manifest_paths)} manifests)"
        )
    specs: list[dict[str, Any]] = []
    for index, manifest_path in enumerate(manifest_paths):
        output_dir = output_dirs[0] if len(output_dirs) == 1 else output_dirs[index]
        specs.extend(
            _shard_specs_from_manifest(
                manifest_path,
                output_dir,
                method=method,
            )
        )
    return specs


def _shard_specs_from_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    method: str,
) -> list[dict[str, Any]]:
    payload = _load_json(manifest_path)
    shards = payload.get("shards")
    if not isinstance(shards, list):
        raise ValueError(f"{method} shard manifest missing shards list: {manifest_path}")
    specs: list[dict[str, Any]] = []
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError(f"{method} shard manifest contains non-object shard")
        shard_path = shard.get("path")
        if not isinstance(shard_path, str):
            raise ValueError(f"{method} shard manifest shard missing path")
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
    missing_case_ids: list[str] = []
    unexpected_case_ids: list[str] = []
    shard_reports = []
    expected_total = 0
    present_total = 0
    for spec in specs:
        expected_ids = set(spec["expected_case_ids"])
        expected_total += len(expected_ids)
        output_path = spec["output_path"]
        if not output_path.exists():
            missing = sorted(expected_ids)
            present_ids: set[str] = set()
        else:
            present_ids = {prediction.id for prediction in lab.load_qa_predictions(output_path)}
            missing = sorted(expected_ids - present_ids)
            unexpected_case_ids.extend(sorted(present_ids - expected_ids))
        present_total += len(expected_ids) - len(missing)
        missing_case_ids.extend(missing)
        shard_reports.append(
            {
                "expected_case_count": len(expected_ids),
                "missing_case_count": len(missing),
                "missing_case_ids": missing,
                "output_path": str(output_path),
                "prediction_case_count": len(present_ids),
                "request_bundle": str(spec["request_bundle"]),
                "shard_manifest": str(spec["manifest_path"]),
                "shard_output_dir": str(spec["output_dir"]),
                "shard_index": spec.get("shard_index"),
            }
        )
    return {
        "expected_case_count": expected_total,
        "missing_case_count": len(missing_case_ids),
        "missing_case_ids": sorted(missing_case_ids),
        "prediction_case_count": present_total,
        "prediction_coverage_rate": _ratio(present_total, expected_total),
        "shards": shard_reports,
        "unexpected_case_count": len(unexpected_case_ids),
        "unexpected_case_ids": sorted(set(unexpected_case_ids)),
    }


def _require_path(path: Path | None, flag: str) -> None:
    if path is None:
        raise ValueError(f"{flag} is required when all shard outputs are present")
    if not path.exists():
        raise ValueError(f"{flag} does not exist: {path}")


def _load_peer_script(filename: str) -> ModuleType:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(f"p55_shard_finalize_{path.stem}", path)
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
