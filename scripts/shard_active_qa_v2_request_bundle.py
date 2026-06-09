#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ACTIVE_QA_V2_REQUEST_BUNDLE_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.active-qa-v2-vlm-request-bundle.v1"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Shard a leak-free active QA v2 request bundle for resumable external prediction runs.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=250)
    parser.add_argument("--prefix", default="active-qa-v2-shard")
    parser.add_argument("--target-method", default="vlm_only")
    parser.add_argument(
        "--target-crop-filter",
        choices=("all", "with", "without"),
        default="all",
    )
    args = parser.parse_args(argv)

    try:
        source = _load_source_bundle(args.input)
        if args.shard_size <= 0:
            raise ValueError("--shard-size must be greater than zero")
        source_cases = _prediction_cases(source)
        cases = _filter_cases_by_target_crop(
            source_cases,
            target_crop_filter=args.target_crop_filter,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        shards = []
        for index, start in enumerate(range(0, len(cases), args.shard_size), start=1):
            shard_cases = cases[start : start + args.shard_size]
            shard = _shard_bundle(
                source,
                shard_cases,
                shard_index=index,
                shard_count=_shard_count(len(cases), args.shard_size),
                source_bundle_path=args.input,
                target_method=args.target_method,
            )
            shard_path = args.output_dir / f"{args.prefix}-{index:04d}.json"
            shard_path.write_text(json.dumps(shard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            shards.append(
                {
                    "path": str(shard_path),
                    "request_bundle_digest": shard["request_bundle_digest"],
                    "request_count": shard["request_count"],
                    "shard_index": index,
                }
            )
        manifest = {
            "schema_version": "dsg-spatialqa-lab.active-qa-v2-request-bundle-shard-manifest.v1",
            "input": str(args.input),
            "shard_count": len(shards),
            "shard_size": args.shard_size,
            "shards": shards,
            "source_total_case_count": len(source_cases),
            "target_crop_case_count": _target_crop_case_count(cases),
            "target_crop_filter": args.target_crop_filter,
            "target_method": args.target_method,
            "total_case_count": len(cases),
        }
        manifest["manifest_digest"] = _digest_without(manifest, "manifest_digest")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit(
            {
                "action": "shard_active_qa_v2_request_bundle",
                "error": str(exc),
                "ready": False,
            }
        )
        return 1

    _emit(
        {
            "action": "shard_active_qa_v2_request_bundle",
            "manifest": str(args.manifest),
            "ready": True,
            "shard_count": len(shards),
            "source_total_case_count": len(source_cases),
            "target_crop_case_count": manifest["target_crop_case_count"],
            "target_crop_filter": args.target_crop_filter,
            "total_case_count": len(cases),
        }
    )
    return 0


def _load_source_bundle(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request bundle must be a JSON object")
    if payload.get("schema_version") != ACTIVE_QA_V2_REQUEST_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported_request_bundle_schema")
    if payload.get("leak_free") is not True:
        raise ValueError("request_bundle_not_leak_free")
    if payload.get("leak_paths"):
        raise ValueError("request_bundle_has_leak_paths")
    return payload


def _prediction_cases(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    cases = bundle.get("prediction_cases")
    if not isinstance(cases, list):
        raise ValueError("prediction_cases must be a list")
    rows = []
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"prediction case {index} must be an object")
        if not isinstance(case.get("case_id"), str):
            raise ValueError(f"prediction case {index} missing string case_id")
        rows.append(case)
    return rows


def _filter_cases_by_target_crop(
    cases: list[dict[str, Any]],
    *,
    target_crop_filter: str,
) -> list[dict[str, Any]]:
    if target_crop_filter == "all":
        return cases
    if target_crop_filter == "with":
        return [case for case in cases if isinstance(case.get("target_crop"), dict)]
    if target_crop_filter == "without":
        return [case for case in cases if not isinstance(case.get("target_crop"), dict)]
    raise ValueError(f"unsupported target crop filter: {target_crop_filter}")


def _target_crop_case_count(cases: list[dict[str, Any]]) -> int:
    return sum(1 for case in cases if isinstance(case.get("target_crop"), dict))


def _shard_bundle(
    source: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    shard_index: int,
    shard_count: int,
    source_bundle_path: Path,
    target_method: str,
) -> dict[str, Any]:
    shard = {
        "schema_version": ACTIVE_QA_V2_REQUEST_BUNDLE_SCHEMA_VERSION,
        "bundle_kind": "request_bundle_shard",
        "episode_id": source.get("episode_id", "multiple"),
        "leak_free": True,
        "leak_paths": [],
        "prediction_cases": cases,
        "request_count": len(cases),
        "source_bundle_digest": source.get("request_bundle_digest"),
        "source_bundle_path": str(source_bundle_path),
        "target_method": target_method,
        "shard_index": shard_index,
        "shard_count": shard_count,
    }
    shard["request_bundle_digest"] = _digest_without(shard, "request_bundle_digest")
    return shard


def _shard_count(case_count: int, shard_size: int) -> int:
    return (case_count + shard_size - 1) // shard_size


def _digest_without(payload: dict[str, Any], key_to_omit: str) -> str:
    normalized = {key: value for key, value in payload.items() if key != key_to_omit}
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
