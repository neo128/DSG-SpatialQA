#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


DEFAULT_API_KEY_ENV = "DSG_SPATIALQA_DASHSCOPE_API_KEY"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Create or execute an explicit local P55 prediction shard run plan. "
            "Dry-run is deterministic and does not call network/model providers."
        ),
    )
    parser.add_argument("--target-method", choices=("vlm_only", "vlm_dsg_trusted"), required=True)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--source-kind", default="vlm", choices=("vlm", "multi_frame_vlm"))
    parser.add_argument("--vlm-predictions", type=Path)
    parser.add_argument("--graph-predictions", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help=(
            "Maximum number of shard subprocesses to run concurrently during "
            "--execute. Defaults to 1 for the historical sequential behavior."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    args = parser.parse_args(argv)

    try:
        if args.dry_run and args.execute:
            raise ValueError("Choose either --dry-run or --execute, not both.")
        if args.max_parallel < 1:
            raise ValueError("--max-parallel must be >= 1.")
        execute = bool(args.execute)
        dry_run = not execute
        commands = _commands(args, include_allow_network=execute and args.allow_network)
        request_input_summary = _request_input_summary(args.shard_manifest)
        pending_commands = [
            command for command in commands if not _command_output_complete(command)
        ]
        blockers: list[str] = []
        execution_blockers: list[str] = []
        if execute and not args.allow_network:
            blockers.append("network_not_allowed")
            execution_blockers.append("network_not_allowed")
        dedicated_key_set = bool(os.environ.get(args.api_key_env))
        system_key_set = bool(os.environ.get("DASHSCOPE_API_KEY"))
        uses_system_key = args.api_key_env == "DASHSCOPE_API_KEY"
        prediction_input_coverage: dict[str, Any] = {}
        if execute and not dedicated_key_set:
            blockers.append("dedicated_api_key_env_unset")
        if not dedicated_key_set:
            execution_blockers.append("dedicated_api_key_env_unset")
        if uses_system_key:
            blockers.append("system_dashscope_key_not_allowed")
            execution_blockers.append("system_dashscope_key_not_allowed")
        if args.target_method == "vlm_dsg_trusted":
            if args.vlm_predictions is None or not args.vlm_predictions.exists():
                blockers.append("missing_vlm_predictions")
                execution_blockers.append("missing_vlm_predictions")
            if args.graph_predictions is None or not args.graph_predictions.exists():
                blockers.append("missing_graph_predictions")
                execution_blockers.append("missing_graph_predictions")
            if (
                args.vlm_predictions is not None
                and args.vlm_predictions.exists()
                and args.graph_predictions is not None
                and args.graph_predictions.exists()
            ):
                expected_case_ids = _expected_case_ids_from_manifest(args.shard_manifest)
                prediction_input_coverage = {
                    "vlm_predictions": _prediction_coverage(
                        expected_case_ids,
                        args.vlm_predictions,
                    ),
                    "graph_predictions": _prediction_coverage(
                        expected_case_ids,
                        args.graph_predictions,
                    ),
                }
                if (
                    prediction_input_coverage["vlm_predictions"]["missing_case_count"]
                    > 0
                ):
                    blockers.append("missing_vlm_prediction_cases")
                    execution_blockers.append("missing_vlm_prediction_cases")
                if (
                    prediction_input_coverage["graph_predictions"]["missing_case_count"]
                    > 0
                ):
                    blockers.append("missing_graph_prediction_cases")
                    execution_blockers.append("missing_graph_prediction_cases")
        execution_blockers = sorted(set(execution_blockers))
        report: dict[str, Any] = {
            "schema_version": "dsg-spatialqa-lab.p55-prediction-shard-run-plan.v1",
            "api_key_env": args.api_key_env,
            "blockers": sorted(set(blockers)),
            "commands": commands,
            "dedicated_api_key_env_set": dedicated_key_set,
            "dry_run": dry_run,
            "execute": execute,
            "executed_shard_count": 0,
            "execution_blockers": execution_blockers,
            "execution_ready": not execution_blockers,
            "max_parallel": args.max_parallel,
            "pending_shard_count": len(pending_commands),
            "prediction_input_coverage": prediction_input_coverage,
            "ready": not blockers,
            "request_input_summary": request_input_summary,
            "resume": args.resume,
            "shard_count": len(commands),
            "shard_manifest": str(args.shard_manifest),
            "skipped_existing_shard_count": len(commands) - len(pending_commands),
            "system_dashscope_key_set": system_key_set,
            "target_method": args.target_method,
            "uses_system_dashscope_key": uses_system_key,
        }
        if execute and not blockers:
            executed = _execute_commands(pending_commands, max_parallel=args.max_parallel)
            report["executed_shard_count"] = executed["executed_shard_count"]
            report["failed_command"] = executed.get("failed_command")
            if executed["failed_command"] is not None:
                report["blockers"] = ["shard_command_failed"]
                report["ready"] = False
        report["report_digest"] = _stable_digest_without(report, "report_digest")
        _write_json(args.report, report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": "dsg-spatialqa-lab.p55-prediction-shard-run-plan.v1",
            "blockers": ["p55_prediction_shard_runner_error"],
            "error": str(exc),
            "ready": False,
        }
        report["report_digest"] = _stable_digest_without(report, "report_digest")
        _write_json(args.report, report)
        _emit(report)
        return 1

    _emit(
        {
            "action": "run_p55_prediction_shards",
            "blockers": report["blockers"],
            "dry_run": report["dry_run"],
            "execute": report["execute"],
            "execution_ready": report["execution_ready"],
            "pending_shard_count": report["pending_shard_count"],
            "ready": report["ready"],
            "report": str(args.report),
            "target_method": report["target_method"],
        }
    )
    return 0 if report["ready"] is True else 1


def _commands(args: argparse.Namespace, *, include_allow_network: bool) -> list[dict[str, Any]]:
    manifest = _load_json(args.shard_manifest)
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError(f"shard manifest missing shards: {args.shard_manifest}")
    commands: list[dict[str, Any]] = []
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("shard manifest contains non-object shard")
        shard_path_value = shard.get("path")
        if not isinstance(shard_path_value, str):
            raise ValueError("shard manifest shard missing path")
        shard_path = Path(shard_path_value)
        output_path = args.output_dir / (shard_path.stem + ".jsonl")
        trace_path = args.trace_dir / (shard_path.stem + "-trace.jsonl")
        argv = _command_argv(
            args,
            shard_path,
            output_path,
            trace_path,
            include_allow_network=include_allow_network,
        )
        commands.append(
            {
                "argv": argv,
                "output": str(output_path),
                "request_bundle": str(shard_path),
                "shard_index": shard.get("shard_index"),
                "trace_output": str(trace_path),
            }
        )
    return commands


def _command_argv(
    args: argparse.Namespace,
    shard_path: Path,
    output_path: Path,
    trace_path: Path,
    *,
    include_allow_network: bool,
) -> list[str]:
    if args.target_method == "vlm_only":
        argv = [
            "python",
            "external_tools/run_vlm_controls.py",
            "--request-bundle",
            str(shard_path),
            "--output",
            str(output_path),
            "--trace-output",
            str(trace_path),
            "--source-kind",
            args.source_kind,
            "--model",
            args.model,
            "--base-url",
            args.base_url,
            "--api-key-env",
            args.api_key_env,
        ]
    else:
        argv = [
            "python",
            "external_tools/run_vlm_graph_adjudication_active.py",
            "--request-bundle",
            str(shard_path),
            "--vlm-predictions",
            str(args.vlm_predictions),
            "--graph-predictions",
            str(args.graph_predictions),
            "--output",
            str(output_path),
            "--trace-output",
            str(trace_path),
            "--model",
            args.model,
            "--base-url",
            args.base_url,
            "--api-key-env",
            args.api_key_env,
            "--batch-size",
            str(args.batch_size),
        ]
    if args.limit is not None:
        argv.extend(["--limit", str(args.limit)])
    if include_allow_network:
        argv.append("--allow-network")
    if args.resume:
        argv.append("--resume")
    return argv


def _execute_commands(
    commands: list[dict[str, Any]],
    *,
    max_parallel: int,
) -> dict[str, Any]:
    if max_parallel == 1:
        executed = 0
        for command in commands:
            result = _run_command(command)
            if result["returncode"] != 0:
                return {
                    "executed_shard_count": executed,
                    "failed_command": _failed_command(command, result["returncode"]),
                }
            executed += 1
        return {"executed_shard_count": executed, "failed_command": None}

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        results = list(executor.map(_run_command, commands))
    failed_index = next(
        (index for index, result in enumerate(results) if result["returncode"] != 0),
        None,
    )
    if failed_index is not None:
        return {
            "executed_shard_count": sum(
                1 for result in results if result["returncode"] == 0
            ),
            "failed_command": _failed_command(
                commands[failed_index],
                results[failed_index]["returncode"],
            ),
        }
    return {"executed_shard_count": len(commands), "failed_command": None}


def _run_command(command: dict[str, Any]) -> dict[str, int]:
    Path(command["output"]).parent.mkdir(parents=True, exist_ok=True)
    Path(command["trace_output"]).parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command["argv"], check=False)  # noqa: S603
    return {"returncode": result.returncode}


def _failed_command(command: dict[str, Any], returncode: int) -> dict[str, Any]:
    return {
        "argv": command["argv"],
        "returncode": returncode,
        "shard_index": command.get("shard_index"),
    }


def _command_output_complete(command: dict[str, Any]) -> bool:
    output_path = Path(command["output"])
    if not output_path.exists():
        return False
    try:
        expected_case_ids = _case_ids_from_request_bundle(Path(command["request_bundle"]))
        prediction_ids = _prediction_ids(output_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return all(case_id in prediction_ids for case_id in expected_case_ids)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _expected_case_ids_from_manifest(path: Path) -> list[str]:
    case_ids: set[str] = set()
    for case in _manifest_prediction_cases(path):
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"shard prediction case missing case_id: {path}")
        case_ids.add(case_id)
    return sorted(case_ids)


def _case_ids_from_request_bundle(path: Path) -> list[str]:
    payload = _load_json(path)
    rows = payload.get("prediction_cases")
    if not isinstance(rows, list):
        rows = payload.get("case_inputs")
    if not isinstance(rows, list):
        raise ValueError(f"request bundle missing prediction_cases/case_inputs: {path}")
    case_ids = []
    for row in _mapping_sequence(rows):
        case_id = row.get("case_id")
        if isinstance(case_id, str) and case_id:
            case_ids.append(case_id)
    return sorted(set(case_ids))


def _request_input_summary(path: Path) -> dict[str, Any]:
    cases = _manifest_prediction_cases(path)
    target_crop_count = sum(1 for case in cases if isinstance(case.get("target_crop"), dict))
    target_visual_context_count = sum(
        1 for case in cases if isinstance(case.get("target_visual_context"), dict)
    )
    return {
        "missing_target_crop_case_count": len(cases) - target_crop_count,
        "request_case_count": len(cases),
        "target_crop_case_count": target_crop_count,
        "target_visual_context_case_count": target_visual_context_count,
        "target_crop_enriched": len(cases) > 0 and target_crop_count == len(cases),
        "visual_context_fallback_case_count": sum(
            1
            for case in cases
            if not isinstance(case.get("target_crop"), dict)
            and isinstance(case.get("target_visual_context"), dict)
        ),
    }


def _manifest_prediction_cases(path: Path) -> list[dict[str, Any]]:
    manifest = _load_json(path)
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError(f"shard manifest missing shards: {path}")
    cases: list[dict[str, Any]] = []
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("shard manifest contains non-object shard")
        shard_path_value = shard.get("path")
        if not isinstance(shard_path_value, str):
            raise ValueError("shard manifest shard missing path")
        shard_payload = _load_json(Path(shard_path_value))
        cases.extend(_mapping_sequence(shard_payload.get("prediction_cases")))
    return cases


def _prediction_coverage(expected_case_ids: list[str], path: Path) -> dict[str, Any]:
    prediction_ids = _prediction_ids(path)
    missing_case_ids = [
        case_id for case_id in expected_case_ids if case_id not in prediction_ids
    ]
    return {
        "expected_case_count": len(expected_case_ids),
        "missing_case_count": len(missing_case_ids),
        "missing_case_ids": missing_case_ids,
        "path": str(path),
        "prediction_case_count": len(prediction_ids),
    }


def _prediction_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict) and isinstance(payload.get("id"), str):
            ids.add(payload["id"])
    return ids


def _mapping_sequence(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("expected list of JSON objects")
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("expected JSON object in list")
        rows.append(item)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stable_digest_without(payload: dict[str, Any], key_to_omit: str) -> str:
    normalized = {key: value for key, value in payload.items() if key != key_to_omit}
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
