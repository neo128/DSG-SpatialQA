from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab import (
    SpatialQAError,
    load_real_collection_report,
    run_real_experiment_package,
    validate_real_collection_report,
)


REAL_SMALL_EXPERIMENT_RUN_MANIFEST_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-small-experiment-run-manifest.v1"
)
REAL_SMALL_EXPERIMENT_RUN_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.real-small-experiment-run-report.v1"
)
REQUIRED_REAL_CONTROL_KINDS = (
    "caption_memory",
    "graph_text",
    "multi_frame_vlm",
    "vlm",
)
REQUIRED_PREDICTED_INPUT_KINDS = ("observation_sequence",)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Run a manifest-driven real-small DSG-vs-control experiment over "
            "explicit local artifacts."
        ),
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        report = run_real_small_experiment(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            report_path=args.report,
        )
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        report = _error_report(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            report_path=args.report,
            error=exc,
        )
        _save_report(report, args.report)
        _emit_json(report)
        return 1

    _save_report(report, args.report)
    _emit_json(report)
    return 0 if report["ready"] is True else 1


def run_real_small_experiment(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = _load_manifest(path)
    output_root = Path(output_dir)
    run_report_path = Path(report_path)
    data_source_kind = _data_source_kind(manifest)
    not_real_research_result = _not_real_research_result(manifest, data_source_kind)
    required_control_kinds = _required_control_kinds(manifest)
    missing_artifacts = _missing_artifacts(manifest, path)
    blockers = _blockers(
        manifest=manifest,
        manifest_path=path,
        data_source_kind=data_source_kind,
        not_real_research_result=not_real_research_result,
        required_control_kinds=required_control_kinds,
        missing_artifacts=missing_artifacts,
    )
    if blockers or missing_artifacts:
        return _run_report(
            manifest=manifest,
            manifest_path=path,
            output_dir=output_root,
            report_path=run_report_path,
            data_source_kind=data_source_kind,
            not_real_research_result=not_real_research_result,
            ready=False,
            research_ready=False,
            blockers=blockers,
            next_missing_artifacts=missing_artifacts,
            package_result=None,
        )

    try:
        package_result = run_real_experiment_package(
            dataset_name=_optional_str(manifest.get("dataset_name"), "real_small"),
            episode_paths=_path_texts(_sequence(manifest.get("episodes")), path.parent),
            output_dir=output_root,
            manifest_path=output_root / "benchmark-manifest.json",
            readiness_report_path=output_root / "real-experiment-readiness.json",
            summary_report_path=output_root / "final" / "experiment-summary.json",
            record_path=output_root / "final" / "final-experiment-record.json",
            max_qa_per_episode=_optional_int_or_none(
                manifest.get("max_qa_per_episode")
                or _mapping(manifest.get("qa")).get("max_qa_per_episode"),
            ),
            tags=_tags(manifest),
            declared_data_source_kind="real",
            min_episode_count=_optional_int(manifest.get("min_episode_count"), 3),
            min_scene_count=_optional_int(manifest.get("min_scene_count"), 1),
            min_qa_count=_optional_int(manifest.get("min_qa_count"), 30),
            required_control_kinds=required_control_kinds,
            required_predicted_input_kinds=REQUIRED_PREDICTED_INPUT_KINDS,
            active_task_delta_report_paths=_optional_report_paths(
                manifest,
                path.parent,
                ("active_task_delta_report",),
            ),
            dashboard_bundle_paths=_optional_report_paths(
                manifest,
                path.parent,
                ("dashboard_bundle",),
            ),
            error_attribution_report_paths=_optional_report_paths(
                manifest,
                path.parent,
                ("error_attribution_report",),
            ),
            graph_eval_report_paths=_optional_report_paths(
                manifest,
                path.parent,
                ("graph_eval_report",),
            ),
            offline_control_import_manifest_path=_offline_manifest_path(
                manifest,
                path.parent,
            ),
            offline_control_matrix_report_paths=_offline_direct_report_paths(
                manifest,
                path.parent,
                "matrix_report_path",
            ),
            offline_control_result_report_paths=_offline_direct_report_paths(
                manifest,
                path.parent,
                "result_report_path",
            ),
            offline_prediction_import_report_paths=_offline_import_report_paths(
                manifest,
                path.parent,
            ),
            predicted_dsg_detector_run_manifest_path=_predicted_detector_manifest_path(
                manifest,
                path.parent,
            ),
            predicted_dsg_evidence_report_paths=_predicted_direct_report_paths(
                manifest,
                path.parent,
                "evidence_report_path",
            ),
            predicted_graph_report_paths=_predicted_direct_report_paths(
                manifest,
                path.parent,
                "predicted_graph_report_path",
            ),
            real_collection_report_paths=_path_texts(
                _sequence(manifest.get("real_collection_reports")),
                path.parent,
            ),
        )
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        blockers.append(
            {
                "name": "package_run_failed",
                "message": str(exc),
            }
        )
        return _run_report(
            manifest=manifest,
            manifest_path=path,
            output_dir=output_root,
            report_path=run_report_path,
            data_source_kind=data_source_kind,
            not_real_research_result=not_real_research_result,
            ready=False,
            research_ready=False,
            blockers=blockers,
            next_missing_artifacts=(),
            package_result=None,
        )

    ready = package_result.get("ready") is True
    research_ready = (
        ready
        and data_source_kind == "real"
        and not_real_research_result is False
    )
    if not ready:
        blockers.extend(_package_blockers(package_result))
    return _run_report(
        manifest=manifest,
        manifest_path=path,
        output_dir=output_root,
        report_path=run_report_path,
        data_source_kind=data_source_kind,
        not_real_research_result=not_real_research_result,
        ready=ready,
        research_ready=research_ready,
        blockers=blockers,
        next_missing_artifacts=(),
        package_result=package_result,
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Real-small run manifest JSON must be an object")
    manifest = cast(dict[str, Any], payload)
    schema_version = manifest.get("schema_version")
    if schema_version != REAL_SMALL_EXPERIMENT_RUN_MANIFEST_SCHEMA_VERSION:
        raise SpatialQAError(
            "Unsupported real-small run manifest schema version: "
            f"{schema_version}"
        )
    return manifest


def _run_report(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    output_dir: Path,
    report_path: Path,
    data_source_kind: str,
    not_real_research_result: bool,
    ready: bool,
    research_ready: bool,
    blockers: Sequence[Mapping[str, Any]],
    next_missing_artifacts: Sequence[Mapping[str, Any]],
    package_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    record_path = _string_or_none(package_result.get("record_path")) if package_result else None
    final_record_written = record_path is not None and Path(record_path).exists()
    report: dict[str, Any] = {
        "schema_version": REAL_SMALL_EXPERIMENT_RUN_REPORT_SCHEMA_VERSION,
        "action": "run_real_small_experiment",
        "manifest_path": str(manifest_path),
        "manifest_digest": _json_digest(manifest),
        "dataset_name": _optional_str(manifest.get("dataset_name"), "real_small"),
        "data_source_kind": data_source_kind,
        "not_real_research_result": not_real_research_result,
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "ready": ready,
        "research_ready": research_ready,
        "real_package_status": _real_package_status(
            ready,
            research_ready,
            data_source_kind,
            package_result,
        ),
        "final_record_written": final_record_written,
        "final_record_path": record_path,
        "final_record_kind": _final_record_kind(
            final_record_written,
            research_ready,
            data_source_kind,
        ),
        "blockers": _json_safe(list(blockers)),
        "next_missing_artifacts": _json_safe(list(next_missing_artifacts)),
        "package_result": _json_safe(package_result) if package_result is not None else None,
        "digests": _digests(package_result),
    }
    report["report_digest"] = _json_digest(report)
    return report


def _error_report(
    *,
    manifest_path: Path,
    output_dir: Path,
    report_path: Path,
    error: Exception,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": REAL_SMALL_EXPERIMENT_RUN_REPORT_SCHEMA_VERSION,
        "action": "run_real_small_experiment",
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "ready": False,
        "research_ready": False,
        "not_real_research_result": True,
        "final_record_written": False,
        "final_record_path": None,
        "final_record_kind": "none",
        "blockers": [{"name": "run_real_small_experiment_error", "message": str(error)}],
        "next_missing_artifacts": [],
        "package_result": None,
        "digests": {},
    }
    report["report_digest"] = _json_digest(report)
    return report


def _blockers(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    data_source_kind: str,
    not_real_research_result: bool,
    required_control_kinds: tuple[str, ...],
    missing_artifacts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if data_source_kind not in ("real", "synthetic_test_fixture"):
        blockers.append(
            {
                "name": "data_source_kind_supported",
                "expected": ["real", "synthetic_test_fixture"],
                "actual": data_source_kind,
            }
        )
    if data_source_kind == "synthetic_test_fixture" and not_real_research_result is False:
        blockers.append(
            {
                "name": "synthetic_fixture_requires_not_real_research_result",
                "expected": True,
                "actual": not_real_research_result,
            }
        )
    missing_controls = [
        kind for kind in REQUIRED_REAL_CONTROL_KINDS if kind not in required_control_kinds
    ]
    if missing_controls:
        blockers.append(
            {
                "name": "required_control_kinds_complete",
                "expected": list(REQUIRED_REAL_CONTROL_KINDS),
                "actual": list(required_control_kinds),
                "missing": missing_controls,
            }
        )
    if data_source_kind == "real" and not _sequence(manifest.get("real_collection_reports")):
        blockers.append(
            {
                "name": "real_collection_report_required_for_real_data",
                "message": "data_source_kind=real requires at least one real collection report",
            }
        )
    if data_source_kind == "real":
        blockers.extend(_real_collection_real_simulator_blockers(manifest, manifest_path))
    if missing_artifacts:
        blockers.append(
            {
                "name": "required_artifacts_present",
                "missing_count": len(missing_artifacts),
            }
        )
    return blockers


def _real_collection_real_simulator_blockers(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for index, path_text in enumerate(_sequence(manifest.get("real_collection_reports"))):
        report_path = _resolve_path(str(path_text), manifest_path.parent)
        if not report_path.exists():
            continue
        try:
            report = load_real_collection_report(report_path)
            validation = validate_real_collection_report(report)
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(
                {
                    "name": "real_collection_report_valid",
                    "index": index,
                    "path": str(report_path),
                    "message": str(exc),
                }
            )
            continue
        summary = _mapping(report.get("collection_summary"))
        frame_count = _int(summary.get("frame_count"), 0)
        frame_source_kind_counts = _mapping(summary.get("frame_source_kind_counts"))
        if validation.get("valid") is not True:
            blockers.append(
                {
                    "name": "real_collection_report_valid",
                    "index": index,
                    "path": str(report_path),
                    "validation": validation,
                }
            )
        if frame_source_kind_counts != {"real_simulator": frame_count}:
            blockers.append(
                {
                    "name": "real_collection_report_source_kind_real_simulator",
                    "index": index,
                    "path": str(report_path),
                    "expected": {"real_simulator": frame_count},
                    "actual": frame_source_kind_counts,
                }
            )
    return blockers


def _missing_artifacts(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> tuple[dict[str, Any], ...]:
    base_dir = manifest_path.parent
    required_paths = _required_input_paths(manifest, base_dir)
    missing = [
        {"role": role, "path": str(path)}
        for role, path in required_paths
        if not path.exists()
    ]
    return tuple(sorted(missing, key=lambda item: (item["role"], item["path"])))


def _required_input_paths(
    manifest: Mapping[str, Any],
    base_dir: Path,
) -> tuple[tuple[str, Path], ...]:
    paths: list[tuple[str, Path]] = []
    paths.extend(
        (f"episode[{index}]", _resolve_path(str(path), base_dir))
        for index, path in enumerate(_sequence(manifest.get("episodes")))
    )
    paths.extend(
        (f"real_collection_reports[{index}]", _resolve_path(str(path), base_dir))
        for index, path in enumerate(_sequence(manifest.get("real_collection_reports")))
    )
    offline = _mapping(manifest.get("offline_controls"))
    offline_manifest_path = _string_or_none(offline.get("manifest_path"))
    if offline_manifest_path is not None:
        paths.append(("offline_controls.manifest_path", _resolve_path(offline_manifest_path, base_dir)))
    else:
        paths.extend(
            (
                f"offline_controls.import_report_paths[{index}]",
                _resolve_path(str(path), base_dir),
            )
            for index, path in enumerate(_sequence(offline.get("import_report_paths")))
        )
        for key in ("matrix_report_path", "result_report_path"):
            value = _string_or_none(offline.get(key))
            if value is not None:
                paths.append((f"offline_controls.{key}", _resolve_path(value, base_dir)))
    predicted = _mapping(manifest.get("predicted_dsg"))
    predicted_manifest_path = _string_or_none(predicted.get("detector_run_manifest_path"))
    if predicted_manifest_path is not None:
        paths.append(
            (
                "predicted_dsg.detector_run_manifest_path",
                _resolve_path(predicted_manifest_path, base_dir),
            )
        )
    else:
        for key in ("predicted_graph_report_path", "evidence_report_path"):
            value = _string_or_none(predicted.get(key))
            if value is not None:
                paths.append((f"predicted_dsg.{key}", _resolve_path(value, base_dir)))
    reports = _mapping(manifest.get("reports"))
    for key in (
        "active_task_delta_report",
        "dashboard_bundle",
        "error_attribution_report",
        "graph_eval_report",
    ):
        value = _string_or_none(reports.get(key))
        if value is not None:
            paths.append((f"reports.{key}", _resolve_path(value, base_dir)))
    if not _sequence(manifest.get("episodes")):
        paths.append(("episodes", base_dir / "<missing>"))
    if not _sequence(manifest.get("real_collection_reports")):
        paths.append(("real_collection_reports", base_dir / "<missing>"))
    if not offline_manifest_path and not _sequence(offline.get("import_report_paths")):
        paths.append(("offline_controls.manifest_path", base_dir / "<missing>"))
    if not predicted_manifest_path and not _string_or_none(predicted.get("evidence_report_path")):
        paths.append(("predicted_dsg.detector_run_manifest_path", base_dir / "<missing>"))
    return tuple(paths)


def _package_blockers(package_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    readiness = _mapping(package_result.get("readiness"))
    failed_checks = _sequence(readiness.get("failed_checks"))
    return [
        {
            "name": "real_experiment_readiness_failed",
            "failed_checks": list(failed_checks),
        }
    ]


def _data_source_kind(manifest: Mapping[str, Any]) -> str:
    return _optional_str(
        manifest.get("data_source_kind")
        or manifest.get("declared_data_source_kind"),
        "real",
    )


def _not_real_research_result(
    manifest: Mapping[str, Any],
    data_source_kind: str,
) -> bool:
    value = manifest.get("not_real_research_result")
    if isinstance(value, bool):
        return value
    return data_source_kind != "real"


def _required_control_kinds(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    offline = _mapping(manifest.get("offline_controls"))
    values = _sequence(offline.get("required_source_kinds"))
    if not values:
        values = _sequence(manifest.get("required_control_kinds"))
    if not values:
        return REQUIRED_REAL_CONTROL_KINDS
    return tuple(sorted(str(value) for value in values if isinstance(value, str)))


def _tags(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    values = _sequence(manifest.get("tags"))
    if not values:
        return ("benchmark", "real")
    tags = tuple(str(value) for value in values if isinstance(value, str) and value != "")
    return tags or ("benchmark", "real")


def _offline_manifest_path(
    manifest: Mapping[str, Any],
    base_dir: Path,
) -> str | None:
    value = _string_or_none(_mapping(manifest.get("offline_controls")).get("manifest_path"))
    return str(_resolve_path(value, base_dir)) if value is not None else None


def _offline_direct_report_paths(
    manifest: Mapping[str, Any],
    base_dir: Path,
    key: str,
) -> tuple[str, ...]:
    value = _string_or_none(_mapping(manifest.get("offline_controls")).get(key))
    return (str(_resolve_path(value, base_dir)),) if value is not None else ()


def _offline_import_report_paths(
    manifest: Mapping[str, Any],
    base_dir: Path,
) -> tuple[str, ...]:
    return _path_texts(
        _sequence(_mapping(manifest.get("offline_controls")).get("import_report_paths")),
        base_dir,
    )


def _predicted_detector_manifest_path(
    manifest: Mapping[str, Any],
    base_dir: Path,
) -> str | None:
    value = _string_or_none(
        _mapping(manifest.get("predicted_dsg")).get("detector_run_manifest_path")
    )
    return str(_resolve_path(value, base_dir)) if value is not None else None


def _predicted_direct_report_paths(
    manifest: Mapping[str, Any],
    base_dir: Path,
    key: str,
) -> tuple[str, ...]:
    value = _string_or_none(_mapping(manifest.get("predicted_dsg")).get(key))
    return (str(_resolve_path(value, base_dir)),) if value is not None else ()


def _optional_report_paths(
    manifest: Mapping[str, Any],
    base_dir: Path,
    keys: Sequence[str],
) -> tuple[str, ...]:
    reports = _mapping(manifest.get("reports"))
    values = [
        str(_resolve_path(value, base_dir))
        for key in keys
        if (value := _string_or_none(reports.get(key))) is not None
    ]
    return tuple(values)


def _real_package_status(
    ready: bool,
    research_ready: bool,
    data_source_kind: str,
    package_result: Mapping[str, Any] | None,
) -> str:
    if ready and data_source_kind == "synthetic_test_fixture":
        return "synthetic_mechanical_pass"
    if research_ready:
        return "ready"
    if package_result is not None:
        return _optional_str(package_result.get("real_package_status"), "not_ready")
    return "not_ready"


def _final_record_kind(
    final_record_written: bool,
    research_ready: bool,
    data_source_kind: str,
) -> str:
    if not final_record_written:
        return "none"
    if research_ready:
        return "real_claim_record"
    if data_source_kind == "synthetic_test_fixture":
        return "synthetic_mechanical_record"
    return "incomplete_record"


def _digests(package_result: Mapping[str, Any] | None) -> dict[str, str | None]:
    if package_result is None:
        return {}
    return {
        "benchmark_manifest_digest": _string_or_none(
            package_result.get("manifest_digest")
        ),
        "readiness_report_digest": _string_or_none(
            package_result.get("readiness_report_digest")
        ),
        "summary_report_digest": _string_or_none(
            package_result.get("summary_report_digest")
        ),
        "final_record_digest": _string_or_none(package_result.get("record_digest")),
        "offline_control_matrix_digest": _string_or_none(
            package_result.get("generated_offline_control_matrix_report_digest")
        ),
        "predicted_dsg_evidence_digest": _string_or_none(
            package_result.get("generated_predicted_dsg_evidence_report_digest")
        ),
    }


def _path_texts(values: Sequence[object], base_dir: Path) -> tuple[str, ...]:
    return tuple(str(_resolve_path(str(value), base_dir)) for value in values)


def _resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else base_dir / path


def _mapping(value: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    return tuple(value)


def _optional_str(value: object, default: str) -> str:
    return value if isinstance(value, str) and value != "" else default


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _optional_int(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _optional_int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _int(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _json_digest(payload: Mapping[str, Any]) -> str:
    normalized = {key: value for key, value in payload.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _json_safe(value: object) -> Any:
    return json.loads(json.dumps(value, separators=(",", ":"), sort_keys=True))


def _save_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _emit_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())
