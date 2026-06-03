from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.benchmark.qa_generator import (
    QACase,
    generate_qa_cases,
    load_qa_dataset,
    qa_dataset_digest,
    save_qa_dataset,
)
from dsg_spatialqa_lab.episodes import EpisodeFrame, load_episode_sequence
from dsg_spatialqa_lab.eval.error_attribution import (
    error_attribution_report_digest,
    load_error_attribution_report,
)
from dsg_spatialqa_lab.eval.qa_metrics import (
    load_qa_eval_delta_report,
    load_qa_eval_report,
    qa_eval_delta_report_digest,
    qa_eval_report_digest,
)
from dsg_spatialqa_lab.eval.graph_metrics import (
    graph_eval_report_digest,
    load_graph_eval_report,
)
from dsg_spatialqa_lab.eval.offline_control_matrix import (
    load_offline_control_matrix_report,
    offline_control_matrix_report_digest,
)
from dsg_spatialqa_lab.eval.offline_control_result import (
    load_offline_control_result_report,
    offline_control_result_report_digest,
)
from dsg_spatialqa_lab.eval.offline_predictions import (
    load_offline_prediction_import_report,
    offline_prediction_import_report_digest,
)
from dsg_spatialqa_lab.eval.task_metrics import (
    active_task_delta_report_digest,
    active_task_report_digest,
    load_active_task_delta_report,
    load_active_task_report,
)
from dsg_spatialqa_lab.oracle import build_oracle_graph_from_episode
from dsg_spatialqa_lab.predicted import (
    load_predicted_graph_report,
    predicted_graph_report_digest,
)
from dsg_spatialqa_lab.predicted_evidence import (
    load_predicted_dsg_evidence_report,
    predicted_dsg_evidence_report_digest,
)
from dsg_spatialqa_lab.benchmark.real_collection import (
    load_real_collection_report,
    real_collection_report_digest,
)
from dsg_spatialqa_lab.scene_io import graph_json_digest, load_graph_json, save_graph_json
from dsg_spatialqa_lab.schema import SpatialQAError
from dsg_spatialqa_lab.visualization.dashboard_export import (
    dashboard_bundle_digest,
    load_dashboard_bundle,
)


BENCHMARK_MANIFEST_SCHEMA_VERSION = "dsg-spatialqa-lab.benchmark-manifest.v1"
DYNAMIC_QUESTION_TYPES = frozenset(
    {
        "agent_history",
        "agent_timeline",
        "object_history",
        "object_timeline",
        "recent_events",
        "relation_timeline",
        "reobserve_targets",
        "scene_delta",
        "next_action_validity",
    }
)


def build_benchmark_artifacts(
    *,
    dataset_name: str,
    episode_paths: Sequence[str | Path],
    output_dir: str | Path,
    max_qa_per_episode: int | None = None,
    tags: Sequence[str] = ("benchmark", "oracle"),
    qa_eval_report_paths: Sequence[str | Path] = (),
    qa_eval_delta_report_paths: Sequence[str | Path] = (),
    active_task_report_paths: Sequence[str | Path] = (),
    active_task_delta_report_paths: Sequence[str | Path] = (),
    dashboard_bundle_paths: Sequence[str | Path] = (),
    error_attribution_report_paths: Sequence[str | Path] = (),
    graph_eval_report_paths: Sequence[str | Path] = (),
    offline_control_matrix_report_paths: Sequence[str | Path] = (),
    offline_control_result_report_paths: Sequence[str | Path] = (),
    offline_prediction_import_report_paths: Sequence[str | Path] = (),
    predicted_dsg_evidence_report_paths: Sequence[str | Path] = (),
    predicted_graph_report_paths: Sequence[str | Path] = (),
    real_collection_report_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    _validate_non_empty_str(dataset_name, "dataset_name")
    if not episode_paths:
        raise SpatialQAError("Benchmark manifest requires at least one episode path")
    if max_qa_per_episode is not None and max_qa_per_episode <= 0:
        raise SpatialQAError("max_qa_per_episode must be positive")

    root = Path(output_dir)
    graph_dir = root / "graphs"
    qa_dir = root / "qa"
    graph_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, Any]] = []
    all_cases: list[QACase] = []
    for episode_path in episode_paths:
        path = Path(episode_path)
        frames = load_episode_sequence(path)
        scene_id = _frame_scene_id(frames)
        episode_id = _frame_episode_id(frames)
        graph = build_oracle_graph_from_episode(frames)
        artifact_name = _artifact_name(episode_id)
        graph_path = graph_dir / f"{artifact_name}-oracle-graph.json"
        qa_path = qa_dir / f"{artifact_name}-qa.jsonl"
        save_graph_json(graph, graph_path)
        cases = generate_qa_cases(
            graph,
            scene_id=scene_id,
            episode_id=episode_id,
            max_cases=max_qa_per_episode,
            tags=tags,
        )
        save_qa_dataset(cases, qa_path)
        all_cases.extend(cases)
        artifacts.append(
            {
                "episode_id": episode_id,
                "scene_id": scene_id,
                "episode_path": str(path),
                "episode_step_count": len(frames),
                "graph_path": str(graph_path),
                "graph_digest": graph_json_digest(graph),
                "qa_path": str(qa_path),
                "qa_count": len(cases),
                "qa_dataset_digest": qa_dataset_digest(cases),
                "source": "oracle",
            }
        )

    return _manifest(
        dataset_name=dataset_name,
        artifacts=artifacts,
        cases=all_cases,
        max_qa_per_episode=max_qa_per_episode,
        tags=tags,
        experiment_artifacts=_experiment_artifacts(
            qa_eval_report_paths=qa_eval_report_paths,
            qa_eval_delta_report_paths=qa_eval_delta_report_paths,
            active_task_report_paths=active_task_report_paths,
            active_task_delta_report_paths=active_task_delta_report_paths,
            dashboard_bundle_paths=dashboard_bundle_paths,
            error_attribution_report_paths=error_attribution_report_paths,
            graph_eval_report_paths=graph_eval_report_paths,
            offline_control_matrix_report_paths=offline_control_matrix_report_paths,
            offline_control_result_report_paths=offline_control_result_report_paths,
            offline_prediction_import_report_paths=offline_prediction_import_report_paths,
            predicted_dsg_evidence_report_paths=predicted_dsg_evidence_report_paths,
            predicted_graph_report_paths=predicted_graph_report_paths,
            real_collection_report_paths=real_collection_report_paths,
        ),
    )


def benchmark_manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def benchmark_manifest_json(manifest: Mapping[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def save_benchmark_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(benchmark_manifest_json(manifest), encoding="utf-8")
    return output_path


def load_benchmark_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Benchmark manifest JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_benchmark_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = manifest.get("schema_version")
    manifest_digest = _string_or_none(manifest.get("manifest_digest"))
    expected_manifest_digest = benchmark_manifest_digest(manifest)
    artifacts = _artifact_sequence(manifest.get("artifacts"))
    summary = manifest.get("summary")
    experiment_artifacts = _experiment_artifact_sequence(
        manifest.get("experiment_artifacts")
    )
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == BENCHMARK_MANIFEST_SCHEMA_VERSION,
            "expected": BENCHMARK_MANIFEST_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "manifest_digest",
            "passed": manifest_digest == expected_manifest_digest,
            "expected": expected_manifest_digest,
            "actual": manifest_digest,
        },
        {
            "name": "artifact_count",
            "passed": _summary_value(summary, "episode_count") == len(artifacts),
            "expected": len(artifacts),
            "actual": _summary_value(summary, "episode_count"),
        },
        {
            "name": "qa_count",
            "passed": _summary_value(summary, "qa_count") == _artifact_qa_count(artifacts),
            "expected": _artifact_qa_count(artifacts),
            "actual": _summary_value(summary, "qa_count"),
        },
    ]
    if experiment_artifacts:
        expected_experiment_artifact_digests = _experiment_artifact_digests(
            experiment_artifacts
        )
        checks.append(
            {
                "name": "experiment_artifact_count",
                "passed": _summary_value(summary, "experiment_artifact_count")
                == len(experiment_artifacts),
                "expected": len(experiment_artifacts),
                "actual": _summary_value(summary, "experiment_artifact_count"),
            }
        )
        checks.append(
            {
                "name": "experiment_artifact_digests",
                "passed": manifest.get("experiment_artifact_digests")
                == expected_experiment_artifact_digests,
                "expected": expected_experiment_artifact_digests,
                "actual": manifest.get("experiment_artifact_digests"),
            }
        )
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "manifest_digest": manifest_digest,
        "checks": checks,
    }


def compare_benchmark_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_benchmark_manifest(manifest)
    artifacts = _artifact_sequence(manifest.get("artifacts"))
    experiment_artifacts = _experiment_artifact_sequence(
        manifest.get("experiment_artifacts")
    )
    current_artifacts: list[dict[str, Any]] = []
    current_cases: list[QACase] = []
    for artifact in artifacts:
        graph_path = _required_artifact_str(artifact, "graph_path")
        qa_path = _required_artifact_str(artifact, "qa_path")
        cases = load_qa_dataset(qa_path)
        graph = load_graph_json(graph_path)
        current_cases.extend(cases)
        current_artifacts.append(
            {
                **dict(artifact),
                "graph_digest": graph_json_digest(graph),
                "qa_count": len(cases),
                "qa_dataset_digest": qa_dataset_digest(cases),
            }
        )

    current_manifest = _manifest(
        dataset_name=_required_manifest_str(manifest, "dataset_name"),
        artifacts=current_artifacts,
        cases=current_cases,
        max_qa_per_episode=_filter_max_qa_per_episode(manifest),
        tags=_filter_tags(manifest),
        experiment_artifacts=_current_experiment_artifacts(experiment_artifacts),
    )
    saved_digest = _string_or_none(manifest.get("manifest_digest"))
    current_digest = _string_or_none(current_manifest.get("manifest_digest"))
    checks = [
        {
            "name": "manifest_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        _equality_check(
            "graph_digests_match_current",
            manifest.get("graph_digests"),
            current_manifest["graph_digests"],
        ),
        _equality_check(
            "qa_dataset_digests_match_current",
            manifest.get("qa_dataset_digests"),
            current_manifest["qa_dataset_digests"],
        ),
        _equality_check(
            "qa_digest_matches_current",
            manifest.get("qa_digest"),
            current_manifest["qa_digest"],
        ),
        _equality_check("summary_matches_current", manifest.get("summary"), current_manifest["summary"]),
        _equality_check(
            "coverage_matches_current",
            manifest.get("coverage"),
            current_manifest["coverage"],
        ),
    ]
    if experiment_artifacts:
        checks.append(
            _equality_check(
                "experiment_artifacts_match_current",
                manifest.get("experiment_artifacts"),
                current_manifest.get("experiment_artifacts"),
            )
        )
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def _manifest(
    *,
    dataset_name: str,
    artifacts: Sequence[Mapping[str, Any]],
    cases: Sequence[QACase],
    max_qa_per_episode: int | None,
    tags: Sequence[str],
    experiment_artifacts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    artifact_list = [_stable_mapping(artifact) for artifact in artifacts]
    experiment_artifact_list = [
        _stable_mapping(artifact) for artifact in experiment_artifacts
    ]
    scene_ids = sorted({str(artifact["scene_id"]) for artifact in artifact_list})
    episode_ids = [str(artifact["episode_id"]) for artifact in artifact_list]
    graph_digests = {
        str(artifact["episode_id"]): str(artifact["graph_digest"]) for artifact in artifact_list
    }
    qa_dataset_digests = {
        str(artifact["episode_id"]): str(artifact["qa_dataset_digest"])
        for artifact in artifact_list
    }
    summary = {
        "dataset_name": dataset_name,
        "episode_count": len(artifact_list),
        "qa_count": len(cases),
        "scene_count": len(scene_ids),
        "task_count": 0,
    }
    if experiment_artifact_list:
        summary["experiment_artifact_count"] = len(experiment_artifact_list)
    manifest: dict[str, Any] = {
        "schema_version": BENCHMARK_MANIFEST_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "scene_count": len(scene_ids),
        "episode_count": len(episode_ids),
        "qa_count": len(cases),
        "qa_digest": qa_dataset_digest(cases),
        "task_count": 0,
        "graph_digests": graph_digests,
        "qa_dataset_digests": qa_dataset_digests,
        "filters": {
            "max_qa_per_episode": max_qa_per_episode,
            "source": "oracle",
            "tags": list(tags),
        },
        "coverage": _coverage(cases, oracle_graph_count=len(artifact_list)),
        "summary": summary,
        "artifacts": artifact_list,
    }
    if experiment_artifact_list:
        manifest["experiment_artifacts"] = experiment_artifact_list
        manifest["experiment_artifact_digests"] = {
            _experiment_artifact_key(artifact): artifact["digest"]
            for artifact in experiment_artifact_list
        }
    manifest["manifest_digest"] = benchmark_manifest_digest(manifest)
    return manifest


def _coverage(cases: Sequence[QACase], *, oracle_graph_count: int) -> dict[str, Any]:
    dynamic_count = sum(1 for case in cases if _is_dynamic_case(case))
    static_count = len(cases) - dynamic_count
    return {
        "by_question_type": _sorted_counts(case.question_type for case in cases),
        "by_scene": _sorted_counts(case.scene_id for case in cases),
        "by_episode": _sorted_counts(case.episode_id for case in cases),
        "by_reference_frame": _sorted_counts(case.reference_frame or "none" for case in cases),
        "by_tag": _sorted_counts(tag for case in cases for tag in case.tags),
        "dynamic_static": {"dynamic": dynamic_count, "static": static_count},
        "oracle_predicted": {"oracle": oracle_graph_count, "predicted": 0},
    }


def _is_dynamic_case(case: QACase) -> bool:
    return case.question_type in DYNAMIC_QUESTION_TYPES or any(
        tag in {"dynamic", "temporal"} for tag in case.tags
    )


def _frame_scene_id(frames: Sequence[EpisodeFrame]) -> str:
    if not frames:
        raise SpatialQAError("Benchmark episode sequence must contain at least one frame")
    scene_ids = {frame.scene_id for frame in frames}
    if len(scene_ids) != 1:
        raise SpatialQAError("Benchmark episode sequence scene_id must be consistent")
    return frames[0].scene_id


def _frame_episode_id(frames: Sequence[EpisodeFrame]) -> str:
    if not frames:
        raise SpatialQAError("Benchmark episode sequence must contain at least one frame")
    episode_ids = {frame.episode_id for frame in frames}
    if len(episode_ids) != 1:
        raise SpatialQAError("Benchmark episode sequence episode_id must be consistent")
    return frames[0].episode_id


def _artifact_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _artifact_sequence(value: object) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    artifacts: list[Mapping[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            artifacts.append(cast(Mapping[str, Any], item))
    return tuple(artifacts)


def _experiment_artifacts(
    *,
    qa_eval_report_paths: Sequence[str | Path],
    qa_eval_delta_report_paths: Sequence[str | Path],
    active_task_report_paths: Sequence[str | Path],
    active_task_delta_report_paths: Sequence[str | Path],
    dashboard_bundle_paths: Sequence[str | Path],
    error_attribution_report_paths: Sequence[str | Path],
    graph_eval_report_paths: Sequence[str | Path],
    offline_control_matrix_report_paths: Sequence[str | Path],
    offline_control_result_report_paths: Sequence[str | Path],
    offline_prediction_import_report_paths: Sequence[str | Path],
    predicted_dsg_evidence_report_paths: Sequence[str | Path],
    predicted_graph_report_paths: Sequence[str | Path],
    real_collection_report_paths: Sequence[str | Path],
) -> tuple[dict[str, Any], ...]:
    artifacts: list[dict[str, Any]] = []
    for path in active_task_delta_report_paths:
        artifacts.append(_experiment_artifact_from_path("active_task_delta_report", path))
    for path in active_task_report_paths:
        artifacts.append(_experiment_artifact_from_path("active_task_report", path))
    for path in dashboard_bundle_paths:
        artifacts.append(_experiment_artifact_from_path("dashboard_bundle", path))
    for path in error_attribution_report_paths:
        artifacts.append(_experiment_artifact_from_path("error_attribution_report", path))
    for path in graph_eval_report_paths:
        artifacts.append(_experiment_artifact_from_path("graph_eval_report", path))
    for path in offline_control_matrix_report_paths:
        artifacts.append(
            _experiment_artifact_from_path(
                "offline_control_matrix_report",
                path,
            )
        )
    for path in offline_control_result_report_paths:
        artifacts.append(
            _experiment_artifact_from_path(
                "offline_control_result_report",
                path,
            )
        )
    for path in offline_prediction_import_report_paths:
        artifacts.append(
            _experiment_artifact_from_path(
                "offline_prediction_import_report",
                path,
            )
        )
    for path in predicted_dsg_evidence_report_paths:
        artifacts.append(
            _experiment_artifact_from_path("predicted_dsg_evidence_report", path)
        )
    for path in predicted_graph_report_paths:
        artifacts.append(_experiment_artifact_from_path("predicted_graph_report", path))
    for path in real_collection_report_paths:
        artifacts.append(_experiment_artifact_from_path("real_collection_report", path))
    for path in qa_eval_delta_report_paths:
        artifacts.append(_experiment_artifact_from_path("qa_eval_delta_report", path))
    for path in qa_eval_report_paths:
        artifacts.append(_experiment_artifact_from_path("qa_eval_report", path))
    return tuple(sorted(artifacts, key=lambda item: (item["artifact_type"], item["path"])))


def _experiment_artifact_from_path(
    artifact_type: str,
    path: str | Path,
) -> dict[str, Any]:
    artifact_path = Path(path)
    payload, digest = _load_experiment_artifact(artifact_type, artifact_path)
    return {
        "artifact_type": artifact_type,
        "path": str(artifact_path),
        "schema_version": _string_or_none(payload.get("schema_version")),
        "digest": digest,
    }


def _load_experiment_artifact(
    artifact_type: str,
    path: Path,
) -> tuple[Mapping[str, Any], str]:
    if artifact_type == "active_task_delta_report":
        payload = load_active_task_delta_report(path)
        return payload, active_task_delta_report_digest(payload)
    if artifact_type == "active_task_report":
        payload = load_active_task_report(path)
        return payload, active_task_report_digest(payload)
    if artifact_type == "dashboard_bundle":
        payload = load_dashboard_bundle(path)
        return payload, dashboard_bundle_digest(payload)
    if artifact_type == "error_attribution_report":
        payload = load_error_attribution_report(path)
        return payload, error_attribution_report_digest(payload)
    if artifact_type == "graph_eval_report":
        payload = load_graph_eval_report(path)
        return payload, graph_eval_report_digest(payload)
    if artifact_type == "offline_control_matrix_report":
        payload = load_offline_control_matrix_report(path)
        return payload, offline_control_matrix_report_digest(payload)
    if artifact_type == "offline_control_result_report":
        payload = load_offline_control_result_report(path)
        return payload, offline_control_result_report_digest(payload)
    if artifact_type == "offline_prediction_import_report":
        payload = load_offline_prediction_import_report(path)
        return payload, offline_prediction_import_report_digest(payload)
    if artifact_type == "predicted_dsg_evidence_report":
        payload = load_predicted_dsg_evidence_report(path)
        return payload, predicted_dsg_evidence_report_digest(payload)
    if artifact_type == "predicted_graph_report":
        payload = load_predicted_graph_report(path)
        return payload, predicted_graph_report_digest(payload)
    if artifact_type == "real_collection_report":
        payload = load_real_collection_report(path)
        return payload, real_collection_report_digest(payload)
    if artifact_type == "qa_eval_delta_report":
        payload = load_qa_eval_delta_report(path)
        return payload, qa_eval_delta_report_digest(payload)
    if artifact_type == "qa_eval_report":
        payload = load_qa_eval_report(path)
        return payload, qa_eval_report_digest(payload)
    raise SpatialQAError(f"Unsupported experiment artifact type: {artifact_type}")


def _experiment_artifact_sequence(value: object) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    artifacts: list[Mapping[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            artifacts.append(cast(Mapping[str, Any], item))
    return tuple(artifacts)


def _current_experiment_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    current: list[dict[str, Any]] = []
    for artifact in artifacts:
        current.append(
            _experiment_artifact_from_path(
                _required_experiment_artifact_str(artifact, "artifact_type"),
                _required_experiment_artifact_str(artifact, "path"),
            )
        )
    return tuple(current)


def _experiment_artifact_key(artifact: Mapping[str, Any]) -> str:
    artifact_type = str(artifact["artifact_type"])
    path = Path(str(artifact["path"]))
    return f"{artifact_type}:{path.name}"


def _experiment_artifact_digests(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        _experiment_artifact_key(artifact): artifact.get("digest")
        for artifact in artifacts
    }


def _artifact_qa_count(artifacts: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for artifact in artifacts:
        value = artifact.get("qa_count", 0)
        if isinstance(value, int) and not isinstance(value, bool):
            total += value
    return total


def _required_artifact_str(artifact: Mapping[str, Any], key: str) -> str:
    value = artifact.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Benchmark artifact field must be a non-empty string: {key}")
    return value


def _required_experiment_artifact_str(artifact: Mapping[str, Any], key: str) -> str:
    value = artifact.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(
            f"Benchmark experiment artifact field must be a non-empty string: {key}"
        )
    return value


def _required_manifest_str(manifest: Mapping[str, Any], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Benchmark manifest field must be a non-empty string: {key}")
    return value


def _filter_max_qa_per_episode(manifest: Mapping[str, Any]) -> int | None:
    filters = manifest.get("filters")
    if not isinstance(filters, Mapping):
        return None
    value = filters.get("max_qa_per_episode")
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise SpatialQAError("Benchmark manifest filter max_qa_per_episode must be an integer or null")


def _filter_tags(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    filters = manifest.get("filters")
    if not isinstance(filters, Mapping):
        return ()
    value = filters.get("tags", ())
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SpatialQAError("Benchmark manifest filter tags must be a string sequence")
    tags: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise SpatialQAError("Benchmark manifest filter tags must be a string sequence")
        tags.append(item)
    return tuple(tags)


def _summary_value(summary: object, key: str) -> object:
    if not isinstance(summary, Mapping):
        return None
    return summary.get(key)


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _validate_non_empty_str(value: object, key: str) -> None:
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Benchmark manifest field must be a non-empty string: {key}")


def _stable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _json_value(item)
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _stable_mapping(cast(Mapping[str, Any], value))
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_json_value(item) for item in value]
    return value


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _equality_check(name: str, saved: Any, current: Any) -> dict[str, Any]:
    check: dict[str, Any] = {
        "name": name,
        "passed": saved == current,
        "expected": saved,
        "actual": current,
    }
    if saved != current:
        check["differences"] = _differences(saved, current)
    return check


def _differences(saved: Any, current: Any, path: str = "") -> list[dict[str, Any]]:
    if saved == current:
        return []
    if isinstance(saved, Mapping) and isinstance(current, Mapping):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(saved) | set(current), key=str):
            child_path = f"{path}.{key}" if path else str(key)
            differences.extend(_differences(saved.get(key), current.get(key), child_path))
        return differences
    return [{"path": path or "value", "expected": saved, "actual": current}]
