from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab.benchmark.manifest import (
    build_benchmark_artifacts,
    save_benchmark_manifest,
)
from dsg_spatialqa_lab.benchmark.readiness import (
    real_experiment_readiness_report,
    save_real_experiment_readiness_report,
)


REAL_EXPERIMENT_PACKAGE_SCHEMA_VERSION = "dsg-spatialqa-lab.real-experiment-package.v1"


def assemble_real_experiment_package(
    *,
    dataset_name: str,
    episode_paths: Sequence[str | Path],
    output_dir: str | Path,
    manifest_path: str | Path,
    readiness_report_path: str | Path,
    max_qa_per_episode: int | None = None,
    tags: Sequence[str] = ("benchmark", "real"),
    declared_data_source_kind: str = "real",
    min_episode_count: int = 3,
    min_scene_count: int = 1,
    min_qa_count: int = 30,
    required_control_kinds: Sequence[str] = (
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
        "vlm",
    ),
    required_predicted_input_kinds: Sequence[str] = ("observation_sequence",),
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
    manifest = build_benchmark_artifacts(
        dataset_name=dataset_name,
        episode_paths=episode_paths,
        output_dir=output_dir,
        max_qa_per_episode=max_qa_per_episode,
        tags=tags,
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
    )
    manifest_output_path = save_benchmark_manifest(manifest, manifest_path)
    readiness_report = real_experiment_readiness_report(
        manifest,
        manifest_path=manifest_output_path,
        declared_data_source_kind=declared_data_source_kind,
        min_episode_count=min_episode_count,
        min_scene_count=min_scene_count,
        min_qa_count=min_qa_count,
        required_control_kinds=required_control_kinds,
        required_predicted_input_kinds=required_predicted_input_kinds,
    )
    readiness_output_path = save_real_experiment_readiness_report(
        readiness_report,
        readiness_report_path,
    )
    return _package_result(
        dataset_name=dataset_name,
        manifest_path=manifest_output_path,
        readiness_report_path=readiness_output_path,
        manifest=manifest,
        readiness_report=readiness_report,
    )


def _package_result(
    *,
    dataset_name: str,
    manifest_path: Path,
    readiness_report_path: Path,
    manifest: Mapping[str, Any],
    readiness_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REAL_EXPERIMENT_PACKAGE_SCHEMA_VERSION,
        "action": "assemble_real_experiment_package",
        "dataset_name": dataset_name,
        "manifest_path": str(manifest_path),
        "readiness_report_path": str(readiness_report_path),
        "manifest_digest": manifest["manifest_digest"],
        "readiness_report_digest": readiness_report["report_digest"],
        "ready": readiness_report["readiness"]["ready"],
        "readiness": readiness_report["readiness"],
        "summary": manifest["summary"],
    }
