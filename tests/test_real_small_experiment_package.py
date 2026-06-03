from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

from _pytest.capture import CaptureFixture

import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
REAL_SMALL_SCRIPT = ROOT / "scripts" / "run_real_small_experiment.py"
REAL_PACKAGE_TESTS = ROOT / "tests" / "test_real_experiment_package.py"
EXAMPLE_TEMPLATE = (
    ROOT
    / "examples"
    / "real_small_experiment"
    / "real-small-run-manifest.template.json"
)


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_real_small_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_real_small_experiment_script",
        REAL_SMALL_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_real_package_test_helpers() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "real_experiment_package_test_helpers",
        REAL_PACKAGE_TESTS,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_small_template_manifest_fails_with_missing_artifacts(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_real_small_script()
    main = cast(MainFn, getattr(module, "main"))
    output_dir = tmp_path / "real-small"
    report_path = output_dir / "run-report.json"

    assert main(
        [
            "--manifest",
            str(EXAMPLE_TEMPLATE),
            "--output-dir",
            str(output_dir),
            "--report",
            str(report_path),
        ]
    ) == 1
    payload = json.loads(capsys.readouterr().out)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload == report
    assert report["action"] == "run_real_small_experiment"
    assert report["ready"] is False
    assert report["research_ready"] is False
    assert report["final_record_written"] is False
    assert report["next_missing_artifacts"]
    assert "offline_controls.manifest_path" in {
        item["role"] for item in report["next_missing_artifacts"]
    }
    assert not (output_dir / "final" / "final-experiment-record.json").exists()


def test_real_small_real_manifest_requires_all_four_controls(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_real_small_script()
    main = cast(MainFn, getattr(module, "main"))
    manifest_path = tmp_path / "real-small-missing-controls.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "dsg-spatialqa-lab.real-small-experiment-run-manifest.v1"
                ),
                "dataset_name": "real_small_missing_controls",
                "data_source_kind": "real",
                "episodes": [],
                "offline_controls": {
                    "required_source_kinds": ["vlm"],
                },
                "predicted_dsg": {},
                "real_collection_reports": [],
                "reports": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(
        [
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--report",
            str(tmp_path / "out" / "run-report.json"),
        ]
    ) == 1
    report = json.loads(capsys.readouterr().out)
    blockers = {blocker["name"]: blocker for blocker in report["blockers"]}

    assert report["ready"] is False
    assert blockers["required_control_kinds_complete"]["missing"] == [
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
    ]
    assert "real_collection_report_required_for_real_data" in blockers


def test_real_small_synthetic_fixture_mechanical_pass_is_not_research_result(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    helper_module = load_real_package_test_helpers()
    inputs = cast(Any, getattr(helper_module, "_ready_package_inputs"))(tmp_path)
    offline_manifest_path = cast(
        Any,
        getattr(helper_module, "_offline_control_import_manifest_for_real_run"),
    )(tmp_path, inputs)
    (
        predicted_manifest_path,
        graph_eval_report_path,
        error_attribution_report_path,
    ) = cast(
        Any,
        getattr(helper_module, "_predicted_dsg_detector_run_manifest_for_real_run"),
    )(tmp_path, inputs)
    module = load_real_small_script()
    main = cast(MainFn, getattr(module, "main"))
    output_dir = tmp_path / "synthetic-run"
    manifest_path = tmp_path / "synthetic-real-small-manifest.json"
    manifest = _synthetic_real_small_manifest(
        output_dir=output_dir,
        episode_paths=inputs["episode_paths"],
        real_collection_report_path=inputs["real_collection_report_path"],
        offline_manifest_path=offline_manifest_path,
        predicted_manifest_path=predicted_manifest_path,
        active_delta_report_path=inputs["active_delta_report_path"],
        dashboard_bundle_path=inputs["dashboard_bundle_path"],
        graph_eval_report_path=graph_eval_report_path,
        error_attribution_report_path=error_attribution_report_path,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "final" / "run-report.json"

    assert main(
        [
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--report",
            str(report_path),
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    record_path = output_dir / "final" / "final-experiment-record.json"

    assert saved_report == report
    assert report["ready"] is True
    assert report["data_source_kind"] == "synthetic_test_fixture"
    assert report["not_real_research_result"] is True
    assert report["research_ready"] is False
    assert report["real_package_status"] == "synthetic_mechanical_pass"
    assert report["final_record_written"] is True
    assert report["final_record_kind"] == "synthetic_mechanical_record"
    assert record_path.exists()
    assert lab.validate_experiment_record(lab.load_experiment_record(record_path))[
        "valid"
    ] is True


def _synthetic_real_small_manifest(
    *,
    output_dir: Path,
    episode_paths: tuple[Path, ...],
    real_collection_report_path: Path,
    offline_manifest_path: Path,
    predicted_manifest_path: Path,
    active_delta_report_path: Path,
    dashboard_bundle_path: Path,
    graph_eval_report_path: Path,
    error_attribution_report_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "dsg-spatialqa-lab.real-small-experiment-run-manifest.v1",
        "dataset_name": "synthetic_real_small_fixture",
        "data_source_kind": "synthetic_test_fixture",
        "not_real_research_result": True,
        "episodes": [str(path) for path in episode_paths],
        "max_qa_per_episode": 20,
        "min_episode_count": 1,
        "min_qa_count": 8,
        "min_scene_count": 1,
        "offline_controls": {
            "manifest_path": str(offline_manifest_path),
            "required_source_kinds": [
                "vlm",
                "multi_frame_vlm",
                "caption_memory",
                "graph_text",
            ],
        },
        "predicted_dsg": {
            "detector_run_manifest_path": str(predicted_manifest_path),
        },
        "real_collection_reports": [str(real_collection_report_path)],
        "reports": {
            "active_task_delta_report": str(active_delta_report_path),
            "benchmark_manifest": str(output_dir / "benchmark-manifest.json"),
            "dashboard_bundle": str(dashboard_bundle_path),
            "error_attribution_report": str(error_attribution_report_path),
            "final_record": str(output_dir / "final" / "final-experiment-record.json"),
            "graph_eval_report": str(graph_eval_report_path),
            "readiness_report": str(output_dir / "real-experiment-readiness.json"),
            "summary_report": str(output_dir / "final" / "experiment-summary.json"),
        },
    }
