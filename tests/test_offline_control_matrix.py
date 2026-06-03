from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from _pytest.capture import CaptureFixture

import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
CONTROL_MATRIX_SCRIPT = ROOT / "scripts" / "check_offline_controls.py"
RUN_OFFLINE_CONTROLS_SCRIPT = ROOT / "scripts" / "run_offline_controls.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_control_matrix_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_offline_controls_script",
        CONTROL_MATRIX_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_run_offline_controls_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_offline_controls_script",
        RUN_OFFLINE_CONTROLS_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_offline_control_matrix_report_accepts_complete_real_control_set(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "offline_control_matrix_report")
    assert hasattr(lab, "offline_control_matrix_report_digest")
    assert hasattr(lab, "save_offline_control_matrix_report")
    assert hasattr(lab, "load_offline_control_matrix_report")
    assert hasattr(lab, "validate_offline_control_matrix_report")
    assert hasattr(lab, "compare_offline_control_matrix_report")
    import_report_paths = _write_control_import_reports(tmp_path)
    reports = tuple(
        lab.load_offline_prediction_import_report(path) for path in import_report_paths
    )
    report_path = tmp_path / "offline-control-matrix.json"

    report = lab.offline_control_matrix_report(
        reports,
        report_paths=import_report_paths,
    )
    saved_path = lab.save_offline_control_matrix_report(report, report_path)
    loaded = lab.load_offline_control_matrix_report(report_path)
    validation = lab.validate_offline_control_matrix_report(loaded)
    comparison = lab.compare_offline_control_matrix_report(loaded)

    assert report["schema_version"] == (
        "dsg-spatialqa-lab.offline-control-matrix-report.v1"
    )
    assert report["required_source_kinds"] == [
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
        "vlm",
    ]
    assert report["summary"]["missing_required_source_kinds"] == []
    assert report["summary"]["incomplete_source_keys"] == []
    assert report["readiness"]["ready"] is True
    assert report["source_kind_counts"] == {
        "caption_memory": 1,
        "graph_text": 1,
        "multi_frame_vlm": 1,
        "vlm": 1,
    }
    assert [row["source_key"] for row in report["source_profile_matrix"]] == [
        "caption_memory:caption_fixture",
        "graph_text:graph_text_fixture",
        "multi_frame_vlm:multi_frame_fixture",
        "vlm:vlm_fixture",
    ]
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["required_source_kinds_present"]["passed"] is True
    assert checks["complete_prediction_coverage"]["passed"] is True
    assert checks["qa_digest_consistent"]["passed"] is True
    assert saved_path == report_path
    assert loaded == report
    assert validation["valid"] is True
    assert comparison["matches"] is True


def test_offline_control_matrix_report_rejects_missing_or_incomplete_controls(
    tmp_path: Path,
) -> None:
    import_report_paths = _write_control_import_reports(
        tmp_path,
        source_kinds=("vlm", "caption_memory", "graph_text"),
        incomplete_source_kind="graph_text",
    )
    reports = tuple(
        lab.load_offline_prediction_import_report(path) for path in import_report_paths
    )

    report = lab.offline_control_matrix_report(
        reports,
        report_paths=import_report_paths,
    )

    assert report["readiness"]["ready"] is False
    assert report["summary"]["missing_required_source_kinds"] == ["multi_frame_vlm"]
    assert report["summary"]["incomplete_source_keys"] == [
        "graph_text:graph_text_fixture"
    ]
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["required_source_kinds_present"]["passed"] is False
    assert checks["complete_prediction_coverage"]["passed"] is False
    assert checks["complete_prediction_coverage"]["actual"] == [
        "graph_text:graph_text_fixture"
    ]


def test_offline_control_matrix_cli_writes_valid_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_control_matrix_script()
    main = cast(MainFn, getattr(module, "main"))
    import_report_paths = _write_control_import_reports(tmp_path)
    report_path = tmp_path / "offline-control-matrix.json"
    argv = [
        "--report",
        str(report_path),
    ]
    for path in import_report_paths:
        argv.extend(["--import-report", str(path)])

    assert main(argv) == 0

    output = json.loads(capsys.readouterr().out)
    report = lab.load_offline_control_matrix_report(report_path)
    assert output["action"] == "offline_control_matrix"
    assert output["path"] == str(report_path)
    assert output["ready"] is True
    assert output["report_digest"] == report["report_digest"]
    assert output["summary"] == report["summary"]

    assert main(["--validate-report", str(report_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_offline_control_matrix_report"
    assert validation["valid"] is True

    assert main(["--compare-report", str(report_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_offline_control_matrix_report"
    assert comparison["matches"] is True


def test_offline_control_matrix_cli_returns_structured_json_for_missing_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_control_matrix_script()
    main = cast(MainFn, getattr(module, "main"))
    missing_path = tmp_path / "missing-import-report.json"
    report_path = tmp_path / "offline-control-matrix.json"

    assert main(
        [
            "--import-report",
            str(missing_path),
            "--report",
            str(report_path),
        ]
    ) == 1

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "offline_control_matrix"
    assert output["path"] == str(report_path)
    assert output["valid"] is False
    assert "missing-import-report.json" in output["error"]


def test_run_offline_control_imports_writes_all_reports_and_matrix(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "run_offline_control_imports")
    qa_path, source_specs = _write_control_input_specs(tmp_path)
    matrix_path = tmp_path / "offline-controls" / "offline-control-matrix.json"

    result = lab.run_offline_control_imports(
        qa_path=qa_path,
        source_specs=source_specs,
        output_dir=tmp_path / "offline-controls",
        matrix_report_path=matrix_path,
    )
    matrix_report = lab.load_offline_control_matrix_report(matrix_path)

    assert result["schema_version"] == (
        "dsg-spatialqa-lab.offline-control-import-run.v1"
    )
    assert result["action"] == "run_offline_control_imports"
    assert result["ready"] is True
    assert result["matrix_report_path"] == str(matrix_path)
    assert result["matrix_report_digest"] == matrix_report["report_digest"]
    assert result["matrix_readiness"] == matrix_report["readiness"]
    assert result["readiness"]["ready"] is True
    assert result["source_metadata_summary"] == {
        "missing_metadata_source_keys": [],
        "placeholder_source_keys": [],
    }
    assert result["summary"] == matrix_report["summary"]
    assert [source["source_kind"] for source in result["sources"]] == [
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
        "vlm",
    ]
    for source in result["sources"]:
        assert Path(source["prediction_path"]).exists()
        assert Path(source["import_report_path"]).exists()
        import_report = lab.load_offline_prediction_import_report(
            source["import_report_path"]
        )
        assert source["import_report_digest"] == import_report["report_digest"]
        assert import_report["summary"]["missing_case_count"] == 0
    assert lab.validate_offline_control_matrix_report(matrix_report)["valid"] is True
    assert lab.compare_offline_control_matrix_report(matrix_report)["matches"] is True


def test_run_offline_control_imports_writes_optional_qa_eval_delta_reports(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "OFFLINE_CONTROL_RESULT_REPORT_SCHEMA_VERSION")
    assert hasattr(lab, "load_offline_control_result_report")
    assert hasattr(lab, "validate_offline_control_result_report")
    assert hasattr(lab, "compare_offline_control_result_report")
    qa_path, source_specs = _write_control_input_specs(tmp_path)
    candidate_prediction_path = _write_candidate_prediction_file(
        tmp_path,
        qa_path,
        name="predicted_graph_tool",
    )
    matrix_path = tmp_path / "offline-controls" / "offline-control-matrix.json"
    qa_eval_output_dir = tmp_path / "offline-controls" / "qa-eval"
    result_report_path = tmp_path / "offline-controls" / "offline-control-result.json"

    result = lab.run_offline_control_imports(
        qa_path=qa_path,
        source_specs=source_specs,
        output_dir=tmp_path / "offline-controls",
        matrix_report_path=matrix_path,
        candidate_prediction_path=candidate_prediction_path,
        candidate_name="predicted_graph_tool",
        qa_eval_output_dir=qa_eval_output_dir,
        result_report_path=result_report_path,
    )

    handoff = result["qa_eval_handoff"]
    assert handoff["requested"] is True
    assert handoff["ready"] is True
    assert handoff["candidate_name"] == "predicted_graph_tool"
    assert handoff["candidate_prediction_path"] == str(candidate_prediction_path)
    assert Path(handoff["candidate_qa_eval_report_path"]).exists()
    candidate_report = lab.load_qa_eval_report(
        handoff["candidate_qa_eval_report_path"]
    )
    assert handoff["candidate_qa_eval_report_digest"] == candidate_report[
        "report_digest"
    ]
    assert lab.validate_qa_eval_report(candidate_report)["valid"] is True
    assert result["candidate_qa_eval_report_path"] == handoff[
        "candidate_qa_eval_report_path"
    ]
    assert set(result["qa_eval_delta_report_paths"]) == {
        "caption_memory:caption_memory_ai2thor_trial",
        "graph_text:graph_text_ai2thor_trial",
        "multi_frame_vlm:llava16_multiframe_ai2thor_trial",
        "vlm:llava16_ai2thor_trial",
    }
    delta_reports = [
        lab.load_qa_eval_delta_report(path)
        for path in result["qa_eval_delta_report_paths"].values()
    ]
    assert sorted(report["baseline_name"] for report in delta_reports) == [
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
        "vlm",
    ]
    for report in delta_reports:
        assert report["candidate_name"] == "predicted_graph_tool"
        assert report["candidate_report_path"] == handoff[
            "candidate_qa_eval_report_path"
        ]
        assert Path(str(report["baseline_report_path"])).exists()
        assert lab.validate_qa_eval_delta_report(report)["valid"] is True
    assert result["qa_eval_delta_reports"] == handoff["qa_eval_delta_reports"]
    assert result["offline_control_result_report_path"] == str(result_report_path)
    result_report = lab.load_offline_control_result_report(result_report_path)
    assert result_report["schema_version"] == (
        "dsg-spatialqa-lab.offline-control-result-report.v1"
    )
    assert result_report["candidate_name"] == "predicted_graph_tool"
    assert result_report["matrix_report_path"] == str(matrix_path)
    assert result_report["matrix_report_digest"] == result["matrix_report_digest"]
    assert result_report["candidate_qa_eval_report_path"] == handoff[
        "candidate_qa_eval_report_path"
    ]
    assert result_report["summary"] == {
        "candidate_exact_match_rate": 1.0,
        "delta_report_count": 4,
        "improved_source_count": 0,
        "regressed_source_count": 0,
        "source_count": 4,
        "unchanged_source_count": 4,
    }
    assert result_report["readiness"]["ready"] is True
    assert [row["source_kind"] for row in result_report["source_result_matrix"]] == [
        "caption_memory",
        "graph_text",
        "multi_frame_vlm",
        "vlm",
    ]
    assert {
        row["source_key"]: row["exact_match_rate_delta"]
        for row in result_report["source_result_matrix"]
    } == {
        "caption_memory:caption_memory_ai2thor_trial": 0.0,
        "graph_text:graph_text_ai2thor_trial": 0.0,
        "multi_frame_vlm:llava16_multiframe_ai2thor_trial": 0.0,
        "vlm:llava16_ai2thor_trial": 0.0,
    }
    assert result["offline_control_result_report_digest"] == result_report[
        "report_digest"
    ]
    assert result["offline_control_result_readiness"] == result_report["readiness"]
    assert lab.validate_offline_control_result_report(result_report)["valid"] is True
    assert lab.compare_offline_control_result_report(result_report)["matches"] is True


def test_run_offline_control_imports_accepts_qa_prediction_source_inputs(
    tmp_path: Path,
) -> None:
    qa_path, source_specs = _write_control_input_specs(
        tmp_path,
        input_format="qa_prediction",
    )
    matrix_path = tmp_path / "offline-controls" / "offline-control-matrix.json"

    result = lab.run_offline_control_imports(
        qa_path=qa_path,
        source_specs=source_specs,
        output_dir=tmp_path / "offline-controls",
        matrix_report_path=matrix_path,
    )

    assert result["ready"] is True
    assert [source["input_format"] for source in result["sources"]] == [
        "qa_prediction",
        "qa_prediction",
        "qa_prediction",
        "qa_prediction",
    ]
    for source in result["sources"]:
        import_report = lab.load_offline_prediction_import_report(
            source["import_report_path"]
        )
        input_predictions = tuple(lab.load_qa_predictions(source["input_path"]))
        assert import_report["input_format"] == "qa_prediction"
        assert import_report["input_digest"] == lab.qa_predictions_digest(
            input_predictions
        )
        assert source["prediction_digest"] == lab.qa_predictions_digest(
            input_predictions
        )
        assert lab.validate_offline_prediction_import_report(import_report)[
            "valid"
        ] is True
        assert lab.compare_offline_prediction_import_report(import_report)[
            "matches"
        ] is True


def test_run_offline_control_imports_rejects_placeholder_source_identity(
    tmp_path: Path,
) -> None:
    qa_path, source_specs = _write_control_input_specs(
        tmp_path,
        placeholder_sources=True,
        source_metadata={"model_id": "mock-vlm"},
    )
    matrix_path = tmp_path / "offline-controls" / "offline-control-matrix.json"

    result = lab.run_offline_control_imports(
        qa_path=qa_path,
        source_specs=source_specs,
        output_dir=tmp_path / "offline-controls",
        matrix_report_path=matrix_path,
    )

    expected_source_keys = [
        "caption_memory:caption_fixture",
        "graph_text:graph_text_fixture",
        "multi_frame_vlm:multi_frame_fixture",
        "vlm:vlm_fixture",
    ]
    assert result["ready"] is False
    assert result["matrix_readiness"]["ready"] is True
    assert result["readiness"]["failed_checks"] == [
        "real_source_metadata_present",
        "no_placeholder_source_identity",
    ]
    assert result["source_metadata_summary"] == {
        "missing_metadata_source_keys": expected_source_keys,
        "placeholder_source_keys": expected_source_keys,
    }


def test_run_offline_control_imports_keeps_matrix_diagnostics_when_incomplete(
    tmp_path: Path,
) -> None:
    qa_path, source_specs = _write_control_input_specs(
        tmp_path,
        source_kinds=("vlm", "caption_memory", "graph_text"),
        incomplete_source_kind="graph_text",
    )
    matrix_path = tmp_path / "offline-controls" / "offline-control-matrix.json"

    result = lab.run_offline_control_imports(
        qa_path=qa_path,
        source_specs=source_specs,
        output_dir=tmp_path / "offline-controls",
        matrix_report_path=matrix_path,
    )

    assert result["ready"] is False
    assert matrix_path.exists()
    assert result["summary"]["missing_required_source_kinds"] == ["multi_frame_vlm"]
    assert result["summary"]["incomplete_source_keys"] == [
        "graph_text:graph_text_ai2thor_trial"
    ]
    assert result["readiness"]["failed_checks"] == [
        "required_source_kinds_present",
        "complete_prediction_coverage",
    ]


def test_run_offline_controls_cli_imports_four_sources_and_writes_matrix(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_offline_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    qa_path, source_specs = _write_control_input_specs(tmp_path)
    matrix_path = tmp_path / "cli-controls" / "offline-control-matrix.json"
    argv = [
        "--qa",
        str(qa_path),
        "--output-dir",
        str(tmp_path / "cli-controls"),
        "--matrix-report",
        str(matrix_path),
    ]
    for source in source_specs:
        argv.extend(
            [
                "--source",
                str(source["source_kind"]),
                str(source["source_name"]),
                str(source["input_path"]),
            ]
        )
        metadata = cast(dict[str, object], source["metadata"])
        for key, value in sorted(metadata.items()):
            if key == "capabilities":
                continue
            argv.extend(
                [
                    "--source-metadata",
                    str(source["source_name"]),
                    f"{key}={value}",
                ]
            )

    assert main(argv) == 0

    output = json.loads(capsys.readouterr().out)
    matrix_report = lab.load_offline_control_matrix_report(matrix_path)
    assert output["action"] == "run_offline_control_imports"
    assert output["ready"] is True
    assert output["matrix_report_digest"] == matrix_report["report_digest"]
    assert len(output["sources"]) == 4
    assert output["source_metadata_summary"] == {
        "missing_metadata_source_keys": [],
        "placeholder_source_keys": [],
    }


def test_run_offline_controls_cli_writes_optional_qa_eval_delta_reports(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_offline_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    qa_path, source_specs = _write_control_input_specs(tmp_path)
    candidate_prediction_path = _write_candidate_prediction_file(
        tmp_path,
        qa_path,
        name="predicted_graph_tool",
    )
    matrix_path = tmp_path / "cli-controls" / "offline-control-matrix.json"
    qa_eval_output_dir = tmp_path / "cli-controls" / "qa-eval"
    result_report_path = tmp_path / "cli-controls" / "offline-control-result.json"
    argv = [
        "--qa",
        str(qa_path),
        "--output-dir",
        str(tmp_path / "cli-controls"),
        "--matrix-report",
        str(matrix_path),
        "--candidate-prediction",
        str(candidate_prediction_path),
        "--candidate-name",
        "predicted_graph_tool",
        "--qa-eval-output-dir",
        str(qa_eval_output_dir),
        "--result-report",
        str(result_report_path),
    ]
    for source in source_specs:
        argv.extend(
            [
                "--source",
                str(source["source_kind"]),
                str(source["source_name"]),
                str(source["input_path"]),
            ]
        )
        metadata = cast(dict[str, object], source["metadata"])
        for key, value in sorted(metadata.items()):
            if key == "capabilities":
                continue
            argv.extend(
                [
                    "--source-metadata",
                    str(source["source_name"]),
                    f"{key}={value}",
                ]
            )

    assert main(argv) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["ready"] is True
    assert output["qa_eval_handoff"]["ready"] is True
    assert Path(output["candidate_qa_eval_report_path"]).exists()
    assert output["offline_control_result_report_path"] == str(result_report_path)
    assert Path(output["offline_control_result_report_path"]).exists()
    assert len(output["qa_eval_delta_report_paths"]) == 4
    assert sorted(
        lab.load_qa_eval_delta_report(path)["baseline_name"]
        for path in output["qa_eval_delta_report_paths"].values()
    ) == ["caption_memory", "graph_text", "multi_frame_vlm", "vlm"]
    result_report = lab.load_offline_control_result_report(result_report_path)
    assert result_report["readiness"]["ready"] is True


def test_offline_control_matrix_cli_validates_result_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    run_module = load_run_offline_controls_script()
    run_main = cast(MainFn, getattr(run_module, "main"))
    check_module = load_control_matrix_script()
    check_main = cast(MainFn, getattr(check_module, "main"))
    qa_path, source_specs = _write_control_input_specs(tmp_path)
    candidate_prediction_path = _write_candidate_prediction_file(
        tmp_path,
        qa_path,
        name="predicted_graph_tool",
    )
    matrix_path = tmp_path / "controls" / "offline-control-matrix.json"
    result_report_path = tmp_path / "controls" / "offline-control-result.json"
    argv = [
        "--qa",
        str(qa_path),
        "--output-dir",
        str(tmp_path / "controls"),
        "--matrix-report",
        str(matrix_path),
        "--candidate-prediction",
        str(candidate_prediction_path),
        "--candidate-name",
        "predicted_graph_tool",
        "--result-report",
        str(result_report_path),
    ]
    for source in source_specs:
        argv.extend(
            [
                "--source",
                str(source["source_kind"]),
                str(source["source_name"]),
                str(source["input_path"]),
            ]
        )
        metadata = cast(dict[str, object], source["metadata"])
        for key, value in sorted(metadata.items()):
            if key == "capabilities":
                continue
            argv.extend(
                [
                    "--source-metadata",
                    str(source["source_name"]),
                    f"{key}={value}",
                ]
            )
    assert run_main(argv) == 0
    capsys.readouterr()

    assert check_main(["--validate-result-report", str(result_report_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_offline_control_result_report"
    assert validation["valid"] is True

    assert check_main(["--compare-result-report", str(result_report_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_offline_control_result_report"
    assert comparison["matches"] is True


def test_run_offline_controls_cli_accepts_qa_prediction_source_inputs(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_run_offline_controls_script()
    main = cast(MainFn, getattr(module, "main"))
    qa_path, source_specs = _write_control_input_specs(
        tmp_path,
        input_format="qa_prediction",
    )
    matrix_path = tmp_path / "cli-controls" / "offline-control-matrix.json"
    argv = [
        "--qa",
        str(qa_path),
        "--output-dir",
        str(tmp_path / "cli-controls"),
        "--matrix-report",
        str(matrix_path),
    ]
    for source in source_specs:
        argv.extend(
            [
                "--source",
                str(source["source_kind"]),
                str(source["source_name"]),
                str(source["input_path"]),
                "--source-input-format",
                str(source["source_name"]),
                "qa_prediction",
            ]
        )
        metadata = cast(dict[str, object], source["metadata"])
        for key, value in sorted(metadata.items()):
            if key == "capabilities":
                continue
            argv.extend(
                [
                    "--source-metadata",
                    str(source["source_name"]),
                    f"{key}={value}",
                ]
            )

    assert main(argv) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["ready"] is True
    assert {source["input_format"] for source in output["sources"]} == {
        "qa_prediction"
    }
    assert all(
        lab.compare_offline_prediction_import_report(
            lab.load_offline_prediction_import_report(source["import_report_path"])
        )["matches"]
        is True
        for source in output["sources"]
    )


def _write_control_import_reports(
    tmp_path: Path,
    *,
    source_kinds: tuple[str, ...] = (
        "vlm",
        "multi_frame_vlm",
        "caption_memory",
        "graph_text",
    ),
    incomplete_source_kind: str | None = None,
) -> tuple[Path, ...]:
    cases = _cases()
    qa_path = tmp_path / "qa.jsonl"
    lab.save_qa_dataset(cases, qa_path)
    paths: list[Path] = []
    for source_kind in source_kinds:
        source_name = _source_name(source_kind)
        source_dir = tmp_path / source_kind
        input_path = source_dir / "offline-input.jsonl"
        prediction_path = source_dir / "predictions.jsonl"
        report_path = source_dir / "import-report.json"
        records = _records(cases)
        if source_kind == incomplete_source_kind:
            records = records[:-1]
        lab.save_offline_prediction_records(records, input_path)
        predictions, report = lab.import_offline_predictions(
            cases,
            records,
            source_name=source_name,
            source_kind=source_kind,
            source_metadata={
                "capabilities": ("spatial_qa", "dynamic_memory", "graph_tool_query"),
                "dataset_id": "real_smoke_fixture",
                "model_id": f"{source_kind}-model",
                "prompt_id": f"{source_kind}-prompt",
            },
            qa_path=qa_path,
            input_path=input_path,
            prediction_path=prediction_path,
        )
        lab.save_qa_predictions(predictions, prediction_path)
        lab.save_offline_prediction_import_report(report, report_path)
        paths.append(report_path)
    return tuple(paths)


def _write_control_input_specs(
    tmp_path: Path,
    *,
    source_kinds: tuple[str, ...] = (
        "vlm",
        "multi_frame_vlm",
        "caption_memory",
        "graph_text",
    ),
    incomplete_source_kind: str | None = None,
    input_format: str = "offline_prediction_record",
    placeholder_sources: bool = False,
    source_metadata: dict[str, object] | None = None,
) -> tuple[Path, tuple[dict[str, object], ...]]:
    cases = _cases()
    qa_path = tmp_path / "qa.jsonl"
    lab.save_qa_dataset(cases, qa_path)
    specs: list[dict[str, object]] = []
    for source_kind in source_kinds:
        source_name = (
            _source_name(source_kind)
            if placeholder_sources
            else _real_source_name(source_kind)
        )
        source_dir = tmp_path / "inputs" / source_kind
        input_path = source_dir / "offline-input.jsonl"
        records = _records(cases)
        if source_kind == incomplete_source_kind:
            records = records[:-1]
        spec: dict[str, object] = {
            "source_kind": source_kind,
            "source_name": source_name,
            "input_path": input_path,
            "metadata": source_metadata
            if source_metadata is not None
            else _real_source_metadata(source_kind),
        }
        if input_format == "qa_prediction":
            lab.save_qa_predictions(_predictions_from_records(records), input_path)
            spec["input_format"] = "qa_prediction"
        else:
            lab.save_offline_prediction_records(records, input_path)
        specs.append(spec)
    return qa_path, tuple(specs)


def _predictions_from_records(
    records: tuple[lab.OfflinePredictionRecord, ...],
) -> tuple[lab.QAPrediction, ...]:
    return tuple(
        lab.QAPrediction(
            id=record.case_id,
            answer=dict(record.answer),
            evidence_nodes=record.evidence_nodes,
            evidence_edges=record.evidence_edges,
            confidence=record.confidence,
            error=record.error,
        )
        for record in records
    )


def _write_candidate_prediction_file(
    tmp_path: Path,
    qa_path: Path,
    *,
    name: str,
) -> Path:
    cases = tuple(lab.load_qa_dataset(qa_path))
    predictions = tuple(
        lab.QAPrediction(
            id=case.id,
            answer=case.answer,
            evidence_nodes=case.required_nodes,
            evidence_edges=case.required_edges,
            confidence=0.91,
        )
        for case in cases
    )
    path = tmp_path / "candidate" / f"{name}.jsonl"
    lab.save_qa_predictions(predictions, path)
    return path


def _real_source_name(source_kind: str) -> str:
    names = {
        "caption_memory": "caption_memory_ai2thor_trial",
        "graph_text": "graph_text_ai2thor_trial",
        "multi_frame_vlm": "llava16_multiframe_ai2thor_trial",
        "vlm": "llava16_ai2thor_trial",
    }
    return names[source_kind]


def _real_source_metadata(source_kind: str) -> dict[str, object]:
    model_ids = {
        "caption_memory": "blip2-flan-t5-xl",
        "graph_text": "gpt-4.1-mini",
        "multi_frame_vlm": "llava-v1.6-34b",
        "vlm": "llava-v1.6-34b",
    }
    prompt_ids = {
        "caption_memory": "caption-memory-spatial-v1",
        "graph_text": "graph-text-spatial-qa-v1",
        "multi_frame_vlm": "multi-frame-vlm-spatial-qa-v1",
        "vlm": "vlm-spatial-qa-v1",
    }
    return {
        "capabilities": ("spatial_qa", "dynamic_memory"),
        "dataset_id": "ai2thor-real-trial-v1",
        "model_id": model_ids[source_kind],
        "prompt_id": prompt_ids[source_kind],
    }


def _cases() -> tuple[lab.QACase, ...]:
    graph = lab.load_scene_fixture("tabletop")
    generated = lab.generate_qa_cases(
        graph,
        scene_id="tabletop_scene",
        episode_id="episode_001",
    )
    location = next(case for case in generated if case.question_type == "object_location")
    relation = next(case for case in generated if case.question_type == "relative_relation")
    nearest = next(case for case in generated if case.question_type == "nearest_object")
    return (location, relation, nearest)


def _records(cases: tuple[lab.QACase, ...]) -> tuple[lab.OfflinePredictionRecord, ...]:
    return tuple(
        lab.OfflinePredictionRecord(
            case_id=case.id,
            answer=case.answer,
            evidence_nodes=case.required_nodes,
            evidence_edges=case.required_edges,
            confidence=0.8,
        )
        for case in cases
    )


def _source_name(source_kind: str) -> str:
    names = {
        "caption_memory": "caption_fixture",
        "graph_text": "graph_text_fixture",
        "multi_frame_vlm": "multi_frame_fixture",
        "vlm": "vlm_fixture",
    }
    return names[source_kind]
