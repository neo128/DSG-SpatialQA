from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest
from _pytest.capture import CaptureFixture

from dsg_spatialqa_lab import (
    evaluation_bundle,
    evaluation_bundle_json,
    evaluation_manifest,
    evaluation_manifest_json,
    evaluation_report,
    evaluation_report_json,
    run_evaluation_suite,
)


ROOT = Path(__file__).resolve().parents[1]
EVALUATE_SCRIPT = ROOT / "scripts" / "evaluate.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_evaluate_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_script", EVALUATE_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assert_invalid_artifact_report(
    main: MainFn,
    capsys: CaptureFixture[str],
    args: list[str],
    expected: dict[str, object],
) -> None:
    try:
        result = main(args)
    except Exception as exc:
        pytest.fail(f"expected structured artifact error, got {type(exc).__name__}: {exc}")

    assert result == 1
    assert json.loads(capsys.readouterr().out) == expected


def test_evaluate_cli_prints_filtered_report_json(
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))

    assert main(["--name", "tabletop_object_location"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["summary"] == {
        "total": 1,
        "passed": 1,
        "failed": 0,
        "failed_cases": [],
        "selected_cases": ["tabletop_object_location"],
    }
    assert report["metrics"] == {
        "case_count": 1,
        "passed_case_count": 1,
        "failed_case_count": 0,
        "pass_rate": 1.0,
        "failure_rate": 0.0,
        "by_kind": {
            "qa": {
                "case_count": 1,
                "passed_case_count": 1,
                "failed_case_count": 0,
                "pass_rate": 1.0,
                "failure_rate": 0.0,
            }
        },
        "by_question_type": {
            "object_location": {
                "case_count": 1,
                "passed_case_count": 1,
                "failed_case_count": 0,
                "pass_rate": 1.0,
                "failure_rate": 0.0,
            }
        },
        "by_scene_fixture": {
            "tabletop": {
                "case_count": 1,
                "passed_case_count": 1,
                "failed_case_count": 0,
                "pass_rate": 1.0,
                "failure_rate": 0.0,
            }
        },
        "by_tag": {
            "memory": {
                "case_count": 1,
                "passed_case_count": 1,
                "failed_case_count": 0,
                "pass_rate": 1.0,
                "failure_rate": 0.0,
            },
            "qa": {
                "case_count": 1,
                "passed_case_count": 1,
                "failed_case_count": 0,
                "pass_rate": 1.0,
                "failure_rate": 0.0,
            },
        },
    }
    assert report["failed_cases"] == []
    assert report["runtime_error_categories"] == []


def test_evaluate_cli_filters_by_tags_kinds_and_question_types(
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))

    assert main(["--tag", "qa", "--tag", "dynamic", "--question-type", "scene_delta"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["summary"]["selected_cases"] == [
        "moved_mug_scene_delta",
        "moved_mug_scene_delta_reversed_window_error",
        "multi_room_rearrangement_scene_delta",
    ]
    assert report["summary"]["passed"] == 3


def test_evaluate_cli_writes_report_to_explicit_path(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    report_path = tmp_path / "reports" / "qa-report.json"

    assert main(["--kind", "vla_pick", "--report", str(report_path)]) == 0

    stdout_report = json.loads(capsys.readouterr().out)
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report == stdout_report
    assert saved_report["summary"]["selected_cases"] == [
        "ambiguous_mug_pick_by_label",
        "needs_reobserve_spoon_pick",
        "tabletop_missing_object_pick_error",
        "tabletop_mug_pick",
    ]


def test_evaluate_cli_filters_vla_error_cases(capsys: CaptureFixture[str]) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))

    assert main(["--tag", "vla", "--tag", "error"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["summary"]["selected_cases"] == [
        "tabletop_missing_object_pick_error",
        "tabletop_missing_reference_place_error",
        "tabletop_unsupported_place_relation_error",
    ]
    assert report["summary"]["passed"] == 3
    assert report["runtime_error_categories"] == [
        {
            "category": "missing_object",
            "count": 2,
            "cases": [
                "tabletop_missing_object_pick_error",
                "tabletop_missing_reference_place_error",
            ],
        },
        {
            "category": "unsupported_relation",
            "count": 1,
            "cases": ["tabletop_unsupported_place_relation_error"],
        },
    ]


def test_evaluate_cli_compares_report_with_current_run(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    report = evaluation_report(run_evaluation_suite(names=("tabletop_object_location",)))
    report_path = tmp_path / "report.json"
    report_path.write_text(evaluation_report_json(report), encoding="utf-8")

    assert main(["--compare-report", str(report_path)]) == 0

    comparison = json.loads(capsys.readouterr().out)
    assert comparison["matches"] is True
    assert comparison["filters"] == {"names": ["tabletop_object_location"]}
    assert comparison["saved_digest"] == report["digest"]
    assert comparison["current_digest"] == report["digest"]


def test_evaluate_cli_reports_report_current_run_drift(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    report = evaluation_report(run_evaluation_suite(names=("tabletop_object_location",)))
    drifted_report = json.loads(evaluation_report_json(report))
    drifted_report["summary"]["selected_cases"] = ["tabletop_mug_pick"]
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(drifted_report), encoding="utf-8")

    assert main(["--compare-report", str(report_path)]) == 1

    comparison = json.loads(capsys.readouterr().out)
    assert comparison["matches"] is False
    assert comparison["filters"] == {"names": ["tabletop_mug_pick"]}
    assert comparison["checks"][0]["name"] == "report_digest_matches_current"
    assert comparison["checks"][0]["passed"] is False


def test_evaluate_cli_reports_invalid_explicit_report_file(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    report_path = tmp_path / "report.json"
    report_path.write_text("[]\n", encoding="utf-8")

    assert_invalid_artifact_report(
        main,
        capsys,
        ["--compare-report", str(report_path)],
        {
            "action": "compare_report",
            "path": str(report_path),
            "valid": False,
            "matches": False,
            "error": "Evaluation report JSON must be an object",
        },
    )


def test_evaluate_cli_prints_benchmark_bundle_json(
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))

    assert main(["--bundle", "--tag", "qa", "--tag", "reobserve"]) == 0

    bundle = json.loads(capsys.readouterr().out)
    assert bundle["schema_version"] == "dsg-spatialqa-lab.evaluation-bundle.v1"
    assert bundle["filters"] == {
        "names": [],
        "tags": ["qa", "reobserve"],
        "kinds": [],
        "question_types": [],
    }
    assert [case["name"] for case in bundle["evaluation_cases"]] == [
        "multi_room_rearrangement_reobserve_targets",
        "needs_reobserve_targets",
    ]
    assert bundle["report"]["summary"]["passed"] == 2


def test_evaluate_cli_prints_filtered_manifest_without_suite_execution(
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))

    assert main(["--manifest", "--tag", "qa", "--tag", "relations"]) == 0

    manifest = json.loads(capsys.readouterr().out)
    assert manifest == evaluation_manifest(tags=("qa", "relations"))
    assert [case["name"] for case in manifest["evaluation_cases"]] == [
        "relation_shift_relation_timeline",
        "tabletop_relation_timeline",
        "tabletop_relative_relation_mug_left_of_plate",
    ]
    assert manifest["coverage"]["by_question_type"] == {
        "relation_timeline": 2,
        "relative_relation": 1,
    }
    assert "suite" not in manifest
    assert "report" not in manifest


def test_evaluate_cli_writes_manifest_to_explicit_path(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    manifest_path = tmp_path / "manifests" / "scene-delta.json"

    assert main(
        [
            "--manifest",
            "--question-type",
            "scene_delta",
            "--report",
            str(manifest_path),
        ]
    ) == 0

    stdout_manifest = json.loads(capsys.readouterr().out)
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved_manifest == stdout_manifest
    assert saved_manifest == json.loads(
        evaluation_manifest_json(evaluation_manifest(question_types=("scene_delta",)))
    )


def test_evaluate_cli_validates_explicit_manifest_file(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    manifest = evaluation_manifest(tags=("qa", "relations"))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(evaluation_manifest_json(manifest), encoding="utf-8")

    assert main(["--validate-manifest", str(manifest_path)]) == 0

    validation = json.loads(capsys.readouterr().out)
    assert validation["valid"] is True
    assert validation["digest"] == manifest["digest"]
    assert validation["checks"][1]["name"] == "manifest_digest"


def test_evaluate_cli_rejects_tampered_manifest_digest(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    manifest = evaluation_manifest(tags=("qa", "relations"))
    tampered_manifest = json.loads(evaluation_manifest_json(manifest))
    tampered_manifest["digest"] = "0" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")

    assert main(["--validate-manifest", str(manifest_path)]) == 1

    validation = json.loads(capsys.readouterr().out)
    assert validation["valid"] is False
    assert validation["checks"][1]["name"] == "manifest_digest"
    assert validation["checks"][1]["passed"] is False


def test_evaluate_cli_rejects_tampered_manifest_coverage_with_differences(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    manifest = evaluation_manifest(tags=("qa", "relations"))
    tampered_manifest = json.loads(evaluation_manifest_json(manifest))
    tampered_manifest["coverage"]["by_tag"]["relations"] = 2
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")

    assert main(["--validate-manifest", str(manifest_path)]) == 1

    validation = json.loads(capsys.readouterr().out)
    assert validation["valid"] is False
    assert validation["checks"][-1]["name"] == "coverage_matches_manifest"
    assert validation["checks"][-1]["differences"] == [
        {"path": "by_tag.relations", "expected": 3, "actual": 2}
    ]


def test_evaluate_cli_reports_invalid_explicit_manifest_file_for_validation(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("[]\n", encoding="utf-8")

    assert_invalid_artifact_report(
        main,
        capsys,
        ["--validate-manifest", str(manifest_path)],
        {
            "action": "validate_manifest",
            "path": str(manifest_path),
            "valid": False,
            "error": "Evaluation manifest JSON must be an object",
        },
    )


def test_evaluate_cli_compares_manifest_with_current_metadata(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    manifest = evaluation_manifest(tags=("qa", "relations"))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(evaluation_manifest_json(manifest), encoding="utf-8")

    assert main(["--compare-manifest", str(manifest_path)]) == 0

    comparison = json.loads(capsys.readouterr().out)
    assert comparison["matches"] is True
    assert comparison["saved_digest"] == manifest["digest"]
    assert comparison["current_digest"] == manifest["digest"]
    assert comparison["checks"][1]["name"] == "manifest_digest_matches_current"


def test_evaluate_cli_reports_manifest_current_metadata_drift(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    manifest = evaluation_manifest(tags=("qa", "relations"))
    drifted_manifest = json.loads(evaluation_manifest_json(manifest))
    drifted_manifest["filters"]["tags"] = ["qa", "dynamic"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(drifted_manifest), encoding="utf-8")

    assert main(["--compare-manifest", str(manifest_path)]) == 1

    comparison = json.loads(capsys.readouterr().out)
    assert comparison["matches"] is False
    assert comparison["checks"][0]["name"] == "saved_manifest_valid"
    assert comparison["checks"][0]["passed"] is False
    assert comparison["checks"][1]["name"] == "manifest_digest_matches_current"
    assert comparison["checks"][1]["passed"] is False


def test_evaluate_cli_reports_invalid_explicit_manifest_file_for_compare(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("[]\n", encoding="utf-8")

    assert_invalid_artifact_report(
        main,
        capsys,
        ["--compare-manifest", str(manifest_path)],
        {
            "action": "compare_manifest",
            "path": str(manifest_path),
            "valid": False,
            "matches": False,
            "error": "Evaluation manifest JSON must be an object",
        },
    )


def test_evaluate_cli_validates_explicit_bundle_file(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    bundle = evaluation_bundle(tags=("qa", "reobserve"))
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(evaluation_bundle_json(bundle), encoding="utf-8")

    assert main(["--validate-bundle", str(bundle_path)]) == 0

    validation = json.loads(capsys.readouterr().out)
    assert validation["valid"] is True
    assert validation["digest"] == bundle["suite"]["digest"]


def test_evaluate_cli_reports_invalid_explicit_bundle_file_for_validation(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text("[]\n", encoding="utf-8")

    assert_invalid_artifact_report(
        main,
        capsys,
        ["--validate-bundle", str(bundle_path)],
        {
            "action": "validate_bundle",
            "path": str(bundle_path),
            "valid": False,
            "error": "Evaluation bundle JSON must be an object",
        },
    )


def test_evaluate_cli_rejects_tampered_bundle_digest(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    bundle = evaluation_bundle(tags=("qa", "reobserve"))
    tampered_bundle = json.loads(evaluation_bundle_json(bundle))
    tampered_bundle["suite"]["digest"] = "0" * 64
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(tampered_bundle), encoding="utf-8")

    assert main(["--validate-bundle", str(bundle_path)]) == 1

    validation = json.loads(capsys.readouterr().out)
    assert validation["valid"] is False
    assert validation["checks"][1]["name"] == "suite_digest"
    assert validation["checks"][1]["passed"] is False


def test_evaluate_cli_rejects_tampered_bundle_report_with_differences(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    bundle = evaluation_bundle(tags=("qa", "reobserve"))
    tampered_bundle = json.loads(evaluation_bundle_json(bundle))
    tampered_bundle["report"]["metrics"]["case_count"] = 999
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(tampered_bundle), encoding="utf-8")

    assert main(["--validate-bundle", str(bundle_path)]) == 1

    validation = json.loads(capsys.readouterr().out)
    report_check = next(
        check
        for check in validation["checks"]
        if check["name"] == "report_matches_suite"
    )
    assert validation["valid"] is False
    assert report_check["differences"] == [
        {"path": "metrics.case_count", "expected": 2, "actual": 999}
    ]


def test_evaluate_cli_rejects_tampered_bundle_coverage(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    bundle = evaluation_bundle(tags=("qa", "reobserve"))
    tampered_bundle = json.loads(evaluation_bundle_json(bundle))
    tampered_bundle["coverage"]["case_count"] = 999
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(tampered_bundle), encoding="utf-8")

    assert main(["--validate-bundle", str(bundle_path)]) == 1

    validation = json.loads(capsys.readouterr().out)
    assert validation["valid"] is False
    assert validation["checks"][-1]["name"] == "coverage_matches_manifest"
    assert validation["checks"][-1]["passed"] is False
    assert validation["checks"][-1]["differences"] == [
        {"path": "case_count", "expected": 2, "actual": 999}
    ]


def test_evaluate_cli_compares_bundle_with_current_run(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    bundle = evaluation_bundle(tags=("qa", "reobserve"))
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(evaluation_bundle_json(bundle), encoding="utf-8")

    assert main(["--compare-bundle", str(bundle_path)]) == 0

    comparison = json.loads(capsys.readouterr().out)
    assert comparison["matches"] is True
    assert comparison["saved_digest"] == bundle["suite"]["digest"]
    assert comparison["current_digest"] == bundle["suite"]["digest"]


def test_evaluate_cli_reports_bundle_current_run_drift(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    bundle = evaluation_bundle(tags=("qa", "reobserve"))
    drifted_bundle = json.loads(evaluation_bundle_json(bundle))
    drifted_bundle["filters"]["tags"] = ["qa", "dynamic"]
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(drifted_bundle), encoding="utf-8")

    assert main(["--compare-bundle", str(bundle_path)]) == 1

    comparison = json.loads(capsys.readouterr().out)
    assert comparison["matches"] is False
    assert comparison["checks"][1]["name"] == "suite_digest_matches_current"
    assert comparison["checks"][1]["passed"] is False


def test_evaluate_cli_reports_invalid_explicit_bundle_file_for_compare(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_evaluate_script()
    main = cast(MainFn, getattr(module, "main"))
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text("[]\n", encoding="utf-8")

    assert_invalid_artifact_report(
        main,
        capsys,
        ["--compare-bundle", str(bundle_path)],
        {
            "action": "compare_bundle",
            "path": str(bundle_path),
            "valid": False,
            "matches": False,
            "error": "Evaluation bundle JSON must be an object",
        },
    )
