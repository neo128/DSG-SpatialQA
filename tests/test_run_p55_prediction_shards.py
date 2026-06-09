from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p55_prediction_shards.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_p55_prediction_shards_test_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_p55_prediction_shards_dry_run_writes_deterministic_plan(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path / "shard-manifest.json", "p55-vlm-only-missing")
    report = tmp_path / "run-plan.json"
    monkeypatch.delenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "system-key-must-not-be-used")

    module = load_script()
    exit_code = module.main(
        [
            "--target-method",
            "vlm_only",
            "--shard-manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--trace-dir",
            str(tmp_path / "traces"),
            "--report",
            str(report),
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    plan = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["ready"] is True
    assert plan["dry_run"] is True
    assert plan["execute"] is False
    assert plan["execution_ready"] is False
    assert "dedicated_api_key_env_unset" in plan["execution_blockers"]
    assert plan["pending_shard_count"] == 2
    assert plan["commands"][0]["argv"][:2] == ["python", "external_tools/run_vlm_controls.py"]
    assert "--allow-network" not in plan["commands"][0]["argv"]
    assert payload["execution_ready"] is False
    assert plan["uses_system_dashscope_key"] is False


def test_run_p55_prediction_shards_reports_target_crop_input_coverage(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    manifest = _write_manifest(
        tmp_path / "shard-manifest.json",
        "p55-vlm-only-missing-target-crop-enriched",
        target_crop_case_ids={"case-001"},
    )
    report = tmp_path / "run-plan.json"
    monkeypatch.delenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", raising=False)

    module = load_script()
    exit_code = module.main(
        [
            "--target-method",
            "vlm_only",
            "--shard-manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--trace-dir",
            str(tmp_path / "traces"),
            "--report",
            str(report),
            "--dry-run",
        ]
    )

    _ = json.loads(capsys.readouterr().out)
    plan = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert plan["request_input_summary"] == {
        "missing_target_crop_case_count": 1,
        "request_case_count": 2,
        "target_crop_case_count": 1,
        "target_visual_context_case_count": 0,
        "target_crop_enriched": False,
        "visual_context_fallback_case_count": 0,
    }


def test_run_p55_prediction_shards_reports_no_crop_visual_context_coverage(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    manifest = _write_manifest(
        tmp_path / "shard-manifest.json",
        "p55-vlm-only-missing-no-crop",
        target_visual_context_case_ids={"case-001", "case-002"},
    )
    report = tmp_path / "run-plan.json"
    monkeypatch.delenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", raising=False)

    module = load_script()
    exit_code = module.main(
        [
            "--target-method",
            "vlm_only",
            "--shard-manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--trace-dir",
            str(tmp_path / "traces"),
            "--report",
            str(report),
            "--dry-run",
        ]
    )

    _ = json.loads(capsys.readouterr().out)
    plan = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert plan["request_input_summary"] == {
        "missing_target_crop_case_count": 2,
        "request_case_count": 2,
        "target_crop_case_count": 0,
        "target_visual_context_case_count": 2,
        "target_crop_enriched": False,
        "visual_context_fallback_case_count": 2,
    }


def test_run_p55_prediction_shards_keeps_incomplete_existing_output_pending(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path / "shard-manifest.json", "p55-vlm-only-missing")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    incomplete_output = output_dir / "p55-vlm-only-missing-0001.jsonl"
    incomplete_output.write_text(_prediction_line("wrong-case"), encoding="utf-8")
    report = tmp_path / "run-plan.json"
    monkeypatch.delenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", raising=False)

    module = load_script()
    exit_code = module.main(
        [
            "--target-method",
            "vlm_only",
            "--shard-manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--trace-dir",
            str(tmp_path / "traces"),
            "--report",
            str(report),
            "--dry-run",
        ]
    )

    _ = json.loads(capsys.readouterr().out)
    plan = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert plan["pending_shard_count"] == 2
    assert plan["skipped_existing_shard_count"] == 0


def test_run_p55_prediction_shards_execute_requires_dedicated_key(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path / "shard-manifest.json", "p55-vlm-only-missing")
    report = tmp_path / "run-plan.json"
    monkeypatch.delenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "system-key-must-not-be-used")

    module = load_script()
    exit_code = module.main(
        [
            "--target-method",
            "vlm_only",
            "--shard-manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--trace-dir",
            str(tmp_path / "traces"),
            "--report",
            str(report),
            "--execute",
            "--allow-network",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    plan = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["ready"] is False
    assert "dedicated_api_key_env_unset" in plan["blockers"]
    assert plan["api_key_env"] == "DSG_SPATIALQA_DASHSCOPE_API_KEY"
    assert plan["uses_system_dashscope_key"] is False


def test_run_p55_prediction_shards_execute_supports_limited_parallelism(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path / "shard-manifest.json", "p55-vlm-only-missing")
    report = tmp_path / "run-plan.json"
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "dedicated-key")

    module = load_script()
    active = 0
    max_active = 0
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    calls: list[list[str]] = []

    def fake_run(argv: list[str], *, check: bool) -> SimpleNamespace:
        nonlocal active, max_active
        assert check is False
        with lock:
            active += 1
            max_active = max(max_active, active)
            calls.append(argv)
        try:
            barrier.wait(timeout=1)
        finally:
            with lock:
                active -= 1
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code = module.main(
        [
            "--target-method",
            "vlm_only",
            "--shard-manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--trace-dir",
            str(tmp_path / "traces"),
            "--report",
            str(report),
            "--execute",
            "--allow-network",
            "--max-parallel",
            "2",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    plan = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["ready"] is True
    assert plan["max_parallel"] == 2
    assert plan["executed_shard_count"] == 2
    assert len(calls) == 2
    assert max_active == 2


def test_run_p55_prediction_shards_dsg_trusted_plan_requires_vlm_and_graph_inputs(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path / "shard-manifest.json", "p55-vlm-dsg-trusted-missing")
    report = tmp_path / "run-plan.json"
    vlm_predictions = tmp_path / "vlm.jsonl"
    graph_predictions = tmp_path / "graph.jsonl"
    vlm_predictions.write_text(
        _prediction_line("case-001") + _prediction_line("case-002"),
        encoding="utf-8",
    )
    graph_predictions.write_text(
        _prediction_line("case-001") + _prediction_line("case-002"),
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "dedicated-key")

    module = load_script()
    exit_code = module.main(
        [
            "--target-method",
            "vlm_dsg_trusted",
            "--shard-manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--trace-dir",
            str(tmp_path / "traces"),
            "--vlm-predictions",
            str(vlm_predictions),
            "--graph-predictions",
            str(graph_predictions),
            "--report",
            str(report),
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    plan = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["ready"] is True
    assert payload["execution_ready"] is True
    assert plan["execution_ready"] is True
    assert plan["commands"][0]["argv"][:2] == [
        "python",
        "external_tools/run_vlm_graph_adjudication_active.py",
    ]
    assert "--vlm-predictions" in plan["commands"][0]["argv"]
    assert "--graph-predictions" in plan["commands"][0]["argv"]


def test_run_p55_prediction_shards_dsg_trusted_rejects_incomplete_input_coverage(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path / "shard-manifest.json", "p55-vlm-dsg-trusted-missing")
    report = tmp_path / "run-plan.json"
    vlm_predictions = tmp_path / "vlm.jsonl"
    graph_predictions = tmp_path / "graph.jsonl"
    vlm_predictions.write_text(_prediction_line("case-001"), encoding="utf-8")
    graph_predictions.write_text(
        _prediction_line("case-001") + _prediction_line("case-002"),
        encoding="utf-8",
    )
    monkeypatch.setenv("DSG_SPATIALQA_DASHSCOPE_API_KEY", "dedicated-key")

    module = load_script()
    exit_code = module.main(
        [
            "--target-method",
            "vlm_dsg_trusted",
            "--shard-manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--trace-dir",
            str(tmp_path / "traces"),
            "--vlm-predictions",
            str(vlm_predictions),
            "--graph-predictions",
            str(graph_predictions),
            "--report",
            str(report),
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    plan = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["ready"] is False
    assert payload["execution_ready"] is False
    assert "missing_vlm_prediction_cases" in plan["blockers"]
    assert plan["prediction_input_coverage"]["vlm_predictions"]["missing_case_count"] == 1
    assert plan["prediction_input_coverage"]["graph_predictions"]["missing_case_count"] == 0


def _write_manifest(
    path: Path,
    prefix: str,
    *,
    target_crop_case_ids: set[str] | None = None,
    target_visual_context_case_ids: set[str] | None = None,
) -> Path:
    shards = []
    for index in range(1, 3):
        shard_path = path.parent / f"{prefix}-{index:04d}.json"
        case_id = f"case-{index:03d}"
        case: dict[str, object] = {"case_id": case_id}
        if target_crop_case_ids is not None and case_id in target_crop_case_ids:
            case["target_crop"] = {
                "bbox_2d_xyxy": [1, 2, 3, 4],
                "crop_kind": "segmentation_color_mask",
                "rgb_path": str(path.parent / f"{case_id}-crop.ppm"),
            }
        if (
            target_visual_context_case_ids is not None
            and case_id in target_visual_context_case_ids
        ):
            case["target_visual_context"] = {
                "available": True,
                "context_kind": "primary_frame_without_target_crop",
                "target_crop_available": False,
            }
        shard_path.write_text(
            json.dumps(
                {
                    "leak_free": True,
                    "prediction_cases": [case],
                    "request_count": 1,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        shards.append({"path": str(shard_path), "request_count": 1, "shard_index": index})
    manifest = {
        "schema_version": "dsg-spatialqa-lab.active-qa-v2-request-bundle-shard-manifest.v1",
        "shard_count": 2,
        "shards": shards,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _prediction_line(case_id: str) -> str:
    return json.dumps(
        {
            "answer": {"text": "ok"},
            "confidence": 1.0,
            "error": None,
            "evidence_edges": [],
            "evidence_nodes": [],
            "id": case_id,
            "schema_version": "dsg-spatialqa-lab.qa-prediction.v1",
        },
        sort_keys=True,
    ) + "\n"
