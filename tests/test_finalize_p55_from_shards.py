from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from _pytest.capture import CaptureFixture

import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "finalize_p55_from_shards.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("finalize_p55_from_shards_test_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_finalize_from_shards_reports_missing_outputs(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    vlm_manifest = _write_manifest(tmp_path / "vlm-manifest.json", "vlm-only", ["case-001"])
    trusted_manifest = _write_manifest(
        tmp_path / "trusted-manifest.json",
        "vlm-dsg-trusted",
        ["case-001"],
    )
    report = tmp_path / "report.json"

    module = load_script()
    exit_code = module.main(
        [
            "--vlm-shard-manifest",
            str(vlm_manifest),
            "--vlm-shard-output-dir",
            str(tmp_path / "vlm-outputs"),
            "--trusted-shard-manifest",
            str(trusted_manifest),
            "--trusted-shard-output-dir",
            str(tmp_path / "trusted-outputs"),
            "--report",
            str(report),
            "--finalize-output-dir",
            str(tmp_path / "finalize"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["ready"] is False
    assert report_payload["shard_coverage_ready"] is False
    assert report_payload["finalize_invoked"] is False
    assert report_payload["missing_shard_output_count"] == 2
    assert "missing_vlm_only_shard_outputs" in report_payload["blockers"]
    assert "missing_vlm_dsg_trusted_shard_outputs" in report_payload["blockers"]


def test_finalize_from_complete_shards_invokes_p55_finalize(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    qa_root = tmp_path / "qa"
    _write_case(qa_root, "episode-001", "case-001", "countertop")
    vlm_manifest = _write_manifest(tmp_path / "vlm-manifest.json", "vlm-only", ["case-001"])
    trusted_manifest = _write_manifest(
        tmp_path / "trusted-manifest.json",
        "vlm-dsg-trusted",
        ["case-001"],
    )
    vlm_output_dir = tmp_path / "vlm-outputs"
    trusted_output_dir = tmp_path / "trusted-outputs"
    _write_prediction(vlm_output_dir / "vlm-only-0001.jsonl", "case-001", "wrong")
    _write_prediction(trusted_output_dir / "vlm-dsg-trusted-0001.jsonl", "case-001", "countertop")
    base_vlm = tmp_path / "base-vlm.jsonl"
    base_trusted = tmp_path / "base-trusted.jsonl"
    graph = tmp_path / "graph.jsonl"
    lab.save_qa_predictions([], base_vlm)
    lab.save_qa_predictions([], base_trusted)
    _write_prediction(graph, "case-001", "countertop")
    report = tmp_path / "report.json"
    finalize_dir = tmp_path / "finalize"

    module = load_script()
    exit_code = module.main(
        [
            "--qa-root",
            str(qa_root),
            "--vlm-base-input",
            str(base_vlm),
            "--trusted-base-input",
            str(base_trusted),
            "--graph-predictions",
            str(graph),
            "--required-episode-count",
            "1",
            "--vlm-shard-manifest",
            str(vlm_manifest),
            "--vlm-shard-output-dir",
            str(vlm_output_dir),
            "--trusted-shard-manifest",
            str(trusted_manifest),
            "--trusted-shard-output-dir",
            str(trusted_output_dir),
            "--report",
            str(report),
            "--finalize-output-dir",
            str(finalize_dir),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["ready"] is False
    assert report_payload["shard_coverage_ready"] is True
    assert report_payload["finalize_invoked"] is True
    assert report_payload["finalize_report"]["coverage_ready"] is True
    assert (finalize_dir / "p55-active-qa-v2-finalize-report.json").exists()


def test_finalize_from_split_shard_manifests_invokes_p55_finalize(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    qa_root = tmp_path / "qa"
    _write_case(qa_root, "episode-001", "case-crop", "countertop")
    _write_case(qa_root, "episode-001", "case-no-crop", "sink")
    vlm_crop_manifest = _write_manifest(
        tmp_path / "vlm-crop-manifest.json",
        "vlm-only-crop-ready",
        ["case-crop"],
    )
    vlm_no_crop_manifest = _write_manifest(
        tmp_path / "vlm-no-crop-manifest.json",
        "vlm-only-no-crop",
        ["case-no-crop"],
    )
    trusted_crop_manifest = _write_manifest(
        tmp_path / "trusted-crop-manifest.json",
        "vlm-dsg-trusted-crop-ready",
        ["case-crop"],
    )
    trusted_no_crop_manifest = _write_manifest(
        tmp_path / "trusted-no-crop-manifest.json",
        "vlm-dsg-trusted-no-crop",
        ["case-no-crop"],
    )
    vlm_crop_output_dir = tmp_path / "vlm-crop-outputs"
    vlm_no_crop_output_dir = tmp_path / "vlm-no-crop-outputs"
    trusted_crop_output_dir = tmp_path / "trusted-crop-outputs"
    trusted_no_crop_output_dir = tmp_path / "trusted-no-crop-outputs"
    _write_prediction(vlm_crop_output_dir / "vlm-only-crop-ready-0001.jsonl", "case-crop", "wrong")
    _write_prediction(vlm_no_crop_output_dir / "vlm-only-no-crop-0001.jsonl", "case-no-crop", "wrong")
    _write_prediction(
        trusted_crop_output_dir / "vlm-dsg-trusted-crop-ready-0001.jsonl",
        "case-crop",
        "countertop",
    )
    _write_prediction(
        trusted_no_crop_output_dir / "vlm-dsg-trusted-no-crop-0001.jsonl",
        "case-no-crop",
        "sink",
    )
    base_vlm = tmp_path / "base-vlm.jsonl"
    base_trusted = tmp_path / "base-trusted.jsonl"
    graph = tmp_path / "graph.jsonl"
    lab.save_qa_predictions([], base_vlm)
    lab.save_qa_predictions([], base_trusted)
    _write_prediction(graph, "case-crop", "countertop")
    _write_prediction(graph, "case-no-crop", "sink")
    report = tmp_path / "report.json"
    finalize_dir = tmp_path / "finalize"

    module = load_script()
    exit_code = module.main(
        [
            "--qa-root",
            str(qa_root),
            "--vlm-base-input",
            str(base_vlm),
            "--trusted-base-input",
            str(base_trusted),
            "--graph-predictions",
            str(graph),
            "--required-episode-count",
            "1",
            "--vlm-shard-manifest",
            str(vlm_crop_manifest),
            "--vlm-shard-manifest",
            str(vlm_no_crop_manifest),
            "--vlm-shard-output-dir",
            str(vlm_crop_output_dir),
            "--vlm-shard-output-dir",
            str(vlm_no_crop_output_dir),
            "--trusted-shard-manifest",
            str(trusted_crop_manifest),
            "--trusted-shard-manifest",
            str(trusted_no_crop_manifest),
            "--trusted-shard-output-dir",
            str(trusted_crop_output_dir),
            "--trusted-shard-output-dir",
            str(trusted_no_crop_output_dir),
            "--report",
            str(report),
            "--finalize-output-dir",
            str(finalize_dir),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["ready"] is False
    assert report_payload["shard_coverage_ready"] is True
    assert report_payload["finalize_invoked"] is True
    assert report_payload["shard_manifest_counts"] == {
        "vlm_dsg_trusted": 2,
        "vlm_only": 2,
    }
    assert report_payload["shard_prediction_coverage"]["vlm_only"]["expected_case_count"] == 2
    assert report_payload["finalize_report"]["coverage_ready"] is True


def test_finalize_from_shards_blocks_incomplete_shard_prediction_coverage(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    vlm_manifest = _write_manifest(
        tmp_path / "vlm-manifest.json",
        "vlm-only",
        ["case-001", "case-002"],
    )
    trusted_manifest = _write_manifest(
        tmp_path / "trusted-manifest.json",
        "vlm-dsg-trusted",
        ["case-001"],
    )
    vlm_output_dir = tmp_path / "vlm-outputs"
    trusted_output_dir = tmp_path / "trusted-outputs"
    _write_prediction(vlm_output_dir / "vlm-only-0001.jsonl", "case-001", "countertop")
    _write_prediction(
        trusted_output_dir / "vlm-dsg-trusted-0001.jsonl",
        "case-001",
        "countertop",
    )
    report = tmp_path / "report.json"

    module = load_script()
    exit_code = module.main(
        [
            "--vlm-shard-manifest",
            str(vlm_manifest),
            "--vlm-shard-output-dir",
            str(vlm_output_dir),
            "--trusted-shard-manifest",
            str(trusted_manifest),
            "--trusted-shard-output-dir",
            str(trusted_output_dir),
            "--report",
            str(report),
            "--finalize-output-dir",
            str(tmp_path / "finalize"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["ready"] is False
    assert report_payload["shard_coverage_ready"] is False
    assert report_payload["finalize_invoked"] is False
    assert "incomplete_vlm_only_shard_predictions" in report_payload["blockers"]
    assert report_payload["shard_prediction_coverage"]["vlm_only"]["missing_case_count"] == 1
    assert report_payload["shard_prediction_coverage"]["vlm_only"]["missing_case_ids"] == ["case-002"]


def _write_manifest(path: Path, prefix: str, case_ids: list[str]) -> Path:
    shard_path = path.parent / f"{prefix}-0001.json"
    shard = {
        "case_inputs": [{"case_id": case_id} for case_id in case_ids],
        "leak_free": True,
        "prediction_cases": [{"case_id": case_id} for case_id in case_ids],
        "request_count": len(case_ids),
        "schema_version": "dsg-spatialqa-lab.active-qa-v2-vlm-request-bundle.v1",
    }
    shard_path.write_text(json.dumps(shard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "dsg-spatialqa-lab.active-qa-v2-request-bundle-shard-manifest.v1",
        "shard_count": 1,
        "shards": [{"path": str(shard_path), "request_count": len(case_ids), "shard_index": 1}],
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_case(root: Path, episode_id: str, case_id: str, dst_label: str) -> None:
    episode_dir = root / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "answer": {"dst": f"{dst_label}_001", "dst_label": dst_label, "relation": "ON"},
        "episode_id": episode_id,
        "id": case_id,
        "question_type": "support_relation",
        "split": "observation_aware",
    }
    with (episode_dir / "qa-observation-aware.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_prediction(path: Path, case_id: str, dst_label: str) -> None:
    predictions = lab.load_qa_predictions(path) if path.exists() else []
    predictions.append(
        lab.QAPrediction(
            id=case_id,
            answer={
                "dst": f"{dst_label}_001",
                "dst_label": dst_label,
                "relation": "ON",
                "selected_candidate": "graph_tool_dsg",
            },
            confidence=1.0,
        )
    )
    lab.save_qa_predictions(predictions, path)
