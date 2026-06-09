from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from _pytest.capture import CaptureFixture

import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "merge_p55_prediction_shards.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("merge_p55_prediction_shards_test_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_merge_p55_prediction_shards_merges_base_and_complete_shards(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    qa_root = tmp_path / "qa"
    _write_case(qa_root, "case-base")
    _write_case(qa_root, "case-shard-1")
    _write_case(qa_root, "case-shard-2")
    base = tmp_path / "base.jsonl"
    _write_predictions(base, ["case-base"])
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        "p55-vlm-only",
        ["case-shard-1", "case-shard-2"],
    )
    output_dir = tmp_path / "shard-outputs"
    _write_predictions(output_dir / "p55-vlm-only-0001.jsonl", ["case-shard-1", "case-shard-2"])
    output = tmp_path / "merged.jsonl"
    report = tmp_path / "merge-report.json"

    module = load_script()
    exit_code = module.main(
        [
            "--base-input",
            str(base),
            "--shard-manifest",
            str(manifest),
            "--shard-output-dir",
            str(output_dir),
            "--expected-qa-root",
            str(qa_root),
            "--target-method",
            "vlm_only",
            "--output",
            str(output),
            "--report",
            str(report),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    predictions = lab.load_qa_predictions(output)
    assert exit_code == 0
    assert payload["ready"] is True
    assert [prediction.id for prediction in predictions] == [
        "case-base",
        "case-shard-1",
        "case-shard-2",
    ]
    assert report_payload["ready"] is True
    assert report_payload["shard_prediction_coverage"]["missing_case_count"] == 0
    assert report_payload["expected_qa_coverage"]["missing_case_count"] == 0


def test_merge_p55_prediction_shards_blocks_incomplete_shard_output(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    qa_root = tmp_path / "qa"
    _write_case(qa_root, "case-shard-1")
    _write_case(qa_root, "case-shard-2")
    base = tmp_path / "base.jsonl"
    lab.save_qa_predictions([], base)
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        "p55-vlm-only",
        ["case-shard-1", "case-shard-2"],
    )
    output_dir = tmp_path / "shard-outputs"
    _write_predictions(output_dir / "p55-vlm-only-0001.jsonl", ["case-shard-1"])
    output = tmp_path / "merged.jsonl"
    report = tmp_path / "merge-report.json"

    module = load_script()
    exit_code = module.main(
        [
            "--base-input",
            str(base),
            "--shard-manifest",
            str(manifest),
            "--shard-output-dir",
            str(output_dir),
            "--expected-qa-root",
            str(qa_root),
            "--target-method",
            "vlm_only",
            "--output",
            str(output),
            "--report",
            str(report),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["ready"] is False
    assert "incomplete_shard_predictions" in report_payload["blockers"]
    assert report_payload["shard_prediction_coverage"]["missing_case_ids"] == ["case-shard-2"]
    assert report_payload["ready"] is False


def test_merge_p55_prediction_shards_merges_multiple_manifests(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    qa_root = tmp_path / "qa"
    _write_case(qa_root, "case-base")
    _write_case(qa_root, "case-crop")
    _write_case(qa_root, "case-no-crop")
    base = tmp_path / "base.jsonl"
    _write_predictions(base, ["case-base"])
    crop_manifest = _write_manifest(
        tmp_path / "crop-manifest.json",
        "p55-vlm-only-crop-ready",
        ["case-crop"],
    )
    no_crop_manifest = _write_manifest(
        tmp_path / "no-crop-manifest.json",
        "p55-vlm-only-no-crop",
        ["case-no-crop"],
    )
    crop_output_dir = tmp_path / "crop-outputs"
    no_crop_output_dir = tmp_path / "no-crop-outputs"
    _write_predictions(crop_output_dir / "p55-vlm-only-crop-ready-0001.jsonl", ["case-crop"])
    _write_predictions(
        no_crop_output_dir / "p55-vlm-only-no-crop-0001.jsonl",
        ["case-no-crop"],
    )
    output = tmp_path / "merged.jsonl"
    report = tmp_path / "merge-report.json"

    module = load_script()
    exit_code = module.main(
        [
            "--base-input",
            str(base),
            "--shard-manifest",
            str(crop_manifest),
            "--shard-manifest",
            str(no_crop_manifest),
            "--shard-output-dir",
            str(crop_output_dir),
            "--shard-output-dir",
            str(no_crop_output_dir),
            "--expected-qa-root",
            str(qa_root),
            "--target-method",
            "vlm_only",
            "--output",
            str(output),
            "--report",
            str(report),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    predictions = lab.load_qa_predictions(output)
    assert exit_code == 0
    assert payload["ready"] is True
    assert [prediction.id for prediction in predictions] == [
        "case-base",
        "case-crop",
        "case-no-crop",
    ]
    assert report_payload["ready"] is True
    assert report_payload["shard_manifest_count"] == 2
    assert report_payload["shard_prediction_coverage"]["expected_case_count"] == 2
    assert report_payload["expected_qa_coverage"]["missing_case_count"] == 0


def _write_case(root: Path, case_id: str) -> None:
    episode_dir = root / "episode-001"
    episode_dir.mkdir(parents=True, exist_ok=True)
    with (episode_dir / "qa-observation-aware.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"episode_id": "episode-001", "id": case_id}) + "\n")


def _write_manifest(path: Path, prefix: str, case_ids: list[str]) -> Path:
    shard_path = path.parent / f"{prefix}-0001.json"
    shard_path.write_text(
        json.dumps(
            {
                "leak_free": True,
                "prediction_cases": [{"case_id": case_id} for case_id in case_ids],
                "request_count": len(case_ids),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "dsg-spatialqa-lab.active-qa-v2-request-bundle-shard-manifest.v1",
        "shard_count": 1,
        "shards": [{"path": str(shard_path), "request_count": len(case_ids), "shard_index": 1}],
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_predictions(path: Path, case_ids: list[str]) -> None:
    lab.save_qa_predictions(
        [lab.QAPrediction(id=case_id, answer={"text": case_id}) for case_id in case_ids],
        path,
    )
