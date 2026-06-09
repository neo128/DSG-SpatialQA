from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from _pytest.capture import CaptureFixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "shard_active_qa_v2_request_bundle.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "shard_active_qa_v2_request_bundle_test_script",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shards_active_qa_v2_bundle_and_writes_manifest(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    bundle_path = tmp_path / "request-bundle.json"
    _write_bundle(bundle_path, case_count=5)
    output_dir = tmp_path / "shards"
    manifest = tmp_path / "manifest.json"

    module = load_script()
    exit_code = module.main(
        [
            "--input",
            str(bundle_path),
            "--output-dir",
            str(output_dir),
            "--shard-size",
            "2",
            "--prefix",
            "p55-vlm-only",
            "--manifest",
            str(manifest),
            "--target-method",
            "vlm_only",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    shard_paths = [Path(item["path"]) for item in manifest_payload["shards"]]
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    assert exit_code == 0
    assert payload["ready"] is True
    assert payload["shard_count"] == 3
    assert [shard["request_count"] for shard in shards] == [2, 2, 1]
    assert [case["case_id"] for shard in shards for case in shard["prediction_cases"]] == [
        "case-001",
        "case-002",
        "case-003",
        "case-004",
        "case-005",
    ]
    assert all(shard["leak_free"] is True for shard in shards)
    assert all(not shard["leak_paths"] for shard in shards)
    assert manifest_payload["total_case_count"] == 5
    assert manifest_payload["target_method"] == "vlm_only"


def test_shards_active_qa_v2_bundle_with_target_crop_filter(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    bundle_path = tmp_path / "request-bundle.json"
    _write_bundle(
        bundle_path,
        case_count=4,
        target_crop_case_ids={"case-002", "case-004"},
    )
    output_dir = tmp_path / "shards"
    manifest = tmp_path / "manifest.json"

    module = load_script()
    exit_code = module.main(
        [
            "--input",
            str(bundle_path),
            "--output-dir",
            str(output_dir),
            "--shard-size",
            "1",
            "--prefix",
            "p55-vlm-only-crop-ready",
            "--manifest",
            str(manifest),
            "--target-method",
            "vlm_only",
            "--target-crop-filter",
            "with",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    shard_paths = [Path(item["path"]) for item in manifest_payload["shards"]]
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    assert exit_code == 0
    assert payload["ready"] is True
    assert payload["total_case_count"] == 2
    assert manifest_payload["source_total_case_count"] == 4
    assert manifest_payload["target_crop_case_count"] == 2
    assert manifest_payload["target_crop_filter"] == "with"
    assert [case["case_id"] for shard in shards for case in shard["prediction_cases"]] == [
        "case-002",
        "case-004",
    ]


def test_shard_rejects_non_leak_free_bundle(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    bundle_path = tmp_path / "request-bundle.json"
    _write_bundle(bundle_path, case_count=1, leak_free=False)

    module = load_script()
    exit_code = module.main(
        [
            "--input",
            str(bundle_path),
            "--output-dir",
            str(tmp_path / "shards"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ready"] is False
    assert payload["error"] == "request_bundle_not_leak_free"


def _write_bundle(
    path: Path,
    *,
    case_count: int,
    leak_free: bool = True,
    target_crop_case_ids: set[str] | None = None,
) -> None:
    cases: list[dict[str, object]] = [
        {
            "case_id": f"case-{index:03d}",
            "episode_id": "episode-001",
            "question_text": "Where is the apple?",
            "question_type": "object_location",
        }
        for index in range(1, case_count + 1)
    ]
    for case in cases:
        case_id = str(case["case_id"])
        if target_crop_case_ids is not None and case_id in target_crop_case_ids:
            case["target_crop"] = {
                "bbox_2d_xyxy": [1, 2, 3, 4],
                "crop_kind": "segmentation_color_mask",
                "rgb_path": f"crops/{case_id}.ppm",
            }
    path.write_text(
        json.dumps(
            {
                "leak_free": leak_free,
                "leak_paths": [] if leak_free else ["$.gold_answer"],
                "prediction_cases": cases,
                "request_bundle_digest": "digest",
                "request_count": len(cases),
                "schema_version": "dsg-spatialqa-lab.active-qa-v2-vlm-request-bundle.v1",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
