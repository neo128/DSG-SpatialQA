from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from _pytest.capture import CaptureFixture


ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = ROOT / "scripts" / "compare_active_qa_v2_three_way.py"


def load_compare_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "compare_active_qa_v2_three_way_test_script",
        COMPARE_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compare_multiple_qa_roots_blocks_incomplete_prediction_coverage(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    qa_root_a = tmp_path / "qa-a"
    qa_root_b = tmp_path / "qa-b"
    _write_case(qa_root_a, "episode-001", "case-001", "countertop")
    _write_case(qa_root_b, "episode-002", "case-002", "sink")
    vlm_predictions = tmp_path / "vlm.jsonl"
    trusted_predictions = tmp_path / "trusted.jsonl"
    _write_prediction(vlm_predictions, "case-001", "countertop")
    _write_prediction(trusted_predictions, "case-001", "countertop")
    output = tmp_path / "comparison.json"

    module = load_compare_script()
    exit_code = module.main(
        [
            "--qa-root",
            str(qa_root_a),
            "--qa-root",
            str(qa_root_b),
            "--vlm-predictions",
            str(vlm_predictions),
            "--vlm-dsg-predictions",
            str(trusted_predictions),
            "--required-episode-count",
            "2",
            "--output",
            str(output),
            "--markdown-output",
            str(tmp_path / "comparison.md"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["ready"] is False
    assert report["episode_count"] == 2
    assert "vlm_only_prediction_coverage_incomplete" in report["blockers"]
    assert "vlm_dsg_trusted_prediction_coverage_incomplete" in report["blockers"]
    assert report["prediction_coverage"]["vlm_only"]["missing_case_count"] == 1
    assert report["next_missing_predictions"] == [
        {"episode_id": "episode-002", "method": "vlm_dsg_trusted", "missing_count": 1},
        {"episode_id": "episode-002", "method": "vlm_only", "missing_count": 1},
    ]


def _write_case(root: Path, episode_id: str, case_id: str, dst_label: str) -> None:
    episode_dir = root / episode_id
    episode_dir.mkdir(parents=True)
    row = {
        "answer": {"dst": f"{dst_label}_001", "dst_label": dst_label, "relation": "ON"},
        "episode_id": episode_id,
        "id": case_id,
        "question_type": "support_relation",
    }
    (episode_dir / "qa-observation-aware.jsonl").write_text(
        json.dumps(row, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_prediction(path: Path, case_id: str, dst_label: str) -> None:
    row = {
        "answer": {"dst": f"{dst_label}_001", "dst_label": dst_label, "relation": "ON"},
        "id": case_id,
    }
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
