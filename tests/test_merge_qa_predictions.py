from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from _pytest.capture import CaptureFixture

import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "merge_qa_predictions.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("merge_qa_predictions_test_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_merge_predictions_replaces_by_id_and_validates_expected_qa_coverage(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    qa_root = tmp_path / "qa"
    _write_case(qa_root, "case-001")
    _write_case(qa_root, "case-002")
    base = tmp_path / "base.jsonl"
    supplement = tmp_path / "supplement.jsonl"
    output = tmp_path / "merged.jsonl"
    report = tmp_path / "merge-report.json"
    lab.save_qa_predictions(
        [
            lab.QAPrediction(id="case-001", answer={"text": "old"}),
            lab.QAPrediction(id="case-002", answer={"text": "old"}),
        ],
        base,
    )
    lab.save_qa_predictions(
        [lab.QAPrediction(id="case-002", answer={"text": "new"})],
        supplement,
    )

    module = load_script()
    exit_code = module.main(
        [
            "--input",
            str(base),
            "--input",
            str(supplement),
            "--expected-qa-root",
            str(qa_root),
            "--output",
            str(output),
            "--report",
            str(report),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    predictions = lab.load_qa_predictions(output)
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["ready"] is True
    assert [prediction.id for prediction in predictions] == ["case-001", "case-002"]
    assert predictions[1].answer == {"text": "new"}
    assert report_payload["coverage"]["missing_case_count"] == 0
    assert report_payload["coverage"]["prediction_coverage_rate"] == 1.0


def test_merge_predictions_fails_when_expected_qa_coverage_is_incomplete(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    qa_root = tmp_path / "qa"
    _write_case(qa_root, "case-001")
    _write_case(qa_root, "case-002")
    base = tmp_path / "base.jsonl"
    output = tmp_path / "merged.jsonl"
    report = tmp_path / "merge-report.json"
    lab.save_qa_predictions([lab.QAPrediction(id="case-001", answer={"text": "only"})], base)

    module = load_script()
    exit_code = module.main(
        [
            "--input",
            str(base),
            "--expected-qa-root",
            str(qa_root),
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
    assert report_payload["coverage"]["missing_case_ids"] == ["case-002"]


def _write_case(root: Path, case_id: str) -> None:
    episode_dir = root / "episode-001"
    episode_dir.mkdir(parents=True, exist_ok=True)
    with (episode_dir / "qa-observation-aware.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": case_id, "episode_id": "episode-001"}) + "\n")
