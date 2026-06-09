from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from _pytest.capture import CaptureFixture

import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "finalize_active_qa_v2_p55.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("finalize_active_qa_v2_p55_test_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_finalize_blocks_when_prediction_coverage_is_incomplete(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    qa_root = tmp_path / "qa"
    _write_case(qa_root, "episode-001", "case-001", "countertop")
    _write_case(qa_root, "episode-001", "case-002", "sink")
    vlm = tmp_path / "vlm.jsonl"
    trusted = tmp_path / "trusted.jsonl"
    graph = tmp_path / "graph.jsonl"
    _write_prediction(vlm, "case-001", "countertop")
    _write_prediction(trusted, "case-001", "countertop")
    _write_prediction(graph, "case-001", "countertop")
    report = tmp_path / "p55-report.json"

    module = load_script()
    exit_code = module.main(
        [
            "--qa-root",
            str(qa_root),
            "--vlm-input",
            str(vlm),
            "--trusted-input",
            str(trusted),
            "--graph-predictions",
            str(graph),
            "--required-episode-count",
            "1",
            "--output-dir",
            str(tmp_path / "out"),
            "--report",
            str(report),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["ready"] is False
    assert report_payload["research_ready"] is False
    assert report_payload["final_claim_written"] is False
    assert "vlm_only_prediction_coverage_incomplete" in report_payload["blockers"]
    assert "vlm_dsg_trusted_prediction_coverage_incomplete" in report_payload["blockers"]
    assert report_payload["merge_reports"]["vlm_only"]["coverage"]["missing_case_ids"] == ["case-002"]


def test_finalize_runs_comparison_and_attribution_when_predictions_cover_scope(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    qa_root_a = tmp_path / "qa-a"
    qa_root_b = tmp_path / "qa-b"
    _write_case(qa_root_a, "episode-001", "case-001", "countertop")
    _write_case(qa_root_b, "episode-002", "case-002", "sink")
    vlm = tmp_path / "vlm.jsonl"
    trusted = tmp_path / "trusted.jsonl"
    graph = tmp_path / "graph.jsonl"
    _write_predictions(vlm, [("case-001", "countertop"), ("case-002", "wrong")])
    _write_predictions(trusted, [("case-001", "countertop"), ("case-002", "sink")])
    _write_predictions(graph, [("case-001", "countertop"), ("case-002", "sink")])
    output_dir = tmp_path / "out"
    report = tmp_path / "p55-report.json"

    module = load_script()
    exit_code = module.main(
        [
            "--qa-root",
            str(qa_root_a),
            "--qa-root",
            str(qa_root_b),
            "--vlm-input",
            str(vlm),
            "--trusted-input",
            str(trusted),
            "--graph-predictions",
            str(graph),
            "--required-episode-count",
            "2",
            "--output-dir",
            str(output_dir),
            "--report",
            str(report),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    attribution = json.loads((output_dir / "p55-active-qa-v2-case-attribution.json").read_text(encoding="utf-8"))
    comparison = json.loads((output_dir / "p55-three-way-comparison-active-qa-v2.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["ready"] is False
    assert report_payload["coverage_ready"] is True
    assert report_payload["attribution_ready"] is True
    assert report_payload["final_claim_written"] is False
    assert attribution["case_count"] == 2
    assert comparison["methods"]["vlm_dsg_trusted"]["semantic_match_count"] == 2
    assert "directional_not_significant" in report_payload["blockers"]


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
    _write_predictions(path, [(case_id, dst_label)])


def _write_predictions(path: Path, cases: list[tuple[str, str]]) -> None:
    predictions = [
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
        for case_id, dst_label in cases
    ]
    lab.save_qa_predictions(predictions, path)
