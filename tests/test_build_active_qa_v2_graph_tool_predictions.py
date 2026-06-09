from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from _pytest.capture import CaptureFixture

import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_active_qa_v2_graph_tool_predictions.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_active_qa_v2_graph_tool_predictions_test_script",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_builds_explicit_graph_tool_predictions_from_multiple_active_qa_roots(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    qa_root_a = tmp_path / "qa-a"
    qa_root_b = tmp_path / "qa-b"
    _write_case(qa_root_a, "episode-001", "case-001", "countertop")
    _write_case(qa_root_b, "episode-002", "case-002", "sink")
    output = tmp_path / "graph-tool.jsonl"
    report = tmp_path / "graph-tool-report.json"

    module = load_script()
    exit_code = module.main(
        [
            "--qa-root",
            str(qa_root_a),
            "--qa-root",
            str(qa_root_b),
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
    assert payload["prediction_count"] == 2
    assert [prediction.id for prediction in predictions] == ["case-001", "case-002"]
    assert predictions[0].answer["current_location"] == {
        "dst": "countertop_001",
        "dst_label": "countertop",
        "relation": "ON",
        "step": 1,
    }
    assert predictions[0].answer["source"] == "graph_tool_only_dsg"
    assert predictions[0].evidence_edges == ("apple_001-ON-countertop_001",)
    assert report_payload["qa_roots"] == [str(qa_root_a), str(qa_root_b)]
    assert report_payload["prediction_count"] == 2
    assert report_payload["episode_counts"] == {"episode-001": 1, "episode-002": 1}


def _write_case(root: Path, episode_id: str, case_id: str, dst_label: str) -> None:
    episode_dir = root / episode_id
    episode_dir.mkdir(parents=True)
    row = {
        "answer": {
            "dst": f"{dst_label}_001",
            "dst_label": dst_label,
            "relation": "ON",
            "step": 1,
        },
        "episode_id": episode_id,
        "evidence_frames": [1],
        "id": case_id,
        "question_type": "support_relation",
        "required_evidence": {
            "edges": [f"apple_001-ON-{dst_label}_001"],
            "nodes": ["apple_001", f"{dst_label}_001"],
        },
        "required_edges": [f"apple_001-ON-{dst_label}_001"],
        "required_nodes": ["apple_001", f"{dst_label}_001"],
        "target": {"label": "apple", "object_id": "apple_001"},
    }
    (episode_dir / "qa-observation-aware.jsonl").write_text(
        json.dumps(row, sort_keys=True) + "\n",
        encoding="utf-8",
    )
