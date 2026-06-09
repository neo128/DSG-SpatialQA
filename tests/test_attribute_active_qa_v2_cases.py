from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from _pytest.capture import CaptureFixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "attribute_active_qa_v2_cases.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "attribute_active_qa_v2_cases_test_script",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_attribute_cases_loads_multiple_active_qa_roots(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    qa_root_a = tmp_path / "qa-a"
    qa_root_b = tmp_path / "qa-b"
    _write_case(qa_root_a, "episode-001", "case-001", "countertop")
    _write_case(qa_root_b, "episode-002", "case-002", "sink")
    vlm_predictions = tmp_path / "vlm.jsonl"
    graph_predictions = tmp_path / "graph.jsonl"
    trusted_predictions = tmp_path / "trusted.jsonl"
    adjudicated_predictions = tmp_path / "adjudicated.jsonl"
    for path in (vlm_predictions, graph_predictions, trusted_predictions, adjudicated_predictions):
        _write_predictions(path)
    output = tmp_path / "attribution.json"

    module = load_script()
    exit_code = module.main(
        [
            "--qa-root",
            str(qa_root_a),
            "--qa-root",
            str(qa_root_b),
            "--vlm-predictions",
            str(vlm_predictions),
            "--graph-predictions",
            str(graph_predictions),
            "--trusted-predictions",
            str(trusted_predictions),
            "--adjudicated-predictions",
            str(adjudicated_predictions),
            "--output",
            str(output),
            "--markdown-output",
            str(tmp_path / "attribution.md"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["ready"] is True
    assert report["case_count"] == 2
    assert report["source_paths"]["qa_roots"] == [str(qa_root_a), str(qa_root_b)]


def _write_case(root: Path, episode_id: str, case_id: str, dst_label: str) -> None:
    episode_dir = root / episode_id
    episode_dir.mkdir(parents=True)
    row = {
        "answer": {"dst": f"{dst_label}_001", "dst_label": dst_label, "relation": "ON"},
        "episode_id": episode_id,
        "id": case_id,
        "question_type": "support_relation",
        "split": "observation_aware",
    }
    (episode_dir / "qa-observation-aware.jsonl").write_text(
        json.dumps(row, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_predictions(path: Path) -> None:
    rows = [
        {
            "answer": {
                "dst": "countertop_001",
                "dst_label": "countertop",
                "relation": "ON",
                "selected_candidate": "graph_tool_dsg",
            },
            "id": "case-001",
        },
        {
            "answer": {
                "dst": "sink_001",
                "dst_label": "sink",
                "relation": "ON",
                "selected_candidate": "graph_tool_dsg",
            },
            "id": "case-002",
        },
    ]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
