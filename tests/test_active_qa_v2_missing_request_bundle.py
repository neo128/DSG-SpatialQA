from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from _pytest.capture import CaptureFixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_active_qa_v2_missing_prediction_request_bundle.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_active_qa_v2_missing_prediction_request_bundle_test_script",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_missing_prediction_request_bundle_filters_existing_and_stays_leak_free(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    qa_root_a = tmp_path / "qa-a"
    qa_root_b = tmp_path / "qa-b"
    _write_case(qa_root_a, "episode-001", "case-001", "countertop")
    _write_case(qa_root_b, "episode-002", "case-002", "sink")
    existing_predictions = tmp_path / "predictions.jsonl"
    existing_predictions.write_text(
        json.dumps({"id": "case-001", "answer": {"text": "on the countertop"}}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "missing-request-bundle.json"

    module = load_script()
    exit_code = module.main(
        [
            "--qa-root",
            str(qa_root_a),
            "--qa-root",
            str(qa_root_b),
            "--existing-predictions",
            str(existing_predictions),
            "--output",
            str(output),
            "--target-method",
            "vlm_only",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    bundle = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["ready"] is True
    assert bundle["leak_free"] is True
    assert bundle["missing_case_count"] == 1
    assert bundle["request_count"] == 1
    assert bundle["prediction_cases"][0]["case_id"] == "case-002"
    assert bundle["prediction_cases"][0]["support_candidates"] == [
        {"label": "sink", "relation_hint": "ON"},
        {"label": "floor", "relation_hint": "ON"},
    ]
    assert bundle["missing_by_episode"] == {"episode-002": 1}
    forbidden_text = json.dumps(bundle, sort_keys=True)
    for forbidden in (
        "gold_answer",
        "gold_evidence",
        "required_edges",
        "required_nodes",
        "visible_object_ids",
        "visible_object_labels",
    ):
        assert forbidden not in forbidden_text


def _write_case(root: Path, episode_id: str, case_id: str, dst_label: str) -> None:
    episode_dir = root / episode_id
    episode_dir.mkdir(parents=True)
    row = {
        "answer": {"dst": f"{dst_label}_001", "dst_label": dst_label, "relation": "ON"},
        "answer_options": [
            {"dst_label": dst_label, "option_id": "a", "relation": "ON"},
            {"dst_label": "floor", "option_id": "b", "relation": "ON"},
        ],
        "episode_id": episode_id,
        "gold_answer": {"should": "not leak"},
        "id": case_id,
        "question_text": "Where is the object?",
        "question_type": "support_relation",
        "required_edges": ["secret-edge"],
        "required_nodes": ["secret-node"],
        "scene_id": "FloorPlan1",
        "situation": {"step": 1, "view_frame": {"rgb_path": "rgb/0001.ppm"}},
        "target": {"label": "apple", "object_id": "apple_001"},
        "visible_object_ids": ["secret-object"],
        "visible_object_labels": ["secret-label"],
    }
    (episode_dir / "qa-observation-aware.jsonl").write_text(
        json.dumps(row, sort_keys=True) + "\n",
        encoding="utf-8",
    )
