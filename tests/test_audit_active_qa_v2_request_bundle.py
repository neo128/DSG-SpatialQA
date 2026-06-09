from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from _pytest.capture import CaptureFixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_active_qa_v2_request_bundle.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "audit_active_qa_v2_request_bundle_test_script",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_request_bundle_quality_audit_passes_complete_leak_free_bundle(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    frame = tmp_path / "rgb.ppm"
    frame.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    bundle = tmp_path / "bundle.json"
    _write_bundle(
        bundle,
        [
            _case(
                "case-001",
                frame,
                question_type="support_relation",
                support_candidates=[{"label": "countertop", "relation_hint": "ON"}],
            ),
            _case("case-002", frame, question_type="state_change"),
        ],
    )
    report = tmp_path / "report.json"

    module = load_script()
    exit_code = module.main(
        [
            "--request-bundle",
            str(bundle),
            "--report",
            str(report),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    audited = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["ready"] is True
    assert audited["ready"] is True
    assert audited["summary"]["request_count"] == 2
    assert audited["summary"]["question_task_hint_count"] == 2
    assert audited["summary"]["support_candidates_case_count"] == 1
    assert audited["summary"]["primary_frame_exists_count"] == 2
    assert audited["summary"]["target_crop_case_count"] == 0
    assert audited["summary"]["target_visual_context_case_count"] == 0
    assert audited["blockers"] == []


def test_request_bundle_quality_audit_counts_target_visual_context(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    frame = tmp_path / "rgb.ppm"
    frame.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    bundle = tmp_path / "bundle.json"
    _write_bundle(
        bundle,
        [
            _case(
                "case-001",
                frame,
                question_type="nearest_object",
                target_visual_context={
                    "available": True,
                    "context_kind": "primary_frame_without_target_crop",
                    "target_crop_available": False,
                },
            )
        ],
    )
    report = tmp_path / "report.json"

    module = load_script()
    exit_code = module.main(
        [
            "--request-bundle",
            str(bundle),
            "--report",
            str(report),
        ]
    )

    _ = capsys.readouterr()
    audited = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert audited["summary"]["target_crop_case_count"] == 0
    assert audited["summary"]["target_visual_context_case_count"] == 1


def test_request_bundle_quality_audit_fails_missing_task_hints_and_frames(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    bundle = tmp_path / "bundle.json"
    missing_frame = tmp_path / "missing.ppm"
    row = _case("case-001", missing_frame, question_type="support_relation")
    row.pop("question_task_hint")
    _write_bundle(bundle, [row])
    report = tmp_path / "report.json"

    module = load_script()
    exit_code = module.main(
        [
            "--request-bundle",
            str(bundle),
            "--report",
            str(report),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    audited = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["ready"] is False
    assert audited["ready"] is False
    assert audited["blockers"] == [
        "primary_frame_artifacts_missing",
        "question_task_hints_incomplete",
    ]
    assert audited["summary"]["missing_primary_frame_count"] == 1
    assert audited["summary"]["missing_question_task_hint_count"] == 1


def _case(
    case_id: str,
    frame: Path,
    *,
    question_type: str,
    support_candidates: list[dict[str, str]] | None = None,
    target_visual_context: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "case_id": case_id,
        "episode_id": "episode-001",
        "frames": [{"rgb_path": str(frame), "step": 1}],
        "primary_frame": {"rgb_path": str(frame), "step": 1},
        "question_task_hint": "Use the visible evidence and answer_options.",
        "question_text": "Where is the object?",
        "question_type": question_type,
        "target": {"label": "apple"},
    }
    if support_candidates is not None:
        row["support_candidates"] = support_candidates
    if target_visual_context is not None:
        row["target_visual_context"] = target_visual_context
    return row


def _write_bundle(path: Path, cases: list[dict[str, object]]) -> None:
    payload = {
        "schema_version": "dsg-spatialqa-lab.active-qa-v2-vlm-request-bundle.v1",
        "leak_free": True,
        "leak_paths": [],
        "prediction_cases": cases,
        "request_count": len(cases),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
