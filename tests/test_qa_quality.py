from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from _pytest.capture import CaptureFixture

import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
QA_QUALITY_SCRIPT = ROOT / "scripts" / "audit_qa_quality.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_quality_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "audit_qa_quality_script",
        QA_QUALITY_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qa_quality_report_flags_object_location_only_dataset() -> None:
    cases = (
        _case("case:apple", "apple_1", "countertop_1", relation="ON"),
        _case("case:book", "book_1", "room_1", relation="IN_ROOM"),
        _case("case:mug", "mug_1", "room_1", relation="IN_ROOM"),
    )

    report = lab.qa_quality_report(cases)
    validation = lab.validate_qa_quality_report(report)

    assert report["summary"]["case_count"] == 3
    assert report["summary"]["question_type_counts"] == {"object_location": 3}
    assert report["summary"]["question_type_count"] == 1
    assert report["summary"]["relation_counts"] == {"IN_ROOM": 2, "ON": 1}
    assert report["quality_gates"]["question_type_coverage"]["passed"] is False
    assert report["quality_gates"]["relation_balance"]["passed"] is False
    assert report["summary"]["language_prior_risk_counts"] == {
        "high": 2,
        "medium": 1,
    }
    assert report["summary"]["schema_gap_counts"]["question_text"] == 3
    assert validation["valid"] is True


def test_qa_quality_report_uses_observability_splits() -> None:
    cases = (
        _case("case:apple", "apple_1", "countertop_1", relation="ON"),
        _case("case:book", "book_1", "room_1", relation="IN_ROOM"),
    )
    observability_report = {
        "splits": {
            "evidence_observable": ["case:apple"],
            "target_observable": ["case:apple", "case:book"],
            "target_observable_relation_missing": ["case:book"],
            "target_missing": [],
            "missing_evidence": ["case:book"],
        },
        "cases": [
            {"case_id": "case:apple", "observability_status": "evidence_observable"},
            {
                "case_id": "case:book",
                "observability_status": "target_observable_relation_missing",
            },
        ],
    }

    report = lab.qa_quality_report(cases, observability_report=observability_report)

    assert report["summary"]["split_counts"] == {
        "full_oracle": 2,
        "observation_aware": 1,
        "target_observable": 2,
        "missing_evidence": 1,
        "target_missing": 0,
        "target_observable_relation_missing": 1,
        "situated": 0,
        "temporal": 0,
        "anti_shortcut_candidate": 1,
    }
    assert {
        row["case_id"]: row["recommended_splits"] for row in report["cases"]
    } == {
        "case:apple": ["full_oracle", "observation_aware", "target_observable"],
        "case:book": [
            "full_oracle",
            "target_observable",
            "missing_evidence",
            "target_observable_relation_missing",
        ],
    }


def test_vlm_request_bundle_leak_audit_detects_gold_fields() -> None:
    clean_bundle = {
        "cases": [
            {
                "case_id": "case:apple",
                "question": "Where is the apple?",
                "answer_schema": {"relation": "string", "dst_label": "string"},
            }
        ]
    }
    leaky_bundle = {
        "cases": [
            {
                "case_id": "case:apple",
                "question": "Where is the apple?",
                "gold_answer": {"relation": "ON", "dst": "countertop_1"},
                "required_edges": ["apple_1-ON-countertop_1-1"],
            }
        ]
    }

    clean = lab.audit_vlm_request_bundle_for_gold_leakage(clean_bundle)
    leaky = lab.audit_vlm_request_bundle_for_gold_leakage(leaky_bundle)

    assert clean == {
        "leak_free": True,
        "leak_count": 0,
        "leaked_fields": [],
        "leaks": [],
    }
    assert leaky["leak_free"] is False
    assert leaky["leaked_fields"] == ["gold_answer", "required_edges"]
    assert leaky["leak_count"] == 2


def test_qa_quality_cli_writes_and_validates_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_quality_script()
    main = cast(MainFn, getattr(module, "main"))
    qa_path = tmp_path / "qa.jsonl"
    report_path = tmp_path / "qa-quality-report.json"
    lab.save_qa_dataset((_case("case:apple", "apple_1", "countertop_1"),), qa_path)

    assert main(["--qa", str(qa_path), "--report", str(report_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "qa_quality_report"
    assert output["valid"] is True
    assert output["summary"]["case_count"] == 1

    assert main(["--validate-report", str(report_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_qa_quality_report"
    assert validation["valid"] is True


def _case(
    case_id: str,
    object_id: str,
    dst: str,
    *,
    relation: str = "ON",
    question_type: str = "object_location",
) -> lab.QACase:
    return lab.QACase(
        id=case_id,
        scene_id="FloorPlan1",
        episode_id="episode-001",
        graph_digest="0" * 64,
        step=1,
        question={"type": question_type, "object_id": object_id},
        question_type=question_type,
        answer={
            "object_id": object_id,
            "label": object_id.split("_")[0],
            "current_location": {"relation": relation, "dst": dst, "step": 1},
        },
        answer_type=question_type,
        required_nodes=(object_id, dst),
        required_edges=(f"{object_id}-{relation}-{dst}-1",),
        tags=("generated", "qa", question_type),
    )
