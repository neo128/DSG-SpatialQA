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
IMPORT_PREDICTIONS_SCRIPT = ROOT / "scripts" / "import_predictions.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_import_predictions_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "import_predictions_script",
        IMPORT_PREDICTIONS_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_offline_prediction_import_normalizes_records_and_reports(tmp_path: Path) -> None:
    assert hasattr(lab, "OfflinePredictionRecord")
    assert hasattr(lab, "offline_prediction_record_to_dict")
    assert hasattr(lab, "offline_prediction_record_from_dict")
    assert hasattr(lab, "offline_prediction_records_jsonl")
    assert hasattr(lab, "offline_prediction_records_digest")
    assert hasattr(lab, "offline_prediction_records_from_jsonl")
    assert hasattr(lab, "import_offline_predictions")
    assert hasattr(lab, "offline_prediction_import_report_digest")
    assert hasattr(lab, "save_offline_prediction_import_report")
    assert hasattr(lab, "load_offline_prediction_import_report")
    assert hasattr(lab, "validate_offline_prediction_import_report")
    assert hasattr(lab, "compare_offline_prediction_import_report")
    cases = _cases()
    records = _records(cases)
    qa_path = tmp_path / "qa.jsonl"
    input_path = tmp_path / "offline-input.jsonl"
    pred_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "import-report.json"
    lab.save_qa_dataset(cases, qa_path)
    lab.save_offline_prediction_records(records, input_path)

    predictions, report = lab.import_offline_predictions(
        cases,
        records,
        source_name="vlm_fixture",
        source_kind="vlm",
        source_metadata={"prompt_id": "spatial-qa-v1"},
        qa_path=qa_path,
        input_path=input_path,
        prediction_path=pred_path,
    )
    lab.save_qa_predictions(predictions, pred_path)
    lab.save_offline_prediction_import_report(report, report_path)
    loaded_report = lab.load_offline_prediction_import_report(report_path)
    validation = lab.validate_offline_prediction_import_report(loaded_report)
    comparison = lab.compare_offline_prediction_import_report(loaded_report)

    assert [
        lab.offline_prediction_record_from_dict(lab.offline_prediction_record_to_dict(record))
        for record in records
    ] == list(records)
    assert lab.offline_prediction_records_jsonl(records).endswith("\n")
    assert lab.offline_prediction_records_digest(records) == lab.offline_prediction_records_digest(
        lab.load_offline_prediction_records(input_path)
    )
    assert [prediction.id for prediction in predictions] == [cases[0].id, cases[1].id]
    assert predictions[0].answer == cases[0].answer
    assert predictions[0].evidence_nodes == cases[0].required_nodes
    assert predictions[0].evidence_edges == cases[0].required_edges
    assert report["source"] == {
        "kind": "vlm",
        "metadata": {"prompt_id": "spatial-qa-v1"},
        "name": "vlm_fixture",
    }
    assert report["summary"] == {
        "duplicate_case_count": 0,
        "error_prediction_count": 0,
        "gold_case_count": 3,
        "imported_prediction_count": 2,
        "missing_case_count": 1,
        "record_count": 3,
        "unknown_case_count": 1,
    }
    assert report["missing_case_ids"] == [cases[2].id]
    assert report["unknown_case_ids"] == ["not_in_gold"]
    assert report["records"][-1] == {
        "case_id": "not_in_gold",
        "error": "unknown_case",
        "imported": False,
        "line_number": 3,
    }
    assert report["prediction_digest"] == lab.qa_predictions_digest(predictions)
    assert report["report_digest"] == lab.offline_prediction_import_report_digest(report)
    assert validation["valid"] is True
    assert comparison["matches"] is True


def test_import_predictions_cli_writes_predictions_and_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_import_predictions_script()
    main = cast(MainFn, getattr(module, "main"))
    cases = _cases()
    records = _records(cases)[:2]
    qa_path = tmp_path / "qa.jsonl"
    input_path = tmp_path / "offline-input.jsonl"
    pred_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "import-report.json"
    lab.save_qa_dataset(cases, qa_path)
    lab.save_offline_prediction_records(records, input_path)

    assert main(
        [
            "--qa",
            str(qa_path),
            "--input",
            str(input_path),
            "--source-name",
            "caption_fixture",
            "--source-kind",
            "caption_memory",
            "--metadata",
            "prompt_id=caption-v1",
            "--pred",
            str(pred_path),
            "--report",
            str(report_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    predictions = lab.load_qa_predictions(pred_path)
    report = lab.load_offline_prediction_import_report(report_path)
    assert output == {
        "action": "import_predictions",
        "path": str(pred_path),
        "valid": True,
        "digest": report["report_digest"],
        "prediction_digest": lab.qa_predictions_digest(predictions),
        "summary": report["summary"],
    }
    assert report["source"]["metadata"] == {"prompt_id": "caption-v1"}
    assert report["summary"]["missing_case_count"] == 1

    assert main(["--validate-report", str(report_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_offline_prediction_import_report"
    assert validation["path"] == str(report_path)
    assert validation["valid"] is True

    assert main(["--compare-report", str(report_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_offline_prediction_import_report"
    assert comparison["path"] == str(report_path)
    assert comparison["matches"] is True


def test_import_predictions_cli_returns_structured_json_for_invalid_input(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_import_predictions_script()
    main = cast(MainFn, getattr(module, "main"))
    qa_path = tmp_path / "qa.jsonl"
    input_path = tmp_path / "invalid.jsonl"
    pred_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "import-report.json"
    lab.save_qa_dataset(_cases(), qa_path)
    input_path.write_text("[]\n", encoding="utf-8")

    assert main(
        [
            "--qa",
            str(qa_path),
            "--input",
            str(input_path),
            "--source-name",
            "vlm_fixture",
            "--pred",
            str(pred_path),
            "--report",
            str(report_path),
        ]
    ) == 1

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "action": "import_predictions",
        "path": str(pred_path),
        "valid": False,
        "error": "Offline prediction line 1 must be an object",
    }


def _cases() -> tuple[lab.QACase, ...]:
    graph = lab.load_scene_fixture("tabletop")
    generated = lab.generate_qa_cases(
        graph,
        scene_id="tabletop_scene",
        episode_id="episode_001",
    )
    location = next(case for case in generated if case.question_type == "object_location")
    relation = next(case for case in generated if case.question_type == "relative_relation")
    nearest = next(case for case in generated if case.question_type == "nearest_object")
    return (location, relation, nearest)


def _records(cases: tuple[lab.QACase, ...]) -> tuple[lab.OfflinePredictionRecord, ...]:
    relation_answer = lab.qa_case_from_dict(lab.qa_case_to_dict(cases[1])).answer
    relation_answer["holds"] = not bool(cases[1].answer["holds"])
    return (
        lab.OfflinePredictionRecord(
            case_id=cases[0].id,
            answer=cases[0].answer,
            evidence_nodes=cases[0].required_nodes,
            evidence_edges=cases[0].required_edges,
            confidence=0.91,
        ),
        lab.OfflinePredictionRecord(
            case_id=cases[1].id,
            answer=relation_answer,
            confidence=0.34,
            metadata={"raw_answer": "no"},
        ),
        lab.OfflinePredictionRecord(
            case_id="not_in_gold",
            answer={"text": "extra"},
            confidence=0.2,
        ),
    )
