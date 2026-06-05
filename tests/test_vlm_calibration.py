from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

from _pytest.capture import CaptureFixture
import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
VLM_CALIBRATION_SCRIPT = ROOT / "scripts" / "eval_vlm_calibration.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_vlm_calibration_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "eval_vlm_calibration_test_script",
        VLM_CALIBRATION_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_vlm_observable_slice_keeps_visible_object_location_only() -> None:
    visible = _object_location_case(
        "case-visible",
        visible=True,
        relation="ON",
        dst="countertop_001",
    )
    hidden = _object_location_case(
        "case-hidden",
        visible=False,
        relation="IN_ROOM",
        dst="ai2thor_room",
    )

    report = lab.vlm_observable_slice_report([visible, hidden])

    assert report["summary"] == {
        "case_count": 2,
        "observable_case_count": 1,
        "observable_case_rate": 0.5,
        "object_location_case_count": 2,
        "visible_object_location_case_count": 1,
    }
    assert report["observable_case_ids"] == ["case-visible"]
    assert report["cases"][0]["included"] is True
    assert report["cases"][1]["included"] is False
    assert report["cases"][1]["exclusion_reason"] == "target_not_visible"
    assert lab.validate_vlm_observable_slice_report(report)["valid"] is True


def test_vlm_semantic_eval_accepts_natural_language_location_without_oracle_id() -> None:
    case = _object_location_case(
        "case-apple",
        visible=True,
        relation="ON",
        dst="countertop_001",
    )
    prediction = lab.QAPrediction(
        id="case-apple",
        answer={"model": "qwen3.7-plus", "source": "vlm", "text": "on the countertop"},
        confidence=0.8,
        error=None,
    )

    report = lab.vlm_semantic_eval_report([case], [prediction])

    assert report["summary"]["semantic_match_count"] == 1
    assert report["summary"]["semantic_match_rate"] == 1.0
    assert report["summary"]["strict_exact_match_count"] == 0
    assert report["cases"][0]["semantic_match"] is True
    assert report["cases"][0]["gold"] == {
        "destination_label": "countertop",
        "relation": "ON",
        "target_label": "object",
        "visible": True,
    }
    assert report["cases"][0]["prediction"]["destination_label"] == "countertop"
    assert report["cases"][0]["prediction"]["relation"] == "ON"
    assert lab.validate_vlm_semantic_eval_report(report)["valid"] is True


def test_vlm_semantic_eval_accepts_structured_graph_tool_location_id() -> None:
    case = _object_location_case(
        "case-apple",
        visible=True,
        relation="ON",
        dst="countertop_001",
    )
    prediction = lab.QAPrediction(
        id="case-apple",
        answer={
            "confidence": 1.0,
            "current_location": {
                "dst": "countertop_001",
                "relation": "ON",
                "step": 10,
            },
            "label": "apple",
            "object_id": "apple_001",
        },
        confidence=1.0,
        error=None,
    )

    report = lab.vlm_semantic_eval_report([case], [prediction])

    assert report["summary"]["semantic_match_count"] == 1
    assert report["cases"][0]["semantic_match"] is True
    assert report["cases"][0]["prediction"]["destination_label"] == "countertop"


def test_vlm_semantic_eval_accepts_floor_as_room_level_location() -> None:
    case = _object_location_case(
        "case-chair",
        visible=True,
        relation="IN_ROOM",
        dst="ai2thor_room",
    )
    prediction = lab.QAPrediction(
        id="case-chair",
        answer={
            "current_location": {
                "dst": "floor_001",
                "relation": "ON",
            }
        },
        confidence=1.0,
        error=None,
    )

    report = lab.vlm_semantic_eval_report([case], [prediction])

    assert report["summary"]["semantic_match_count"] == 1
    assert report["cases"][0]["semantic_match"] is True
    assert report["cases"][0]["failure_reason"] is None


def test_vlm_semantic_eval_accepts_dining_room_as_room_level_location() -> None:
    case = _object_location_case(
        "case-chair",
        visible=True,
        relation="IN_ROOM",
        dst="ai2thor_room",
    )
    prediction = lab.QAPrediction(
        id="case-chair",
        answer={
            "current_location": {
                "dst_label": "dining room",
                "relation": "IN_ROOM",
            }
        },
        confidence=1.0,
        error=None,
    )

    report = lab.vlm_semantic_eval_report([case], [prediction])

    assert report["summary"]["semantic_match_count"] == 1
    assert report["cases"][0]["semantic_match"] is True
    assert report["cases"][0]["failure_reason"] is None


def test_vlm_semantic_eval_accepts_affordance_support_as_specific_room_location() -> None:
    case = _object_location_case(
        "case-faucet",
        visible=True,
        relation="IN_ROOM",
        dst="ai2thor_room",
        label="faucet",
    )
    prediction = lab.QAPrediction(
        id="case-faucet",
        answer={
            "current_location": {
                "dst_label": "sink",
                "relation": "ON",
            }
        },
        confidence=1.0,
        error=None,
    )

    report = lab.vlm_semantic_eval_report([case], [prediction])

    assert report["summary"]["semantic_match_count"] == 1
    assert report["cases"][0]["semantic_match"] is True
    assert report["cases"][0]["failure_reason"] is None


def test_vlm_semantic_eval_does_not_treat_every_support_as_room_level() -> None:
    case = _object_location_case(
        "case-chair",
        visible=True,
        relation="IN_ROOM",
        dst="ai2thor_room",
        label="chair",
    )
    prediction = lab.QAPrediction(
        id="case-chair",
        answer={
            "current_location": {
                "dst": "countertop_001",
                "relation": "ON",
            }
        },
        confidence=1.0,
        error=None,
    )

    report = lab.vlm_semantic_eval_report([case], [prediction])

    assert report["summary"]["semantic_match_count"] == 0
    assert report["cases"][0]["semantic_match"] is False
    assert report["cases"][0]["failure_reason"] == "relation_mismatch"


def test_vlm_semantic_eval_excludes_hidden_cases_by_default() -> None:
    hidden = _object_location_case(
        "case-hidden",
        visible=False,
        relation="IN_ROOM",
        dst="ai2thor_room",
    )
    prediction = lab.QAPrediction(
        id="case-hidden",
        answer={"text": "in the bathroom"},
        confidence=0.7,
        error=None,
    )

    report = lab.vlm_semantic_eval_report([hidden], [prediction])

    assert report["summary"]["case_count"] == 0
    assert report["excluded_case_ids"] == ["case-hidden"]


def test_vlm_semantic_eval_delta_report_compares_candidate_to_baseline() -> None:
    candidate_cases = [
        {
            "case_id": "case-1",
            "semantic_match": False,
            "strict_exact_match": False,
        },
        {
            "case_id": "case-2",
            "semantic_match": False,
            "strict_exact_match": False,
        },
    ]
    baseline_cases = [
        {
            "case_id": "case-1",
            "semantic_match": True,
            "strict_exact_match": False,
        },
        {
            "case_id": "case-2",
            "semantic_match": False,
            "strict_exact_match": False,
        },
    ]
    candidate = {
        "schema_version": "dsg-spatialqa-lab.vlm-semantic-eval-report.v1",
        "gold_digest": "a" * 64,
        "prediction_digest": "b" * 64,
        "summary": {
            "case_count": 2,
            "matched_prediction_count": 2,
            "prediction_count": 2,
            "semantic_match_count": 0,
            "semantic_match_rate": 0.0,
            "strict_exact_match_count": 0,
            "strict_exact_match_rate": 0.0,
        },
        "cases": candidate_cases,
    }
    baseline = {
        "schema_version": "dsg-spatialqa-lab.vlm-semantic-eval-report.v1",
        "gold_digest": "a" * 64,
        "prediction_digest": "c" * 64,
        "summary": {
            "case_count": 2,
            "matched_prediction_count": 2,
            "prediction_count": 2,
            "semantic_match_count": 1,
            "semantic_match_rate": 0.5,
            "strict_exact_match_count": 0,
            "strict_exact_match_rate": 0.0,
        },
        "cases": baseline_cases,
    }
    candidate["report_digest"] = lab.vlm_semantic_eval_report_digest(candidate)
    baseline["report_digest"] = lab.vlm_semantic_eval_report_digest(baseline)

    report = lab.vlm_semantic_eval_delta_report(
        candidate,
        baseline,
        candidate_name="dsg_candidate",
        baseline_name="vlm",
    )

    assert report["candidate_name"] == "dsg_candidate"
    assert report["baseline_name"] == "vlm"
    assert report["gold_digest_match"] is True
    assert report["case_count_match"] is True
    assert report["summary_delta"] == {
        "baseline_semantic_match_count": 1,
        "baseline_semantic_match_rate": 0.5,
        "candidate_semantic_match_count": 0,
        "candidate_semantic_match_rate": 0.0,
        "semantic_match_count_delta": -1,
        "semantic_match_rate_delta": -0.5,
    }
    assert report["paired"] == {
        "case_count": 2,
        "paired_losses": 1,
        "paired_ties": 1,
        "paired_wins": 0,
    }
    assert report["paired_significance"] == {
        "candidate_loss_count": 1,
        "candidate_win_count": 0,
        "discordant_case_count": 1,
        "method": "exact_paired_sign_test_mcnemar_like",
        "significant_at_0_05": False,
        "two_sided_p_value": 1.0,
    }
    assert report["decision"] == "candidate_regressed"
    assert report["report_digest"] == lab.vlm_semantic_eval_delta_report_digest(report)
    assert lab.validate_vlm_semantic_eval_delta_report(report)["valid"] is True


def test_vlm_semantic_eval_delta_report_flags_strong_paired_lift() -> None:
    candidate_cases = [
        {"case_id": f"case-{index}", "semantic_match": True, "strict_exact_match": False}
        for index in range(10)
    ]
    baseline_cases = [
        {"case_id": f"case-{index}", "semantic_match": False, "strict_exact_match": False}
        for index in range(10)
    ]
    candidate = {
        "schema_version": "dsg-spatialqa-lab.vlm-semantic-eval-report.v1",
        "gold_digest": "a" * 64,
        "prediction_digest": "b" * 64,
        "summary": {
            "case_count": 10,
            "matched_prediction_count": 10,
            "prediction_count": 10,
            "semantic_match_count": 10,
            "semantic_match_rate": 1.0,
            "strict_exact_match_count": 0,
            "strict_exact_match_rate": 0.0,
        },
        "cases": candidate_cases,
    }
    baseline = {
        "schema_version": "dsg-spatialqa-lab.vlm-semantic-eval-report.v1",
        "gold_digest": "a" * 64,
        "prediction_digest": "c" * 64,
        "summary": {
            "case_count": 10,
            "matched_prediction_count": 10,
            "prediction_count": 10,
            "semantic_match_count": 0,
            "semantic_match_rate": 0.0,
            "strict_exact_match_count": 0,
            "strict_exact_match_rate": 0.0,
        },
        "cases": baseline_cases,
    }
    candidate["report_digest"] = lab.vlm_semantic_eval_report_digest(candidate)
    baseline["report_digest"] = lab.vlm_semantic_eval_report_digest(baseline)

    report = lab.vlm_semantic_eval_delta_report(candidate, baseline)

    assert report["paired"] == {
        "case_count": 10,
        "paired_losses": 0,
        "paired_ties": 0,
        "paired_wins": 10,
    }
    assert report["paired_significance"] == {
        "candidate_loss_count": 0,
        "candidate_win_count": 10,
        "discordant_case_count": 10,
        "method": "exact_paired_sign_test_mcnemar_like",
        "significant_at_0_05": True,
        "two_sided_p_value": 0.001953,
    }
    assert lab.validate_vlm_semantic_eval_delta_report(report)["valid"] is True


def test_vlm_semantic_eval_delta_report_groups_by_question_type() -> None:
    candidate_cases = [
        {
            "case_id": "case-location-win",
            "question_type": "object_location",
            "semantic_match": True,
            "strict_exact_match": False,
        },
        {
            "case_id": "case-location-tie",
            "question_type": "object_location",
            "semantic_match": True,
            "strict_exact_match": False,
        },
        {
            "case_id": "case-memory-loss",
            "question_type": "dynamic_memory",
            "semantic_match": False,
            "strict_exact_match": False,
        },
    ]
    baseline_cases = [
        {
            "case_id": "case-location-win",
            "question_type": "object_location",
            "semantic_match": False,
            "strict_exact_match": False,
        },
        {
            "case_id": "case-location-tie",
            "question_type": "object_location",
            "semantic_match": True,
            "strict_exact_match": False,
        },
        {
            "case_id": "case-memory-loss",
            "question_type": "dynamic_memory",
            "semantic_match": True,
            "strict_exact_match": False,
        },
    ]
    candidate = {
        "schema_version": "dsg-spatialqa-lab.vlm-semantic-eval-report.v1",
        "gold_digest": "a" * 64,
        "prediction_digest": "b" * 64,
        "summary": {
            "case_count": 3,
            "matched_prediction_count": 3,
            "prediction_count": 3,
            "semantic_match_count": 2,
            "semantic_match_rate": 2 / 3,
            "strict_exact_match_count": 0,
            "strict_exact_match_rate": 0.0,
        },
        "cases": candidate_cases,
    }
    baseline = {
        "schema_version": "dsg-spatialqa-lab.vlm-semantic-eval-report.v1",
        "gold_digest": "a" * 64,
        "prediction_digest": "c" * 64,
        "summary": {
            "case_count": 3,
            "matched_prediction_count": 3,
            "prediction_count": 3,
            "semantic_match_count": 2,
            "semantic_match_rate": 2 / 3,
            "strict_exact_match_count": 0,
            "strict_exact_match_rate": 0.0,
        },
        "cases": baseline_cases,
    }
    candidate["report_digest"] = lab.vlm_semantic_eval_report_digest(candidate)
    baseline["report_digest"] = lab.vlm_semantic_eval_report_digest(baseline)

    report = lab.vlm_semantic_eval_delta_report(candidate, baseline)

    assert report["question_type_groups"] == [
        {
            "baseline_semantic_match_count": 1,
            "candidate_semantic_match_count": 0,
            "case_count": 1,
            "decision": "candidate_regressed",
            "paired": {
                "case_count": 1,
                "paired_losses": 1,
                "paired_ties": 0,
                "paired_wins": 0,
            },
            "paired_significance": {
                "candidate_loss_count": 1,
                "candidate_win_count": 0,
                "discordant_case_count": 1,
                "method": "exact_paired_sign_test_mcnemar_like",
                "significant_at_0_05": False,
                "two_sided_p_value": 1.0,
            },
            "question_type": "dynamic_memory",
            "semantic_match_count_delta": -1,
        },
        {
            "baseline_semantic_match_count": 1,
            "candidate_semantic_match_count": 2,
            "case_count": 2,
            "decision": "candidate_improved",
            "paired": {
                "case_count": 2,
                "paired_losses": 0,
                "paired_ties": 1,
                "paired_wins": 1,
            },
            "paired_significance": {
                "candidate_loss_count": 0,
                "candidate_win_count": 1,
                "discordant_case_count": 1,
                "method": "exact_paired_sign_test_mcnemar_like",
                "significant_at_0_05": False,
                "two_sided_p_value": 1.0,
            },
            "question_type": "object_location",
            "semantic_match_count_delta": 1,
        },
    ]
    assert lab.validate_vlm_semantic_eval_delta_report(report)["valid"] is True


def test_vlm_observable_request_bundle_filters_cases_without_gold_fields() -> None:
    bundle: dict[str, Any] = {
        "action": "offline_control_prediction_request_bundle",
        "case_count": 2,
        "case_inputs": [
            {
                "case_id": "case-visible",
                "question_text": "Where is the apple?",
                "target": {"label": "apple", "object_id": "apple_001"},
            },
            {
                "case_id": "case-hidden",
                "question_text": "Where is the book?",
                "target": {"label": "book", "object_id": "book_001"},
            },
        ],
        "prediction_templates": {
            "qa_prediction": [
                {"id": "case-visible", "answer": {}, "evidence_nodes": []},
                {"id": "case-hidden", "answer": {}, "evidence_nodes": []},
            ]
        },
        "request_bundle_digest": "old",
    }

    filtered = lab.vlm_observable_request_bundle(
        bundle,
        observable_case_ids=("case-visible",),
    )

    assert filtered["case_count"] == 1
    assert [case["case_id"] for case in filtered["case_inputs"]] == ["case-visible"]
    assert [case["id"] for case in filtered["prediction_templates"]["qa_prediction"]] == [
        "case-visible"
    ]
    serialized = lab.vlm_observable_request_bundle_json(filtered)
    assert "gold_answer" not in serialized
    assert "gold_evidence" not in serialized
    assert "\"answer\":{\"" not in serialized
    assert filtered["request_bundle_digest"] == lab.vlm_observable_request_bundle_digest(
        filtered
    )


def test_vlm_support_candidate_request_bundle_adds_visible_support_labels_without_gold() -> None:
    bundle: dict[str, Any] = {
        "action": "offline_control_prediction_request_bundle",
        "case_count": 1,
        "case_inputs": [
            {
                "case_id": "case-apple",
                "episode_id": "episode-001",
                "scene_id": "FloorPlan1",
                "question_type": "object_location",
                "question_text": "Where is the apple?",
                "target": {"label": "apple", "object_id": "apple_1"},
                "primary_frame": {
                    "episode_id": "episode-001",
                    "scene_id": "FloorPlan1",
                    "step": 2,
                },
            }
        ],
        "request_bundle_digest": "old",
    }
    frame_index = [
        {
            "episode_id": "episode-001",
            "scene_id": "FloorPlan1",
            "step": 2,
            "visible_object_ids": ["apple_1", "countertop_1", "sink_1"],
            "visible_object_labels": ["apple", "countertop", "sink"],
        }
    ]

    enriched = lab.vlm_support_candidate_request_bundle(
        bundle,
        frame_index,
        max_candidates_per_case=4,
    )

    assert enriched["case_inputs"][0]["support_candidates"] == [
        {
            "label": "countertop",
            "relation_hint": "ON",
            "source": "primary_frame_visible_label",
        },
        {
            "label": "sink",
            "relation_hint": "INSIDE",
            "source": "primary_frame_visible_label",
        },
    ]
    assert enriched["vlm_support_candidate_enrichment"]["summary"] == {
        "case_count": 1,
        "cases_with_primary_frame": 1,
        "cases_with_support_candidates": 1,
        "support_candidate_count": 2,
    }
    assert enriched["request_bundle_digest"] == lab.vlm_support_candidate_request_bundle_digest(
        enriched
    )
    serialized = lab.vlm_support_candidate_request_bundle_json(enriched)
    assert "gold_answer" not in serialized
    assert "gold_evidence" not in serialized
    assert "countertop_1" not in serialized
    assert "sink_1" not in serialized


def test_vlm_calibration_cli_writes_retry_request_bundle_without_gold(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    request_bundle_path = tmp_path / "request-bundle.json"
    semantic_report_path = tmp_path / "semantic-report.json"
    output_path = tmp_path / "retry-bundle.json"
    request_bundle_path.write_text(
        json.dumps(
            {
                "case_count": 2,
                "case_inputs": [
                    {"case_id": "case-ok", "question_text": "Where is the apple?"},
                    {
                        "case_id": "case-retry",
                        "question_text": "Where is the bottle?",
                        "gold_answer": "bottle ON shelf",
                    },
                ],
                "request_bundle_digest": "old",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    semantic_report_path.write_text(
        json.dumps(
            {
                "report_digest": "b" * 64,
                "summary": {"case_count": 2},
                "cases": [
                    {"case_id": "case-ok", "semantic_match": True},
                    {
                        "case_id": "case-retry",
                        "semantic_match": False,
                        "failure_reason": "target_not_observed",
                        "gold": {"relation": "ON", "destination_label": "shelf"},
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    module = load_vlm_calibration_script()
    main = cast(MainFn, getattr(module, "main"))

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle_path),
            "--retry-semantic-eval-report",
            str(semantic_report_path),
            "--retry-request-bundle-output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"action": "build_vlm_retry_request_bundle"' in output
    retry = json.loads(output_path.read_text(encoding="utf-8"))
    assert retry["case_count"] == 1
    assert retry["case_inputs"] == [
        {"case_id": "case-retry", "question_text": "Where is the bottle?"}
    ]
    serialized = json.dumps(retry, sort_keys=True)
    assert "gold_answer" not in serialized
    assert "gold_evidence" not in serialized
    assert "target_not_observed" not in serialized


def test_vlm_calibration_cli_merges_retry_predictions(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    original_path = tmp_path / "original.jsonl"
    retry_path = tmp_path / "retry.jsonl"
    merged_path = tmp_path / "merged.jsonl"
    semantic_report_path = tmp_path / "semantic-report.json"
    merge_report_path = tmp_path / "merge-report.json"
    original_predictions = [
        lab.QAPrediction(id="case-ok", answer={"text": "ok"}, confidence=0.8),
        lab.QAPrediction(id="case-retry", answer={"text": "old"}, confidence=0.2),
    ]
    retry_predictions = [
        lab.QAPrediction(id="case-retry", answer={"text": "new"}, confidence=0.9)
    ]
    lab.save_qa_predictions(original_predictions, original_path)
    lab.save_qa_predictions(retry_predictions, retry_path)
    semantic_report_path.write_text(
        json.dumps(
            {
                "report_digest": "c" * 64,
                "cases": [
                    {"case_id": "case-ok", "semantic_match": True},
                    {"case_id": "case-retry", "semantic_match": False},
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    module = load_vlm_calibration_script()
    main = cast(MainFn, getattr(module, "main"))

    exit_code = main(
        [
            "--retry-semantic-eval-report",
            str(semantic_report_path),
            "--merge-original-predictions",
            str(original_path),
            "--merge-retry-predictions",
            str(retry_path),
            "--merged-predictions-output",
            str(merged_path),
            "--retry-merge-report",
            str(merge_report_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"action": "merge_vlm_retry_predictions"' in output
    merged = lab.load_qa_predictions(merged_path)
    assert [prediction.answer for prediction in merged] == [
        {"text": "ok"},
        {"text": "new"},
    ]
    report = lab.load_vlm_retry_merge_report(merge_report_path)
    assert report["ready"] is True
    assert report["summary"]["replaced_retry_case_count"] == 1
    assert lab.validate_vlm_retry_merge_report(report)["valid"] is True


def test_vlm_calibration_cli_merges_retry_predictions_with_request_bundle_scope(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    original_path = tmp_path / "original.jsonl"
    retry_path = tmp_path / "retry.jsonl"
    merged_path = tmp_path / "merged.jsonl"
    semantic_report_path = tmp_path / "semantic-report.json"
    request_bundle_path = tmp_path / "retry-request-bundle.json"
    merge_report_path = tmp_path / "merge-report.json"
    original_predictions = [
        lab.QAPrediction(id="case-success", answer={"text": "ok"}),
        lab.QAPrediction(id="case-ready-retry", answer={"text": "old"}),
        lab.QAPrediction(id="case-blocked-crop", answer={"text": "blocked old"}),
    ]
    retry_predictions = [
        lab.QAPrediction(id="case-ready-retry", answer={"text": "new"})
    ]
    lab.save_qa_predictions(original_predictions, original_path)
    lab.save_qa_predictions(retry_predictions, retry_path)
    semantic_report_path.write_text(
        json.dumps(
            {
                "report_digest": "d" * 64,
                "cases": [
                    {"case_id": "case-success", "semantic_match": True},
                    {"case_id": "case-ready-retry", "semantic_match": False},
                    {"case_id": "case-blocked-crop", "semantic_match": False},
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    request_bundle_path.write_text(
        json.dumps(
            {
                "request_bundle_digest": "e" * 64,
                "case_inputs": [{"case_id": "case-ready-retry"}],
                "vlm_retry_enrichment": {
                    "retry_case_ids": [
                        "case-ready-retry",
                        "case-blocked-crop",
                    ]
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    module = load_vlm_calibration_script()
    main = cast(MainFn, getattr(module, "main"))

    exit_code = main(
        [
            "--retry-semantic-eval-report",
            str(semantic_report_path),
            "--merge-original-predictions",
            str(original_path),
            "--merge-retry-predictions",
            str(retry_path),
            "--merge-retry-request-bundle",
            str(request_bundle_path),
            "--merged-predictions-output",
            str(merged_path),
            "--retry-merge-report",
            str(merge_report_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"retry_scope_kind": "explicit_retry_case_ids"' in output
    merged = lab.load_qa_predictions(merged_path)
    assert [prediction.answer for prediction in merged] == [
        {"text": "ok"},
        {"text": "new"},
        {"text": "blocked old"},
    ]
    report = lab.load_vlm_retry_merge_report(merge_report_path)
    assert report["ready"] is True
    assert report["retry_case_ids"] == ["case-ready-retry"]
    assert report["out_of_scope_failed_case_ids"] == ["case-blocked-crop"]


def test_vlm_answer_option_request_bundle_constrains_visible_support_choices_without_gold() -> None:
    bundle: dict[str, Any] = {
        "action": "offline_control_prediction_request_bundle",
        "case_count": 1,
        "case_inputs": [
            {
                "case_id": "case-apple",
                "question_type": "object_location",
                "question_text": "Where is the apple?",
                "target": {"label": "apple", "object_id": "apple_1"},
                "support_candidates": [
                    {
                        "label": "countertop",
                        "object_id": "countertop_1",
                        "relation_hint": "ON",
                    },
                    {
                        "label": "sink",
                        "object_id": "sink_1",
                        "relation_hint": "INSIDE",
                    },
                ],
            }
        ],
        "request_bundle_digest": "old",
    }

    enriched = lab.vlm_answer_option_request_bundle(bundle)

    assert enriched["case_inputs"][0]["answer_options"] == [
        {
            "destination_label": "countertop",
            "option_id": "locopt_001",
            "relation": "ON",
            "source": "support_candidate",
        },
        {
            "destination_label": "sink",
            "option_id": "locopt_002",
            "relation": "INSIDE",
            "source": "support_candidate",
        },
        {
            "destination_label": "room",
            "option_id": "locopt_003",
            "relation": "IN_ROOM",
            "source": "fallback_room",
        },
    ]
    assert enriched["vlm_answer_option_enrichment"]["summary"] == {
        "answer_option_count": 3,
        "case_count": 1,
        "cases_with_answer_options": 1,
    }
    assert enriched["case_inputs"][0]["answer_option_response_schema"] == {
        "answer_current_location_rule": (
            "Copy relation and destination_label from the selected answer option."
        ),
        "answer_option_id_field": "answer.answer_option_id",
        "allowed_answer_option_ids": ["locopt_001", "locopt_002", "locopt_003"],
        "required_when_answer_options_present": True,
    }
    assert (
        enriched["case_inputs"][0]["answer_schema_hint"]["answer_option_rule"]
        == "When answer_options is non-empty, choose answer.answer_option_id from "
        "allowed_answer_option_ids and copy that option's relation and "
        "destination_label into answer.current_location. Use room only for large "
        "standalone objects or when no more specific visible support is clear."
    )
    assert enriched["request_bundle_digest"] == lab.vlm_answer_option_request_bundle_digest(
        enriched
    )
    serialized = lab.vlm_answer_option_request_bundle_json(enriched)
    assert "gold_answer" not in serialized
    assert "gold_evidence" not in serialized
    assert "countertop_1" not in serialized
    assert "sink_1" not in serialized


def test_vlm_answer_option_request_bundle_can_add_ambiguous_support_relations_without_gold() -> None:
    bundle: dict[str, Any] = {
        "case_inputs": [
            {
                "case_id": "case-cloth",
                "question_type": "object_location",
                "target": {"label": "cloth", "object_id": "cloth_1"},
                "support_candidates": [
                    {
                        "label": "bathtub",
                        "object_id": "bathtub_1",
                        "relation_hint": "INSIDE",
                    }
                ],
            }
        ],
        "request_bundle_digest": "old",
    }

    enriched = lab.vlm_answer_option_request_bundle(
        bundle,
        include_ambiguous_support_relations=True,
    )

    assert enriched["case_inputs"][0]["answer_options"] == [
        {
            "destination_label": "bathtub",
            "option_id": "locopt_001",
            "relation": "INSIDE",
            "source": "support_candidate",
        },
        {
            "destination_label": "bathtub",
            "option_id": "locopt_002",
            "relation": "ON",
            "source": "ambiguous_support_relation",
        },
        {
            "destination_label": "room",
            "option_id": "locopt_003",
            "relation": "IN_ROOM",
            "source": "fallback_room",
        },
    ]
    serialized = lab.vlm_answer_option_request_bundle_json(enriched)
    assert "bathtub_1" not in serialized
    assert "gold_answer" not in serialized


def test_vlm_answer_option_request_bundle_can_add_target_affordance_options_without_gold() -> None:
    bundle: dict[str, Any] = {
        "case_inputs": [
            {
                "case_id": "case-handtowel",
                "question_type": "object_location",
                "target": {"label": "handtowel", "object_id": "handtowel_1"},
            }
        ],
        "request_bundle_digest": "old",
    }

    enriched = lab.vlm_answer_option_request_bundle(
        bundle,
        include_target_affordance_options=True,
    )

    assert enriched["case_inputs"][0]["answer_options"] == [
        {
            "destination_label": "handtowelholder",
            "option_id": "locopt_001",
            "relation": "ON",
            "source": "target_affordance_prior",
        },
        {
            "destination_label": "room",
            "option_id": "locopt_002",
            "relation": "IN_ROOM",
            "source": "fallback_room",
        },
    ]
    serialized = lab.vlm_answer_option_request_bundle_json(enriched)
    assert "gold_answer" not in serialized
    assert "gold_evidence" not in serialized


def test_vlm_retry_request_bundle_filters_failed_cases_without_evaluator_leak() -> None:
    bundle: dict[str, Any] = {
        "action": "offline_control_prediction_request_bundle",
        "case_count": 2,
        "case_inputs": [
            {
                "case_id": "case-success",
                "question_text": "Where is the apple?",
                "target": {"label": "apple"},
                "answer_options": [{"option_id": "locopt_001"}],
            },
            {
                "case_id": "case-failed",
                "question_text": "Where is the bottle?",
                "target": {"label": "bottle"},
                "answer_options": [{"option_id": "locopt_001"}],
                "gold_answer": "bottle ON shelf",
                "failure_reason": "do not leak",
            },
        ],
        "prediction_templates": {
            "qa_prediction": [
                {"id": "case-success", "answer": {}, "evidence_nodes": []},
                {"id": "case-failed", "answer": {}, "evidence_nodes": []},
            ]
        },
        "request_bundle_digest": "old",
    }
    semantic_report: dict[str, Any] = {
        "report_digest": "a" * 64,
        "summary": {"case_count": 2},
        "cases": [
            {
                "case_id": "case-success",
                "semantic_match": True,
                "gold": {"relation": "ON", "destination_label": "countertop"},
                "prediction": {"raw_text": "on the countertop"},
            },
            {
                "case_id": "case-failed",
                "semantic_match": False,
                "failure_reason": "destination_mismatch",
                "gold": {"relation": "ON", "destination_label": "shelf"},
                "prediction": {"raw_text": "on the table"},
            },
        ],
    }

    retry = lab.vlm_retry_request_bundle(bundle, semantic_report)

    assert retry["case_count"] == 1
    assert [case["case_id"] for case in retry["case_inputs"]] == ["case-failed"]
    assert [row["id"] for row in retry["prediction_templates"]["qa_prediction"]] == [
        "case-failed"
    ]
    assert retry["vlm_retry_enrichment"]["summary"] == {
        "case_count": 2,
        "retry_case_count": 1,
        "skipped_success_count": 1,
    }
    serialized = lab.vlm_retry_request_bundle_json(retry)
    assert "gold_answer" not in serialized
    assert "gold_evidence" not in serialized
    assert '"gold"' not in serialized
    assert "destination_mismatch" not in serialized
    assert "semantic_match" not in serialized
    assert "on the table" not in serialized
    assert retry["request_bundle_digest"] == lab.vlm_retry_request_bundle_digest(
        retry
    )


def test_vlm_retry_merge_replaces_failed_predictions_and_reports_readiness() -> None:
    original_predictions = [
        lab.QAPrediction(
            id="case-success",
            answer={"text": "on the countertop"},
            confidence=0.8,
            error=None,
        ),
        lab.QAPrediction(
            id="case-failed",
            answer={"text": "on the table"},
            confidence=0.4,
            error=None,
        ),
    ]
    retry_predictions = [
        lab.QAPrediction(
            id="case-failed",
            answer={
                "current_location": {
                    "destination_label": "shelf",
                    "relation": "ON",
                }
            },
            confidence=0.9,
            error=None,
        )
    ]
    semantic_report: dict[str, Any] = {
        "report_digest": "a" * 64,
        "cases": [
            {
                "case_id": "case-success",
                "semantic_match": True,
                "gold": {"relation": "ON", "destination_label": "countertop"},
                "prediction": {"raw_text": "on the countertop"},
            },
            {
                "case_id": "case-failed",
                "semantic_match": False,
                "failure_reason": "destination_mismatch",
                "gold": {"relation": "ON", "destination_label": "shelf"},
                "prediction": {"raw_text": "on the table"},
            },
        ],
    }

    merged = lab.vlm_retry_merged_predictions(
        original_predictions,
        retry_predictions,
        semantic_report,
    )
    report = lab.vlm_retry_merge_report(
        original_predictions,
        retry_predictions,
        merged,
        semantic_report,
    )

    assert [prediction.id for prediction in merged] == ["case-success", "case-failed"]
    assert merged[0].answer == {"text": "on the countertop"}
    assert merged[1].answer == {
        "current_location": {
            "destination_label": "shelf",
            "relation": "ON",
        }
    }
    assert report["ready"] is True
    assert report["summary"] == {
        "expected_case_count": 2,
        "kept_original_success_count": 1,
        "merged_prediction_count": 2,
        "missing_original_case_count": 0,
        "missing_retry_case_count": 0,
        "original_prediction_count": 2,
        "out_of_scope_failed_case_count": 0,
        "replaced_retry_case_count": 1,
        "retry_expected_case_count": 1,
        "retry_prediction_count": 1,
        "unexpected_retry_case_count": 0,
    }
    assert report["retry_case_ids"] == ["case-failed"]
    assert report["kept_original_success_case_ids"] == ["case-success"]
    assert report["report_digest"] == lab.vlm_retry_merge_report_digest(report)
    serialized = lab.vlm_retry_merge_report_json(report)
    assert "gold" not in serialized
    assert "destination_mismatch" not in serialized
    assert "on the table" not in serialized
    assert lab.validate_vlm_retry_merge_report(report)["valid"] is True


def test_vlm_retry_merge_report_marks_missing_retry_predictions_not_ready() -> None:
    original_predictions = [
        lab.QAPrediction(id="case-success", answer={"text": "ok"}),
        lab.QAPrediction(id="case-failed", answer={"text": "old"}),
    ]
    semantic_report: dict[str, Any] = {
        "report_digest": "b" * 64,
        "cases": [
            {"case_id": "case-success", "semantic_match": True},
            {"case_id": "case-failed", "semantic_match": False},
        ],
    }

    merged = lab.vlm_retry_merged_predictions(
        original_predictions,
        [],
        semantic_report,
    )
    report = lab.vlm_retry_merge_report(
        original_predictions,
        [],
        merged,
        semantic_report,
    )

    assert [prediction.answer for prediction in merged] == [
        {"text": "ok"},
        {"text": "old"},
    ]
    assert report["ready"] is False
    assert report["missing_retry_case_ids"] == ["case-failed"]
    assert report["summary"]["missing_retry_case_count"] == 1
    assert lab.validate_vlm_retry_merge_report(report)["valid"] is True


def test_vlm_retry_merge_can_use_explicit_retry_scope_for_crop_ready_subset() -> None:
    original_predictions = [
        lab.QAPrediction(id="case-success", answer={"text": "ok"}),
        lab.QAPrediction(id="case-ready-retry", answer={"text": "old"}),
        lab.QAPrediction(id="case-blocked-crop", answer={"text": "blocked old"}),
    ]
    retry_predictions = [
        lab.QAPrediction(id="case-ready-retry", answer={"text": "new"})
    ]
    semantic_report: dict[str, Any] = {
        "report_digest": "c" * 64,
        "cases": [
            {"case_id": "case-success", "semantic_match": True},
            {"case_id": "case-ready-retry", "semantic_match": False},
            {"case_id": "case-blocked-crop", "semantic_match": False},
        ],
    }

    merged = lab.vlm_retry_merged_predictions(
        original_predictions,
        retry_predictions,
        semantic_report,
        retry_case_ids=["case-ready-retry"],
    )
    report = lab.vlm_retry_merge_report(
        original_predictions,
        retry_predictions,
        merged,
        semantic_report,
        retry_case_ids=["case-ready-retry"],
    )

    assert [prediction.answer for prediction in merged] == [
        {"text": "ok"},
        {"text": "new"},
        {"text": "blocked old"},
    ]
    assert report["ready"] is True
    assert report["retry_scope_kind"] == "explicit_retry_case_ids"
    assert report["retry_case_ids"] == ["case-ready-retry"]
    assert report["out_of_scope_failed_case_ids"] == ["case-blocked-crop"]
    assert report["summary"]["retry_expected_case_count"] == 1
    assert report["summary"]["out_of_scope_failed_case_count"] == 1
    assert report["summary"]["missing_retry_case_count"] == 0
    assert lab.validate_vlm_retry_merge_report(report)["valid"] is True


def test_vlm_retry_input_gap_report_lists_missing_visual_inputs_without_gold() -> None:
    bundle: dict[str, Any] = {
        "request_bundle_digest": "old",
        "case_inputs": [
            {
                "case_id": "case-ready",
                "episode_id": "episode-001",
                "scene_id": "FloorPlan1",
                "question_text": "Where is the apple?",
                "target": {"label": "apple", "object_id": "apple_1"},
                "primary_frame": {"step": 2},
                "frames": [{"step": 2}],
                "target_crop": {"rgb_path": "crop.ppm"},
                "support_candidates": [{"label": "countertop"}],
                "answer_options": [{"option_id": "locopt_001"}],
            },
            {
                "case_id": "case-gap",
                "episode_id": "episode-001",
                "scene_id": "FloorPlan1",
                "question_text": "Where is the bottle?",
                "target": {"label": "bottle", "object_id": "bottle_1"},
                "primary_frame": {"step": 3},
                "frames": [{"step": 3}],
                "answer_options": [{"option_id": "locopt_001"}],
                "gold_answer": "bottle ON shelf",
            },
        ],
    }

    report = lab.vlm_retry_input_gap_report(bundle)

    assert report["summary"] == {
        "case_count": 2,
        "cases_with_answer_options": 2,
        "cases_with_frames": 2,
        "cases_with_primary_frame": 2,
        "cases_with_support_candidates": 1,
        "support_candidates_required_count": 2,
        "support_candidates_not_applicable_count": 0,
        "cases_with_target_crop": 1,
        "missing_answer_options_count": 0,
        "missing_frames_count": 0,
        "missing_primary_frame_count": 0,
        "missing_support_candidates_count": 1,
        "missing_target_crop_count": 1,
    }
    assert report["ready"] is False
    assert report["next_collection_targets"] == [
        {
            "case_id": "case-gap",
            "episode_id": "episode-001",
            "missing_input_kinds": ["support_candidates", "target_crop"],
            "primary_frame_step": 3,
            "scene_id": "FloorPlan1",
            "target_label": "bottle",
            "target_object_id": "bottle_1",
        }
    ]
    assert report["report_digest"] == lab.vlm_retry_input_gap_report_digest(report)
    serialized = lab.vlm_retry_input_gap_report_json(report)
    assert "gold_answer" not in serialized
    assert "gold_evidence" not in serialized
    assert lab.validate_vlm_retry_input_gap_report(report)["valid"] is True


def test_vlm_retry_input_gap_report_treats_room_only_support_as_not_applicable() -> None:
    bundle: dict[str, Any] = {
        "request_bundle_digest": "old",
        "case_inputs": [
            {
                "case_id": "case-room-only",
                "episode_id": "episode-001",
                "scene_id": "FloorPlan1",
                "target": {"label": "armchair", "object_id": "armchair_1"},
                "primary_frame": {"step": 3},
                "frames": [{"step": 3}],
                "target_crop": {"rgb_path": "crop.ppm"},
                "answer_options": [
                    {
                        "destination_label": "room",
                        "option_id": "locopt_001",
                        "relation": "IN_ROOM",
                        "source": "fallback_room",
                    }
                ],
            },
            {
                "case_id": "case-support-required",
                "episode_id": "episode-001",
                "scene_id": "FloorPlan1",
                "target": {"label": "mug", "object_id": "mug_1"},
                "primary_frame": {"step": 4},
                "frames": [{"step": 4}],
                "target_crop": {"rgb_path": "crop.ppm"},
                "answer_options": [
                    {
                        "destination_label": "countertop",
                        "option_id": "locopt_001",
                        "relation": "ON",
                        "source": "support_candidate",
                    }
                ],
            },
        ],
    }

    report = lab.vlm_retry_input_gap_report(bundle)

    assert report["summary"]["missing_support_candidates_count"] == 1
    assert report["summary"]["support_candidates_not_applicable_count"] == 1
    assert report["cases"][0]["support_candidates_required"] is False
    assert report["cases"][0]["missing_input_kinds"] == []
    assert report["cases"][1]["support_candidates_required"] is True
    assert report["cases"][1]["missing_input_kinds"] == ["support_candidates"]
    assert report["next_collection_targets"] == [
        {
            "case_id": "case-support-required",
            "episode_id": "episode-001",
            "missing_input_kinds": ["support_candidates"],
            "primary_frame_step": 4,
            "scene_id": "FloorPlan1",
            "target_label": "mug",
            "target_object_id": "mug_1",
        }
    ]
    assert lab.validate_vlm_retry_input_gap_report(report)["valid"] is True


def test_vlm_calibration_cli_writes_retry_input_gap_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    request_bundle_path = tmp_path / "request-bundle.json"
    report_path = tmp_path / "input-gap-report.json"
    request_bundle_path.write_text(
        json.dumps(
            {
                "request_bundle_digest": "old",
                "case_inputs": [
                    {
                        "case_id": "case-gap",
                        "episode_id": "episode-001",
                        "scene_id": "FloorPlan1",
                        "target": {"label": "bottle", "object_id": "bottle_1"},
                        "primary_frame": {"step": 3},
                        "frames": [{"step": 3}],
                        "answer_options": [{"option_id": "locopt_001"}],
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    module = load_vlm_calibration_script()
    main = cast(MainFn, getattr(module, "main"))

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle_path),
            "--retry-input-gap-report",
            str(report_path),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert '"action": "build_vlm_retry_input_gap_report"' in output
    report = lab.load_vlm_retry_input_gap_report(report_path)
    assert report["ready"] is False
    assert report["summary"]["missing_target_crop_count"] == 1
    assert report["summary"]["missing_support_candidates_count"] == 1
    assert lab.validate_vlm_retry_input_gap_report(report)["valid"] is True


def test_vlm_answer_option_fallback_fills_single_room_option_without_gold() -> None:
    predictions = [
        lab.QAPrediction(
            id="case-room-only",
            answer={"source": "vlm", "text": "I cannot determine the location."},
            evidence_nodes=("frame-001",),
            confidence=0.42,
            error="target_not_observed",
        )
    ]
    request_bundle: dict[str, Any] = {
        "request_bundle_digest": "old",
        "case_inputs": [
            {
                "case_id": "case-room-only",
                "primary_frame": {"step": 12},
                "answer_options": [
                    {
                        "destination_label": "room",
                        "option_id": "locopt_001",
                        "relation": "IN_ROOM",
                        "source": "fallback_room",
                    }
                ],
                "gold_answer": "hidden evaluator answer",
            }
        ],
    }

    calibrated, report = lab.vlm_answer_option_fallback_predictions(
        predictions,
        request_bundle,
        prediction_path="predictions.jsonl",
        request_bundle_path="request-bundle.json",
    )

    assert calibrated[0].id == "case-room-only"
    assert calibrated[0].confidence == 0.42
    assert calibrated[0].error is None
    assert calibrated[0].evidence_nodes == ("frame-001",)
    assert calibrated[0].answer["answer_option_id"] == "locopt_001"
    assert calibrated[0].answer["current_location"] == {
        "dst": "room",
        "dst_label": "room",
        "relation": "IN_ROOM",
        "step": 12,
    }
    assert calibrated[0].answer["vlm_answer_option_fallback"] == {
        "case_id": "case-room-only",
        "source": "single_room_answer_option",
    }
    assert report["summary"] == {
        "applied_fallback_count": 1,
        "case_input_count": 1,
        "input_prediction_count": 1,
        "missing_case_input_count": 0,
        "output_prediction_count": 1,
        "skipped_existing_location_count": 0,
        "skipped_no_single_room_option_count": 0,
    }
    assert report["applied_case_ids"] == ["case-room-only"]
    assert report["report_digest"] == lab.vlm_answer_option_fallback_report_digest(
        report
    )
    assert lab.validate_vlm_answer_option_fallback_report(report)["valid"] is True
    serialized_predictions = lab.qa_predictions_jsonl(calibrated)
    serialized_report = lab.vlm_answer_option_fallback_report_json(report)
    assert "gold_answer" not in serialized_predictions
    assert "gold_answer" not in serialized_report
    assert "hidden evaluator answer" not in serialized_report


def test_vlm_answer_option_fallback_skips_existing_location_and_ambiguous_options() -> None:
    predictions = [
        lab.QAPrediction(
            id="case-existing",
            answer={"current_location": {"relation": "ON", "dst_label": "countertop"}},
        ),
        lab.QAPrediction(id="case-multiple", answer={"text": "not sure"}),
        lab.QAPrediction(id="case-missing", answer={"text": "not sure"}),
    ]
    request_bundle: dict[str, Any] = {
        "case_inputs": [
            {
                "case_id": "case-existing",
                "primary_frame": {"step": 1},
                "answer_options": [
                    {
                        "destination_label": "room",
                        "option_id": "locopt_001",
                        "relation": "IN_ROOM",
                        "source": "fallback_room",
                    }
                ],
            },
            {
                "case_id": "case-multiple",
                "primary_frame": {"step": 2},
                "answer_options": [
                    {
                        "destination_label": "room",
                        "option_id": "locopt_001",
                        "relation": "IN_ROOM",
                        "source": "fallback_room",
                    },
                    {
                        "destination_label": "countertop",
                        "option_id": "locopt_002",
                        "relation": "ON",
                        "source": "support_candidate",
                    },
                ],
            },
        ],
    }

    calibrated, report = lab.vlm_answer_option_fallback_predictions(
        predictions,
        request_bundle,
    )

    assert [prediction.answer for prediction in calibrated] == [
        {"current_location": {"relation": "ON", "dst_label": "countertop"}},
        {"text": "not sure"},
        {"text": "not sure"},
    ]
    assert report["applied_case_ids"] == []
    assert report["missing_case_input_ids"] == ["case-missing"]
    assert report["skipped_existing_location_case_ids"] == ["case-existing"]
    assert report["skipped_no_single_room_option_case_ids"] == ["case-multiple"]
    assert report["summary"] == {
        "applied_fallback_count": 0,
        "case_input_count": 2,
        "input_prediction_count": 3,
        "missing_case_input_count": 1,
        "output_prediction_count": 3,
        "skipped_existing_location_count": 1,
        "skipped_no_single_room_option_count": 1,
    }
    assert lab.validate_vlm_answer_option_fallback_report(report)["valid"] is True


def test_vlm_room_level_target_fallback_fills_room_option_for_room_fixtures_without_gold() -> None:
    predictions = [
        lab.QAPrediction(
            id="case-cabinet",
            answer={
                "current_location": {
                    "dst_label": "currentframe",
                    "relation": "INSIDE",
                },
                "reasoning_summary": "The cabinet is not clearly visible.",
            },
            confidence=0.21,
            error="target_not_observed",
        )
    ]
    request_bundle: dict[str, Any] = {
        "request_bundle_digest": "old",
        "case_inputs": [
            {
                "case_id": "case-cabinet",
                "target": {"label": "cabinet", "object_id": "cabinet_001"},
                "primary_frame": {"step": 40},
                "answer_options": [
                    {
                        "destination_label": "microwave",
                        "option_id": "locopt_001",
                        "relation": "INSIDE",
                        "source": "support_candidate",
                    },
                    {
                        "destination_label": "room",
                        "option_id": "locopt_002",
                        "relation": "IN_ROOM",
                        "source": "fallback_room",
                    },
                ],
                "gold_answer": "hidden evaluator answer",
            }
        ],
    }

    calibrated, report = lab.vlm_room_level_target_fallback_predictions(
        predictions,
        request_bundle,
        prediction_path="predictions.jsonl",
        request_bundle_path="request-bundle.json",
    )

    assert calibrated[0].answer["answer_option_id"] == "locopt_002"
    assert calibrated[0].answer["current_location"] == {
        "dst": "room",
        "dst_label": "room",
        "relation": "IN_ROOM",
        "step": 40,
    }
    assert calibrated[0].answer["vlm_room_level_target_fallback"] == {
        "case_id": "case-cabinet",
        "source": "room_level_target_prior",
        "target_label": "cabinet",
    }
    assert calibrated[0].error is None
    assert report["summary"] == {
        "applied_fallback_count": 1,
        "case_input_count": 1,
        "input_prediction_count": 1,
        "missing_case_input_count": 0,
        "output_prediction_count": 1,
        "skipped_existing_location_count": 0,
        "skipped_ineligible_error_count": 0,
        "skipped_no_room_option_count": 0,
        "skipped_non_room_level_target_count": 0,
    }
    assert report["applied_case_ids"] == ["case-cabinet"]
    assert report["report_digest"] == lab.vlm_room_level_target_fallback_report_digest(
        report
    )
    assert lab.validate_vlm_room_level_target_fallback_report(report)["valid"] is True
    serialized = lab.vlm_room_level_target_fallback_report_json(report)
    assert "gold_answer" not in serialized
    assert "hidden evaluator answer" not in serialized


def test_vlm_room_level_target_fallback_skips_small_targets_and_existing_locations() -> None:
    predictions = [
        lab.QAPrediction(
            id="case-creditcard",
            answer={"text": "not visible"},
            error="target_not_observed",
        ),
        lab.QAPrediction(
            id="case-chair",
            answer={"current_location": {"dst_label": "floor", "relation": "ON"}},
            error=None,
        ),
    ]
    request_bundle: dict[str, Any] = {
        "case_inputs": [
            {
                "case_id": "case-creditcard",
                "target": {"label": "creditcard"},
                "primary_frame": {"step": 10},
                "answer_options": [
                    {
                        "destination_label": "countertop",
                        "option_id": "locopt_001",
                        "relation": "ON",
                        "source": "support_candidate",
                    },
                    {
                        "destination_label": "room",
                        "option_id": "locopt_002",
                        "relation": "IN_ROOM",
                        "source": "fallback_room",
                    },
                ],
            },
            {
                "case_id": "case-chair",
                "target": {"label": "chair"},
                "primary_frame": {"step": 11},
                "answer_options": [
                    {
                        "destination_label": "room",
                        "option_id": "locopt_001",
                        "relation": "IN_ROOM",
                        "source": "fallback_room",
                    }
                ],
            },
        ],
    }

    calibrated, report = lab.vlm_room_level_target_fallback_predictions(
        predictions,
        request_bundle,
    )

    assert [prediction.answer for prediction in calibrated] == [
        {"text": "not visible"},
        {"current_location": {"dst_label": "floor", "relation": "ON"}},
    ]
    assert report["applied_case_ids"] == []
    assert report["skipped_non_room_level_target_case_ids"] == ["case-creditcard"]
    assert report["skipped_existing_location_case_ids"] == ["case-chair"]
    assert report["summary"] == {
        "applied_fallback_count": 0,
        "case_input_count": 2,
        "input_prediction_count": 2,
        "missing_case_input_count": 0,
        "output_prediction_count": 2,
        "skipped_existing_location_count": 1,
        "skipped_ineligible_error_count": 0,
        "skipped_no_room_option_count": 0,
        "skipped_non_room_level_target_count": 1,
    }
    assert lab.validate_vlm_room_level_target_fallback_report(report)["valid"] is True


def test_vlm_single_support_option_fallback_fills_only_support_candidate_without_gold() -> None:
    predictions = [
        lab.QAPrediction(
            id="case-sponge",
            answer={"text": "The sponge is not visible."},
            confidence=0.17,
            error="target_not_observed",
        )
    ]
    request_bundle: dict[str, Any] = {
        "request_bundle_digest": "old",
        "case_inputs": [
            {
                "case_id": "case-sponge",
                "target": {"label": "dishsponge"},
                "primary_frame": {"step": 44},
                "answer_options": [
                    {
                        "destination_label": "sidetable",
                        "option_id": "locopt_001",
                        "relation": "ON",
                        "source": "support_candidate",
                    },
                    {
                        "destination_label": "room",
                        "option_id": "locopt_002",
                        "relation": "IN_ROOM",
                        "source": "fallback_room",
                    },
                ],
                "gold_answer": "hidden evaluator answer",
            }
        ],
    }

    calibrated, report = lab.vlm_single_support_option_fallback_predictions(
        predictions,
        request_bundle,
        prediction_path="predictions.jsonl",
        request_bundle_path="request-bundle.json",
    )

    assert calibrated[0].answer["answer_option_id"] == "locopt_001"
    assert calibrated[0].answer["current_location"] == {
        "dst": "sidetable",
        "dst_label": "sidetable",
        "relation": "ON",
        "step": 44,
    }
    assert calibrated[0].answer["vlm_single_support_option_fallback"] == {
        "case_id": "case-sponge",
        "source": "single_support_candidate_option",
    }
    assert calibrated[0].error is None
    assert report["summary"] == {
        "applied_fallback_count": 1,
        "case_input_count": 1,
        "input_prediction_count": 1,
        "missing_case_input_count": 0,
        "output_prediction_count": 1,
        "skipped_existing_location_count": 0,
        "skipped_ineligible_error_count": 0,
        "skipped_no_single_support_option_count": 0,
        "skipped_room_level_target_count": 0,
    }
    assert report["applied_case_ids"] == ["case-sponge"]
    assert report["report_digest"] == lab.vlm_single_support_option_fallback_report_digest(
        report
    )
    assert lab.validate_vlm_single_support_option_fallback_report(report)["valid"] is True
    serialized = lab.vlm_single_support_option_fallback_report_json(report)
    assert "gold_answer" not in serialized
    assert "hidden evaluator answer" not in serialized


def test_vlm_single_support_option_fallback_skips_multiple_options_room_targets_and_existing_locations() -> None:
    predictions = [
        lab.QAPrediction(
            id="case-creditcard",
            answer={"text": "not visible"},
            error="target_not_observed",
        ),
        lab.QAPrediction(
            id="case-cabinet",
            answer={"text": "not visible"},
            error="target_not_observed",
        ),
        lab.QAPrediction(
            id="case-bowl",
            answer={"current_location": {"dst_label": "countertop", "relation": "ON"}},
            error=None,
        ),
        lab.QAPrediction(
            id="case-cloth",
            answer={"text": "not visible"},
            error="target_not_observed",
        ),
    ]
    request_bundle: dict[str, Any] = {
        "case_inputs": [
            {
                "case_id": "case-creditcard",
                "target": {"label": "creditcard"},
                "answer_options": [
                    {
                        "destination_label": "countertop",
                        "option_id": "locopt_001",
                        "relation": "ON",
                        "source": "support_candidate",
                    },
                    {
                        "destination_label": "floor",
                        "option_id": "locopt_002",
                        "relation": "ON",
                        "source": "support_candidate",
                    },
                    {
                        "destination_label": "room",
                        "option_id": "locopt_003",
                        "relation": "IN_ROOM",
                        "source": "fallback_room",
                    },
                ],
            },
            {
                "case_id": "case-cabinet",
                "target": {"label": "cabinet"},
                "answer_options": [
                    {
                        "destination_label": "microwave",
                        "option_id": "locopt_001",
                        "relation": "INSIDE",
                        "source": "support_candidate",
                    },
                    {
                        "destination_label": "room",
                        "option_id": "locopt_002",
                        "relation": "IN_ROOM",
                        "source": "fallback_room",
                    },
                ],
            },
            {
                "case_id": "case-bowl",
                "target": {"label": "bowl"},
                "answer_options": [
                    {
                        "destination_label": "desk",
                        "option_id": "locopt_001",
                        "relation": "ON",
                        "source": "support_candidate",
                    },
                    {
                        "destination_label": "room",
                        "option_id": "locopt_002",
                        "relation": "IN_ROOM",
                        "source": "fallback_room",
                    },
                ],
            },
            {
                "case_id": "case-cloth",
                "target": {"label": "cloth"},
                "answer_options": [
                    {
                        "destination_label": "bathtub",
                        "option_id": "locopt_001",
                        "relation": "INSIDE",
                        "source": "support_candidate",
                    },
                    {
                        "destination_label": "bathtub",
                        "option_id": "locopt_002",
                        "relation": "ON",
                        "source": "ambiguous_support_relation",
                    },
                    {
                        "destination_label": "room",
                        "option_id": "locopt_003",
                        "relation": "IN_ROOM",
                        "source": "fallback_room",
                    },
                ],
            },
        ],
    }

    calibrated, report = lab.vlm_single_support_option_fallback_predictions(
        predictions,
        request_bundle,
    )

    assert [prediction.answer for prediction in calibrated] == [
        {"text": "not visible"},
        {"text": "not visible"},
        {"current_location": {"dst_label": "countertop", "relation": "ON"}},
        {"text": "not visible"},
    ]
    assert report["applied_case_ids"] == []
    assert report["skipped_no_single_support_option_case_ids"] == [
        "case-creditcard",
        "case-cloth",
    ]
    assert report["skipped_room_level_target_case_ids"] == ["case-cabinet"]
    assert report["skipped_existing_location_case_ids"] == ["case-bowl"]
    assert report["summary"] == {
        "applied_fallback_count": 0,
        "case_input_count": 4,
        "input_prediction_count": 4,
        "missing_case_input_count": 0,
        "output_prediction_count": 4,
        "skipped_existing_location_count": 1,
        "skipped_ineligible_error_count": 0,
        "skipped_no_single_support_option_count": 2,
        "skipped_room_level_target_count": 1,
    }
    assert lab.validate_vlm_single_support_option_fallback_report(report)["valid"] is True


def test_vlm_text_option_alignment_uses_unique_option_label_mentioned_in_reasoning_without_gold() -> None:
    predictions = [
        lab.QAPrediction(
            id="case-bottle",
            answer={
                "current_location": {
                    "dst_label": "sidetable",
                    "relation": "ON",
                    "step": 37,
                },
                "reasoning_summary": (
                    "The bottle is on a dark structure and sitting on one of its shelves."
                ),
            },
            confidence=0.64,
            error=None,
        )
    ]
    request_bundle: dict[str, Any] = {
        "request_bundle_digest": "old",
        "case_inputs": [
            {
                "case_id": "case-bottle",
                "primary_frame": {"step": 37},
                "answer_options": [
                    {
                        "destination_label": "countertop",
                        "option_id": "locopt_001",
                        "relation": "ON",
                        "source": "support_candidate",
                    },
                    {
                        "destination_label": "shelf",
                        "option_id": "locopt_002",
                        "relation": "ON",
                        "source": "support_candidate",
                    },
                    {
                        "destination_label": "room",
                        "option_id": "locopt_003",
                        "relation": "IN_ROOM",
                        "source": "fallback_room",
                    },
                ],
                "gold_answer": "hidden evaluator answer",
            }
        ],
    }

    calibrated, report = lab.vlm_text_option_alignment_predictions(
        predictions,
        request_bundle,
        prediction_path="predictions.jsonl",
        request_bundle_path="request-bundle.json",
    )

    assert calibrated[0].answer["answer_option_id"] == "locopt_002"
    assert calibrated[0].answer["current_location"] == {
        "dst": "shelf",
        "dst_label": "shelf",
        "relation": "ON",
        "step": 37,
    }
    assert calibrated[0].answer["vlm_text_option_alignment"] == {
        "case_id": "case-bottle",
        "matched_destination_label": "shelf",
        "source": "unique_text_mentioned_answer_option",
    }
    assert calibrated[0].error is None
    assert report["summary"] == {
        "aligned_prediction_count": 1,
        "case_input_count": 1,
        "input_prediction_count": 1,
        "missing_case_input_count": 0,
        "output_prediction_count": 1,
        "skipped_ambiguous_text_match_count": 0,
        "skipped_existing_option_location_count": 0,
        "skipped_ineligible_error_count": 0,
        "skipped_missing_location_count": 0,
        "skipped_no_text_match_count": 0,
    }
    assert report["aligned_case_ids"] == ["case-bottle"]
    assert report["report_digest"] == lab.vlm_text_option_alignment_report_digest(report)
    assert lab.validate_vlm_text_option_alignment_report(report)["valid"] is True
    serialized = lab.vlm_text_option_alignment_report_json(report)
    assert "gold_answer" not in serialized
    assert "hidden evaluator answer" not in serialized


def test_vlm_text_option_alignment_skips_generic_table_and_existing_option_locations() -> None:
    predictions = [
        lab.QAPrediction(
            id="case-book",
            answer={
                "current_location": {"dst_label": "table", "relation": "ON"},
                "reasoning_summary": "The book is on the table.",
            },
            error=None,
        ),
        lab.QAPrediction(
            id="case-apple",
            answer={
                "current_location": {"dst_label": "countertop", "relation": "ON"},
                "reasoning_summary": "The apple is on the countertop.",
            },
            error=None,
        ),
        lab.QAPrediction(
            id="case-missing",
            answer={"text": "The target is not visible."},
            error="target_not_observed",
        ),
    ]
    request_bundle: dict[str, Any] = {
        "case_inputs": [
            {
                "case_id": "case-book",
                "answer_options": [
                    {
                        "destination_label": "chair",
                        "option_id": "locopt_001",
                        "relation": "ON",
                        "source": "support_candidate",
                    },
                    {
                        "destination_label": "diningtable",
                        "option_id": "locopt_002",
                        "relation": "ON",
                        "source": "support_candidate",
                    },
                ],
            },
            {
                "case_id": "case-apple",
                "answer_options": [
                    {
                        "destination_label": "countertop",
                        "option_id": "locopt_001",
                        "relation": "ON",
                        "source": "support_candidate",
                    }
                ],
            },
            {
                "case_id": "case-missing",
                "answer_options": [
                    {
                        "destination_label": "desk",
                        "option_id": "locopt_001",
                        "relation": "ON",
                        "source": "support_candidate",
                    }
                ],
            },
        ],
    }

    calibrated, report = lab.vlm_text_option_alignment_predictions(
        predictions,
        request_bundle,
    )

    assert [prediction.answer for prediction in calibrated] == [
        {
            "current_location": {"dst_label": "table", "relation": "ON"},
            "reasoning_summary": "The book is on the table.",
        },
        {
            "current_location": {"dst_label": "countertop", "relation": "ON"},
            "reasoning_summary": "The apple is on the countertop.",
        },
        {"text": "The target is not visible."},
    ]
    assert report["aligned_case_ids"] == []
    assert report["skipped_no_text_match_case_ids"] == ["case-book"]
    assert report["skipped_existing_option_location_case_ids"] == ["case-apple"]
    assert report["skipped_ineligible_error_case_ids"] == ["case-missing"]
    assert report["summary"] == {
        "aligned_prediction_count": 0,
        "case_input_count": 3,
        "input_prediction_count": 3,
        "missing_case_input_count": 0,
        "output_prediction_count": 3,
        "skipped_ambiguous_text_match_count": 0,
        "skipped_existing_option_location_count": 1,
        "skipped_ineligible_error_count": 1,
        "skipped_missing_location_count": 0,
        "skipped_no_text_match_count": 1,
    }
    assert lab.validate_vlm_text_option_alignment_report(report)["valid"] is True


def test_vlm_affordance_option_fallback_uses_public_target_prior_without_gold() -> None:
    predictions = [
        lab.QAPrediction(
            id="case-pen",
            answer={"text": "The pen is not visible in the crop."},
            confidence=0.31,
            error="target_not_observed",
        )
    ]
    request_bundle: dict[str, Any] = {
        "request_bundle_digest": "old",
        "case_inputs": [
            {
                "case_id": "case-pen",
                "target": {"label": "pen", "object_id": "pen_001"},
                "primary_frame": {"step": 29},
                "answer_options": [
                    {
                        "destination_label": "desk",
                        "option_id": "locopt_001",
                        "relation": "ON",
                        "source": "support_candidate",
                    },
                    {
                        "destination_label": "shelf",
                        "option_id": "locopt_002",
                        "relation": "ON",
                        "source": "support_candidate",
                    },
                    {
                        "destination_label": "room",
                        "option_id": "locopt_003",
                        "relation": "IN_ROOM",
                        "source": "fallback_room",
                    },
                ],
                "gold_answer": "hidden evaluator answer",
            }
        ],
    }

    calibrated, report = lab.vlm_affordance_option_fallback_predictions(
        predictions,
        request_bundle,
        prediction_path="predictions.jsonl",
        request_bundle_path="request-bundle.json",
    )

    assert calibrated[0].answer["answer_option_id"] == "locopt_001"
    assert calibrated[0].answer["current_location"] == {
        "dst": "desk",
        "dst_label": "desk",
        "relation": "ON",
        "step": 29,
    }
    assert calibrated[0].answer["vlm_affordance_option_fallback"] == {
        "case_id": "case-pen",
        "destination_label": "desk",
        "source": "target_affordance_public_option_prior",
        "target_label": "pen",
    }
    assert calibrated[0].error is None
    assert report["summary"] == {
        "applied_fallback_count": 1,
        "case_input_count": 1,
        "input_prediction_count": 1,
        "missing_case_input_count": 0,
        "output_prediction_count": 1,
        "skipped_ambiguous_affordance_option_count": 0,
        "skipped_existing_location_count": 0,
        "skipped_ineligible_error_count": 0,
        "skipped_no_affordance_option_count": 0,
    }
    assert report["applied_case_ids"] == ["case-pen"]
    assert report["report_digest"] == lab.vlm_affordance_option_fallback_report_digest(
        report
    )
    assert lab.validate_vlm_affordance_option_fallback_report(report)["valid"] is True
    serialized = lab.vlm_affordance_option_fallback_report_json(report)
    assert "gold_answer" not in serialized
    assert "hidden evaluator answer" not in serialized


def test_vlm_affordance_option_fallback_skips_ambiguous_existing_and_non_error_predictions() -> None:
    predictions = [
        lab.QAPrediction(
            id="case-bowl-ambiguous",
            answer={"text": "not visible"},
            error="target_not_observed",
        ),
        lab.QAPrediction(
            id="case-bowl-existing",
            answer={"current_location": {"dst_label": "countertop", "relation": "ON"}},
            error="target_not_observed",
        ),
        lab.QAPrediction(
            id="case-bowl-no-error",
            answer={"text": "not visible"},
            error=None,
        ),
        lab.QAPrediction(
            id="case-missing",
            answer={"text": "not visible"},
            error="target_not_observed",
        ),
    ]
    request_bundle: dict[str, Any] = {
        "case_inputs": [
            {
                "case_id": "case-bowl-ambiguous",
                "target": {"label": "bowl"},
                "answer_options": [
                    {
                        "destination_label": "coffeetable",
                        "option_id": "locopt_001",
                        "relation": "ON",
                        "source": "support_candidate",
                    },
                    {
                        "destination_label": "desk",
                        "option_id": "locopt_002",
                        "relation": "ON",
                        "source": "support_candidate",
                    },
                ],
            },
            {
                "case_id": "case-bowl-existing",
                "target": {"label": "bowl"},
                "answer_options": [
                    {
                        "destination_label": "desk",
                        "option_id": "locopt_001",
                        "relation": "ON",
                        "source": "support_candidate",
                    }
                ],
            },
            {
                "case_id": "case-bowl-no-error",
                "target": {"label": "bowl"},
                "answer_options": [
                    {
                        "destination_label": "desk",
                        "option_id": "locopt_001",
                        "relation": "ON",
                        "source": "support_candidate",
                    }
                ],
            },
        ],
    }

    calibrated, report = lab.vlm_affordance_option_fallback_predictions(
        predictions,
        request_bundle,
    )

    assert [prediction.answer for prediction in calibrated] == [
        {"text": "not visible"},
        {"current_location": {"dst_label": "countertop", "relation": "ON"}},
        {"text": "not visible"},
        {"text": "not visible"},
    ]
    assert report["applied_case_ids"] == []
    assert report["skipped_ambiguous_affordance_option_case_ids"] == [
        "case-bowl-ambiguous"
    ]
    assert report["skipped_existing_location_case_ids"] == ["case-bowl-existing"]
    assert report["skipped_ineligible_error_case_ids"] == ["case-bowl-no-error"]
    assert report["missing_case_input_ids"] == ["case-missing"]
    assert report["summary"] == {
        "applied_fallback_count": 0,
        "case_input_count": 3,
        "input_prediction_count": 4,
        "missing_case_input_count": 1,
        "output_prediction_count": 4,
        "skipped_ambiguous_affordance_option_count": 1,
        "skipped_existing_location_count": 1,
        "skipped_ineligible_error_count": 1,
        "skipped_no_affordance_option_count": 0,
    }
    assert lab.validate_vlm_affordance_option_fallback_report(report)["valid"] is True


def test_vlm_answer_option_coverage_report_measures_gold_option_recall() -> None:
    cases = [
        _object_location_case(
            "case-apple",
            visible=True,
            relation="ON",
            dst="countertop_001",
        ),
        _object_location_case(
            "case-book",
            visible=True,
            relation="ON",
            dst="desk_001",
        ),
    ]
    bundle: dict[str, Any] = {
        "case_inputs": [
            {
                "case_id": "case-apple",
                "answer_options": [
                    {
                        "destination_label": "countertop",
                        "option_id": "locopt_001",
                        "relation": "ON",
                        "source": "support_candidate",
                    }
                ],
            },
            {
                "case_id": "case-book",
                "answer_options": [
                    {
                        "destination_label": "room",
                        "option_id": "locopt_001",
                        "relation": "IN_ROOM",
                        "source": "fallback_room",
                    }
                ],
            },
        ],
        "request_bundle_digest": "old",
    }

    report = lab.vlm_answer_option_coverage_report(cases, bundle)

    assert report["summary"] == {
        "case_count": 2,
        "covered_case_count": 1,
        "covered_case_rate": 0.5,
        "missing_case_count": 1,
        "option_count": 2,
    }
    assert report["missing_case_ids"] == ["case-book"]
    assert report["cases"][0]["covered"] is True
    assert report["cases"][0]["matched_option_source"] == "support_candidate"
    assert report["cases"][1]["covered"] is False
    assert report["cases"][1]["failure_reason"] == "gold_option_missing"
    assert report["report_digest"] == lab.vlm_answer_option_coverage_report_digest(
        report
    )
    assert lab.validate_vlm_answer_option_coverage_report(report)["valid"] is True


def test_vlm_calibration_cli_writes_support_candidate_request_bundle(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    request_bundle_path = tmp_path / "request-bundle.json"
    frame_index_path = tmp_path / "frame-index.jsonl"
    output_path = tmp_path / "request-bundle-support.json"
    request_bundle_path.write_text(
        """{
  "action": "offline_control_prediction_request_bundle",
  "case_count": 1,
  "case_inputs": [
    {
      "case_id": "case-apple",
      "episode_id": "episode-001",
      "scene_id": "FloorPlan1",
      "question_type": "object_location",
      "question_text": "Where is the apple?",
      "target": {"label": "apple", "object_id": "apple_1"},
      "primary_frame": {
        "episode_id": "episode-001",
        "scene_id": "FloorPlan1",
        "step": 2
      }
    }
  ],
  "request_bundle_digest": "old"
}
""",
        encoding="utf-8",
    )
    frame_index_path.write_text(
        (
            '{"episode_id":"episode-001","scene_id":"FloorPlan1","step":2,'
            '"visible_object_ids":["apple_1","countertop_1"],'
            '"visible_object_labels":["apple","countertop"]}\n'
        ),
        encoding="utf-8",
    )

    module = load_vlm_calibration_script()
    main = cast(MainFn, getattr(module, "main"))

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle_path),
            "--frame-index",
            str(frame_index_path),
            "--support-candidate-request-bundle-output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"action": "build_vlm_support_candidate_request_bundle"' in output
    bundle = json.loads(output_path.read_text(encoding="utf-8"))
    assert bundle["case_inputs"][0]["support_candidates"] == [
        {
            "label": "countertop",
            "relation_hint": "ON",
            "source": "primary_frame_visible_label",
        }
    ]


def test_vlm_calibration_cli_writes_answer_option_request_bundle(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    request_bundle_path = tmp_path / "request-bundle-support-crop.json"
    output_path = tmp_path / "request-bundle-options.json"
    request_bundle_path.write_text(
        """{
  "action": "offline_control_prediction_request_bundle",
  "case_count": 1,
  "case_inputs": [
    {
      "case_id": "case-apple",
      "question_type": "object_location",
      "question_text": "Where is the apple?",
      "target": {"label": "apple", "object_id": "apple_1"},
      "support_candidates": [
        {"label": "countertop", "object_id": "countertop_1", "relation_hint": "ON"}
      ]
    }
  ],
  "request_bundle_digest": "old"
}
""",
        encoding="utf-8",
    )
    module = load_vlm_calibration_script()
    main = cast(MainFn, getattr(module, "main"))

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle_path),
            "--answer-option-request-bundle-output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"action": "build_vlm_answer_option_request_bundle"' in output
    bundle = json.loads(output_path.read_text(encoding="utf-8"))
    assert bundle["case_inputs"][0]["answer_options"] == [
        {
            "destination_label": "countertop",
            "option_id": "locopt_001",
            "relation": "ON",
            "source": "support_candidate",
        },
        {
            "destination_label": "room",
            "option_id": "locopt_002",
            "relation": "IN_ROOM",
            "source": "fallback_room",
        },
    ]
    assert "countertop_1" not in json.dumps(bundle, sort_keys=True)


def test_vlm_support_detector_handoff_groups_support_labels_by_frame_without_ids() -> None:
    detector_handoff: dict[str, Any] = {
        "schema_version": "dsg-spatialqa-lab.independent-detector-rgbd-handoff.v1",
        "required_frames": [
            {
                "episode_id": "episode-001",
                "scene_id": "FloorPlan1",
                "frame_step": 2,
                "target_labels": ["apple"],
                "target_object_ids": ["apple_1"],
            }
        ],
    }
    request_bundle: dict[str, Any] = {
        "case_inputs": [
            {
                "case_id": "case-apple",
                "primary_frame": {
                    "episode_id": "episode-001",
                    "scene_id": "FloorPlan1",
                    "step": 2,
                },
                "support_candidates": [
                    {"label": "countertop", "relation_hint": "ON"},
                    {"label": "sink", "relation_hint": "INSIDE"},
                ],
            },
            {
                "case_id": "case-book",
                "primary_frame": {
                    "episode_id": "episode-001",
                    "scene_id": "FloorPlan1",
                    "step": 2,
                },
                "support_candidates": [
                    {"label": "countertop", "relation_hint": "ON"},
                    {"label": "cabinet", "relation_hint": "INSIDE"},
                ],
            },
        ],
        "request_bundle_digest": "a" * 64,
    }

    enriched = lab.vlm_support_detector_handoff(
        detector_handoff,
        request_bundle,
        max_support_labels_per_frame=3,
    )

    assert enriched["required_frames"][0]["support_labels"] == [
        "countertop",
        "sink",
        "cabinet",
    ]
    assert enriched["vlm_support_detector_handoff_enrichment"]["summary"] == {
        "frame_count": 1,
        "frames_with_support_labels": 1,
        "support_label_count": 3,
    }
    serialized = json.dumps(enriched, sort_keys=True)
    assert "countertop_1" not in serialized
    assert "sink_1" not in serialized
    assert "gold_answer" not in serialized
    assert "gold_evidence" not in serialized


def test_vlm_target_crop_request_bundle_writes_detector_crop_without_gold(
    tmp_path: Path,
) -> None:
    rgb_path = tmp_path / "frames" / "0002.ppm"
    rgb_path.parent.mkdir(parents=True)
    rgb_path.write_text(
        "P3\n4 3\n255\n"
        "0 0 0 10 0 0 20 0 0 30 0 0\n"
        "0 10 0 10 10 0 20 10 0 30 10 0\n"
        "0 20 0 10 20 0 20 20 0 30 20 0\n",
        encoding="utf-8",
    )
    request_bundle: dict[str, Any] = {
        "case_count": 1,
        "case_inputs": [
            {
                "case_id": "case-apple",
                "episode_id": "episode-001",
                "scene_id": "FloorPlan1",
                "target": {"label": "apple", "object_id": "apple_1"},
                "primary_frame": {
                    "episode_id": "episode-001",
                    "frame_id": "episode-001:FloorPlan1:0002",
                    "scene_id": "FloorPlan1",
                    "step": 2,
                },
            }
        ],
        "request_bundle_digest": "old",
    }
    detector_records = [
        {
            "detector_name": "qwen37_vlm_rgbd_detector",
            "episode_id": "episode-001",
            "rgb_path": str(rgb_path),
            "scene_id": "FloorPlan1",
            "step": 2,
            "detections": [
                {
                    "bbox_2d_xyxy": [1, 1, 2, 2],
                    "confidence": 0.91,
                    "label": "apple",
                    "object_id": "apple_1",
                    "visible": True,
                }
            ],
        }
    ]

    enriched = lab.vlm_target_crop_request_bundle(
        request_bundle,
        detector_records,
        crop_root=tmp_path / "crops",
        padding_pixels=0,
    )

    crop = enriched["case_inputs"][0]["target_crop"]
    assert crop == {
        "bbox_2d_xyxy": [1, 1, 2, 2],
        "confidence": 0.91,
        "detector_name": "qwen37_vlm_rgbd_detector",
        "rgb_path": str(tmp_path / "crops" / "episode-001" / "000002-case_apple.ppm"),
        "source": "detector_bbox_crop",
        "source_frame_id": "episode-001:FloorPlan1:0002",
    }
    assert Path(crop["rgb_path"]).read_text(encoding="utf-8") == (
        "P3\n2 2\n255\n"
        "10 10 0 20 10 0\n"
        "10 20 0 20 20 0\n"
    )
    assert enriched["vlm_target_crop_enrichment"]["summary"] == {
        "case_count": 1,
        "cases_with_target_crop": 1,
        "missing_target_detection_count": 0,
    }
    assert enriched["request_bundle_digest"] == lab.vlm_target_crop_request_bundle_digest(
        enriched
    )
    serialized = lab.vlm_target_crop_request_bundle_json(enriched)
    assert "gold_answer" not in serialized
    assert "gold_evidence" not in serialized


def test_vlm_target_crop_request_bundle_skips_fully_out_of_bounds_bbox(
    tmp_path: Path,
) -> None:
    rgb_path = tmp_path / "frames" / "0002.ppm"
    rgb_path.parent.mkdir(parents=True)
    rgb_path.write_text("P3\n2 1\n255\n10 0 0 20 0 0\n", encoding="utf-8")
    request_bundle: dict[str, Any] = {
        "case_count": 1,
        "case_inputs": [
            {
                "case_id": "case-apple",
                "target": {"label": "apple", "object_id": "apple_1"},
                "primary_frame": {
                    "episode_id": "episode-001",
                    "scene_id": "FloorPlan1",
                    "step": 2,
                },
            }
        ],
        "request_bundle_digest": "old",
    }
    detector_records = [
        {
            "detector_name": "qwen37_vlm_rgbd_detector",
            "episode_id": "episode-001",
            "rgb_path": str(rgb_path),
            "scene_id": "FloorPlan1",
            "step": 2,
            "detections": [
                {
                    "bbox_2d_xyxy": [4, 4, 5, 5],
                    "confidence": 0.91,
                    "label": "apple",
                    "object_id": "apple_1",
                    "visible": True,
                }
            ],
        }
    ]

    enriched = lab.vlm_target_crop_request_bundle(
        request_bundle,
        detector_records,
        crop_root=tmp_path / "crops",
        padding_pixels=0,
    )

    assert "target_crop" not in enriched["case_inputs"][0]
    assert enriched["vlm_target_crop_enrichment"]["summary"] == {
        "case_count": 1,
        "cases_with_target_crop": 0,
        "missing_target_detection_count": 1,
    }


def test_vlm_target_crop_request_bundle_skips_detection_without_bbox(
    tmp_path: Path,
) -> None:
    rgb_path = tmp_path / "frames" / "0002.ppm"
    rgb_path.parent.mkdir(parents=True)
    rgb_path.write_text(
        "P3\n3 2\n255\n"
        "0 0 0 10 0 0 20 0 0\n"
        "0 10 0 10 10 0 20 10 0\n",
        encoding="utf-8",
    )
    request_bundle: dict[str, Any] = {
        "case_count": 2,
        "case_inputs": [
            {
                "case_id": "case-apple",
                "target": {"label": "apple", "object_id": "apple_1"},
                "primary_frame": {
                    "episode_id": "episode-001",
                    "scene_id": "FloorPlan1",
                    "step": 2,
                },
            },
            {
                "case_id": "case-mug",
                "target": {"label": "mug", "object_id": "mug_1"},
                "primary_frame": {
                    "episode_id": "episode-001",
                    "scene_id": "FloorPlan1",
                    "step": 2,
                },
            },
        ],
        "request_bundle_digest": "old",
    }
    detector_records = [
        {
            "detector_name": "visible_segmentation_rgbd",
            "episode_id": "episode-001",
            "rgb_path": str(rgb_path),
            "scene_id": "FloorPlan1",
            "step": 2,
            "detections": [
                {
                    "bbox_2d_xyxy": None,
                    "confidence": 1.0,
                    "label": "apple",
                    "object_id": "apple_1",
                    "visible": True,
                },
                {
                    "bbox_2d_xyxy": [1, 0, 2, 1],
                    "confidence": 0.9,
                    "label": "mug",
                    "object_id": "mug_1",
                    "visible": True,
                },
            ],
        }
    ]

    enriched = lab.vlm_target_crop_request_bundle(
        request_bundle,
        detector_records,
        crop_root=tmp_path / "crops",
        padding_pixels=0,
    )

    assert "target_crop" not in enriched["case_inputs"][0]
    assert enriched["case_inputs"][1]["target_crop"]["bbox_2d_xyxy"] == [1, 0, 2, 1]
    assert enriched["vlm_target_crop_enrichment"]["summary"] == {
        "case_count": 2,
        "cases_with_target_crop": 1,
        "missing_target_detection_count": 1,
    }


def test_vlm_target_crop_request_bundle_uses_detector_metadata_frame_ids(
    tmp_path: Path,
) -> None:
    rgb_path = tmp_path / "frames" / "0002.ppm"
    rgb_path.parent.mkdir(parents=True)
    rgb_path.write_text("P3\n2 1\n255\n10 0 0 20 0 0\n", encoding="utf-8")
    request_bundle: dict[str, Any] = {
        "case_count": 1,
        "case_inputs": [
            {
                "case_id": "case-apple",
                "target": {"label": "apple", "object_id": "apple_1"},
                "primary_frame": {
                    "episode_id": "episode-001",
                    "scene_id": "FloorPlan1",
                    "step": 2,
                },
            }
        ],
        "request_bundle_digest": "old",
    }
    detector_records = [
        {
            "metadata": {
                "detector": "ai2thor_metadata_coverage_objects",
                "episode_id": "episode-001",
                "scene_id": "FloorPlan1",
                "source_kind": "ai2thor_metadata_coverage",
            },
            "rgb_path": str(rgb_path),
            "step": 2,
            "detections": [
                {
                    "bbox_2d_xyxy": [0, 0, 1, 0],
                    "confidence": 0.91,
                    "label": "apple",
                    "object_id": "apple_1",
                    "visible": True,
                }
            ],
        }
    ]

    enriched = lab.vlm_target_crop_request_bundle(
        request_bundle,
        detector_records,
        crop_root=tmp_path / "crops",
        padding_pixels=0,
    )

    crop = enriched["case_inputs"][0]["target_crop"]
    assert crop["detector_name"] == "ai2thor_metadata_coverage_objects"
    assert crop["source_frame_id"] == "episode-001:FloorPlan1:0002"
    assert enriched["vlm_target_crop_enrichment"]["summary"] == {
        "case_count": 1,
        "cases_with_target_crop": 1,
        "missing_target_detection_count": 0,
    }


def test_vlm_target_crop_request_bundle_rescales_detector_bbox_from_1000_space(
    tmp_path: Path,
) -> None:
    rgb_path = tmp_path / "frames" / "0002.ppm"
    rgb_path.parent.mkdir(parents=True)
    rgb_path.write_text(
        "P3\n4 3\n255\n"
        "0 0 0 10 0 0 20 0 0 30 0 0\n"
        "0 10 0 10 10 0 20 10 0 30 10 0\n"
        "0 20 0 10 20 0 20 20 0 30 20 0\n",
        encoding="utf-8",
    )
    request_bundle: dict[str, Any] = {
        "case_count": 1,
        "case_inputs": [
            {
                "case_id": "case-faucet",
                "target": {"label": "faucet", "object_id": "faucet_1"},
                "primary_frame": {
                    "episode_id": "episode-001",
                    "frame_id": "episode-001:FloorPlan1:0002",
                    "scene_id": "FloorPlan1",
                    "step": 2,
                },
            }
        ],
        "request_bundle_digest": "old",
    }
    detector_records = [
        {
            "detector_name": "qwen37_vlm_rgbd_detector",
            "episode_id": "episode-001",
            "rgb_path": str(rgb_path),
            "scene_id": "FloorPlan1",
            "step": 2,
            "detections": [
                {
                    "bbox_2d_xyxy": [250, 660, 760, 999],
                    "confidence": 0.84,
                    "label": "faucet",
                    "object_id": "faucet_1",
                    "visible": True,
                }
            ],
        }
    ]

    enriched = lab.vlm_target_crop_request_bundle(
        request_bundle,
        detector_records,
        crop_root=tmp_path / "crops",
        padding_pixels=0,
    )

    crop = enriched["case_inputs"][0]["target_crop"]
    assert crop["bbox_2d_xyxy"] == [1, 2, 3, 2]
    assert Path(crop["rgb_path"]).read_text(encoding="utf-8") == (
        "P3\n3 1\n255\n"
        "10 20 0 20 20 0 30 20 0\n"
    )
    assert enriched["vlm_target_crop_enrichment"]["summary"] == {
        "case_count": 1,
        "cases_with_target_crop": 1,
        "missing_target_detection_count": 0,
    }


def test_vlm_calibration_cli_writes_target_crop_request_bundle(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    rgb_path = tmp_path / "frames" / "0002.ppm"
    rgb_path.parent.mkdir(parents=True)
    rgb_path.write_text(
        "P3\n2 1\n255\n10 0 0 20 0 0\n",
        encoding="utf-8",
    )
    request_bundle_path = tmp_path / "request-bundle.json"
    detector_jsonl_path = tmp_path / "detector.jsonl"
    output_path = tmp_path / "request-bundle-crops.json"
    crop_root = tmp_path / "crops"
    request_bundle_path.write_text(
        json.dumps(
            {
                "case_count": 1,
                "case_inputs": [
                    {
                        "case_id": "case-apple",
                        "target": {"label": "apple", "object_id": "apple_1"},
                        "primary_frame": {
                            "episode_id": "episode-001",
                            "frame_id": "episode-001:FloorPlan1:0002",
                            "scene_id": "FloorPlan1",
                            "step": 2,
                        },
                    }
                ],
                "request_bundle_digest": "old",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    detector_jsonl_path.write_text(
        json.dumps(
            {
                "detector_name": "qwen37_vlm_rgbd_detector",
                "episode_id": "episode-001",
                "rgb_path": str(rgb_path),
                "scene_id": "FloorPlan1",
                "step": 2,
                "detections": [
                    {
                        "bbox_2d_xyxy": [0, 0, 0, 0],
                        "confidence": 0.91,
                        "label": "apple",
                        "object_id": "apple_1",
                        "visible": True,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    module = load_vlm_calibration_script()
    main = cast(MainFn, getattr(module, "main"))

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle_path),
            "--target-crop-detector-jsonl",
            str(detector_jsonl_path),
            "--target-crop-root",
            str(crop_root),
            "--target-crop-request-bundle-output",
            str(output_path),
            "--target-crop-padding-pixels",
            "0",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"action": "build_vlm_target_crop_request_bundle"' in output
    enriched = json.loads(output_path.read_text(encoding="utf-8"))
    crop = enriched["case_inputs"][0]["target_crop"]
    assert Path(crop["rgb_path"]).exists()
    assert enriched["vlm_target_crop_enrichment"]["summary"] == {
        "case_count": 1,
        "cases_with_target_crop": 1,
        "missing_target_detection_count": 0,
    }


def test_vlm_calibration_cli_writes_enhanced_visual_request_bundle(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    rgb_path = tmp_path / "frames" / "0002.ppm"
    rgb_path.parent.mkdir(parents=True)
    rgb_path.write_text(
        "P3\n3 2\n255\n"
        "0 0 0 10 0 0 20 0 0\n"
        "0 10 0 10 10 0 20 10 0\n",
        encoding="utf-8",
    )
    request_bundle_path = tmp_path / "request-bundle.json"
    frame_index_path = tmp_path / "frame-index.jsonl"
    detector_jsonl_path = tmp_path / "detector.jsonl"
    output_path = tmp_path / "request-bundle-visual.json"
    crop_root = tmp_path / "crops"
    request_bundle_path.write_text(
        json.dumps(
            {
                "action": "offline_control_prediction_request_bundle",
                "case_count": 1,
                "case_inputs": [
                    {
                        "case_id": "case-apple",
                        "episode_id": "episode-001",
                        "scene_id": "FloorPlan1",
                        "question_type": "object_location",
                        "question_text": "Where is the apple?",
                        "target": {"label": "apple", "object_id": "apple_1"},
                        "primary_frame": {
                            "episode_id": "episode-001",
                            "frame_id": "episode-001:FloorPlan1:0002",
                            "scene_id": "FloorPlan1",
                            "step": 2,
                        },
                    }
                ],
                "request_bundle_digest": "old",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    frame_index_path.write_text(
        (
            '{"episode_id":"episode-001","scene_id":"FloorPlan1","step":2,'
            '"visible_object_ids":["apple_1","countertop_1"],'
            '"visible_object_labels":["apple","countertop"]}\n'
        ),
        encoding="utf-8",
    )
    detector_jsonl_path.write_text(
        json.dumps(
            {
                "detector_name": "qwen37_vlm_rgbd_detector",
                "episode_id": "episode-001",
                "rgb_path": str(rgb_path),
                "scene_id": "FloorPlan1",
                "step": 2,
                "detections": [
                    {
                        "bbox_2d_xyxy": [1, 0, 2, 1],
                        "confidence": 0.91,
                        "label": "apple",
                        "object_id": "apple_1",
                        "visible": True,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    module = load_vlm_calibration_script()
    main = cast(MainFn, getattr(module, "main"))

    exit_code = main(
        [
            "--request-bundle",
            str(request_bundle_path),
            "--frame-index",
            str(frame_index_path),
            "--target-crop-detector-jsonl",
            str(detector_jsonl_path),
            "--target-crop-root",
            str(crop_root),
            "--enhanced-vlm-request-bundle-output",
            str(output_path),
            "--target-crop-padding-pixels",
            "0",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"action": "build_enhanced_vlm_request_bundle"' in output
    enriched = json.loads(output_path.read_text(encoding="utf-8"))
    case = enriched["case_inputs"][0]
    assert case["support_candidates"] == [
        {
            "label": "countertop",
            "relation_hint": "ON",
        }
    ]
    assert case["answer_options"] == [
        {
            "destination_label": "countertop",
            "option_id": "locopt_001",
            "relation": "ON",
            "source": "support_candidate",
        },
        {
            "destination_label": "room",
            "option_id": "locopt_002",
            "relation": "IN_ROOM",
            "source": "fallback_room",
        },
    ]
    assert Path(case["target_crop"]["rgb_path"]).exists()
    assert enriched["vlm_visual_request_enrichment"]["summary"] == {
        "answer_option_count": 2,
        "case_count": 1,
        "cases_with_answer_options": 1,
        "cases_with_support_candidates": 1,
        "cases_with_target_crop": 1,
    }
    serialized = json.dumps(enriched, sort_keys=True)
    assert "gold_answer" not in serialized
    assert "gold_evidence" not in serialized
    assert "countertop_1" not in serialized


def test_vlm_calibration_cli_writes_support_detector_handoff(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    detector_handoff_path = tmp_path / "detector-handoff.json"
    support_bundle_path = tmp_path / "support-bundle.json"
    output_path = tmp_path / "detector-handoff-support.json"
    detector_handoff_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "dsg-spatialqa-lab.independent-detector-rgbd-handoff.v1"
                ),
                "required_frames": [
                    {
                        "episode_id": "episode-001",
                        "scene_id": "FloorPlan1",
                        "frame_step": 2,
                        "target_labels": ["apple"],
                        "target_object_ids": ["apple_1"],
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    support_bundle_path.write_text(
        json.dumps(
            {
                "case_inputs": [
                    {
                        "case_id": "case-apple",
                        "primary_frame": {
                            "episode_id": "episode-001",
                            "scene_id": "FloorPlan1",
                            "step": 2,
                        },
                        "support_candidates": [
                            {"label": "countertop", "relation_hint": "ON"}
                        ],
                    }
                ],
                "request_bundle_digest": "a" * 64,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    module = load_vlm_calibration_script()
    main = cast(MainFn, getattr(module, "main"))

    exit_code = main(
        [
            "--detector-handoff",
            str(detector_handoff_path),
            "--support-candidate-request-bundle",
            str(support_bundle_path),
            "--support-detector-handoff-output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"action": "build_vlm_support_detector_handoff"' in output
    enriched = json.loads(output_path.read_text(encoding="utf-8"))
    assert enriched["required_frames"][0]["support_labels"] == ["countertop"]


def test_vlm_primary_frame_visibility_report_counts_target_visible_inputs() -> None:
    request_bundle: dict[str, Any] = {
        "case_inputs": [
            {
                "case_id": "case-visible",
                "episode_id": "episode-001",
                "primary_frame": {
                    "episode_id": "episode-001",
                    "scene_id": "FloorPlan1",
                    "step": 2,
                },
                "question_type": "object_location",
                "scene_id": "FloorPlan1",
                "target": {"label": "apple", "object_id": "apple_1"},
            },
            {
                "case_id": "case-hidden",
                "episode_id": "episode-001",
                "primary_frame": {
                    "episode_id": "episode-001",
                    "scene_id": "FloorPlan1",
                    "step": 3,
                },
                "question_type": "object_location",
                "scene_id": "FloorPlan1",
                "target": {"label": "book", "object_id": "book_1"},
            },
            {
                "case_id": "case-relative",
                "episode_id": "episode-001",
                "primary_frame": {
                    "episode_id": "episode-001",
                    "scene_id": "FloorPlan1",
                    "step": 2,
                },
                "question_type": "relative_relation",
                "scene_id": "FloorPlan1",
                "target": {"label": "apple", "object_id": "apple_1"},
            },
        ]
    }
    frame_index = [
        {
            "episode_id": "episode-001",
            "scene_id": "FloorPlan1",
            "step": 2,
            "visible_object_ids": ["apple_1"],
            "visible_object_labels": ["apple"],
        },
        {
            "episode_id": "episode-001",
            "scene_id": "FloorPlan1",
            "step": 3,
            "visible_object_ids": ["chair_1"],
            "visible_object_labels": ["chair"],
        },
    ]

    report = lab.vlm_primary_frame_visibility_report(request_bundle, frame_index)

    assert report["summary"] == {
        "case_count": 3,
        "object_location_case_count": 2,
        "primary_frame_missing_count": 0,
        "target_visible_primary_frame_count": 1,
        "target_visible_primary_frame_rate": 0.5,
    }
    assert report["target_visible_primary_frame_case_ids"] == ["case-visible"]
    serialized = lab.vlm_primary_frame_visibility_report_json(report)
    assert "visible_object_ids" not in serialized
    assert "visible_object_labels" not in serialized
    assert lab.validate_vlm_primary_frame_visibility_report(report)["valid"] is True


def test_vlm_frame_index_rows_from_detector_records_are_vlm_safe() -> None:
    detector_record = {
        "episode_id": "episode-001",
        "scene_id": "FloorPlan1",
        "step": 42,
        "rgb_path": "frames/0042.rgb.ppm",
        "depth_path": "frames/0042.depth.npy",
        "segmentation_path": "frames/0042.seg.ppm",
        "metadata": {
            "dataset_id": "ai2thor-real-small",
            "source_name": "visible_rgbd_detector",
        },
        "detections": [
            {"object_id": "apple_1", "label": "apple", "visible": True},
            {"object_id": "book_1", "label": "book", "visible": False},
            {"object_id": "mug_1", "label": "mug"},
        ],
    }

    rows = lab.vlm_frame_index_rows_from_detector_records([detector_record])
    report = lab.vlm_frame_index_report(
        rows,
        source_path="detector.jsonl",
        existing_frame_index_path="frame-index.jsonl",
    )

    assert rows == [
        {
            "schema_version": "dsg-spatialqa-lab.real-experiment-frame-trace.v1",
            "dataset_id": "ai2thor-real-small",
            "episode_id": "episode-001",
            "scene_id": "FloorPlan1",
            "step": 42,
            "asset_paths": {
                "depth": "frames/0042.depth.npy",
                "rgb": "frames/0042.rgb.ppm",
                "segmentation": "frames/0042.seg.ppm",
            },
            "asset_present": {
                "depth": False,
                "rgb": False,
                "segmentation": False,
            },
            "detector_depth_path": "frames/0042.depth.npy",
            "detector_object_count": 2,
            "detector_rgb_path": "frames/0042.rgb.ppm",
            "detector_segmentation_path": "frames/0042.seg.ppm",
            "source_name": "visible_rgbd_detector",
            "visible_object_ids": ["apple_1", "mug_1"],
            "visible_object_labels": ["apple", "mug"],
        }
    ]
    assert report["summary"] == {
        "frame_count": 1,
        "source_path": "detector.jsonl",
        "target_visible_frame_count": 1,
        "visible_object_id_count": 2,
    }
    assert report["existing_frame_index_path"] == "frame-index.jsonl"
    assert report["frame_index_digest"] == lab.vlm_frame_index_rows_digest(rows)
    assert report["report_digest"] == lab.vlm_frame_index_report_digest(report)
    assert lab.validate_vlm_frame_index_report(report)["valid"] is True


def test_vlm_support_gap_report_marks_present_support_missing_relation() -> None:
    case = _object_location_case(
        "case-apple",
        visible=True,
        relation="ON",
        dst="countertop_001",
    )
    prediction = lab.QAPrediction(
        id="case-apple",
        answer={
            "current_location": {
                "dst": "ai2thor_room",
                "relation": "IN_ROOM",
            }
        },
        confidence=0.8,
        error=None,
    )
    semantic_report = lab.vlm_semantic_eval_report([case], [prediction])
    graph = lab.DynamicSceneGraph()
    graph.upsert_object(
        "object_apple",
        "apple",
        lab.Pose3D(0.0, 1.0, 0.0),
        lab.BBox3D(lab.Pose3D(0.0, 1.0, 0.0), (0.1, 0.1, 0.1)),
        confidence=0.9,
        visible=True,
        step=10,
        attributes={
            "evidence_kinds": ["depth", "detector", "rgb"],
            "scene_id": "FloorPlan1",
            "source_kind": "detector",
        },
    )
    graph.upsert_object(
        "countertop_001",
        "countertop",
        lab.Pose3D(0.0, 0.9, 0.0),
        lab.BBox3D(lab.Pose3D(0.0, 0.9, 0.0), (1.0, 0.1, 1.0)),
        confidence=0.9,
        visible=True,
        step=10,
        attributes={
            "evidence_kinds": ["depth", "detector", "rgb"],
            "scene_id": "FloorPlan1",
            "source_kind": "detector",
        },
    )

    report = lab.vlm_support_gap_report([case], semantic_report, graph)

    assert report["summary"] == {
        "case_count": 1,
        "failed_on_case_count": 1,
        "support_missing_count": 0,
        "support_present_relation_missing_count": 1,
        "target_missing_count": 0,
    }
    assert report["support_labels_to_collect"] == []
    assert report["cases"][0]["gap_kind"] == "support_present_but_relation_missing"
    assert report["cases"][0]["support_node_ids"] == ["countertop_001"]
    assert report["cases"][0]["evaluator_only"] is True
    assert report["report_digest"] == lab.vlm_support_gap_report_digest(report)
    assert lab.validate_vlm_support_gap_report(report)["valid"] is True


def test_vlm_calibration_cli_writes_support_gap_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    case = _object_location_case(
        "case-apple",
        visible=True,
        relation="ON",
        dst="countertop_001",
    )
    prediction = lab.QAPrediction(
        id="case-apple",
        answer={
            "current_location": {
                "dst": "ai2thor_room",
                "relation": "IN_ROOM",
            }
        },
        confidence=0.8,
        error=None,
    )
    semantic_report = lab.vlm_semantic_eval_report([case], [prediction])
    graph = lab.DynamicSceneGraph()
    graph.upsert_object(
        "object_apple",
        "apple",
        lab.Pose3D(0.0, 1.0, 0.0),
        lab.BBox3D(lab.Pose3D(0.0, 1.0, 0.0), (0.1, 0.1, 0.1)),
        confidence=0.9,
        visible=True,
        step=10,
        attributes={
            "evidence_kinds": ["depth", "detector", "rgb"],
            "scene_id": "FloorPlan1",
            "source_kind": "detector",
        },
    )

    qa_path = tmp_path / "qa.jsonl"
    semantic_path = tmp_path / "semantic.json"
    graph_path = tmp_path / "predicted-graph.json"
    report_path = tmp_path / "support-gap.json"
    qa_path.write_text(lab.qa_dataset_jsonl([case]), encoding="utf-8")
    lab.save_vlm_semantic_eval_report(semantic_report, semantic_path)
    lab.save_graph_json(graph, graph_path)
    module = load_vlm_calibration_script()
    main = cast(MainFn, getattr(module, "main"))

    exit_code = main(
        [
            "--qa",
            str(qa_path),
            "--support-gap-semantic-eval-report",
            str(semantic_path),
            "--support-gap-predicted-graph",
            str(graph_path),
            "--support-gap-report-output",
            str(report_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"action": "build_vlm_support_gap_report"' in output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["target_missing_count"] == 0
    assert report["summary"]["support_missing_count"] == 1
    assert report["support_labels_to_collect"] == ["countertop"]
    assert lab.validate_vlm_support_gap_report(report)["valid"] is True


def _object_location_case(
    case_id: str,
    *,
    visible: bool,
    relation: str,
    dst: str,
    label: str | None = None,
) -> lab.QACase:
    object_label = label if label is not None else "object"
    object_id = case_id.replace("case-", f"{object_label}_")
    return lab.QACase(
        id=case_id,
        scene_id="FloorPlan1",
        episode_id="episode-001",
        graph_digest="0" * 64,
        step=10,
        question={"type": "object_location", "object_id": object_id},
        question_type="object_location",
        answer={
            "confidence": 0.9,
            "current_location": {"dst": dst, "relation": relation, "step": 10},
            "label": object_label,
            "last_seen_step": 10 if visible else None,
            "object_id": object_id,
            "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0, "z": 0.0},
            "state_step": 10,
            "visible": visible,
        },
        answer_type="object_location",
        required_nodes=(object_id,),
        required_edges=(f"{object_id}-{relation}-{dst}-10",),
        tags=("real", "qa", "object_location"),
    )
