from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol

from _pytest.capture import CaptureFixture
import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
FUSION_SCRIPT = ROOT / "scripts" / "fuse_vlm_graph_predictions.py"
BUILD_EVIDENCE_SCRIPT = ROOT / "scripts" / "build_vlm_graph_evidence.py"
ADJUDICATION_SCRIPT = ROOT / "scripts" / "evaluate_vlm_graph_adjudication.py"
ADJUDICATION_ALL_SCRIPT = ROOT / "scripts" / "evaluate_vlm_graph_adjudication_all_episodes.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_fusion_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "fuse_vlm_graph_predictions_test_script",
        FUSION_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_evidence_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_vlm_graph_evidence_test_script",
        BUILD_EVIDENCE_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_adjudication_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "evaluate_vlm_graph_adjudication_test_script",
        ADJUDICATION_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_adjudication_all_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "evaluate_vlm_graph_adjudication_all_episodes_test_script",
        ADJUDICATION_ALL_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_vlm_graph_fusion_uses_explicit_graph_location() -> None:
    vlm = lab.QAPrediction(
        id="episode:scene:0001:object_location:apple_001",
        answer={"source": "vlm", "text": "unknown"},
        confidence=0.2,
    )
    graph = lab.QAPrediction(
        id="episode:scene:0001:object_location:apple_001:observation_aware:42",
        answer={
            "current_location": {"dst": "countertop_001", "relation": "ON", "step": 42},
            "source": "graph_tool",
        },
        evidence_nodes=("apple_001", "countertop_001"),
        evidence_edges=("apple_001-ON-countertop_001-42",),
        confidence=0.9,
    )

    fused = lab.fuse_vlm_graph_predictions([vlm], [graph])

    assert len(fused) == 1
    assert fused[0].id == graph.id
    assert fused[0].answer["current_location"] == {
        "dst": "countertop_001",
        "relation": "ON",
        "step": 42,
    }
    assert fused[0].answer["fusion"] == {
        "fusion_policy": "explicit_graph_relation_or_vlm_fallback",
        "fusion_source": "graph_tool",
        "graph_prediction_id": graph.id,
        "vlm_prediction_id": vlm.id,
    }
    assert fused[0].evidence_edges == graph.evidence_edges
    assert fused[0].confidence == 0.9


def test_vlm_graph_fusion_falls_back_to_vlm_for_room_only_graph_location() -> None:
    vlm = lab.QAPrediction(
        id="episode:scene:0001:object_location:apple_001",
        answer={"source": "vlm", "text": "on the countertop"},
        confidence=0.8,
    )
    graph = lab.QAPrediction(
        id="episode:scene:0001:object_location:apple_001:observation_aware:42",
        answer={
            "current_location": {"dst": "ai2thor_room", "relation": "IN_ROOM", "step": 42},
            "source": "graph_tool",
        },
        confidence=0.9,
    )

    fused = lab.fuse_vlm_graph_predictions([vlm], [graph])

    assert len(fused) == 1
    assert fused[0].id == graph.id
    assert fused[0].answer["text"] == "on the countertop"
    assert fused[0].answer["fusion"]["fusion_source"] == "vlm"
    assert fused[0].answer["fusion"]["graph_fallback_reason"] == "room_level_graph_location"
    assert fused[0].confidence == 0.8


def test_trusted_vlm_graph_fusion_rejects_implausible_graph_support() -> None:
    vlm = lab.QAPrediction(
        id="episode:scene:0001:object_location:bread_001",
        answer={"source": "vlm", "text": "on the countertop"},
        confidence=0.8,
    )
    graph = lab.QAPrediction(
        id="episode:scene:0001:object_location:bread_001:observation_aware:42",
        answer={
            "current_location": {"dst": "egg_001", "relation": "ON", "step": 42},
            "label": "bread",
            "object_id": "bread_001",
            "source": "graph_tool",
        },
        confidence=0.9,
    )

    fused = lab.fuse_vlm_graph_predictions(
        [vlm],
        [graph],
        fusion_policy=lab.VLM_GRAPH_TRUSTED_FUSION_POLICY,
    )

    assert fused[0].answer["text"] == "on the countertop"
    assert fused[0].answer["fusion"]["fusion_source"] == "vlm"
    assert fused[0].answer["fusion"]["graph_fallback_reason"] == "implausible_support_label"


def test_trusted_vlm_graph_fusion_uses_plausible_graph_when_vlm_is_unknown() -> None:
    vlm = lab.QAPrediction(
        id="episode:scene:0001:object_location:laptop_001",
        answer={"source": "vlm", "text": "unknown"},
        confidence=0.1,
    )
    graph = lab.QAPrediction(
        id="episode:scene:0001:object_location:laptop_001:observation_aware:42",
        answer={
            "current_location": {"dst": "chair_001", "relation": "ON", "step": 42},
            "label": "laptop",
            "object_id": "laptop_001",
            "source": "graph_tool",
        },
        confidence=0.9,
    )

    fused = lab.fuse_vlm_graph_predictions(
        [vlm],
        [graph],
        fusion_policy=lab.VLM_GRAPH_TRUSTED_FUSION_POLICY,
    )

    assert fused[0].answer["current_location"] == {
        "dst": "chair_001",
        "relation": "ON",
        "step": 42,
    }
    assert fused[0].answer["fusion"]["fusion_source"] == "graph_tool"
    assert fused[0].answer["fusion"]["fusion_policy"] == lab.VLM_GRAPH_TRUSTED_FUSION_POLICY


def test_fuse_vlm_graph_predictions_script_writes_stable_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    vlm_path = tmp_path / "vlm.jsonl"
    graph_path = tmp_path / "graph.jsonl"
    output_path = tmp_path / "fusion.jsonl"
    report_path = tmp_path / "fusion-report.json"
    vlm = [
        lab.QAPrediction(
            id="episode:scene:0001:object_location:apple_001",
            answer={"text": "unknown"},
            confidence=0.2,
        ),
        lab.QAPrediction(
            id="episode:scene:0002:object_location:book_001",
            answer={"text": "on the chair"},
            confidence=0.7,
        ),
    ]
    graph = [
        lab.QAPrediction(
            id="episode:scene:0001:object_location:apple_001:observation_aware:42",
            answer={"current_location": {"dst": "countertop_001", "relation": "ON"}},
            confidence=0.9,
        ),
        lab.QAPrediction(
            id="episode:scene:0002:object_location:book_001:observation_aware:43",
            answer={"current_location": {"dst": "ai2thor_room", "relation": "IN_ROOM"}},
            confidence=0.9,
        ),
    ]
    lab.save_qa_predictions(vlm, vlm_path)
    lab.save_qa_predictions(graph, graph_path)

    module = load_fusion_script()
    exit_code = module.main(
        [
            "--vlm-predictions",
            str(vlm_path),
            "--graph-predictions",
            str(graph_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    predictions = lab.load_qa_predictions(output_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert [prediction.answer["fusion"]["fusion_source"] for prediction in predictions] == [
        "graph_tool",
        "vlm",
    ]
    assert payload["summary"] == {
        "fused_prediction_count": 2,
        "graph_prediction_count": 2,
        "graph_tool_source_count": 1,
        "unmatched_graph_prediction_count": 0,
        "unmatched_vlm_prediction_count": 0,
        "vlm_prediction_count": 2,
        "vlm_source_count": 1,
    }
    assert report["prediction_digest"] == lab.qa_predictions_digest(predictions)
    assert lab.validate_vlm_graph_fusion_report(report)["valid"] is True


def test_fuse_vlm_graph_predictions_script_accepts_trusted_policy(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    vlm_path = tmp_path / "vlm.jsonl"
    graph_path = tmp_path / "graph.jsonl"
    output_path = tmp_path / "fusion.jsonl"
    report_path = tmp_path / "fusion-report.json"
    lab.save_qa_predictions(
        [
            lab.QAPrediction(
                id="episode:scene:0001:object_location:bread_001",
                answer={"text": "on the countertop"},
                confidence=0.8,
            )
        ],
        vlm_path,
    )
    lab.save_qa_predictions(
        [
            lab.QAPrediction(
                id="episode:scene:0001:object_location:bread_001:observation_aware:42",
                answer={
                    "current_location": {"dst": "egg_001", "relation": "ON"},
                    "label": "bread",
                    "object_id": "bread_001",
                },
                confidence=0.9,
            )
        ],
        graph_path,
    )

    module = load_fusion_script()
    exit_code = module.main(
        [
            "--vlm-predictions",
            str(vlm_path),
            "--graph-predictions",
            str(graph_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--fusion-policy",
            lab.VLM_GRAPH_TRUSTED_FUSION_POLICY,
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    predictions = lab.load_qa_predictions(output_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert predictions[0].answer["fusion"]["fusion_source"] == "vlm"
    assert predictions[0].answer["fusion"]["graph_fallback_reason"] == (
        "implausible_support_label"
    )
    assert payload["fusion_policy"] == lab.VLM_GRAPH_TRUSTED_FUSION_POLICY
    assert report["fusion_policy"] == lab.VLM_GRAPH_TRUSTED_FUSION_POLICY


def test_adjudication_all_episodes_missing_predictions_is_not_ready(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_adjudication_all_script()
    report_path = tmp_path / "adjudication-readiness.json"
    missing_path = tmp_path / "missing-adjudicated.jsonl"

    exit_code = module.main(
        [
            "--qa-root",
            str(tmp_path / "qa-v2-active"),
            "--adjudicated-predictions",
            str(missing_path),
            "--report",
            str(report_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["ready"] is False
    assert payload["research_ready"] is False
    assert payload["blockers"] == ["missing_adjudicated_predictions"]
    assert payload["final_record_written"] is False
    assert report == payload


def test_vlm_graph_evidence_score_report_explains_trust_and_reject_reasons() -> None:
    cases = [
        _object_location_case("episode:scene:0001:object_location:laptop_001", "laptop_001", 42),
        _object_location_case("episode:scene:0002:object_location:bread_001", "bread_001", 42),
    ]
    graph_predictions = [
        lab.QAPrediction(
            id="episode:scene:0001:object_location:laptop_001:observation_aware:42",
            answer={
                "current_location": {"dst": "chair_001", "relation": "ON", "step": 42},
                "label": "laptop",
                "object_id": "laptop_001",
            },
            confidence=0.9,
        ),
        lab.QAPrediction(
            id="episode:scene:0002:object_location:bread_001:observation_aware:42",
            answer={
                "current_location": {"dst": "egg_001", "relation": "ON", "step": 42},
                "label": "bread",
                "object_id": "bread_001",
            },
            confidence=0.9,
        ),
    ]
    detector_records = [
        _detector_record("episode", "scene", 42, "laptop_001", "laptop"),
        _detector_record("episode", "scene", 42, "chair_001", "chair"),
        _detector_record("episode", "scene", 42, "bread_001", "bread"),
        _detector_record("episode", "scene", 42, "egg_001", "egg"),
    ]

    report = lab.vlm_graph_evidence_score_report(
        cases,
        graph_predictions,
        detector_records=detector_records,
    )

    rows = {row["case_id"]: row for row in report["cases"]}
    laptop = rows["episode:scene:0001:object_location:laptop_001"]
    bread = rows["episode:scene:0002:object_location:bread_001"]
    assert laptop["reject_reason"] is None
    assert laptop["trusted"] is True
    assert laptop["graph_trust_score"] >= 0.8
    assert laptop["checks"]["target_label_match"] is True
    assert laptop["checks"]["support_label_plausible"] is True
    assert laptop["checks"]["required_evidence_kinds_present"] is True
    assert bread["reject_reason"] == "implausible_support_label"
    assert bread["trusted"] is False
    assert bread["checks"]["support_label_plausible"] is False
    assert report["summary"]["trusted_case_count"] == 1
    assert report["summary"]["rejected_case_count"] == 1
    assert lab.validate_vlm_graph_evidence_score_report(report)["valid"] is True


def test_vlm_graph_evidence_score_report_trusts_dresser_and_towel_holder_supports() -> None:
    cases = [
        _object_location_case(
            "episode:scene:0001:object_location:alarmclock_001",
            "alarmclock_001",
            42,
        ),
        _object_location_case(
            "episode:scene:0002:object_location:handtowel_001",
            "handtowel_001",
            42,
        ),
    ]
    graph_predictions = [
        lab.QAPrediction(
            id=f"{cases[0].id}:observation_aware:42",
            answer={
                "current_location": {"dst": "dresser_001", "relation": "ON", "step": 42},
                "label": "alarmclock",
                "object_id": "alarmclock_001",
            },
            confidence=0.9,
        ),
        lab.QAPrediction(
            id=f"{cases[1].id}:observation_aware:42",
            answer={
                "current_location": {
                    "dst": "handtowelholder_001",
                    "relation": "ON",
                    "step": 42,
                },
                "label": "handtowel",
                "object_id": "handtowel_001",
            },
            confidence=0.9,
        ),
    ]
    detector_records = [
        _detector_record("episode", "scene", 42, "alarmclock_001", "alarmclock"),
        _detector_record("episode", "scene", 42, "dresser_001", "dresser"),
        _detector_record("episode", "scene", 42, "handtowel_001", "handtowel"),
        _detector_record(
            "episode",
            "scene",
            42,
            "handtowelholder_001",
            "handtowelholder",
        ),
    ]

    report = lab.vlm_graph_evidence_score_report(
        cases,
        graph_predictions,
        detector_records=detector_records,
    )

    rows = {row["case_id"]: row for row in report["cases"]}
    assert rows[cases[0].id]["trusted"] is True
    assert rows[cases[0].id]["checks"]["support_label_plausible"] is True
    assert rows[cases[1].id]["trusted"] is True
    assert rows[cases[1].id]["checks"]["support_label_plausible"] is True
    assert report["summary"]["trusted_case_count"] == 2
    assert report["summary"]["rejected_case_count"] == 0


def test_vlm_graph_evidence_score_report_reads_detector_observation_record_schema() -> None:
    case = _object_location_case(
        "episode:scene:0001:object_location:laptop_001",
        "laptop_001",
        42,
    )
    graph_prediction = lab.QAPrediction(
        id=f"{case.id}:observation_aware:42",
        answer={
            "current_location": {"dst": "chair_001", "relation": "ON", "step": 42},
            "label": "laptop",
            "object_id": "laptop_001",
        },
        confidence=0.9,
    )
    detector_records = [
        _detector_observation_record("episode", "scene", 42, "laptop_001", "laptop"),
        _detector_observation_record("episode", "scene", 42, "chair_001", "chair"),
    ]

    report = lab.vlm_graph_evidence_score_report(
        [case],
        [graph_prediction],
        detector_records=detector_records,
    )

    row = report["cases"][0]
    assert row["trusted"] is True
    assert row["target_object_evidence"]["bbox_3d_center"] == {
        "x": 1.0,
        "y": 1.0,
        "z": 1.0,
        "yaw": 0.0,
    }
    assert row["target_object_evidence"]["bbox_3d_size"] == [0.1, 0.2, 0.3]
    assert row["target_object_evidence"]["evidence_kinds"] == [
        "depth",
        "detector",
        "rgb",
    ]


def test_vlm_graph_evidence_score_report_trusts_room_level_target_evidence() -> None:
    case = _object_location_case(
        "episode:scene:0001:object_location:cabinet_001",
        "cabinet_001",
        42,
    )
    graph_prediction = lab.QAPrediction(
        id=f"{case.id}:observation_aware:42",
        answer={
            "current_location": {"dst": "ai2thor_room", "relation": "IN_ROOM", "step": 42},
            "label": "cabinet",
            "object_id": "cabinet_001",
        },
        confidence=0.9,
    )
    detector_records = [
        _detector_observation_record("episode", "scene", 42, "cabinet_001", "cabinet"),
    ]

    report = lab.vlm_graph_evidence_score_report(
        [case],
        [graph_prediction],
        detector_records=detector_records,
    )

    row = report["cases"][0]
    assert row["trusted"] is True
    assert row["trust_scope"] == "room_level_target"
    assert row["relation"] == "IN_ROOM"
    assert row["reject_reason"] is None
    assert row["checks"]["room_level_target_plausible"] is True
    assert row["support_object_evidence"] is None
    assert row["target_object_evidence"]["object_id"] == "cabinet_001"
    assert report["summary"]["trusted_case_count"] == 1
    assert report["summary"]["explicit_trusted_relation_count"] == 0
    assert report["summary"]["trusted_room_level_count"] == 1


def test_vlm_graph_conflict_report_counts_outcomes_and_opportunities() -> None:
    cases = [
        _object_location_case("case-vlm-wrong-graph-correct", "laptop_001", 1),
        _object_location_case("case-vlm-correct-graph-wrong", "bread_001", 1),
        _object_location_case("case-vlm-unknown-graph-plausible", "book_001", 1),
    ]
    vlm_predictions = [
        lab.QAPrediction(id=cases[0].id, answer={"text": "on the table"}, confidence=0.8),
        lab.QAPrediction(id=cases[1].id, answer={"text": "on the countertop"}, confidence=0.8),
        lab.QAPrediction(id=cases[2].id, answer={"text": "unknown"}, confidence=0.1),
    ]
    graph_predictions = [
        lab.QAPrediction(
            id=cases[0].id,
            answer={"current_location": {"dst": "chair_001", "relation": "ON"}},
            confidence=0.9,
        ),
        lab.QAPrediction(
            id=cases[1].id,
            answer={"current_location": {"dst": "egg_001", "relation": "ON"}},
            confidence=0.9,
        ),
        lab.QAPrediction(
            id=cases[2].id,
            answer={"current_location": {"dst": "shelf_001", "relation": "ON"}},
            confidence=0.9,
        ),
    ]
    score_report = {
        "schema_version": lab.VLM_GRAPH_EVIDENCE_SCORE_REPORT_SCHEMA_VERSION,
        "cases": [
            {"case_id": cases[0].id, "graph_trust_score": 0.9, "trusted": True},
            {"case_id": cases[1].id, "graph_trust_score": 0.2, "trusted": False},
            {"case_id": cases[2].id, "graph_trust_score": 0.85, "trusted": True},
        ],
        "report_digest": "0" * 64,
    }
    vlm_report = _semantic_report(
        [
            (cases[0].id, False),
            (cases[1].id, True),
            (cases[2].id, False),
        ]
    )
    graph_report = _semantic_report(
        [
            (cases[0].id, True),
            (cases[1].id, False),
            (cases[2].id, True),
        ]
    )

    report = lab.vlm_graph_conflict_report(
        cases,
        vlm_predictions,
        graph_predictions,
        vlm_report,
        graph_report,
        score_report,
    )

    assert report["summary"]["vlm_wrong_graph_correct_count"] == 2
    assert report["summary"]["vlm_correct_graph_wrong_count"] == 1
    assert report["summary"]["vlm_unknown_graph_plausible_count"] == 1
    assert report["opportunity_case_ids"] == [
        "case-vlm-unknown-graph-plausible",
        "case-vlm-wrong-graph-correct",
    ]
    assert lab.validate_vlm_graph_conflict_report(report)["valid"] is True


def test_vlm_graph_evidence_request_bundle_keeps_evidence_without_gold() -> None:
    case = _object_location_case(
        "episode:scene:0001:object_location:laptop_001",
        "laptop_001",
        42,
    )
    vlm_prediction = lab.QAPrediction(
        id=case.id,
        answer={"text": "unknown"},
        confidence=0.1,
    )
    graph_prediction = lab.QAPrediction(
        id=f"{case.id}:observation_aware:42",
        answer={
            "current_location": {"dst": "chair_001", "relation": "ON", "step": 42},
            "label": "laptop",
            "object_id": "laptop_001",
        },
        confidence=0.9,
    )
    score_report = {
        "schema_version": lab.VLM_GRAPH_EVIDENCE_SCORE_REPORT_SCHEMA_VERSION,
        "cases": [
            {
                "case_id": case.id,
                "checks": {"support_label_plausible": True},
                "dsg_candidate": {
                    "current_location": {"dst": "chair_001", "relation": "ON", "step": 42},
                    "support_label": "chair",
                    "target_label": "laptop",
                },
                "graph_trust_score": 0.9,
                "reject_reason": None,
                "trusted": True,
            }
        ],
        "report_digest": "1" * 64,
    }
    conflict_report = {
        "schema_version": lab.VLM_GRAPH_CONFLICT_REPORT_SCHEMA_VERSION,
        "cases": [
            {
                "case_id": case.id,
                "conflict_reason": "vlm_unknown_graph_plausible",
                "needs_vlm_dsg_adjudication": True,
            }
        ],
        "report_digest": "2" * 64,
    }
    detector_records = [
        _detector_record("episode", "scene", 42, "laptop_001", "laptop"),
        _detector_record("episode", "scene", 42, "chair_001", "chair"),
    ]

    bundle = lab.vlm_graph_evidence_request_bundle(
        [case],
        [vlm_prediction],
        [graph_prediction],
        score_report,
        conflict_report,
        detector_records=detector_records,
    )

    serialized = json.dumps(bundle, sort_keys=True)
    request_case = bundle["case_inputs"][0]
    assert request_case["case_id"] == case.id
    assert request_case["question"] == case.question
    assert request_case["vlm_initial_answer"] == vlm_prediction.answer
    assert request_case["dsg_candidate"]["current_location"]["dst"] == "chair_001"
    assert request_case["frame_refs"] == [
        {"depth_path": "depth/0042.npy", "rgb_path": "frames/0042.png", "step": 42}
    ]
    assert request_case["crop_refs"] == [
        {
            "bbox_2d_xyxy": [1, 2, 3, 4],
            "object_id": "laptop_001",
            "rgb_path": "frames/0042.png",
            "role": "target",
            "step": 42,
        },
        {
            "bbox_2d_xyxy": [1, 2, 3, 4],
            "object_id": "chair_001",
            "rgb_path": "frames/0042.png",
            "role": "support",
            "step": 42,
        },
    ]
    assert request_case["support_object_evidence"]["object_id"] == "chair_001"
    assert request_case["graph_trust_score"] == 0.9
    assert "gold" not in serialized
    assert "required_nodes" not in serialized
    assert "required_edges" not in serialized
    assert "semantic_match" not in serialized
    assert lab.validate_vlm_graph_evidence_request_bundle(bundle)["valid"] is True


def test_build_vlm_graph_evidence_script_merges_repeated_detector_jsonl(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    qa_path = tmp_path / "qa.jsonl"
    vlm_path = tmp_path / "vlm.jsonl"
    graph_path = tmp_path / "graph.jsonl"
    vlm_report_path = tmp_path / "vlm-report.json"
    graph_report_path = tmp_path / "graph-report.json"
    target_detector_path = tmp_path / "target-detector.jsonl"
    support_detector_path = tmp_path / "support-detector.jsonl"
    score_path = tmp_path / "score.json"
    conflict_path = tmp_path / "conflict.json"
    bundle_path = tmp_path / "bundle.json"
    case = _object_location_case(
        "episode:scene:0001:object_location:laptop_001",
        "laptop_001",
        42,
    )
    lab.save_qa_dataset([case], qa_path)
    lab.save_qa_predictions(
        [lab.QAPrediction(id=case.id, answer={"text": "unknown"}, confidence=0.1)],
        vlm_path,
    )
    lab.save_qa_predictions(
        [
            lab.QAPrediction(
                id=f"{case.id}:observation_aware:42",
                answer={
                    "current_location": {"dst": "chair_001", "relation": "ON", "step": 42},
                    "label": "laptop",
                    "object_id": "laptop_001",
                },
                confidence=0.9,
            )
        ],
        graph_path,
    )
    vlm_report_path.write_text(
        json.dumps(_semantic_report([(case.id, False)])),
        encoding="utf-8",
    )
    graph_report_path.write_text(
        json.dumps(_semantic_report([(case.id, True)])),
        encoding="utf-8",
    )
    target_detector_path.write_text(
        json.dumps(_detector_observation_record("episode", "scene", 42, "laptop_001", "laptop"))
        + "\n",
        encoding="utf-8",
    )
    support_detector_path.write_text(
        json.dumps(_detector_observation_record("episode", "scene", 42, "chair_001", "chair"))
        + "\n",
        encoding="utf-8",
    )

    module = load_evidence_script()
    exit_code = module.main(
        [
            "--qa",
            str(qa_path),
            "--vlm-predictions",
            str(vlm_path),
            "--graph-predictions",
            str(graph_path),
            "--vlm-semantic-report",
            str(vlm_report_path),
            "--graph-semantic-report",
            str(graph_report_path),
            "--detector-jsonl",
            str(target_detector_path),
            "--detector-jsonl",
            str(support_detector_path),
            "--score-report",
            str(score_path),
            "--conflict-report",
            str(conflict_path),
            "--request-bundle",
            str(bundle_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    score_report = json.loads(score_path.read_text(encoding="utf-8"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["detector_record_count"] == 2
    assert score_report["summary"]["trusted_case_count"] == 1
    assert bundle["case_count"] == 1


def test_evaluate_vlm_graph_adjudication_script_reports_missing_predictions(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    qa_path = tmp_path / "qa.jsonl"
    bundle_path = tmp_path / "bundle.json"
    vlm_predictions_path = tmp_path / "vlm.jsonl"
    missing_adjudicated_path = tmp_path / "missing-p46.jsonl"
    vlm_report_path = tmp_path / "vlm-report.json"
    graph_report_path = tmp_path / "graph-report.json"
    report_path = tmp_path / "comparison.json"
    case = _object_location_case("episode:scene:0001:object_location:laptop_001", "laptop_001", 42)
    lab.save_qa_dataset([case], qa_path)
    vlm_predictions = [lab.QAPrediction(id=case.id, answer={"text": "unknown"}, confidence=0.1)]
    lab.save_qa_predictions(vlm_predictions, vlm_predictions_path)
    _write_json(
        bundle_path,
        {
            "schema_version": lab.VLM_GRAPH_EVIDENCE_REQUEST_BUNDLE_SCHEMA_VERSION,
            "case_count": 1,
            "case_inputs": [{"case_id": case.id, "question": case.question}],
            "forbidden_fields_absent": True,
        },
    )
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["request_bundle_digest"] = lab.vlm_graph_evidence_request_bundle_digest(bundle)
    _write_json(bundle_path, bundle)
    vlm_report = lab.vlm_semantic_eval_report([case], vlm_predictions)
    graph_report = lab.vlm_semantic_eval_report(
        [case],
        [
            lab.QAPrediction(
                id=case.id,
                answer={"current_location": {"dst": "countertop_001", "relation": "ON"}},
                confidence=0.9,
            )
        ],
    )
    lab.save_vlm_semantic_eval_report(vlm_report, vlm_report_path)
    lab.save_vlm_semantic_eval_report(graph_report, graph_report_path)

    module = load_adjudication_script()
    exit_code = module.main(
        [
            "--qa",
            str(qa_path),
            "--request-bundle",
            str(bundle_path),
            "--vlm-predictions",
            str(vlm_predictions_path),
            "--adjudicated-predictions",
            str(missing_adjudicated_path),
            "--vlm-semantic-report",
            str(vlm_report_path),
            "--graph-semantic-report",
            str(graph_report_path),
            "--comparison-report",
            str(report_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ready"] is False
    assert payload["research_ready"] is False
    assert payload["blockers"] == ["missing_adjudicated_predictions"]
    assert payload["next_missing_artifacts"] == [str(missing_adjudicated_path)]
    assert payload["final_record_written"] is False
    assert report_path.exists() is False


def test_evaluate_vlm_graph_adjudication_script_rejects_unstructured_adjudication(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    qa_path = tmp_path / "qa.jsonl"
    bundle_path = tmp_path / "bundle.json"
    vlm_predictions_path = tmp_path / "vlm.jsonl"
    adjudicated_predictions_path = tmp_path / "bad-p46.jsonl"
    vlm_report_path = tmp_path / "vlm-report.json"
    graph_report_path = tmp_path / "graph-report.json"
    report_path = tmp_path / "comparison.json"
    case = _object_location_case(
        "episode:scene:0001:object_location:laptop_001",
        "laptop_001",
        42,
    )
    lab.save_qa_dataset([case], qa_path)
    vlm_predictions = [lab.QAPrediction(id=case.id, answer={"text": "unknown"}, confidence=0.1)]
    lab.save_qa_predictions(vlm_predictions, vlm_predictions_path)
    lab.save_qa_predictions(
        [lab.QAPrediction(id=case.id, answer={"text": "on the countertop"}, confidence=0.8)],
        adjudicated_predictions_path,
    )
    bundle = {
        "schema_version": lab.VLM_GRAPH_EVIDENCE_REQUEST_BUNDLE_SCHEMA_VERSION,
        "case_count": 1,
        "case_inputs": [{"case_id": case.id, "question": case.question}],
        "forbidden_fields_absent": True,
    }
    bundle["request_bundle_digest"] = lab.vlm_graph_evidence_request_bundle_digest(bundle)
    _write_json(bundle_path, bundle)
    lab.save_vlm_semantic_eval_report(
        lab.vlm_semantic_eval_report([case], vlm_predictions),
        vlm_report_path,
    )
    lab.save_vlm_semantic_eval_report(
        lab.vlm_semantic_eval_report(
            [case],
            [
                lab.QAPrediction(
                    id=case.id,
                    answer={"current_location": {"dst": "countertop_001", "relation": "ON"}},
                    confidence=0.9,
                )
            ],
        ),
        graph_report_path,
    )

    module = load_adjudication_script()
    exit_code = module.main(
        [
            "--qa",
            str(qa_path),
            "--request-bundle",
            str(bundle_path),
            "--vlm-predictions",
            str(vlm_predictions_path),
            "--adjudicated-predictions",
            str(adjudicated_predictions_path),
            "--vlm-semantic-report",
            str(vlm_report_path),
            "--graph-semantic-report",
            str(graph_report_path),
            "--comparison-report",
            str(report_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ready"] is False
    assert payload["research_ready"] is False
    assert payload["blockers"] == ["invalid_adjudicated_prediction_schema"]
    assert payload["invalid_adjudicated_case_ids"] == [case.id]
    assert payload["final_record_written"] is False
    assert report_path.exists() is False


def test_evaluate_vlm_graph_adjudication_script_builds_three_way_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    qa_path = tmp_path / "qa.jsonl"
    bundle_path = tmp_path / "bundle.json"
    vlm_predictions_path = tmp_path / "vlm.jsonl"
    adjudicated_predictions_path = tmp_path / "p46.jsonl"
    vlm_report_path = tmp_path / "vlm-report.json"
    graph_report_path = tmp_path / "graph-report.json"
    merged_path = tmp_path / "merged.jsonl"
    adjudicated_semantic_path = tmp_path / "adjudicated-semantic.json"
    delta_vs_vlm_path = tmp_path / "delta-vs-vlm.json"
    delta_vs_graph_path = tmp_path / "delta-vs-graph.json"
    comparison_path = tmp_path / "comparison.json"
    case_a = _object_location_case(
        "episode:scene:0001:object_location:laptop_001",
        "laptop_001",
        42,
    )
    case_b = _object_location_case(
        "episode:scene:0002:object_location:book_001",
        "book_001",
        43,
    )
    lab.save_qa_dataset([case_a, case_b], qa_path)
    vlm_predictions = [
        lab.QAPrediction(id=case_a.id, answer={"text": "unknown"}, confidence=0.1),
        lab.QAPrediction(id=case_b.id, answer={"text": "on the countertop"}, confidence=0.8),
    ]
    graph_predictions = [
        lab.QAPrediction(
            id=case_a.id,
            answer={"current_location": {"dst": "countertop_001", "relation": "ON"}},
            confidence=0.9,
        ),
        lab.QAPrediction(
            id=case_b.id,
            answer={"current_location": {"dst": "countertop_001", "relation": "ON"}},
            confidence=0.9,
        ),
    ]
    adjudicated_predictions = [
        lab.QAPrediction(
            id=case_a.id,
            answer={
                "current_location": {"dst": "countertop_001", "relation": "ON"},
                "decision": "accept_dsg",
                "evidence_summary": "DSG relation is supported by RGB-D evidence.",
            },
            confidence=0.85,
        )
    ]
    lab.save_qa_predictions(vlm_predictions, vlm_predictions_path)
    lab.save_qa_predictions(adjudicated_predictions, adjudicated_predictions_path)
    bundle = {
        "schema_version": lab.VLM_GRAPH_EVIDENCE_REQUEST_BUNDLE_SCHEMA_VERSION,
        "case_count": 1,
        "case_inputs": [{"case_id": case_a.id, "question": case_a.question}],
        "forbidden_fields_absent": True,
    }
    bundle["request_bundle_digest"] = lab.vlm_graph_evidence_request_bundle_digest(bundle)
    _write_json(bundle_path, bundle)
    lab.save_vlm_semantic_eval_report(
        lab.vlm_semantic_eval_report([case_a, case_b], vlm_predictions),
        vlm_report_path,
    )
    lab.save_vlm_semantic_eval_report(
        lab.vlm_semantic_eval_report([case_a, case_b], graph_predictions),
        graph_report_path,
    )

    module = load_adjudication_script()
    exit_code = module.main(
        [
            "--qa",
            str(qa_path),
            "--request-bundle",
            str(bundle_path),
            "--vlm-predictions",
            str(vlm_predictions_path),
            "--adjudicated-predictions",
            str(adjudicated_predictions_path),
            "--vlm-semantic-report",
            str(vlm_report_path),
            "--graph-semantic-report",
            str(graph_report_path),
            "--merged-predictions-output",
            str(merged_path),
            "--adjudicated-semantic-report",
            str(adjudicated_semantic_path),
            "--delta-vs-vlm",
            str(delta_vs_vlm_path),
            "--delta-vs-graph",
            str(delta_vs_graph_path),
            "--comparison-report",
            str(comparison_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    merged_predictions = lab.load_qa_predictions(merged_path)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["ready"] is True
    assert payload["research_ready"] is False
    assert payload["summary"]["adjudicated_request_case_count"] == 1
    assert payload["summary"]["fallback_vlm_case_count"] == 1
    assert [prediction.answer["adjudication"]["source"] for prediction in merged_predictions] == [
        "vlm_graph_adjudication",
        "vlm_fallback",
    ]
    assert comparison["summary"]["semantic_match_count"] == 2
    assert comparison["deltas"]["adjudicated_vlm_graph_vs_vlm_only"]["summary_delta"][
        "semantic_match_count_delta"
    ] == 1
    assert comparison["deltas"]["adjudicated_vlm_graph_vs_graph_tool_only"]["summary_delta"][
        "semantic_match_count_delta"
    ] == 0
    assert comparison["final_record_written"] is False
    assert lab.validate_vlm_semantic_eval_delta_report(
        json.loads(delta_vs_vlm_path.read_text(encoding="utf-8"))
    )["valid"] is True


def _object_location_case(case_id: str, object_id: str, step: int) -> lab.QACase:
    return lab.QACase(
        id=case_id,
        scene_id="scene",
        episode_id="episode",
        graph_digest="g" * 64,
        step=step,
        question={"type": "object_location", "object_id": object_id},
        question_type="object_location",
        answer={
            "current_location": {"dst": "countertop_001", "relation": "ON", "step": step},
            "label": _label_from_object_id(object_id),
            "object_id": object_id,
            "visible": True,
        },
        answer_type="object_location",
        required_nodes=("secret_gold_node",),
        required_edges=("secret_gold_edge",),
        tags=("test",),
    )


def _label_from_object_id(object_id: str) -> str:
    return object_id.split("_", maxsplit=1)[0]


def _detector_record(
    episode_id: str,
    scene_id: str,
    step: int,
    object_id: str,
    label: str,
) -> dict[str, object]:
    return {
        "schema_version": "dsg-spatialqa-lab.external-detector-frame.v1",
        "episode_id": episode_id,
        "scene_id": scene_id,
        "step": step,
        "rgb_path": f"frames/{step:04d}.png",
        "depth_path": f"depth/{step:04d}.npy",
        "detector_name": "unit_detector",
        "detections": [
            {
                "detection_id": f"{object_id}-det",
                "object_id": object_id,
                "label": label,
                "confidence": 0.9,
                "bbox_2d_xyxy": [1, 2, 3, 4],
                "bbox_3d_center": {"x": 1.0, "y": 1.0, "z": 1.0, "yaw": 0.0},
                "bbox_3d_size": [0.1, 0.2, 0.3],
                "visible": True,
                "evidence_kinds": ["rgb", "depth", "detector"],
            }
        ],
    }


def _detector_observation_record(
    episode_id: str,
    scene_id: str,
    step: int,
    object_id: str,
    label: str,
) -> dict[str, object]:
    return {
        "schema_version": "dsg-spatialqa-lab.detector-observation-record.v1",
        "episode_id": episode_id,
        "scene_id": scene_id,
        "step": step,
        "rgb_path": f"frames/{step:04d}.png",
        "depth_path": f"depth/{step:04d}.npy",
        "metadata": {
            "source_kind": "detector",
            "source_name": "visible_segmentation_rgbd",
        },
        "detections": [
            {
                "object_id": object_id,
                "label": label,
                "confidence": 0.9,
                "bbox_2d_xyxy": [1, 2, 3, 4],
                "bbox": {
                    "center": {"x": 1.0, "y": 1.0, "z": 1.0, "yaw": 0.0},
                    "size": [0.1, 0.2, 0.3],
                },
                "visible": True,
                "attributes": {
                    "evidence_kinds": ["rgb", "depth", "detector"],
                    "source_kind": "detector",
                    "source_name": "visible_segmentation_rgbd",
                },
            }
        ],
    }


def _semantic_report(rows: list[tuple[str, bool]]) -> dict[str, object]:
    return {
        "schema_version": "dsg-spatialqa-lab.vlm-semantic-eval-report.v1",
        "cases": [
            {
                "case_id": case_id,
                "failure_reason": None if matched else "semantic_mismatch",
                "semantic_match": matched,
                "strict_exact_match": False,
            }
            for case_id, matched in rows
        ],
        "gold_digest": "a" * 64,
        "prediction_digest": "b" * 64,
        "report_digest": "c" * 64,
        "summary": {
            "case_count": len(rows),
            "semantic_match_count": sum(1 for _, matched in rows if matched),
            "semantic_match_rate": 0.0,
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
