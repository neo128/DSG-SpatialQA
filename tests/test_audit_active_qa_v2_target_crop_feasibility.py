from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from _pytest.capture import CaptureFixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_active_qa_v2_target_crop_feasibility.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "audit_active_qa_v2_target_crop_feasibility_test_script",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_target_crop_feasibility_audit_finds_bbox_and_mask_evidence(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    rgb_path = tmp_path / "frames" / "rgb.ppm"
    mask_path = tmp_path / "frames" / "mug-mask.ppm"
    segmentation_path = tmp_path / "frames" / "segmentation.ppm"
    _write_ppm(rgb_path)
    _write_ppm(mask_path)
    _write_ppm(segmentation_path)
    bundle = tmp_path / "request-bundle.json"
    _write_bundle(
        bundle,
        [
            _case(
                "episode-001:FloorPlan1:1:support_relation:mug_1:relation_centric",
                rgb_path,
                target={"label": "mug", "object_id": "mug_1"},
            )
        ],
    )
    observations = tmp_path / "observations.json"
    _write_observations(
        observations,
        [
            _observation(
                step=1,
                object_id="mug_1",
                rgb_path=rgb_path,
                segmentation_path=segmentation_path,
                attributes={
                    "bbox_2d_xyxy": [1, 2, 10, 12],
                    "mask_path": str(mask_path),
                    "segmentation_color_rgb": [10, 20, 30],
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
            "--observation-sequence",
            str(observations),
            "--report",
            str(report),
        ]
    )

    emitted = json.loads(capsys.readouterr().out)
    audited = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert emitted["ready"] is True
    assert audited["ready"] is True
    assert audited["crop_generation_ready"] is True
    assert audited["summary"]["request_count"] == 1
    assert audited["summary"]["cases_with_target_id"] == 1
    assert audited["summary"]["cases_with_matching_observation_object"] == 1
    assert audited["summary"]["cases_with_bbox_2d"] == 1
    assert audited["summary"]["cases_with_existing_mask_path"] == 1
    assert audited["summary"]["feasible_target_crop_case_count"] == 1
    assert audited["blockers"] == []


def test_target_crop_feasibility_audit_reports_only_3d_bbox_as_infeasible(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    rgb_path = tmp_path / "frames" / "rgb.ppm"
    segmentation_path = tmp_path / "frames" / "segmentation.ppm"
    _write_ppm(rgb_path)
    _write_ppm(segmentation_path)
    bundle = tmp_path / "request-bundle.json"
    _write_bundle(
        bundle,
        [
            _case(
                "episode-001:FloorPlan1:1:support_relation:mug_1:relation_centric",
                rgb_path,
                target={"label": "mug"},
            )
        ],
    )
    observations = tmp_path / "observations.json"
    _write_observations(
        observations,
        [
            _observation(
                step=1,
                object_id="mug_1",
                rgb_path=rgb_path,
                segmentation_path=segmentation_path,
                attributes={},
            )
        ],
    )
    report = tmp_path / "report.json"

    module = load_script()
    exit_code = module.main(
        [
            "--request-bundle",
            str(bundle),
            "--observation-sequence",
            str(observations),
            "--report",
            str(report),
        ]
    )

    emitted = json.loads(capsys.readouterr().out)
    audited = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert emitted["ready"] is True
    assert audited["ready"] is True
    assert audited["crop_generation_ready"] is False
    assert audited["blockers"] == ["target_crop_artifacts_missing"]
    assert audited["summary"]["cases_with_target_id"] == 1
    assert audited["summary"]["cases_with_matching_observation_object"] == 1
    assert audited["summary"]["cases_with_existing_rgb_path"] == 1
    assert audited["summary"]["cases_with_segmentation_path"] == 1
    assert audited["summary"]["cases_with_bbox_3d_only"] == 1
    assert audited["summary"]["cases_with_bbox_2d"] == 0
    assert audited["summary"]["feasible_target_crop_case_count"] == 0
    assert audited["infeasible_reasons"] == {
        "missing_bbox_2d_mask_or_segmentation_color": 1
    }


def _case(case_id: str, frame: Path, *, target: dict[str, str]) -> dict[str, object]:
    return {
        "case_id": case_id,
        "episode_id": "episode-001",
        "frames": [{"rgb_path": str(frame), "step": 1}],
        "primary_frame": {"rgb_path": str(frame), "step": 1},
        "question_task_hint": "Use visible evidence.",
        "question_text": "What is the mug on?",
        "question_type": "support_relation",
        "target": target,
    }


def _write_bundle(path: Path, cases: list[dict[str, object]]) -> None:
    payload = {
        "schema_version": "dsg-spatialqa-lab.active-qa-v2-vlm-request-bundle.v1",
        "leak_free": True,
        "leak_paths": [],
        "prediction_cases": cases,
        "request_count": len(cases),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _observation(
    *,
    step: int,
    object_id: str,
    rgb_path: Path,
    segmentation_path: Path,
    attributes: dict[str, object],
) -> dict[str, object]:
    merged_attributes: dict[str, object] = {
        "rgb_path": str(rgb_path),
        "segmentation_path": str(segmentation_path),
        **attributes,
    }
    return {
        "step": step,
        "agent_pose": {"x": 0.0, "y": 0.9, "z": 0.0, "yaw": 0.0},
        "objects": [
            {
                "object_id": object_id,
                "label": "mug",
                "visible": True,
                "bbox": {
                    "center": {"x": 0.0, "y": 0.9, "z": 1.0, "yaw": 0.0},
                    "size": [0.1, 0.1, 0.1],
                },
                "attributes": merged_attributes,
            }
        ],
    }


def _write_observations(path: Path, observations: list[dict[str, object]]) -> None:
    payload = {
        "schema_version": "dsg-spatialqa-lab.scene-observation-sequence.v1",
        "observation_count": len(observations),
        "observations": observations,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_ppm(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
