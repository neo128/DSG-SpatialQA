from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from _pytest.capture import CaptureFixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "enrich_active_qa_v2_request_bundle_with_target_crops.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "enrich_active_qa_v2_request_bundle_with_target_crops_test_script",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_enriches_feasible_case_with_segmentation_color_crop(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    rgb_path = tmp_path / "frames" / "rgb.ppm"
    segmentation_path = tmp_path / "frames" / "segmentation.ppm"
    _write_ppm(
        rgb_path,
        width=4,
        height=4,
        pixels=[
            (0, 0, 0),
            (10, 0, 0),
            (20, 0, 0),
            (30, 0, 0),
            (0, 10, 0),
            (10, 10, 0),
            (20, 10, 0),
            (30, 10, 0),
            (0, 20, 0),
            (10, 20, 0),
            (20, 20, 0),
            (30, 20, 0),
            (0, 30, 0),
            (10, 30, 0),
            (20, 30, 0),
            (30, 30, 0),
        ],
    )
    _write_ppm(
        segmentation_path,
        width=4,
        height=4,
        pixels=[
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (7, 8, 9),
            (7, 8, 9),
            (0, 0, 0),
            (0, 0, 0),
            (7, 8, 9),
            (7, 8, 9),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
        ],
    )
    request_bundle = tmp_path / "request-bundle.json"
    _write_bundle(
        request_bundle,
        [
            _case(
                case_id=(
                    "ai2thor-real-small-episode-006:FloorPlan201:200001:"
                    "nearest_object:apple_1:relation_centric"
                ),
                rgb_path=rgb_path,
            )
        ],
    )
    observation_sequence = tmp_path / "observations.json"
    _write_observation_sequence(
        observation_sequence,
        rgb_path=rgb_path,
        segmentation_path=segmentation_path,
        color=[7, 8, 9],
    )

    module = load_script()
    output_bundle = tmp_path / "enriched-bundle.json"
    report = tmp_path / "report.json"
    exit_code = module.main(
        [
            "--request-bundle",
            str(request_bundle),
            "--observation-sequence",
            str(observation_sequence),
            "--crop-root",
            str(tmp_path / "crops"),
            "--output",
            str(output_bundle),
            "--report",
            str(report),
            "--padding-pixels",
            "0",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    enriched = json.loads(output_bundle.read_text(encoding="utf-8"))
    audited = json.loads(report.read_text(encoding="utf-8"))
    case = enriched["prediction_cases"][0]
    target_crop = case["target_crop"]
    crop_path = Path(target_crop["rgb_path"])
    assert exit_code == 0
    assert payload["ready"] is True
    assert payload["enriched_target_crop_count"] == 1
    assert enriched["leak_free"] is True
    assert enriched["request_count"] == 1
    assert enriched["target_crop_enrichment"]["summary"]["cases_with_target_crop"] == 1
    assert target_crop["bbox_2d_xyxy"] == [1, 1, 3, 3]
    assert target_crop["crop_kind"] == "segmentation_color_mask"
    assert target_crop["object_id"] == "apple_1"
    assert crop_path.exists()
    assert _ppm_dimensions(crop_path) == (2, 2)
    assert audited["summary"]["cases_with_target_crop"] == 1
    assert audited["blockers"] == []
    assert "gold" not in json.dumps(enriched, sort_keys=True)
    assert "required_edges" not in json.dumps(enriched, sort_keys=True)
    assert "required_nodes" not in json.dumps(enriched, sort_keys=True)


def test_reports_partial_when_segmentation_color_is_not_visible(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    rgb_path = tmp_path / "frames" / "rgb.ppm"
    segmentation_path = tmp_path / "frames" / "segmentation.ppm"
    _write_ppm(rgb_path, width=1, height=1, pixels=[(1, 2, 3)])
    _write_ppm(segmentation_path, width=1, height=1, pixels=[(0, 0, 0)])
    request_bundle = tmp_path / "request-bundle.json"
    _write_bundle(
        request_bundle,
        [
            _case(
                case_id=(
                    "ai2thor-real-small-episode-006:FloorPlan201:200001:"
                    "nearest_object:apple_1:relation_centric"
                ),
                rgb_path=rgb_path,
            )
        ],
    )
    observation_sequence = tmp_path / "observations.json"
    _write_observation_sequence(
        observation_sequence,
        rgb_path=rgb_path,
        segmentation_path=segmentation_path,
        color=[7, 8, 9],
    )

    module = load_script()
    exit_code = module.main(
        [
            "--request-bundle",
            str(request_bundle),
            "--observation-sequence",
            str(observation_sequence),
            "--crop-root",
            str(tmp_path / "crops"),
            "--output",
            str(tmp_path / "enriched-bundle.json"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ready"] is False
    assert payload["blockers"] == ["target_crop_artifacts_missing"]
    assert payload["infeasible_reasons"] == {"segmentation_color_not_found": 1}


def test_adds_no_crop_visual_context_without_detector_id_leak(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    rgb_path = tmp_path / "frames" / "rgb.ppm"
    segmentation_path = tmp_path / "frames" / "segmentation.ppm"
    _write_ppm(rgb_path, width=1, height=1, pixels=[(1, 2, 3)])
    _write_ppm(segmentation_path, width=1, height=1, pixels=[(0, 0, 0)])
    request_bundle = tmp_path / "request-bundle.json"
    _write_bundle(
        request_bundle,
        [
            _case(
                case_id=(
                    "ai2thor-real-small-episode-006:FloorPlan201:200001:"
                    "nearest_object:apple_1:relation_centric"
                ),
                rgb_path=rgb_path,
            )
        ],
    )
    observation_sequence = tmp_path / "observations.json"
    _write_observation_sequence(
        observation_sequence,
        rgb_path=rgb_path,
        segmentation_path=segmentation_path,
        color=[7, 8, 9],
    )

    module = load_script()
    output_bundle = tmp_path / "enriched-bundle.json"
    exit_code = module.main(
        [
            "--request-bundle",
            str(request_bundle),
            "--observation-sequence",
            str(observation_sequence),
            "--crop-root",
            str(tmp_path / "crops"),
            "--output",
            str(output_bundle),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )

    _ = capsys.readouterr()
    enriched = json.loads(output_bundle.read_text(encoding="utf-8"))
    case = enriched["prediction_cases"][0]
    context = case["target_visual_context"]
    serialized_context = json.dumps(context, sort_keys=True)
    assert exit_code == 1
    assert context == {
        "available": True,
        "context_kind": "primary_frame_without_target_crop",
        "instruction": (
            "No local target crop is available. Inspect only the primary RGB "
            "frame; if the target is not visually clear, return "
            "target_not_observed instead of guessing."
        ),
        "primary_frame_role": "primary_frame",
        "target_crop_available": False,
        "target_crop_unavailable_reason": "segmentation_color_not_found",
        "target_label": "apple",
    }
    assert "apple_1" not in serialized_context
    assert "AI2Thor" not in serialized_context
    assert "visible_object" not in serialized_context


def test_uses_latest_duplicate_observation_object_with_segmentation_color(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    rgb_path = tmp_path / "frames" / "rgb.ppm"
    segmentation_path = tmp_path / "frames" / "segmentation.ppm"
    _write_ppm(rgb_path, width=1, height=1, pixels=[(1, 2, 3)])
    _write_ppm(segmentation_path, width=1, height=1, pixels=[(7, 8, 9)])
    request_bundle = tmp_path / "request-bundle.json"
    _write_bundle(
        request_bundle,
        [
            _case(
                case_id=(
                    "ai2thor-real-small-episode-006:FloorPlan201:200001:"
                    "nearest_object:apple_1:relation_centric"
                ),
                rgb_path=rgb_path,
            )
        ],
    )
    observation_sequence = tmp_path / "observations.json"
    _write_observation_sequence(
        observation_sequence,
        rgb_path=rgb_path,
        segmentation_path=segmentation_path,
        color=[7, 8, 9],
        include_uncolored_duplicate=True,
    )

    module = load_script()
    exit_code = module.main(
        [
            "--request-bundle",
            str(request_bundle),
            "--observation-sequence",
            str(observation_sequence),
            "--crop-root",
            str(tmp_path / "crops"),
            "--output",
            str(tmp_path / "enriched-bundle.json"),
            "--report",
            str(tmp_path / "report.json"),
            "--padding-pixels",
            "0",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ready"] is True
    assert payload["enriched_target_crop_count"] == 1


def _case(*, case_id: str, rgb_path: Path) -> dict[str, object]:
    return {
        "case_id": case_id,
        "episode_id": "ai2thor-real-small-episode-006",
        "frames": [{"rgb_path": str(rgb_path), "step": 200001}],
        "leak_free": True,
        "primary_frame": {
            "frame_id": "200001",
            "rgb_path": str(rgb_path),
            "scene_id": "FloorPlan201",
            "step": 200001,
        },
        "question_task_hint": "Use the primary frame and target crop.",
        "question_text": "Which observed object is nearest to the apple?",
        "question_type": "nearest_object",
        "target": {"label": "apple"},
    }


def _write_bundle(path: Path, cases: list[dict[str, object]]) -> None:
    payload = {
        "leak_free": True,
        "leak_paths": [],
        "prediction_cases": cases,
        "request_count": len(cases),
        "schema_version": "dsg-spatialqa-lab.active-qa-v2-vlm-request-bundle.v1",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_observation_sequence(
    path: Path,
    *,
    rgb_path: Path,
    segmentation_path: Path,
    color: list[int],
    include_uncolored_duplicate: bool = False,
) -> None:
    objects: list[dict[str, object]] = []
    if include_uncolored_duplicate:
        objects.append(
            {
                "attributes": {
                    "rgb_path": str(rgb_path),
                    "segmentation_path": str(segmentation_path),
                },
                "object_id": "apple_1",
            }
        )
    objects.append(
        {
            "attributes": {
                "rgb_path": str(rgb_path),
                "segmentation_color_rgb": color,
                "segmentation_path": str(segmentation_path),
            },
            "object_id": "apple_1",
        }
    )
    payload = {
        "observations": [
            {
                "objects": objects,
                "step": 200001,
            }
        ],
        "schema_version": "dsg-spatialqa-lab.scene-observation-sequence.v1",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_ppm(
    path: Path,
    *,
    width: int,
    height: int,
    pixels: list[tuple[int, int, int]],
) -> None:
    assert len(pixels) == width * height
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bytes(channel for pixel in pixels for channel in pixel)
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + payload)


def _ppm_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    parts = data.split(maxsplit=4)
    return int(parts[1]), int(parts[2])
