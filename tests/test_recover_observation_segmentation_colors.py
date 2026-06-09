from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from _pytest.capture import CaptureFixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "recover_observation_segmentation_colors.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "recover_observation_segmentation_colors_test_script",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recovers_segmentation_colors_by_exact_ai2thor_object_id(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    episode = tmp_path / "episode-001.jsonl"
    _write_episode(
        episode,
        episode_id="episode-001",
        color_map=[
            {"object_id": "Mug|+00.10|+00.90|+01.20", "rgb": [10, 20, 30]},
            {"object_id": "Plate|+00.30|+00.90|+01.30", "rgb": [40, 50, 60]},
        ],
    )
    observations = tmp_path / "observations.json"
    _write_observations(
        observations,
        episode_id="episode-001",
        objects=[
            _object("mug_1", "Mug|+00.10|+00.90|+01.20"),
            _object("plate_1", "Plate|+00.30|+00.90|+01.30"),
        ],
    )
    output_root = tmp_path / "out"
    report = tmp_path / "report.json"

    module = load_script()
    exit_code = module.main(
        [
            "--episode-jsonl",
            str(episode),
            "--observation-sequence",
            str(observations),
            "--output-root",
            str(output_root),
            "--report",
            str(report),
        ]
    )

    emitted = json.loads(capsys.readouterr().out)
    audited = json.loads(report.read_text(encoding="utf-8"))
    enriched = json.loads(
        Path(audited["enriched_observation_sequences"][0]["output_path"]).read_text(
            encoding="utf-8"
        )
    )
    attrs = [
        obj["attributes"]
        for obj in enriched["observations"][0]["objects"]
    ]
    assert exit_code == 0
    assert emitted["ready"] is True
    assert audited["summary"]["object_count"] == 2
    assert audited["summary"]["recovered_object_count"] == 2
    assert attrs[0]["segmentation_color_rgb"] == [10, 20, 30]
    assert attrs[0]["segmentation_color_recovery_source"] == (
        "episode_jsonl_segmentation_color_map"
    )
    assert attrs[1]["segmentation_color_rgb"] == [40, 50, 60]


def test_recovery_does_not_guess_by_label_or_overwrite_existing_color(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    episode = tmp_path / "episode-001.jsonl"
    _write_episode(
        episode,
        episode_id="episode-001",
        color_map=[
            {"object_id": "Mug|+00.10|+00.90|+01.20", "rgb": [10, 20, 30]},
        ],
    )
    observations = tmp_path / "observations.json"
    _write_observations(
        observations,
        episode_id="episode-001",
        objects=[
            _object(
                "mug_existing",
                "Mug|+00.10|+00.90|+01.20",
                attributes={"segmentation_color_rgb": [1, 2, 3]},
            ),
            _object("mug_unmatched", "Mug|+09.99|+09.99|+09.99"),
        ],
    )
    output_root = tmp_path / "out"
    report = tmp_path / "report.json"

    module = load_script()
    exit_code = module.main(
        [
            "--episode-jsonl",
            str(episode),
            "--observation-sequence",
            str(observations),
            "--output-root",
            str(output_root),
            "--report",
            str(report),
        ]
    )

    emitted = json.loads(capsys.readouterr().out)
    audited = json.loads(report.read_text(encoding="utf-8"))
    enriched = json.loads(
        Path(audited["enriched_observation_sequences"][0]["output_path"]).read_text(
            encoding="utf-8"
        )
    )
    attrs = [
        obj["attributes"]
        for obj in enriched["observations"][0]["objects"]
    ]
    assert exit_code == 0
    assert emitted["ready"] is True
    assert audited["summary"]["already_colored_object_count"] == 1
    assert audited["summary"]["recovered_object_count"] == 0
    assert audited["summary"]["unmatched_object_count"] == 1
    assert attrs[0]["segmentation_color_rgb"] == [1, 2, 3]
    assert "segmentation_color_rgb" not in attrs[1]


def test_recovery_uses_matching_episode_color_map_when_raw_ids_repeat(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    raw_id = "Mug|+00.10|+00.90|+01.20"
    episode_one = tmp_path / "episode-001.jsonl"
    episode_two = tmp_path / "episode-002.jsonl"
    _write_episode(
        episode_one,
        episode_id="episode-001",
        color_map=[{"object_id": raw_id, "rgb": [10, 20, 30]}],
    )
    _write_episode(
        episode_two,
        episode_id="episode-002",
        color_map=[{"object_id": raw_id, "rgb": [40, 50, 60]}],
    )
    observations = tmp_path / "detector-observations-real-ai2thor-reachable-nbv-episode002.json"
    _write_observations(
        observations,
        episode_id="episode-002",
        objects=[_object("mug_1", raw_id)],
    )
    output_root = tmp_path / "out"
    report = tmp_path / "report.json"

    module = load_script()
    exit_code = module.main(
        [
            "--episode-jsonl",
            str(episode_one),
            "--episode-jsonl",
            str(episode_two),
            "--observation-sequence",
            str(observations),
            "--output-root",
            str(output_root),
            "--report",
            str(report),
        ]
    )

    _ = json.loads(capsys.readouterr().out)
    audited = json.loads(report.read_text(encoding="utf-8"))
    enriched = json.loads(
        Path(audited["enriched_observation_sequences"][0]["output_path"]).read_text(
            encoding="utf-8"
        )
    )
    attrs = enriched["observations"][0]["objects"][0]["attributes"]
    assert exit_code == 0
    assert attrs["segmentation_color_rgb"] == [40, 50, 60]


def _write_episode(path: Path, *, episode_id: str, color_map: list[dict[str, object]]) -> None:
    payload = {
        "episode_id": episode_id,
        "metadata": {
            "segmentation_color_map": color_map,
        },
        "schema_version": "dsg-spatialqa-lab.episode-frame.v1",
        "step": 1,
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_observations(
    path: Path,
    *,
    episode_id: str,
    objects: list[dict[str, object]],
) -> None:
    payload = {
        "schema_version": "dsg-spatialqa-lab.scene-observation-sequence.v1",
        "observation_count": 1,
        "observations": [
            {
                "episode_id": episode_id,
                "objects": objects,
                "step": 1,
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _object(
    object_id: str,
    ai2thor_object_id: str,
    *,
    attributes: dict[str, object] | None = None,
) -> dict[str, object]:
    merged_attributes = {
        "ai2thor_object_id": ai2thor_object_id,
        "rgb_path": "rgb.ppm",
        "segmentation_path": "segmentation.ppm",
        **(attributes or {}),
    }
    return {
        "attributes": merged_attributes,
        "label": "mug",
        "object_id": object_id,
        "visible": True,
    }
