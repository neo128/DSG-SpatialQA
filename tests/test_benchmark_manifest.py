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
BUILD_BENCHMARK_SCRIPT = ROOT / "scripts" / "build_benchmark.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_build_benchmark_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_benchmark_script",
        BUILD_BENCHMARK_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_artifacts_build_small_manifest_and_coverage(tmp_path: Path) -> None:
    assert hasattr(lab, "build_benchmark_artifacts")
    assert hasattr(lab, "benchmark_manifest_digest")
    assert hasattr(lab, "validate_benchmark_manifest")
    episode_paths = _write_mock_episodes(tmp_path)
    output_dir = tmp_path / "benchmark"

    manifest = lab.build_benchmark_artifacts(
        dataset_name="mock_benchmark",
        episode_paths=episode_paths,
        output_dir=output_dir,
        max_qa_per_episode=4,
        tags=("benchmark", "oracle"),
    )
    validation = lab.validate_benchmark_manifest(manifest)

    assert manifest["schema_version"] == "dsg-spatialqa-lab.benchmark-manifest.v1"
    assert manifest["dataset_name"] == "mock_benchmark"
    assert manifest["summary"] == {
        "dataset_name": "mock_benchmark",
        "episode_count": 2,
        "qa_count": 8,
        "scene_count": 2,
        "task_count": 0,
    }
    assert manifest["scene_count"] == 2
    assert manifest["episode_count"] == 2
    assert manifest["qa_count"] == 8
    assert manifest["task_count"] == 0
    assert manifest["filters"] == {
        "max_qa_per_episode": 4,
        "source": "oracle",
        "tags": ["benchmark", "oracle"],
    }
    assert manifest["coverage"]["by_episode"] == {
        "ai2thor_mock_001": 4,
        "habitat_mock_001": 4,
    }
    assert manifest["coverage"]["by_scene"] == {"FloorPlan1": 4, "apartment_0": 4}
    assert manifest["coverage"]["by_question_type"] == {
        "object_location": 4,
        "object_room": 2,
        "relative_relation": 2,
    }
    assert manifest["coverage"]["by_reference_frame"] == {"none": 6, "world": 2}
    assert manifest["coverage"]["by_tag"]["benchmark"] == 8
    assert manifest["coverage"]["dynamic_static"] == {"dynamic": 0, "static": 8}
    assert manifest["coverage"]["oracle_predicted"] == {"oracle": 2, "predicted": 0}
    assert sorted(manifest["graph_digests"]) == ["ai2thor_mock_001", "habitat_mock_001"]
    assert sorted(manifest["qa_dataset_digests"]) == [
        "ai2thor_mock_001",
        "habitat_mock_001",
    ]
    assert validation["valid"] is True
    assert manifest["manifest_digest"] == lab.benchmark_manifest_digest(manifest)
    for artifact in manifest["artifacts"]:
        assert Path(artifact["episode_path"]).exists()
        assert Path(artifact["graph_path"]).exists()
        assert Path(artifact["qa_path"]).exists()


def test_benchmark_manifest_json_save_load_and_compare_detects_qa_drift(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "benchmark_manifest_json")
    assert hasattr(lab, "save_benchmark_manifest")
    assert hasattr(lab, "load_benchmark_manifest")
    assert hasattr(lab, "compare_benchmark_manifest")
    manifest_path = tmp_path / "benchmark-manifest.json"
    manifest = lab.build_benchmark_artifacts(
        dataset_name="mock_benchmark",
        episode_paths=_write_mock_episodes(tmp_path),
        output_dir=tmp_path / "benchmark",
        max_qa_per_episode=3,
    )

    lab.save_benchmark_manifest(manifest, manifest_path)
    loaded = lab.load_benchmark_manifest(manifest_path)
    comparison = lab.compare_benchmark_manifest(loaded)

    assert json.loads(lab.benchmark_manifest_json(manifest)) == manifest
    assert loaded == manifest
    assert comparison["matches"] is True

    first_artifact = cast(dict[str, object], loaded["artifacts"][0])
    qa_path = Path(cast(str, first_artifact["qa_path"]))
    cases = lab.load_qa_dataset(qa_path)
    cases[0].answer["object_id"] = "changed"
    lab.save_qa_dataset(cases, qa_path)

    drift = lab.compare_benchmark_manifest(loaded)
    checks = {check["name"]: check for check in drift["checks"]}
    assert drift["matches"] is False
    assert checks["qa_dataset_digests_match_current"]["passed"] is False
    assert checks["coverage_matches_current"]["passed"] is True


def test_build_benchmark_cli_outputs_validates_compares_and_reports_invalid_json(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_build_benchmark_script()
    main = cast(MainFn, getattr(module, "main"))
    episode_paths = _write_mock_episodes(tmp_path)
    output_dir = tmp_path / "benchmark"
    manifest_path = tmp_path / "benchmark-manifest.json"

    assert main(
        [
            "--episodes",
            str(episode_paths[0]),
            "--episodes",
            str(episode_paths[1]),
            "--dataset-name",
            "mock_benchmark",
            "--output-dir",
            str(output_dir),
            "--max-qa-per-episode",
            "3",
            "--manifest",
            str(manifest_path),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    manifest = lab.load_benchmark_manifest(manifest_path)
    assert output == {
        "action": "build_benchmark",
        "path": str(manifest_path),
        "valid": True,
        "digest": manifest["manifest_digest"],
        "summary": manifest["summary"],
    }

    assert main(["--validate-manifest", str(manifest_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_benchmark_manifest"
    assert validation["valid"] is True

    assert main(["--compare-manifest", str(manifest_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_benchmark_manifest"
    assert comparison["matches"] is True

    invalid_path = tmp_path / "invalid-manifest.json"
    invalid_path.write_text("[]\n", encoding="utf-8")
    assert main(["--validate-manifest", str(invalid_path)]) == 1
    invalid = json.loads(capsys.readouterr().out)
    assert invalid == {
        "action": "validate_benchmark_manifest",
        "path": str(invalid_path),
        "valid": False,
        "error": "Benchmark manifest JSON must be an object",
    }


def _write_mock_episodes(tmp_path: Path) -> tuple[Path, Path]:
    episode_dir = tmp_path / "episodes"
    ai2thor_path = episode_dir / "ai2thor.jsonl"
    habitat_path = episode_dir / "habitat.jsonl"
    lab.save_episode_sequence(
        lab.build_mock_ai2thor_episode(
            lab.AI2ThorAdapterConfig(
                scene_id="FloorPlan1",
                episode_id="ai2thor_mock_001",
                steps=(1, 2),
                actions=("Initialize", "MoveAhead"),
            )
        ),
        ai2thor_path,
    )
    lab.save_episode_sequence(
        lab.build_mock_habitat_episode(
            lab.HabitatAdapterConfig(
                scene_id="apartment_0",
                episode_id="habitat_mock_001",
                steps=(1, 2),
                actions=("reset", "turn_left"),
            )
        ),
        habitat_path,
    )
    return ai2thor_path, habitat_path
