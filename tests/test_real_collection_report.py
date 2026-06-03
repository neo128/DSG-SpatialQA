from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from _pytest.capture import CaptureFixture

import dsg_spatialqa_lab as lab


ROOT = Path(__file__).resolve().parents[1]
REAL_COLLECTION_SCRIPT = ROOT / "scripts" / "check_real_collection.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


def load_real_collection_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_real_collection_script",
        REAL_COLLECTION_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_collection_report_accepts_ai2thor_rgbd_segmentation_collection(
    tmp_path: Path,
) -> None:
    assert hasattr(lab, "real_collection_report")
    assert hasattr(lab, "real_collection_report_digest")
    assert hasattr(lab, "save_real_collection_report")
    assert hasattr(lab, "load_real_collection_report")
    assert hasattr(lab, "validate_real_collection_report")
    assert hasattr(lab, "compare_real_collection_report")
    episode_paths = _write_real_ai2thor_episodes(tmp_path)

    report = lab.real_collection_report(
        dataset_name="ai2thor_real_smoke",
        episode_paths=episode_paths,
        source_kind="ai2thor",
        min_episode_count=2,
        min_scene_count=2,
        min_frame_count=4,
    )
    report_path = tmp_path / "real-collection.json"
    saved_path = lab.save_real_collection_report(report, report_path)
    loaded = lab.load_real_collection_report(report_path)
    validation = lab.validate_real_collection_report(loaded)
    comparison = lab.compare_real_collection_report(loaded)

    assert report["schema_version"] == "dsg-spatialqa-lab.real-collection-report.v1"
    assert report["dataset_name"] == "ai2thor_real_smoke"
    assert report["source_kind"] == "ai2thor"
    assert report["collection_summary"] == {
        "action_counts": {"Initialize": 2, "MoveAhead": 2},
        "adapter_counts": {"ai2thor": 4},
        "agent_pose_frame_count": 4,
        "asset_summary": {
            "asset_kind_counts": {"depth": 4, "rgb": 4, "segmentation": 4},
            "asset_path_count": 12,
            "missing_asset_count": 0,
            "missing_assets": [],
            "present_asset_count": 12,
        },
        "collection_kind_counts": {"real": 4},
        "depth_frame_count": 4,
        "episode_count": 2,
        "episode_digests": {
            str(episode_paths[0]): lab.episode_sequence_digest(
                lab.load_episode_sequence(episode_paths[0])
            ),
            str(episode_paths[1]): lab.episode_sequence_digest(
                lab.load_episode_sequence(episode_paths[1])
            ),
        },
        "frame_count": 4,
        "frame_source_kind_counts": {"ai2thor": 4},
        "rgb_frame_count": 4,
        "scene_count": 2,
        "segmentation_frame_count": 4,
        "simulator_counts": {"unspecified": 4},
        "source_kind_counts": {"ai2thor": 4},
        "visible_object_frame_count": 4,
        "visible_object_nonempty_ratio": 1.0,
    }
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["source_kind_supported"]["passed"] is True
    assert checks["source_kind_matches_frames"]["passed"] is True
    assert checks["collection_kind_real"]["passed"] is True
    assert checks["required_frame_evidence_present"]["passed"] is True
    assert checks["frame_assets_present"] == {
        "name": "frame_assets_present",
        "passed": True,
        "asset_path_count": 12,
        "missing": [],
        "missing_asset_count": 0,
        "present_asset_count": 12,
    }
    assert report["readiness"] == {
        "ready": True,
        "failed_check_count": 0,
        "failed_checks": [],
    }
    assert saved_path == report_path
    assert loaded == report
    assert validation["valid"] is True
    assert comparison["matches"] is True


def test_real_collection_report_required_adapter_accepts_real_simulator_metadata(
    tmp_path: Path,
) -> None:
    episode_path = tmp_path / "episode-001.jsonl"
    frames = tuple(
        replace(
            frame,
            metadata={
                **frame.metadata,
                "collection_kind": "real",
                "source_kind": "real_simulator",
                "simulator": "ai2thor",
            },
        )
        for frame in _real_episode_frames(
            scene_id="FloorPlan1",
            episode_id="ai2thor_real_001",
            root="real-ai2thor",
        )
    )
    lab.save_episode_sequence(frames, episode_path)
    _write_frame_assets(episode_path, frames)

    report = lab.real_collection_report(
        dataset_name="ai2thor_real_smoke",
        episode_paths=(episode_path,),
        source_kind="ai2thor",
        required_adapter="ai2thor",
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=2,
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["required_adapter"] == "ai2thor"
    assert report["readiness"]["ready"] is True
    assert report["collection_summary"]["adapter_counts"] == {"ai2thor": 2}
    assert report["collection_summary"]["frame_source_kind_counts"] == {
        "real_simulator": 2,
    }
    assert report["collection_summary"]["simulator_counts"] == {"ai2thor": 2}
    assert report["collection_summary"]["visible_object_frame_count"] == 2
    assert report["collection_summary"]["agent_pose_frame_count"] == 2
    assert report["collection_summary"]["action_counts"] == {
        "Initialize": 1,
        "MoveAhead": 1,
    }
    assert checks["adapter_supported"]["passed"] is True
    assert checks["required_adapter_matches_frames"]["passed"] is True
    assert checks["frame_source_kind_real_simulator"]["passed"] is True
    assert checks["simulator_matches_required_adapter"]["passed"] is True
    assert checks["visible_object_ids_observed"]["passed"] is True
    assert checks["agent_pose_present"]["passed"] is True
    assert checks["action_coverage"]["passed"] is True
    assert lab.validate_real_collection_report(report)["valid"] is True
    assert lab.compare_real_collection_report(report)["matches"] is True


def test_real_collection_report_required_adapter_rejects_mock_metadata(
    tmp_path: Path,
) -> None:
    mock_episode_path = tmp_path / "mock-episode.jsonl"
    frames = lab.build_mock_ai2thor_episode(
        lab.AI2ThorAdapterConfig(
            scene_id="FloorPlan1",
            episode_id="ai2thor_mock_001",
            steps=(1, 2),
            artifact_root="mock-artifacts",
        )
    )
    lab.save_episode_sequence(frames, mock_episode_path)

    report = lab.real_collection_report(
        dataset_name="mock_benchmark",
        episode_paths=(mock_episode_path,),
        source_kind="ai2thor",
        required_adapter="ai2thor",
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=2,
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert checks["frame_source_kind_real_simulator"]["passed"] is False
    assert checks["frame_source_kind_real_simulator"]["actual"] == {"unspecified": 2}
    assert checks["mock_markers_absent"]["passed"] is False


def test_real_collection_report_rejects_mock_or_incomplete_collection(
    tmp_path: Path,
) -> None:
    mock_episode_path = tmp_path / "mock-episode.jsonl"
    frames = lab.build_mock_ai2thor_episode(
        lab.AI2ThorAdapterConfig(
            scene_id="FloorPlan1",
            episode_id="ai2thor_mock_001",
            steps=(1, 2),
            artifact_root="mock-artifacts",
        )
    )
    lab.save_episode_sequence(frames, mock_episode_path)

    report = lab.real_collection_report(
        dataset_name="mock_benchmark",
        episode_paths=(mock_episode_path,),
        source_kind="ai2thor",
        min_episode_count=2,
        min_scene_count=2,
        min_frame_count=4,
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert checks["episode_count_minimum"]["actual"] == 1
    assert checks["scene_count_minimum"]["actual"] == 1
    assert checks["frame_count_minimum"]["actual"] == 2
    assert checks["collection_kind_real"]["passed"] is False
    assert checks["collection_kind_real"]["actual"] == {"unspecified": 2}
    assert checks["mock_markers_absent"]["passed"] is False
    assert checks["mock_markers_absent"]["actual"] == [
        "episode_id:ai2thor_mock_001",
        "path:mock-artifacts/ai2thor_mock_001/depth/0001.npy",
        "path:mock-artifacts/ai2thor_mock_001/depth/0002.npy",
        "path:mock-artifacts/ai2thor_mock_001/rgb/0001.png",
        "path:mock-artifacts/ai2thor_mock_001/rgb/0002.png",
        "path:mock-artifacts/ai2thor_mock_001/segmentation/0001.png",
        "path:mock-artifacts/ai2thor_mock_001/segmentation/0002.png",
    ]


def test_real_collection_report_rejects_synthetic_or_placeholder_markers(
    tmp_path: Path,
) -> None:
    episode_path = tmp_path / "episode-001.jsonl"
    frames = tuple(
        replace(
            frame,
            metadata={
                **frame.metadata,
                "capture_run": "SyntheticPlaceholderTrial",
            },
        )
        for frame in _real_episode_frames(
            scene_id="FloorPlan1",
            episode_id="ai2thor_real_001",
            root="real-ai2thor",
        )
    )
    lab.save_episode_sequence(frames, episode_path)
    _write_frame_assets(episode_path, frames)

    report = lab.real_collection_report(
        dataset_name="ai2thor_real_smoke",
        episode_paths=(episode_path,),
        source_kind="ai2thor",
        min_episode_count=1,
        min_scene_count=1,
        min_frame_count=2,
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"]["ready"] is False
    assert "non_real_markers_absent" in report["readiness"]["failed_checks"]
    assert checks["non_real_markers_absent"] == {
        "name": "non_real_markers_absent",
        "passed": False,
        "actual": ["metadata:capture_run:SyntheticPlaceholderTrial"],
    }


def test_real_collection_report_rejects_missing_frame_asset(
    tmp_path: Path,
) -> None:
    episode_paths = _write_real_ai2thor_episodes(tmp_path)
    missing_asset_path = (
        tmp_path / "real-ai2thor" / "ai2thor_real_001" / "000001.rgb.png"
    )
    missing_asset_path.unlink()

    report = lab.real_collection_report(
        dataset_name="ai2thor_real_smoke",
        episode_paths=episode_paths,
        source_kind="ai2thor",
        min_episode_count=2,
        min_scene_count=2,
        min_frame_count=4,
    )
    checks = {check["name"]: check for check in report["checks"]}

    assert report["readiness"] == {
        "ready": False,
        "failed_check_count": 1,
        "failed_checks": ["frame_assets_present"],
    }
    assert report["collection_summary"]["asset_summary"] == {
        "asset_kind_counts": {"depth": 4, "rgb": 4, "segmentation": 4},
        "asset_path_count": 12,
        "missing_asset_count": 1,
        "missing_assets": [
            {
                "episode_path": str(episode_paths[0]),
                "kind": "rgb",
                "path": "real-ai2thor/ai2thor_real_001/000001.rgb.png",
                "resolved_path": str(missing_asset_path),
                "step": 1,
            },
        ],
        "present_asset_count": 11,
    }
    assert checks["frame_assets_present"] == {
        "name": "frame_assets_present",
        "passed": False,
        "asset_path_count": 12,
        "missing": report["collection_summary"]["asset_summary"]["missing_assets"],
        "missing_asset_count": 1,
        "present_asset_count": 11,
    }


def test_real_collection_cli_writes_validates_and_compares_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_real_collection_script()
    main = cast(MainFn, getattr(module, "main"))
    episode_paths = _write_real_ai2thor_episodes(tmp_path)
    report_path = tmp_path / "real-collection.json"

    assert main(
        [
            "--dataset-name",
            "ai2thor_real_smoke",
            "--source-kind",
            "ai2thor",
            "--episode",
            str(episode_paths[0]),
            "--episode",
            str(episode_paths[1]),
            "--report",
            str(report_path),
            "--min-episode-count",
            "2",
            "--min-scene-count",
            "2",
            "--min-frame-count",
            "4",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    report = lab.load_real_collection_report(report_path)
    assert output["action"] == "real_collection"
    assert output["ready"] is True
    assert output["report_digest"] == report["report_digest"]

    assert main(["--validate-report", str(report_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["action"] == "validate_real_collection_report"
    assert validation["valid"] is True

    assert main(["--compare-report", str(report_path)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["action"] == "compare_real_collection_report"
    assert comparison["matches"] is True


def test_real_collection_cli_supports_required_adapter_without_source_kind(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_real_collection_script()
    main = cast(MainFn, getattr(module, "main"))
    episode_path = tmp_path / "episode-001.jsonl"
    frames = tuple(
        replace(
            frame,
            metadata={
                **frame.metadata,
                "source_kind": "real_simulator",
                "simulator": "ai2thor",
            },
        )
        for frame in _real_episode_frames(
            scene_id="FloorPlan1",
            episode_id="ai2thor_real_001",
            root="real-ai2thor",
        )
    )
    lab.save_episode_sequence(frames, episode_path)
    _write_frame_assets(episode_path, frames)
    report_path = tmp_path / "real-collection.json"

    assert main(
        [
            "--dataset-name",
            "ai2thor_real_smoke",
            "--episode",
            str(episode_path),
            "--report",
            str(report_path),
            "--min-episode-count",
            "1",
            "--min-scene-count",
            "1",
            "--min-frame-count",
            "2",
            "--required-adapter",
            "ai2thor",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    report = lab.load_real_collection_report(report_path)
    assert output["ready"] is True
    assert report["source_kind"] == "ai2thor"
    assert report["required_adapter"] == "ai2thor"


def test_real_collection_request_bundle_exports_collection_templates(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    assert hasattr(lab, "REAL_COLLECTION_REQUEST_BUNDLE_SCHEMA_VERSION")
    assert hasattr(lab, "real_collection_request_bundle")
    assert hasattr(lab, "real_collection_request_bundle_digest")
    assert hasattr(lab, "save_real_collection_request_bundle")
    assert hasattr(lab, "load_real_collection_request_bundle")
    assert hasattr(lab, "validate_real_collection_request_bundle")
    assert hasattr(lab, "compare_real_collection_request_bundle")
    episode_paths = (
        tmp_path / "inputs" / "episodes" / "FloorPlan1.jsonl",
        tmp_path / "inputs" / "episodes" / "FloorPlan2.jsonl",
    )

    bundle = lab.real_collection_request_bundle(
        dataset_name="ai2thor_real_smoke",
        episode_paths=episode_paths,
        source_kind="ai2thor",
        report_path=tmp_path / "inputs" / "real-collection-report.json",
        min_episode_count=2,
        min_scene_count=2,
        min_frame_count=4,
    )

    assert bundle["schema_version"] == (
        "dsg-spatialqa-lab.real-collection-request-bundle.v1"
    )
    assert bundle["action"] == "real_collection_request_bundle"
    assert bundle["dataset_name"] == "ai2thor_real_smoke"
    assert bundle["source_kind"] == "ai2thor"
    assert bundle["episode_paths"] == [str(path) for path in episode_paths]
    assert bundle["report_path"] == str(
        tmp_path / "inputs" / "real-collection-report.json"
    )
    assert bundle["thresholds"] == {
        "min_episode_count": 2,
        "min_frame_count": 4,
        "min_scene_count": 2,
    }
    assert bundle["required_frame_evidence"] == [
        "depth",
        "rgb",
        "segmentation",
    ]
    assert bundle["frame_asset_fields"] == [
        "rgb_path",
        "depth_path",
        "segmentation_path",
    ]
    assert bundle["commands"] == {
        "collection_report": (
            "python scripts/check_real_collection.py "
            "--dataset-name ai2thor_real_smoke "
            "--source-kind ai2thor "
            f"--episode {episode_paths[0]} "
            f"--episode {episode_paths[1]} "
            f"--report {tmp_path / 'inputs' / 'real-collection-report.json'} "
            "--min-episode-count 2 "
            "--min-scene-count 2 "
            "--min-frame-count 4 "
            "--required-frame-evidence depth "
            "--required-frame-evidence rgb "
            "--required-frame-evidence segmentation"
        ),
        "compare_report": (
            "python scripts/check_real_collection.py --compare-report "
            f"{tmp_path / 'inputs' / 'real-collection-report.json'}"
        ),
        "validate_report": (
            "python scripts/check_real_collection.py --validate-report "
            f"{tmp_path / 'inputs' / 'real-collection-report.json'}"
        ),
    }
    assert bundle["episode_record_template"] == {
        "schema_version": "dsg-spatialqa-lab.episode-frame.v1",
        "episode_id": "episode_001",
        "scene_id": "scene_001",
        "step": 1,
        "rgb_path": "frames/000001.rgb.png",
        "depth_path": "frames/000001.depth.png",
        "segmentation_path": "frames/000001.segmentation.png",
        "agent_id": "agent",
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        "action": "Initialize",
        "visible_object_ids": [],
        "metadata": {
            "adapter": "ai2thor",
            "collection_kind": "real",
            "simulator": "ai2thor",
            "source_kind": "real_simulator",
        },
    }
    assert bundle["request_bundle_digest"] == (
        lab.real_collection_request_bundle_digest(bundle)
    )
    bundle_path = tmp_path / "handoff" / "real-collection-request-bundle.json"
    saved_path = lab.save_real_collection_request_bundle(bundle, bundle_path)
    loaded_bundle = lab.load_real_collection_request_bundle(bundle_path)
    validation = lab.validate_real_collection_request_bundle(loaded_bundle)
    comparison = lab.compare_real_collection_request_bundle(loaded_bundle)
    assert saved_path == bundle_path
    assert loaded_bundle == bundle
    assert validation["action"] == "validate_real_collection_request_bundle"
    assert validation["valid"] is True
    assert comparison["action"] == "compare_real_collection_request_bundle"
    assert comparison["matches"] is True

    module = load_real_collection_script()
    main = cast(MainFn, getattr(module, "main"))
    cli_bundle_path = tmp_path / "handoff" / "cli-real-collection-request-bundle.json"
    exit_code = main(
        [
            "--request-bundle",
            str(cli_bundle_path),
            "--dataset-name",
            "ai2thor_real_smoke",
            "--source-kind",
            "ai2thor",
            "--episode",
            str(episode_paths[0]),
            "--episode",
            str(episode_paths[1]),
            "--report",
            str(tmp_path / "inputs" / "real-collection-report.json"),
            "--min-episode-count",
            "2",
            "--min-scene-count",
            "2",
            "--min-frame-count",
            "4",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["action"] == "real_collection_request_bundle"
    assert payload["request_bundle_path"] == str(cli_bundle_path)
    assert payload["bundle"] == bundle
    assert lab.load_real_collection_request_bundle(cli_bundle_path) == bundle

    assert main(["--validate-request-bundle", str(cli_bundle_path)]) == 0
    cli_validation = json.loads(capsys.readouterr().out)
    assert cli_validation["action"] == "validate_real_collection_request_bundle"
    assert cli_validation["path"] == str(cli_bundle_path)
    assert cli_validation["valid"] is True

    assert main(["--compare-request-bundle", str(cli_bundle_path)]) == 0
    cli_comparison = json.loads(capsys.readouterr().out)
    assert cli_comparison["action"] == "compare_real_collection_request_bundle"
    assert cli_comparison["path"] == str(cli_bundle_path)
    assert cli_comparison["matches"] is True

    tampered_bundle = {
        **bundle,
        "thresholds": {
            **bundle["thresholds"],
            "min_frame_count": 5,
        },
    }
    tampered_bundle["request_bundle_digest"] = (
        lab.real_collection_request_bundle_digest(tampered_bundle)
    )
    tampered_path = tmp_path / "handoff" / "tampered-real-collection-request.json"
    lab.save_real_collection_request_bundle(tampered_bundle, tampered_path)

    tampered_validation = lab.validate_real_collection_request_bundle(
        lab.load_real_collection_request_bundle(tampered_path)
    )
    tampered_comparison = lab.compare_real_collection_request_bundle(
        lab.load_real_collection_request_bundle(tampered_path)
    )
    assert tampered_validation["valid"] is False
    assert tampered_comparison["matches"] is False

    assert main(["--validate-request-bundle", str(tampered_path)]) == 1
    cli_tampered_validation = json.loads(capsys.readouterr().out)
    assert cli_tampered_validation["valid"] is False

    assert main(["--compare-request-bundle", str(tampered_path)]) == 1
    cli_tampered_comparison = json.loads(capsys.readouterr().out)
    assert cli_tampered_comparison["matches"] is False


def _write_real_ai2thor_episodes(tmp_path: Path) -> tuple[Path, Path]:
    first_path = tmp_path / "episode-001.jsonl"
    second_path = tmp_path / "episode-002.jsonl"
    lab.save_episode_sequence(
        first_frames := _real_episode_frames(
            scene_id="FloorPlan1",
            episode_id="ai2thor_real_001",
            root="real-ai2thor",
        ),
        first_path,
    )
    lab.save_episode_sequence(
        second_frames := _real_episode_frames(
            scene_id="FloorPlan2",
            episode_id="ai2thor_real_002",
            root="real-ai2thor",
        ),
        second_path,
    )
    for episode_path, frames in (
        (first_path, first_frames),
        (second_path, second_frames),
    ):
        _write_frame_assets(episode_path, frames)
    return first_path, second_path


def _write_frame_assets(
    episode_path: Path,
    frames: tuple[lab.EpisodeFrame, ...],
) -> None:
    for frame in frames:
        for asset_path_text in (
            frame.depth_path,
            frame.rgb_path,
            frame.segmentation_path,
        ):
            assert asset_path_text is not None
            asset_path = episode_path.parent / asset_path_text
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.write_text(f"{frame.step}\n", encoding="utf-8")


def _real_episode_frames(
    *,
    scene_id: str,
    episode_id: str,
    root: str,
) -> tuple[lab.EpisodeFrame, ...]:
    return (
        _real_frame(scene_id=scene_id, episode_id=episode_id, root=root, step=1),
        _real_frame(scene_id=scene_id, episode_id=episode_id, root=root, step=2),
    )


def _real_frame(
    *,
    scene_id: str,
    episode_id: str,
    root: str,
    step: int,
) -> lab.EpisodeFrame:
    stem = f"{root}/{episode_id}/{step:06d}"
    return lab.EpisodeFrame(
        episode_id=episode_id,
        scene_id=scene_id,
        step=step,
        rgb_path=f"{stem}.rgb.png",
        depth_path=f"{stem}.depth.png",
        segmentation_path=f"{stem}.segmentation.png",
        agent_id="agent",
        agent_pose=lab.Pose3D(float(step), 0.0, 0.0),
        action="MoveAhead" if step == 2 else "Initialize",
        visible_object_ids=("mug_1", "table_1"),
        metadata={
            "adapter": "ai2thor",
            "collection_kind": "real",
            "source_kind": "ai2thor",
            "objects": [
                {
                    "object_id": "mug_1",
                    "label": "mug",
                    "pose": {"x": 0.1, "y": 1.0, "z": 0.8, "yaw": 0.0},
                    "bbox": {
                        "center": {"x": 0.1, "y": 1.0, "z": 0.8, "yaw": 0.0},
                        "size": [0.1, 0.1, 0.2],
                    },
                    "confidence": 1.0,
                    "visible": True,
                }
            ],
        },
    )
