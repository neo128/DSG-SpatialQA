import json
from pathlib import Path
import sys

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import audit_reachable_nbv_all_episodes as audit_all  # noqa: E402
import compare_reachable_nbv_all_episodes as compare_all  # noqa: E402
import run_reachable_nbv_all_episodes as run_all  # noqa: E402


def test_reachable_nbv_all_episodes_dry_run_accepts_twenty_episode_plan(tmp_path: Path) -> None:
    plan = {
        "schema_version": "dsg-spatialqa-lab.ai2thor-episode-plan.v1",
        "episodes": [
            {
                "short_id": f"episode{index:03d}",
                "episode_id": f"ai2thor-real-small-episode-{index:03d}",
                "scene_id": f"FloorPlan{index}",
            }
            for index in range(1, 21)
        ],
    }
    plan_path = tmp_path / "episode-plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    report_path = tmp_path / "dry-run-report.json"

    exit_code = run_all.main(
        [
            "--episode-plan",
            str(plan_path),
            "--dry-run",
            "--input-root",
            str(tmp_path / "inputs"),
            "--output-root",
            str(tmp_path / "navigation"),
            "--predicted-root",
            str(tmp_path / "predicted-dsg"),
            "--qa",
            str(tmp_path / "qa.jsonl"),
            "--report",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["runtime_kind"] == "dry_run"
    assert report["episode_count"] == 20
    assert report["valid"] is True
    assert report["episodes"][19]["scene_id"] == "FloorPlan20"
    assert report["episodes"][19]["paths"]["trajectory"].endswith(
        "reachable-nbv-real-ai2thor-trajectory-episode020.json"
    )


def test_reachable_nbv_all_episodes_dry_run_reports_active_qa_v2_paths(
    tmp_path: Path,
) -> None:
    plan = {
        "schema_version": "dsg-spatialqa-lab.ai2thor-episode-plan.v1",
        "episodes": [
            {
                "short_id": "episode007",
                "episode_id": "ai2thor-real-small-episode-007",
                "scene_id": "FloorPlan302",
            }
        ],
    }
    plan_path = tmp_path / "episode-plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    report_path = tmp_path / "dry-run-report.json"

    exit_code = run_all.main(
        [
            "--episode-plan",
            str(plan_path),
            "--dry-run",
            "--build-active-qa-v2",
            "--input-root",
            str(tmp_path / "inputs"),
            "--output-root",
            str(tmp_path / "navigation"),
            "--predicted-root",
            str(tmp_path / "predicted-dsg"),
            "--active-qa-root",
            str(tmp_path / "inputs" / "qa-v2-active-p54"),
            "--quality-report-root",
            str(tmp_path / "diagnostics"),
            "--report",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    paths = report["episodes"][0]["paths"]
    assert exit_code == 0
    assert report["valid"] is True
    assert paths["active_qa_observation_aware"].endswith(
        "qa-v2-active-p54/ai2thor-real-small-episode-007/qa-observation-aware.jsonl"
    )
    assert paths["active_qa_quality_report"].endswith(
        "qa-v2-active-p54-quality-report-ai2thor-real-small-episode-007.json"
    )
    assert paths["active_qa_vlm_request_bundle"].endswith(
        "qa-v2-active-p54/ai2thor-real-small-episode-007/vlm-request-bundle.json"
    )


def test_reachable_nbv_audit_and_compare_use_episode_plan(tmp_path: Path) -> None:
    plan = {
        "schema_version": "dsg-spatialqa-lab.ai2thor-episode-plan.v1",
        "episodes": [
            {
                "short_id": "episode001",
                "episode_id": "ai2thor-real-small-episode-001",
                "scene_id": "FloorPlan1",
            },
            {
                "short_id": "episode006",
                "episode_id": "ai2thor-real-small-episode-006",
                "scene_id": "FloorPlan6",
            },
        ],
    }
    plan_path = tmp_path / "episode-plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    output_root = tmp_path / "navigation"
    audit_report = tmp_path / "audit-report.json"
    compare_report = tmp_path / "compare-report.json"
    compare_md = tmp_path / "compare-report.zh.md"

    audit_code = audit_all.main(
        [
            "--episode-plan",
            str(plan_path),
            "--output-root",
            str(output_root),
            "--report",
            str(audit_report),
        ]
    )
    compare_code = compare_all.main(
        [
            "--episode-plan",
            str(plan_path),
            "--output-root",
            str(output_root),
            "--output",
            str(compare_report),
            "--markdown-output",
            str(compare_md),
        ]
    )

    audit_payload = json.loads(audit_report.read_text(encoding="utf-8"))
    compare_payload = json.loads(compare_report.read_text(encoding="utf-8"))
    assert audit_code == 1
    assert compare_code == 1
    assert [row["short_id"] for row in audit_payload["episodes"]] == [
        "episode001",
        "episode006",
    ]
    assert [row["short_id"] for row in compare_payload["episodes"]] == [
        "episode001",
        "episode006",
    ]


def test_fixed_vs_nbv_overlay_generates_missing_fixed_topdown(tmp_path: Path) -> None:
    episode_path = tmp_path / "inputs" / "episodes" / "episode.jsonl"
    episode_path.parent.mkdir(parents=True)
    episode_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "agent_pose": {"x": 0.0, "z": 0.0, "yaw": 0.0},
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "agent_pose": {"x": 0.5, "z": 0.0, "yaw": 90.0},
                    },
                    sort_keys=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trajectory_path = tmp_path / "navigation" / "trajectory.json"
    trajectory_path.parent.mkdir(parents=True)
    trajectory_path.write_text(
        json.dumps(
            {
                "steps": [
                    {"selected_viewpoint": {"x": 0.0, "z": 0.0}},
                    {"selected_viewpoint": {"x": 0.5, "z": 0.5}},
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    paths = {
        "episode": episode_path,
        "trajectory": trajectory_path,
        "fixed_episode_topdown": tmp_path / "inputs" / "episodes" / "episode-topdown-path.png",
        "fixed_vs_overlay_png": tmp_path / "inputs" / "episodes" / "episode-overlay.png",
        "fixed_vs_overlay_json": tmp_path / "inputs" / "episodes" / "episode-overlay.json",
    }

    valid = run_all._write_fixed_vs_nbv_overlay(  # noqa: SLF001
        paths,
        "episode",
        "episode001",
        "FloorPlan1",
    )

    assert valid is True
    assert paths["fixed_episode_topdown"].exists()
    assert paths["fixed_vs_overlay_png"].exists()
    overlay_meta = json.loads(paths["fixed_vs_overlay_json"].read_text(encoding="utf-8"))
    assert overlay_meta["fixed_topdown_generated"] is True
