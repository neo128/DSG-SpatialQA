from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import AI2ThorAdapterConfig, SpatialQAError, run_mock_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run a deterministic mock DSG-SpatialQA experiment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Explicit output directory for all generated artifacts.",
    )
    parser.add_argument(
        "--dataset-name",
        default="mock_experiment",
        help="Deterministic dataset name recorded in the benchmark manifest.",
    )
    parser.add_argument(
        "--max-qa-per-episode",
        type=int,
        default=3,
        help="Positive QA case cap for the generated mock episode.",
    )
    parser.add_argument(
        "--episode-count",
        type=int,
        default=1,
        help="Positive number of deterministic mock episodes to generate.",
    )
    parser.add_argument(
        "--qa-baseline",
        action="append",
        dest="qa_baselines",
        help=(
            "QA baseline name to compare against graph_tool. Repeat for a "
            "deterministic QA agent matrix."
        ),
    )
    args = parser.parse_args(argv)

    try:
        result = run_mock_experiment(
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            max_qa_per_episode=args.max_qa_per_episode,
            episode_configs=_episode_configs(args.episode_count),
            qa_baseline_names=(
                tuple(args.qa_baselines) if args.qa_baselines is not None else None
            ),
        )
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(
            {
                "action": "run_mock_experiment",
                "output_dir": str(args.output_dir),
                "valid": False,
                "error": str(exc),
            }
        )
        return 1

    _emit_json(
        {
            "action": "run_mock_experiment",
            "valid": True,
            **result,
        }
    )
    return 0


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


def _episode_configs(count: int) -> tuple[AI2ThorAdapterConfig, ...]:
    if count <= 0:
        raise ValueError("episode-count must be positive")
    return tuple(
        AI2ThorAdapterConfig(
            scene_id=f"FloorPlan{index}",
            episode_id=f"ai2thor_mock_{index:03d}",
            steps=(1, 2),
            actions=("Initialize", "MoveAhead"),
        )
        for index in range(1, count + 1)
    )


if __name__ == "__main__":
    raise SystemExit(main())
