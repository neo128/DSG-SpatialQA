from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    VLM_GRAPH_FUSION_POLICY,
    VLM_GRAPH_TRUSTED_FUSION_POLICY,
    load_qa_predictions,
    qa_predictions_digest,
    save_vlm_graph_fusion_predictions,
    validate_vlm_graph_fusion_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Fuse local VLM-only predictions with local GraphTool-only DSG predictions.",
    )
    parser.add_argument("--vlm-predictions", type=Path, required=True)
    parser.add_argument("--graph-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--fusion-policy",
        choices=(VLM_GRAPH_FUSION_POLICY, VLM_GRAPH_TRUSTED_FUSION_POLICY),
        default=VLM_GRAPH_FUSION_POLICY,
    )
    args = parser.parse_args(argv)

    try:
        vlm_predictions = load_qa_predictions(args.vlm_predictions)
        graph_predictions = load_qa_predictions(args.graph_predictions)
        report = save_vlm_graph_fusion_predictions(
            vlm_predictions,
            graph_predictions,
            args.output,
            report_path=args.report,
            vlm_prediction_path=args.vlm_predictions,
            graph_prediction_path=args.graph_predictions,
            fusion_policy=args.fusion_policy,
        )
        validation = validate_vlm_graph_fusion_report(report)
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(
            {
                "action": "fuse_vlm_graph_predictions",
                "error": str(exc),
                "output": str(args.output),
                "ready": False,
                "valid": False,
            }
        )
        return 1
    _emit_json(
        {
            "action": "fuse_vlm_graph_predictions",
            "fusion_policy": report["fusion_policy"],
            "output": str(args.output),
            "prediction_digest": report["prediction_digest"],
            "report_digest": report["report_digest"],
            "report_path": str(args.report) if args.report is not None else None,
            "saved_prediction_digest": qa_predictions_digest(load_qa_predictions(args.output)),
            "summary": report["summary"],
            "valid": validation["valid"],
        }
    )
    return 0 if validation["valid"] is True else 1


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
