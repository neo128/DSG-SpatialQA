from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    compare_research_conclusion_report,
    load_research_conclusion_report,
    research_conclusion_report,
    research_conclusion_report_digest,
    save_research_conclusion_markdown,
    save_research_conclusion_report,
    validate_research_conclusion_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Conclude whether DSG is superior to VLM/video-memory controls for "
            "an explicit local real experiment package."
        ),
    )
    parser.add_argument("--real-readiness-report", type=Path)
    parser.add_argument("--offline-control-result-report", type=Path)
    parser.add_argument("--predicted-dsg-evidence-report", type=Path)
    parser.add_argument("--graph-eval-report", type=Path)
    parser.add_argument("--error-attribution-report", type=Path)
    parser.add_argument("--qa-observability-report", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--markdown-report", type=Path)
    parser.add_argument("--required-source-kind", action="append", dest="required_source_kinds")
    parser.add_argument(
        "--evaluation-scope",
        choices=("full_oracle", "observation_aware"),
        default="full_oracle",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-candidate-exact-match-rate", type=float, default=0.2)
    parser.add_argument("--min-candidate-exact-match-count", type=int, default=15)
    parser.add_argument("--min-exact-match-rate-delta", type=float, default=0.05)
    parser.add_argument("--min-graph-object-recall", type=float, default=0.3)
    parser.add_argument("--min-observation-aware-case-count", type=int, default=30)
    parser.add_argument("--validate-report", type=Path)
    parser.add_argument("--compare-report", type=Path)
    args = parser.parse_args(argv)

    if args.validate_report is not None:
        try:
            validation = validate_research_conclusion_report(
                load_research_conclusion_report(args.validate_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(_error_payload("validate_research_conclusion_report", exc))
            return 1
        _emit_json(
            {
                "action": "validate_research_conclusion_report",
                "path": str(args.validate_report),
                **validation,
            }
        )
        return 0 if validation["valid"] is True else 1

    if args.compare_report is not None:
        try:
            comparison = compare_research_conclusion_report(
                load_research_conclusion_report(args.compare_report)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                {
                    **_error_payload("compare_research_conclusion_report", exc),
                    "matches": False,
                }
            )
            return 1
        _emit_json(
            {
                "action": "compare_research_conclusion_report",
                "path": str(args.compare_report),
                **comparison,
            }
        )
        return 0 if comparison["matches"] is True else 1

    if args.real_readiness_report is None:
        parser.error("--real-readiness-report is required")
    if args.offline_control_result_report is None:
        parser.error("--offline-control-result-report is required")
    if args.report is None:
        parser.error("--report is required")

    try:
        report = research_conclusion_report(
            _load_json(args.real_readiness_report),
            _load_json(args.offline_control_result_report),
            predicted_dsg_evidence_report=_load_optional_json(
                args.predicted_dsg_evidence_report
            ),
            graph_eval_report=_load_optional_json(args.graph_eval_report),
            error_attribution_report=_load_optional_json(
                args.error_attribution_report
            ),
            qa_observability_report=_load_optional_json(
                args.qa_observability_report
            ),
            real_readiness_report_path=args.real_readiness_report,
            offline_control_result_report_path=args.offline_control_result_report,
            predicted_dsg_evidence_report_path=args.predicted_dsg_evidence_report,
            graph_eval_report_path=args.graph_eval_report,
            error_attribution_report_path=args.error_attribution_report,
            qa_observability_report_path=args.qa_observability_report,
            required_source_kinds=args.required_source_kinds
            or ("caption_memory", "graph_text", "multi_frame_vlm", "vlm"),
            evaluation_scope=args.evaluation_scope,
            alpha=args.alpha,
            min_candidate_exact_match_rate=args.min_candidate_exact_match_rate,
            min_candidate_exact_match_count=args.min_candidate_exact_match_count,
            min_exact_match_rate_delta=args.min_exact_match_rate_delta,
            min_graph_object_recall=args.min_graph_object_recall,
            min_observation_aware_case_count=args.min_observation_aware_case_count,
        )
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("research_conclusion_report", exc))
        return 1

    save_research_conclusion_report(report, args.report)
    if args.markdown_report is not None:
        save_research_conclusion_markdown(report, args.markdown_report)
    validation = validate_research_conclusion_report(report)
    _emit_json(
        {
            "action": "research_conclusion_report",
            "path": str(args.report),
            "markdown_path": (
                str(args.markdown_report) if args.markdown_report is not None else None
            ),
            "valid": validation["valid"],
            "digest": research_conclusion_report_digest(report),
            "verdict": report["conclusion"]["verdict"],
            "dsg_superiority_claim_allowed": report["conclusion"][
                "dsg_superiority_claim_allowed"
            ],
        }
    )
    return 0 if validation["valid"] is True else 1


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError(f"Conclusion input must be a JSON object: {path}")
    return payload


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    return _load_json(path) if path is not None else None


def _error_payload(action: str, error: Exception) -> dict[str, Any]:
    return {
        "action": action,
        "valid": False,
        "error": str(error),
    }


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
