from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    load_graph_json,
    load_qa_predictions,
    load_qa_dataset,
    load_vlm_frame_index_report,
    load_vlm_observable_slice_report,
    load_vlm_semantic_eval_report,
    load_vlm_support_gap_report,
    merge_vlm_frame_index_rows,
    run_vlm_calibration,
    save_vlm_answer_option_request_bundle,
    save_vlm_answer_option_coverage_report,
    save_vlm_frame_index_report,
    save_vlm_frame_index_rows,
    save_vlm_primary_frame_visibility_report,
    save_qa_predictions,
    save_vlm_retry_input_gap_report,
    save_vlm_retry_merge_report,
    save_vlm_retry_request_bundle,
    save_vlm_support_candidate_request_bundle,
    save_vlm_support_detector_handoff,
    save_vlm_support_gap_report,
    save_vlm_target_crop_request_bundle,
    validate_vlm_frame_index_report,
    load_vlm_answer_option_coverage_report,
    vlm_answer_option_request_bundle_digest,
    vlm_frame_index_report,
    vlm_frame_index_rows_from_detector_records,
    vlm_support_candidate_request_bundle,
    vlm_support_detector_handoff,
    vlm_support_gap_report,
    vlm_target_crop_request_bundle,
    validate_vlm_observable_slice_report,
    validate_vlm_semantic_eval_report,
    validate_vlm_support_gap_report,
    validate_vlm_answer_option_coverage_report,
    vlm_answer_option_request_bundle,
    vlm_answer_option_coverage_report,
    vlm_primary_frame_visibility_report,
    vlm_retry_input_gap_report,
    vlm_retry_merge_report,
    vlm_retry_merged_predictions,
    vlm_retry_request_bundle,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Build a single-frame VLM-observable QA slice and optional semantic "
            "location eval over explicit local artifacts."
        ),
    )
    parser.add_argument("--qa", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--request-bundle", type=Path)
    parser.add_argument("--frame-index", type=Path)
    parser.add_argument("--observable-qa-output", type=Path)
    parser.add_argument("--observable-request-bundle-output", type=Path)
    parser.add_argument("--observable-slice-report", type=Path)
    parser.add_argument("--primary-frame-visibility-report", type=Path)
    parser.add_argument("--semantic-eval-report", type=Path)
    parser.add_argument("--detector-frame-index-jsonl", type=Path)
    parser.add_argument("--existing-frame-index", type=Path)
    parser.add_argument("--vlm-frame-index-output", type=Path)
    parser.add_argument("--vlm-frame-index-report", type=Path)
    parser.add_argument("--support-candidate-request-bundle-output", type=Path)
    parser.add_argument("--max-support-candidates-per-case", type=int, default=6)
    parser.add_argument("--detector-handoff", type=Path)
    parser.add_argument("--support-candidate-request-bundle", type=Path)
    parser.add_argument("--support-detector-handoff-output", type=Path)
    parser.add_argument("--max-support-labels-per-frame", type=int, default=6)
    parser.add_argument("--support-gap-semantic-eval-report", type=Path)
    parser.add_argument("--support-gap-predicted-graph", type=Path)
    parser.add_argument("--support-gap-report-output", type=Path)
    parser.add_argument("--target-crop-detector-jsonl", type=Path)
    parser.add_argument("--target-crop-root", type=Path)
    parser.add_argument("--target-crop-request-bundle-output", type=Path)
    parser.add_argument("--target-crop-padding-pixels", type=int, default=4)
    parser.add_argument("--enhanced-vlm-request-bundle-output", type=Path)
    parser.add_argument("--answer-option-request-bundle-output", type=Path)
    parser.add_argument("--max-answer-options-per-case", type=int, default=8)
    parser.add_argument(
        "--include-ambiguous-support-relations",
        action="store_true",
        help="Add extra non-gold relation choices for known ambiguous support labels.",
    )
    parser.add_argument(
        "--include-target-affordance-options",
        action="store_true",
        help="Add non-gold target-label affordance prior answer choices.",
    )
    parser.add_argument("--answer-option-coverage-report", type=Path)
    parser.add_argument("--retry-semantic-eval-report", type=Path)
    parser.add_argument("--retry-request-bundle-output", type=Path)
    parser.add_argument("--retry-input-gap-report", type=Path)
    parser.add_argument("--merge-original-predictions", type=Path)
    parser.add_argument("--merge-retry-predictions", type=Path)
    parser.add_argument("--merge-retry-request-bundle", type=Path)
    parser.add_argument("--merged-predictions-output", type=Path)
    parser.add_argument("--retry-merge-report", type=Path)
    parser.add_argument(
        "--no-room-answer-option",
        action="store_true",
        help="Do not add the fallback IN_ROOM/room option to VLM answer choices.",
    )
    parser.add_argument("--validate-vlm-frame-index-report", type=Path)
    parser.add_argument("--validate-answer-option-coverage-report", type=Path)
    parser.add_argument("--validate-support-gap-report", type=Path)
    parser.add_argument("--validate-observable-slice-report", type=Path)
    parser.add_argument("--validate-semantic-eval-report", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.validate_vlm_frame_index_report is not None:
            report = load_vlm_frame_index_report(args.validate_vlm_frame_index_report)
            validation = validate_vlm_frame_index_report(report)
            _emit_json(
                {
                    "action": "validate_vlm_frame_index_report",
                    "path": str(args.validate_vlm_frame_index_report),
                    **validation,
                }
            )
            return 0 if validation["valid"] is True else 1
        if args.validate_answer_option_coverage_report is not None:
            report = load_vlm_answer_option_coverage_report(
                args.validate_answer_option_coverage_report
            )
            validation = validate_vlm_answer_option_coverage_report(report)
            _emit_json(
                {
                    "action": "validate_vlm_answer_option_coverage_report",
                    "path": str(args.validate_answer_option_coverage_report),
                    **validation,
                }
            )
            return 0 if validation["valid"] is True else 1
        if args.validate_support_gap_report is not None:
            report = load_vlm_support_gap_report(args.validate_support_gap_report)
            validation = validate_vlm_support_gap_report(report)
            _emit_json(
                {
                    "action": "validate_vlm_support_gap_report",
                    "path": str(args.validate_support_gap_report),
                    **validation,
                }
            )
            return 0 if validation["valid"] is True else 1
        if args.validate_observable_slice_report is not None:
            report = load_vlm_observable_slice_report(
                args.validate_observable_slice_report
            )
            validation = validate_vlm_observable_slice_report(report)
            _emit_json(
                {
                    "action": "validate_vlm_observable_slice_report",
                    "path": str(args.validate_observable_slice_report),
                    **validation,
                }
            )
            return 0 if validation["valid"] is True else 1
        if args.validate_semantic_eval_report is not None:
            report = load_vlm_semantic_eval_report(args.validate_semantic_eval_report)
            validation = validate_vlm_semantic_eval_report(report)
            _emit_json(
                {
                    "action": "validate_vlm_semantic_eval_report",
                    "path": str(args.validate_semantic_eval_report),
                    **validation,
                }
            )
            return 0 if validation["valid"] is True else 1
        if args.support_gap_report_output is not None:
            if args.qa is None:
                raise SpatialQAError("--support-gap-report-output requires --qa")
            if args.support_gap_semantic_eval_report is None:
                raise SpatialQAError(
                    "--support-gap-report-output requires "
                    "--support-gap-semantic-eval-report"
                )
            if args.support_gap_predicted_graph is None:
                raise SpatialQAError(
                    "--support-gap-report-output requires --support-gap-predicted-graph"
                )
            cases = load_qa_dataset(args.qa)
            semantic_report = load_vlm_semantic_eval_report(
                args.support_gap_semantic_eval_report
            )
            predicted_graph = load_graph_json(args.support_gap_predicted_graph)
            report = vlm_support_gap_report(
                cases,
                semantic_report,
                predicted_graph,
                qa_path=args.qa,
                semantic_eval_report_path=args.support_gap_semantic_eval_report,
                predicted_graph_path=args.support_gap_predicted_graph,
            )
            save_vlm_support_gap_report(report, args.support_gap_report_output)
            _emit_json(
                {
                    "action": "build_vlm_support_gap_report",
                    "output": str(args.support_gap_report_output),
                    "report_digest": report["report_digest"],
                    "summary": report["summary"],
                    "ready": True,
                }
            )
            return 0
        if args.vlm_frame_index_output is not None:
            if args.detector_frame_index_jsonl is None:
                raise SpatialQAError(
                    "--vlm-frame-index-output requires --detector-frame-index-jsonl"
                )
            existing_rows = (
                _load_jsonl_mappings(args.existing_frame_index)
                if args.existing_frame_index is not None
                else []
            )
            detector_records = _load_jsonl_mappings(args.detector_frame_index_jsonl)
            detector_rows = vlm_frame_index_rows_from_detector_records(
                detector_records,
            )
            rows = merge_vlm_frame_index_rows(existing_rows, detector_rows)
            save_vlm_frame_index_rows(rows, args.vlm_frame_index_output)
            report = vlm_frame_index_report(
                rows,
                source_path=args.detector_frame_index_jsonl,
                existing_frame_index_path=args.existing_frame_index,
            )
            if args.vlm_frame_index_report is not None:
                save_vlm_frame_index_report(report, args.vlm_frame_index_report)
            _emit_json(
                {
                    "action": "build_vlm_frame_index",
                    "detector_frame_index_jsonl": str(args.detector_frame_index_jsonl),
                    "existing_frame_index": (
                        str(args.existing_frame_index)
                        if args.existing_frame_index is not None
                        else None
                    ),
                    "frame_index_digest": report["frame_index_digest"],
                    "report": report,
                    "report_path": (
                        str(args.vlm_frame_index_report)
                        if args.vlm_frame_index_report is not None
                        else None
                    ),
                    "ready": True,
                    "vlm_frame_index_output": str(args.vlm_frame_index_output),
                }
            )
            return 0
        if args.support_candidate_request_bundle_output is not None:
            if args.request_bundle is None or args.frame_index is None:
                raise SpatialQAError(
                    "--support-candidate-request-bundle-output requires "
                    "--request-bundle and --frame-index"
                )
            request_bundle = _load_json_mapping(args.request_bundle)
            frame_index = _load_jsonl_mappings(args.frame_index)
            enriched_bundle = vlm_support_candidate_request_bundle(
                request_bundle,
                frame_index,
                max_candidates_per_case=args.max_support_candidates_per_case,
            )
            save_vlm_support_candidate_request_bundle(
                enriched_bundle,
                args.support_candidate_request_bundle_output,
            )
            enrichment = enriched_bundle["vlm_support_candidate_enrichment"]
            _emit_json(
                {
                    "action": "build_vlm_support_candidate_request_bundle",
                    "frame_index": str(args.frame_index),
                    "request_bundle": str(args.request_bundle),
                    "output": str(args.support_candidate_request_bundle_output),
                    "request_bundle_digest": enriched_bundle[
                        "request_bundle_digest"
                    ],
                    "summary": enrichment["summary"],
                    "ready": True,
                }
            )
            return 0
        if args.target_crop_request_bundle_output is not None:
            if args.request_bundle is None or args.target_crop_detector_jsonl is None:
                raise SpatialQAError(
                    "--target-crop-request-bundle-output requires "
                    "--request-bundle and --target-crop-detector-jsonl"
                )
            if args.target_crop_root is None:
                raise SpatialQAError(
                    "--target-crop-request-bundle-output requires --target-crop-root"
                )
            request_bundle = _load_json_mapping(args.request_bundle)
            detector_records = _load_jsonl_mappings(args.target_crop_detector_jsonl)
            enriched_bundle = vlm_target_crop_request_bundle(
                request_bundle,
                detector_records,
                crop_root=args.target_crop_root,
                padding_pixels=args.target_crop_padding_pixels,
            )
            save_vlm_target_crop_request_bundle(
                enriched_bundle,
                args.target_crop_request_bundle_output,
            )
            enrichment = enriched_bundle["vlm_target_crop_enrichment"]
            _emit_json(
                {
                    "action": "build_vlm_target_crop_request_bundle",
                    "detector_jsonl": str(args.target_crop_detector_jsonl),
                    "request_bundle": str(args.request_bundle),
                    "crop_root": str(args.target_crop_root),
                    "output": str(args.target_crop_request_bundle_output),
                    "request_bundle_digest": enriched_bundle[
                        "request_bundle_digest"
                    ],
                    "summary": enrichment["summary"],
                    "ready": True,
                }
            )
            return 0
        if args.enhanced_vlm_request_bundle_output is not None:
            if args.request_bundle is None:
                raise SpatialQAError(
                    "--enhanced-vlm-request-bundle-output requires --request-bundle"
                )
            if args.frame_index is None:
                raise SpatialQAError(
                    "--enhanced-vlm-request-bundle-output requires --frame-index"
                )
            if args.target_crop_detector_jsonl is None:
                raise SpatialQAError(
                    "--enhanced-vlm-request-bundle-output requires "
                    "--target-crop-detector-jsonl"
                )
            if args.target_crop_root is None:
                raise SpatialQAError(
                    "--enhanced-vlm-request-bundle-output requires --target-crop-root"
                )
            request_bundle = _load_json_mapping(args.request_bundle)
            frame_index = _load_jsonl_mappings(args.frame_index)
            detector_records = _load_jsonl_mappings(args.target_crop_detector_jsonl)
            support_bundle = vlm_support_candidate_request_bundle(
                request_bundle,
                frame_index,
                max_candidates_per_case=args.max_support_candidates_per_case,
            )
            crop_bundle = vlm_target_crop_request_bundle(
                support_bundle,
                detector_records,
                crop_root=args.target_crop_root,
                padding_pixels=args.target_crop_padding_pixels,
            )
            enriched_bundle = vlm_answer_option_request_bundle(
                crop_bundle,
                include_room_option=not args.no_room_answer_option,
                include_ambiguous_support_relations=(
                    args.include_ambiguous_support_relations
                ),
                include_target_affordance_options=(
                    args.include_target_affordance_options
                ),
                max_options_per_case=args.max_answer_options_per_case,
            )
            support_summary = support_bundle["vlm_support_candidate_enrichment"][
                "summary"
            ]
            crop_summary = crop_bundle["vlm_target_crop_enrichment"]["summary"]
            option_summary = enriched_bundle["vlm_answer_option_enrichment"][
                "summary"
            ]
            enriched_bundle["vlm_visual_request_enrichment"] = {
                "schema_version": (
                    "dsg-spatialqa-lab.vlm-visual-request-enrichment.v1"
                ),
                "summary": {
                    "answer_option_count": option_summary["answer_option_count"],
                    "case_count": option_summary["case_count"],
                    "cases_with_answer_options": option_summary[
                        "cases_with_answer_options"
                    ],
                    "cases_with_support_candidates": support_summary[
                        "cases_with_support_candidates"
                    ],
                    "cases_with_target_crop": crop_summary[
                        "cases_with_target_crop"
                    ],
                },
            }
            enriched_bundle["request_bundle_digest"] = (
                vlm_answer_option_request_bundle_digest(enriched_bundle)
            )
            save_vlm_answer_option_request_bundle(
                enriched_bundle,
                args.enhanced_vlm_request_bundle_output,
            )
            _emit_json(
                {
                    "action": "build_enhanced_vlm_request_bundle",
                    "crop_root": str(args.target_crop_root),
                    "detector_jsonl": str(args.target_crop_detector_jsonl),
                    "frame_index": str(args.frame_index),
                    "output": str(args.enhanced_vlm_request_bundle_output),
                    "request_bundle": str(args.request_bundle),
                    "request_bundle_digest": enriched_bundle[
                        "request_bundle_digest"
                    ],
                    "summary": enriched_bundle["vlm_visual_request_enrichment"][
                        "summary"
                    ],
                    "ready": True,
                }
            )
            return 0
        if args.answer_option_request_bundle_output is not None:
            if args.request_bundle is None:
                raise SpatialQAError(
                    "--answer-option-request-bundle-output requires --request-bundle"
                )
            request_bundle = _load_json_mapping(args.request_bundle)
            enriched_bundle = vlm_answer_option_request_bundle(
                request_bundle,
                include_room_option=not args.no_room_answer_option,
                include_ambiguous_support_relations=(
                    args.include_ambiguous_support_relations
                ),
                include_target_affordance_options=(
                    args.include_target_affordance_options
                ),
                max_options_per_case=args.max_answer_options_per_case,
            )
            save_vlm_answer_option_request_bundle(
                enriched_bundle,
                args.answer_option_request_bundle_output,
            )
            enrichment = enriched_bundle["vlm_answer_option_enrichment"]
            _emit_json(
                {
                    "action": "build_vlm_answer_option_request_bundle",
                    "request_bundle": str(args.request_bundle),
                    "output": str(args.answer_option_request_bundle_output),
                    "request_bundle_digest": enriched_bundle[
                        "request_bundle_digest"
                    ],
                    "summary": enrichment["summary"],
                    "ready": True,
                }
            )
            return 0
        if args.answer_option_coverage_report is not None:
            if args.qa is None or args.request_bundle is None:
                raise SpatialQAError(
                    "--answer-option-coverage-report requires --qa and --request-bundle"
                )
            cases = load_qa_dataset(args.qa)
            request_bundle = _load_json_mapping(args.request_bundle)
            report = vlm_answer_option_coverage_report(
                cases,
                request_bundle,
                qa_path=args.qa,
                request_bundle_path=args.request_bundle,
            )
            save_vlm_answer_option_coverage_report(
                report,
                args.answer_option_coverage_report,
            )
            _emit_json(
                {
                    "action": "build_vlm_answer_option_coverage_report",
                    "qa": str(args.qa),
                    "request_bundle": str(args.request_bundle),
                    "output": str(args.answer_option_coverage_report),
                    "report_digest": report["report_digest"],
                    "summary": report["summary"],
                    "ready": True,
                }
            )
            return 0
        if args.retry_request_bundle_output is not None:
            if args.request_bundle is None or args.retry_semantic_eval_report is None:
                raise SpatialQAError(
                    "--retry-request-bundle-output requires --request-bundle "
                    "and --retry-semantic-eval-report"
                )
            request_bundle = _load_json_mapping(args.request_bundle)
            semantic_report = load_vlm_semantic_eval_report(
                args.retry_semantic_eval_report
            )
            retry_bundle = vlm_retry_request_bundle(
                request_bundle,
                semantic_report,
            )
            save_vlm_retry_request_bundle(
                retry_bundle,
                args.retry_request_bundle_output,
            )
            _emit_json(
                {
                    "action": "build_vlm_retry_request_bundle",
                    "request_bundle": str(args.request_bundle),
                    "semantic_eval_report": str(args.retry_semantic_eval_report),
                    "output": str(args.retry_request_bundle_output),
                    "request_bundle_digest": retry_bundle["request_bundle_digest"],
                    "summary": retry_bundle["vlm_retry_enrichment"]["summary"],
                    "ready": True,
                }
            )
            return 0
        if args.retry_input_gap_report is not None:
            if args.request_bundle is None:
                raise SpatialQAError(
                    "--retry-input-gap-report requires --request-bundle"
                )
            request_bundle = _load_json_mapping(args.request_bundle)
            report = vlm_retry_input_gap_report(
                request_bundle,
                request_bundle_path=args.request_bundle,
            )
            save_vlm_retry_input_gap_report(report, args.retry_input_gap_report)
            _emit_json(
                {
                    "action": "build_vlm_retry_input_gap_report",
                    "output": str(args.retry_input_gap_report),
                    "ready": report["ready"],
                    "report_digest": report["report_digest"],
                    "summary": report["summary"],
                }
            )
            return 0 if report["ready"] is True else 1
        if args.retry_merge_report is not None:
            if args.retry_semantic_eval_report is None:
                raise SpatialQAError(
                    "--retry-merge-report requires --retry-semantic-eval-report"
                )
            if args.merge_original_predictions is None:
                raise SpatialQAError(
                    "--retry-merge-report requires --merge-original-predictions"
                )
            if args.merge_retry_predictions is None:
                raise SpatialQAError(
                    "--retry-merge-report requires --merge-retry-predictions"
                )
            if args.merged_predictions_output is None:
                raise SpatialQAError(
                    "--retry-merge-report requires --merged-predictions-output"
                )
            semantic_report = load_vlm_semantic_eval_report(
                args.retry_semantic_eval_report
            )
            original_predictions = load_qa_predictions(
                args.merge_original_predictions
            )
            retry_predictions = load_qa_predictions(args.merge_retry_predictions)
            retry_case_ids = None
            if args.merge_retry_request_bundle is not None:
                retry_bundle = _load_json_mapping(args.merge_retry_request_bundle)
                retry_case_ids = _retry_case_ids_from_request_bundle(retry_bundle)
            merged_predictions = vlm_retry_merged_predictions(
                original_predictions,
                retry_predictions,
                semantic_report,
                retry_case_ids=retry_case_ids,
            )
            save_qa_predictions(merged_predictions, args.merged_predictions_output)
            report = vlm_retry_merge_report(
                original_predictions,
                retry_predictions,
                merged_predictions,
                semantic_report,
                original_prediction_path=args.merge_original_predictions,
                retry_prediction_path=args.merge_retry_predictions,
                merged_prediction_path=args.merged_predictions_output,
                semantic_eval_report_path=args.retry_semantic_eval_report,
                retry_case_ids=retry_case_ids,
            )
            save_vlm_retry_merge_report(report, args.retry_merge_report)
            _emit_json(
                {
                    "action": "merge_vlm_retry_predictions",
                    "merged_prediction_digest": report["merged_prediction_digest"],
                    "merged_predictions_output": str(args.merged_predictions_output),
                    "ready": report["ready"],
                    "report": str(args.retry_merge_report),
                    "report_digest": report["report_digest"],
                    "retry_scope_kind": report["retry_scope_kind"],
                    "summary": report["summary"],
                }
            )
            return 0 if report["ready"] is True else 1
        if args.support_detector_handoff_output is not None:
            if (
                args.detector_handoff is None
                or args.support_candidate_request_bundle is None
            ):
                raise SpatialQAError(
                    "--support-detector-handoff-output requires "
                    "--detector-handoff and --support-candidate-request-bundle"
                )
            detector_handoff = _load_json_mapping(args.detector_handoff)
            support_bundle = _load_json_mapping(args.support_candidate_request_bundle)
            enriched_handoff = vlm_support_detector_handoff(
                detector_handoff,
                support_bundle,
                max_support_labels_per_frame=args.max_support_labels_per_frame,
            )
            save_vlm_support_detector_handoff(
                enriched_handoff,
                args.support_detector_handoff_output,
            )
            enrichment = enriched_handoff[
                "vlm_support_detector_handoff_enrichment"
            ]
            _emit_json(
                {
                    "action": "build_vlm_support_detector_handoff",
                    "detector_handoff": str(args.detector_handoff),
                    "support_candidate_request_bundle": str(
                        args.support_candidate_request_bundle
                    ),
                    "output": str(args.support_detector_handoff_output),
                    "support_detector_handoff_digest": enriched_handoff[
                        "support_detector_handoff_digest"
                    ],
                    "summary": enrichment["summary"],
                    "ready": True,
                }
            )
            return 0
        if args.qa is None:
            raise SpatialQAError("--qa is required unless validating a report")
        result = run_vlm_calibration(
            qa_path=args.qa,
            prediction_path=args.predictions,
            request_bundle_path=args.request_bundle,
            observable_qa_output=args.observable_qa_output,
            observable_request_bundle_output=args.observable_request_bundle_output,
            observable_slice_report_output=args.observable_slice_report,
            semantic_eval_report_output=args.semantic_eval_report,
        )
        if (
            args.request_bundle is not None
            and args.frame_index is not None
            and args.primary_frame_visibility_report is not None
        ):
            request_bundle = _load_json_mapping(args.request_bundle)
            frame_index = _load_jsonl_mappings(args.frame_index)
            primary_report = vlm_primary_frame_visibility_report(
                request_bundle,
                frame_index,
            )
            save_vlm_primary_frame_visibility_report(
                primary_report,
                args.primary_frame_visibility_report,
            )
            result["primary_frame_visibility_report_digest"] = primary_report[
                "report_digest"
            ]
            result["primary_frame_visibility_summary"] = primary_report["summary"]
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(
            {
                "action": "eval_vlm_calibration",
                "ready": False,
                "error": str(exc),
            }
        )
        return 1
    _emit_json({**result, "ready": True})
    return 0


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialQAError("JSON file must contain an object")
    return payload


def _load_jsonl_mappings(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line == "":
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise SpatialQAError(f"JSONL line {line_number} must contain an object")
        rows.append(payload)
    return rows


def _retry_case_ids_from_request_bundle(bundle: dict[str, Any]) -> list[str]:
    case_inputs = bundle.get("case_inputs")
    if isinstance(case_inputs, list):
        case_ids: list[str] = []
        for index, case_input in enumerate(case_inputs):
            if not isinstance(case_input, dict):
                raise SpatialQAError(
                    f"retry request bundle case_inputs[{index}] must be an object"
                )
            case_id = case_input.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise SpatialQAError(
                    f"retry request bundle case_inputs[{index}] must contain case_id"
                )
            case_ids.append(case_id)
        return case_ids
    enrichment = bundle.get("vlm_retry_enrichment")
    if isinstance(enrichment, dict):
        retry_case_ids = enrichment.get("retry_case_ids")
        if isinstance(retry_case_ids, list):
            return [str(case_id) for case_id in retry_case_ids]
    raise SpatialQAError("retry request bundle must contain case_inputs")


if __name__ == "__main__":
    raise SystemExit(main())
