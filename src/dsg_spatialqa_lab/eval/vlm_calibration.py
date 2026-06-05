from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from math import comb
from pathlib import Path
import re
from typing import Any, cast

from dsg_spatialqa_lab.benchmark import (
    QACase,
    load_qa_dataset,
    qa_dataset_digest,
    save_qa_dataset,
)
from dsg_spatialqa_lab.eval.qa_metrics import (
    QAPrediction,
    load_qa_predictions,
    qa_predictions_digest,
)
from dsg_spatialqa_lab.memory import CONTAINMENT_RELATIONS, DynamicSceneGraph
from dsg_spatialqa_lab.scene_io import graph_json_digest
from dsg_spatialqa_lab.schema import SpatialQAError


VLM_OBSERVABLE_SLICE_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-observable-slice-report.v1"
)
VLM_SEMANTIC_EVAL_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-semantic-eval-report.v1"
)
VLM_SEMANTIC_EVAL_DELTA_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-semantic-eval-delta-report.v1"
)
VLM_PRIMARY_FRAME_VISIBILITY_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-primary-frame-visibility-report.v1"
)
VLM_FRAME_INDEX_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-frame-index-report.v1"
)
VLM_VISIBLE_RELATIONS = frozenset(("IN_ROOM", "INSIDE", "ON"))
REAL_FRAME_TRACE_SCHEMA_VERSION = "dsg-spatialqa-lab.real-experiment-frame-trace.v1"
VLM_SUPPORT_CANDIDATE_ENRICHMENT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-support-candidate-enrichment.v1"
)
VLM_SUPPORT_DETECTOR_HANDOFF_ENRICHMENT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-support-detector-handoff-enrichment.v1"
)
VLM_TARGET_CROP_ENRICHMENT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-target-crop-enrichment.v1"
)
VLM_ANSWER_OPTION_ENRICHMENT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-answer-option-enrichment.v1"
)
VLM_ANSWER_OPTION_COVERAGE_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-answer-option-coverage-report.v1"
)
VLM_SUPPORT_GAP_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-support-gap-report.v1"
)
VLM_RETRY_ENRICHMENT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-retry-enrichment.v1"
)
VLM_RETRY_MERGE_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-retry-merge-report.v1"
)
VLM_RETRY_INPUT_GAP_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-retry-input-gap-report.v1"
)
VLM_ANSWER_OPTION_FALLBACK_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-answer-option-fallback-report.v1"
)
VLM_ROOM_LEVEL_TARGET_FALLBACK_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-room-level-target-fallback-report.v1"
)
VLM_SINGLE_SUPPORT_OPTION_FALLBACK_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-single-support-option-fallback-report.v1"
)
VLM_TEXT_OPTION_ALIGNMENT_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-text-option-alignment-report.v1"
)
VLM_AFFORDANCE_OPTION_FALLBACK_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.vlm-affordance-option-fallback-report.v1"
)
_TARGET_AFFORDANCE_OPTION_PRIORS = {
    "bowl": ("coffeetable", "countertop", "desk", "diningtable"),
    "bread": ("countertop", "diningtable"),
    "butterknife": ("countertop", "diningtable", "desk"),
    "cd": ("desk", "shelf"),
    "cellphone": ("bed", "desk", "coffeetable", "sidetable"),
    "creditcard": ("countertop", "desk"),
    "keychain": ("desk", "sidetable", "countertop"),
    "laptop": ("bed", "desk"),
    "mug": ("desk", "coffeetable", "countertop", "diningtable"),
    "pen": ("desk", "diningtable", "countertop"),
    "pencil": ("desk", "diningtable", "countertop"),
}
_SUPPORT_RELATION_HINTS = {
    "armchair": "ON",
    "bathtub": "INSIDE",
    "bed": "ON",
    "cabinet": "INSIDE",
    "chair": "ON",
    "coffeetable": "ON",
    "countertop": "ON",
    "desk": "ON",
    "diningtable": "ON",
    "drawer": "INSIDE",
    "dresser": "ON",
    "floor": "ON",
    "fridge": "INSIDE",
    "garbagecan": "INSIDE",
    "microwave": "INSIDE",
    "shelf": "ON",
    "sidetable": "ON",
    "sink": "INSIDE",
    "sofa": "ON",
    "stoveburner": "ON",
    "stovetop": "ON",
    "table": "ON",
}
_ADDITIONAL_SUPPORT_RELATION_HINTS = {
    "bathtub": ("ON",),
}
_TARGET_AFFORDANCE_ANSWER_OPTIONS = {
    "handtowel": (("ON", "handtowelholder"),),
}
_ROOM_LEVEL_SPECIFIC_SUPPORT_MATCHES = frozenset(
    (
        ("candle", "ON", "shelf"),
        ("faucet", "ON", "bathtub"),
        ("faucet", "ON", "sink"),
    )
)
_ROOM_LEVEL_TARGET_FALLBACK_LABELS = frozenset(
    (
        "armchair",
        "bathtub",
        "bed",
        "cabinet",
        "chair",
        "coffeetable",
        "countertop",
        "desk",
        "diningtable",
        "drawer",
        "dresser",
        "floor",
        "fridge",
        "garbagecan",
        "microwave",
        "shelf",
        "sidetable",
        "sink",
        "sofa",
        "stoveburner",
        "stovetop",
        "table",
    )
)
_NON_LOCATION_DESTINATION_LABELS = frozenset(
    (
        "currentframe",
        "currentview",
        "image",
        "unknown",
    )
)


def vlm_observable_slice_report(
    cases: Sequence[QACase],
    *,
    allowed_relations: Sequence[str] = tuple(sorted(VLM_VISIBLE_RELATIONS)),
    qa_path: str | Path | None = None,
) -> dict[str, Any]:
    allowed = frozenset(
        normalized
        for relation in allowed_relations
        if (normalized := _normalize_relation(relation)) is not None
    )
    rows = [
        _observable_case_row(case, allowed_relations=allowed)
        for case in cases
    ]
    observable_case_ids = [
        str(row["case_id"]) for row in rows if row.get("included") is True
    ]
    object_location_count = sum(
        1 for case in cases if case.question_type == "object_location"
    )
    visible_object_location_count = sum(
        1
        for case in cases
        if case.question_type == "object_location"
        and _mapping(case.answer).get("visible") is True
    )
    report: dict[str, Any] = {
        "schema_version": VLM_OBSERVABLE_SLICE_REPORT_SCHEMA_VERSION,
        "allowed_relations": sorted(allowed),
        "qa_digest": qa_dataset_digest(cases),
        "qa_path": str(qa_path) if qa_path is not None else None,
        "summary": {
            "case_count": len(cases),
            "observable_case_count": len(observable_case_ids),
            "observable_case_rate": _rate(len(observable_case_ids), len(cases)),
            "object_location_case_count": object_location_count,
            "visible_object_location_case_count": visible_object_location_count,
        },
        "observable_case_ids": observable_case_ids,
        "excluded_case_ids": [
            str(row["case_id"]) for row in rows if row.get("included") is not True
        ],
        "cases": rows,
    }
    report["report_digest"] = vlm_observable_slice_report_digest(report)
    return report


def vlm_observable_slice_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return _digest(payload)


def vlm_observable_slice_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_vlm_observable_slice_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(vlm_observable_slice_report_json(report), encoding="utf-8")
    return output_path


def load_vlm_observable_slice_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("VLM observable slice report must be a JSON object")
    return cast(dict[str, Any], payload)


def validate_vlm_observable_slice_report(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = report.get("cases")
    case_rows = _mapping_rows(rows)
    observable_case_ids = _string_list(report.get("observable_case_ids"))
    expected_observable_case_ids = [
        str(row["case_id"]) for row in case_rows if row.get("included") is True
    ]
    expected_digest = vlm_observable_slice_report_digest(report)
    summary = _mapping(report.get("summary"))
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == VLM_OBSERVABLE_SLICE_REPORT_SCHEMA_VERSION,
            "expected": VLM_OBSERVABLE_SLICE_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "report_digest",
            "passed": report.get("report_digest") == expected_digest,
            "expected": expected_digest,
            "actual": report.get("report_digest"),
        },
        {
            "name": "observable_case_ids",
            "passed": observable_case_ids == expected_observable_case_ids,
            "expected": expected_observable_case_ids,
            "actual": observable_case_ids,
        },
        {
            "name": "case_count",
            "passed": summary.get("case_count") == len(case_rows),
            "expected": len(case_rows),
            "actual": summary.get("case_count"),
        },
        {
            "name": "observable_case_count",
            "passed": summary.get("observable_case_count")
            == len(expected_observable_case_ids),
            "expected": len(expected_observable_case_ids),
            "actual": summary.get("observable_case_count"),
        },
    ]
    return {
        "action": "validate_vlm_observable_slice_report",
        "valid": all(check["passed"] for check in checks),
        "checks": checks,
    }


def vlm_observable_qa_cases(
    cases: Sequence[QACase],
    observable_case_ids: Sequence[str],
) -> list[QACase]:
    wanted = set(observable_case_ids)
    return [case for case in cases if case.id in wanted]


def save_vlm_observable_qa_dataset(
    cases: Sequence[QACase],
    observable_case_ids: Sequence[str],
    path: str | Path,
) -> Path:
    return save_qa_dataset(vlm_observable_qa_cases(cases, observable_case_ids), path)


def vlm_observable_request_bundle(
    bundle: Mapping[str, Any],
    *,
    observable_case_ids: Sequence[str],
) -> dict[str, Any]:
    wanted = set(observable_case_ids)
    filtered = json.loads(
        json.dumps(bundle, separators=(",", ":"), sort_keys=True)
    )
    if not isinstance(filtered, Mapping):
        raise SpatialQAError("VLM request bundle must be a JSON object")
    result = cast(dict[str, Any], filtered)
    result["case_inputs"] = [
        case
        for case in _mapping_rows(result.get("case_inputs"))
        if str(case.get("case_id")) in wanted
    ]
    result["case_count"] = len(result["case_inputs"])
    templates = result.get("prediction_templates")
    if isinstance(templates, Mapping):
        result["prediction_templates"] = {
            str(key): _filter_template_rows(value, wanted)
            for key, value in templates.items()
        }
    result["vlm_observable_filter"] = {
        "observable_case_count": result["case_count"],
        "observable_case_ids": sorted(wanted),
    }
    result["request_bundle_digest"] = vlm_observable_request_bundle_digest(result)
    return result


def vlm_observable_request_bundle_digest(bundle: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in bundle.items() if key != "request_bundle_digest"}
    return _digest(payload)


def vlm_observable_request_bundle_json(bundle: Mapping[str, Any]) -> str:
    return json.dumps(bundle, indent=2, sort_keys=True) + "\n"


def save_vlm_observable_request_bundle(
    bundle: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(vlm_observable_request_bundle_json(bundle), encoding="utf-8")
    return output_path


def vlm_support_candidate_request_bundle(
    bundle: Mapping[str, Any],
    frame_index: Sequence[Mapping[str, Any]],
    *,
    max_candidates_per_case: int = 6,
) -> dict[str, Any]:
    if max_candidates_per_case < 0:
        raise SpatialQAError("max_candidates_per_case must be non-negative")
    enriched = json.loads(
        json.dumps(bundle, separators=(",", ":"), sort_keys=True)
    )
    if not isinstance(enriched, Mapping):
        raise SpatialQAError("VLM request bundle must be a JSON object")
    result = cast(dict[str, Any], enriched)
    labels_by_frame = _visible_object_labels_by_frame(frame_index)
    case_inputs: list[dict[str, Any]] = []
    cases_with_primary_frame = 0
    cases_with_support = 0
    support_count = 0
    for case in _mapping_rows(result.get("case_inputs")):
        mutable_case = dict(case)
        primary_key = _primary_frame_key(_mapping(mutable_case.get("primary_frame")))
        if primary_key is not None:
            cases_with_primary_frame += 1
        target_label = _canonical_destination_label(
            _optional_str(_mapping(mutable_case.get("target")).get("label"))
        )
        frame_labels = labels_by_frame.get(primary_key, ()) if primary_key is not None else ()
        candidates = _support_candidates_for_labels(
            frame_labels,
            target_label=target_label,
            max_candidates=max_candidates_per_case,
        )
        if candidates:
            mutable_case["support_candidates"] = candidates
            cases_with_support += 1
            support_count += len(candidates)
        else:
            mutable_case.pop("support_candidates", None)
        case_inputs.append(mutable_case)
    result["case_inputs"] = case_inputs
    result["case_count"] = len(case_inputs)
    result["vlm_support_candidate_enrichment"] = {
        "schema_version": VLM_SUPPORT_CANDIDATE_ENRICHMENT_SCHEMA_VERSION,
        "frame_index_digest": vlm_frame_index_rows_digest(frame_index),
        "max_candidates_per_case": max_candidates_per_case,
        "summary": {
            "case_count": len(case_inputs),
            "cases_with_primary_frame": cases_with_primary_frame,
            "cases_with_support_candidates": cases_with_support,
            "support_candidate_count": support_count,
        },
    }
    result["request_bundle_digest"] = vlm_support_candidate_request_bundle_digest(
        result
    )
    return result


def vlm_support_candidate_request_bundle_digest(bundle: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in bundle.items() if key != "request_bundle_digest"}
    return _digest(payload)


def vlm_support_candidate_request_bundle_json(bundle: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(bundle), indent=2, sort_keys=True) + "\n"


def save_vlm_support_candidate_request_bundle(
    bundle: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        vlm_support_candidate_request_bundle_json(bundle),
        encoding="utf-8",
    )
    return output_path


def vlm_answer_option_request_bundle(
    bundle: Mapping[str, Any],
    *,
    include_room_option: bool = True,
    include_ambiguous_support_relations: bool = False,
    include_target_affordance_options: bool = False,
    max_options_per_case: int = 8,
) -> dict[str, Any]:
    if max_options_per_case < 0:
        raise SpatialQAError("max_options_per_case must be non-negative")
    enriched = json.loads(
        json.dumps(bundle, separators=(",", ":"), sort_keys=True)
    )
    if not isinstance(enriched, Mapping):
        raise SpatialQAError("VLM request bundle must be a JSON object")
    result = cast(dict[str, Any], enriched)
    cases: list[dict[str, Any]] = []
    cases_with_options = 0
    option_count = 0
    for case in _mapping_rows(result.get("case_inputs")):
        mutable_case = dict(case)
        _sanitize_support_candidates(mutable_case)
        options = _answer_options_for_case(
            mutable_case,
            include_room_option=include_room_option,
            include_ambiguous_support_relations=include_ambiguous_support_relations,
            include_target_affordance_options=include_target_affordance_options,
            max_options=max_options_per_case,
        )
        if options:
            mutable_case["answer_options"] = options
            cases_with_options += 1
            option_count += len(options)
            _add_answer_option_contract(mutable_case, options)
        else:
            mutable_case.pop("answer_options", None)
        cases.append(mutable_case)
    result["case_inputs"] = cases
    result["case_count"] = len(cases)
    result["vlm_answer_option_enrichment"] = {
        "schema_version": VLM_ANSWER_OPTION_ENRICHMENT_SCHEMA_VERSION,
        "include_room_option": include_room_option,
        "include_ambiguous_support_relations": include_ambiguous_support_relations,
        "include_target_affordance_options": include_target_affordance_options,
        "max_options_per_case": max_options_per_case,
        "source_request_bundle_digest": bundle.get("request_bundle_digest"),
        "summary": {
            "answer_option_count": option_count,
            "case_count": len(cases),
            "cases_with_answer_options": cases_with_options,
        },
    }
    result["request_bundle_digest"] = vlm_answer_option_request_bundle_digest(result)
    return result


def vlm_answer_option_request_bundle_digest(bundle: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in bundle.items() if key != "request_bundle_digest"}
    return _digest(payload)


def vlm_answer_option_request_bundle_json(bundle: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(bundle), indent=2, sort_keys=True) + "\n"


def save_vlm_answer_option_request_bundle(
    bundle: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        vlm_answer_option_request_bundle_json(bundle),
        encoding="utf-8",
    )
    return output_path


def vlm_retry_request_bundle(
    bundle: Mapping[str, Any],
    semantic_eval_report: Mapping[str, Any],
) -> dict[str, Any]:
    failed_case_ids = {
        case_id
        for row in _mapping_rows(semantic_eval_report.get("cases"))
        if row.get("semantic_match") is False
        if (case_id := _optional_str(row.get("case_id"))) is not None
    }
    copied = json.loads(json.dumps(bundle, separators=(",", ":"), sort_keys=True))
    if not isinstance(copied, Mapping):
        raise SpatialQAError("VLM request bundle must be a JSON object")
    sanitized = _strip_vlm_external_forbidden_fields(cast(dict[str, Any], copied))
    if not isinstance(sanitized, dict):
        raise SpatialQAError("VLM request bundle must be a JSON object")
    result = cast(dict[str, Any], sanitized)
    case_inputs = [
        _mapping(case)
        for case in _mapping_rows(result.get("case_inputs"))
        if _optional_str(case.get("case_id")) in failed_case_ids
    ]
    result["case_inputs"] = [
        cast(dict[str, Any], _strip_vlm_external_forbidden_fields(case))
        for case in case_inputs
    ]
    result["case_count"] = len(result["case_inputs"])
    templates = result.get("prediction_templates")
    if isinstance(templates, Mapping):
        result["prediction_templates"] = {
            str(key): [
                _strip_vlm_external_forbidden_fields(row)
                for row in _filter_template_rows(value, failed_case_ids)
            ]
            for key, value in templates.items()
        }
    source_case_count = len(_mapping_rows(bundle.get("case_inputs")))
    result["vlm_retry_enrichment"] = {
        "schema_version": VLM_RETRY_ENRICHMENT_SCHEMA_VERSION,
        "source_request_bundle_digest": bundle.get("request_bundle_digest"),
        "source_semantic_eval_report_digest": semantic_eval_report.get(
            "report_digest"
        ),
        "retry_case_ids": [
            str(case.get("case_id")) for case in result["case_inputs"]
        ],
        "retry_policy": {
            "gold_free": True,
            "rule": (
                "Retry only the listed case inputs. Use the provided local "
                "images, target_crop, support_candidates, answer_options, and "
                "answer_schema_hint. Do not use evaluator-only fields."
            ),
        },
        "summary": {
            "case_count": source_case_count,
            "retry_case_count": len(result["case_inputs"]),
            "skipped_success_count": source_case_count - len(result["case_inputs"]),
        },
    }
    result["request_bundle_digest"] = vlm_retry_request_bundle_digest(result)
    return result


def vlm_retry_request_bundle_digest(bundle: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in bundle.items() if key != "request_bundle_digest"}
    return _digest(payload)


def vlm_retry_request_bundle_json(bundle: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(bundle), indent=2, sort_keys=True) + "\n"


def save_vlm_retry_request_bundle(
    bundle: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(vlm_retry_request_bundle_json(bundle), encoding="utf-8")
    return output_path


def vlm_retry_merged_predictions(
    original_predictions: Sequence[QAPrediction],
    retry_predictions: Sequence[QAPrediction],
    semantic_eval_report: Mapping[str, Any],
    *,
    retry_case_ids: Sequence[str] | None = None,
) -> list[QAPrediction]:
    original_by_id = {prediction.id: prediction for prediction in original_predictions}
    retry_by_id = {prediction.id: prediction for prediction in retry_predictions}
    retry_case_id_set = set(
        _retry_case_ids_from_semantic_report(semantic_eval_report)
        if retry_case_ids is None
        else [str(case_id) for case_id in retry_case_ids]
    )
    merged: list[QAPrediction] = []
    for case_id in _semantic_report_case_ids(semantic_eval_report):
        if case_id in retry_case_id_set and case_id in retry_by_id:
            merged.append(retry_by_id[case_id])
        elif case_id in original_by_id:
            merged.append(original_by_id[case_id])
    return merged


def vlm_retry_merge_report(
    original_predictions: Sequence[QAPrediction],
    retry_predictions: Sequence[QAPrediction],
    merged_predictions: Sequence[QAPrediction],
    semantic_eval_report: Mapping[str, Any],
    *,
    original_prediction_path: str | Path | None = None,
    retry_prediction_path: str | Path | None = None,
    merged_prediction_path: str | Path | None = None,
    semantic_eval_report_path: str | Path | None = None,
    retry_case_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    expected_case_ids = _semantic_report_case_ids(semantic_eval_report)
    failed_case_ids = _retry_case_ids_from_semantic_report(semantic_eval_report)
    retry_scope_kind = (
        "semantic_failures" if retry_case_ids is None else "explicit_retry_case_ids"
    )
    effective_retry_case_ids = (
        failed_case_ids
        if retry_case_ids is None
        else [str(case_id) for case_id in retry_case_ids]
    )
    success_case_ids = [
        case_id
        for case_id in expected_case_ids
        if case_id not in set(failed_case_ids)
    ]
    out_of_scope_failed_case_ids = sorted(
        set(failed_case_ids) - set(effective_retry_case_ids)
    )
    original_ids = {prediction.id for prediction in original_predictions}
    retry_ids = {prediction.id for prediction in retry_predictions}
    merged_ids = {prediction.id for prediction in merged_predictions}
    retry_expected = set(effective_retry_case_ids)
    expected = set(expected_case_ids)
    missing_retry_case_ids = sorted(retry_expected - retry_ids)
    unexpected_retry_case_ids = sorted(retry_ids - retry_expected)
    missing_original_case_ids = sorted(
        case_id
        for case_id in expected
        if case_id not in retry_ids and case_id not in original_ids
    )
    replaced_retry_case_ids = sorted(retry_expected & retry_ids & merged_ids)
    kept_original_success_case_ids = sorted(set(success_case_ids) & original_ids & merged_ids)
    report: dict[str, Any] = {
        "schema_version": VLM_RETRY_MERGE_REPORT_SCHEMA_VERSION,
        "ready": (
            not missing_retry_case_ids
            and not unexpected_retry_case_ids
            and not missing_original_case_ids
        ),
        "semantic_eval_report_digest": semantic_eval_report.get("report_digest"),
        "semantic_eval_report_path": (
            str(semantic_eval_report_path)
            if semantic_eval_report_path is not None
            else None
        ),
        "retry_scope_kind": retry_scope_kind,
        "original_prediction_digest": qa_predictions_digest(original_predictions),
        "original_prediction_path": (
            str(original_prediction_path)
            if original_prediction_path is not None
            else None
        ),
        "retry_prediction_digest": qa_predictions_digest(retry_predictions),
        "retry_prediction_path": (
            str(retry_prediction_path) if retry_prediction_path is not None else None
        ),
        "merged_prediction_digest": qa_predictions_digest(merged_predictions),
        "merged_prediction_path": (
            str(merged_prediction_path) if merged_prediction_path is not None else None
        ),
        "expected_case_ids": expected_case_ids,
        "failed_case_ids": failed_case_ids,
        "retry_case_ids": effective_retry_case_ids,
        "kept_original_success_case_ids": kept_original_success_case_ids,
        "out_of_scope_failed_case_ids": out_of_scope_failed_case_ids,
        "replaced_retry_case_ids": replaced_retry_case_ids,
        "missing_original_case_ids": missing_original_case_ids,
        "missing_retry_case_ids": missing_retry_case_ids,
        "unexpected_retry_case_ids": unexpected_retry_case_ids,
        "summary": {
            "expected_case_count": len(expected_case_ids),
            "kept_original_success_count": len(kept_original_success_case_ids),
            "merged_prediction_count": len(merged_predictions),
            "missing_original_case_count": len(missing_original_case_ids),
            "missing_retry_case_count": len(missing_retry_case_ids),
            "original_prediction_count": len(original_predictions),
            "out_of_scope_failed_case_count": len(out_of_scope_failed_case_ids),
            "replaced_retry_case_count": len(replaced_retry_case_ids),
            "retry_expected_case_count": len(effective_retry_case_ids),
            "retry_prediction_count": len(retry_predictions),
            "unexpected_retry_case_count": len(unexpected_retry_case_ids),
        },
    }
    report["report_digest"] = vlm_retry_merge_report_digest(report)
    return report


def vlm_retry_merge_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return _digest(payload)


def vlm_retry_merge_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n"


def save_vlm_retry_merge_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(vlm_retry_merge_report_json(report), encoding="utf-8")
    return output_path


def load_vlm_retry_merge_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("VLM retry merge report must be a JSON object")
    return cast(dict[str, Any], payload)


def validate_vlm_retry_merge_report(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(report.get("summary"))
    retry_case_ids = _string_list(report.get("retry_case_ids"))
    kept_case_ids = _string_list(report.get("kept_original_success_case_ids"))
    out_of_scope_failed_case_ids = _string_list(
        report.get("out_of_scope_failed_case_ids")
    )
    replaced_case_ids = _string_list(report.get("replaced_retry_case_ids"))
    missing_original_case_ids = _string_list(report.get("missing_original_case_ids"))
    missing_retry_case_ids = _string_list(report.get("missing_retry_case_ids"))
    unexpected_retry_case_ids = _string_list(report.get("unexpected_retry_case_ids"))
    expected_digest = vlm_retry_merge_report_digest(report)
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == VLM_RETRY_MERGE_REPORT_SCHEMA_VERSION,
            "expected": VLM_RETRY_MERGE_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "report_digest",
            "passed": report.get("report_digest") == expected_digest,
            "expected": expected_digest,
            "actual": report.get("report_digest"),
        },
        {
            "name": "ready",
            "passed": isinstance(report.get("ready"), bool),
            "expected": "boolean",
            "actual": report.get("ready"),
        },
        {
            "name": "retry_expected_case_count",
            "passed": summary.get("retry_expected_case_count") == len(retry_case_ids),
            "expected": len(retry_case_ids),
            "actual": summary.get("retry_expected_case_count"),
        },
        {
            "name": "kept_original_success_count",
            "passed": summary.get("kept_original_success_count") == len(kept_case_ids),
            "expected": len(kept_case_ids),
            "actual": summary.get("kept_original_success_count"),
        },
        {
            "name": "out_of_scope_failed_case_count",
            "passed": (
                summary.get("out_of_scope_failed_case_count")
                == len(out_of_scope_failed_case_ids)
                or (
                    summary.get("out_of_scope_failed_case_count") is None
                    and len(out_of_scope_failed_case_ids) == 0
                )
            ),
            "expected": len(out_of_scope_failed_case_ids),
            "actual": summary.get("out_of_scope_failed_case_count"),
        },
        {
            "name": "replaced_retry_case_count",
            "passed": summary.get("replaced_retry_case_count") == len(replaced_case_ids),
            "expected": len(replaced_case_ids),
            "actual": summary.get("replaced_retry_case_count"),
        },
        {
            "name": "missing_original_case_count",
            "passed": summary.get("missing_original_case_count")
            == len(missing_original_case_ids),
            "expected": len(missing_original_case_ids),
            "actual": summary.get("missing_original_case_count"),
        },
        {
            "name": "missing_retry_case_count",
            "passed": summary.get("missing_retry_case_count")
            == len(missing_retry_case_ids),
            "expected": len(missing_retry_case_ids),
            "actual": summary.get("missing_retry_case_count"),
        },
        {
            "name": "unexpected_retry_case_count",
            "passed": summary.get("unexpected_retry_case_count")
            == len(unexpected_retry_case_ids),
            "expected": len(unexpected_retry_case_ids),
            "actual": summary.get("unexpected_retry_case_count"),
        },
    ]
    return {
        "action": "validate_vlm_retry_merge_report",
        "valid": all(check["passed"] for check in checks),
        "checks": checks,
    }


def vlm_retry_input_gap_report(
    bundle: Mapping[str, Any],
    *,
    request_bundle_path: str | Path | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    next_targets: list[dict[str, Any]] = []
    for case in _mapping_rows(bundle.get("case_inputs")):
        row = _retry_input_gap_case_row(case)
        rows.append(row)
        missing_kinds = _string_list(row.get("missing_input_kinds"))
        if missing_kinds:
            target = _mapping(case.get("target"))
            primary_frame = _mapping(case.get("primary_frame"))
            next_targets.append(
                {
                    "case_id": row["case_id"],
                    "episode_id": _optional_str(case.get("episode_id")),
                    "missing_input_kinds": missing_kinds,
                    "primary_frame_step": (
                        primary_frame.get("step")
                        if isinstance(primary_frame.get("step"), int)
                        and not isinstance(primary_frame.get("step"), bool)
                        else None
                    ),
                    "scene_id": _optional_str(case.get("scene_id")),
                    "target_label": _optional_str(target.get("label")),
                    "target_object_id": _optional_str(target.get("object_id")),
                }
            )
    summary = {
        "case_count": len(rows),
        "cases_with_answer_options": sum(
            1 for row in rows if row["has_answer_options"] is True
        ),
        "cases_with_frames": sum(1 for row in rows if row["has_frames"] is True),
        "cases_with_primary_frame": sum(
            1 for row in rows if row["has_primary_frame"] is True
        ),
        "cases_with_support_candidates": sum(
            1 for row in rows if row["has_support_candidates"] is True
        ),
        "support_candidates_required_count": sum(
            1 for row in rows if row["support_candidates_required"] is True
        ),
        "support_candidates_not_applicable_count": sum(
            1 for row in rows if row["support_candidates_required"] is not True
        ),
        "cases_with_target_crop": sum(
            1 for row in rows if row["has_target_crop"] is True
        ),
        "missing_answer_options_count": sum(
            1 for row in rows if row["has_answer_options"] is not True
        ),
        "missing_frames_count": sum(
            1 for row in rows if row["has_frames"] is not True
        ),
        "missing_primary_frame_count": sum(
            1 for row in rows if row["has_primary_frame"] is not True
        ),
        "missing_support_candidates_count": sum(
            1
            for row in rows
            if row["support_candidates_required"] is True
            and row["has_support_candidates"] is not True
        ),
        "missing_target_crop_count": sum(
            1 for row in rows if row["has_target_crop"] is not True
        ),
    }
    report: dict[str, Any] = {
        "schema_version": VLM_RETRY_INPUT_GAP_REPORT_SCHEMA_VERSION,
        "ready": not next_targets,
        "request_bundle_digest": bundle.get("request_bundle_digest"),
        "request_bundle_path": (
            str(request_bundle_path) if request_bundle_path is not None else None
        ),
        "summary": summary,
        "next_collection_targets": next_targets,
        "cases": rows,
    }
    report["report_digest"] = vlm_retry_input_gap_report_digest(report)
    return report


def vlm_retry_input_gap_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return _digest(payload)


def vlm_retry_input_gap_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n"


def save_vlm_retry_input_gap_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(vlm_retry_input_gap_report_json(report), encoding="utf-8")
    return output_path


def load_vlm_retry_input_gap_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("VLM retry input gap report must be a JSON object")
    return cast(dict[str, Any], payload)


def validate_vlm_retry_input_gap_report(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(report.get("summary"))
    rows = _mapping_rows(report.get("cases"))
    next_targets = _mapping_rows(report.get("next_collection_targets"))
    expected_digest = vlm_retry_input_gap_report_digest(report)
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == VLM_RETRY_INPUT_GAP_REPORT_SCHEMA_VERSION,
            "expected": VLM_RETRY_INPUT_GAP_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "report_digest",
            "passed": report.get("report_digest") == expected_digest,
            "expected": expected_digest,
            "actual": report.get("report_digest"),
        },
        {
            "name": "case_count",
            "passed": summary.get("case_count") == len(rows),
            "expected": len(rows),
            "actual": summary.get("case_count"),
        },
        {
            "name": "ready",
            "passed": report.get("ready") == (len(next_targets) == 0),
            "expected": len(next_targets) == 0,
            "actual": report.get("ready"),
        },
        {
            "name": "missing_target_crop_count",
            "passed": summary.get("missing_target_crop_count")
            == sum(1 for row in rows if row.get("has_target_crop") is not True),
            "expected": sum(
                1 for row in rows if row.get("has_target_crop") is not True
            ),
            "actual": summary.get("missing_target_crop_count"),
        },
        {
            "name": "missing_support_candidates_count",
            "passed": summary.get("missing_support_candidates_count")
            == sum(
                1
                for row in rows
                if row.get("support_candidates_required") is True
                and row.get("has_support_candidates") is not True
            ),
            "expected": sum(
                1
                for row in rows
                if row.get("support_candidates_required") is True
                and row.get("has_support_candidates") is not True
            ),
            "actual": summary.get("missing_support_candidates_count"),
        },
    ]
    return {
        "action": "validate_vlm_retry_input_gap_report",
        "valid": all(check["passed"] for check in checks),
        "checks": checks,
    }


def vlm_answer_option_fallback_predictions(
    predictions: Sequence[QAPrediction],
    request_bundle: Mapping[str, Any],
    *,
    prediction_path: str | Path | None = None,
    request_bundle_path: str | Path | None = None,
) -> tuple[list[QAPrediction], dict[str, Any]]:
    case_inputs = {
        str(case["case_id"]): case
        for case in _mapping_rows(request_bundle.get("case_inputs"))
        if isinstance(case.get("case_id"), str)
    }
    calibrated: list[QAPrediction] = []
    applied_case_ids: list[str] = []
    skipped_existing_location_case_ids: list[str] = []
    skipped_no_single_room_option_case_ids: list[str] = []
    missing_case_input_ids: list[str] = []

    for prediction in predictions:
        case = case_inputs.get(prediction.id)
        if case is None:
            calibrated.append(prediction)
            missing_case_input_ids.append(prediction.id)
            continue
        if _prediction_has_usable_current_location(prediction):
            calibrated.append(prediction)
            skipped_existing_location_case_ids.append(prediction.id)
            continue
        option = _single_room_fallback_option(case)
        if option is None:
            calibrated.append(prediction)
            skipped_no_single_room_option_case_ids.append(prediction.id)
            continue
        answer = dict(prediction.answer)
        answer["answer_option_id"] = option["option_id"]
        answer["current_location"] = {
            "dst": "room",
            "dst_label": "room",
            "relation": "IN_ROOM",
            "step": _case_primary_step(case),
        }
        answer["vlm_answer_option_fallback"] = {
            "case_id": prediction.id,
            "source": "single_room_answer_option",
        }
        calibrated.append(
            QAPrediction(
                id=prediction.id,
                answer=answer,
                evidence_nodes=prediction.evidence_nodes,
                evidence_edges=prediction.evidence_edges,
                confidence=prediction.confidence,
                error=None,
            )
        )
        applied_case_ids.append(prediction.id)

    report: dict[str, Any] = {
        "schema_version": VLM_ANSWER_OPTION_FALLBACK_REPORT_SCHEMA_VERSION,
        "request_bundle_digest": request_bundle.get("request_bundle_digest"),
        "request_bundle_path": (
            str(request_bundle_path) if request_bundle_path is not None else None
        ),
        "input_prediction_digest": qa_predictions_digest(predictions),
        "input_prediction_path": (
            str(prediction_path) if prediction_path is not None else None
        ),
        "output_prediction_digest": qa_predictions_digest(calibrated),
        "applied_case_ids": applied_case_ids,
        "missing_case_input_ids": missing_case_input_ids,
        "skipped_existing_location_case_ids": skipped_existing_location_case_ids,
        "skipped_no_single_room_option_case_ids": skipped_no_single_room_option_case_ids,
        "summary": {
            "applied_fallback_count": len(applied_case_ids),
            "case_input_count": len(case_inputs),
            "input_prediction_count": len(predictions),
            "missing_case_input_count": len(missing_case_input_ids),
            "output_prediction_count": len(calibrated),
            "skipped_existing_location_count": len(
                skipped_existing_location_case_ids
            ),
            "skipped_no_single_room_option_count": len(
                skipped_no_single_room_option_case_ids
            ),
        },
    }
    report["report_digest"] = vlm_answer_option_fallback_report_digest(report)
    return calibrated, report


def vlm_answer_option_fallback_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return _digest(payload)


def vlm_answer_option_fallback_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n"


def save_vlm_answer_option_fallback_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        vlm_answer_option_fallback_report_json(report),
        encoding="utf-8",
    )
    return output_path


def load_vlm_answer_option_fallback_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("VLM answer option fallback report must be a JSON object")
    return cast(dict[str, Any], payload)


def validate_vlm_answer_option_fallback_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _mapping(report.get("summary"))
    applied_case_ids = _string_list(report.get("applied_case_ids"))
    missing_case_input_ids = _string_list(report.get("missing_case_input_ids"))
    skipped_existing_location_case_ids = _string_list(
        report.get("skipped_existing_location_case_ids")
    )
    skipped_no_single_room_option_case_ids = _string_list(
        report.get("skipped_no_single_room_option_case_ids")
    )
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == VLM_ANSWER_OPTION_FALLBACK_REPORT_SCHEMA_VERSION,
            "expected": VLM_ANSWER_OPTION_FALLBACK_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "report_digest",
            "passed": report.get("report_digest")
            == vlm_answer_option_fallback_report_digest(report),
            "expected": vlm_answer_option_fallback_report_digest(report),
            "actual": report.get("report_digest"),
        },
        {
            "name": "applied_fallback_count",
            "passed": summary.get("applied_fallback_count")
            == len(applied_case_ids),
            "expected": len(applied_case_ids),
            "actual": summary.get("applied_fallback_count"),
        },
        {
            "name": "missing_case_input_count",
            "passed": summary.get("missing_case_input_count")
            == len(missing_case_input_ids),
            "expected": len(missing_case_input_ids),
            "actual": summary.get("missing_case_input_count"),
        },
        {
            "name": "skipped_existing_location_count",
            "passed": summary.get("skipped_existing_location_count")
            == len(skipped_existing_location_case_ids),
            "expected": len(skipped_existing_location_case_ids),
            "actual": summary.get("skipped_existing_location_count"),
        },
        {
            "name": "skipped_no_single_room_option_count",
            "passed": summary.get("skipped_no_single_room_option_count")
            == len(skipped_no_single_room_option_case_ids),
            "expected": len(skipped_no_single_room_option_case_ids),
            "actual": summary.get("skipped_no_single_room_option_count"),
        },
    ]
    return {
        "action": "validate_vlm_answer_option_fallback_report",
        "valid": all(check["passed"] for check in checks),
        "checks": checks,
    }


def vlm_room_level_target_fallback_predictions(
    predictions: Sequence[QAPrediction],
    request_bundle: Mapping[str, Any],
    *,
    prediction_path: str | Path | None = None,
    request_bundle_path: str | Path | None = None,
) -> tuple[list[QAPrediction], dict[str, Any]]:
    case_inputs = {
        str(case["case_id"]): case
        for case in _mapping_rows(request_bundle.get("case_inputs"))
        if isinstance(case.get("case_id"), str)
    }
    calibrated: list[QAPrediction] = []
    applied_case_ids: list[str] = []
    missing_case_input_ids: list[str] = []
    skipped_existing_location_case_ids: list[str] = []
    skipped_ineligible_error_case_ids: list[str] = []
    skipped_no_room_option_case_ids: list[str] = []
    skipped_non_room_level_target_case_ids: list[str] = []

    for prediction in predictions:
        case = case_inputs.get(prediction.id)
        if case is None:
            calibrated.append(prediction)
            missing_case_input_ids.append(prediction.id)
            continue
        if _prediction_has_usable_current_location(prediction):
            calibrated.append(prediction)
            skipped_existing_location_case_ids.append(prediction.id)
            continue
        if prediction.error not in {"target_not_observed", "relation_not_observed"}:
            calibrated.append(prediction)
            skipped_ineligible_error_case_ids.append(prediction.id)
            continue
        target_label = _canonical_destination_label(
            _optional_str(_mapping(case.get("target")).get("label"))
        )
        if target_label not in _ROOM_LEVEL_TARGET_FALLBACK_LABELS:
            calibrated.append(prediction)
            skipped_non_room_level_target_case_ids.append(prediction.id)
            continue
        option = _room_fallback_option(case)
        if option is None:
            calibrated.append(prediction)
            skipped_no_room_option_case_ids.append(prediction.id)
            continue

        answer = dict(prediction.answer)
        answer["answer_option_id"] = option["option_id"]
        answer["current_location"] = {
            "dst": "room",
            "dst_label": "room",
            "relation": "IN_ROOM",
            "step": _case_primary_step(case),
        }
        answer["vlm_room_level_target_fallback"] = {
            "case_id": prediction.id,
            "source": "room_level_target_prior",
            "target_label": target_label,
        }
        calibrated.append(
            QAPrediction(
                id=prediction.id,
                answer=answer,
                evidence_nodes=prediction.evidence_nodes,
                evidence_edges=prediction.evidence_edges,
                confidence=prediction.confidence,
                error=None,
            )
        )
        applied_case_ids.append(prediction.id)

    report: dict[str, Any] = {
        "schema_version": VLM_ROOM_LEVEL_TARGET_FALLBACK_REPORT_SCHEMA_VERSION,
        "request_bundle_digest": request_bundle.get("request_bundle_digest"),
        "request_bundle_path": (
            str(request_bundle_path) if request_bundle_path is not None else None
        ),
        "input_prediction_digest": qa_predictions_digest(predictions),
        "input_prediction_path": (
            str(prediction_path) if prediction_path is not None else None
        ),
        "output_prediction_digest": qa_predictions_digest(calibrated),
        "applied_case_ids": applied_case_ids,
        "missing_case_input_ids": missing_case_input_ids,
        "skipped_existing_location_case_ids": skipped_existing_location_case_ids,
        "skipped_ineligible_error_case_ids": skipped_ineligible_error_case_ids,
        "skipped_no_room_option_case_ids": skipped_no_room_option_case_ids,
        "skipped_non_room_level_target_case_ids": (
            skipped_non_room_level_target_case_ids
        ),
        "room_level_target_labels": sorted(_ROOM_LEVEL_TARGET_FALLBACK_LABELS),
        "summary": {
            "applied_fallback_count": len(applied_case_ids),
            "case_input_count": len(case_inputs),
            "input_prediction_count": len(predictions),
            "missing_case_input_count": len(missing_case_input_ids),
            "output_prediction_count": len(calibrated),
            "skipped_existing_location_count": len(
                skipped_existing_location_case_ids
            ),
            "skipped_ineligible_error_count": len(
                skipped_ineligible_error_case_ids
            ),
            "skipped_no_room_option_count": len(skipped_no_room_option_case_ids),
            "skipped_non_room_level_target_count": len(
                skipped_non_room_level_target_case_ids
            ),
        },
    }
    report["report_digest"] = vlm_room_level_target_fallback_report_digest(report)
    return calibrated, report


def vlm_room_level_target_fallback_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return _digest(payload)


def vlm_room_level_target_fallback_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n"


def save_vlm_room_level_target_fallback_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        vlm_room_level_target_fallback_report_json(report),
        encoding="utf-8",
    )
    return output_path


def load_vlm_room_level_target_fallback_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(
            "VLM room-level target fallback report must be a JSON object"
        )
    return cast(dict[str, Any], payload)


def validate_vlm_room_level_target_fallback_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _mapping(report.get("summary"))
    applied_case_ids = _string_list(report.get("applied_case_ids"))
    missing_case_input_ids = _string_list(report.get("missing_case_input_ids"))
    skipped_existing_location_case_ids = _string_list(
        report.get("skipped_existing_location_case_ids")
    )
    skipped_ineligible_error_case_ids = _string_list(
        report.get("skipped_ineligible_error_case_ids")
    )
    skipped_no_room_option_case_ids = _string_list(
        report.get("skipped_no_room_option_case_ids")
    )
    skipped_non_room_level_target_case_ids = _string_list(
        report.get("skipped_non_room_level_target_case_ids")
    )
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == VLM_ROOM_LEVEL_TARGET_FALLBACK_REPORT_SCHEMA_VERSION,
            "expected": VLM_ROOM_LEVEL_TARGET_FALLBACK_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "report_digest",
            "passed": report.get("report_digest")
            == vlm_room_level_target_fallback_report_digest(report),
            "expected": vlm_room_level_target_fallback_report_digest(report),
            "actual": report.get("report_digest"),
        },
        {
            "name": "applied_fallback_count",
            "passed": summary.get("applied_fallback_count")
            == len(applied_case_ids),
            "expected": len(applied_case_ids),
            "actual": summary.get("applied_fallback_count"),
        },
        {
            "name": "missing_case_input_count",
            "passed": summary.get("missing_case_input_count")
            == len(missing_case_input_ids),
            "expected": len(missing_case_input_ids),
            "actual": summary.get("missing_case_input_count"),
        },
        {
            "name": "skipped_existing_location_count",
            "passed": summary.get("skipped_existing_location_count")
            == len(skipped_existing_location_case_ids),
            "expected": len(skipped_existing_location_case_ids),
            "actual": summary.get("skipped_existing_location_count"),
        },
        {
            "name": "skipped_ineligible_error_count",
            "passed": summary.get("skipped_ineligible_error_count")
            == len(skipped_ineligible_error_case_ids),
            "expected": len(skipped_ineligible_error_case_ids),
            "actual": summary.get("skipped_ineligible_error_count"),
        },
        {
            "name": "skipped_no_room_option_count",
            "passed": summary.get("skipped_no_room_option_count")
            == len(skipped_no_room_option_case_ids),
            "expected": len(skipped_no_room_option_case_ids),
            "actual": summary.get("skipped_no_room_option_count"),
        },
        {
            "name": "skipped_non_room_level_target_count",
            "passed": summary.get("skipped_non_room_level_target_count")
            == len(skipped_non_room_level_target_case_ids),
            "expected": len(skipped_non_room_level_target_case_ids),
            "actual": summary.get("skipped_non_room_level_target_count"),
        },
    ]
    return {
        "action": "validate_vlm_room_level_target_fallback_report",
        "valid": all(check["passed"] for check in checks),
        "checks": checks,
    }


def vlm_single_support_option_fallback_predictions(
    predictions: Sequence[QAPrediction],
    request_bundle: Mapping[str, Any],
    *,
    prediction_path: str | Path | None = None,
    request_bundle_path: str | Path | None = None,
) -> tuple[list[QAPrediction], dict[str, Any]]:
    case_inputs = {
        str(case["case_id"]): case
        for case in _mapping_rows(request_bundle.get("case_inputs"))
        if isinstance(case.get("case_id"), str)
    }
    calibrated: list[QAPrediction] = []
    applied_case_ids: list[str] = []
    missing_case_input_ids: list[str] = []
    skipped_existing_location_case_ids: list[str] = []
    skipped_ineligible_error_case_ids: list[str] = []
    skipped_no_single_support_option_case_ids: list[str] = []
    skipped_room_level_target_case_ids: list[str] = []

    for prediction in predictions:
        case = case_inputs.get(prediction.id)
        if case is None:
            calibrated.append(prediction)
            missing_case_input_ids.append(prediction.id)
            continue
        if _prediction_has_usable_current_location(prediction):
            calibrated.append(prediction)
            skipped_existing_location_case_ids.append(prediction.id)
            continue
        if prediction.error not in {"target_not_observed", "relation_not_observed"}:
            calibrated.append(prediction)
            skipped_ineligible_error_case_ids.append(prediction.id)
            continue
        target_label = _canonical_destination_label(
            _optional_str(_mapping(case.get("target")).get("label"))
        )
        if target_label in _ROOM_LEVEL_TARGET_FALLBACK_LABELS:
            calibrated.append(prediction)
            skipped_room_level_target_case_ids.append(prediction.id)
            continue
        option = _single_support_candidate_option(case)
        if option is None:
            calibrated.append(prediction)
            skipped_no_single_support_option_case_ids.append(prediction.id)
            continue

        answer = dict(prediction.answer)
        answer["answer_option_id"] = option["option_id"]
        answer["current_location"] = {
            "dst": option["destination_label"],
            "dst_label": option["destination_label"],
            "relation": option["relation"],
            "step": _case_primary_step(case),
        }
        answer["vlm_single_support_option_fallback"] = {
            "case_id": prediction.id,
            "source": "single_support_candidate_option",
        }
        calibrated.append(
            QAPrediction(
                id=prediction.id,
                answer=answer,
                evidence_nodes=prediction.evidence_nodes,
                evidence_edges=prediction.evidence_edges,
                confidence=prediction.confidence,
                error=None,
            )
        )
        applied_case_ids.append(prediction.id)

    report: dict[str, Any] = {
        "schema_version": VLM_SINGLE_SUPPORT_OPTION_FALLBACK_REPORT_SCHEMA_VERSION,
        "request_bundle_digest": request_bundle.get("request_bundle_digest"),
        "request_bundle_path": (
            str(request_bundle_path) if request_bundle_path is not None else None
        ),
        "input_prediction_digest": qa_predictions_digest(predictions),
        "input_prediction_path": (
            str(prediction_path) if prediction_path is not None else None
        ),
        "output_prediction_digest": qa_predictions_digest(calibrated),
        "applied_case_ids": applied_case_ids,
        "missing_case_input_ids": missing_case_input_ids,
        "skipped_existing_location_case_ids": skipped_existing_location_case_ids,
        "skipped_ineligible_error_case_ids": skipped_ineligible_error_case_ids,
        "skipped_no_single_support_option_case_ids": (
            skipped_no_single_support_option_case_ids
        ),
        "skipped_room_level_target_case_ids": skipped_room_level_target_case_ids,
        "summary": {
            "applied_fallback_count": len(applied_case_ids),
            "case_input_count": len(case_inputs),
            "input_prediction_count": len(predictions),
            "missing_case_input_count": len(missing_case_input_ids),
            "output_prediction_count": len(calibrated),
            "skipped_existing_location_count": len(
                skipped_existing_location_case_ids
            ),
            "skipped_ineligible_error_count": len(
                skipped_ineligible_error_case_ids
            ),
            "skipped_no_single_support_option_count": len(
                skipped_no_single_support_option_case_ids
            ),
            "skipped_room_level_target_count": len(
                skipped_room_level_target_case_ids
            ),
        },
    }
    report["report_digest"] = vlm_single_support_option_fallback_report_digest(report)
    return calibrated, report


def vlm_single_support_option_fallback_report_digest(
    report: Mapping[str, Any],
) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return _digest(payload)


def vlm_single_support_option_fallback_report_json(
    report: Mapping[str, Any],
) -> str:
    return json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n"


def save_vlm_single_support_option_fallback_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        vlm_single_support_option_fallback_report_json(report),
        encoding="utf-8",
    )
    return output_path


def load_vlm_single_support_option_fallback_report(
    path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(
            "VLM single support option fallback report must be a JSON object"
        )
    return cast(dict[str, Any], payload)


def validate_vlm_single_support_option_fallback_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _mapping(report.get("summary"))
    applied_case_ids = _string_list(report.get("applied_case_ids"))
    missing_case_input_ids = _string_list(report.get("missing_case_input_ids"))
    skipped_existing_location_case_ids = _string_list(
        report.get("skipped_existing_location_case_ids")
    )
    skipped_ineligible_error_case_ids = _string_list(
        report.get("skipped_ineligible_error_case_ids")
    )
    skipped_no_single_support_option_case_ids = _string_list(
        report.get("skipped_no_single_support_option_case_ids")
    )
    skipped_room_level_target_case_ids = _string_list(
        report.get("skipped_room_level_target_case_ids")
    )
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == VLM_SINGLE_SUPPORT_OPTION_FALLBACK_REPORT_SCHEMA_VERSION,
            "expected": VLM_SINGLE_SUPPORT_OPTION_FALLBACK_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "report_digest",
            "passed": report.get("report_digest")
            == vlm_single_support_option_fallback_report_digest(report),
            "expected": vlm_single_support_option_fallback_report_digest(report),
            "actual": report.get("report_digest"),
        },
        {
            "name": "applied_fallback_count",
            "passed": summary.get("applied_fallback_count")
            == len(applied_case_ids),
            "expected": len(applied_case_ids),
            "actual": summary.get("applied_fallback_count"),
        },
        {
            "name": "missing_case_input_count",
            "passed": summary.get("missing_case_input_count")
            == len(missing_case_input_ids),
            "expected": len(missing_case_input_ids),
            "actual": summary.get("missing_case_input_count"),
        },
        {
            "name": "skipped_existing_location_count",
            "passed": summary.get("skipped_existing_location_count")
            == len(skipped_existing_location_case_ids),
            "expected": len(skipped_existing_location_case_ids),
            "actual": summary.get("skipped_existing_location_count"),
        },
        {
            "name": "skipped_ineligible_error_count",
            "passed": summary.get("skipped_ineligible_error_count")
            == len(skipped_ineligible_error_case_ids),
            "expected": len(skipped_ineligible_error_case_ids),
            "actual": summary.get("skipped_ineligible_error_count"),
        },
        {
            "name": "skipped_no_single_support_option_count",
            "passed": summary.get("skipped_no_single_support_option_count")
            == len(skipped_no_single_support_option_case_ids),
            "expected": len(skipped_no_single_support_option_case_ids),
            "actual": summary.get("skipped_no_single_support_option_count"),
        },
        {
            "name": "skipped_room_level_target_count",
            "passed": summary.get("skipped_room_level_target_count")
            == len(skipped_room_level_target_case_ids),
            "expected": len(skipped_room_level_target_case_ids),
            "actual": summary.get("skipped_room_level_target_count"),
        },
    ]
    return {
        "action": "validate_vlm_single_support_option_fallback_report",
        "valid": all(check["passed"] for check in checks),
        "checks": checks,
    }


def vlm_text_option_alignment_predictions(
    predictions: Sequence[QAPrediction],
    request_bundle: Mapping[str, Any],
    *,
    prediction_path: str | Path | None = None,
    request_bundle_path: str | Path | None = None,
) -> tuple[list[QAPrediction], dict[str, Any]]:
    case_inputs = {
        str(case["case_id"]): case
        for case in _mapping_rows(request_bundle.get("case_inputs"))
        if isinstance(case.get("case_id"), str)
    }
    calibrated: list[QAPrediction] = []
    aligned_case_ids: list[str] = []
    missing_case_input_ids: list[str] = []
    skipped_ambiguous_text_match_case_ids: list[str] = []
    skipped_existing_option_location_case_ids: list[str] = []
    skipped_ineligible_error_case_ids: list[str] = []
    skipped_missing_location_case_ids: list[str] = []
    skipped_no_text_match_case_ids: list[str] = []

    for prediction in predictions:
        case = case_inputs.get(prediction.id)
        if case is None:
            calibrated.append(prediction)
            missing_case_input_ids.append(prediction.id)
            continue
        if prediction.error is not None:
            calibrated.append(prediction)
            skipped_ineligible_error_case_ids.append(prediction.id)
            continue

        current = _prediction_location_semantics(prediction.answer)
        current_relation = _normalize_relation(current.get("relation"))
        current_label = _canonical_destination_label(
            _optional_str(current.get("destination_label"))
        )
        if current_relation is None or current_label is None:
            calibrated.append(prediction)
            skipped_missing_location_case_ids.append(prediction.id)
            continue
        if current_label in _NON_LOCATION_DESTINATION_LABELS:
            calibrated.append(prediction)
            skipped_missing_location_case_ids.append(prediction.id)
            continue

        options = _text_alignable_options(case, relation=current_relation)
        option_labels = {option["destination_label"] for option in options}
        if current_label in option_labels:
            calibrated.append(prediction)
            skipped_existing_option_location_case_ids.append(prediction.id)
            continue

        matches = _text_mentioned_options(prediction.answer, options)
        if not matches:
            calibrated.append(prediction)
            skipped_no_text_match_case_ids.append(prediction.id)
            continue
        if len(matches) != 1:
            calibrated.append(prediction)
            skipped_ambiguous_text_match_case_ids.append(prediction.id)
            continue

        option = matches[0]
        answer = dict(prediction.answer)
        answer["answer_option_id"] = option["option_id"]
        answer["current_location"] = {
            "dst": option["destination_label"],
            "dst_label": option["destination_label"],
            "relation": option["relation"],
            "step": _case_primary_step(case),
        }
        answer["vlm_text_option_alignment"] = {
            "case_id": prediction.id,
            "matched_destination_label": option["destination_label"],
            "source": "unique_text_mentioned_answer_option",
        }
        calibrated.append(
            QAPrediction(
                id=prediction.id,
                answer=answer,
                evidence_nodes=prediction.evidence_nodes,
                evidence_edges=prediction.evidence_edges,
                confidence=prediction.confidence,
                error=None,
            )
        )
        aligned_case_ids.append(prediction.id)

    report: dict[str, Any] = {
        "schema_version": VLM_TEXT_OPTION_ALIGNMENT_REPORT_SCHEMA_VERSION,
        "request_bundle_digest": request_bundle.get("request_bundle_digest"),
        "request_bundle_path": (
            str(request_bundle_path) if request_bundle_path is not None else None
        ),
        "input_prediction_digest": qa_predictions_digest(predictions),
        "input_prediction_path": (
            str(prediction_path) if prediction_path is not None else None
        ),
        "output_prediction_digest": qa_predictions_digest(calibrated),
        "aligned_case_ids": aligned_case_ids,
        "missing_case_input_ids": missing_case_input_ids,
        "skipped_ambiguous_text_match_case_ids": (
            skipped_ambiguous_text_match_case_ids
        ),
        "skipped_existing_option_location_case_ids": (
            skipped_existing_option_location_case_ids
        ),
        "skipped_ineligible_error_case_ids": skipped_ineligible_error_case_ids,
        "skipped_missing_location_case_ids": skipped_missing_location_case_ids,
        "skipped_no_text_match_case_ids": skipped_no_text_match_case_ids,
        "summary": {
            "aligned_prediction_count": len(aligned_case_ids),
            "case_input_count": len(case_inputs),
            "input_prediction_count": len(predictions),
            "missing_case_input_count": len(missing_case_input_ids),
            "output_prediction_count": len(calibrated),
            "skipped_ambiguous_text_match_count": len(
                skipped_ambiguous_text_match_case_ids
            ),
            "skipped_existing_option_location_count": len(
                skipped_existing_option_location_case_ids
            ),
            "skipped_ineligible_error_count": len(
                skipped_ineligible_error_case_ids
            ),
            "skipped_missing_location_count": len(skipped_missing_location_case_ids),
            "skipped_no_text_match_count": len(skipped_no_text_match_case_ids),
        },
    }
    report["report_digest"] = vlm_text_option_alignment_report_digest(report)
    return calibrated, report


def vlm_text_option_alignment_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return _digest(payload)


def vlm_text_option_alignment_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n"


def save_vlm_text_option_alignment_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        vlm_text_option_alignment_report_json(report),
        encoding="utf-8",
    )
    return output_path


def load_vlm_text_option_alignment_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("VLM text option alignment report must be a JSON object")
    return cast(dict[str, Any], payload)


def validate_vlm_text_option_alignment_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _mapping(report.get("summary"))
    aligned_case_ids = _string_list(report.get("aligned_case_ids"))
    missing_case_input_ids = _string_list(report.get("missing_case_input_ids"))
    skipped_ambiguous_text_match_case_ids = _string_list(
        report.get("skipped_ambiguous_text_match_case_ids")
    )
    skipped_existing_option_location_case_ids = _string_list(
        report.get("skipped_existing_option_location_case_ids")
    )
    skipped_ineligible_error_case_ids = _string_list(
        report.get("skipped_ineligible_error_case_ids")
    )
    skipped_missing_location_case_ids = _string_list(
        report.get("skipped_missing_location_case_ids")
    )
    skipped_no_text_match_case_ids = _string_list(
        report.get("skipped_no_text_match_case_ids")
    )
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == VLM_TEXT_OPTION_ALIGNMENT_REPORT_SCHEMA_VERSION,
            "expected": VLM_TEXT_OPTION_ALIGNMENT_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "report_digest",
            "passed": report.get("report_digest")
            == vlm_text_option_alignment_report_digest(report),
            "expected": vlm_text_option_alignment_report_digest(report),
            "actual": report.get("report_digest"),
        },
        {
            "name": "aligned_prediction_count",
            "passed": summary.get("aligned_prediction_count")
            == len(aligned_case_ids),
            "expected": len(aligned_case_ids),
            "actual": summary.get("aligned_prediction_count"),
        },
        {
            "name": "missing_case_input_count",
            "passed": summary.get("missing_case_input_count")
            == len(missing_case_input_ids),
            "expected": len(missing_case_input_ids),
            "actual": summary.get("missing_case_input_count"),
        },
        {
            "name": "skipped_ambiguous_text_match_count",
            "passed": summary.get("skipped_ambiguous_text_match_count")
            == len(skipped_ambiguous_text_match_case_ids),
            "expected": len(skipped_ambiguous_text_match_case_ids),
            "actual": summary.get("skipped_ambiguous_text_match_count"),
        },
        {
            "name": "skipped_existing_option_location_count",
            "passed": summary.get("skipped_existing_option_location_count")
            == len(skipped_existing_option_location_case_ids),
            "expected": len(skipped_existing_option_location_case_ids),
            "actual": summary.get("skipped_existing_option_location_count"),
        },
        {
            "name": "skipped_ineligible_error_count",
            "passed": summary.get("skipped_ineligible_error_count")
            == len(skipped_ineligible_error_case_ids),
            "expected": len(skipped_ineligible_error_case_ids),
            "actual": summary.get("skipped_ineligible_error_count"),
        },
        {
            "name": "skipped_missing_location_count",
            "passed": summary.get("skipped_missing_location_count")
            == len(skipped_missing_location_case_ids),
            "expected": len(skipped_missing_location_case_ids),
            "actual": summary.get("skipped_missing_location_count"),
        },
        {
            "name": "skipped_no_text_match_count",
            "passed": summary.get("skipped_no_text_match_count")
            == len(skipped_no_text_match_case_ids),
            "expected": len(skipped_no_text_match_case_ids),
            "actual": summary.get("skipped_no_text_match_count"),
        },
    ]
    return {
        "action": "validate_vlm_text_option_alignment_report",
        "valid": all(check["passed"] for check in checks),
        "checks": checks,
    }


def vlm_affordance_option_fallback_predictions(
    predictions: Sequence[QAPrediction],
    request_bundle: Mapping[str, Any],
    *,
    prediction_path: str | Path | None = None,
    request_bundle_path: str | Path | None = None,
) -> tuple[list[QAPrediction], dict[str, Any]]:
    case_inputs = {
        str(case["case_id"]): case
        for case in _mapping_rows(request_bundle.get("case_inputs"))
        if isinstance(case.get("case_id"), str)
    }
    calibrated: list[QAPrediction] = []
    applied_case_ids: list[str] = []
    missing_case_input_ids: list[str] = []
    skipped_ambiguous_affordance_option_case_ids: list[str] = []
    skipped_existing_location_case_ids: list[str] = []
    skipped_ineligible_error_case_ids: list[str] = []
    skipped_no_affordance_option_case_ids: list[str] = []

    for prediction in predictions:
        case = case_inputs.get(prediction.id)
        if case is None:
            calibrated.append(prediction)
            missing_case_input_ids.append(prediction.id)
            continue
        if prediction.error not in {"target_not_observed", "relation_not_observed"}:
            calibrated.append(prediction)
            skipped_ineligible_error_case_ids.append(prediction.id)
            continue
        if _prediction_has_usable_current_location(prediction):
            calibrated.append(prediction)
            skipped_existing_location_case_ids.append(prediction.id)
            continue
        option_result = _affordance_option_for_case(case)
        if isinstance(option_result, str):
            calibrated.append(prediction)
            skipped_ambiguous_affordance_option_case_ids.append(prediction.id)
            continue
        if option_result is None:
            calibrated.append(prediction)
            skipped_no_affordance_option_case_ids.append(prediction.id)
            continue

        option = option_result
        target_label = _canonical_destination_label(
            _optional_str(_mapping(case.get("target")).get("label"))
        )
        answer = dict(prediction.answer)
        answer["answer_option_id"] = option["option_id"]
        answer["current_location"] = {
            "dst": option["destination_label"],
            "dst_label": option["destination_label"],
            "relation": option["relation"],
            "step": _case_primary_step(case),
        }
        answer["vlm_affordance_option_fallback"] = {
            "case_id": prediction.id,
            "destination_label": option["destination_label"],
            "source": "target_affordance_public_option_prior",
            "target_label": target_label,
        }
        calibrated.append(
            QAPrediction(
                id=prediction.id,
                answer=answer,
                evidence_nodes=prediction.evidence_nodes,
                evidence_edges=prediction.evidence_edges,
                confidence=prediction.confidence,
                error=None,
            )
        )
        applied_case_ids.append(prediction.id)

    report: dict[str, Any] = {
        "schema_version": VLM_AFFORDANCE_OPTION_FALLBACK_REPORT_SCHEMA_VERSION,
        "request_bundle_digest": request_bundle.get("request_bundle_digest"),
        "request_bundle_path": (
            str(request_bundle_path) if request_bundle_path is not None else None
        ),
        "input_prediction_digest": qa_predictions_digest(predictions),
        "input_prediction_path": (
            str(prediction_path) if prediction_path is not None else None
        ),
        "output_prediction_digest": qa_predictions_digest(calibrated),
        "applied_case_ids": applied_case_ids,
        "missing_case_input_ids": missing_case_input_ids,
        "skipped_ambiguous_affordance_option_case_ids": (
            skipped_ambiguous_affordance_option_case_ids
        ),
        "skipped_existing_location_case_ids": skipped_existing_location_case_ids,
        "skipped_ineligible_error_case_ids": skipped_ineligible_error_case_ids,
        "skipped_no_affordance_option_case_ids": (
            skipped_no_affordance_option_case_ids
        ),
        "target_affordance_option_priors": {
            key: list(value)
            for key, value in sorted(_TARGET_AFFORDANCE_OPTION_PRIORS.items())
        },
        "summary": {
            "applied_fallback_count": len(applied_case_ids),
            "case_input_count": len(case_inputs),
            "input_prediction_count": len(predictions),
            "missing_case_input_count": len(missing_case_input_ids),
            "output_prediction_count": len(calibrated),
            "skipped_ambiguous_affordance_option_count": len(
                skipped_ambiguous_affordance_option_case_ids
            ),
            "skipped_existing_location_count": len(
                skipped_existing_location_case_ids
            ),
            "skipped_ineligible_error_count": len(
                skipped_ineligible_error_case_ids
            ),
            "skipped_no_affordance_option_count": len(
                skipped_no_affordance_option_case_ids
            ),
        },
    }
    report["report_digest"] = vlm_affordance_option_fallback_report_digest(report)
    return calibrated, report


def vlm_affordance_option_fallback_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return _digest(payload)


def vlm_affordance_option_fallback_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n"


def save_vlm_affordance_option_fallback_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        vlm_affordance_option_fallback_report_json(report),
        encoding="utf-8",
    )
    return output_path


def load_vlm_affordance_option_fallback_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError(
            "VLM affordance option fallback report must be a JSON object"
        )
    return cast(dict[str, Any], payload)


def validate_vlm_affordance_option_fallback_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _mapping(report.get("summary"))
    applied_case_ids = _string_list(report.get("applied_case_ids"))
    missing_case_input_ids = _string_list(report.get("missing_case_input_ids"))
    skipped_ambiguous_affordance_option_case_ids = _string_list(
        report.get("skipped_ambiguous_affordance_option_case_ids")
    )
    skipped_existing_location_case_ids = _string_list(
        report.get("skipped_existing_location_case_ids")
    )
    skipped_ineligible_error_case_ids = _string_list(
        report.get("skipped_ineligible_error_case_ids")
    )
    skipped_no_affordance_option_case_ids = _string_list(
        report.get("skipped_no_affordance_option_case_ids")
    )
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == VLM_AFFORDANCE_OPTION_FALLBACK_REPORT_SCHEMA_VERSION,
            "expected": VLM_AFFORDANCE_OPTION_FALLBACK_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "report_digest",
            "passed": report.get("report_digest")
            == vlm_affordance_option_fallback_report_digest(report),
            "expected": vlm_affordance_option_fallback_report_digest(report),
            "actual": report.get("report_digest"),
        },
        {
            "name": "applied_fallback_count",
            "passed": summary.get("applied_fallback_count")
            == len(applied_case_ids),
            "expected": len(applied_case_ids),
            "actual": summary.get("applied_fallback_count"),
        },
        {
            "name": "missing_case_input_count",
            "passed": summary.get("missing_case_input_count")
            == len(missing_case_input_ids),
            "expected": len(missing_case_input_ids),
            "actual": summary.get("missing_case_input_count"),
        },
        {
            "name": "skipped_ambiguous_affordance_option_count",
            "passed": summary.get("skipped_ambiguous_affordance_option_count")
            == len(skipped_ambiguous_affordance_option_case_ids),
            "expected": len(skipped_ambiguous_affordance_option_case_ids),
            "actual": summary.get("skipped_ambiguous_affordance_option_count"),
        },
        {
            "name": "skipped_existing_location_count",
            "passed": summary.get("skipped_existing_location_count")
            == len(skipped_existing_location_case_ids),
            "expected": len(skipped_existing_location_case_ids),
            "actual": summary.get("skipped_existing_location_count"),
        },
        {
            "name": "skipped_ineligible_error_count",
            "passed": summary.get("skipped_ineligible_error_count")
            == len(skipped_ineligible_error_case_ids),
            "expected": len(skipped_ineligible_error_case_ids),
            "actual": summary.get("skipped_ineligible_error_count"),
        },
        {
            "name": "skipped_no_affordance_option_count",
            "passed": summary.get("skipped_no_affordance_option_count")
            == len(skipped_no_affordance_option_case_ids),
            "expected": len(skipped_no_affordance_option_case_ids),
            "actual": summary.get("skipped_no_affordance_option_count"),
        },
    ]
    return {
        "action": "validate_vlm_affordance_option_fallback_report",
        "valid": all(check["passed"] for check in checks),
        "checks": checks,
    }


def vlm_answer_option_coverage_report(
    cases: Sequence[QACase],
    bundle: Mapping[str, Any],
    *,
    qa_path: str | Path | None = None,
    request_bundle_path: str | Path | None = None,
) -> dict[str, Any]:
    bundle_cases = {
        str(case.get("case_id")): case
        for case in _mapping_rows(bundle.get("case_inputs"))
        if isinstance(case.get("case_id"), str)
    }
    rows = [
        _answer_option_coverage_case_row(case, bundle_cases.get(case.id))
        for case in cases
    ]
    covered_count = sum(1 for row in rows if row.get("covered") is True)
    option_count = sum(_int_value(row.get("option_count")) for row in rows)
    missing_case_ids = [
        str(row["case_id"]) for row in rows if row.get("covered") is not True
    ]
    report: dict[str, Any] = {
        "schema_version": VLM_ANSWER_OPTION_COVERAGE_REPORT_SCHEMA_VERSION,
        "qa_digest": qa_dataset_digest(cases),
        "qa_path": str(qa_path) if qa_path is not None else None,
        "request_bundle_digest": bundle.get("request_bundle_digest"),
        "request_bundle_path": (
            str(request_bundle_path) if request_bundle_path is not None else None
        ),
        "summary": {
            "case_count": len(rows),
            "covered_case_count": covered_count,
            "covered_case_rate": _rate(covered_count, len(rows)),
            "missing_case_count": len(missing_case_ids),
            "option_count": option_count,
        },
        "missing_case_ids": missing_case_ids,
        "cases": rows,
    }
    report["report_digest"] = vlm_answer_option_coverage_report_digest(report)
    return report


def vlm_answer_option_coverage_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return _digest(payload)


def vlm_answer_option_coverage_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n"


def save_vlm_answer_option_coverage_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        vlm_answer_option_coverage_report_json(report),
        encoding="utf-8",
    )
    return output_path


def load_vlm_answer_option_coverage_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("VLM answer option coverage report must be a JSON object")
    return cast(dict[str, Any], payload)


def validate_vlm_answer_option_coverage_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    rows = _mapping_rows(report.get("cases"))
    summary = _mapping(report.get("summary"))
    covered_count = sum(1 for row in rows if row.get("covered") is True)
    missing_case_ids = [
        str(row["case_id"])
        for row in rows
        if isinstance(row.get("case_id"), str) and row.get("covered") is not True
    ]
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == VLM_ANSWER_OPTION_COVERAGE_REPORT_SCHEMA_VERSION,
            "expected": VLM_ANSWER_OPTION_COVERAGE_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "report_digest",
            "passed": report.get("report_digest")
            == vlm_answer_option_coverage_report_digest(report),
            "expected": vlm_answer_option_coverage_report_digest(report),
            "actual": report.get("report_digest"),
        },
        {
            "name": "case_count",
            "passed": summary.get("case_count") == len(rows),
            "expected": len(rows),
            "actual": summary.get("case_count"),
        },
        {
            "name": "covered_case_count",
            "passed": summary.get("covered_case_count") == covered_count,
            "expected": covered_count,
            "actual": summary.get("covered_case_count"),
        },
        {
            "name": "missing_case_ids",
            "passed": _string_list(report.get("missing_case_ids")) == missing_case_ids,
            "expected": missing_case_ids,
            "actual": _string_list(report.get("missing_case_ids")),
        },
    ]
    return {
        "action": "validate_vlm_answer_option_coverage_report",
        "valid": all(check["passed"] for check in checks),
        "checks": checks,
    }


def vlm_support_gap_report(
    cases: Sequence[QACase],
    semantic_eval_report: Mapping[str, Any],
    predicted_graph: DynamicSceneGraph,
    *,
    qa_path: str | Path | None = None,
    semantic_eval_report_path: str | Path | None = None,
    predicted_graph_path: str | Path | None = None,
) -> dict[str, Any]:
    semantic_rows = {
        str(row["case_id"]): row
        for row in _mapping_rows(semantic_eval_report.get("cases"))
        if isinstance(row.get("case_id"), str)
    }
    rows = [
        row
        for case in cases
        if (
            row := _support_gap_case_row(
                case,
                semantic_rows.get(case.id),
                predicted_graph,
            )
        )
        is not None
    ]
    support_missing_rows = [
        row for row in rows if row.get("gap_kind") == "support_missing"
    ]
    support_labels = sorted(
        {
            str(row["support_label"])
            for row in support_missing_rows
            if isinstance(row.get("support_label"), str)
        }
    )
    summary = {
        "case_count": len(rows),
        "failed_on_case_count": len(rows),
        "support_missing_count": len(support_missing_rows),
        "support_present_relation_missing_count": sum(
            1
            for row in rows
            if row.get("gap_kind") == "support_present_but_relation_missing"
        ),
        "target_missing_count": sum(
            1 for row in rows if row.get("gap_kind") == "target_missing"
        ),
    }
    report: dict[str, Any] = {
        "schema_version": VLM_SUPPORT_GAP_REPORT_SCHEMA_VERSION,
        "evaluator_only": True,
        "qa_digest": qa_dataset_digest(cases),
        "qa_path": str(qa_path) if qa_path is not None else None,
        "semantic_eval_report_digest": semantic_eval_report.get("report_digest"),
        "semantic_eval_report_path": (
            str(semantic_eval_report_path)
            if semantic_eval_report_path is not None
            else None
        ),
        "predicted_graph_digest": graph_json_digest(predicted_graph),
        "predicted_graph_path": (
            str(predicted_graph_path) if predicted_graph_path is not None else None
        ),
        "support_labels_to_collect": support_labels,
        "summary": summary,
        "cases": rows,
    }
    report["report_digest"] = vlm_support_gap_report_digest(report)
    return report


def vlm_support_gap_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return _digest(payload)


def vlm_support_gap_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n"


def save_vlm_support_gap_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(vlm_support_gap_report_json(report), encoding="utf-8")
    return output_path


def load_vlm_support_gap_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("VLM support gap report must be a JSON object")
    return cast(dict[str, Any], payload)


def validate_vlm_support_gap_report(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = _mapping_rows(report.get("cases"))
    summary = _mapping(report.get("summary"))
    support_missing_count = sum(
        1 for row in rows if row.get("gap_kind") == "support_missing"
    )
    support_present_relation_missing_count = sum(
        1
        for row in rows
        if row.get("gap_kind") == "support_present_but_relation_missing"
    )
    target_missing_count = sum(
        1 for row in rows if row.get("gap_kind") == "target_missing"
    )
    support_labels_to_collect = sorted(
        {
            str(row["support_label"])
            for row in rows
            if row.get("gap_kind") == "support_missing"
            and isinstance(row.get("support_label"), str)
        }
    )
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == VLM_SUPPORT_GAP_REPORT_SCHEMA_VERSION,
            "expected": VLM_SUPPORT_GAP_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "report_digest",
            "passed": report.get("report_digest") == vlm_support_gap_report_digest(report),
            "expected": vlm_support_gap_report_digest(report),
            "actual": report.get("report_digest"),
        },
        {
            "name": "evaluator_only",
            "passed": report.get("evaluator_only") is True,
            "expected": True,
            "actual": report.get("evaluator_only"),
        },
        {
            "name": "case_count",
            "passed": summary.get("case_count") == len(rows),
            "expected": len(rows),
            "actual": summary.get("case_count"),
        },
        {
            "name": "failed_on_case_count",
            "passed": summary.get("failed_on_case_count") == len(rows),
            "expected": len(rows),
            "actual": summary.get("failed_on_case_count"),
        },
        {
            "name": "support_missing_count",
            "passed": summary.get("support_missing_count") == support_missing_count,
            "expected": support_missing_count,
            "actual": summary.get("support_missing_count"),
        },
        {
            "name": "support_present_relation_missing_count",
            "passed": summary.get("support_present_relation_missing_count")
            == support_present_relation_missing_count,
            "expected": support_present_relation_missing_count,
            "actual": summary.get("support_present_relation_missing_count"),
        },
        {
            "name": "target_missing_count",
            "passed": summary.get("target_missing_count") == target_missing_count,
            "expected": target_missing_count,
            "actual": summary.get("target_missing_count"),
        },
        {
            "name": "support_labels_to_collect",
            "passed": _string_list(report.get("support_labels_to_collect"))
            == support_labels_to_collect,
            "expected": support_labels_to_collect,
            "actual": _string_list(report.get("support_labels_to_collect")),
        },
    ]
    return {
        "action": "validate_vlm_support_gap_report",
        "valid": all(check["passed"] for check in checks),
        "checks": checks,
    }


def vlm_support_detector_handoff(
    detector_handoff: Mapping[str, Any],
    support_candidate_request_bundle: Mapping[str, Any],
    *,
    max_support_labels_per_frame: int = 6,
) -> dict[str, Any]:
    if max_support_labels_per_frame < 0:
        raise SpatialQAError("max_support_labels_per_frame must be non-negative")
    enriched = json.loads(
        json.dumps(detector_handoff, separators=(",", ":"), sort_keys=True)
    )
    if not isinstance(enriched, Mapping):
        raise SpatialQAError("Detector handoff must be a JSON object")
    result = cast(dict[str, Any], enriched)
    labels_by_frame = _support_labels_by_primary_frame(support_candidate_request_bundle)
    frames: list[dict[str, Any]] = []
    frames_with_support = 0
    support_label_count = 0
    for frame in _mapping_rows(result.get("required_frames")):
        mutable_frame = dict(frame)
        key = _detector_handoff_frame_key(mutable_frame)
        labels = labels_by_frame.get(key, ()) if key is not None else ()
        if labels:
            support_labels = list(labels[:max_support_labels_per_frame])
            mutable_frame["support_labels"] = support_labels
            frames_with_support += 1
            support_label_count += len(support_labels)
        else:
            mutable_frame.pop("support_labels", None)
        frames.append(mutable_frame)
    result["required_frames"] = frames
    result["vlm_support_detector_handoff_enrichment"] = {
        "schema_version": VLM_SUPPORT_DETECTOR_HANDOFF_ENRICHMENT_SCHEMA_VERSION,
        "support_candidate_request_bundle_digest": support_candidate_request_bundle.get(
            "request_bundle_digest"
        ),
        "max_support_labels_per_frame": max_support_labels_per_frame,
        "summary": {
            "frame_count": len(frames),
            "frames_with_support_labels": frames_with_support,
            "support_label_count": support_label_count,
        },
    }
    result["support_detector_handoff_digest"] = vlm_support_detector_handoff_digest(
        result
    )
    return result


def vlm_support_detector_handoff_digest(handoff: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in handoff.items()
        if key != "support_detector_handoff_digest"
    }
    return _digest(payload)


def vlm_support_detector_handoff_json(handoff: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(handoff), indent=2, sort_keys=True) + "\n"


def save_vlm_support_detector_handoff(
    handoff: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(vlm_support_detector_handoff_json(handoff), encoding="utf-8")
    return output_path


def vlm_target_crop_request_bundle(
    bundle: Mapping[str, Any],
    detector_records: Sequence[Mapping[str, Any]],
    *,
    crop_root: str | Path,
    padding_pixels: int = 4,
) -> dict[str, Any]:
    if padding_pixels < 0:
        raise SpatialQAError("padding_pixels must be non-negative")
    enriched = json.loads(
        json.dumps(bundle, separators=(",", ":"), sort_keys=True)
    )
    if not isinstance(enriched, Mapping):
        raise SpatialQAError("VLM request bundle must be a JSON object")
    result = cast(dict[str, Any], enriched)
    crop_root_path = Path(crop_root)
    detection_index = _target_detection_index(detector_records)
    cases: list[dict[str, Any]] = []
    crop_count = 0
    missing_count = 0
    for case in _mapping_rows(result.get("case_inputs")):
        mutable_case = dict(case)
        crop = _target_crop_for_case(
            mutable_case,
            detection_index,
            crop_root=crop_root_path,
            padding_pixels=padding_pixels,
        )
        if crop is None:
            mutable_case.pop("target_crop", None)
            missing_count += 1
        else:
            mutable_case["target_crop"] = crop
            crop_count += 1
        cases.append(mutable_case)
    result["case_inputs"] = cases
    result["case_count"] = len(cases)
    result["vlm_target_crop_enrichment"] = {
        "schema_version": VLM_TARGET_CROP_ENRICHMENT_SCHEMA_VERSION,
        "detector_records_digest": _digest(
            {"detector_records": [_json_value(record) for record in detector_records]}
        ),
        "padding_pixels": padding_pixels,
        "crop_root": str(crop_root_path),
        "summary": {
            "case_count": len(cases),
            "cases_with_target_crop": crop_count,
            "missing_target_detection_count": missing_count,
        },
    }
    result["request_bundle_digest"] = vlm_target_crop_request_bundle_digest(result)
    return result


def vlm_target_crop_request_bundle_digest(bundle: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in bundle.items() if key != "request_bundle_digest"}
    return _digest(payload)


def vlm_target_crop_request_bundle_json(bundle: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(bundle), indent=2, sort_keys=True) + "\n"


def save_vlm_target_crop_request_bundle(
    bundle: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(vlm_target_crop_request_bundle_json(bundle), encoding="utf-8")
    return output_path


def vlm_primary_frame_visibility_report(
    request_bundle: Mapping[str, Any],
    frame_index: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    visible_by_frame = _visible_object_ids_by_frame(frame_index)
    rows: list[dict[str, Any]] = []
    visible_case_ids: list[str] = []
    primary_missing_count = 0
    for case in _mapping_rows(request_bundle.get("case_inputs")):
        if case.get("question_type") != "object_location":
            continue
        target = _mapping(case.get("target"))
        target_id = _optional_str(target.get("object_id"))
        primary_frame = _mapping(case.get("primary_frame"))
        primary_key = _primary_frame_key(primary_frame)
        primary_missing = primary_key is None
        if primary_missing:
            primary_missing_count += 1
        target_visible = (
            target_id is not None
            and primary_key is not None
            and target_id in visible_by_frame.get(primary_key, frozenset())
        )
        if target_visible:
            visible_case_ids.append(str(case.get("case_id")))
        rows.append(
            {
                "case_id": str(case.get("case_id")),
                "primary_frame": (
                    {
                        "episode_id": primary_key[0],
                        "scene_id": primary_key[1],
                        "step": primary_key[2],
                    }
                    if primary_key is not None
                    else None
                ),
                "target_label": _optional_str(target.get("label")),
                "target_object_id": target_id,
                "target_visible_in_primary_frame": target_visible,
            }
        )
    report: dict[str, Any] = {
        "schema_version": VLM_PRIMARY_FRAME_VISIBILITY_REPORT_SCHEMA_VERSION,
        "summary": {
            "case_count": len(_mapping_rows(request_bundle.get("case_inputs"))),
            "object_location_case_count": len(rows),
            "primary_frame_missing_count": primary_missing_count,
            "target_visible_primary_frame_count": len(visible_case_ids),
            "target_visible_primary_frame_rate": _rate(len(visible_case_ids), len(rows)),
        },
        "target_visible_primary_frame_case_ids": visible_case_ids,
        "cases": rows,
    }
    report["report_digest"] = vlm_primary_frame_visibility_report_digest(report)
    return report


def vlm_primary_frame_visibility_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return _digest(payload)


def vlm_primary_frame_visibility_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_vlm_primary_frame_visibility_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        vlm_primary_frame_visibility_report_json(report),
        encoding="utf-8",
    )
    return output_path


def validate_vlm_primary_frame_visibility_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    rows = _mapping_rows(report.get("cases"))
    visible_ids = _string_list(report.get("target_visible_primary_frame_case_ids"))
    expected_visible_ids = [
        str(row["case_id"])
        for row in rows
        if row.get("target_visible_in_primary_frame") is True
    ]
    summary = _mapping(report.get("summary"))
    expected_digest = vlm_primary_frame_visibility_report_digest(report)
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == VLM_PRIMARY_FRAME_VISIBILITY_REPORT_SCHEMA_VERSION,
            "expected": VLM_PRIMARY_FRAME_VISIBILITY_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "report_digest",
            "passed": report.get("report_digest") == expected_digest,
            "expected": expected_digest,
            "actual": report.get("report_digest"),
        },
        {
            "name": "object_location_case_count",
            "passed": summary.get("object_location_case_count") == len(rows),
            "expected": len(rows),
            "actual": summary.get("object_location_case_count"),
        },
        {
            "name": "target_visible_primary_frame_case_ids",
            "passed": visible_ids == expected_visible_ids,
            "expected": expected_visible_ids,
            "actual": visible_ids,
        },
        {
            "name": "target_visible_primary_frame_count",
            "passed": summary.get("target_visible_primary_frame_count")
            == len(expected_visible_ids),
            "expected": len(expected_visible_ids),
            "actual": summary.get("target_visible_primary_frame_count"),
        },
    ]
    return {
        "action": "validate_vlm_primary_frame_visibility_report",
        "valid": all(check["passed"] for check in checks),
        "checks": checks,
    }


def vlm_frame_index_rows_from_detector_records(
    detector_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        (
            _detector_record_frame_index_row(record, record_index=index)
            for index, record in enumerate(detector_records, start=1)
        ),
        key=lambda row: (
            str(row["episode_id"]),
            str(row["scene_id"]),
            _required_int(row.get("step"), "frame index step"),
            str(_mapping(row.get("asset_paths")).get("rgb")),
        ),
    )


def merge_vlm_frame_index_rows(
    existing_rows: Sequence[Mapping[str, Any]],
    additional_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for row in (*existing_rows, *additional_rows):
        normalized = dict(row)
        key = _frame_index_row_key(normalized)
        merged[key] = normalized
    return sorted(
        merged.values(),
        key=lambda row: (
            str(row["episode_id"]),
            str(row["scene_id"]),
            _required_int(row.get("step"), "frame index step"),
            str(_mapping(row.get("asset_paths")).get("rgb")),
        ),
    )


def vlm_frame_index_rows_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            [_json_value(row) for row in rows],
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def vlm_frame_index_rows_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(_json_value(row), separators=(",", ":"), sort_keys=True) + "\n"
        for row in rows
    )


def save_vlm_frame_index_rows(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(vlm_frame_index_rows_jsonl(rows), encoding="utf-8")
    return output_path


def vlm_frame_index_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_path: str | Path | None = None,
    existing_frame_index_path: str | Path | None = None,
) -> dict[str, Any]:
    visible_ids = sorted(
        {
            object_id
            for row in rows
            for object_id in _string_list(row.get("visible_object_ids"))
        }
    )
    report: dict[str, Any] = {
        "schema_version": VLM_FRAME_INDEX_REPORT_SCHEMA_VERSION,
        "existing_frame_index_path": (
            str(existing_frame_index_path)
            if existing_frame_index_path is not None
            else None
        ),
        "frame_index_digest": vlm_frame_index_rows_digest(rows),
        "summary": {
            "frame_count": len(rows),
            "source_path": str(source_path) if source_path is not None else None,
            "target_visible_frame_count": sum(
                1 for row in rows if _string_list(row.get("visible_object_ids"))
            ),
            "visible_object_id_count": len(visible_ids),
        },
    }
    report["report_digest"] = vlm_frame_index_report_digest(report)
    return report


def vlm_frame_index_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return _digest(cast(Mapping[str, Any], payload))


def vlm_frame_index_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n"


def save_vlm_frame_index_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(vlm_frame_index_report_json(report), encoding="utf-8")
    return output_path


def load_vlm_frame_index_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("VLM frame index report must be a JSON object")
    return cast(dict[str, Any], payload)


def validate_vlm_frame_index_report(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(report.get("summary"))
    expected_digest = vlm_frame_index_report_digest(report)
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == VLM_FRAME_INDEX_REPORT_SCHEMA_VERSION,
            "expected": VLM_FRAME_INDEX_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "report_digest",
            "passed": report.get("report_digest") == expected_digest,
            "expected": expected_digest,
            "actual": report.get("report_digest"),
        },
        {
            "name": "frame_index_digest",
            "passed": isinstance(report.get("frame_index_digest"), str)
            and report.get("frame_index_digest") != "",
            "expected": "non-empty digest",
            "actual": report.get("frame_index_digest"),
        },
        {
            "name": "frame_count",
            "passed": _non_negative_int(summary.get("frame_count")),
            "expected": "non-negative integer",
            "actual": summary.get("frame_count"),
        },
        {
            "name": "target_visible_frame_count",
            "passed": _non_negative_int(summary.get("target_visible_frame_count")),
            "expected": "non-negative integer",
            "actual": summary.get("target_visible_frame_count"),
        },
        {
            "name": "visible_object_id_count",
            "passed": _non_negative_int(summary.get("visible_object_id_count")),
            "expected": "non-negative integer",
            "actual": summary.get("visible_object_id_count"),
        },
    ]
    return {
        "action": "validate_vlm_frame_index_report",
        "valid": all(check["passed"] for check in checks),
        "checks": checks,
    }


def vlm_semantic_eval_report(
    cases: Sequence[QACase],
    predictions: Sequence[QAPrediction],
    *,
    include_hidden: bool = False,
    gold_path: str | Path | None = None,
    prediction_path: str | Path | None = None,
) -> dict[str, Any]:
    prediction_by_id = {prediction.id: prediction for prediction in predictions}
    rows: list[dict[str, Any]] = []
    excluded_case_ids: list[str] = []
    strict_exact_count = 0
    semantic_match_count = 0
    matched_prediction_count = 0
    for case in cases:
        included, exclusion_reason = _case_included_for_semantic_eval(
            case,
            include_hidden=include_hidden,
        )
        if not included:
            excluded_case_ids.append(case.id)
            continue
        prediction = prediction_by_id.get(case.id)
        row = _semantic_case_row(case, prediction)
        rows.append(row)
        if prediction is not None:
            matched_prediction_count += 1
        if row["strict_exact_match"] is True:
            strict_exact_count += 1
        if row["semantic_match"] is True:
            semantic_match_count += 1
        if exclusion_reason is not None:
            row["exclusion_reason"] = exclusion_reason
    report: dict[str, Any] = {
        "schema_version": VLM_SEMANTIC_EVAL_REPORT_SCHEMA_VERSION,
        "gold_digest": qa_dataset_digest(cases),
        "gold_path": str(gold_path) if gold_path is not None else None,
        "prediction_digest": qa_predictions_digest(predictions),
        "prediction_path": str(prediction_path) if prediction_path is not None else None,
        "include_hidden": include_hidden,
        "summary": {
            "case_count": len(rows),
            "excluded_case_count": len(excluded_case_ids),
            "matched_prediction_count": matched_prediction_count,
            "prediction_count": len(predictions),
            "semantic_match_count": semantic_match_count,
            "semantic_match_rate": _rate(semantic_match_count, len(rows)),
            "strict_exact_match_count": strict_exact_count,
            "strict_exact_match_rate": _rate(strict_exact_count, len(rows)),
        },
        "excluded_case_ids": excluded_case_ids,
        "cases": rows,
    }
    report["report_digest"] = vlm_semantic_eval_report_digest(report)
    return report


def vlm_semantic_eval_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return _digest(payload)


def vlm_semantic_eval_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def vlm_semantic_eval_delta_report(
    candidate_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    *,
    candidate_name: str = "candidate",
    baseline_name: str = "baseline",
) -> dict[str, Any]:
    candidate_summary = _mapping(candidate_report.get("summary"))
    baseline_summary = _mapping(baseline_report.get("summary"))
    candidate_cases = _semantic_case_match_map(candidate_report)
    baseline_cases = _semantic_case_match_map(baseline_report)
    shared_case_ids = sorted(set(candidate_cases) & set(baseline_cases))
    paired_wins = sum(
        1
        for case_id in shared_case_ids
        if candidate_cases[case_id] is True and baseline_cases[case_id] is not True
    )
    paired_losses = sum(
        1
        for case_id in shared_case_ids
        if candidate_cases[case_id] is not True and baseline_cases[case_id] is True
    )
    paired_ties = len(shared_case_ids) - paired_wins - paired_losses
    candidate_count = _int_summary_value(candidate_summary, "semantic_match_count")
    baseline_count = _int_summary_value(baseline_summary, "semantic_match_count")
    candidate_rate = _float_summary_value(candidate_summary, "semantic_match_rate")
    baseline_rate = _float_summary_value(baseline_summary, "semantic_match_rate")
    report: dict[str, Any] = {
        "schema_version": VLM_SEMANTIC_EVAL_DELTA_REPORT_SCHEMA_VERSION,
        "baseline_name": baseline_name,
        "baseline_report_digest": baseline_report.get("report_digest"),
        "baseline_semantic_eval_report_path": baseline_report.get("prediction_path"),
        "candidate_name": candidate_name,
        "candidate_report_digest": candidate_report.get("report_digest"),
        "candidate_semantic_eval_report_path": candidate_report.get("prediction_path"),
        "case_count_match": candidate_summary.get("case_count")
        == baseline_summary.get("case_count"),
        "gold_digest_match": candidate_report.get("gold_digest")
        == baseline_report.get("gold_digest"),
        "summary_delta": {
            "baseline_semantic_match_count": baseline_count,
            "baseline_semantic_match_rate": baseline_rate,
            "candidate_semantic_match_count": candidate_count,
            "candidate_semantic_match_rate": candidate_rate,
            "semantic_match_count_delta": candidate_count - baseline_count,
            "semantic_match_rate_delta": _round_float(candidate_rate - baseline_rate),
        },
        "paired": {
            "case_count": len(shared_case_ids),
            "paired_losses": paired_losses,
            "paired_ties": paired_ties,
            "paired_wins": paired_wins,
        },
        "paired_significance": _paired_significance_report(
            paired_wins,
            paired_losses,
        ),
        "question_type_groups": _semantic_delta_question_type_groups(
            candidate_report,
            baseline_report,
            candidate_cases,
            baseline_cases,
            shared_case_ids,
        ),
        "decision": _semantic_delta_decision(paired_wins, paired_losses),
    }
    report["report_digest"] = vlm_semantic_eval_delta_report_digest(report)
    return report


def vlm_semantic_eval_delta_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return _digest(payload)


def vlm_semantic_eval_delta_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def validate_vlm_semantic_eval_delta_report(report: Mapping[str, Any]) -> dict[str, Any]:
    summary_delta = _mapping(report.get("summary_delta"))
    paired = _mapping(report.get("paired"))
    paired_significance = _mapping(report.get("paired_significance"))
    expected_digest = vlm_semantic_eval_delta_report_digest(report)
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == VLM_SEMANTIC_EVAL_DELTA_REPORT_SCHEMA_VERSION,
            "expected": VLM_SEMANTIC_EVAL_DELTA_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "report_digest",
            "passed": report.get("report_digest") == expected_digest,
            "expected": expected_digest,
            "actual": report.get("report_digest"),
        },
        {
            "name": "gold_digest_match",
            "passed": isinstance(report.get("gold_digest_match"), bool),
            "expected": "boolean",
            "actual": report.get("gold_digest_match"),
        },
        {
            "name": "case_count_match",
            "passed": isinstance(report.get("case_count_match"), bool),
            "expected": "boolean",
            "actual": report.get("case_count_match"),
        },
        {
            "name": "semantic_match_count_delta",
            "passed": isinstance(
                summary_delta.get("semantic_match_count_delta"),
                int,
            )
            and not isinstance(summary_delta.get("semantic_match_count_delta"), bool),
            "expected": "integer",
            "actual": summary_delta.get("semantic_match_count_delta"),
        },
        {
            "name": "paired_case_count",
            "passed": isinstance(paired.get("case_count"), int)
            and not isinstance(paired.get("case_count"), bool),
            "expected": "integer",
            "actual": paired.get("case_count"),
        },
        {
            "name": "paired_significance_method",
            "passed": paired_significance.get("method")
            == "exact_paired_sign_test_mcnemar_like",
            "expected": "exact_paired_sign_test_mcnemar_like",
            "actual": paired_significance.get("method"),
        },
        {
            "name": "paired_significance_p_value",
            "passed": isinstance(
                paired_significance.get("two_sided_p_value"),
                (int, float),
            )
            and not isinstance(paired_significance.get("two_sided_p_value"), bool),
            "expected": "number",
            "actual": paired_significance.get("two_sided_p_value"),
        },
        {
            "name": "question_type_groups",
            "passed": all(
                isinstance(row.get("question_type"), str)
                and isinstance(row.get("paired"), Mapping)
                and isinstance(row.get("paired_significance"), Mapping)
                for row in _mapping_rows(report.get("question_type_groups"))
            ),
            "expected": "list of question type group rows",
            "actual": report.get("question_type_groups"),
        },
    ]
    return {
        "action": "validate_vlm_semantic_eval_delta_report",
        "valid": all(check["passed"] for check in checks),
        "checks": checks,
    }


def save_vlm_semantic_eval_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(vlm_semantic_eval_report_json(report), encoding="utf-8")
    return output_path


def load_vlm_semantic_eval_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("VLM semantic eval report must be a JSON object")
    return cast(dict[str, Any], payload)


def validate_vlm_semantic_eval_report(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = _mapping_rows(report.get("cases"))
    semantic_count = sum(1 for row in rows if row.get("semantic_match") is True)
    strict_count = sum(1 for row in rows if row.get("strict_exact_match") is True)
    summary = _mapping(report.get("summary"))
    expected_digest = vlm_semantic_eval_report_digest(report)
    checks = [
        {
            "name": "schema_version",
            "passed": report.get("schema_version")
            == VLM_SEMANTIC_EVAL_REPORT_SCHEMA_VERSION,
            "expected": VLM_SEMANTIC_EVAL_REPORT_SCHEMA_VERSION,
            "actual": report.get("schema_version"),
        },
        {
            "name": "report_digest",
            "passed": report.get("report_digest") == expected_digest,
            "expected": expected_digest,
            "actual": report.get("report_digest"),
        },
        {
            "name": "case_count",
            "passed": summary.get("case_count") == len(rows),
            "expected": len(rows),
            "actual": summary.get("case_count"),
        },
        {
            "name": "semantic_match_count",
            "passed": summary.get("semantic_match_count") == semantic_count,
            "expected": semantic_count,
            "actual": summary.get("semantic_match_count"),
        },
        {
            "name": "strict_exact_match_count",
            "passed": summary.get("strict_exact_match_count") == strict_count,
            "expected": strict_count,
            "actual": summary.get("strict_exact_match_count"),
        },
    ]
    return {
        "action": "validate_vlm_semantic_eval_report",
        "valid": all(check["passed"] for check in checks),
        "checks": checks,
    }


def run_vlm_calibration(
    *,
    qa_path: str | Path,
    prediction_path: str | Path | None = None,
    request_bundle_path: str | Path | None = None,
    observable_qa_output: str | Path | None = None,
    observable_request_bundle_output: str | Path | None = None,
    observable_slice_report_output: str | Path | None = None,
    semantic_eval_report_output: str | Path | None = None,
) -> dict[str, Any]:
    cases = load_qa_dataset(qa_path)
    slice_report = vlm_observable_slice_report(cases, qa_path=qa_path)
    observable_ids = _string_list(slice_report.get("observable_case_ids"))
    if observable_slice_report_output is not None:
        save_vlm_observable_slice_report(slice_report, observable_slice_report_output)
    if observable_qa_output is not None:
        save_vlm_observable_qa_dataset(cases, observable_ids, observable_qa_output)
    request_bundle_digest = None
    if request_bundle_path is not None and observable_request_bundle_output is not None:
        bundle = _load_json_mapping(request_bundle_path)
        filtered_bundle = vlm_observable_request_bundle(
            bundle,
            observable_case_ids=observable_ids,
        )
        save_vlm_observable_request_bundle(
            filtered_bundle,
            observable_request_bundle_output,
        )
        request_bundle_digest = filtered_bundle["request_bundle_digest"]
    semantic_report = None
    if prediction_path is not None:
        predictions = load_qa_predictions(prediction_path)
        semantic_report = vlm_semantic_eval_report(
            cases,
            predictions,
            gold_path=qa_path,
            prediction_path=prediction_path,
        )
        if semantic_eval_report_output is not None:
            save_vlm_semantic_eval_report(
                semantic_report,
                semantic_eval_report_output,
            )
    return {
        "action": "run_vlm_calibration",
        "observable_case_count": len(observable_ids),
        "observable_case_ids": observable_ids,
        "observable_slice_report_digest": slice_report["report_digest"],
        "observable_request_bundle_digest": request_bundle_digest,
        "semantic_eval_report_digest": (
            semantic_report["report_digest"] if semantic_report is not None else None
        ),
        "semantic_summary": (
            semantic_report["summary"] if semantic_report is not None else None
        ),
    }


def _observable_case_row(
    case: QACase,
    *,
    allowed_relations: frozenset[str],
) -> dict[str, Any]:
    included, reason = _case_included_for_semantic_eval(
        case,
        include_hidden=False,
        allowed_relations=allowed_relations,
    )
    semantics = _gold_location_semantics(case)
    return {
        "case_id": case.id,
        "question_type": case.question_type,
        "answer_type": case.answer_type,
        "included": included,
        "exclusion_reason": reason,
        "gold": semantics,
    }


def _case_included_for_semantic_eval(
    case: QACase,
    *,
    include_hidden: bool,
    allowed_relations: frozenset[str] = VLM_VISIBLE_RELATIONS,
) -> tuple[bool, str | None]:
    if case.question_type != "object_location":
        return False, "unsupported_question_type"
    semantics = _gold_location_semantics(case)
    if semantics["relation"] not in allowed_relations:
        return False, "unsupported_relation"
    if semantics["visible"] is not True and include_hidden is not True:
        return False, "target_not_visible"
    if semantics["destination_label"] is None:
        return False, "destination_missing"
    return True, None


def _semantic_case_row(
    case: QACase,
    prediction: QAPrediction | None,
) -> dict[str, Any]:
    gold = _gold_location_semantics(case)
    if prediction is None:
        prediction_semantics = _empty_prediction_semantics("missing_prediction")
        strict_exact = False
        semantic_match = False
        failure_reason: str | None = "missing_prediction"
    else:
        prediction_semantics = _prediction_location_semantics(prediction.answer)
        strict_exact = prediction.error is None and prediction.answer == case.answer
        semantic_match = (
            prediction.error is None
            and _location_semantics_match(gold, prediction_semantics)
        )
        failure_reason = None if semantic_match else _semantic_failure_reason(
            prediction.error,
            gold,
            prediction_semantics,
        )
    return {
        "case_id": case.id,
        "question_type": case.question_type,
        "strict_exact_match": strict_exact,
        "semantic_match": semantic_match,
        "failure_reason": failure_reason,
        "gold": gold,
        "prediction": prediction_semantics,
    }


def _gold_location_semantics(case: QACase) -> dict[str, Any]:
    answer = _mapping(case.answer)
    location = _mapping(answer.get("current_location"))
    relation = _normalize_relation(location.get("relation"))
    dst = _optional_str(location.get("dst"))
    return {
        "destination_label": _destination_label_from_id(dst),
        "relation": relation,
        "target_label": _canonical_destination_label(_optional_str(answer.get("label"))),
        "visible": answer.get("visible") is True,
    }


def _prediction_location_semantics(answer: Mapping[str, Any]) -> dict[str, Any]:
    location = _mapping(answer.get("current_location"))
    relation = _normalize_relation(
        location.get("relation")
        or answer.get("relation")
        or answer.get("relation_label")
    )
    destination = (
        location.get("dst")
        or location.get("dst_label")
        or location.get("destination_label")
        or answer.get("destination_label")
        or answer.get("location_label")
        or answer.get("place")
    )
    raw_text = _answer_text(answer)
    text_relation, text_destination = _parse_location_text(raw_text)
    if relation is None:
        relation = text_relation
    destination_label = _destination_label_from_id(_optional_str(destination))
    if destination_label is None:
        destination_label = text_destination
    return {
        "destination_label": destination_label,
        "raw_text": raw_text,
        "relation": relation,
    }


def _empty_prediction_semantics(reason: str) -> dict[str, Any]:
    return {
        "destination_label": None,
        "failure_reason": reason,
        "raw_text": None,
        "relation": None,
    }


def _semantic_case_match_map(report: Mapping[str, Any]) -> dict[str, bool]:
    matches: dict[str, bool] = {}
    for row in _mapping_rows(report.get("cases")):
        case_id = row.get("case_id")
        if isinstance(case_id, str) and case_id not in matches:
            matches[case_id] = row.get("semantic_match") is True
    return matches


def _semantic_delta_decision(paired_wins: int, paired_losses: int) -> str:
    if paired_wins > paired_losses:
        return "candidate_improved"
    if paired_losses > paired_wins:
        return "candidate_regressed"
    return "candidate_unchanged"


def _paired_significance_report(paired_wins: int, paired_losses: int) -> dict[str, Any]:
    discordant_count = paired_wins + paired_losses
    p_value = _exact_two_sided_sign_test_p_value(paired_wins, paired_losses)
    return {
        "candidate_loss_count": paired_losses,
        "candidate_win_count": paired_wins,
        "discordant_case_count": discordant_count,
        "method": "exact_paired_sign_test_mcnemar_like",
        "significant_at_0_05": p_value < 0.05,
        "two_sided_p_value": _round_float(p_value),
    }


def _exact_two_sided_sign_test_p_value(paired_wins: int, paired_losses: int) -> float:
    discordant_count = paired_wins + paired_losses
    if discordant_count == 0:
        return 1.0
    smaller_side_count = min(paired_wins, paired_losses)
    tail_probability = float(
        sum(
        comb(discordant_count, count) for count in range(smaller_side_count + 1)
        )
    ) / float(2**discordant_count)
    return min(1.0, 2.0 * tail_probability)


def _semantic_delta_question_type_groups(
    candidate_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    candidate_cases: Mapping[str, bool],
    baseline_cases: Mapping[str, bool],
    shared_case_ids: Sequence[str],
) -> list[dict[str, Any]]:
    candidate_rows = _semantic_case_row_map(candidate_report)
    baseline_rows = _semantic_case_row_map(baseline_report)
    grouped_case_ids: dict[str, list[str]] = {}
    for case_id in shared_case_ids:
        question_type = _semantic_question_type(
            candidate_rows.get(case_id),
            baseline_rows.get(case_id),
        )
        grouped_case_ids.setdefault(question_type, []).append(case_id)
    groups: list[dict[str, Any]] = []
    for question_type in sorted(grouped_case_ids):
        case_ids = grouped_case_ids[question_type]
        paired_wins = sum(
            1
            for case_id in case_ids
            if candidate_cases[case_id] is True and baseline_cases[case_id] is not True
        )
        paired_losses = sum(
            1
            for case_id in case_ids
            if candidate_cases[case_id] is not True and baseline_cases[case_id] is True
        )
        paired_ties = len(case_ids) - paired_wins - paired_losses
        candidate_count = sum(1 for case_id in case_ids if candidate_cases[case_id] is True)
        baseline_count = sum(1 for case_id in case_ids if baseline_cases[case_id] is True)
        groups.append(
            {
                "baseline_semantic_match_count": baseline_count,
                "candidate_semantic_match_count": candidate_count,
                "case_count": len(case_ids),
                "decision": _semantic_delta_decision(paired_wins, paired_losses),
                "paired": {
                    "case_count": len(case_ids),
                    "paired_losses": paired_losses,
                    "paired_ties": paired_ties,
                    "paired_wins": paired_wins,
                },
                "paired_significance": _paired_significance_report(
                    paired_wins,
                    paired_losses,
                ),
                "question_type": question_type,
                "semantic_match_count_delta": candidate_count - baseline_count,
            }
        )
    return groups


def _semantic_case_row_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for row in _mapping_rows(report.get("cases")):
        case_id = _optional_str(row.get("case_id"))
        if case_id is not None and case_id not in rows:
            rows[case_id] = row
    return rows


def _semantic_question_type(
    candidate_row: Mapping[str, Any] | None,
    baseline_row: Mapping[str, Any] | None,
) -> str:
    if candidate_row is not None:
        candidate_question_type = _optional_str(candidate_row.get("question_type"))
        if candidate_question_type is not None:
            return candidate_question_type
    if baseline_row is not None:
        baseline_question_type = _optional_str(baseline_row.get("question_type"))
        if baseline_question_type is not None:
            return baseline_question_type
    return "unknown"


def _int_summary_value(summary: Mapping[str, Any], key: str) -> int:
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _int_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _float_summary_value(summary: Mapping[str, Any], key: str) -> float:
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _round_float(value: float) -> float:
    return round(value, 6)


def _semantic_failure_reason(
    prediction_error: str | None,
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> str:
    if prediction_error is not None:
        return prediction_error
    if prediction.get("relation") is None and prediction.get("destination_label") is None:
        return "location_not_parseable"
    if not _relation_matches(gold, prediction):
        return "relation_mismatch"
    if not _destination_matches(gold, prediction):
        return "destination_mismatch"
    return "semantic_mismatch"


def _location_semantics_match(
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> bool:
    return (
        _room_floor_location_matches(gold, prediction)
        or _room_level_specific_support_matches(gold, prediction)
        or (
            _relation_matches(gold, prediction)
            and _destination_matches(gold, prediction)
        )
    )


def _room_floor_location_matches(
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> bool:
    return (
        gold.get("relation") == "IN_ROOM"
        and _optional_str(gold.get("destination_label")) == "room"
        and prediction.get("relation") == "ON"
        and _optional_str(prediction.get("destination_label")) == "floor"
    )


def _room_level_specific_support_matches(
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> bool:
    if (
        gold.get("relation") != "IN_ROOM"
        or _optional_str(gold.get("destination_label")) != "room"
    ):
        return False
    key = (
        _optional_str(gold.get("target_label")) or "",
        _optional_str(prediction.get("relation")) or "",
        _optional_str(prediction.get("destination_label")) or "",
    )
    return key in _ROOM_LEVEL_SPECIFIC_SUPPORT_MATCHES


def _relation_matches(gold: Mapping[str, Any], prediction: Mapping[str, Any]) -> bool:
    gold_relation = gold.get("relation")
    prediction_relation = prediction.get("relation")
    if gold_relation == prediction_relation:
        return True
    return gold_relation == "IN_ROOM" and prediction_relation == "INSIDE"


def _destination_matches(gold: Mapping[str, Any], prediction: Mapping[str, Any]) -> bool:
    gold_label = _optional_str(gold.get("destination_label"))
    prediction_label = _optional_str(prediction.get("destination_label"))
    if gold_label is None or prediction_label is None:
        return False
    if gold_label == prediction_label:
        return True
    return gold_label == "room" and prediction_label in {
        "bathroom",
        "bedroom",
        "diningroom",
        "kitchen",
        "livingroom",
        "room",
    }


def _parse_location_text(value: str | None) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    normalized = _normalize_text(value)
    patterns = (
        ("ON", r"\bon top of\b\s+(?P<dst>.+)$"),
        ("ON", r"\bon\b\s+(?P<dst>.+)$"),
        ("INSIDE", r"\binside\b\s+(?P<dst>.+)$"),
        ("INSIDE", r"\bin\b\s+(?P<dst>.+)$"),
    )
    for relation, pattern in patterns:
        match = re.search(pattern, normalized)
        if match is None:
            continue
        destination = _canonical_destination_label(match.group("dst"))
        if relation == "INSIDE" and destination in {
            "bathroom",
            "bedroom",
            "kitchen",
            "livingroom",
            "room",
        }:
            return "IN_ROOM", destination
        return relation, destination
    return None, _canonical_destination_label(normalized)


def _answer_text(answer: Mapping[str, Any]) -> str | None:
    for key in ("answer_text", "text", "location_text"):
        value = answer.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _destination_label_from_id(value: str | None) -> str | None:
    if value is None:
        return None
    if value == "ai2thor_room":
        return "room"
    parts = value.split("_")
    label_parts: list[str] = []
    for part in parts:
        if part == "" or part[:1].isdigit():
            break
        label_parts.append(part)
    label = "_".join(label_parts) if label_parts else value
    return _canonical_destination_label(label)


def _canonical_destination_label(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _normalize_text(value)
    normalized = re.sub(r"\b(the|a|an)\b", " ", normalized)
    normalized = re.sub(r"\b(on|in|inside|top|of|at|near|under)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    aliases = {
        "bath room": "bathroom",
        "counter": "countertop",
        "counter top": "countertop",
        "kitchen counter": "countertop",
        "living room": "livingroom",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == "":
        return None
    return normalized.replace(" ", "")


def _normalize_text(value: str) -> str:
    lowered = value.lower().replace("_", " ").replace("-", " ")
    return re.sub(r"[^a-z0-9 ]+", " ", lowered)


def _normalize_relation(value: object) -> str | None:
    if not isinstance(value, str) or value == "":
        return None
    normalized = value.strip().upper().replace(" ", "_")
    aliases = {
        "IN": "INSIDE",
        "IN_THE_ROOM": "IN_ROOM",
        "ROOM": "IN_ROOM",
        "ON_TOP_OF": "ON",
    }
    return aliases.get(normalized, normalized)


def _detector_record_frame_index_row(
    record: Mapping[str, Any],
    *,
    record_index: int,
) -> dict[str, Any]:
    metadata = _mapping(record.get("metadata"))
    episode_id = _required_text(
        record.get("episode_id") or metadata.get("episode_id"),
        f"detector record {record_index} episode_id",
    )
    scene_id = _required_text(
        record.get("scene_id") or metadata.get("scene_id"),
        f"detector record {record_index} scene_id",
    )
    step = _required_int(record.get("step"), f"detector record {record_index} step")
    rgb_path = _optional_str(record.get("rgb_path"))
    depth_path = _optional_str(record.get("depth_path"))
    segmentation_path = _optional_str(record.get("segmentation_path"))
    visible_detections = _visible_detector_detections(record.get("detections"))
    visible_ids = sorted(
        {
            object_id
            for detection in visible_detections
            if (object_id := _optional_str(detection.get("object_id"))) is not None
        }
    )
    visible_labels = sorted(
        {
            label
            for detection in visible_detections
            if (label := _optional_str(detection.get("label"))) is not None
        }
    )
    asset_paths = _asset_path_mapping(
        rgb_path=rgb_path,
        depth_path=depth_path,
        segmentation_path=segmentation_path,
    )
    row: dict[str, Any] = {
        "schema_version": REAL_FRAME_TRACE_SCHEMA_VERSION,
        "dataset_id": _optional_str(metadata.get("dataset_id"))
        or _optional_str(record.get("dataset_id"))
        or "detector_frame_index",
        "episode_id": episode_id,
        "scene_id": scene_id,
        "step": step,
        "asset_paths": asset_paths,
        "asset_present": {
            key: Path(value).exists()
            for key, value in asset_paths.items()
        },
        "detector_object_count": len(visible_ids),
        "visible_object_ids": visible_ids,
        "visible_object_labels": visible_labels,
    }
    source_name = (
        _optional_str(metadata.get("source_name"))
        or _optional_str(record.get("detector_name"))
        or _optional_str(metadata.get("detector"))
    )
    if source_name is not None:
        row["source_name"] = source_name
    if rgb_path is not None:
        row["detector_rgb_path"] = rgb_path
    if depth_path is not None:
        row["detector_depth_path"] = depth_path
    if segmentation_path is not None:
        row["detector_segmentation_path"] = segmentation_path
    return row


def _visible_detector_detections(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    detections: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        if item.get("visible") is False:
            continue
        detections.append(cast(Mapping[str, Any], item))
    return detections


def _asset_path_mapping(
    *,
    rgb_path: str | None,
    depth_path: str | None,
    segmentation_path: str | None,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    if depth_path is not None:
        paths["depth"] = depth_path
    if rgb_path is not None:
        paths["rgb"] = rgb_path
    if segmentation_path is not None:
        paths["segmentation"] = segmentation_path
    return paths


def _frame_index_row_key(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    asset_paths = _mapping(row.get("asset_paths"))
    return (
        _required_text(row.get("episode_id"), "frame index episode_id"),
        _required_text(row.get("scene_id"), "frame index scene_id"),
        _required_int(row.get("step"), "frame index step"),
        str(asset_paths.get("rgb") or row.get("detector_rgb_path") or ""),
    )


def _visible_object_ids_by_frame(
    frame_index: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int], frozenset[str]]:
    visible_by_frame: dict[tuple[str, str, int], frozenset[str]] = {}
    for row in frame_index:
        key = _primary_frame_key(row)
        if key is None:
            continue
        visible_by_frame[key] = frozenset(_string_list(row.get("visible_object_ids")))
    return visible_by_frame


def _visible_object_labels_by_frame(
    frame_index: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int], tuple[str, ...]]:
    labels_by_frame: dict[tuple[str, str, int], list[str]] = {}
    for row in frame_index:
        key = _primary_frame_key(row)
        if key is None:
            continue
        labels = labels_by_frame.setdefault(key, [])
        for label in _string_list(row.get("visible_object_labels")):
            canonical = _canonical_destination_label(label)
            if canonical is not None and canonical not in labels:
                labels.append(canonical)
        for object_id in _string_list(row.get("visible_object_ids")):
            canonical = _destination_label_from_id(object_id)
            if canonical is not None and canonical not in labels:
                labels.append(canonical)
    return {key: tuple(labels) for key, labels in labels_by_frame.items()}


def _support_labels_by_primary_frame(
    request_bundle: Mapping[str, Any],
) -> dict[tuple[str, str, int], tuple[str, ...]]:
    labels_by_frame: dict[tuple[str, str, int], list[str]] = {}
    for case in _mapping_rows(request_bundle.get("case_inputs")):
        key = _primary_frame_key(_mapping(case.get("primary_frame")))
        if key is None:
            continue
        labels = labels_by_frame.setdefault(key, [])
        for candidate in _mapping_rows(case.get("support_candidates")):
            label = _canonical_destination_label(_optional_str(candidate.get("label")))
            if label is not None and label not in labels:
                labels.append(label)
    return {key: tuple(labels) for key, labels in labels_by_frame.items()}


def _answer_options_for_case(
    case: Mapping[str, Any],
    *,
    include_room_option: bool,
    include_ambiguous_support_relations: bool,
    include_target_affordance_options: bool,
    max_options: int,
) -> list[dict[str, Any]]:
    if case.get("question_type") not in {None, "object_location"}:
        return []
    options: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add_option(relation: str, label: str, source: str) -> None:
        if relation not in {"IN_ROOM", "INSIDE", "ON"}:
            return
        key = (relation, label)
        if key in seen:
            return
        seen.add(key)
        options.append(
            {
                "destination_label": label,
                "relation": relation,
                "source": source,
            }
        )

    for candidate in _mapping_rows(case.get("support_candidates")):
        label = _canonical_destination_label(_optional_str(candidate.get("label")))
        relation = _normalize_relation(candidate.get("relation_hint")) or "ON"
        if label is None or relation not in {"IN_ROOM", "INSIDE", "ON"}:
            continue
        add_option(relation, label, "support_candidate")
        if include_ambiguous_support_relations:
            for additional_relation in _ADDITIONAL_SUPPORT_RELATION_HINTS.get(
                label,
                (),
            ):
                add_option(additional_relation, label, "ambiguous_support_relation")
    if include_target_affordance_options:
        target_label = _canonical_destination_label(
            _optional_str(_mapping(case.get("target")).get("label"))
        )
        for relation, label in _TARGET_AFFORDANCE_ANSWER_OPTIONS.get(
            target_label or "",
            (),
        ):
            add_option(relation, label, "target_affordance_prior")
    if include_room_option and ("IN_ROOM", "room") not in seen:
        options.append(
            {
                "destination_label": "room",
                "relation": "IN_ROOM",
                "source": "fallback_room",
            }
        )
    limited = options[:max_options]
    return [
        {
            "destination_label": option["destination_label"],
            "option_id": f"locopt_{index:03d}",
            "relation": option["relation"],
            "source": option["source"],
        }
        for index, option in enumerate(limited, start=1)
    ]


def _answer_option_coverage_case_row(
    case: QACase,
    bundle_case: Mapping[str, Any] | None,
) -> dict[str, Any]:
    gold = _gold_location_semantics(case)
    if bundle_case is None:
        return {
            "case_id": case.id,
            "covered": False,
            "failure_reason": "missing_case_input",
            "gold": gold,
            "matched_option_id": None,
            "matched_option_source": None,
            "option_count": 0,
        }
    options = _mapping_rows(bundle_case.get("answer_options"))
    option_rows = [_answer_option_row(option) for option in options]
    gold_relation = gold.get("relation")
    gold_label = _optional_str(gold.get("destination_label"))
    matched = next(
        (
            option
            for option in option_rows
            if option.get("relation") == gold_relation
            and option.get("destination_label") == gold_label
        ),
        None,
    )
    if matched is not None:
        failure_reason = None
    elif not option_rows:
        failure_reason = "no_answer_options"
    elif gold_label is None or gold_relation is None:
        failure_reason = "gold_location_missing"
    else:
        failure_reason = "gold_option_missing"
    return {
        "case_id": case.id,
        "covered": matched is not None,
        "failure_reason": failure_reason,
        "gold": gold,
        "matched_option_id": matched.get("option_id") if matched is not None else None,
        "matched_option_source": (
            matched.get("source") if matched is not None else None
        ),
        "option_count": len(option_rows),
        "options": option_rows,
    }


def _answer_option_row(option: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "destination_label": _canonical_destination_label(
            _optional_str(option.get("destination_label"))
        ),
        "option_id": _optional_str(option.get("option_id")),
        "relation": _normalize_relation(option.get("relation")),
        "source": _optional_str(option.get("source")),
    }


def _support_gap_case_row(
    case: QACase,
    semantic_row: Mapping[str, Any] | None,
    predicted_graph: DynamicSceneGraph,
) -> dict[str, Any] | None:
    if semantic_row is None or semantic_row.get("semantic_match") is True:
        return None
    gold = _gold_location_semantics(case)
    if gold.get("relation") != "ON":
        return None
    support_label = _optional_str(gold.get("destination_label"))
    if support_label is None:
        return None
    target_id = _case_target_object_id(case)
    support_node_ids = _support_node_ids_for_label(
        predicted_graph,
        support_label,
        scene_id=case.scene_id,
    )
    matching_edge_ids = (
        _matching_location_edge_ids(predicted_graph, target_id, support_node_ids)
        if target_id is not None
        else []
    )
    target_present = target_id is not None and target_id in predicted_graph.nodes
    if not target_present:
        gap_kind = "target_missing"
    elif not support_node_ids:
        gap_kind = "support_missing"
    elif matching_edge_ids:
        gap_kind = "relation_present_eval_mismatch"
    else:
        gap_kind = "support_present_but_relation_missing"
    return {
        "case_id": case.id,
        "episode_id": case.episode_id,
        "evaluator_only": True,
        "failure_reason": _optional_str(semantic_row.get("failure_reason")),
        "gap_kind": gap_kind,
        "gold_location": {
            "destination_label": support_label,
            "relation": "ON",
        },
        "matching_location_edge_ids": matching_edge_ids,
        "scene_id": case.scene_id,
        "step": case.step,
        "support_label": support_label,
        "support_node_ids": support_node_ids,
        "target_detector_evidence_ready": (
            _node_has_detector_rgbd_evidence(predicted_graph, target_id)
            if target_id is not None
            else False
        ),
        "target_label": _case_target_label(case),
        "target_object_id": target_id,
        "target_present": target_present,
    }


def _case_target_object_id(case: QACase) -> str | None:
    question_object_id = _optional_str(_mapping(case.question).get("object_id"))
    if question_object_id is not None:
        return question_object_id
    return _optional_str(_mapping(case.answer).get("object_id"))


def _case_target_label(case: QACase) -> str | None:
    return _canonical_destination_label(_optional_str(_mapping(case.answer).get("label")))


def _support_node_ids_for_label(
    graph: DynamicSceneGraph,
    support_label: str,
    *,
    scene_id: str,
) -> list[str]:
    return sorted(
        node.id
        for node in graph.nodes.values()
        if node.type == "object"
        and _canonical_destination_label(node.label) == support_label
        and node.attributes.get("scene_id") == scene_id
    )


def _matching_location_edge_ids(
    graph: DynamicSceneGraph,
    target_id: str,
    support_node_ids: Sequence[str],
) -> list[str]:
    support_ids = set(support_node_ids)
    return sorted(
        edge.id
        for edge in graph.edges
        if edge.src == target_id
        and edge.dst in support_ids
        and edge.relation in CONTAINMENT_RELATIONS
    )


def _node_has_detector_rgbd_evidence(
    graph: DynamicSceneGraph,
    node_id: str | None,
) -> bool:
    if node_id is None:
        return False
    node = graph.nodes.get(node_id)
    if node is None or node.attributes.get("source_kind") != "detector":
        return False
    evidence_kinds = node.attributes.get("evidence_kinds")
    return isinstance(evidence_kinds, list) and {
        "depth",
        "detector",
        "rgb",
    }.issubset(set(evidence_kinds))


def _sanitize_support_candidates(case: dict[str, Any]) -> None:
    candidates: list[dict[str, Any]] = []
    for candidate in _mapping_rows(case.get("support_candidates")):
        label = _canonical_destination_label(_optional_str(candidate.get("label")))
        if label is None:
            continue
        clean: dict[str, Any] = {"label": label}
        relation = _normalize_relation(candidate.get("relation_hint"))
        if relation is not None:
            clean["relation_hint"] = relation
        confidence = candidate.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            clean["confidence"] = float(confidence)
        candidates.append(clean)
    if candidates:
        case["support_candidates"] = candidates
    else:
        case.pop("support_candidates", None)


def _add_answer_option_contract(
    case: dict[str, Any],
    options: Sequence[Mapping[str, Any]],
) -> None:
    hint = dict(_mapping(case.get("answer_schema_hint")))
    hint["answer_option_rule"] = (
        "When answer_options is non-empty, choose answer.answer_option_id from "
        "allowed_answer_option_ids and copy that option's relation and "
        "destination_label into answer.current_location. Use room only for large "
        "standalone objects or when no more specific visible support is clear."
    )
    hint["non_gold_rule"] = (
        "answer_options are non-gold visible candidate labels. They are not gold "
        "answers. Do not use hidden object ids or evaluator-only fields."
    )
    option_ids = [
        option_id
        for option in options
        if (option_id := _optional_str(option.get("option_id"))) is not None
    ]
    hint["allowed_answer_option_ids"] = option_ids
    case["answer_schema_hint"] = hint
    case["answer_option_response_schema"] = {
        "answer_current_location_rule": (
            "Copy relation and destination_label from the selected answer option."
        ),
        "answer_option_id_field": "answer.answer_option_id",
        "allowed_answer_option_ids": option_ids,
        "required_when_answer_options_present": True,
    }


def _detector_handoff_frame_key(frame: Mapping[str, Any]) -> tuple[str, str, int] | None:
    episode_id = _optional_str(frame.get("episode_id"))
    scene_id = _optional_str(frame.get("scene_id"))
    step = frame.get("frame_step")
    if (
        episode_id is None
        or scene_id is None
        or isinstance(step, bool)
        or not isinstance(step, int)
    ):
        return None
    return (episode_id, scene_id, step)


def _target_detection_index(
    detector_records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int, str], tuple[Mapping[str, Any], Mapping[str, Any]]]:
    index: dict[tuple[str, str, int, str], tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for record in detector_records:
        key = _detector_record_frame_key(record)
        if key is None:
            continue
        detections = sorted(
            _mapping_rows(record.get("detections")),
            key=lambda detection: (
                -_float_value(detection.get("confidence"), default=0.0),
                str(detection.get("detection_id") or detection.get("object_id") or ""),
            ),
        )
        for detection in detections:
            if detection.get("visible") is not True:
                continue
            object_id = _optional_str(detection.get("object_id"))
            if object_id is None:
                continue
            detection_key = (*key, object_id)
            index.setdefault(detection_key, (record, detection))
    return index


def _detector_record_frame_key(
    record: Mapping[str, Any],
) -> tuple[str, str, int] | None:
    metadata = _mapping(record.get("metadata"))
    episode_id = _optional_str(record.get("episode_id")) or _optional_str(
        metadata.get("episode_id")
    )
    scene_id = _optional_str(record.get("scene_id")) or _optional_str(
        metadata.get("scene_id")
    )
    step = record.get("step")
    if (
        episode_id is None
        or scene_id is None
        or isinstance(step, bool)
        or not isinstance(step, int)
    ):
        return None
    return (episode_id, scene_id, step)


def _target_crop_for_case(
    case: Mapping[str, Any],
    detection_index: Mapping[
        tuple[str, str, int, str],
        tuple[Mapping[str, Any], Mapping[str, Any]],
    ],
    *,
    crop_root: Path,
    padding_pixels: int,
) -> dict[str, Any] | None:
    primary_key = _primary_frame_key(_mapping(case.get("primary_frame")))
    object_id = _optional_str(_mapping(case.get("target")).get("object_id"))
    if primary_key is None or object_id is None:
        return None
    match = detection_index.get((*primary_key, object_id))
    if match is None:
        return None
    record, detection = match
    rgb_path = _optional_str(record.get("rgb_path"))
    if rgb_path is None:
        return None
    pixels = _read_ppm_pixels(Path(rgb_path))
    image_width = len(pixels[0]) if pixels else 0
    image_height = len(pixels)
    detection_bbox = _optional_bbox_2d_xyxy(detection.get("bbox_2d_xyxy"))
    if detection_bbox is None:
        detection_bbox = _optional_bbox_2d_xyxy(
            _mapping(detection.get("attributes")).get("bbox_2d_xyxy")
        )
    if detection_bbox is None:
        return None
    bbox = _expanded_bbox(
        _bbox_to_image_space(
            detection_bbox,
            width=image_width,
            height=image_height,
        ),
        width=image_width,
        height=image_height,
        padding=padding_pixels,
    )
    if bbox is None:
        return None
    case_id = _required_text(case.get("case_id"), "case_id")
    crop_path = (
        crop_root
        / primary_key[0]
        / f"{primary_key[2]:06d}-{_safe_path_token(case_id)}.ppm"
    )
    _write_ppm_crop(pixels, bbox, crop_path)
    source_frame_id = _optional_str(_mapping(case.get("primary_frame")).get("frame_id"))
    if source_frame_id is None:
        source_frame_id = f"{primary_key[0]}:{primary_key[1]}:{primary_key[2]:04d}"
    crop: dict[str, Any] = {
        "bbox_2d_xyxy": list(bbox),
        "confidence": _float_value(detection.get("confidence"), default=0.0),
        "detector_name": _detector_record_name(record),
        "rgb_path": str(crop_path),
        "source": "detector_bbox_crop",
        "source_frame_id": source_frame_id,
    }
    return crop


def _detector_record_name(record: Mapping[str, Any]) -> str:
    metadata = _mapping(record.get("metadata"))
    return (
        _optional_str(record.get("detector_name"))
        or _optional_str(metadata.get("detector"))
        or _optional_str(metadata.get("source_name"))
        or "detector"
    )


def _bbox_2d_xyxy(value: object) -> tuple[int, int, int, int]:
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 4:
        raise SpatialQAError("bbox_2d_xyxy must contain four numbers")
    numbers: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise SpatialQAError("bbox_2d_xyxy values must be numeric")
        numbers.append(int(round(float(item))))
    x1, y1, x2, y2 = numbers
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _optional_bbox_2d_xyxy(value: object) -> tuple[int, int, int, int] | None:
    try:
        return _bbox_2d_xyxy(value)
    except SpatialQAError:
        return None


def _bbox_to_image_space(
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    if width <= 0 or height <= 0:
        return bbox
    if x2 < width and y2 < height:
        return bbox
    max_coord = max(x1, y1, x2, y2)
    if max_coord < 100 or max_coord > 1000:
        return bbox
    return (
        int(round((x1 / 1000) * width)),
        int(round((y1 / 1000) * height)),
        int(round((x2 / 1000) * width)),
        int(round((y2 / 1000) * height)),
    )


def _expanded_bbox(
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    padding: int,
) -> tuple[int, int, int, int] | None:
    if width <= 0 or height <= 0:
        raise SpatialQAError("PPM image must be non-empty")
    x1, y1, x2, y2 = bbox
    if x2 < 0 or y2 < 0 or x1 >= width or y1 >= height:
        return None
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width - 1, x2 + padding),
        min(height - 1, y2 + padding),
    )


def _read_ppm_pixels(path: Path) -> list[list[tuple[int, int, int]]]:
    data = path.read_bytes()
    magic, offset = _ppm_token(data, 0)
    width_token, offset = _ppm_token(data, offset)
    height_token, offset = _ppm_token(data, offset)
    max_value_token, offset = _ppm_token(data, offset)
    if magic not in {b"P3", b"P6"}:
        raise SpatialQAError("PPM image must use P3 or P6 format")
    width = _positive_int_token(width_token, "PPM width")
    height = _positive_int_token(height_token, "PPM height")
    max_value = _positive_int_token(max_value_token, "PPM max value")
    if max_value != 255:
        raise SpatialQAError("PPM max value must be 255")
    if magic == b"P6":
        while offset < len(data) and data[offset] in b" \t\r\n":
            offset += 1
        payload = data[offset : offset + width * height * 3]
        if len(payload) != width * height * 3:
            raise SpatialQAError("P6 PPM payload size does not match dimensions")
        rows: list[list[tuple[int, int, int]]] = []
        cursor = 0
        for _ in range(height):
            row: list[tuple[int, int, int]] = []
            for _ in range(width):
                row.append((payload[cursor], payload[cursor + 1], payload[cursor + 2]))
                cursor += 3
            rows.append(row)
        return rows
    tokens: list[bytes] = []
    while len(tokens) < width * height * 3:
        token, offset = _ppm_token(data, offset)
        tokens.append(token)
    channels = [_positive_or_zero_int_token(token, "P3 PPM channel") for token in tokens]
    if any(channel > 255 for channel in channels):
        raise SpatialQAError("P3 PPM channel must be in 0..255")
    rows = []
    cursor = 0
    for _ in range(height):
        row = []
        for _ in range(width):
            row.append((channels[cursor], channels[cursor + 1], channels[cursor + 2]))
            cursor += 3
        rows.append(row)
    return rows


def _write_ppm_crop(
    pixels: Sequence[Sequence[tuple[int, int, int]]],
    bbox: tuple[int, int, int, int],
    path: Path,
) -> None:
    x1, y1, x2, y2 = bbox
    crop_rows = [row[x1 : x2 + 1] for row in pixels[y1 : y2 + 1]]
    width = len(crop_rows[0]) if crop_rows else 0
    height = len(crop_rows)
    if width <= 0 or height <= 0:
        raise SpatialQAError("target crop must be non-empty")
    lines = [
        " ".join(f"{r} {g} {b}" for r, g, b in row)
        for row in crop_rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"P3\n{width} {height}\n255\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _ppm_token(data: bytes, offset: int) -> tuple[bytes, int]:
    cursor = offset
    while cursor < len(data):
        byte = data[cursor]
        if byte in b" \t\r\n":
            cursor += 1
            continue
        if byte == ord("#"):
            while cursor < len(data) and data[cursor] not in b"\r\n":
                cursor += 1
            continue
        break
    if cursor >= len(data):
        raise SpatialQAError("PPM file ended before expected token")
    start = cursor
    while cursor < len(data) and data[cursor] not in b" \t\r\n":
        cursor += 1
    return data[start:cursor], cursor


def _positive_int_token(token: bytes, field: str) -> int:
    try:
        value = int(token.decode("ascii"))
    except ValueError as exc:
        raise SpatialQAError(f"{field} must be an integer") from exc
    if value <= 0:
        raise SpatialQAError(f"{field} must be positive")
    return value


def _positive_or_zero_int_token(token: bytes, field: str) -> int:
    try:
        value = int(token.decode("ascii"))
    except ValueError as exc:
        raise SpatialQAError(f"{field} must be an integer") from exc
    if value < 0:
        raise SpatialQAError(f"{field} must be non-negative")
    return value


def _safe_path_token(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return cleaned.strip("_") or "case"


def _float_value(value: object, *, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return float(value)


def _support_candidates_for_labels(
    labels: Sequence[str],
    *,
    target_label: str | None,
    max_candidates: int,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for label in sorted(
        labels,
        key=lambda item: (
            _support_candidate_priority(item),
            item,
        ),
    ):
        canonical = _canonical_destination_label(label)
        if canonical is None or canonical == target_label or canonical in seen:
            continue
        relation_hint = _SUPPORT_RELATION_HINTS.get(canonical)
        if relation_hint is None:
            continue
        candidates.append(
            {
                "label": canonical,
                "relation_hint": relation_hint,
                "source": "primary_frame_visible_label",
            }
        )
        seen.add(canonical)
        if len(candidates) >= max_candidates:
            break
    return candidates


def _support_candidate_priority(label: str) -> int:
    relation_hint = _SUPPORT_RELATION_HINTS.get(label)
    if relation_hint == "ON":
        return 0
    if relation_hint == "INSIDE":
        return 1
    return 2


def _primary_frame_key(frame: Mapping[str, Any]) -> tuple[str, str, int] | None:
    episode_id = _optional_str(frame.get("episode_id"))
    scene_id = _optional_str(frame.get("scene_id"))
    step = frame.get("step")
    if (
        episode_id is None
        or scene_id is None
        or isinstance(step, bool)
        or not isinstance(step, int)
    ):
        return None
    return (episode_id, scene_id, step)


def _filter_template_rows(value: object, wanted: set[str]) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    rows: list[Any] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        item_id = item.get("id") or item.get("case_id")
        if isinstance(item_id, str) and item_id in wanted:
            rows.append(dict(item))
    return rows


def _strip_vlm_external_forbidden_fields(value: object) -> object:
    forbidden = {
        "evaluator_only",
        "failure_reason",
        "gold",
        "gold_answer",
        "gold_evidence",
        "gold_evidence_edges",
        "gold_evidence_nodes",
        "prediction",
        "semantic_match",
        "strict_exact_match",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _strip_vlm_external_forbidden_fields(item)
            for key, item in value.items()
            if str(key) not in forbidden
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_strip_vlm_external_forbidden_fields(item) for item in value]
    return value


def _semantic_report_case_ids(report: Mapping[str, Any]) -> list[str]:
    return [
        case_id
        for row in _mapping_rows(report.get("cases"))
        if (case_id := _optional_str(row.get("case_id"))) is not None
    ]


def _retry_case_ids_from_semantic_report(report: Mapping[str, Any]) -> list[str]:
    return [
        case_id
        for row in _mapping_rows(report.get("cases"))
        if row.get("semantic_match") is False
        if (case_id := _optional_str(row.get("case_id"))) is not None
    ]


def _retry_input_gap_case_row(case: Mapping[str, Any]) -> dict[str, Any]:
    case_id = _optional_str(case.get("case_id")) or ""
    has_primary_frame = bool(_mapping(case.get("primary_frame")))
    has_frames = bool(_mapping_rows(case.get("frames")))
    has_target_crop = bool(_mapping(case.get("target_crop")))
    has_support = bool(_mapping_rows(case.get("support_candidates")))
    has_options = bool(_mapping_rows(case.get("answer_options")))
    support_required = _support_candidates_required_for_retry_case(case)
    missing: list[str] = []
    if not has_primary_frame:
        missing.append("primary_frame")
    if not has_frames:
        missing.append("frames")
    if support_required and not has_support:
        missing.append("support_candidates")
    if not has_target_crop:
        missing.append("target_crop")
    if not has_options:
        missing.append("answer_options")
    return {
        "case_id": case_id,
        "has_answer_options": has_options,
        "has_frames": has_frames,
        "has_primary_frame": has_primary_frame,
        "has_support_candidates": has_support,
        "support_candidates_required": support_required,
        "has_target_crop": has_target_crop,
        "missing_input_kinds": missing,
    }


def _support_candidates_required_for_retry_case(case: Mapping[str, Any]) -> bool:
    options = _mapping_rows(case.get("answer_options"))
    if not options:
        return True
    for option in options:
        relation = _normalize_relation(option.get("relation"))
        label = _canonical_destination_label(
            _optional_str(option.get("destination_label"))
        )
        source = _optional_str(option.get("source"))
        if relation != "IN_ROOM" or label != "room" or source != "fallback_room":
            return True
    return False


def _prediction_has_usable_current_location(prediction: QAPrediction) -> bool:
    location = _mapping(prediction.answer.get("current_location"))
    relation = _normalize_relation(location.get("relation"))
    destination = (
        _optional_str(location.get("dst"))
        or _optional_str(location.get("dst_label"))
        or _optional_str(location.get("destination_label"))
    )
    destination_label = _canonical_destination_label(destination)
    return (
        relation is not None
        and destination_label is not None
        and destination_label not in _NON_LOCATION_DESTINATION_LABELS
    )


def _single_room_fallback_option(case: Mapping[str, Any]) -> dict[str, str] | None:
    options = _mapping_rows(case.get("answer_options"))
    if len(options) != 1:
        return None
    return _room_fallback_option(options[0])


def _room_fallback_option(case_or_option: Mapping[str, Any]) -> dict[str, str] | None:
    if "answer_options" in case_or_option:
        options = _mapping_rows(case_or_option.get("answer_options"))
        option = next(
            (
                candidate
                for candidate in options
                if _room_fallback_option(candidate) is not None
            ),
            None,
        )
        if option is None:
            return None
    else:
        option = case_or_option
    relation = _normalize_relation(option.get("relation"))
    label = _canonical_destination_label(_optional_str(option.get("destination_label")))
    source = _optional_str(option.get("source"))
    option_id = _optional_str(option.get("option_id"))
    if (
        relation != "IN_ROOM"
        or label != "room"
        or source != "fallback_room"
        or option_id is None
    ):
        return None
    return {
        "destination_label": "room",
        "option_id": option_id,
        "relation": "IN_ROOM",
        "source": "fallback_room",
    }


def _single_support_candidate_option(case: Mapping[str, Any]) -> dict[str, str] | None:
    options = _mapping_rows(case.get("answer_options"))
    non_room_option_count = 0
    support_options: list[dict[str, str]] = []
    room_option_count = 0
    for option in options:
        relation = _normalize_relation(option.get("relation"))
        label = _canonical_destination_label(
            _optional_str(option.get("destination_label"))
        )
        source = _optional_str(option.get("source"))
        option_id = _optional_str(option.get("option_id"))
        if (
            relation == "IN_ROOM"
            and label == "room"
            and source == "fallback_room"
        ):
            room_option_count += 1
            continue
        if relation in {"INSIDE", "ON"} and label is not None:
            non_room_option_count += 1
        if (
            relation in {"INSIDE", "ON"}
            and label is not None
            and source == "support_candidate"
            and option_id is not None
        ):
            support_options.append(
                {
                    "destination_label": label,
                    "option_id": option_id,
                    "relation": relation,
                    "source": "support_candidate",
                }
            )
    if room_option_count != 1 or non_room_option_count != 1 or len(support_options) != 1:
        return None
    return support_options[0]


def _text_alignable_options(
    case: Mapping[str, Any],
    *,
    relation: str,
) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for option in _mapping_rows(case.get("answer_options")):
        option_relation = _normalize_relation(option.get("relation"))
        label = _canonical_destination_label(
            _optional_str(option.get("destination_label"))
        )
        option_id = _optional_str(option.get("option_id"))
        source = _optional_str(option.get("source"))
        if (
            option_relation != relation
            or label is None
            or option_id is None
            or source == "fallback_room"
        ):
            continue
        options.append(
            {
                "destination_label": label,
                "option_id": option_id,
                "relation": relation,
            }
        )
    return options


def _affordance_option_for_case(
    case: Mapping[str, Any],
) -> dict[str, str] | str | None:
    target_label = _canonical_destination_label(
        _optional_str(_mapping(case.get("target")).get("label"))
    )
    if target_label is None:
        return None
    preferred_labels = _TARGET_AFFORDANCE_OPTION_PRIORS.get(target_label)
    if preferred_labels is None:
        return None
    options = _mapping_rows(case.get("answer_options"))
    matches: list[dict[str, str]] = []
    for preferred_label in preferred_labels:
        for option in options:
            relation = _normalize_relation(option.get("relation"))
            destination_label = _canonical_destination_label(
                _optional_str(option.get("destination_label"))
            )
            option_id = _optional_str(option.get("option_id"))
            source = _optional_str(option.get("source"))
            if (
                relation == "ON"
                and destination_label == preferred_label
                and option_id is not None
                and source != "fallback_room"
            ):
                matches.append(
                    {
                        "destination_label": preferred_label,
                        "option_id": option_id,
                        "relation": "ON",
                    }
                )
    if len(matches) > 1:
        return "ambiguous"
    return matches[0] if matches else None


def _text_mentioned_options(
    answer: Mapping[str, Any],
    options: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    text = _alignment_text(answer)
    if text is None:
        return []
    normalized_text = f" {_normalize_text(text)} "
    matches: list[dict[str, str]] = []
    for option in options:
        label = option["destination_label"]
        if any(
            _normalized_text_mentions(normalized_text, variant)
            for variant in _answer_option_text_variants(label)
        ):
            matches.append(dict(option))
    return matches


def _alignment_text(answer: Mapping[str, Any]) -> str | None:
    values: list[str] = []
    for key in (
        "reasoning_summary",
        "answer_text",
        "text",
        "location_text",
        "raw_response",
        "response_text",
    ):
        value = answer.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value)
    return "\n".join(values) if values else None


def _answer_option_text_variants(label: str) -> tuple[str, ...]:
    explicit_variants = {
        "coffeetable": ("coffeetable", "coffee table"),
        "countertop": ("countertop", "counter top", "counter"),
        "diningtable": ("diningtable", "dining table"),
        "sidetable": ("sidetable", "side table"),
        "shelf": ("shelf", "shelves"),
    }
    if label in explicit_variants:
        return explicit_variants[label]
    return (label.replace("_", " "),)


def _normalized_text_mentions(normalized_text: str, variant: str) -> bool:
    normalized_variant = _normalize_text(variant).strip()
    if normalized_variant == "":
        return False
    return f" {normalized_variant} " in normalized_text


def _case_primary_step(case: Mapping[str, Any]) -> int | None:
    primary_step = _mapping(case.get("primary_frame")).get("step")
    if isinstance(primary_step, int) and not isinstance(primary_step, bool):
        return primary_step
    step = case.get("step")
    if isinstance(step, int) and not isinstance(step, bool):
        return step
    frames = _mapping_rows(case.get("frames"))
    if frames:
        frame_step = frames[0].get("step")
        if isinstance(frame_step, int) and not isinstance(frame_step, bool):
            return frame_step
    return None


def _mapping(value: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def _mapping_rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [cast(Mapping[str, Any], row) for row in value if isinstance(row, Mapping)]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, str)]


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"VLM frame index field must be a non-empty string: {field}")
    return value


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpatialQAError(f"VLM frame index field must be an integer: {field}")
    return value


def _non_negative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _load_json_mapping(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("JSON file must contain an object")
    return cast(Mapping[str, Any], payload)


def _rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise SpatialQAError("VLM frame index contains non-JSON value")
