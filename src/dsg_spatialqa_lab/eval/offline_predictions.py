from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from dsg_spatialqa_lab.benchmark import QACase, load_qa_dataset, qa_dataset_digest
from dsg_spatialqa_lab.eval.qa_metrics import (
    QAPrediction,
    load_qa_predictions,
    qa_predictions_digest,
)
from dsg_spatialqa_lab.schema import SpatialQAError


OFFLINE_PREDICTION_RECORD_SCHEMA_VERSION = "dsg-spatialqa-lab.offline-prediction-record.v1"
OFFLINE_PREDICTION_IMPORT_REPORT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.offline-prediction-import-report.v1"
)
OFFLINE_PREDICTION_RECORD_INPUT_FORMAT = "offline_prediction_record"
QA_PREDICTION_INPUT_FORMAT = "qa_prediction"


@dataclass(frozen=True)
class OfflinePredictionRecord:
    case_id: str
    answer: Mapping[str, Any] = field(default_factory=dict)
    evidence_nodes: tuple[str, ...] = field(default_factory=tuple)
    evidence_edges: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.0
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.case_id, "case_id")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise SpatialQAError("Offline prediction confidence must be a number")
        if self.error is not None and (not isinstance(self.error, str) or self.error == ""):
            raise SpatialQAError("Offline prediction error must be a non-empty string or null")
        object.__setattr__(self, "answer", _json_mapping(self.answer))
        object.__setattr__(self, "evidence_nodes", _string_tuple(self.evidence_nodes, "evidence_nodes"))
        object.__setattr__(self, "evidence_edges", _string_tuple(self.evidence_edges, "evidence_edges"))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata))


def offline_prediction_record_to_dict(record: OfflinePredictionRecord) -> dict[str, Any]:
    return {
        "schema_version": OFFLINE_PREDICTION_RECORD_SCHEMA_VERSION,
        "case_id": record.case_id,
        "answer": _json_mapping(record.answer),
        "evidence_nodes": list(record.evidence_nodes),
        "evidence_edges": list(record.evidence_edges),
        "confidence": record.confidence,
        "error": record.error,
        "metadata": _json_mapping(record.metadata),
    }


def offline_prediction_record_from_dict(payload: Mapping[str, Any]) -> OfflinePredictionRecord:
    schema_version = _required_str(payload, "schema_version")
    if schema_version != OFFLINE_PREDICTION_RECORD_SCHEMA_VERSION:
        raise SpatialQAError(f"Unsupported offline prediction schema version: {schema_version}")
    return OfflinePredictionRecord(
        case_id=_required_str(payload, "case_id"),
        answer=_optional_mapping(payload, "answer"),
        evidence_nodes=_optional_string_tuple(payload, "evidence_nodes"),
        evidence_edges=_optional_string_tuple(payload, "evidence_edges"),
        confidence=_optional_float(payload, "confidence"),
        error=_optional_str(payload, "error"),
        metadata=_optional_mapping(payload, "metadata"),
    )


def offline_prediction_records_jsonl(records: Sequence[OfflinePredictionRecord]) -> str:
    return "".join(
        json.dumps(
            offline_prediction_record_to_dict(record),
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
        for record in records
    )


def offline_prediction_records_from_jsonl(payload: str) -> list[OfflinePredictionRecord]:
    records: list[OfflinePredictionRecord] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if line == "":
            continue
        item = json.loads(line)
        if not isinstance(item, Mapping):
            raise SpatialQAError(f"Offline prediction line {line_number} must be an object")
        records.append(offline_prediction_record_from_dict(cast(Mapping[str, Any], item)))
    return records


def offline_prediction_records_digest(records: Sequence[OfflinePredictionRecord]) -> str:
    return hashlib.sha256(offline_prediction_records_jsonl(records).encode("utf-8")).hexdigest()


def save_offline_prediction_records(
    records: Sequence[OfflinePredictionRecord],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(offline_prediction_records_jsonl(records), encoding="utf-8")
    return output_path


def load_offline_prediction_records(path: str | Path) -> list[OfflinePredictionRecord]:
    return offline_prediction_records_from_jsonl(Path(path).read_text(encoding="utf-8"))


def offline_prediction_source_profile(
    source_name: str,
    source_kind: str = "offline",
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_non_empty_str(source_name, "source_name")
    _validate_non_empty_str(source_kind, "source_kind")
    metadata = _json_mapping(source_metadata or {})
    return {
        "adapter": _metadata_text(metadata, ("adapter", "source_adapter"), source_kind),
        "capability_axes": _capability_axes(metadata),
        "dataset_id": _metadata_text(metadata, ("dataset_id", "dataset"), "unspecified"),
        "kind": source_kind,
        "metadata_keys": sorted(str(key) for key in metadata),
        "model_id": _metadata_text(
            metadata,
            ("model_id", "model", "provider_model"),
            "unspecified",
        ),
        "name": source_name,
        "prompt_id": _metadata_text(metadata, ("prompt_id", "prompt"), "unspecified"),
        "source_key": f"{source_kind}:{source_name}",
    }


def import_offline_predictions(
    gold_cases: Sequence[QACase],
    records: Sequence[OfflinePredictionRecord],
    *,
    source_name: str,
    source_kind: str = "offline",
    source_metadata: Mapping[str, Any] | None = None,
    qa_path: str | Path | None = None,
    input_path: str | Path | None = None,
    prediction_path: str | Path | None = None,
) -> tuple[list[QAPrediction], dict[str, Any]]:
    _validate_non_empty_str(source_name, "source_name")
    _validate_non_empty_str(source_kind, "source_kind")
    normalized_metadata = _json_mapping(source_metadata or {})
    case_ids = [case.id for case in gold_cases]
    known_case_ids = set(case_ids)
    imported_case_ids: set[str] = set()
    duplicate_case_ids: set[str] = set()
    unknown_case_ids: set[str] = set()
    predictions: list[QAPrediction] = []
    record_results: list[dict[str, Any]] = []
    for line_number, record in enumerate(records, start=1):
        if record.case_id not in known_case_ids:
            unknown_case_ids.add(record.case_id)
            record_results.append(_record_result(record.case_id, line_number, imported=False, error="unknown_case"))
            continue
        if record.case_id in imported_case_ids:
            duplicate_case_ids.add(record.case_id)
            record_results.append(
                _record_result(record.case_id, line_number, imported=False, error="duplicate_case")
            )
            continue
        imported_case_ids.add(record.case_id)
        predictions.append(
            QAPrediction(
                id=record.case_id,
                answer=_json_mapping(record.answer),
                evidence_nodes=record.evidence_nodes,
                evidence_edges=record.evidence_edges,
                confidence=record.confidence,
                error=record.error,
            )
        )
        record_results.append(_record_result(record.case_id, line_number, imported=True, error=None))
    missing_case_ids = [case_id for case_id in case_ids if case_id not in imported_case_ids]
    report: dict[str, Any] = {
        "schema_version": OFFLINE_PREDICTION_IMPORT_REPORT_SCHEMA_VERSION,
        "source": {
            "kind": source_kind,
            "metadata": normalized_metadata,
            "name": source_name,
        },
        "source_profile": offline_prediction_source_profile(
            source_name,
            source_kind,
            normalized_metadata,
        ),
        "qa_path": str(qa_path) if qa_path is not None else None,
        "input_path": str(input_path) if input_path is not None else None,
        "prediction_path": str(prediction_path) if prediction_path is not None else None,
        "qa_digest": qa_dataset_digest(gold_cases),
        "input_digest": offline_prediction_records_digest(records),
        "prediction_digest": qa_predictions_digest(predictions),
        "summary": {
            "duplicate_case_count": len(duplicate_case_ids),
            "error_prediction_count": sum(1 for prediction in predictions if prediction.error is not None),
            "gold_case_count": len(gold_cases),
            "imported_prediction_count": len(predictions),
            "missing_case_count": len(missing_case_ids),
            "record_count": len(records),
            "unknown_case_count": len(unknown_case_ids),
        },
        "missing_case_ids": missing_case_ids,
        "unknown_case_ids": sorted(unknown_case_ids),
        "duplicate_case_ids": sorted(duplicate_case_ids),
        "records": record_results,
    }
    report["report_digest"] = offline_prediction_import_report_digest(report)
    return predictions, report


def import_qa_prediction_inputs(
    gold_cases: Sequence[QACase],
    predictions: Sequence[QAPrediction],
    *,
    source_name: str,
    source_kind: str = "offline",
    source_metadata: Mapping[str, Any] | None = None,
    qa_path: str | Path | None = None,
    input_path: str | Path | None = None,
    prediction_path: str | Path | None = None,
) -> tuple[list[QAPrediction], dict[str, Any]]:
    normalized_predictions, report = import_offline_predictions(
        gold_cases,
        tuple(_record_from_prediction(prediction) for prediction in predictions),
        source_name=source_name,
        source_kind=source_kind,
        source_metadata=source_metadata,
        qa_path=qa_path,
        input_path=input_path,
        prediction_path=prediction_path,
    )
    report["input_format"] = QA_PREDICTION_INPUT_FORMAT
    report["input_digest"] = qa_predictions_digest(predictions)
    report["report_digest"] = offline_prediction_import_report_digest(report)
    return normalized_predictions, report


def offline_prediction_import_report_digest(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_digest"}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def offline_prediction_import_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def save_offline_prediction_import_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(offline_prediction_import_report_json(report), encoding="utf-8")
    return output_path


def load_offline_prediction_import_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SpatialQAError("Offline prediction import report JSON must be an object")
    return cast(dict[str, Any], payload)


def validate_offline_prediction_import_report(report: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = report.get("schema_version")
    report_digest = _string_or_none(report.get("report_digest"))
    expected_digest = offline_prediction_import_report_digest(report)
    expected_source_profile = _expected_source_profile(report.get("source"))
    records = _mapping_items(report.get("records"))
    summary = report.get("summary")
    missing_case_ids = _string_items(report.get("missing_case_ids"))
    unknown_case_ids = _string_items(report.get("unknown_case_ids"))
    duplicate_case_ids = _string_items(report.get("duplicate_case_ids"))
    imported_count = sum(1 for item in records if item.get("imported") is True)
    checks = [
        {
            "name": "schema_version",
            "passed": schema_version == OFFLINE_PREDICTION_IMPORT_REPORT_SCHEMA_VERSION,
            "expected": OFFLINE_PREDICTION_IMPORT_REPORT_SCHEMA_VERSION,
            "actual": schema_version,
        },
        {
            "name": "report_digest",
            "passed": report_digest == expected_digest,
            "expected": expected_digest,
            "actual": report_digest,
        },
        {
            "name": "record_count",
            "passed": _summary_value(summary, "record_count") == len(records),
            "expected": len(records),
            "actual": _summary_value(summary, "record_count"),
        },
        {
            "name": "imported_prediction_count",
            "passed": _summary_value(summary, "imported_prediction_count") == imported_count,
            "expected": imported_count,
            "actual": _summary_value(summary, "imported_prediction_count"),
        },
        {
            "name": "missing_case_count",
            "passed": _summary_value(summary, "missing_case_count") == len(missing_case_ids),
            "expected": len(missing_case_ids),
            "actual": _summary_value(summary, "missing_case_count"),
        },
        {
            "name": "unknown_case_count",
            "passed": _summary_value(summary, "unknown_case_count") == len(unknown_case_ids),
            "expected": len(unknown_case_ids),
            "actual": _summary_value(summary, "unknown_case_count"),
        },
        {
            "name": "duplicate_case_count",
            "passed": _summary_value(summary, "duplicate_case_count") == len(duplicate_case_ids),
            "expected": len(duplicate_case_ids),
            "actual": _summary_value(summary, "duplicate_case_count"),
        },
        {
            "name": "source_profile",
            "passed": report.get("source_profile") == expected_source_profile,
            "expected": expected_source_profile,
            "actual": report.get("source_profile"),
        },
    ]
    return {
        "valid": all(check["passed"] is True for check in checks),
        "schema_version": schema_version,
        "report_digest": report_digest,
        "checks": checks,
    }


def compare_offline_prediction_import_report(report: Mapping[str, Any]) -> dict[str, Any]:
    qa_path = _required_report_path(report, "qa_path")
    input_path = _required_report_path(report, "input_path")
    prediction_path = _required_report_path(report, "prediction_path")
    source = _required_mapping(report, "source")
    input_format = _prediction_input_format(report.get("input_format"))
    if input_format == QA_PREDICTION_INPUT_FORMAT:
        predictions, current_report = import_qa_prediction_inputs(
            load_qa_dataset(qa_path),
            load_qa_predictions(input_path),
            source_name=_required_str(source, "name"),
            source_kind=_required_str(source, "kind"),
            source_metadata=_optional_mapping(source, "metadata"),
            qa_path=qa_path,
            input_path=input_path,
            prediction_path=prediction_path,
        )
    else:
        predictions, current_report = import_offline_predictions(
            load_qa_dataset(qa_path),
            load_offline_prediction_records(input_path),
            source_name=_required_str(source, "name"),
            source_kind=_required_str(source, "kind"),
            source_metadata=_optional_mapping(source, "metadata"),
            qa_path=qa_path,
            input_path=input_path,
            prediction_path=prediction_path,
        )
    validation = validate_offline_prediction_import_report(report)
    saved_digest = _string_or_none(report.get("report_digest"))
    current_digest = _string_or_none(current_report.get("report_digest"))
    saved_prediction_digest = _string_or_none(report.get("prediction_digest"))
    current_prediction_digest = qa_predictions_digest(predictions)
    file_prediction_digest = qa_predictions_digest(load_qa_predictions(prediction_path))
    checks = [
        {
            "name": "report_valid",
            "passed": validation["valid"] is True,
            "expected": True,
            "actual": validation["valid"],
        },
        {
            "name": "report_digest_matches_current",
            "passed": saved_digest == current_digest,
            "expected": saved_digest,
            "actual": current_digest,
        },
        {
            "name": "prediction_digest_matches_current",
            "passed": saved_prediction_digest == current_prediction_digest,
            "expected": saved_prediction_digest,
            "actual": current_prediction_digest,
        },
        {
            "name": "prediction_file_matches_current",
            "passed": file_prediction_digest == current_prediction_digest,
            "expected": current_prediction_digest,
            "actual": file_prediction_digest,
        },
        _equality_check("summary_matches_current", report.get("summary"), current_report["summary"]),
        _equality_check("records_match_current", report.get("records"), current_report["records"]),
        _equality_check(
            "missing_case_ids_match_current",
            report.get("missing_case_ids"),
            current_report["missing_case_ids"],
        ),
        _equality_check(
            "source_profile_matches_current",
            report.get("source_profile"),
            current_report["source_profile"],
        ),
    ]
    return {
        "matches": all(check["passed"] is True for check in checks),
        "saved_digest": saved_digest,
        "current_digest": current_digest,
        "validation": validation,
        "checks": checks,
    }


def _record_from_prediction(prediction: QAPrediction) -> OfflinePredictionRecord:
    return OfflinePredictionRecord(
        case_id=prediction.id,
        answer=prediction.answer,
        evidence_nodes=prediction.evidence_nodes,
        evidence_edges=prediction.evidence_edges,
        confidence=prediction.confidence,
        error=prediction.error,
    )


def _record_result(
    case_id: str,
    line_number: int,
    *,
    imported: bool,
    error: str | None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "error": error,
        "imported": imported,
        "line_number": line_number,
    }


def _expected_source_profile(source: object) -> dict[str, Any] | None:
    if not isinstance(source, Mapping):
        return None
    name = source.get("name")
    kind = source.get("kind")
    metadata = source.get("metadata", {})
    if not isinstance(name, str) or name == "":
        return None
    if not isinstance(kind, str) or kind == "":
        return None
    if not isinstance(metadata, Mapping):
        return None
    return offline_prediction_source_profile(
        name,
        kind,
        cast(Mapping[str, Any], metadata),
    )


def _equality_check(name: str, expected: object, actual: object) -> dict[str, Any]:
    return {
        "name": name,
        "passed": expected == actual,
        "expected": expected,
        "actual": actual,
    }


def _required_report_path(report: Mapping[str, Any], key: str) -> Path:
    value = report.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Offline prediction import report missing path: {key}")
    return Path(value)


def _required_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"Offline prediction field must be an object: {key}")
    return _json_mapping(cast(Mapping[str, Any], value))


def _optional_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, Mapping):
        raise SpatialQAError(f"Offline prediction field must be an object: {key}")
    return _json_mapping(cast(Mapping[str, Any], value))


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Offline prediction field must be a non-empty string: {key}")
    return value


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Offline prediction field must be a non-empty string or null: {key}")
    return value


def _optional_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpatialQAError(f"Offline prediction field must be a number: {key}")
    return float(value)


def _optional_string_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key, ())
    return _string_tuple(value, key)


def _string_tuple(value: object, key: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SpatialQAError(f"Offline prediction field must be a string sequence: {key}")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise SpatialQAError(f"Offline prediction field must be a string sequence: {key}")
        items.append(item)
    return tuple(items)


def _validate_non_empty_str(value: object, key: str) -> None:
    if not isinstance(value, str) or value == "":
        raise SpatialQAError(f"Offline prediction field must be a non-empty string: {key}")


def _mapping_items(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    return tuple(cast(Mapping[str, Any], item) for item in value if isinstance(item, Mapping))


def _string_items(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, str)]


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _summary_value(summary: object, key: str) -> object:
    return summary.get(key) if isinstance(summary, Mapping) else None


def _prediction_input_format(value: object) -> str:
    if value is None:
        return OFFLINE_PREDICTION_RECORD_INPUT_FORMAT
    if value in (
        OFFLINE_PREDICTION_RECORD_INPUT_FORMAT,
        QA_PREDICTION_INPUT_FORMAT,
    ):
        return str(value)
    raise SpatialQAError(f"Unsupported offline prediction input format: {value}")


def _metadata_text(
    metadata: Mapping[str, Any],
    keys: Sequence[str],
    default: str,
) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value != "":
            return value
    return default


def _capability_axes(metadata: Mapping[str, Any]) -> list[str]:
    axes: set[str] = set()
    for key in ("capability_axes", "capabilities", "research_axes"):
        value = metadata.get(key)
        if isinstance(value, str):
            axes.update(item.strip() for item in value.split(",") if item.strip())
        elif isinstance(value, Sequence):
            axes.update(item for item in value if isinstance(item, str) and item != "")
    return sorted(axes)


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], _json_value(value))


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_json_value(item) for item in value]
    return value
