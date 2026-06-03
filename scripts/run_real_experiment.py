from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab import (
    SpatialQAError,
    compare_real_experiment_claim_readiness,
    compare_real_experiment_execution_packet,
    compare_real_experiment_execution_receipt,
    compare_real_experiment_operator_checklist,
    compare_real_experiment_operator_progress_report,
    compare_real_experiment_primary_evidence_acceptance_report,
    compare_real_experiment_primary_evidence_request_package,
    compare_real_experiment_primary_evidence_return_checklist,
    compare_real_experiment_primary_evidence_return_progress_report,
    compare_real_experiment_primary_evidence_status,
    compare_real_experiment_research_review,
    compare_real_experiment_run_ledger,
    compare_real_experiment_smoke_run_checklist,
    compare_real_experiment_smoke_run_runbook,
    compare_real_experiment_external_artifact_contracts,
    compare_real_experiment_external_artifact_launch_report,
    load_real_experiment_claim_readiness,
    load_real_experiment_execution_packet,
    load_real_experiment_execution_receipt,
    load_real_experiment_operator_checklist,
    load_real_experiment_operator_progress_report,
    load_real_experiment_primary_evidence_acceptance_report,
    load_real_experiment_primary_evidence_request_package,
    load_real_experiment_primary_evidence_return_checklist,
    load_real_experiment_primary_evidence_return_progress_report,
    load_real_experiment_primary_evidence_status,
    load_real_experiment_research_review,
    load_real_experiment_run_ledger,
    load_real_experiment_smoke_run_checklist,
    load_real_experiment_smoke_run_runbook,
    load_real_experiment_external_artifact_contracts,
    load_real_experiment_external_artifact_launch_report,
    real_experiment_claim_readiness,
    real_experiment_execution_packet,
    real_experiment_execution_receipt,
    real_experiment_operator_progress_report,
    real_experiment_primary_evidence_acceptance_report,
    real_experiment_primary_evidence_request_package,
    real_experiment_primary_evidence_return_checklist,
    real_experiment_primary_evidence_return_progress_report,
    real_experiment_primary_evidence_status,
    real_experiment_research_review,
    real_experiment_smoke_run_checklist,
    real_experiment_smoke_run_runbook,
    real_experiment_external_artifact_launch_report,
    real_experiment_run_manifest_preflight,
    run_real_experiment_manifest,
    run_real_experiment_package,
    save_real_experiment_claim_readiness,
    save_real_experiment_execution_packet,
    save_real_experiment_execution_receipt,
    save_real_experiment_operator_progress_report,
    save_real_experiment_primary_evidence_acceptance_report,
    save_real_experiment_primary_evidence_request_package,
    save_real_experiment_primary_evidence_return_checklist,
    save_real_experiment_primary_evidence_return_progress_report,
    save_real_experiment_primary_evidence_status,
    save_real_experiment_research_review,
    save_real_experiment_smoke_run_checklist,
    save_real_experiment_smoke_run_runbook,
    save_real_experiment_external_artifact_launch_report,
    validate_real_experiment_claim_readiness,
    validate_real_experiment_execution_packet,
    validate_real_experiment_execution_receipt,
    validate_real_experiment_operator_checklist,
    validate_real_experiment_operator_progress_report,
    validate_real_experiment_primary_evidence_acceptance_report,
    validate_real_experiment_primary_evidence_request_package,
    validate_real_experiment_primary_evidence_return_checklist,
    validate_real_experiment_primary_evidence_return_progress_report,
    validate_real_experiment_primary_evidence_status,
    validate_real_experiment_research_review,
    validate_real_experiment_run_ledger,
    validate_real_experiment_smoke_run_checklist,
    validate_real_experiment_smoke_run_runbook,
    validate_real_experiment_external_artifact_contracts,
    validate_real_experiment_external_artifact_launch_report,
    write_real_experiment_primary_evidence_request_bundles,
    write_real_experiment_handoff_manifests,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Run the deterministic import pipeline for explicit local "
            "real-experiment artifacts."
        ),
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        help=(
            "Explicit local real experiment run manifest. When supplied, the "
            "manifest provides all run inputs, output paths, thresholds, and "
            "artifact handoff manifests."
        ),
    )
    parser.add_argument(
        "--approved-execution-packet",
        type=Path,
        help=(
            "Optional saved execution packet that must validate, be ready, "
            "and match --run-manifest before the manifest run is allowed."
        ),
    )
    parser.add_argument(
        "--run-ledger-output",
        type=Path,
        help=(
            "Optional top-level real-experiment run ledger output path for "
            "--run-manifest. If omitted, the run manifest's "
            "real_experiment_run_ledger_path is used when present."
        ),
    )
    parser.add_argument(
        "--validate-run-ledger",
        type=Path,
        help="Validate a saved top-level real-experiment run ledger JSON.",
    )
    parser.add_argument(
        "--compare-run-ledger",
        type=Path,
        help=(
            "Compare a saved top-level real-experiment run ledger against the "
            "current run manifest and approval state recorded inside it."
        ),
    )
    parser.add_argument(
        "--preflight-run-manifest",
        type=Path,
        help=(
            "Explicit local real experiment run manifest to audit before "
            "execution. This reads manifests and checks declared local inputs "
            "without importing controls, building predicted DSGs, or writing "
            "experiment outputs."
        ),
    )
    parser.add_argument(
        "--write-handoff-manifests",
        action="store_true",
        help=(
            "Write a portable real-experiment handoff manifest set under "
            "--handoff-root without reading or generating external artifacts."
        ),
    )
    parser.add_argument(
        "--validate-external-artifact-contracts",
        type=Path,
        help="Validate a saved real-experiment external artifact contracts JSON.",
    )
    parser.add_argument(
        "--compare-external-artifact-contracts",
        type=Path,
        help=(
            "Compare a saved real-experiment external artifact contracts JSON "
            "against the current manifests and checklist recorded inside it."
        ),
    )
    parser.add_argument(
        "--validate-operator-checklist",
        type=Path,
        help="Validate a saved real-experiment operator checklist JSON.",
    )
    parser.add_argument(
        "--compare-operator-checklist",
        type=Path,
        help=(
            "Compare a saved real-experiment operator checklist against the "
            "current run manifest and contract paths recorded inside it."
        ),
    )
    parser.add_argument(
        "--operator-progress-report",
        type=Path,
        help=(
            "Build a deterministic progress report from a saved "
            "real-experiment operator checklist."
        ),
    )
    parser.add_argument(
        "--operator-progress-output",
        type=Path,
        help="Optional output path for --operator-progress-report JSON.",
    )
    parser.add_argument(
        "--validate-operator-progress-report",
        type=Path,
        help="Validate a saved real-experiment operator progress report JSON.",
    )
    parser.add_argument(
        "--compare-operator-progress-report",
        type=Path,
        help=(
            "Compare a saved real-experiment operator progress report against "
            "the current checklist and target-file state recorded inside it."
        ),
    )
    parser.add_argument(
        "--external-artifact-launch-report",
        type=Path,
        help=(
            "Summarize current launch readiness by evidence track from a saved "
            "real-experiment external artifact contracts JSON."
        ),
    )
    parser.add_argument(
        "--launch-report-output",
        type=Path,
        help=(
            "Optional output path for --external-artifact-launch-report JSON."
        ),
    )
    parser.add_argument(
        "--validate-external-artifact-launch-report",
        type=Path,
        help="Validate a saved real-experiment external artifact launch report JSON.",
    )
    parser.add_argument(
        "--compare-external-artifact-launch-report",
        type=Path,
        help=(
            "Compare a saved real-experiment external artifact launch report "
            "against the current manifests and receipt bundles recorded inside it."
        ),
    )
    parser.add_argument(
        "--primary-evidence-status",
        type=Path,
        help=(
            "Summarize the three primary evidence tracks from a saved "
            "real-experiment external artifact launch report."
        ),
    )
    parser.add_argument(
        "--primary-evidence-status-output",
        type=Path,
        help="Optional output path for --primary-evidence-status JSON.",
    )
    parser.add_argument(
        "--validate-primary-evidence-status",
        type=Path,
        help="Validate a saved real-experiment primary evidence status JSON.",
    )
    parser.add_argument(
        "--compare-primary-evidence-status",
        type=Path,
        help=(
            "Compare a saved real-experiment primary evidence status against "
            "current launch-report, manifest, and receipt state."
        ),
    )
    parser.add_argument(
        "--primary-evidence-request-package",
        type=Path,
        help=(
            "Build a deterministic request package for the three primary "
            "evidence tracks from a saved launch report."
        ),
    )
    parser.add_argument(
        "--primary-evidence-request-package-output",
        type=Path,
        help="Optional output path for --primary-evidence-request-package JSON.",
    )
    parser.add_argument(
        "--validate-primary-evidence-request-package",
        type=Path,
        help=(
            "Validate a saved real-experiment primary evidence request package "
            "JSON."
        ),
    )
    parser.add_argument(
        "--compare-primary-evidence-request-package",
        type=Path,
        help=(
            "Compare a saved real-experiment primary evidence request package "
            "against current launch-report and input state."
        ),
    )
    parser.add_argument(
        "--write-primary-evidence-request-bundles",
        type=Path,
        help=(
            "Write embedded ready child request bundles from a saved primary "
            "evidence request package to their declared local paths."
        ),
    )
    parser.add_argument(
        "--primary-evidence-return-checklist",
        type=Path,
        help=(
            "Build a deterministic return checklist from a saved primary "
            "evidence request package."
        ),
    )
    parser.add_argument(
        "--primary-evidence-return-checklist-output",
        type=Path,
        help="Optional output path for --primary-evidence-return-checklist JSON.",
    )
    parser.add_argument(
        "--validate-primary-evidence-return-checklist",
        type=Path,
        help=(
            "Validate a saved real-experiment primary evidence return "
            "checklist JSON."
        ),
    )
    parser.add_argument(
        "--compare-primary-evidence-return-checklist",
        type=Path,
        help=(
            "Compare a saved real-experiment primary evidence return checklist "
            "against current request-package and launch-report state."
        ),
    )
    parser.add_argument(
        "--primary-evidence-return-progress-report",
        type=Path,
        help=(
            "Build a deterministic path-presence progress report from a saved "
            "primary evidence return checklist."
        ),
    )
    parser.add_argument(
        "--primary-evidence-return-progress-output",
        type=Path,
        help=(
            "Optional output path for --primary-evidence-return-progress-report "
            "JSON."
        ),
    )
    parser.add_argument(
        "--validate-primary-evidence-return-progress-report",
        type=Path,
        help=(
            "Validate a saved real-experiment primary evidence return progress "
            "report JSON."
        ),
    )
    parser.add_argument(
        "--compare-primary-evidence-return-progress-report",
        type=Path,
        help=(
            "Compare a saved real-experiment primary evidence return progress "
            "report against current return-checklist target-file state."
        ),
    )
    parser.add_argument(
        "--primary-evidence-acceptance-report",
        type=Path,
        help=(
            "Build a deterministic receipt-validation acceptance report from "
            "a saved primary evidence return progress report."
        ),
    )
    parser.add_argument(
        "--primary-evidence-acceptance-output",
        type=Path,
        help="Optional output path for --primary-evidence-acceptance-report JSON.",
    )
    parser.add_argument(
        "--validate-primary-evidence-acceptance-report",
        type=Path,
        help=(
            "Validate a saved real-experiment primary evidence acceptance "
            "report JSON."
        ),
    )
    parser.add_argument(
        "--compare-primary-evidence-acceptance-report",
        type=Path,
        help=(
            "Compare a saved real-experiment primary evidence acceptance "
            "report against current receipt/report validation state."
        ),
    )
    parser.add_argument(
        "--execution-packet",
        type=Path,
        help=(
            "Build a deterministic execution packet from a saved "
            "real-experiment external artifact launch report."
        ),
    )
    parser.add_argument(
        "--execution-packet-primary-evidence-acceptance-report",
        type=Path,
        help=(
            "Optional primary-evidence acceptance report path for "
            "--execution-packet. Defaults to "
            "real-experiment-primary-evidence-acceptance-report.json beside "
            "the launch report."
        ),
    )
    parser.add_argument(
        "--execution-packet-output",
        type=Path,
        help="Optional output path for --execution-packet JSON.",
    )
    parser.add_argument(
        "--validate-execution-packet",
        type=Path,
        help="Validate a saved real-experiment execution packet JSON.",
    )
    parser.add_argument(
        "--compare-execution-packet",
        type=Path,
        help=(
            "Compare a saved real-experiment execution packet against the "
            "current launch report and receipt state recorded inside it."
        ),
    )
    parser.add_argument(
        "--execution-receipt",
        type=Path,
        help=(
            "Build a deterministic post-run receipt from a saved "
            "real-experiment execution packet."
        ),
    )
    parser.add_argument(
        "--execution-receipt-output",
        type=Path,
        help="Optional output path for --execution-receipt JSON.",
    )
    parser.add_argument(
        "--validate-execution-receipt",
        type=Path,
        help="Validate a saved real-experiment execution receipt JSON.",
    )
    parser.add_argument(
        "--compare-execution-receipt",
        type=Path,
        help=(
            "Compare a saved real-experiment execution receipt against the "
            "current output artifacts recorded by its execution packet."
        ),
    )
    parser.add_argument(
        "--smoke-run-checklist",
        type=Path,
        help=(
            "Build a deterministic smoke-run checklist from a saved "
            "real-experiment execution packet."
        ),
    )
    parser.add_argument(
        "--smoke-run-checklist-output",
        type=Path,
        help="Optional output path for --smoke-run-checklist JSON.",
    )
    parser.add_argument(
        "--smoke-run-checklist-receipt-output",
        type=Path,
        help=(
            "Optional execution receipt output path to record in "
            "--smoke-run-checklist."
        ),
    )
    parser.add_argument(
        "--validate-smoke-run-checklist",
        type=Path,
        help="Validate a saved real-experiment smoke-run checklist JSON.",
    )
    parser.add_argument(
        "--compare-smoke-run-checklist",
        type=Path,
        help=(
            "Compare a saved real-experiment smoke-run checklist against the "
            "current execution packet recorded inside it."
        ),
    )
    parser.add_argument(
        "--smoke-run-runbook",
        type=Path,
        help=(
            "Build a deterministic command runbook from a saved "
            "real-experiment smoke-run checklist."
        ),
    )
    parser.add_argument(
        "--smoke-run-runbook-output",
        type=Path,
        help="Optional output path for --smoke-run-runbook JSON.",
    )
    parser.add_argument(
        "--validate-smoke-run-runbook",
        type=Path,
        help="Validate a saved real-experiment smoke-run runbook JSON.",
    )
    parser.add_argument(
        "--compare-smoke-run-runbook",
        type=Path,
        help=(
            "Compare a saved real-experiment smoke-run runbook against the "
            "current smoke-run checklist recorded inside it."
        ),
    )
    parser.add_argument(
        "--research-review",
        type=Path,
        help=(
            "Build a deterministic research review packet from a saved "
            "real-experiment execution receipt."
        ),
    )
    parser.add_argument(
        "--research-review-output",
        type=Path,
        help="Optional output path for --research-review JSON.",
    )
    parser.add_argument(
        "--validate-research-review",
        type=Path,
        help="Validate a saved real-experiment research review JSON.",
    )
    parser.add_argument(
        "--compare-research-review",
        type=Path,
        help=(
            "Compare a saved real-experiment research review against the "
            "current execution receipt and run outputs recorded inside it."
        ),
    )
    parser.add_argument(
        "--claim-readiness",
        type=Path,
        help=(
            "Build a deterministic claim-readiness report from a saved "
            "real-experiment research review."
        ),
    )
    parser.add_argument(
        "--claim-readiness-output",
        type=Path,
        help="Optional output path for --claim-readiness JSON.",
    )
    parser.add_argument(
        "--validate-claim-readiness",
        type=Path,
        help="Validate a saved real-experiment claim-readiness JSON.",
    )
    parser.add_argument(
        "--compare-claim-readiness",
        type=Path,
        help=(
            "Compare a saved real-experiment claim-readiness report against "
            "the current research review and benchmark manifest."
        ),
    )
    parser.add_argument("--claim-min-episode-count", type=int, default=3)
    parser.add_argument("--claim-min-scene-count", type=int, default=1)
    parser.add_argument("--claim-min-qa-count", type=int, default=30)
    parser.add_argument("--claim-min-dynamic-qa-count", type=int, default=1)
    parser.add_argument(
        "--handoff-root",
        type=Path,
        help="Output directory for --write-handoff-manifests.",
    )
    parser.add_argument(
        "--offline-qa",
        type=Path,
        help="Gold QA JSONL path to record in the offline-control manifest.",
    )
    parser.add_argument(
        "--candidate-prediction",
        type=Path,
        help="Candidate DSG GraphTool prediction JSONL path for control deltas.",
    )
    parser.add_argument(
        "--detector-jsonl",
        type=Path,
        help="Detector/RGB-D JSONL path for the predicted DSG detector manifest.",
    )
    parser.add_argument(
        "--episode",
        "--episodes",
        action="append",
        type=Path,
        dest="episodes",
        help="Explicit episode JSONL path. May be repeated.",
    )
    parser.add_argument("--dataset-name", default="real_experiment")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--readiness-report", type=Path)
    parser.add_argument("--summary-report", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--max-qa-per-episode", type=int)
    parser.add_argument("--tag", action="append", dest="tags")
    parser.add_argument("--data-source-kind", default="real")
    parser.add_argument(
        "--real-collection-source-kind",
        default="ai2thor",
        help=(
            "Real collection source kind recorded in handoff manifests "
            "for check_real_collection.py, for example ai2thor or habitat."
        ),
    )
    parser.add_argument("--min-episode-count", type=int, default=3)
    parser.add_argument("--min-scene-count", type=int, default=1)
    parser.add_argument("--real-collection-min-frame-count", type=int, default=30)
    parser.add_argument("--min-qa-count", type=int, default=30)
    parser.add_argument(
        "--required-control-kind",
        action="append",
        dest="required_control_kinds",
    )
    parser.add_argument(
        "--required-predicted-input-kind",
        action="append",
        dest="required_predicted_input_kinds",
    )
    parser.add_argument(
        "--qa-eval-report",
        action="append",
        type=Path,
        dest="qa_eval_reports",
    )
    parser.add_argument(
        "--qa-eval-delta-report",
        action="append",
        type=Path,
        dest="qa_eval_delta_reports",
    )
    parser.add_argument(
        "--active-task-report",
        action="append",
        type=Path,
        dest="active_task_reports",
    )
    parser.add_argument(
        "--active-task-delta-report",
        action="append",
        type=Path,
        dest="active_task_delta_reports",
    )
    parser.add_argument(
        "--dashboard-bundle",
        action="append",
        type=Path,
        dest="dashboard_bundles",
    )
    parser.add_argument(
        "--error-attribution-report",
        action="append",
        type=Path,
        dest="error_attribution_reports",
    )
    parser.add_argument(
        "--graph-eval-report",
        action="append",
        type=Path,
        dest="graph_eval_reports",
    )
    parser.add_argument(
        "--offline-prediction-import-report",
        action="append",
        type=Path,
        dest="offline_prediction_import_reports",
    )
    parser.add_argument(
        "--offline-control-import-manifest",
        type=Path,
        help=(
            "Optional offline control import manifest. When supplied, the "
            "real run first imports offline controls and adds the generated "
            "matrix, import reports, QA eval reports, and QA delta reports to "
            "the real package handoff."
        ),
    )
    parser.add_argument(
        "--offline-control-import-run-ledger",
        type=Path,
        help=(
            "Optional output path for the offline-control import run ledger "
            "when --offline-control-import-manifest is supplied."
        ),
    )
    parser.add_argument(
        "--offline-control-matrix-report",
        action="append",
        type=Path,
        dest="offline_control_matrix_reports",
    )
    parser.add_argument(
        "--offline-control-result-report",
        action="append",
        type=Path,
        dest="offline_control_result_reports",
    )
    parser.add_argument(
        "--predicted-graph-report",
        action="append",
        type=Path,
        dest="predicted_graph_reports",
    )
    parser.add_argument(
        "--predicted-dsg-detector-run-manifest",
        type=Path,
        help=(
            "Optional predicted DSG detector-run manifest. When supplied, "
            "the real run first generates the predicted graph report and "
            "predicted DSG evidence report, then adds them to package assembly."
        ),
    )
    parser.add_argument(
        "--predicted-dsg-detector-run-ledger",
        type=Path,
        help=(
            "Optional output path for the predicted DSG detector-run ledger "
            "when --predicted-dsg-detector-run-manifest is supplied."
        ),
    )
    parser.add_argument(
        "--predicted-dsg-evidence-report",
        action="append",
        type=Path,
        dest="predicted_dsg_evidence_reports",
    )
    parser.add_argument(
        "--real-collection-report",
        action="append",
        type=Path,
        dest="real_collection_reports",
    )
    args = parser.parse_args(argv)

    if args.validate_run_ledger is not None:
        try:
            result = validate_real_experiment_run_ledger(
                load_real_experiment_run_ledger(args.validate_run_ledger)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_real_experiment_run_ledger",
                    args.validate_run_ledger,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["valid"] is True else 1

    if args.compare_run_ledger is not None:
        try:
            result = compare_real_experiment_run_ledger(
                load_real_experiment_run_ledger(args.compare_run_ledger)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "compare_real_experiment_run_ledger",
                    args.compare_run_ledger,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["matches"] is True else 1

    if args.validate_external_artifact_contracts is not None:
        try:
            result = validate_real_experiment_external_artifact_contracts(
                load_real_experiment_external_artifact_contracts(
                    args.validate_external_artifact_contracts
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_real_experiment_external_artifact_contracts",
                    args.validate_external_artifact_contracts,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["valid"] is True else 1

    if args.compare_external_artifact_contracts is not None:
        try:
            result = compare_real_experiment_external_artifact_contracts(
                load_real_experiment_external_artifact_contracts(
                    args.compare_external_artifact_contracts
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "compare_real_experiment_external_artifact_contracts",
                    args.compare_external_artifact_contracts,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["matches"] is True else 1

    if args.validate_operator_checklist is not None:
        try:
            result = validate_real_experiment_operator_checklist(
                load_real_experiment_operator_checklist(
                    args.validate_operator_checklist
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_real_experiment_operator_checklist",
                    args.validate_operator_checklist,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["valid"] is True else 1

    if args.compare_operator_checklist is not None:
        try:
            result = compare_real_experiment_operator_checklist(
                load_real_experiment_operator_checklist(
                    args.compare_operator_checklist
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "compare_real_experiment_operator_checklist",
                    args.compare_operator_checklist,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["matches"] is True else 1

    if args.operator_progress_report is not None:
        try:
            result = real_experiment_operator_progress_report(
                load_real_experiment_operator_checklist(
                    args.operator_progress_report
                ),
                checklist_path=args.operator_progress_report,
            )
            if args.operator_progress_output is not None:
                save_real_experiment_operator_progress_report(
                    result,
                    args.operator_progress_output,
                )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "real_experiment_operator_progress_report",
                    args.operator_progress_report,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["summary"]["all_targets_present"] is True else 1

    if args.validate_operator_progress_report is not None:
        try:
            result = validate_real_experiment_operator_progress_report(
                load_real_experiment_operator_progress_report(
                    args.validate_operator_progress_report
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_real_experiment_operator_progress_report",
                    args.validate_operator_progress_report,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["valid"] is True else 1

    if args.compare_operator_progress_report is not None:
        try:
            result = compare_real_experiment_operator_progress_report(
                load_real_experiment_operator_progress_report(
                    args.compare_operator_progress_report
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "compare_real_experiment_operator_progress_report",
                    args.compare_operator_progress_report,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["matches"] is True else 1

    if args.validate_external_artifact_launch_report is not None:
        try:
            result = validate_real_experiment_external_artifact_launch_report(
                load_real_experiment_external_artifact_launch_report(
                    args.validate_external_artifact_launch_report
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_real_experiment_external_artifact_launch_report",
                    args.validate_external_artifact_launch_report,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["valid"] is True else 1

    if args.compare_external_artifact_launch_report is not None:
        try:
            result = compare_real_experiment_external_artifact_launch_report(
                load_real_experiment_external_artifact_launch_report(
                    args.compare_external_artifact_launch_report
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "compare_real_experiment_external_artifact_launch_report",
                    args.compare_external_artifact_launch_report,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["matches"] is True else 1

    if args.primary_evidence_status is not None:
        try:
            result = real_experiment_primary_evidence_status(
                load_real_experiment_external_artifact_launch_report(
                    args.primary_evidence_status
                ),
                launch_report_path=args.primary_evidence_status,
            )
            if args.primary_evidence_status_output is not None:
                save_real_experiment_primary_evidence_status(
                    result,
                    args.primary_evidence_status_output,
                )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "real_experiment_primary_evidence_status",
                    args.primary_evidence_status,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["summary"]["ready"] is True else 1

    if args.validate_primary_evidence_status is not None:
        try:
            result = validate_real_experiment_primary_evidence_status(
                load_real_experiment_primary_evidence_status(
                    args.validate_primary_evidence_status
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_real_experiment_primary_evidence_status",
                    args.validate_primary_evidence_status,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["valid"] is True else 1

    if args.compare_primary_evidence_status is not None:
        try:
            result = compare_real_experiment_primary_evidence_status(
                load_real_experiment_primary_evidence_status(
                    args.compare_primary_evidence_status
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "compare_real_experiment_primary_evidence_status",
                    args.compare_primary_evidence_status,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["matches"] is True else 1

    if args.primary_evidence_request_package is not None:
        try:
            result = real_experiment_primary_evidence_request_package(
                load_real_experiment_external_artifact_launch_report(
                    args.primary_evidence_request_package
                ),
                launch_report_path=args.primary_evidence_request_package,
            )
            if args.primary_evidence_request_package_output is not None:
                save_real_experiment_primary_evidence_request_package(
                    result,
                    args.primary_evidence_request_package_output,
                )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "real_experiment_primary_evidence_request_package",
                    args.primary_evidence_request_package,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["summary"]["all_request_tracks_ready"] is True else 1

    if args.validate_primary_evidence_request_package is not None:
        try:
            result = validate_real_experiment_primary_evidence_request_package(
                load_real_experiment_primary_evidence_request_package(
                    args.validate_primary_evidence_request_package
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_real_experiment_primary_evidence_request_package",
                    args.validate_primary_evidence_request_package,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["valid"] is True else 1

    if args.compare_primary_evidence_request_package is not None:
        try:
            result = compare_real_experiment_primary_evidence_request_package(
                load_real_experiment_primary_evidence_request_package(
                    args.compare_primary_evidence_request_package
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "compare_real_experiment_primary_evidence_request_package",
                    args.compare_primary_evidence_request_package,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["matches"] is True else 1

    if args.write_primary_evidence_request_bundles is not None:
        try:
            result = write_real_experiment_primary_evidence_request_bundles(
                load_real_experiment_primary_evidence_request_package(
                    args.write_primary_evidence_request_bundles
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "write_real_experiment_primary_evidence_request_bundles",
                    args.write_primary_evidence_request_bundles,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["summary"]["all_request_bundles_written"] is True else 1

    if args.primary_evidence_return_checklist is not None:
        try:
            result = real_experiment_primary_evidence_return_checklist(
                load_real_experiment_primary_evidence_request_package(
                    args.primary_evidence_return_checklist
                ),
                request_package_path=args.primary_evidence_return_checklist,
            )
            if args.primary_evidence_return_checklist_output is not None:
                save_real_experiment_primary_evidence_return_checklist(
                    result,
                    args.primary_evidence_return_checklist_output,
                )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "real_experiment_primary_evidence_return_checklist",
                    args.primary_evidence_return_checklist,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return (
            0
            if result["summary"]["all_return_tracks_actionable"] is True
            else 1
        )

    if args.validate_primary_evidence_return_checklist is not None:
        try:
            result = validate_real_experiment_primary_evidence_return_checklist(
                load_real_experiment_primary_evidence_return_checklist(
                    args.validate_primary_evidence_return_checklist
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_real_experiment_primary_evidence_return_checklist",
                    args.validate_primary_evidence_return_checklist,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["valid"] is True else 1

    if args.compare_primary_evidence_return_checklist is not None:
        try:
            result = compare_real_experiment_primary_evidence_return_checklist(
                load_real_experiment_primary_evidence_return_checklist(
                    args.compare_primary_evidence_return_checklist
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "compare_real_experiment_primary_evidence_return_checklist",
                    args.compare_primary_evidence_return_checklist,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["matches"] is True else 1

    if args.primary_evidence_return_progress_report is not None:
        try:
            result = real_experiment_primary_evidence_return_progress_report(
                load_real_experiment_primary_evidence_return_checklist(
                    args.primary_evidence_return_progress_report
                ),
                return_checklist_path=args.primary_evidence_return_progress_report,
            )
            if args.primary_evidence_return_progress_output is not None:
                save_real_experiment_primary_evidence_return_progress_report(
                    result,
                    args.primary_evidence_return_progress_output,
                )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "real_experiment_primary_evidence_return_progress_report",
                    args.primary_evidence_return_progress_report,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["summary"]["ready_for_launch_refresh"] is True else 1

    if args.validate_primary_evidence_return_progress_report is not None:
        try:
            result = (
                validate_real_experiment_primary_evidence_return_progress_report(
                    load_real_experiment_primary_evidence_return_progress_report(
                        args.validate_primary_evidence_return_progress_report
                    )
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_real_experiment_primary_evidence_return_progress_report",
                    args.validate_primary_evidence_return_progress_report,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["valid"] is True else 1

    if args.compare_primary_evidence_return_progress_report is not None:
        try:
            result = (
                compare_real_experiment_primary_evidence_return_progress_report(
                    load_real_experiment_primary_evidence_return_progress_report(
                        args.compare_primary_evidence_return_progress_report
                    )
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "compare_real_experiment_primary_evidence_return_progress_report",
                    args.compare_primary_evidence_return_progress_report,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["matches"] is True else 1

    if args.primary_evidence_acceptance_report is not None:
        try:
            result = real_experiment_primary_evidence_acceptance_report(
                load_real_experiment_primary_evidence_return_progress_report(
                    args.primary_evidence_acceptance_report
                ),
                return_progress_path=args.primary_evidence_acceptance_report,
            )
            if args.primary_evidence_acceptance_output is not None:
                save_real_experiment_primary_evidence_acceptance_report(
                    result,
                    args.primary_evidence_acceptance_output,
                )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "real_experiment_primary_evidence_acceptance_report",
                    args.primary_evidence_acceptance_report,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return (
            0
            if result["summary"]["ready_for_launch_refresh"] is True
            else 1
        )

    if args.validate_primary_evidence_acceptance_report is not None:
        try:
            result = validate_real_experiment_primary_evidence_acceptance_report(
                load_real_experiment_primary_evidence_acceptance_report(
                    args.validate_primary_evidence_acceptance_report
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_real_experiment_primary_evidence_acceptance_report",
                    args.validate_primary_evidence_acceptance_report,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["valid"] is True else 1

    if args.compare_primary_evidence_acceptance_report is not None:
        try:
            result = compare_real_experiment_primary_evidence_acceptance_report(
                load_real_experiment_primary_evidence_acceptance_report(
                    args.compare_primary_evidence_acceptance_report
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "compare_real_experiment_primary_evidence_acceptance_report",
                    args.compare_primary_evidence_acceptance_report,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["matches"] is True else 1

    if args.validate_execution_packet is not None:
        try:
            result = validate_real_experiment_execution_packet(
                load_real_experiment_execution_packet(args.validate_execution_packet)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_real_experiment_execution_packet",
                    args.validate_execution_packet,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["valid"] is True else 1

    if args.compare_execution_packet is not None:
        try:
            result = compare_real_experiment_execution_packet(
                load_real_experiment_execution_packet(args.compare_execution_packet)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "compare_real_experiment_execution_packet",
                    args.compare_execution_packet,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["matches"] is True else 1

    if args.validate_execution_receipt is not None:
        try:
            result = validate_real_experiment_execution_receipt(
                load_real_experiment_execution_receipt(args.validate_execution_receipt)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_real_experiment_execution_receipt",
                    args.validate_execution_receipt,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["valid"] is True else 1

    if args.compare_execution_receipt is not None:
        try:
            result = compare_real_experiment_execution_receipt(
                load_real_experiment_execution_receipt(args.compare_execution_receipt)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "compare_real_experiment_execution_receipt",
                    args.compare_execution_receipt,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["matches"] is True else 1

    if args.validate_smoke_run_checklist is not None:
        try:
            result = validate_real_experiment_smoke_run_checklist(
                load_real_experiment_smoke_run_checklist(
                    args.validate_smoke_run_checklist
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_real_experiment_smoke_run_checklist",
                    args.validate_smoke_run_checklist,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["valid"] is True else 1

    if args.compare_smoke_run_checklist is not None:
        try:
            result = compare_real_experiment_smoke_run_checklist(
                load_real_experiment_smoke_run_checklist(
                    args.compare_smoke_run_checklist
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "compare_real_experiment_smoke_run_checklist",
                    args.compare_smoke_run_checklist,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["matches"] is True else 1

    if args.validate_smoke_run_runbook is not None:
        try:
            result = validate_real_experiment_smoke_run_runbook(
                load_real_experiment_smoke_run_runbook(
                    args.validate_smoke_run_runbook
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_real_experiment_smoke_run_runbook",
                    args.validate_smoke_run_runbook,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["valid"] is True else 1

    if args.compare_smoke_run_runbook is not None:
        try:
            result = compare_real_experiment_smoke_run_runbook(
                load_real_experiment_smoke_run_runbook(
                    args.compare_smoke_run_runbook
                )
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "compare_real_experiment_smoke_run_runbook",
                    args.compare_smoke_run_runbook,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["matches"] is True else 1

    if args.smoke_run_runbook is not None:
        try:
            result = real_experiment_smoke_run_runbook(
                load_real_experiment_smoke_run_checklist(args.smoke_run_runbook),
                smoke_run_checklist_path=args.smoke_run_runbook,
            )
            if args.smoke_run_runbook_output is not None:
                save_real_experiment_smoke_run_runbook(
                    result,
                    args.smoke_run_runbook_output,
                )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "real_experiment_smoke_run_runbook",
                    args.smoke_run_runbook,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["ready_to_start"] is True else 1

    if args.smoke_run_checklist is not None:
        try:
            result = real_experiment_smoke_run_checklist(
                load_real_experiment_execution_packet(args.smoke_run_checklist),
                execution_packet_path=args.smoke_run_checklist,
                execution_receipt_output_path=(
                    args.smoke_run_checklist_receipt_output
                ),
            )
            if args.smoke_run_checklist_output is not None:
                save_real_experiment_smoke_run_checklist(
                    result,
                    args.smoke_run_checklist_output,
                )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "real_experiment_smoke_run_checklist",
                    args.smoke_run_checklist,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["ready_to_start"] is True else 1

    if args.validate_research_review is not None:
        try:
            result = validate_real_experiment_research_review(
                load_real_experiment_research_review(args.validate_research_review)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_real_experiment_research_review",
                    args.validate_research_review,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["valid"] is True else 1

    if args.compare_research_review is not None:
        try:
            result = compare_real_experiment_research_review(
                load_real_experiment_research_review(args.compare_research_review)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "compare_real_experiment_research_review",
                    args.compare_research_review,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["matches"] is True else 1

    if args.research_review is not None:
        try:
            result = real_experiment_research_review(
                load_real_experiment_execution_receipt(args.research_review),
                execution_receipt_path=args.research_review,
            )
            if args.research_review_output is not None:
                save_real_experiment_research_review(
                    result,
                    args.research_review_output,
                )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "real_experiment_research_review",
                    args.research_review,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["ready_for_research_review"] is True else 1

    if args.validate_claim_readiness is not None:
        try:
            result = validate_real_experiment_claim_readiness(
                load_real_experiment_claim_readiness(args.validate_claim_readiness)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "validate_real_experiment_claim_readiness",
                    args.validate_claim_readiness,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["valid"] is True else 1

    if args.compare_claim_readiness is not None:
        try:
            result = compare_real_experiment_claim_readiness(
                load_real_experiment_claim_readiness(args.compare_claim_readiness)
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "compare_real_experiment_claim_readiness",
                    args.compare_claim_readiness,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["matches"] is True else 1

    if args.claim_readiness is not None:
        try:
            result = real_experiment_claim_readiness(
                load_real_experiment_research_review(args.claim_readiness),
                research_review_path=args.claim_readiness,
                min_dynamic_qa_count=args.claim_min_dynamic_qa_count,
                min_episode_count=args.claim_min_episode_count,
                min_qa_count=args.claim_min_qa_count,
                min_scene_count=args.claim_min_scene_count,
            )
            if args.claim_readiness_output is not None:
                save_real_experiment_claim_readiness(
                    result,
                    args.claim_readiness_output,
                )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "real_experiment_claim_readiness",
                    args.claim_readiness,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["claim_ready"] is True else 1

    if args.execution_receipt is not None:
        try:
            result = real_experiment_execution_receipt(
                load_real_experiment_execution_packet(args.execution_receipt),
                execution_packet_path=args.execution_receipt,
            )
            if args.execution_receipt_output is not None:
                save_real_experiment_execution_receipt(
                    result,
                    args.execution_receipt_output,
                )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "real_experiment_execution_receipt",
                    args.execution_receipt,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["ready_to_review"] is True else 1

    if args.execution_packet is not None:
        try:
            result = real_experiment_execution_packet(
                load_real_experiment_external_artifact_launch_report(
                    args.execution_packet
                ),
                launch_report_path=args.execution_packet,
                primary_evidence_acceptance_report_path=(
                    args.execution_packet_primary_evidence_acceptance_report
                ),
                execution_packet_path=args.execution_packet_output,
            )
            if args.execution_packet_output is not None:
                save_real_experiment_execution_packet(
                    result,
                    args.execution_packet_output,
                )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "real_experiment_execution_packet",
                    args.execution_packet,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["ready_to_execute"] is True else 1

    if args.external_artifact_launch_report is not None:
        try:
            result = real_experiment_external_artifact_launch_report(
                load_real_experiment_external_artifact_contracts(
                    args.external_artifact_launch_report
                ),
                contracts_path=args.external_artifact_launch_report,
            )
            if args.launch_report_output is not None:
                save_real_experiment_external_artifact_launch_report(
                    result,
                    args.launch_report_output,
                )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "real_experiment_external_artifact_launch_report",
                    args.external_artifact_launch_report,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["ready_to_run"] is True else 1

    if args.launch_report_output is not None:
        parser.error("--launch-report-output requires --external-artifact-launch-report")
    if args.operator_progress_output is not None:
        parser.error("--operator-progress-output requires --operator-progress-report")
    if args.primary_evidence_status_output is not None:
        parser.error(
            "--primary-evidence-status-output requires --primary-evidence-status"
        )
    if args.primary_evidence_request_package_output is not None:
        parser.error(
            "--primary-evidence-request-package-output requires "
            "--primary-evidence-request-package"
        )
    if args.primary_evidence_return_checklist_output is not None:
        parser.error(
            "--primary-evidence-return-checklist-output requires "
            "--primary-evidence-return-checklist"
        )
    if args.primary_evidence_return_progress_output is not None:
        parser.error(
            "--primary-evidence-return-progress-output requires "
            "--primary-evidence-return-progress-report"
        )
    if args.primary_evidence_acceptance_output is not None:
        parser.error(
            "--primary-evidence-acceptance-output requires "
            "--primary-evidence-acceptance-report"
        )
    if args.execution_packet_output is not None:
        parser.error("--execution-packet-output requires --execution-packet")
    if args.execution_packet_primary_evidence_acceptance_report is not None:
        parser.error(
            "--execution-packet-primary-evidence-acceptance-report requires "
            "--execution-packet"
        )
    if args.approved_execution_packet is not None and args.run_manifest is None:
        parser.error("--approved-execution-packet requires --run-manifest")
    if args.run_ledger_output is not None and args.run_manifest is None:
        parser.error("--run-ledger-output requires --run-manifest")
    if args.execution_receipt_output is not None:
        parser.error("--execution-receipt-output requires --execution-receipt")
    if args.smoke_run_checklist_output is not None:
        parser.error("--smoke-run-checklist-output requires --smoke-run-checklist")
    if args.smoke_run_checklist_receipt_output is not None:
        parser.error(
            "--smoke-run-checklist-receipt-output requires --smoke-run-checklist"
        )
    if args.smoke_run_runbook_output is not None:
        parser.error("--smoke-run-runbook-output requires --smoke-run-runbook")
    if args.research_review_output is not None:
        parser.error("--research-review-output requires --research-review")
    if args.claim_readiness_output is not None:
        parser.error("--claim-readiness-output requires --claim-readiness")

    if args.write_handoff_manifests:
        if args.handoff_root is None:
            parser.error("--handoff-root is required with --write-handoff-manifests")
        if not args.episodes:
            parser.error("--episode is required with --write-handoff-manifests")
        try:
            result = write_real_experiment_handoff_manifests(
                root=args.handoff_root,
                dataset_name=args.dataset_name,
                episode_paths=tuple(args.episodes),
                offline_qa_path=args.offline_qa or "inputs/qa.jsonl",
                candidate_prediction_path=(
                    args.candidate_prediction
                    or "inputs/candidate/predicted-graph-tool.jsonl"
                ),
                detector_jsonl_path=(
                    args.detector_jsonl
                    or "inputs/predicted-dsg/detector-rgbd.jsonl"
                ),
                max_qa_per_episode=args.max_qa_per_episode,
                tags=tuple(args.tags or ("benchmark", "real")),
                declared_data_source_kind=args.data_source_kind,
                real_collection_source_kind=args.real_collection_source_kind,
                min_episode_count=args.min_episode_count,
                min_scene_count=args.min_scene_count,
                min_frame_count=args.real_collection_min_frame_count,
                min_qa_count=args.min_qa_count,
                required_control_kinds=tuple(
                    args.required_control_kinds
                    or ("caption_memory", "graph_text", "multi_frame_vlm", "vlm")
                ),
                required_predicted_input_kinds=tuple(
                    args.required_predicted_input_kinds
                    or ("observation_sequence",)
                ),
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "write_real_experiment_handoff_manifests",
                    args.handoff_root,
                    exc,
                )
            )
            return 1
        _emit_json(result)
        return 0

    if args.preflight_run_manifest is not None:
        try:
            result = real_experiment_run_manifest_preflight(
                args.preflight_run_manifest
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "real_experiment_run_manifest_preflight",
                    args.preflight_run_manifest,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["ready_to_run"] is True else 1

    if args.run_manifest is not None:
        try:
            result = run_real_experiment_manifest(
                args.run_manifest,
                approved_execution_packet_path=args.approved_execution_packet,
                run_ledger_output_path=args.run_ledger_output,
            )
        except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
            _emit_json(
                _error_payload(
                    "run_real_experiment_manifest",
                    args.run_manifest,
                    exc,
                )
            )
            return 1

        _emit_json(result)
        return 0 if result["ready"] is True else 1

    if not args.episodes:
        parser.error("--episode is required")
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if args.manifest is None:
        parser.error("--manifest is required")
    if args.readiness_report is None:
        parser.error("--readiness-report is required")
    if args.summary_report is None:
        parser.error("--summary-report is required")
    if args.record is None:
        parser.error("--record is required")

    try:
        result = run_real_experiment_package(
            dataset_name=args.dataset_name,
            episode_paths=tuple(args.episodes),
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            readiness_report_path=args.readiness_report,
            summary_report_path=args.summary_report,
            record_path=args.record,
            max_qa_per_episode=args.max_qa_per_episode,
            tags=tuple(args.tags or ("benchmark", "real")),
            declared_data_source_kind=args.data_source_kind,
            min_episode_count=args.min_episode_count,
            min_scene_count=args.min_scene_count,
            min_qa_count=args.min_qa_count,
            required_control_kinds=tuple(
                args.required_control_kinds
                or ("caption_memory", "graph_text", "multi_frame_vlm", "vlm")
            ),
            required_predicted_input_kinds=tuple(
                args.required_predicted_input_kinds or ("observation_sequence",)
            ),
            qa_eval_report_paths=tuple(args.qa_eval_reports or ()),
            qa_eval_delta_report_paths=tuple(args.qa_eval_delta_reports or ()),
            active_task_report_paths=tuple(args.active_task_reports or ()),
            active_task_delta_report_paths=tuple(args.active_task_delta_reports or ()),
            dashboard_bundle_paths=tuple(args.dashboard_bundles or ()),
            error_attribution_report_paths=tuple(args.error_attribution_reports or ()),
            graph_eval_report_paths=tuple(args.graph_eval_reports or ()),
            offline_control_import_manifest_path=(
                args.offline_control_import_manifest
            ),
            offline_control_import_run_ledger_path=(
                args.offline_control_import_run_ledger
            ),
            offline_control_matrix_report_paths=tuple(
                args.offline_control_matrix_reports or ()
            ),
            offline_control_result_report_paths=tuple(
                args.offline_control_result_reports or ()
            ),
            offline_prediction_import_report_paths=tuple(
                args.offline_prediction_import_reports or ()
            ),
            predicted_dsg_detector_run_manifest_path=(
                args.predicted_dsg_detector_run_manifest
            ),
            predicted_dsg_detector_run_ledger_path=(
                args.predicted_dsg_detector_run_ledger
            ),
            predicted_dsg_evidence_report_paths=tuple(
                args.predicted_dsg_evidence_reports or ()
            ),
            predicted_graph_report_paths=tuple(args.predicted_graph_reports or ()),
            real_collection_report_paths=tuple(args.real_collection_reports or ()),
        )
    except (OSError, SpatialQAError, ValueError, json.JSONDecodeError) as exc:
        _emit_json(_error_payload("run_real_experiment_package", args.record, exc))
        return 1

    _emit_json(result)
    return 0 if result["ready"] is True else 1


def _error_payload(action: str, path: Path, error: Exception) -> dict[str, Any]:
    return {
        "action": action,
        "path": str(path),
        "valid": False,
        "error": str(error),
    }


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
