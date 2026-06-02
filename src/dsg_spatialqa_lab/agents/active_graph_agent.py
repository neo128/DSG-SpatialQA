from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from dsg_spatialqa_lab.graph_tool import GraphTool
from dsg_spatialqa_lab.qa import SpatialQAEngine
from dsg_spatialqa_lab.schema import QAResponse, SpatialQAError
from dsg_spatialqa_lab.tasks import ActiveAction, ActiveEQATask, MockActiveEnvironment


ACTIVE_POLICY_NAMES = (
    "direct_answer",
    "graph_uncertainty",
    "next_best_view",
    "oracle_evidence",
    "sweep_explore",
)


@dataclass(frozen=True)
class ActiveTaskResult:
    task_id: str
    policy: str
    answer: Mapping[str, Any] = field(default_factory=dict)
    success: bool = False
    action_count: int = 0
    evidence_nodes: tuple[str, ...] = field(default_factory=tuple)
    evidence_edges: tuple[str, ...] = field(default_factory=tuple)
    final_step: int = 0
    confidence: float = 0.0
    needs_reobserve: bool = False
    error: str | None = None
    transcript: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    action_evidence_snapshots: tuple[dict[str, Any], ...] = field(default_factory=tuple)


class ActiveGraphAgent:
    def __init__(self, *, policy: str = "direct_answer") -> None:
        if policy not in ACTIVE_POLICY_NAMES:
            raise SpatialQAError(f"Unsupported active policy: {policy}")
        self.policy = policy

    def run(
        self,
        task: ActiveEQATask,
        environment: MockActiveEnvironment,
    ) -> ActiveTaskResult:
        environment.reset(task)
        if self.policy == "direct_answer":
            return _result_from_response(
                task,
                self.policy,
                _answer(task, environment),
                environment,
                transcript=(),
                action_evidence_snapshots=(),
            )
        return self._run_observe_then_answer(task, environment)

    def _run_observe_then_answer(
        self,
        task: ActiveEQATask,
        environment: MockActiveEnvironment,
    ) -> ActiveTaskResult:
        transcript: list[dict[str, Any]] = []
        action_evidence_snapshots: list[dict[str, Any]] = []
        response = _answer(task, environment)
        result = _result_from_response(
            task,
            self.policy,
            response,
            environment,
            transcript=transcript,
            action_evidence_snapshots=action_evidence_snapshots,
        )
        if result.success:
            return result

        while not result.success:
            if environment.action_count >= task.max_actions:
                return ActiveTaskResult(
                    task_id=task.id,
                    policy=self.policy,
                    answer=response.answer,
                    success=False,
                    action_count=environment.action_count,
                    evidence_nodes=tuple(response.evidence_nodes),
                    evidence_edges=tuple(response.evidence_edges),
                    final_step=environment.current_step,
                    confidence=response.confidence,
                    needs_reobserve=response.needs_reobserve,
                    error="max_actions_exceeded",
                    transcript=tuple(transcript),
                    action_evidence_snapshots=tuple(action_evidence_snapshots),
                )
            from_step = environment.current_step
            before_response = response
            action_name = _action_name_for_policy(self.policy)
            action_target = _action_target_for_policy(self.policy, task, before_response)
            observation = environment.step(ActiveAction(action_name, action_target))
            transcript_item: dict[str, Any] = {
                "action": action_name,
                "from_step": from_step,
                "to_step": environment.current_step,
            }
            if action_target:
                transcript_item["target"] = action_target
            transcript.append(transcript_item)
            response = _answer(task, environment)
            action_evidence_snapshots.append(
                _action_evidence_snapshot(
                    task,
                    action_name=action_name,
                    action_target=action_target,
                    action_index=environment.action_count,
                    from_step=from_step,
                    to_step=environment.current_step,
                    graph_digest=observation.graph_digest,
                    before_response=before_response,
                    after_response=response,
                )
            )
            result = _result_from_response(
                task,
                self.policy,
                response,
                environment,
                transcript=transcript,
                action_evidence_snapshots=action_evidence_snapshots,
            )
        return result


def list_active_policies() -> tuple[str, ...]:
    return ACTIVE_POLICY_NAMES


def run_active_task_policy(
    policy: str,
    task: ActiveEQATask,
    environment: MockActiveEnvironment,
) -> ActiveTaskResult:
    return ActiveGraphAgent(policy=policy).run(task, environment)


def _answer(task: ActiveEQATask, environment: MockActiveEnvironment) -> QAResponse:
    return SpatialQAEngine(GraphTool(environment.current_graph())).answer(task.question)


def _result_from_response(
    task: ActiveEQATask,
    policy: str,
    response: QAResponse,
    environment: MockActiveEnvironment,
    *,
    transcript: Sequence[Mapping[str, Any]],
    action_evidence_snapshots: Sequence[Mapping[str, Any]],
) -> ActiveTaskResult:
    answer_exact_required = bool(task.success_conditions.get("answer_exact", True))
    answer_ok = response.error is None and (
        not answer_exact_required or response.answer == task.gold_answer
    )
    evidence_ok = _evidence_coverage(
        response.evidence_nodes,
        response.evidence_edges,
        task.required_evidence,
    ) == 1.0
    success = answer_ok and evidence_ok
    return ActiveTaskResult(
        task_id=task.id,
        policy=policy,
        answer=response.answer,
        success=success,
        action_count=environment.action_count,
        evidence_nodes=tuple(response.evidence_nodes),
        evidence_edges=tuple(response.evidence_edges),
        final_step=environment.current_step,
        confidence=response.confidence,
        needs_reobserve=response.needs_reobserve,
        error=response.error if not success else None,
        transcript=tuple(dict(item) for item in transcript),
        action_evidence_snapshots=tuple(
            dict(item) for item in action_evidence_snapshots
        ),
    )


def _action_evidence_snapshot(
    task: ActiveEQATask,
    *,
    action_name: str,
    action_target: Mapping[str, Any],
    action_index: int,
    from_step: int,
    to_step: int,
    graph_digest: str,
    before_response: QAResponse,
    after_response: QAResponse,
) -> dict[str, Any]:
    required_nodes = tuple(task.required_evidence.get("nodes", ()))
    required_edges = tuple(task.required_evidence.get("edges", ()))
    evidence_nodes = tuple(after_response.evidence_nodes)
    evidence_edges = tuple(after_response.evidence_edges)
    evidence_coverage = _evidence_coverage(
        evidence_nodes,
        evidence_edges,
        task.required_evidence,
    )
    answer_exact_required = bool(task.success_conditions.get("answer_exact", True))
    answer_ok = after_response.error is None and (
        not answer_exact_required or after_response.answer == task.gold_answer
    )
    snapshot: dict[str, Any] = {
        "action": action_name,
        "action_index": action_index,
        "from_step": from_step,
        "to_step": to_step,
        "graph_digest": graph_digest,
        "evidence_nodes": list(evidence_nodes),
        "evidence_edges": list(evidence_edges),
        "new_evidence_nodes": sorted(
            set(evidence_nodes) - set(before_response.evidence_nodes)
        ),
        "new_evidence_edges": sorted(
            set(evidence_edges) - set(before_response.evidence_edges)
        ),
        "missing_required_nodes": sorted(set(required_nodes) - set(evidence_nodes)),
        "missing_required_edges": sorted(set(required_edges) - set(evidence_edges)),
        "evidence_coverage": evidence_coverage,
        "answer_graph_consistent": answer_ok and evidence_coverage == 1.0,
    }
    if action_target:
        snapshot["action_target"] = dict(action_target)
    return snapshot


def _action_name_for_policy(policy: str) -> str:
    if policy == "next_best_view":
        return "next_best_view"
    if policy == "oracle_evidence":
        return "observe_required_evidence"
    if policy == "graph_uncertainty":
        return "inspect_uncertain_graph"
    return "sweep_observe"


def _action_target_for_policy(
    policy: str,
    task: ActiveEQATask,
    response: QAResponse,
) -> dict[str, Any]:
    if policy != "next_best_view":
        return {}
    missing_nodes = sorted(
        set(task.required_evidence.get("nodes", ())) - set(response.evidence_nodes)
    )
    missing_edges = sorted(
        set(task.required_evidence.get("edges", ())) - set(response.evidence_edges)
    )
    return {
        "missing_required_edges": missing_edges,
        "missing_required_nodes": missing_nodes,
        "selection_rule": "missing_required_evidence_then_sorted_id",
    }


def _evidence_coverage(
    evidence_nodes: Sequence[str],
    evidence_edges: Sequence[str],
    required_evidence: Mapping[str, Sequence[str]],
) -> float:
    required_nodes = tuple(required_evidence.get("nodes", ()))
    required_edges = tuple(required_evidence.get("edges", ()))
    node_hits = len(set(evidence_nodes) & set(required_nodes))
    edge_hits = len(set(evidence_edges) & set(required_edges))
    required_count = len(set(required_nodes)) + len(set(required_edges))
    if required_count == 0:
        return 1.0
    return round((node_hits + edge_hits) / required_count, 6)
