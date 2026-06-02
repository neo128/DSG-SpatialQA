from __future__ import annotations

from collections.abc import Sequence

from dsg_spatialqa_lab.agents.active_graph_agent import (
    ActiveGraphAgent as ActiveGraphAgent,
    ActiveTaskResult as ActiveTaskResult,
    list_active_policies as list_active_policies,
    run_active_task_policy as run_active_task_policy,
)
from dsg_spatialqa_lab.agents.base import BaselineAgent, BaselineSpec, baseline_spec_to_dict
from dsg_spatialqa_lab.agents.caption_memory_agent import CaptionMemoryBaselineAgent
from dsg_spatialqa_lab.agents.graph_text_agent import GraphTextBaselineAgent
from dsg_spatialqa_lab.agents.graph_tool_agent import GraphToolBaselineAgent
from dsg_spatialqa_lab.agents.majority_agent import MajorityBaselineAgent
from dsg_spatialqa_lab.benchmark import QACase
from dsg_spatialqa_lab.eval import QAPrediction
from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.schema import SpatialQAError


def list_baselines() -> tuple[BaselineSpec, ...]:
    return tuple(agent.spec for agent in _baseline_agents())


def get_baseline_agent(name: str) -> BaselineAgent:
    for agent in _baseline_agents():
        if agent.spec.name == name:
            return agent
    raise SpatialQAError(f"Unsupported baseline: {name}")


def run_baseline_predictions(
    name: str,
    *,
    graph: DynamicSceneGraph,
    cases: Sequence[QACase],
) -> list[QAPrediction]:
    return get_baseline_agent(name).predict(graph, cases)


def _baseline_agents() -> tuple[BaselineAgent, ...]:
    return (
        CaptionMemoryBaselineAgent(),
        GraphTextBaselineAgent(),
        GraphToolBaselineAgent(),
        MajorityBaselineAgent(),
    )


__all__ = [
    "ActiveGraphAgent",
    "ActiveTaskResult",
    "BaselineAgent",
    "BaselineSpec",
    "CaptionMemoryBaselineAgent",
    "GraphTextBaselineAgent",
    "GraphToolBaselineAgent",
    "MajorityBaselineAgent",
    "baseline_spec_to_dict",
    "get_baseline_agent",
    "list_active_policies",
    "list_baselines",
    "run_active_task_policy",
    "run_baseline_predictions",
]
