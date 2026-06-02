from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from dsg_spatialqa_lab.benchmark import QACase
from dsg_spatialqa_lab.eval import QAPrediction
from dsg_spatialqa_lab.memory import DynamicSceneGraph


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    kind: str
    enabled: bool


class BaselineAgent(Protocol):
    spec: BaselineSpec

    def predict(
        self,
        graph: DynamicSceneGraph,
        cases: Sequence[QACase],
    ) -> list[QAPrediction]: ...


def baseline_spec_to_dict(spec: BaselineSpec) -> dict[str, object]:
    return {
        "enabled": spec.enabled,
        "kind": spec.kind,
        "name": spec.name,
    }
