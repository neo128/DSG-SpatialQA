from __future__ import annotations

from collections.abc import Sequence

from dsg_spatialqa_lab.agents.base import BaselineSpec
from dsg_spatialqa_lab.benchmark import QACase
from dsg_spatialqa_lab.eval import QAPrediction
from dsg_spatialqa_lab.memory import DynamicSceneGraph


class CaptionMemoryBaselineAgent:
    spec = BaselineSpec(name="caption_memory", kind="interface", enabled=False)

    def predict(
        self,
        graph: DynamicSceneGraph,
        cases: Sequence[QACase],
    ) -> list[QAPrediction]:
        _ = graph
        return [
            QAPrediction(
                id=case.id,
                confidence=0.0,
                error="baseline_disabled",
            )
            for case in cases
        ]
