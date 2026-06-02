from __future__ import annotations

from collections.abc import Sequence

from dsg_spatialqa_lab.agents.base import BaselineSpec
from dsg_spatialqa_lab.benchmark import QACase
from dsg_spatialqa_lab.eval import QAPrediction
from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.scene_io import graph_summary


class GraphTextBaselineAgent:
    spec = BaselineSpec(name="graph_text", kind="local", enabled=True)

    def predict(
        self,
        graph: DynamicSceneGraph,
        cases: Sequence[QACase],
    ) -> list[QAPrediction]:
        summary = graph_summary(graph)
        return [
            QAPrediction(
                id=case.id,
                answer={"graph_summary": summary},
                confidence=0.0,
                error="unsupported_question",
            )
            for case in cases
        ]
