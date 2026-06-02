from __future__ import annotations

from collections.abc import Sequence

from dsg_spatialqa_lab.agents.base import BaselineSpec
from dsg_spatialqa_lab.benchmark import QACase
from dsg_spatialqa_lab.eval import QAPrediction
from dsg_spatialqa_lab.graph_tool import GraphTool
from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.qa import SpatialQAEngine


class GraphToolBaselineAgent:
    spec = BaselineSpec(name="graph_tool", kind="local", enabled=True)

    def predict(
        self,
        graph: DynamicSceneGraph,
        cases: Sequence[QACase],
    ) -> list[QAPrediction]:
        qa = SpatialQAEngine(GraphTool(graph))
        predictions: list[QAPrediction] = []
        for case in cases:
            response = qa.answer(case.question)
            if response.error is not None:
                predictions.append(
                    QAPrediction(
                        id=case.id,
                        confidence=0.0,
                        error=response.error,
                    )
                )
                continue
            predictions.append(
                QAPrediction(
                    id=case.id,
                    answer=response.answer,
                    evidence_nodes=tuple(response.evidence_nodes),
                    evidence_edges=tuple(response.evidence_edges),
                    confidence=response.confidence,
                )
            )
        return predictions
