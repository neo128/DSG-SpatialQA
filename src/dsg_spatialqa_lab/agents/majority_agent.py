from __future__ import annotations

from collections.abc import Sequence

from dsg_spatialqa_lab.agents.base import BaselineSpec
from dsg_spatialqa_lab.benchmark import QACase
from dsg_spatialqa_lab.eval import QAPrediction
from dsg_spatialqa_lab.memory import DynamicSceneGraph


class MajorityBaselineAgent:
    spec = BaselineSpec(name="majority", kind="local", enabled=True)

    def predict(
        self,
        graph: DynamicSceneGraph,
        cases: Sequence[QACase],
    ) -> list[QAPrediction]:
        _ = graph
        predictions: list[QAPrediction] = []
        for case in cases:
            if case.choices:
                predictions.append(
                    QAPrediction(
                        id=case.id,
                        answer={"choice": case.choices[0], "strategy": "first_choice"},
                        confidence=0.0,
                    )
                )
                continue
            predictions.append(
                QAPrediction(
                    id=case.id,
                    confidence=0.0,
                    error="unsupported_question",
                )
            )
        return predictions
