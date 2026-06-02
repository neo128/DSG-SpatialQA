from __future__ import annotations

from dsg_spatialqa_lab.graph_tool import GraphTool
from dsg_spatialqa_lab.schema import PlannerResult, Pose3D, SkillCommand, SpatialQAError


class VLAAnchorPlanner:
    _PLACE_RELATIONS = {"LEFT_OF", "RIGHT_OF", "FRONT_OF", "BEHIND", "NEAR"}

    def __init__(self, graph_tool: GraphTool, *, place_margin: float = 0.1) -> None:
        self.graph_tool = graph_tool
        self.place_margin = place_margin

    def plan_pick(
        self,
        *,
        target_object: str | None = None,
        label: str | None = None,
    ) -> PlannerResult:
        try:
            resolved_target = self._resolve_target(target_object, label)
            if isinstance(resolved_target, PlannerResult):
                return resolved_target

            state = self.graph_tool.get_object(resolved_target)
            precondition_failure = self._target_precondition_failure(resolved_target)
            if precondition_failure is not None:
                return precondition_failure

            command = SkillCommand(
                skill="pick",
                target_object=resolved_target,
                target_pose=state.pose,
                preconditions=self._pick_preconditions(resolved_target),
                evidence=[resolved_target],
            )
            return PlannerResult(status="ok", command=command)
        except SpatialQAError as exc:
            return PlannerResult(status="error", error=str(exc))

    def plan_place_relative(
        self,
        object_id: str | None,
        reference_object_id: str | None,
        relation: str,
        *,
        target_label: str | None = None,
        reference_label: str | None = None,
    ) -> PlannerResult:
        try:
            resolved_target = self._resolve_target(
                object_id,
                target_label,
                missing_error="target_object or target_label is required",
            )
            if isinstance(resolved_target, PlannerResult):
                return resolved_target
            resolved_reference = self._resolve_target(
                reference_object_id,
                reference_label,
                missing_error="reference_object or reference_label is required",
            )
            if isinstance(resolved_reference, PlannerResult):
                return resolved_reference

            normalized = relation.upper()
            if normalized not in self._PLACE_RELATIONS:
                return PlannerResult(status="error", error=f"Unsupported place relation: {relation}")

            target_failure = self._target_precondition_failure(resolved_target)
            if target_failure is not None:
                return target_failure
            reference_failure = self._target_precondition_failure(
                resolved_reference,
                detail_object_key="reference_object",
            )
            if reference_failure is not None:
                return reference_failure

            target_state = self.graph_tool.get_object(resolved_target)
            anchor_pose = self._anchor_pose(target_state.bbox.size, resolved_reference, normalized)
            command = SkillCommand(
                skill="place_relative",
                target_object=resolved_target,
                target_pose=anchor_pose,
                reference_object=resolved_reference,
                preconditions=self._pick_preconditions(resolved_target)
                + self._pick_preconditions(resolved_reference),
                evidence=[resolved_target, resolved_reference],
                parameters={"relation": normalized},
            )
            return PlannerResult(status="ok", command=command)
        except SpatialQAError as exc:
            return PlannerResult(status="error", error=str(exc))

    def validate(self, command: SkillCommand) -> PlannerResult:
        if command.skill == "place_relative":
            return self._validate_place_relative(command)

        validation = self.graph_tool.validate_next_action(command)
        if validation.valid:
            return PlannerResult(status="ok", command=command)
        return PlannerResult(
            status="needs_replan" if validation.needs_replan else "invalid",
            error=validation.reason,
            needs_reobserve=validation.reason == "needs_reobserve",
            needs_replan=validation.needs_replan,
            details=validation.details,
        )

    def _validate_place_relative(self, command: SkillCommand) -> PlannerResult:
        try:
            reference_object = command.reference_object
            if reference_object is None:
                return PlannerResult(status="error", error="place_relative missing reference_object")
            relation = command.parameters.get("relation")
            if not isinstance(relation, str):
                return PlannerResult(status="error", error="place_relative missing relation")
            normalized = relation.upper()
            if normalized not in self._PLACE_RELATIONS:
                return PlannerResult(status="error", error=f"Unsupported place relation: {relation}")

            target_failure = self._target_precondition_failure(command.target_object)
            if target_failure is not None:
                return target_failure
            reference_failure = self._target_precondition_failure(
                reference_object,
                detail_object_key="reference_object",
            )
            if reference_failure is not None:
                return reference_failure

            target_expected_step = self._object_precondition_value(
                command,
                command.target_object,
                "last_seen_step",
            )
            target_current_step = self.graph_tool.get_object(command.target_object).last_seen_step
            if target_expected_step is not None and target_expected_step != target_current_step:
                return PlannerResult(
                    status="needs_replan",
                    error="stale_object_state",
                    needs_replan=True,
                    details={
                        "target_object": command.target_object,
                        "expected_last_seen_step": target_expected_step,
                        "current_last_seen_step": target_current_step,
                    },
                )

            current_anchor_pose = self._anchor_pose(
                self.graph_tool.get_object(command.target_object).bbox.size,
                reference_object,
                normalized,
            )
            reference_expected_step = self._object_precondition_value(
                command,
                reference_object,
                "last_seen_step",
            )
            reference_current_step = self.graph_tool.get_object(reference_object).last_seen_step
            if (
                reference_expected_step is not None
                and reference_expected_step != reference_current_step
            ) or not current_anchor_pose.almost_equals(command.target_pose):
                return PlannerResult(
                    status="needs_replan",
                    error="stale_reference_state",
                    needs_replan=True,
                    details={
                        "target_object": command.target_object,
                        "reference_object": reference_object,
                        "relation": normalized,
                        "expected_anchor_pose": command.target_pose.to_dict(),
                        "current_anchor_pose": current_anchor_pose.to_dict(),
                        "expected_reference_last_seen_step": reference_expected_step,
                        "current_reference_last_seen_step": reference_current_step,
                    },
                )

            return PlannerResult(status="ok", command=command)
        except SpatialQAError as exc:
            return PlannerResult(status="error", error=str(exc))

    def _resolve_target(
        self,
        target_object: str | None,
        label: str | None,
        *,
        missing_error: str = "target_object or label is required",
    ) -> str | PlannerResult:
        if target_object is not None:
            return target_object
        if label is None:
            return PlannerResult(status="error", error=missing_error)

        matches = self.graph_tool.find_objects(label=label)
        if not matches:
            return PlannerResult(status="error", error=f"Object label not found: {label}")
        if len(matches) > 1:
            return PlannerResult(
                status="ambiguous",
                error=f"Ambiguous label: {label}",
                ambiguous_ids=[state.object_id for state in matches],
                details={
                    "label": label,
                    "candidate_count": len(matches),
                    "candidates": [self._candidate_details(state.object_id) for state in matches],
                },
            )
        return matches[0].object_id

    def _candidate_details(self, object_id: str) -> dict[str, object]:
        state = self.graph_tool.get_object(object_id)
        return {
            "object_id": state.object_id,
            "label": state.label,
            "pose": state.pose.to_dict(),
            "visible": state.visible,
            "confidence": state.confidence,
            "last_seen_step": state.last_seen_step,
            "needs_reobserve": self.graph_tool.needs_reobserve(object_id),
        }

    def _target_precondition_failure(
        self,
        object_id: str,
        *,
        detail_object_key: str = "target_object",
    ) -> PlannerResult | None:
        state = self.graph_tool.get_object(object_id)
        if self.graph_tool.needs_reobserve(object_id):
            return PlannerResult(
                status="needs_reobserve",
                error="needs_reobserve",
                needs_reobserve=True,
                details=self._precondition_details(
                    object_id,
                    object_key=detail_object_key,
                ),
            )
        if not state.visible:
            return PlannerResult(
                status="needs_replan",
                error="target_not_visible",
                needs_replan=True,
                details=self._precondition_details(
                    object_id,
                    object_key=detail_object_key,
                ),
            )
        if state.confidence < self.graph_tool.reobserve_confidence_threshold:
            return PlannerResult(
                status="needs_replan",
                error="low_confidence",
                needs_replan=True,
                details=self._precondition_details(
                    object_id,
                    object_key=detail_object_key,
                ),
            )
        return None

    def _precondition_details(
        self,
        object_id: str,
        *,
        object_key: str = "target_object",
    ) -> dict[str, object]:
        state = self.graph_tool.get_object(object_id)
        return {
            object_key: object_id,
            "visible": state.visible,
            "confidence": state.confidence,
            "min_confidence": self.graph_tool.reobserve_confidence_threshold,
            "last_seen_step": state.last_seen_step,
            "current_step": state.step,
        }

    def _pick_preconditions(self, object_id: str) -> list[dict[str, object]]:
        state = self.graph_tool.get_object(object_id)
        return [
            {"type": "visible", "object_id": object_id, "value": True},
            {
                "type": "min_confidence",
                "object_id": object_id,
                "value": self.graph_tool.reobserve_confidence_threshold,
            },
            {"type": "last_seen_step", "object_id": object_id, "value": state.last_seen_step},
        ]

    @staticmethod
    def _object_precondition_value(
        command: SkillCommand,
        object_id: str,
        precondition_type: str,
    ) -> object | None:
        for precondition in command.preconditions:
            if (
                precondition.get("object_id") == object_id
                and precondition.get("type") == precondition_type
            ):
                return precondition.get("value")
        return None

    def _anchor_pose(
        self,
        target_size: tuple[float, float, float],
        reference_object_id: str,
        relation: str,
    ) -> Pose3D:
        reference = self.graph_tool.get_object(reference_object_id)
        x = reference.pose.x
        y = reference.pose.y
        x_offset = reference.bbox.half_x + target_size[0] / 2.0 + self.place_margin
        y_offset = reference.bbox.half_y + target_size[1] / 2.0 + self.place_margin

        if relation == "RIGHT_OF":
            x += x_offset
        elif relation == "LEFT_OF":
            x -= x_offset
        elif relation == "FRONT_OF":
            y += y_offset
        elif relation == "BEHIND":
            y -= y_offset
        elif relation == "NEAR":
            x += x_offset

        z = reference.bbox.max_z + target_size[2] / 2.0
        return Pose3D(round(x, 6), round(y, 6), round(z, 6), yaw=reference.pose.yaw)
