from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.action.registry import action_actuator_id
from education.intervention import normalize_education_intervention


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(float(low), min(float(high), float(value)))


@dataclass
class SkillScaffoldState:
    """Mutable teacher-side state for one temporary AP skill scaffold.

    This object lives outside AP core. It watches AP traces like a teacher and
    emits ordinary education interventions. AP may accept, ignore, or learn
    from those interventions through its normal action and feedback pathways.
    """

    skill_id: str
    enabled: bool = False
    base_strength: float = 0.74
    mastery_estimate: float = 0.0
    success_steps: int = 0
    failure_steps: int = 0
    demonstration_count: int = 0
    last_step_id: str = ""
    last_biases: list[dict] = field(default_factory=list)

    def effective_strength(self) -> float:
        return _round4(_clamp(self.base_strength, 0.0, 1.0) * (1.0 - 0.72 * _clamp(self.mastery_estimate)))


class SkillScaffoldController:
    """External rule teacher for the draft reread-before-commit skill.

    The controller is intentionally a teacher, not an AP subsystem. It emits
    `education_intervention/v1` payloads containing state hints, soft drive
    biases, and optional feedback. The AP runtime consumes those payloads
    through a generic interface shared with LLM and human teachers.
    """

    DRAFT_SKILL_ID = "skill::draft_reread_before_commit"

    def __init__(self) -> None:
        self._states: dict[str, SkillScaffoldState] = {
            self.DRAFT_SKILL_ID: SkillScaffoldState(skill_id=self.DRAFT_SKILL_ID)
        }

    def enable(self, skill_id: str = DRAFT_SKILL_ID, *, strength: float | None = None) -> dict:
        state = self._state(skill_id)
        state.enabled = True
        if strength is not None:
            state.base_strength = _clamp(float(strength), 0.0, 1.0)
        return self.trace(skill_id=skill_id, reason="teacher_enable")

    def disable(self, skill_id: str = DRAFT_SKILL_ID) -> dict:
        state = self._state(skill_id)
        state.enabled = False
        state.last_biases = []
        state.last_step_id = ""
        return self.trace(skill_id=skill_id, reason="teacher_disable")

    def trace(self, *, skill_id: str = DRAFT_SKILL_ID, reason: str = "status") -> dict:
        state = self._state(skill_id)
        return {
            "schema_id": "skill_scaffold_state/v1",
            "skill_id": skill_id,
            "teacher_boundary": "external_teacher_not_ap_core",
            "enabled": bool(state.enabled),
            "base_strength": _round4(state.base_strength),
            "effective_strength": state.effective_strength(),
            "mastery_estimate": _round4(state.mastery_estimate),
            "success_steps": int(state.success_steps),
            "failure_steps": int(state.failure_steps),
            "demonstration_count": int(state.demonstration_count),
            "last_step_id": str(state.last_step_id),
            "last_biases": [dict(row) for row in state.last_biases],
            "reason": str(reason),
        }

    def state_items(self, *, tick_index: int, skill_id: str = DRAFT_SKILL_ID) -> list[dict]:
        state = self._state(skill_id)
        if not state.enabled and not state.last_step_id:
            return []
        trace = self.trace(skill_id=skill_id, reason="state_field_projection")
        energy = 0.18 + (state.effective_strength() * 0.24 if state.enabled else 0.04)
        return [
            {
                "sa_label": f"education_hint::{skill_id.split('::')[-1]}",
                "display_text": f"education scaffold:{skill_id}",
                "family": "education_intervention",
                "source_type": "external_teacher",
                "real_energy": _round4(energy),
                "cognitive_pressure": _round4(energy * (0.45 if state.enabled else 0.16)),
                "anchor_meta": {
                    **trace,
                    "tick_index": int(tick_index),
                    "meaning": "teacher_hint_first_class_state_item_not_a_hard_gate",
                },
            }
        ]

    def build_draft_intervention(
        self,
        *,
        tick_index: int,
        draft_context: dict,
        expected_text: dict,
        cognitive_feelings: dict | None = None,
        skill_id: str = DRAFT_SKILL_ID,
    ) -> dict:
        state = self._state(skill_id)
        biases = self._build_draft_biases(
            tick_index=tick_index,
            draft_context=draft_context,
            expected_text=expected_text,
            cognitive_feelings=cognitive_feelings,
            skill_id=skill_id,
        )
        intervention = {
            "schema_id": "education_intervention/v1",
            "source": "skill_scaffold_controller",
            "teacher_kind": "rule_scaffold",
            "goal": "teach draft writing rhythm: write -> reread -> revise if needed -> internal commit",
            "tick_index": int(tick_index),
            "state_items": self.state_items(tick_index=tick_index, skill_id=skill_id),
            "action_biases": biases,
            "feedback": {},
            "notes": [
                "external_teacher_scaffold",
                "teacher_does_not_execute_actions",
                "content_from_current_cstar_or_recent_mismatch",
                f"enabled={state.enabled}",
            ],
        }
        return normalize_education_intervention(intervention, tick_index=tick_index)

    def build_draft_biases(
        self,
        *,
        tick_index: int,
        draft_context: dict,
        expected_text: dict,
        cognitive_feelings: dict | None = None,
        skill_id: str = DRAFT_SKILL_ID,
    ) -> dict:
        """Compatibility trace for older experiments.

        New code should prefer ``build_draft_intervention`` and pass the result
        through AP's generic education-intervention queue.
        """

        intervention = self.build_draft_intervention(
            tick_index=tick_index,
            draft_context=draft_context,
            expected_text=expected_text,
            cognitive_feelings=cognitive_feelings,
            skill_id=skill_id,
        )
        return {
            "schema_id": "skill_scaffold_bias_trace/v1",
            "skill_id": skill_id,
            "teacher_boundary": "external_teacher_not_ap_core",
            "enabled": bool(self._state(skill_id).enabled),
            "tick_index": int(tick_index),
            "effective_strength": self._state(skill_id).effective_strength(),
            "mastery_estimate": _round4(self._state(skill_id).mastery_estimate),
            "biases": list(intervention.get("action_biases", []) or []),
            "state_item": list(intervention.get("state_items", []) or []),
            "intervention": intervention,
            "meaning": "soft_bias_only_planner_may_ignore",
        }

    def _build_draft_biases(
        self,
        *,
        tick_index: int,
        draft_context: dict,
        expected_text: dict,
        cognitive_feelings: dict | None = None,
        skill_id: str = DRAFT_SKILL_ID,
    ) -> list[dict]:
        state = self._state(skill_id)
        if not state.enabled:
            state.last_biases = []
            return []

        draft = dict(draft_context or {})
        expected = dict(expected_text or {})
        feelings = dict((cognitive_feelings or {}).get("channels", {}) or {})
        effective = state.effective_strength()
        visible_length = int(draft.get("visible_length", 0) or 0)
        last_event_type = str(draft.get("last_event_type", "") or "")
        last_reread_age = int(draft.get("last_reread_age", 9999) or 9999)
        last_insert_age = int(draft.get("last_insert_age", 9999) or 9999)
        last_replace_age = int(draft.get("last_replace_age", 9999) or 9999)
        mismatch_count = int(draft.get("mismatch_count", 0) or 0)
        mismatch_index = int(draft.get("latest_mismatch_index", -1) or -1)
        mismatch_tick = int(draft.get("latest_mismatch_tick", -1) or -1)
        last_reread_tick = int(draft.get("last_reread_tick", -1) or -1)
        mismatch_expected_token = str(draft.get("latest_mismatch_expected_token", "") or "")
        expected_token = str(expected.get("token", "") or "")
        expected_strength = _clamp(float(expected.get("strength", 0.0) or 0.0), 0.0, 1.2)
        decisive = bool(expected.get("decisive", False))
        ambiguity = _clamp(float(expected.get("ambiguity", 0.0) or 0.0), 0.0, 1.0)
        pressure = _clamp(float(feelings.get("pressure", 0.0) or 0.0), 0.0, 1.0)
        correctness = _clamp(float(feelings.get("correctness", 0.0) or 0.0), 0.0, 1.0)
        dissonance = _clamp(float(feelings.get("dissonance", 0.0) or 0.0), 0.0, 1.0)

        biases: list[dict] = []
        if expected_token and (visible_length <= 0 or (decisive and expected_strength >= 0.38 and last_event_type not in {"insert", "replace"})):
            biases.append(
                self._bias(
                    skill_id=skill_id,
                    step_id="continue_from_cstar",
                    action_id="action::text_insert",
                    drive_delta=0.22 + expected_strength * 0.24,
                    strength=effective,
                    params={"token": expected_token, "reason": "teacher_hint_expected_token"},
                    notes=["content_from_current_cstar_not_skill_script", f"expected_token={expected_token}"],
                )
            )

        just_wrote = last_event_type in {"insert", "replace", "write_revision"} and min(last_insert_age, last_replace_age) <= 2
        if visible_length > 0 and (just_wrote or (last_reread_age > 3 and ambiguity >= 0.24)):
            biases.append(
                self._bias(
                    skill_id=skill_id,
                    step_id="review_after_write",
                    action_id="action::text_reread",
                    drive_delta=0.26 + min(0.16, visible_length * 0.025) + ambiguity * 0.10,
                    strength=effective,
                    params={"span": [0, visible_length], "reason": "teacher_hint_review_before_commit"},
                    notes=["reread_before_commit_teaching", "humanlike_pause_after_writing"],
                )
            )

        revision_token = mismatch_expected_token or expected_token
        reread_after_mismatch = bool(mismatch_tick >= 0 and last_reread_tick >= mismatch_tick)
        if visible_length > 0 and revision_token and reread_after_mismatch and last_reread_age <= 3 and mismatch_count > 0 and dissonance >= 0.22:
            span_start = mismatch_index if 0 <= mismatch_index < visible_length else max(0, visible_length - 1)
            biases.append(
                self._bias(
                    skill_id=skill_id,
                    step_id="revise_after_reread",
                    action_id="action::text_replace",
                    drive_delta=2.24 + expected_strength * 0.12 + dissonance * 0.20,
                    strength=effective,
                    params={"span": [span_start, span_start + 1], "new_text": revision_token, "reason": "teacher_hint_local_revision"},
                    notes=[
                        "local_revision_after_reread",
                        "parameterized_action_hint",
                        "replacement_from_prior_expected_token" if mismatch_expected_token else "replacement_from_current_cstar",
                    ],
                )
            )
            biases.append(
                self._bias(
                    skill_id=skill_id,
                    step_id="suppress_commit_until_revision",
                    action_id="action::text_commit",
                    drive_delta=-0.62,
                    strength=effective,
                    params={"reason": "teacher_hint_unresolved_mismatch"},
                    notes=["do_not_commit_unresolved_mismatch", "negative_bias_only_if_commit_candidate_exists"],
                )
            )

        no_clear_successor = not expected_token or (not decisive and expected_strength < 0.46) or ambiguity >= 0.42
        if visible_length > 0 and last_reread_age <= 3 and no_clear_successor and mismatch_count <= 0:
            commit_delta = 0.20 + correctness * 0.14 + max(0.0, 0.35 - pressure) * 0.10
            biases.append(
                self._bias(
                    skill_id=skill_id,
                    step_id="commit_after_reread",
                    action_id="action::text_commit",
                    drive_delta=commit_delta,
                    strength=effective,
                    params={"target_channel": "internal_draft", "reason": "teacher_hint_commit_after_reread"},
                    notes=["commit_only_after_reread", "internal_commit_not_real_send"],
                )
            )

        if visible_length > 0 and not expected_token and last_reread_age > 3:
            biases.append(
                self._bias(
                    skill_id=skill_id,
                    step_id="wait_when_ambiguous",
                    action_id="action::wait",
                    drive_delta=0.12,
                    strength=effective,
                    params={"reason": "teacher_hint_no_clear_successor_pause"},
                    notes=["pause_when_no_clear_successor"],
                )
            )

        state.last_biases = [dict(row) for row in biases]
        state.last_step_id = str(biases[0].get("step_id", "") if biases else "observe")
        return biases

    def observe_step_result(
        self,
        *,
        selected_actions: list[dict],
        text_output: dict | None,
        observed_feedback: dict | None = None,
        skill_id: str = DRAFT_SKILL_ID,
    ) -> dict:
        state = self._state(skill_id)
        selected_ids = {str(row.get("action_id", "") or "") for row in selected_actions or [] if isinstance(row, dict)}
        hinted_ids = {str(row.get("action_id", "") or "") for row in state.last_biases if isinstance(row, dict)}
        recent_events = [dict(row) for row in list((text_output or {}).get("recent_events", []) or []) if isinstance(row, dict)]
        reward = float((observed_feedback or {}).get("reward", 0.0) or 0.0)
        punishment = float((observed_feedback or {}).get("punishment", 0.0) or 0.0)
        correctness = float((observed_feedback or {}).get("correctness", 0.0) or 0.0)
        selected_hint = bool(selected_ids & hinted_ids)
        useful_event = any(str(event.get("event_type", "") or "") in {"insert", "reread", "replace", "commit"} for event in recent_events)
        success_score = (0.45 if selected_hint else 0.0) + (0.25 if useful_event else 0.0) + max(0.0, reward) * 0.16 + max(0.0, correctness) * 0.14 - max(0.0, punishment) * 0.28
        succeeded = success_score >= 0.42
        if succeeded:
            state.success_steps += 1
            state.demonstration_count += 1
        elif hinted_ids:
            state.failure_steps += 1
        total = max(1, state.success_steps + state.failure_steps)
        success_rate = state.success_steps / total
        target_mastery = _clamp(success_rate * min(1.0, state.success_steps / 8.0), 0.0, 1.0)
        state.mastery_estimate = _round4(state.mastery_estimate * 0.72 + target_mastery * 0.28)
        return {
            "schema_id": "skill_scaffold_step_feedback/v1",
            "teacher_boundary": "external_teacher_not_ap_core",
            "skill_id": skill_id,
            "selected_hint": selected_hint,
            "hinted_action_ids": sorted(hinted_ids),
            "selected_action_ids": sorted(selected_ids),
            "recent_event_types": [str(event.get("event_type", "") or "") for event in recent_events],
            "success_score": _round4(success_score),
            "succeeded": bool(succeeded),
            "mastery_estimate": _round4(state.mastery_estimate),
            "effective_strength_after": state.effective_strength(),
        }

    def _bias(
        self,
        *,
        skill_id: str,
        step_id: str,
        action_id: str,
        drive_delta: float,
        strength: float,
        params: dict | None = None,
        notes: list[str] | None = None,
    ) -> dict:
        effective_delta = _round4(float(drive_delta) * _clamp(float(strength), 0.0, 1.0))
        return {
            "schema_id": "education_action_bias/v1",
            "skill_id": skill_id,
            "step_id": step_id,
            "action_id": action_id,
            "actuator_id": action_actuator_id(action_id, "actuator::legacy_internal"),
            "drive_delta": effective_delta,
            "strength": _round4(strength),
            "params": dict(params or {}),
            "notes": [
                "education_intervention_bias",
                "soft_drive_bias_only",
                "parameterized_action_hint" if params else "unparameterized_action_hint",
                f"skill_id={skill_id}",
                f"step_id={step_id}",
            ]
            + list(notes or []),
        }

    def _state(self, skill_id: str) -> SkillScaffoldState:
        key = str(skill_id or self.DRAFT_SKILL_ID)
        if key not in self._states:
            self._states[key] = SkillScaffoldState(skill_id=key)
        return self._states[key]

