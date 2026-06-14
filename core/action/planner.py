from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from core.action.outcome_memory import ActionOutcomeMemory
from core.action.parameter_memory import ActionParameterMemory
from core.action.registry import action_actuator_id, action_meta, actuator_meta, is_external_action


PASSIVE_MAINTENANCE_ACTIONS = {
    "action::hold_gaze",
    "action::lock_audio_band",
    "action::wait",
}


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


class ActionConsequencePlanner:
    def __init__(
        self,
        *,
        enabled: bool,
        selection_threshold: float,
        max_selected_actions: int,
        fatigue_decay: float,
        fatigue_step: float,
        bias_learning_rate: float,
        bias_gain: float,
        confidence_gain: float,
        wait_base_drive: float,
        outcome_memory_enabled: bool = True,
        outcome_memory_learning_rate: float = 0.18,
        outcome_memory_decay_per_tick: float = 0.992,
        outcome_memory_support_scale: float = 6.0,
        outcome_memory_max_drive_bias: float = 0.75,
    ) -> None:
        self.enabled = bool(enabled)
        self.selection_threshold = max(0.0, float(selection_threshold))
        self.max_selected_actions = max(1, int(max_selected_actions))
        self.fatigue_decay = _clamp(fatigue_decay, 0.0, 1.0)
        self.fatigue_step = max(0.0, float(fatigue_step))
        self.bias_learning_rate = max(0.0, float(bias_learning_rate))
        self.bias_gain = max(0.0, float(bias_gain))
        self.confidence_gain = max(0.0, float(confidence_gain))
        self.wait_base_drive = max(0.0, float(wait_base_drive))
        self._actuator_fatigue: dict[str, float] = defaultdict(float)
        self._drive_bias: dict[str, float] = defaultdict(float)
        self._feedback_modulation: dict[str, dict[str, float | int]] = {}
        self._visual_target_fatigue: dict[str, float] = defaultdict(float)
        self._parameter_action_fatigue: dict[str, dict] = {}
        self._last_tick = -1
        self._last_raw_expected_text_context: dict = {}
        self._last_expected_text_context: dict = {}
        self._last_draft_context: dict = {}
        self._last_revision_opportunities: list[dict] = []
        self._last_draft_eval: dict = {}
        self._last_output_mismatch_context: dict = {}
        self._outcome_memory = ActionOutcomeMemory(
            enabled=outcome_memory_enabled,
            learning_rate=outcome_memory_learning_rate,
            decay_per_tick=outcome_memory_decay_per_tick,
            support_scale=outcome_memory_support_scale,
            max_drive_bias=outcome_memory_max_drive_bias,
        )
        self._parameter_memory = ActionParameterMemory(
            enabled=outcome_memory_enabled,
            learning_rate=outcome_memory_learning_rate,
            decay_per_tick=outcome_memory_decay_per_tick,
        )

    def plan(
        self,
        *,
        tick_index: int,
        state_snapshot_items: list[dict],
        attention_trace: dict | None = None,
        fast_bn: list[dict],
        fast_cn: list[dict],
        slow_bn: list[dict],
        slow_cn: list[dict],
        cognitive_feelings: dict,
        rhythm_trace: dict,
        time_trace: dict,
        expectation_pressure_trace: dict | None = None,
        residual_summary: dict | None = None,
        prediction_trace: dict | None = None,
        action_consequence_trace: dict | None = None,
        emotion_modulation: dict | None = None,
        innate_action_nodes: list[dict] | None = None,
        innate_action_biases: list[dict] | None = None,
        recent_thought_readback: dict | None = None,
        short_term_memory_readback: dict | None = None,
        memory_action_drive_gain: float = 0.28,
    ) -> dict:
        self._advance_tick(int(tick_index))
        if not self.enabled:
            return {"candidates": [], "selected_actions": [], "feedback_items": [], "drive_state": self._drive_snapshot()}
        candidates = self._build_candidates(
            tick_index=int(tick_index),
            state_snapshot_items=state_snapshot_items,
            attention_trace=attention_trace,
            fast_bn=fast_bn,
            fast_cn=fast_cn,
            slow_bn=slow_bn,
            slow_cn=slow_cn,
            cognitive_feelings=cognitive_feelings,
            expectation_pressure_trace=expectation_pressure_trace,
            rhythm_trace=rhythm_trace,
            time_trace=time_trace,
            residual_summary=residual_summary,
            prediction_trace=prediction_trace,
            action_consequence_trace=action_consequence_trace,
            emotion_modulation=emotion_modulation,
            innate_action_nodes=innate_action_nodes,
            innate_action_biases=innate_action_biases,
            recent_thought_readback=recent_thought_readback,
            short_term_memory_readback=short_term_memory_readback,
            memory_action_drive_gain=memory_action_drive_gain,
        )
        # Apply emotion modulation to selection threshold (8-channel NT system)
        # emotion_modulation format: {"attention": {...}, "hdb": {...}, "action": {...}}
        action_mod = (emotion_modulation or {}).get("action", {})
        threshold_adjustment = float(action_mod.get("threshold_adjustment", 0.0))
        effective_threshold = max(0.1, self.selection_threshold + threshold_adjustment)
        candidates.sort(key=lambda item: (-float(item["drive"]), item["action_id"]))
        selected, competition_trace = self._select_with_competition(
            candidates=candidates,
            effective_threshold=effective_threshold,
            max_selected_actions=self.max_selected_actions,
        )
        action_items = self._build_action_items(selected, tick_index=int(tick_index))
        return {
            "candidates": candidates,
            "selected_actions": selected,
            "action_items": action_items,
            "feedback_items": [],
            "drive_state": self._drive_snapshot(),
            "effective_threshold": _round4(effective_threshold),
            "consequence_trace": dict(action_consequence_trace or {}),
            "competition_trace": competition_trace,
            "planner_text_context": {
                "schema_id": "planner_text_context_trace/v1",
                "raw_expected_text": dict(self._last_raw_expected_text_context),
                "expected_text": dict(self._last_expected_text_context),
                "draft_context": dict(self._last_draft_context),
                "revision_opportunities": [dict(row) for row in self._last_revision_opportunities[:8]],
                "draft_eval": dict(self._last_draft_eval),
                "output_mismatch": dict(self._last_output_mismatch_context),
                "audit_only": True,
                "used_as_ap_input": False,
            },
        }

    def record_feedback(self, *, selected_actions: list[dict], observed_feedback: dict, parameter_events: list[dict] | None = None) -> dict:
        reward = float(observed_feedback.get("reward", 0.0) or 0.0)
        punishment = float(observed_feedback.get("punishment", 0.0) or 0.0)
        correctness = float(observed_feedback.get("correctness", 0.0) or 0.0)
        confidence = float(observed_feedback.get("confidence", 0.0) or 0.0)
        utility = reward + correctness * 0.4 - punishment
        events_by_action = self._parameter_events_by_action(parameter_events or [])
        parameter_estimates = []
        for row in selected_actions:
            action_id = str(row.get("action_id", "") or "")
            if not action_id:
                continue
            outcome_estimate = self._outcome_memory.record(
                action_id=action_id,
                observed_feedback=observed_feedback,
                predicted_outcome=dict(row.get("predicted_outcome", {}) or {}),
            )
            actuator_id = str(row.get("actuator_id", "") or "")
            self._drive_bias[action_id] = _clamp(
                float(self._drive_bias[action_id]) + utility * self.bias_learning_rate,
                -1.0,
                1.0,
            )
            self._actuator_fatigue[actuator_id] = _clamp(
                float(self._actuator_fatigue[actuator_id]) + self.fatigue_step * max(0.5, confidence),
                0.0,
                1.0,
            )
            self._record_parameter_action_fatigue(
                action_id=action_id,
                actuator_id=actuator_id,
                params=dict(row.get("params", {}) or {}),
                confidence=confidence,
                utility=utility,
            )
            modulation = 1.0
            ttl = 0
            if utility < -0.04:
                modulation = 0.48
                ttl = 2
            elif utility < 0.04:
                modulation = 0.72
                ttl = 1
            elif utility > 0.32:
                modulation = 1.08
                ttl = 1
            self._feedback_modulation[action_id] = {
                "modulation": _round4(modulation),
                "ttl": int(ttl),
                "last_utility": _round4(utility),
                "outcome_support": _round4(float(outcome_estimate.get("support", 0.0) or 0.0)),
                "outcome_drive_bias": _round4(float(outcome_estimate.get("drive_bias", 0.0) or 0.0)),
                }
            for event in events_by_action.get(action_id, []):
                parameter_estimates.append(
                    self._parameter_memory.record(
                        action_id=action_id,
                        selected_action=row,
                        control_event=event,
                        observed_feedback=observed_feedback,
                        tick_index=event.get("tick_index"),
                    )
                )
        self._update_visual_target_fatigue(
            selected_actions=selected_actions,
            parameter_events=parameter_events or [],
            observed_feedback=observed_feedback,
        )
        return {
            "updated_bias": {key: _round4(value) for key, value in self._drive_bias.items()},
            "updated_fatigue": {key: _round4(value) for key, value in self._actuator_fatigue.items()},
            "feedback_modulation": {key: dict(value) for key, value in self._feedback_modulation.items()},
            "outcome_memory": self._outcome_memory.snapshot(),
            "parameter_memory": self._parameter_memory.snapshot(),
            "parameter_estimates": parameter_estimates,
        }

    def _select_with_competition(
        self,
        *,
        candidates: list[dict],
        effective_threshold: float,
        max_selected_actions: int,
    ) -> tuple[list[dict], dict]:
        """
        Select actions through AP actuator conflict domains.

        A humanlike action field can contain two strong impulses at once. We do
        not erase the losing impulse; we record the competition cost so the
        later feedback/learning chain can know "this wanted to happen too".
        """

        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in candidates or []:
            domain = self._conflict_domain(row)
            grouped[domain].append(row)

        selected: list[dict] = []
        domain_rows: list[dict] = []
        epsilon = 0.0001
        for domain, rows in sorted(grouped.items(), key=lambda item: item[0]):
            rows = self._merge_compatible_candidates_for_competition(rows)
            ordered = sorted(rows, key=lambda item: (-float(item.get("drive", 0.0) or 0.0), str(item.get("action_id", "") or "")))
            if not ordered:
                continue
            winner = dict(ordered[0])
            second = dict(ordered[1]) if len(ordered) > 1 else {}
            winner_drive = float(winner.get("drive", 0.0) or 0.0)
            second_drive = float(second.get("drive", 0.0) or 0.0)
            competition_cost = max(0.0, second_drive - float(effective_threshold)) if second_drive >= float(effective_threshold) else 0.0
            # Same-actuator competition is shared inhibition over every
            # over-threshold impulse in the domain. The shared cost is the
            # amount needed to push the runner-up to the threshold, so the
            # strongest impulse is also deducted but remains executable only
            # when its remaining drive is still strictly above threshold.
            winner_after = max(0.0, winner_drive - competition_cost)
            suppressed = [
                str(row.get("action_id", "") or "")
                for row in ordered[1:]
                if float(row.get("drive", 0.0) or 0.0) >= float(effective_threshold)
            ]
            domain_trace = {
                "conflict_domain": domain,
                "winner_action_id": str(winner.get("action_id", "") or ""),
                "winner_drive_before": _round4(winner_drive),
                "winner_drive_after_competition": _round4(winner_after),
                "second_action_id": str(second.get("action_id", "") or ""),
                "second_drive": _round4(second_drive),
                "competition_cost": _round4(competition_cost),
                "competition_cost_applied_to": "all_same_domain_over_threshold_candidates",
                "second_drive_after_competition": _round4(
                    min(second_drive, float(effective_threshold))
                    if second_drive >= float(effective_threshold)
                    else second_drive
                ),
                "suppressed_action_ids": suppressed,
                "candidate_count": len(ordered),
                "compatible_candidate_merge_count": sum(
                    int(row.get("compatible_candidate_count", 1) or 1) - 1
                    for row in ordered
                    if isinstance(row, dict)
                ),
            }
            domain_rows.append(domain_trace)
            if winner_after <= float(effective_threshold):
                continue
            winner["drive_before_competition"] = _round4(winner_drive)
            winner["drive"] = _round4(winner_after)
            winner["competition_cost"] = _round4(competition_cost)
            winner["same_domain_shared_suppression"] = _round4(competition_cost)
            winner["suppressed_action_ids"] = suppressed
            winner["conflict_domain"] = domain
            winner["planner_selected"] = True
            winner["effective_threshold"] = _round4(effective_threshold)
            winner["effective_decisiveness"] = _round4(max(0.0, winner_after - float(effective_threshold)))
            winner.setdefault("notes", [])
            if competition_cost > 0.0:
                winner["notes"] = list(winner.get("notes", []) or []) + ["same_domain_shared_inhibition_applied_to_winner_and_loser"]
            selected.append(winner)

        selected.sort(key=lambda item: (-float(item.get("drive", 0.0) or 0.0), str(item.get("action_id", "") or "")))
        selected_ids = {str(row.get("action_id", "") or "") for row in selected}
        return selected, {
            "schema_id": "action_competition_trace/v1",
            "effective_threshold": _round4(effective_threshold),
            "policy": "one_winner_per_conflict_domain_independent_threshold_no_global_topn_clip",
            "max_selected_actions_observability_only": int(max_selected_actions),
            "selected_action_ids": sorted(selected_ids),
            "parallel_channel_floor": self._parallel_channel_floor_trace(selected),
            "domains": domain_rows,
            "suppressed_action_ids": sorted(
                {
                    action_id
                    for domain in domain_rows
                    for action_id in list(domain.get("suppressed_action_ids", []) or [])
                    if action_id and action_id not in selected_ids
                }
            ),
        }

    def _merge_compatible_candidates_for_competition(self, rows: list[dict]) -> list[dict]:
        """
        Merge same-effect impulses before same-domain competition.

        Two independent memories can propose the same low-grain action in the
        same tick. That is convergent evidence, not a motor conflict. Without
        this merge, two identical text_insert candidates can subtract each
        other's drive and suppress the very action both pathways support.
        Different text tokens, different explicit cursors, and different edit
        operations still compete normally.
        """

        groups: list[list[dict]] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            placed = False
            for group in groups:
                if group and self._are_compatible_candidate_effects(group[0], row):
                    group.append(row)
                    placed = True
                    break
            if not placed:
                groups.append([row])
        merged: list[dict] = []
        for group in groups:
            if len(group) <= 1:
                merged.append(dict(group[0]))
                continue
            ordered = sorted(
                [dict(row) for row in group],
                key=lambda item: (-float(item.get("drive", 0.0) or 0.0), -len(dict(item.get("params", {}) or {})), str(item.get("source", "") or "")),
            )
            primary = dict(ordered[0])
            primary_params = dict(primary.get("params", {}) or {})
            for support in ordered[1:]:
                for key, value in dict(support.get("params", {}) or {}).items():
                    current = primary_params.get(key)
                    if key not in primary_params or current is None or current == "" or current == []:
                        primary_params[key] = value
            support_bonus = min(0.18, 0.045 * (len(ordered) - 1))
            primary["params"] = primary_params
            primary["base_drive"] = _round4(max(float(row.get("base_drive", 0.0) or 0.0) for row in ordered) + support_bonus)
            primary["drive"] = _round4(_clamp(max(float(row.get("drive", 0.0) or 0.0) for row in ordered) + support_bonus, 0.0, 1.8))
            primary["compatible_candidate_count"] = len(ordered)
            primary["merged_candidate_evidence"] = [
                {
                    "action_id": str(row.get("action_id", "") or ""),
                    "drive": _round4(float(row.get("drive", 0.0) or 0.0)),
                    "params": dict(row.get("params", {}) or {}),
                    "notes": list(row.get("notes", []) or [])[:6],
                }
                for row in ordered
            ]
            primary["notes"] = list(primary.get("notes", []) or []) + [
                "compatible_same_effect_candidate_support_merged",
                f"compatible_candidate_count={len(ordered)}",
                f"compatible_support_bonus={_round4(support_bonus)}",
            ]
            merged.append(primary)
        return merged

    def _are_compatible_candidate_effects(self, left: dict, right: dict) -> bool:
        left_action = str((left or {}).get("action_id", "") or "")
        right_action = str((right or {}).get("action_id", "") or "")
        if left_action != right_action:
            return False
        if self._conflict_domain(left) != self._conflict_domain(right):
            return False
        left_params = dict((left or {}).get("params", {}) or {})
        right_params = dict((right or {}).get("params", {}) or {})
        if left_action == "action::text_insert":
            left_token = str(left_params.get("token", left_params.get("text", "")) or "")
            right_token = str(right_params.get("token", right_params.get("text", "")) or "")
            if not left_token or left_token != right_token:
                return False
            left_cursor = self._optional_int(left_params.get("cursor", left_params.get("cursor_hint", None)))
            right_cursor = self._optional_int(right_params.get("cursor", right_params.get("cursor_hint", None)))
            return bool(left_cursor is None or right_cursor is None or left_cursor == right_cursor)
        return False

    def _optional_int(self, value) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _apply_parallel_channel_floor(self, selected: list[dict], *, max_selected_actions: int) -> list[dict]:
        """
        Compatibility shim for the older global-clip era.

        APV2.1 now treats conflict domains as independent execution lanes:
        after each lane chooses its winner, there is no cross-lane TopN clip.
        This method remains only so older traces/tests can see that the previous
        sensorimotor-floor workaround is deliberately inactive.
        """

        return [dict(row) for row in selected]

    def _parallel_floor_replace_index(self, rows: list[dict]) -> int | None:
        protected_domains = {
            "single_visual_center",
            "visual_sampling_scale",
            "single_auditory_band_center",
            "auditory_sampling_width",
        }
        replaceable = [
            (idx, float(row.get("drive", 0.0) or 0.0), str(row.get("action_id", "") or ""))
            for idx, row in enumerate(rows)
            if str(row.get("conflict_domain", "") or self._conflict_domain(row)) not in protected_domains
            and str(row.get("conflict_domain", "") or self._conflict_domain(row))
            in {"attention_focus_width_and_anchor", "legacy_internal_prediction"}
            and str(row.get("action_id", "") or "") != "action::wait"
        ]
        if not replaceable:
            replaceable = [
                (idx, float(row.get("drive", 0.0) or 0.0), str(row.get("action_id", "") or ""))
                for idx, row in enumerate(rows)
                if str(row.get("conflict_domain", "") or self._conflict_domain(row)) not in protected_domains
                and str(row.get("conflict_domain", "") or self._conflict_domain(row))
                in {"attention_focus_width_and_anchor", "legacy_internal_prediction"}
            ]
        if not replaceable:
            return None
        replaceable.sort(key=lambda item: (item[1], item[2]))
        return int(replaceable[0][0])

    def _parallel_channel_floor_trace(self, selected: list[dict]) -> dict:
        return {
            "schema_id": "sensorimotor_parallel_channel_floor/v1",
            "enabled": False,
            "disabled_reason": "no_global_topn_clip_conflict_domains_are_independent_execution_lanes",
            "protected_domains": [
                "single_visual_center",
                "visual_sampling_scale",
                "single_auditory_band_center",
                "auditory_sampling_width",
            ],
            "selected_by_floor": [
                str(row.get("action_id", "") or "")
                for row in selected
                if bool(row.get("parallel_channel_floor", False))
            ],
        }

    def _conflict_domain(self, candidate: dict) -> str:
        actuator_id = str((candidate or {}).get("actuator_id", "") or "")
        meta = actuator_meta(actuator_id)
        domain = str(meta.get("conflict_domain", "") or "")
        return domain or actuator_id or "unknown_action_domain"

    def _build_candidates(
        self,
        *,
        tick_index: int,
        state_snapshot_items: list[dict],
        attention_trace: dict | None = None,
        fast_bn: list[dict],
        fast_cn: list[dict],
        slow_bn: list[dict],
        slow_cn: list[dict],
        cognitive_feelings: dict,
        rhythm_trace: dict,
        time_trace: dict,
        expectation_pressure_trace: dict | None = None,
        residual_summary: dict | None = None,
        prediction_trace: dict | None = None,
        action_consequence_trace: dict | None = None,
        emotion_modulation: dict | None = None,
        innate_action_nodes: list[dict] | None = None,
        innate_action_biases: list[dict] | None = None,
        recent_thought_readback: dict | None = None,
        short_term_memory_readback: dict | None = None,
        memory_action_drive_gain: float = 0.28,
    ) -> list[dict]:
        pressure = float((cognitive_feelings.get("channels", {}) or {}).get("pressure", 0.0) or 0.0)
        surprise = float((cognitive_feelings.get("channels", {}) or {}).get("surprise", 0.0) or 0.0)
        coherence = float((cognitive_feelings.get("channels", {}) or {}).get("coherence", 0.0) or 0.0)
        dissonance = float((cognitive_feelings.get("channels", {}) or {}).get("dissonance", 0.0) or 0.0)
        expectation = float((cognitive_feelings.get("channels", {}) or {}).get("expectation", 0.0) or 0.0)
        correctness = float((cognitive_feelings.get("channels", {}) or {}).get("correctness", 0.0) or 0.0)
        grasp = float((cognitive_feelings.get("channels", {}) or {}).get("grasp", 0.0) or 0.0)
        uncertainty = float((cognitive_feelings.get("channels", {}) or {}).get("uncertainty", 0.0) or 0.0)
        evidence_gap = float((cognitive_feelings.get("channels", {}) or {}).get("evidence_gap", 0.0) or 0.0)
        boredom = _clamp(float((cognitive_feelings.get("channels", {}) or {}).get("boredom", 0.0) or 0.0), 0.0, 1.0)
        fulfillment = _clamp(float((cognitive_feelings.get("channels", {}) or {}).get("fulfillment", 0.0) or 0.0), 0.0, 1.0)
        task_available = _clamp(float((cognitive_feelings.get("channels", {}) or {}).get("task_available", 0.0) or 0.0), 0.0, 1.0)
        unfinished_strength = _clamp(float((cognitive_feelings.get("channels", {}) or {}).get("unfinished_strength", 0.0) or 0.0), 0.0, 1.0)
        ep_channels = dict((expectation_pressure_trace or {}).get("channels", {}) or {})
        expectation_level = float(ep_channels.get("expectation_level", 0.0) or 0.0)
        pressure_level = float(ep_channels.get("pressure_level", 0.0) or 0.0)
        satisfaction_level = float(ep_channels.get("satisfaction_level", 0.0) or 0.0)
        expectation_gap = float(ep_channels.get("expectation_gap", 0.0) or 0.0)
        expectation_anchor_trace = dict((expectation_pressure_trace or {}).get("anchor_verification", {}) or {})
        expectation_anchors = self._active_expectation_anchors(expectation_anchor_trace)
        top_anchor_level = max([float(anchor.get("level", 0.0) or 0.0) for anchor in expectation_anchors] or [0.0])
        pressure_anchor_level = max(
            [
                float(anchor.get("level", 0.0) or 0.0)
                for anchor in expectation_anchors
                if str(anchor.get("anchor_type", "") or "") == "pressure"
            ]
            or [0.0]
        )
        expectation_anchor_level = max(
            [
                float(anchor.get("level", 0.0) or 0.0)
                for anchor in expectation_anchors
                if str(anchor.get("anchor_type", "") or "") == "expectation"
            ]
            or [0.0]
        )
        expectation = max(expectation, expectation_level)
        pressure = max(pressure, pressure_level)
        rhythm_expect = float((rhythm_trace.get("channels", {}) or {}).get("phase_expectation", 0.0) or 0.0)
        time_conf = float((time_trace.get("channels", {}) or {}).get("confidence", 0.0) or 0.0)
        dominant_time_peak = dict((time_trace or {}).get("dominant_peak", {}) or {})
        target_delta_t = dominant_time_peak.get("center_delta_t")
        time_sigma = dominant_time_peak.get("sigma", 1.0)
        predicted_mass = sum(len(branch.get("predicted_items", []) or []) for branch in fast_cn) + sum(
            len(branch.get("predicted_items", []) or []) for branch in slow_cn
        )
        focusable_count = sum(1 for item in state_snapshot_items if float(item.get("cognitive_pressure", 0.0) or 0.0) > 0.0)
        residual = dict(residual_summary or {})
        trace = dict(prediction_trace or {})
        residual_mass = float(residual.get("total_unresolved_mass", 0.0) or 0.0)
        residual_count = int(residual.get("count", 0) or 0)
        residual_drive = _clamp(residual_mass / max(1.0, residual_mass + len(state_snapshot_items)), 0.0, 1.0)
        current_labels = {str(item.get("sa_label", "") or "") for item in state_snapshot_items or [] if str(item.get("sa_label", "") or "")}
        protective_need = 0.0
        if "state::high_stakes_or_destructive" in current_labels:
            protective_need += 0.42
        if "desktop::risky_operation" in current_labels:
            protective_need += 0.34
        if "future_feedback::bad_if_wrong_action" in current_labels:
            protective_need += 0.28
        if "goal::avoid_wrong_action" in current_labels:
            protective_need += 0.22
        if "state::permission_uncertain" in current_labels:
            protective_need += 0.12
        protective_need = _clamp(protective_need, 0.0, 1.0)
        mismatch_ratio = float(trace.get("mismatch_ratio", 0.0) or 0.0)
        alignment_score = float(trace.get("alignment_score", 0.0) or 0.0)
        output_mismatch = self._output_mismatch_context(state_snapshot_items)
        correction_pressure = float(output_mismatch.get("correction_pressure", 0.0) or 0.0)
        latest_expected_token = str(output_mismatch.get("latest_expected_token", "") or "")
        reread_after_mismatch = bool(output_mismatch.get("reread_after_mismatch", False))
        draft_context = self._draft_writing_context(state_snapshot_items, current_tick=int(tick_index))
        expected_text = self._expected_text_context(
            fast_bn=fast_bn,
            slow_bn=slow_bn,
            fast_cn=fast_cn,
            slow_cn=slow_cn,
            draft_context=draft_context,
        )
        raw_expected_text = dict(expected_text)
        expected_text = self._advance_expected_text_after_visible_closure(expected_text, draft_context)
        expected_token = str(expected_text.get("token", "") or "")
        expected_strength = float(expected_text.get("strength", 0.0) or 0.0)
        expected_top_share = float(expected_text.get("top_share", 0.0) or 0.0)
        visible_length = int(draft_context.get("visible_length", 0) or 0)
        revision_opportunities = self._text_revision_opportunities(state_snapshot_items)
        self._last_raw_expected_text_context = dict(raw_expected_text)
        self._last_expected_text_context = dict(expected_text)
        self._last_draft_context = dict(draft_context)
        self._last_revision_opportunities = [dict(row) for row in revision_opportunities[:12]]
        self._last_output_mismatch_context = dict(output_mismatch)
        draft_eval = self._draft_self_evaluation(
            draft_context,
            expected_text,
            correctness=correctness,
            grasp=grasp,
            pressure=pressure,
            dissonance=dissonance,
            uncertainty=uncertainty,
        )
        self._last_draft_eval = dict(draft_eval)
        continuation_readiness = float(draft_eval.get("continuation_readiness", 0.0) or 0.0)
        ambiguity_pause = float(draft_eval.get("ambiguity_pause", 0.0) or 0.0)
        cleanup_pressure = float(draft_eval.get("cleanup_pressure", 0.0) or 0.0)
        draft_satisfaction = float(draft_eval.get("satisfaction", 0.0) or 0.0)
        successor_decisive = bool(expected_text.get("decisive", False))
        recent_thought = dict(recent_thought_readback or {})
        readback_available = bool(recent_thought.get("available", False))
        readback_strength = _clamp(float(recent_thought.get("strength", 0.0) or 0.0), 0.0, 1.0)
        readback_drift = _clamp(float(recent_thought.get("drift_score", 0.0) or 0.0), 0.0, 1.0)
        readback_branch_end = _clamp(float(recent_thought.get("branch_end_score", 0.0) or 0.0), 0.0, 1.0)
        short_term_readback = dict(short_term_memory_readback or {})
        short_term_available = bool(short_term_readback.get("available", False))
        short_term_strength = _clamp(
            max([float(event.get("score", 0.0) or 0.0) for event in list(short_term_readback.get("selected_events", []) or []) if isinstance(event, dict)] or [0.0]) / 3.0,
            0.0,
            1.0,
        )
        short_term_candidate_pressure = _clamp(float(short_term_readback.get("candidate_count", 0) or 0) / 6.0, 0.0, 1.0)

        # Extract emotion modulation (reward gain affects utility calculation)
        action_mod = (emotion_modulation or {}).get("action", {})
        reward_gain_multiplier = float(action_mod.get("reward_gain_multiplier", 1.0))
        exploration_bias = float(action_mod.get("exploration_bias", 0.0) or 0.0)
        consequence_estimates = dict((action_consequence_trace or {}).get("action_estimates", {}) or {})
        commit_outcome_estimate = self._outcome_memory.estimate("action::text_commit")
        draft_goal_alignment = self._draft_goal_alignment(
            state_snapshot_items=state_snapshot_items,
            draft_context=draft_context,
            fast_cn=fast_cn,
            slow_cn=slow_cn,
            consequence_estimates=consequence_estimates,
            outcome_estimate=commit_outcome_estimate,
        )
        draft_satisfaction_field = self._draft_satisfaction_field(
            draft_eval=draft_eval,
            draft_goal_alignment=draft_goal_alignment,
            correctness=correctness,
            grasp=grasp,
            pressure=pressure,
            dissonance=dissonance,
            uncertainty=uncertainty,
            pressure_anchor_level=pressure_anchor_level,
            expectation_gap=expectation_gap,
        )
        visual_target = self._visual_gaze_target_context(
            state_snapshot_items=state_snapshot_items,
            attention_trace=attention_trace or {},
            draft_context=draft_context,
        )

        candidates = []
        replay_base_drive = (
            0.18
            + time_conf * 0.26
            + expectation * 0.14
            + expectation_gap * 0.12
            + correction_pressure * 0.42
            + max(0.0, 1.0 - correctness) * 0.18
            + dissonance * 0.12
        )
        if correction_pressure > 0.0:
            # Output-side mismatch is not generic prediction stabilization. It is
            # specifically a reread/revise situation, so replay gets an explicit
            # local priority instead of being crowded out by high predicted_mass.
            replay_base_drive += 0.34 + correction_pressure * 0.22
        focus_targets = self._attention_anchor_target_labels(state_snapshot_items, limit=6)
        release_targets = self._attention_release_target_labels(
            state_snapshot_items=state_snapshot_items,
            attention_trace=attention_trace or {},
            limit=6,
        )
        learned_attention_process = any(
            float((dict((action_consequence_trace or {}).get("action_estimates", {}) or {}).get(action_id, {}) or {}).get("support", 0.0) or 0.0) >= 0.25
            for action_id in (
                "action::focus_anchor",
                "action::continue_focus",
                "action::release_focus",
                "action::diverge_attention",
            )
        )
        low_grasp_need = max(uncertainty, evidence_gap, max(0.0, dissonance - coherence * 0.45))
        release_pressure = min(1.0, len(release_targets) / 6.0) if release_targets else 0.0
        release_need = max(
            boredom,
            fulfillment,
            release_pressure,
            max(0.0, pressure - correctness),
        )
        continue_need = max(expectation, grasp, rhythm_expect, satisfaction_level, fulfillment, task_available)
        attention_process_need = self._attention_action_need_scores(
            state_snapshot_items=state_snapshot_items,
            focus_targets=focus_targets,
            release_targets=release_targets,
            surprise=surprise,
            pressure=pressure,
            expectation=expectation,
            grasp=grasp,
            uncertainty=uncertainty,
            evidence_gap=evidence_gap,
            dissonance=dissonance,
            coherence=coherence,
            correctness=correctness,
            fulfillment=fulfillment,
            task_available=task_available,
            rhythm_expect=rhythm_expect,
            learned_attention_process=learned_attention_process,
        )
        focus_process_need = float(attention_process_need.get("action::focus_anchor", 0.0) or 0.0)
        continue_process_need = float(attention_process_need.get("action::continue_focus", 0.0) or 0.0)
        release_process_need = float(attention_process_need.get("action::release_focus", 0.0) or 0.0)
        diverge_process_need = float(attention_process_need.get("action::diverge_attention", 0.0) or 0.0)
        if focus_targets and (surprise >= 0.35 or pressure >= 0.42 or learned_attention_process):
            candidates.append(
                self._candidate(
                    action_id="action::focus_anchor",
                    actuator_id=action_actuator_id("action::focus_anchor", "actuator::attention_allocation"),
                    base_drive=0.24
                    + surprise * 0.34
                    + pressure * 0.10
                    + max(0.0, 0.55 - grasp) * 0.08
                    + (0.12 if learned_attention_process else 0.0)
                    + focus_process_need * 0.10
                    - max(0.0, max(continue_process_need, release_process_need, diverge_process_need) - focus_process_need) * 0.08,
                    predicted={
                        "reward": (0.16 + surprise * 0.08 + min(0.14, pressure * 0.08)) * reward_gain_multiplier,
                        "punishment": max(0.025, 0.07 - surprise * 0.02),
                        "expectation": expectation * 0.34 + surprise * 0.16,
                        "pressure": max(0.0, pressure * 0.36 - surprise * 0.04),
                        "correctness": correctness * 0.26 + grasp * 0.08 + 0.08,
                        "confidence": 0.30 + surprise * 0.18 + min(0.18, len(focus_targets) * 0.03),
                    },
                    notes=[
                        "current_process_attention_anchor",
                        "focus_action_from_surprise_or_learned_attention_process",
                        f"surprise={_round4(surprise)}",
                        f"focus_target_count={len(focus_targets)}",
                        f"learned_attention_process={learned_attention_process}",
                        f"attention_process_need={_round4(focus_process_need)}",
                    ],
                    consequence_estimates=consequence_estimates,
                    params={"target_labels": focus_targets, "target_family": self._dominant_label_family(focus_targets)},
                )
            )
        candidates.append(
            self._candidate(
                action_id="action::continue_focus",
                actuator_id=action_actuator_id("action::continue_focus", "actuator::attention_allocation"),
                base_drive=0.18
                + expectation * 0.32
                + grasp * 0.24
                + rhythm_expect * 0.18
                + satisfaction_level * 0.08
                + fulfillment * 0.10
                + max(0.0, task_available - 0.42) * 0.08
                + (0.20 if continue_need >= max(low_grasp_need, release_need, surprise * 0.7) else 0.0)
                + continue_process_need * 0.14
                - max(0.0, max(release_process_need, diverge_process_need) - continue_process_need) * 0.10
                - boredom * 0.06,
                predicted={
                    "reward": (0.22 + grasp * 0.28) * reward_gain_multiplier,
                    "punishment": max(0.0, 0.12 - grasp * 0.08),
                    "expectation": expectation * 0.85,
                    "pressure": max(0.0, pressure * 0.65 - grasp * 0.18),
                    "correctness": correctness * 0.72 + grasp * 0.18,
                    "confidence": 0.35 + grasp * self.confidence_gain + rhythm_expect * 0.12,
                },
                notes=[
                    "slow_continuation",
                    "focus_development",
                    f"expectation_level={_round4(expectation_level)}",
                    f"satisfaction_level={_round4(satisfaction_level)}",
                    f"fulfillment={_round4(fulfillment)}",
                    f"task_available={_round4(task_available)}",
                    f"attention_process_need={_round4(continue_process_need)}",
                ],
                consequence_estimates=consequence_estimates,
                params={"source_focus_labels": focus_targets, "target_labels": focus_targets},
            )
        )
        if release_targets and (
            boredom >= 0.16
            or fulfillment >= 0.32
            or any("old_" in label or "residue" in label for label in release_targets)
            or learned_attention_process
        ):
            candidates.append(
                self._candidate(
                    action_id="action::release_focus",
                    actuator_id=action_actuator_id("action::release_focus", "actuator::attention_allocation"),
                    base_drive=0.12
                    + boredom * 0.18
                    + fulfillment * 0.16
                    + max(0.0, pressure - correctness) * 0.08
                    + (0.16 if release_targets else 0.0)
                    + (0.10 if learned_attention_process else 0.0)
                    + release_process_need * 0.18,
                    predicted={
                        "reward": (0.10 + boredom * 0.08 + fulfillment * 0.08) * reward_gain_multiplier,
                        "punishment": max(0.025, 0.08 - fulfillment * 0.03),
                        "expectation": expectation * 0.22,
                        "pressure": max(0.0, pressure * 0.32 - fulfillment * 0.06),
                        "correctness": correctness * 0.22 + min(0.16, fulfillment * 0.10),
                        "confidence": 0.26 + boredom * 0.12 + fulfillment * 0.12,
                    },
                    notes=[
                        "current_process_release_old_focus",
                        "release_focus_from_repetition_completion_or_residue",
                        f"release_target_count={len(release_targets)}",
                        f"boredom={_round4(boredom)}",
                        f"fulfillment={_round4(fulfillment)}",
                        f"attention_process_need={_round4(release_process_need)}",
                    ],
                    consequence_estimates=consequence_estimates,
                    params={"suppress_labels": release_targets, "target_labels": focus_targets, "release_reason": "current_process_residue_or_completion"},
                )
            )
        if uncertainty >= 0.42 or evidence_gap >= 0.20 or (dissonance >= 0.48 and coherence < 0.78):
            candidates.append(
                self._candidate(
                    action_id="action::diverge_attention",
                    actuator_id=action_actuator_id("action::diverge_attention", "actuator::attention_allocation"),
                    base_drive=0.18
                    + uncertainty * 0.26
                    + evidence_gap * 0.24
                    + max(0.0, dissonance - coherence * 0.45) * 0.12
                    + (0.24 if low_grasp_need >= max(release_need, continue_need, surprise * 0.65) else 0.0)
                    + (0.08 if learned_attention_process else 0.0)
                    + diverge_process_need * 0.20,
                    predicted={
                        "reward": (0.10 + uncertainty * 0.07 + evidence_gap * 0.08) * reward_gain_multiplier,
                        "punishment": max(0.03, 0.09 - uncertainty * 0.02),
                        "expectation": expectation * 0.20,
                        "pressure": max(0.0, pressure * 0.42 - evidence_gap * 0.05),
                        "correctness": correctness * 0.20 + min(0.12, evidence_gap * 0.12),
                        "confidence": 0.24 + uncertainty * 0.12 + evidence_gap * 0.12,
                    },
                    notes=[
                        "current_process_attention_divergence",
                        "diverge_from_low_grasp_or_evidence_gap",
                        f"uncertainty={_round4(uncertainty)}",
                        f"evidence_gap={_round4(evidence_gap)}",
                        f"dissonance={_round4(dissonance)}",
                        f"attention_process_need={_round4(diverge_process_need)}",
                    ],
                    consequence_estimates=consequence_estimates,
                    params={"target_labels": focus_targets, "top_n_scale": 1.22, "reason": "low_grasp_or_evidence_gap"},
                )
            )
        residual_pair_pressure = min(1.0, mismatch_ratio * 0.35 + max(0.0, dissonance - surprise * 0.35) * 0.45)
        candidates.append(
            self._candidate(
                action_id="action::inspect_residual",
                actuator_id=action_actuator_id("action::inspect_residual", "actuator::attention_allocation"),
                base_drive=0.10
                + residual_pair_pressure * 0.44
                + pressure * 0.12
                + residual_drive * 0.12
                + pressure_level * 0.10
                + min(0.10, residual_count * 0.012)
                + max(0.0, predicted_mass - 1) * 0.01,
                predicted={
                    "reward": (0.14 + dissonance * 0.16 + residual_drive * 0.14 + mismatch_ratio * 0.08) * reward_gain_multiplier,
                    "punishment": max(0.0, 0.08 - dissonance * 0.03 + max(0.0, alignment_score - 0.3) * 0.02),
                    "expectation": expectation * 0.42,
                    "pressure": pressure * 0.88 + expectation_gap * 0.08,
                    "correctness": correctness * 0.52 + dissonance * 0.08,
                    "confidence": 0.28 + dissonance * self.confidence_gain,
                },
                notes=[
                    "residual_probe",
                    "mismatch_resolution",
                    f"residual_mass={_round4(residual_mass)}",
                    f"mismatch_ratio={_round4(mismatch_ratio)}",
                    f"pressure_level={_round4(pressure_level)}",
                ],
                consequence_estimates=consequence_estimates,
            )
        )
        if readback_available or short_term_available:
            no_clear_successor = bool(int(expected_text.get("candidate_count", 0) or 0) <= 0 or not successor_decisive)
            readback_drive = (
                0.12
                + readback_strength * 0.30
                + readback_drift * 0.28
                + readback_branch_end * 0.24
                + short_term_strength * 0.22
                + short_term_candidate_pressure * 0.10
                + boredom * 0.14
                + unfinished_strength * 0.20
                + uncertainty * 0.14
                + ambiguity_pause * 0.18
                + (0.12 if no_clear_successor else 0.0)
                + max(0.0, 0.42 - continuation_readiness) * 0.12
                - pressure * 0.06
            )
            candidates.append(
                self._candidate(
                    action_id="action::recall_recent_context",
                    actuator_id=action_actuator_id("action::recall_recent_context", "actuator::memory_recall"),
                    base_drive=max(0.0, readback_drive),
                    predicted={
                        "reward": (0.08 + readback_strength * 0.10 + short_term_strength * 0.06 + readback_drift * 0.04 + readback_branch_end * 0.05) * reward_gain_multiplier,
                        "punishment": max(0.018, 0.05 - readback_strength * 0.015),
                        "expectation": max(expectation * 0.20, readback_strength * 0.20, short_term_strength * 0.18),
                        "pressure": max(0.0, pressure * 0.24 + ambiguity_pause * 0.04),
                        "correctness": correctness * 0.18 + readback_strength * 0.18 + short_term_strength * 0.10,
                        "confidence": 0.22 + readback_strength * 0.20 + short_term_strength * 0.12 + max(readback_drift, readback_branch_end) * 0.12,
                    },
                    notes=[
                        "recent_thought_readback",
                        "short_term_memory_self_observation",
                        "multimodal_short_term_memory" if short_term_available else "focus_only_short_term_memory",
                        "branch_end_or_drift" if (readback_drift >= 0.18 or readback_branch_end >= 0.34 or no_clear_successor) else "ordinary_recent_context_readback",
                        "unfinished_thought_recovery_bias" if unfinished_strength >= 0.12 else "ordinary_no_param_recall_bias",
                        "boredom_drives_self_probe" if boredom >= 0.12 else "task_feeling_low",
                        f"readback_strength={_round4(readback_strength)}",
                        f"short_term_strength={_round4(short_term_strength)}",
                        f"unfinished_strength={_round4(unfinished_strength)}",
                        f"boredom={_round4(boredom)}",
                        f"drift_score={_round4(readback_drift)}",
                        f"branch_end_score={_round4(readback_branch_end)}",
                        f"successor_decisive={successor_decisive}",
                    ],
                    consequence_estimates=consequence_estimates,
                    params={
                        "horizon": int(recent_thought.get("horizon", 6) or 6),
                        "reason": "recent_thought_readback",
                        "recall_mode": "unfinished_soft_recovery" if unfinished_strength >= 0.12 else ("no_param_recent_context" if no_clear_successor else "cued_recent_context"),
                        "labels": list(recent_thought.get("labels", []) or [])[:8],
                        "active_episode_id": int(recent_thought.get("active_episode_id", -1) or -1),
                        "short_term_event_ids": [
                            str(event.get("event_id", "") or "")
                            for event in list(short_term_readback.get("selected_events", []) or [])[:4]
                            if isinstance(event, dict)
                        ],
                    },
                )
            )
        active_text_goal = bool(
            bool(draft_goal_alignment.get("current_turn_active", False))
            or int(draft_goal_alignment.get("task_anchor_count", 0) or 0) > 0
            or int(draft_goal_alignment.get("dialogue_anchor_count", 0) or 0) > 0
            or float(draft_goal_alignment.get("dialogue_closure_need", 0.0) or 0.0) > 0.0
        )
        if expected_token and correction_pressure <= 0.0:
            last_visible_token = str(draft_context.get("last_visible_token", "") or "")
            last_insert_tick = int(draft_context.get("last_insert_tick", -1) or -1)
            last_reread_tick = int(draft_context.get("last_reread_tick", -1) or -1)
            last_commit_tick = int(draft_context.get("last_commit_tick", -1) or -1)
            last_commit_age = int(draft_context.get("last_commit_age", 9999) or 9999)
            visible_length = int(draft_context.get("visible_length", 0) or 0)
            insert_count = int(draft_context.get("insert_count", 0) or 0)
            has_internal_draft = bool(draft_context.get("has_internal_draft", False))
            recently_closed_empty_surface = bool(
                visible_length <= 0
                and last_commit_tick >= 0
                and last_commit_age <= 64
                and not active_text_goal
            )
            same_token_waiting_for_reread = bool(
                last_visible_token
                and expected_token == last_visible_token
                and last_insert_tick >= 0
                and last_insert_tick > max(last_reread_tick, last_commit_tick)
            )
            if not same_token_waiting_for_reread and not recently_closed_empty_surface:
                continuation_shift = dict(expected_text.get("continuation_shift", {}) or {})
                continuation_shift_bonus = 0.0
                if continuation_shift and has_internal_draft:
                    continuation_shift_bonus = 0.34 + min(0.18, visible_length * 0.04)
                continuation_bonus = (0.14 if has_internal_draft else 0.0) + continuation_shift_bonus
                early_draft_bonus = 0.08 if 0 < visible_length < 5 else 0.0
                new_draft_bonus = 0.06 if insert_count <= 0 else 0.0
                ambiguous_successor_cost = 0.0
                if not successor_decisive:
                    ambiguous_successor_cost = 0.20 + uncertainty * 0.18 + ambiguity_pause * 0.22 + max(0.0, 0.46 - grasp) * 0.12
                    if continuation_shift:
                        ambiguous_successor_cost *= 0.45
                expected_drive = (
                    0.36
                    + expected_strength * 0.62
                    + continuation_readiness * 0.28
                    + expectation * 0.16
                    + grasp * 0.10
                    + correctness * 0.08
                    + continuation_bonus
                    + early_draft_bonus
                    + new_draft_bonus
                    - pressure * 0.28
                    - uncertainty * 0.10
                    - ambiguity_pause * 0.30
                    - cleanup_pressure * 0.36
                    - ambiguous_successor_cost
                )
                if expected_drive > 0.02:
                    # Text insert is a one-token draft action. The candidate is
                    # intentionally local and parameterized by live Cn/Cn'
                    # evidence. If the successor is so ambiguous that its
                    # base write impulse collapses, no write candidate is
                    # emitted; wait/recall/resampling can still win instead.
                    candidates.append(
                        self._candidate(
                            action_id="action::text_insert",
                            actuator_id=action_actuator_id("action::text_insert", "actuator::text_editor"),
                            base_drive=max(0.0, expected_drive),
                            predicted={
                                "reward": (0.15 + expected_strength * 0.14 + continuation_readiness * 0.08 + continuation_bonus * 0.30) * reward_gain_multiplier,
                                "punishment": max(0.025, 0.08 + pressure * 0.04 + ambiguity_pause * 0.03 - expected_strength * 0.03),
                                "expectation": max(expectation * 0.56, expected_strength * 0.62 + continuation_readiness * 0.18),
                                "pressure": max(0.0, pressure * 0.42 + uncertainty * 0.04 + ambiguity_pause * 0.08 + cleanup_pressure * 0.08),
                                "correctness": correctness * 0.30 + expected_strength * 0.25 + continuation_readiness * 0.12 + grasp * 0.08,
                                "confidence": 0.34 + expected_strength * 0.28 + continuation_readiness * 0.18 + grasp * 0.08 - ambiguity_pause * 0.08,
                            },
                            notes=[
                                "draft_expected_token_write",
                                "one_token_internal_draft_action",
                                "successor_decisive" if successor_decisive else "successor_ambiguous",
                                f"expected_token={expected_token}",
                                f"expected_strength={_round4(expected_strength)}",
                                f"top_share={_round4(float(expected_text.get('top_share', 0.0) or 0.0))}",
                                f"dominance_gap={_round4(float(expected_text.get('dominance_gap', 0.0) or 0.0))}",
                                f"ambiguity_pause={_round4(ambiguity_pause)}",
                                f"ambiguous_successor_cost={_round4(ambiguous_successor_cost)}",
                                f"continuation_shift_bonus={_round4(continuation_shift_bonus)}",
                                f"visible_length={visible_length}",
                            ],
                            consequence_estimates=consequence_estimates,
                            params={"token": expected_token, "reason": "expected_token_draft_write"},
                        )
                    )
        if bool(draft_context.get("has_internal_draft", False)) and correction_pressure <= 0.0:
            visible_length = int(draft_context.get("visible_length", 0) or 0)
            last_event_type = str(draft_context.get("last_event_type", "") or "")
            last_insert_age = int(draft_context.get("last_insert_age", 9999) or 9999)
            last_reread_age = int(draft_context.get("last_reread_age", 9999) or 9999)
            last_delete_age = int(draft_context.get("last_delete_age", 9999) or 9999)
            last_replace_age = int(draft_context.get("last_replace_age", 9999) or 9999)
            last_mutation_tick = int(draft_context.get("last_mutation_tick", -1) or -1)
            last_commit_tick = int(draft_context.get("last_commit_tick", -1) or -1)
            last_commit_age = int(draft_context.get("last_commit_age", 9999) or 9999)
            reread_count = int(draft_context.get("reread_count", 0) or 0)
            just_inserted = last_insert_age <= 2 and last_event_type in {"insert", "replace"}
            review_due = just_inserted or (visible_length >= 2 and last_reread_age > 2)
            if review_due or ambiguity_pause >= 0.32 or cleanup_pressure >= 0.28:
                repeat_reread_penalty = 0.22 if (last_event_type == "reread" and last_reread_age <= 1) else 0.0
                review_drive = (
                    0.44
                    + (0.46 if just_inserted else 0.12)
                    + min(0.18, visible_length * 0.04)
                    + uncertainty * 0.10
                    + ambiguity_pause * 0.28
                    + cleanup_pressure * 0.20
                    + max(0.0, 0.5 - correctness) * 0.08
                    - pressure * 0.10
                    - repeat_reread_penalty
                )
                candidates.append(
                    self._candidate(
                        action_id="action::text_reread",
                        actuator_id=action_actuator_id("action::text_reread", "actuator::text_editor"),
                        base_drive=max(0.0, review_drive),
                        predicted={
                            "reward": (0.12 + min(0.18, visible_length * 0.03) + (0.06 if just_inserted else 0.0)) * reward_gain_multiplier,
                            "punishment": max(0.015, 0.05 - min(0.02, reread_count * 0.004)),
                            "expectation": expectation * 0.34 + expected_strength * 0.18,
                            "pressure": max(0.0, pressure * 0.28 - 0.04),
                            "correctness": correctness * 0.34 + grasp * 0.10 + 0.10,
                            "confidence": 0.38 + grasp * 0.12 + min(0.20, visible_length * 0.04),
                        },
                        notes=[
                            "draft_reread_for_review",
                            "humanlike_pause_after_writing",
                            "successor_distribution_review" if ambiguity_pause >= 0.32 else "draft_surface_review",
                            f"last_event_type={last_event_type}",
                            f"last_insert_age={last_insert_age}",
                            f"ambiguity_pause={_round4(ambiguity_pause)}",
                            f"cleanup_pressure={_round4(cleanup_pressure)}",
                            f"visible_length={visible_length}",
                            f"repeat_reread_penalty={_round4(repeat_reread_penalty)}",
                        ],
                        consequence_estimates=consequence_estimates,
                        params={"reason": "draft_review"},
                    )
                )
            trailing_repeat_count = int(draft_context.get("trailing_repeat_count", 0) or 0)
            trailing_repeat_token = str(draft_context.get("trailing_repeat_token", "") or "")
            duplicate_ratio = _clamp(float(draft_context.get("duplicate_ratio", 0.0) or 0.0), 0.0, 1.0)
            visible_tokens_for_tail = [
                str(token or "")
                for token in list(draft_context.get("visible_tokens", []) or [])
                if str(token or "")
            ]
            tail_token_seen_before = bool(
                visible_length >= 3
                and visible_tokens_for_tail
                and visible_tokens_for_tail[-1] in set(visible_tokens_for_tail[:-1])
            )
            pure_consecutive_repeat = bool(
                visible_tokens_for_tail
                and len(visible_tokens_for_tail) == visible_length
                and len(set(visible_tokens_for_tail)) == 1
            )
            mismatch_count = int(draft_context.get("mismatch_count", 0) or 0)
            expressive_repeat_protected = bool(
                pure_consecutive_repeat
                and mismatch_count <= 0
                and dissonance < 0.82
            )
            if trailing_repeat_count >= 2 and cleanup_pressure >= 0.24 and last_delete_age > 1 and not expressive_repeat_protected:
                delete_start = max(0, visible_length - 1)
                delete_drive = (
                    0.34
                    + cleanup_pressure * 0.62
                    + (0.10 if last_reread_age <= 3 else 0.0)
                    + dissonance * 0.08
                    - pressure * 0.10
                    - (0.18 if last_delete_age <= 3 else 0.0)
                )
                candidates.append(
                    self._candidate(
                        action_id="action::text_delete",
                        actuator_id=action_actuator_id("action::text_delete", "actuator::text_editor"),
                        base_drive=max(0.0, delete_drive),
                        predicted={
                            "reward": (0.11 + cleanup_pressure * 0.20 + (0.04 if last_reread_age <= 3 else 0.0)) * reward_gain_multiplier,
                            "punishment": max(0.03, pressure * 0.05),
                            "expectation": expectation * 0.20,
                            "pressure": max(0.0, pressure * 0.36 - cleanup_pressure * 0.06),
                            "correctness": correctness * 0.24 + cleanup_pressure * 0.24,
                            "confidence": 0.34 + cleanup_pressure * 0.28,
                        },
                        notes=[
                            "draft_tail_repetition_cleanup",
                            "local_delete_not_sentence_reset",
                            f"trailing_repeat_token={trailing_repeat_token}",
                            f"trailing_repeat_count={trailing_repeat_count}",
                            f"cleanup_pressure={_round4(cleanup_pressure)}",
                        ],
                        consequence_estimates=consequence_estimates,
                        params={"span": [delete_start, visible_length], "reason": "tail_repetition_cleanup"},
                    )
                )
            elif tail_token_seen_before and cleanup_pressure >= 0.16 and last_delete_age > 1 and last_reread_age <= 4 and not expressive_repeat_protected:
                delete_start = max(0, visible_length - 1)
                delete_drive = (
                    0.30
                    + cleanup_pressure * 0.54
                    + max(0.0, duplicate_ratio - 0.18) * 0.36
                    + dissonance * 0.10
                    + (0.08 if last_reread_age <= 3 else 0.0)
                    - pressure * 0.10
                    - (0.18 if last_delete_age <= 3 else 0.0)
                )
                candidates.append(
                    self._candidate(
                        action_id="action::text_delete",
                        actuator_id=action_actuator_id("action::text_delete", "actuator::text_editor"),
                        base_drive=max(0.0, delete_drive),
                        predicted={
                            "reward": (0.10 + cleanup_pressure * 0.18 + duplicate_ratio * 0.08) * reward_gain_multiplier,
                            "punishment": max(0.028, pressure * 0.05),
                            "expectation": expectation * 0.18,
                            "pressure": max(0.0, pressure * 0.34 - cleanup_pressure * 0.05),
                            "correctness": correctness * 0.22 + cleanup_pressure * 0.22 + duplicate_ratio * 0.08,
                            "confidence": 0.32 + cleanup_pressure * 0.24 + duplicate_ratio * 0.10,
                        },
                        notes=[
                            "draft_tail_reentry_cleanup",
                            "local_delete_not_sentence_reset",
                            "tail_token_seen_earlier_in_visible_draft",
                            f"tail_token={visible_tokens_for_tail[-1] if visible_tokens_for_tail else ''}",
                            f"duplicate_ratio={_round4(duplicate_ratio)}",
                            f"cleanup_pressure={_round4(cleanup_pressure)}",
                            f"last_reread_age={last_reread_age}",
                        ],
                        consequence_estimates=consequence_estimates,
                        params={"span": [delete_start, visible_length], "reason": "tail_reentry_cleanup"},
                    )
                )
            last_visible_token = str(draft_context.get("last_visible_token", "") or "")
            expected_position_notes = {
                str(note or "")
                for alt in list(expected_text.get("alternatives", []) or [])
                if isinstance(alt, dict) and str(alt.get("token", "") or "") == expected_token
                for note in list(alt.get("position_notes", []) or [])
            }
            expected_is_cursor_continuation = bool(
                "cursor_aligned_next_unread_region" in expected_position_notes
                or "cursor_aligned_next_unread_region" in str(expected_text.get("source", "") or "")
                or bool(dict(expected_text.get("continuation_shift", {}) or {}).get("cursor_aligned_shift", False))
            )
            if (
                expected_token
                and last_visible_token
                and expected_token != last_visible_token
                and not expected_is_cursor_continuation
                and last_reread_age <= 3
                and last_replace_age > 2
                and cleanup_pressure < 0.55
                and ambiguity_pause <= 0.42
                and successor_decisive
            ):
                replace_drive = (
                    0.30
                    + continuation_readiness * 0.24
                    + expected_strength * 0.20
                    + correctness * 0.08
                    - pressure * 0.12
                )
                replace_params = {"span": [max(0, visible_length - 1), visible_length], "new_text": expected_token, "expected_token": expected_token, "from_token": last_visible_token, "reason": "local_successor_revision"}
                replace_parameter_estimate = self._parameter_memory.estimate(action_id="action::text_replace", proposed_params=replace_params)
                replace_drive += float(replace_parameter_estimate.get("drive_bias", 0.0) or 0.0)
                candidates.append(
                    self._candidate(
                        action_id="action::text_replace",
                        actuator_id=action_actuator_id("action::text_replace", "actuator::text_editor"),
                        base_drive=max(0.0, replace_drive),
                        predicted={
                            "reward": (0.10 + continuation_readiness * 0.10 + expected_strength * 0.06) * reward_gain_multiplier,
                            "punishment": max(0.035, pressure * 0.05 + ambiguity_pause * 0.03),
                            "expectation": expectation * 0.22 + expected_strength * 0.16,
                            "pressure": max(0.0, pressure * 0.38 + ambiguity_pause * 0.04),
                            "correctness": correctness * 0.26 + continuation_readiness * 0.18,
                            "confidence": 0.34 + continuation_readiness * 0.18 + expected_strength * 0.10,
                        },
                        notes=[
                            "draft_local_replace_after_reread",
                            "successor_decisive_local_revision",
                            f"from_token={last_visible_token}",
                            f"to_token={expected_token}",
                            f"last_reread_age={last_reread_age}",
                        ]
                        + (
                            [
                                "parameter_memory_bias",
                                f"parameter_drive_bias={_round4(float(replace_parameter_estimate.get('drive_bias', 0.0) or 0.0))}",
                                f"parameter_similarity={_round4(float(replace_parameter_estimate.get('similarity', 0.0) or 0.0))}",
                            ]
                            if float(replace_parameter_estimate.get("support", 0.0) or 0.0) > 0.0
                            else []
                        ),
                        consequence_estimates=consequence_estimates,
                        params=replace_params,
                    )
                )
            dialogue_closure_need_for_commit = _clamp(
                float(draft_goal_alignment.get("dialogue_closure_need", 0.0) or 0.0),
                0.0,
                1.0,
            )
            stale_reviewed_draft_ready = bool(
                dialogue_closure_need_for_commit > 0.0
                and reread_count > 0
                and last_insert_age >= 3
                and last_delete_age > 2
                and last_replace_age > 2
                and cleanup_pressure < 0.42
                and ambiguity_pause < 0.62
            )
            stable_after_reread = (
                last_reread_age <= 3
                or (last_insert_age >= 3 and reread_count > 0)
                or stale_reviewed_draft_ready
            )
            if visible_length > 0 and stable_after_reread:
                pending_revision_pressure = 0.0
                if revision_opportunities:
                    pending_revision_pressure = _clamp(
                        max(float(row.get("support", 0.0) or 0.0) for row in revision_opportunities)
                        * (0.72 if last_reread_age <= 4 else 0.42),
                        0.0,
                        1.0,
                    )
                field_satisfaction = _clamp(float(draft_satisfaction_field.get("satisfaction", 0.0) or 0.0), 0.0, 1.0)
                closure_pressure = _clamp(float(draft_satisfaction_field.get("closure_pressure", 0.0) or 0.0), 0.0, 1.0)
                goal_alignment = _clamp(float(draft_satisfaction_field.get("goal_alignment", 0.0) or 0.0), 0.0, 1.0)
                continuation_pressure = _clamp(float(draft_satisfaction_field.get("continuation_pressure", 0.0) or 0.0), 0.0, 1.0)
                revision_pressure = _clamp(max(float(draft_satisfaction_field.get("revision_pressure", 0.0) or 0.0), pending_revision_pressure), 0.0, 1.0)
                habitual_commit_pressure = _clamp(float(draft_satisfaction_field.get("habitual_commit_pressure", 0.0) or 0.0), 0.0, 1.0)
                outcome_commit_pressure = _clamp(float(draft_satisfaction_field.get("outcome_commit_pressure", 0.0) or 0.0), 0.0, 1.0)
                risk_commit_pressure = _clamp(max(float(draft_satisfaction_field.get("risk_commit_pressure", 0.0) or 0.0), pending_revision_pressure * 0.82), 0.0, 1.0)
                unexpressed_successor_pressure = _clamp(
                    float(draft_goal_alignment.get("unexpressed_successor_pressure", 0.0) or 0.0),
                    0.0,
                    1.0,
                )
                same_draft_already_committed = bool(last_commit_tick >= 0 and last_commit_tick >= last_mutation_tick)
                recent_commit_fatigue = 0.0
                same_draft_commit_fatigue = 0.0
                commit_params = {
                    "target_channel": "draft",
                    "reason": "draft_satisfaction_field_commit_ready",
                    "satisfaction_field": dict(draft_satisfaction_field),
                    "goal_alignment": dict(draft_goal_alignment),
                    "pending_revision_pressure": _round4(pending_revision_pressure),
                    "draft_signature": str(draft_context.get("visible_text", "") or ""),
                    "task_context_signature": self._draft_task_context_signature(draft_goal_alignment),
                }
                commit_drive = (
                    0.28
                    + min(0.22, visible_length * 0.05)
                    + draft_satisfaction * 0.24
                    + field_satisfaction * 0.22
                    + closure_pressure * 0.18
                    + goal_alignment * 0.12
                    + habitual_commit_pressure * 0.24
                    + outcome_commit_pressure * 0.16
                    + correctness * 0.18
                    + grasp * 0.18
                    + (0.16 if last_reread_age <= 2 else 0.0)
                    + (0.14 * dialogue_closure_need_for_commit if stale_reviewed_draft_ready else 0.0)
                    - max(0.0, expected_strength - 0.35) * 0.24
                    - continuation_pressure * 0.62
                    - unexpressed_successor_pressure * 0.74
                    - revision_pressure * 0.24
                    - risk_commit_pressure * 0.58
                    - ambiguity_pause * 0.22
                    - cleanup_pressure * 0.34
                    - pressure * 0.12
                    - dissonance * 0.10
                    - uncertainty * 0.08
                    - unfinished_strength * 0.30
                    - recent_commit_fatigue
                    - same_draft_commit_fatigue
                    - pending_revision_pressure * 0.68
                )
                candidates.append(
                    self._candidate(
                        action_id="action::text_commit",
                        actuator_id=action_actuator_id("action::text_commit", "actuator::text_editor"),
                        base_drive=max(0.0, commit_drive),
                        predicted={
                            "reward": (
                                0.12
                                + draft_satisfaction * 0.08
                                + field_satisfaction * 0.08
                                + closure_pressure * 0.06
                                + goal_alignment * 0.05
                                + outcome_commit_pressure * 0.06
                                + correctness * 0.18
                                + grasp * 0.12
                            )
                            * reward_gain_multiplier,
                            "punishment": max(
                                0.06,
                                pressure * 0.12
                                + uncertainty * 0.04
                                + cleanup_pressure * 0.04
                                + revision_pressure * 0.05
                                + risk_commit_pressure * 0.16,
                            ),
                            "expectation": expectation * 0.24,
                            "pressure": max(
                                0.0,
                                pressure * 0.48
                                + ambiguity_pause * 0.06
                                + cleanup_pressure * 0.08
                                + continuation_pressure * 0.05
                                + risk_commit_pressure * 0.22
                                + pending_revision_pressure * 0.18
                                + unexpressed_successor_pressure * 0.20
                                + unfinished_strength * 0.10
                                + 0.04,
                            ),
                            "correctness": correctness * 0.50 + grasp * 0.18 + field_satisfaction * 0.12 + goal_alignment * 0.08 + min(0.14, visible_length * 0.02) - pending_revision_pressure * 0.12 - unexpressed_successor_pressure * 0.16,
                            "confidence": 0.42 + grasp * 0.16 + correctness * 0.14 + closure_pressure * 0.08 + goal_alignment * 0.06 - pending_revision_pressure * 0.08 - unexpressed_successor_pressure * 0.10,
                        },
                        notes=[
                            "draft_commit_ready",
                            "draft_satisfaction_field_commit_ready",
                            "commit_still_external_safety_gate_boundary",
                            "commit_is_internal_draft_closure_not_external_send",
                            f"field_satisfaction={_round4(field_satisfaction)}",
                            f"goal_alignment={_round4(goal_alignment)}",
                            f"closure_pressure={_round4(closure_pressure)}",
                            f"habitual_commit_pressure={_round4(habitual_commit_pressure)}",
                            f"outcome_commit_pressure={_round4(outcome_commit_pressure)}",
                            f"risk_commit_pressure={_round4(risk_commit_pressure)}",
                            f"continuation_pressure={_round4(continuation_pressure)}",
                            f"unexpressed_successor_pressure={_round4(unexpressed_successor_pressure)}",
                            f"revision_pressure={_round4(revision_pressure)}",
                            f"pending_revision_pressure={_round4(pending_revision_pressure)}",
                            "commit_risk_from_text_revision_opportunity" if pending_revision_pressure > 0.0 else "no_pending_revision_opportunity",
                            f"habit_scope={str(draft_goal_alignment.get('habit_scope', 'none') or 'none')}",
                            f"draft_satisfaction={_round4(draft_satisfaction)}",
                            f"ambiguity_pause={_round4(ambiguity_pause)}",
                            f"cleanup_pressure={_round4(cleanup_pressure)}",
                            f"last_reread_age={last_reread_age}",
                            f"stale_reviewed_draft_ready={stale_reviewed_draft_ready}",
                            f"dialogue_closure_need={_round4(dialogue_closure_need_for_commit)}",
                            f"last_commit_tick={last_commit_tick}",
                            f"last_mutation_tick={last_mutation_tick}",
                            f"last_commit_age={last_commit_age}",
                            f"visible_length={visible_length}",
                            f"recent_commit_fatigue={_round4(recent_commit_fatigue)}",
                            f"same_draft_commit_fatigue={_round4(same_draft_commit_fatigue)}",
                            f"unfinished_strength={_round4(unfinished_strength)}",
                            "unfinished_soft_bias_suppresses_premature_commit"
                            if unfinished_strength > 0.0
                            else "no_unfinished_commit_suppression",
                            "same_draft_already_committed_suppression"
                            if same_draft_already_committed
                            else "draft_changed_since_last_commit",
                        ],
                        consequence_estimates=consequence_estimates,
                        params=commit_params,
                    )
                )
        if correction_pressure > 0.0 and latest_expected_token:
            # Output mismatch is a local draft-editing problem. We create text
            # editor candidates from generic mismatch evidence instead of
            # hard-coding any sequence answer; normal action competition still
            # decides whether AP rereads, waits, recalls, or actually revises.
            reread_drive = (
                0.38
                + correction_pressure * 0.54
                + min(0.22, dissonance * 0.16 + mismatch_ratio * 0.12)
                + uncertainty * 0.08
            )
            replace_drive = (
                0.28
                + correction_pressure * 0.44
                + (0.44 if reread_after_mismatch else 0.0)
                + min(0.26, dissonance * 0.12 + mismatch_ratio * 0.10 + max(0.0, 1.0 - correctness) * 0.06)
            )
            mismatch_span_start = max(0, int(draft_context.get("latest_mismatch_index", -1) or -1))
            if mismatch_span_start < 0:
                mismatch_span_start = max(0, visible_length - 1)
            mismatch_token = str(draft_context.get("latest_mismatch_token", "") or latest_mismatch_token)
            output_replace_params = {
                "span": [mismatch_span_start, mismatch_span_start + 1],
                "new_text": latest_expected_token,
                "expected_token": latest_expected_token,
                "candidate_token": latest_expected_token,
                "from_token": mismatch_token,
                "conflict_index": mismatch_span_start,
                "reason": "real_virtual_conflict_revision",
            }
            output_parameter_estimate = self._parameter_memory.estimate(action_id="action::text_replace", proposed_params=output_replace_params)
            replace_drive += float(output_parameter_estimate.get("drive_bias", 0.0) or 0.0)
            if reread_after_mismatch:
                # Once AP has looked back at the doubtful draft, continuing to
                # reread should fatigue relative to revision. This preserves
                # humanlike hesitation while avoiding an endless "just looking"
                # loop when the expected replacement is clear.
                reread_drive -= 0.18
            candidates.append(
                self._candidate(
                    action_id="action::text_reread",
                    actuator_id=action_actuator_id("action::text_reread", "actuator::text_editor"),
                    base_drive=reread_drive,
                    predicted={
                        "reward": (0.14 + correction_pressure * 0.18 + dissonance * 0.04) * reward_gain_multiplier,
                        "punishment": max(0.02, 0.07 - correction_pressure * 0.02),
                        "expectation": expectation * 0.34 + correction_pressure * 0.22,
                        "pressure": max(0.0, pressure * 0.42 - correction_pressure * 0.08),
                        "correctness": correctness * 0.34 + correction_pressure * 0.24,
                        "confidence": 0.42 + correction_pressure * 0.26 + min(0.12, dissonance * 0.06),
                    },
                    notes=[
                        "output_side_revision_pressure",
                        "humanlike_reread_before_revise",
                        f"correction_pressure={_round4(correction_pressure)}",
                        f"latest_expected_token={latest_expected_token}",
                    ],
                    consequence_estimates=consequence_estimates,
                    params={
                        "reason": "write_mismatch",
                        "expected_token": latest_expected_token,
                    },
                )
            )
            candidates.append(
                self._candidate(
                    action_id="action::text_replace",
                    actuator_id=action_actuator_id("action::text_replace", "actuator::text_editor"),
                    base_drive=replace_drive,
                    predicted={
                        "reward": (0.16 + correction_pressure * 0.24 + (0.06 if reread_after_mismatch else 0.0)) * reward_gain_multiplier,
                        "punishment": max(0.035, 0.10 - correction_pressure * 0.025),
                        "expectation": expectation * 0.30 + correction_pressure * 0.26,
                        "pressure": max(0.0, pressure * 0.48 - correction_pressure * 0.05),
                        "correctness": correctness * 0.28 + correction_pressure * 0.36 + (0.08 if reread_after_mismatch else 0.0),
                        "confidence": 0.40 + correction_pressure * 0.24 + (0.14 if reread_after_mismatch else 0.0),
                    },
                    notes=[
                        "output_side_revision_pressure",
                        "revise_unresolved_write_mismatch",
                        f"correction_pressure={_round4(correction_pressure)}",
                        f"reread_after_mismatch={reread_after_mismatch}",
                        f"latest_expected_token={latest_expected_token}",
                    ]
                    + (
                        [
                            "parameter_memory_bias",
                            f"parameter_drive_bias={_round4(float(output_parameter_estimate.get('drive_bias', 0.0) or 0.0))}",
                            f"parameter_similarity={_round4(float(output_parameter_estimate.get('similarity', 0.0) or 0.0))}",
                        ]
                        if float(output_parameter_estimate.get("support", 0.0) or 0.0) > 0.0
                        else []
                    ),
                    consequence_estimates=consequence_estimates,
                    params=output_replace_params,
                )
            )
        if revision_opportunities:
            visible_length = int(draft_context.get("visible_length", 0) or 0)
            last_reread_age = int(draft_context.get("last_reread_age", 9999) or 9999)
            last_reread_tick = int(draft_context.get("last_reread_tick", -1) or -1)
            has_recent_reread = last_reread_age <= 4 and last_reread_tick >= 0
            top_opportunity = revision_opportunities[0]
            top_support = _clamp(float(top_opportunity.get("support", 0.0) or 0.0), 0.0, 1.2)
            top_kind = str(top_opportunity.get("conflict_kind", "") or top_opportunity.get("operation", "") or "")
            reread_drive = (
                0.30
                + top_support * 0.26
                + min(0.20, len(revision_opportunities) * 0.05)
                + dissonance * 0.10
                + uncertainty * 0.08
                - (0.18 if has_recent_reread else 0.0)
            )
            if visible_length > 0 and not has_recent_reread:
                candidates.append(
                    self._candidate(
                        action_id="action::text_reread",
                        actuator_id=action_actuator_id("action::text_reread", "actuator::text_editor"),
                        base_drive=max(0.0, reread_drive),
                        predicted={
                            "reward": (0.10 + top_support * 0.08 + min(0.08, len(revision_opportunities) * 0.02)) * reward_gain_multiplier,
                            "punishment": max(0.018, 0.055 - top_support * 0.012),
                            "expectation": expectation * 0.28 + top_support * 0.16,
                            "pressure": max(0.0, pressure * 0.34 - top_support * 0.03),
                            "correctness": correctness * 0.28 + top_support * 0.16,
                            "confidence": 0.34 + top_support * 0.18,
                        },
                        notes=[
                            "text_revision_opportunity_review",
                            "humanlike_reread_before_multi_edit",
                            f"opportunity_kind={top_kind}",
                            f"opportunity_count={len(revision_opportunities)}",
                            f"top_support={_round4(top_support)}",
                        ],
                        consequence_estimates=consequence_estimates,
                        params={"span": [0, visible_length], "reason": "text_revision_opportunity_review"},
                    )
                )
            if has_recent_reread:
                expected_cursor_aligned = "cursor_aligned_next_unread_region" in {
                    str(note or "")
                    for alt in list(expected_text.get("alternatives", []) or [])
                    if isinstance(alt, dict) and str(alt.get("token", "") or "") == expected_token
                    for note in list(alt.get("position_notes", []) or []) + list(alt.get("sources", []) or [])
                }
                for opportunity in revision_opportunities[:4]:
                    operation = str(opportunity.get("operation", "") or "")
                    candidate_text = str(opportunity.get("candidate_text", "") or "")
                    from_text = str(opportunity.get("from_text", "") or "")
                    support = _clamp(float(opportunity.get("support", 0.0) or 0.0), 0.0, 1.2)
                    span = self._text_span(opportunity.get("span"))
                    cursor = int(opportunity.get("cursor", span[0]) or 0)
                    conflict_kind = str(opportunity.get("conflict_kind", operation) or operation)
                    if operation == "insert" and candidate_text:
                        if cursor > visible_length:
                            continue
                        if conflict_kind == "continue_after_visible_prefix":
                            last_visible_token = str(draft_context.get("last_visible_token", "") or "")
                            if not expected_token or candidate_text != expected_token or candidate_text == last_visible_token or not expected_cursor_aligned:
                                continue
                        visible_tokens = [
                            str(token or "")
                            for token in list(draft_context.get("visible_tokens", []) or [])
                            if str(token or "")
                        ]
                        candidate_already_closed = bool(candidate_text in set(visible_tokens))
                        params = {
                            "cursor": max(0, cursor),
                            "token": candidate_text,
                            "expected_token": candidate_text,
                            "candidate_token": candidate_text,
                            "parameter_kind": "text_insert",
                            "reason": f"text_revision_opportunity::{conflict_kind}",
                        }
                        parameter_estimate = self._parameter_memory.estimate(action_id="action::text_insert", proposed_params=params)
                        repetition_cost = 0.0
                        if candidate_already_closed and conflict_kind == "continue_after_visible_prefix" and visible_length >= 2:
                            repetition_cost = 0.36 + min(0.22, support * 0.18)
                        drive = 0.36 + support * 0.46 + dissonance * 0.10 + float(parameter_estimate.get("drive_bias", 0.0) or 0.0) - repetition_cost
                        candidates.append(
                            self._candidate(
                                action_id="action::text_insert",
                                actuator_id=action_actuator_id("action::text_insert", "actuator::text_editor"),
                                base_drive=max(0.0, drive),
                                predicted={
                                    "reward": (0.12 + support * 0.14) * reward_gain_multiplier,
                                    "punishment": max(0.035, pressure * 0.04 + ambiguity_pause * 0.02),
                                    "expectation": expectation * 0.22 + support * 0.20,
                                    "pressure": max(0.0, pressure * 0.36 - support * 0.04),
                                    "correctness": correctness * 0.24 + support * 0.26,
                                    "confidence": 0.38 + support * 0.24,
                                },
                                notes=self._revision_opportunity_notes(opportunity, parameter_estimate, "insert")
                                + (
                                    [
                                        "visible_prefix_repetition_soft_cost",
                                        f"candidate_already_closed={candidate_text}",
                                        f"repetition_cost={_round4(repetition_cost)}",
                                    ]
                                    if repetition_cost > 0.0
                                    else []
                                ),
                                consequence_estimates=consequence_estimates,
                                params=params,
                            )
                        )
                if expected_token and expected_cursor_aligned:
                    cursor = max(0, min(visible_length, int(draft_context.get("cursor_index", visible_length) or visible_length)))
                    wanted_conflict = "start_empty_draft" if visible_length <= 0 else "continue_after_visible_prefix"
                    continuation_opportunities = [
                        row
                        for row in revision_opportunities
                        if str(row.get("conflict_kind", "") or "") == wanted_conflict
                        or (
                            visible_length > 0
                            and str(row.get("conflict_kind", "") or "") == "continue_after_visible_prefix"
                        )
                    ]
                    if continuation_opportunities and (
                        not str(draft_context.get("last_visible_token", "") or "")
                        or expected_token != str(draft_context.get("last_visible_token", "") or "")
                    ):
                        continuation_support = _clamp(
                            max(float(row.get("support", 0.0) or 0.0) for row in continuation_opportunities)
                            + continuation_readiness * 0.34
                            + expected_strength * 0.22
                            + expected_top_share * 0.12,
                            0.0,
                            1.2,
                        )
                        params = {
                            "cursor": cursor,
                            "token": expected_token,
                            "expected_token": expected_token,
                            "candidate_token": expected_token,
                            "parameter_kind": "text_insert",
                            "reason": f"text_revision_opportunity::{wanted_conflict}",
                        }
                        parameter_estimate = self._parameter_memory.estimate(action_id="action::text_insert", proposed_params=params)
                        drive = (
                            0.34
                            + continuation_support * 0.44
                            + expected_strength * 0.18
                            + continuation_readiness * 0.18
                            + dissonance * 0.06
                            - pressure * 0.08
                            - ambiguity_pause * 0.10
                            + float(parameter_estimate.get("drive_bias", 0.0) or 0.0)
                        )
                        candidates.append(
                            self._candidate(
                                action_id="action::text_insert",
                                actuator_id=action_actuator_id("action::text_insert", "actuator::text_editor"),
                                base_drive=max(0.0, drive),
                                predicted={
                                    "reward": (0.12 + continuation_support * 0.14 + expected_strength * 0.06) * reward_gain_multiplier,
                                    "punishment": max(0.035, pressure * 0.04 + ambiguity_pause * 0.02),
                                    "expectation": expectation * 0.22 + continuation_support * 0.22 + expected_strength * 0.16,
                                    "pressure": max(0.0, pressure * 0.34 - continuation_support * 0.04),
                                    "correctness": correctness * 0.24 + continuation_support * 0.25 + expected_strength * 0.10,
                                    "confidence": 0.36 + continuation_support * 0.22 + expected_strength * 0.10,
                                },
                                notes=[
                                    "text_revision_opportunity_action",
                                    wanted_conflict,
                                    "candidate_from_memory_prediction_not_teacher_input",
                                    "requires_recent_reread",
                                    f"cursor={cursor}",
                                    f"expected_token={expected_token}",
                                    f"continuation_support={_round4(continuation_support)}",
                                ]
                                + (
                                    [
                                        "parameter_memory_bias",
                                        f"parameter_drive_bias={_round4(float(parameter_estimate.get('drive_bias', 0.0) or 0.0))}",
                                        f"parameter_similarity={_round4(float(parameter_estimate.get('similarity', 0.0) or 0.0))}",
                                    ]
                                    if float(parameter_estimate.get("support", 0.0) or 0.0) > 0.0
                                    else []
                                ),
                                consequence_estimates=consequence_estimates,
                                params=params,
                            )
                        )
                    elif operation == "delete":
                        params = {
                            "span": list(span),
                            "from_token": from_text,
                            "parameter_kind": "text_delete",
                            "reason": f"text_revision_opportunity::{conflict_kind}",
                        }
                        parameter_estimate = self._parameter_memory.estimate(action_id="action::text_delete", proposed_params=params)
                        drive = 0.36 + support * 0.48 + cleanup_pressure * 0.10 + dissonance * 0.08 + float(parameter_estimate.get("drive_bias", 0.0) or 0.0)
                        candidates.append(
                            self._candidate(
                                action_id="action::text_delete",
                                actuator_id=action_actuator_id("action::text_delete", "actuator::text_editor"),
                                base_drive=max(0.0, drive),
                                predicted={
                                    "reward": (0.12 + support * 0.15 + cleanup_pressure * 0.05) * reward_gain_multiplier,
                                    "punishment": max(0.035, pressure * 0.04),
                                    "expectation": expectation * 0.18 + support * 0.18,
                                    "pressure": max(0.0, pressure * 0.34 - support * 0.04),
                                    "correctness": correctness * 0.24 + support * 0.28,
                                    "confidence": 0.38 + support * 0.24,
                                },
                                notes=self._revision_opportunity_notes(opportunity, parameter_estimate, "delete"),
                                consequence_estimates=consequence_estimates,
                                params=params,
                            )
                        )
                    elif operation == "replace" and candidate_text:
                        params = {
                            "span": list(span),
                            "new_text": candidate_text,
                            "expected_token": candidate_text,
                            "candidate_token": candidate_text,
                            "from_token": from_text,
                            "conflict_index": int(span[0]),
                            "parameter_kind": "text_replace",
                            "reason": f"text_revision_opportunity::{conflict_kind}",
                        }
                        parameter_estimate = self._parameter_memory.estimate(action_id="action::text_replace", proposed_params=params)
                        drive = 0.34 + support * 0.48 + dissonance * 0.10 + float(parameter_estimate.get("drive_bias", 0.0) or 0.0)
                        candidates.append(
                            self._candidate(
                                action_id="action::text_replace",
                                actuator_id=action_actuator_id("action::text_replace", "actuator::text_editor"),
                                base_drive=max(0.0, drive),
                                predicted={
                                    "reward": (0.12 + support * 0.16) * reward_gain_multiplier,
                                    "punishment": max(0.04, pressure * 0.045 + ambiguity_pause * 0.02),
                                    "expectation": expectation * 0.20 + support * 0.20,
                                    "pressure": max(0.0, pressure * 0.36 - support * 0.035),
                                    "correctness": correctness * 0.24 + support * 0.28,
                                    "confidence": 0.38 + support * 0.24,
                                },
                                notes=self._revision_opportunity_notes(opportunity, parameter_estimate, "replace"),
                                consequence_estimates=consequence_estimates,
                                params=params,
                            )
                        )
        candidates.append(
            self._candidate(
                action_id="action::replay_recent_context",
                actuator_id=action_actuator_id("action::replay_recent_context", "actuator::memory_recall"),
                base_drive=replay_base_drive + boredom * 0.08 + unfinished_strength * 0.10,
                predicted={
                    "reward": (0.16 + time_conf * 0.24 + correction_pressure * 0.22) * reward_gain_multiplier,
                    "punishment": max(0.0, 0.09 - time_conf * 0.04 - correction_pressure * 0.03),
                    "expectation": expectation * 0.58,
                    "pressure": max(0.0, pressure * 0.72 - time_conf * 0.12),
                    "correctness": correctness * 0.44 + time_conf * 0.24,
                    "confidence": 0.26 + time_conf * self.confidence_gain,
                },
                notes=[
                    "temporal_replay",
                    "memory_only_continuation",
                    f"expectation_gap={_round4(expectation_gap)}",
                    f"output_mismatch_pressure={_round4(correction_pressure)}",
                    f"boredom={_round4(boredom)}",
                    f"unfinished_strength={_round4(unfinished_strength)}",
                ],
                consequence_estimates=consequence_estimates,
            )
        )
        if time_conf > 0.0 or dominant_time_peak:
            candidates.append(
                self._candidate(
                    action_id="action::recall_by_timefelt",
                    actuator_id=action_actuator_id("action::recall_by_timefelt", "actuator::memory_recall"),
                    base_drive=0.12 + time_conf * 0.54 + rhythm_expect * 0.12 + max(0.0, 1.0 - grasp) * 0.08,
                    predicted={
                        "reward": (0.11 + time_conf * 0.26 + rhythm_expect * 0.06) * reward_gain_multiplier,
                        "punishment": max(0.015, 0.08 - time_conf * 0.04),
                        "expectation": max(expectation * 0.35, rhythm_expect * 0.52),
                        "pressure": max(0.0, pressure * 0.36 - time_conf * 0.08),
                        "correctness": correctness * 0.28 + time_conf * 0.26,
                        "confidence": 0.22 + time_conf * self.confidence_gain + rhythm_expect * 0.10,
                    },
                    notes=[
                        "timefelt_recall",
                        "temporal_interval_memory_query",
                        f"time_confidence={_round4(time_conf)}",
                        f"target_delta_t={_round4(float(target_delta_t or 0.0))}",
                    ],
                    consequence_estimates=consequence_estimates,
                    params={
                        "delta_t": target_delta_t,
                        "sigma": time_sigma,
                        "confidence": _round4(time_conf),
                    },
                )
            )
        if expectation_anchors:
            anchor_drive = 0.18 + top_anchor_level * 0.44 + expectation_anchor_level * 0.14 + pressure_anchor_level * 0.18 + expectation_gap * 0.12
            candidates.append(
                self._candidate(
                    action_id="action::recall_by_expectation",
                    actuator_id=action_actuator_id("action::recall_by_expectation", "actuator::memory_recall"),
                    base_drive=anchor_drive,
                    predicted={
                        "reward": (0.13 + expectation_anchor_level * 0.20 + pressure_anchor_level * 0.08) * reward_gain_multiplier,
                        "punishment": max(0.02, pressure_anchor_level * 0.08),
                        "expectation": max(expectation, top_anchor_level * 0.82),
                        "pressure": max(0.0, pressure * 0.62 + pressure_anchor_level * 0.28),
                        "correctness": correctness * 0.34 + top_anchor_level * 0.22,
                        "confidence": 0.28 + top_anchor_level * self.confidence_gain + min(0.18, len(expectation_anchors) * 0.03),
                    },
                    notes=[
                        "b_anchor_recall",
                        "expectation_pressure_self_query",
                        f"top_anchor_level={_round4(top_anchor_level)}",
                        f"pressure_anchor_level={_round4(pressure_anchor_level)}",
                    ],
                    consequence_estimates=consequence_estimates,
                    params={
                        "b_anchor": str(expectation_anchors[0].get("anchor_id", "") or ""),
                        "source_memory_id": str(expectation_anchors[0].get("source_memory_id", "") or ""),
                    },
                    supporting_anchors=expectation_anchors[:4],
                )
            )
        if protective_need > 0.0:
            candidates.append(
                self._candidate(
                    action_id="action::avoid",
                    actuator_id=action_actuator_id("action::avoid", "actuator::protective_orientation"),
                    base_drive=0.14
                    + protective_need * 0.62
                    + pressure * 0.10
                    + uncertainty * 0.08
                    + max(0.0, 1.0 - correctness) * 0.10,
                    predicted={
                        "reward": (0.12 + protective_need * 0.24 + pressure * 0.04) * reward_gain_multiplier,
                        "punishment": max(0.025, 0.06 - protective_need * 0.02),
                        "expectation": expectation * 0.18,
                        "pressure": max(0.0, pressure * 0.36 - protective_need * 0.10),
                        "correctness": correctness * 0.22 + protective_need * 0.24,
                        "confidence": 0.28 + protective_need * self.confidence_gain + uncertainty * 0.06,
                    },
                    notes=[
                        "protective_avoid_from_risk_feeling",
                        "process_grounded_high_stakes_permission_uncertain",
                        f"protective_need={_round4(protective_need)}",
                    ],
                    consequence_estimates=consequence_estimates,
                    params={
                        "target": "current_risk",
                        "reason": "high_stakes_or_destructive_permission_uncertain",
                        "protective_need": _round4(protective_need),
                    },
                )
            )
        evidence_gap = self._evidence_gap_context(
            state_snapshot_items=state_snapshot_items,
            expected_text=expected_text,
            draft_context=draft_context,
            uncertainty=uncertainty,
            dissonance=dissonance,
            pressure=pressure,
            ambiguity_pause=ambiguity_pause,
            revision_opportunities=revision_opportunities,
        )
        if evidence_gap.get("available"):
            gap_strength = float(evidence_gap.get("strength", 0.0) or 0.0)
            missing_visual = float(evidence_gap.get("missing_visual", 0.0) or 0.0)
            missing_audio = float(evidence_gap.get("missing_audio", 0.0) or 0.0)
            conflict_strength = float(evidence_gap.get("conflict_strength", 0.0) or 0.0)
            low_grasp = float(evidence_gap.get("low_grasp", 0.0) or 0.0)
            candidates.append(
                self._candidate(
                    action_id="action::llm_think",
                    actuator_id=action_actuator_id("action::llm_think", "actuator::llm_call"),
                    base_drive=max(
                        0.0,
                        0.20
                        + gap_strength * 0.62
                        + conflict_strength * 0.22
                        + low_grasp * 0.16
                        + pressure * 0.06
                        - grasp * 0.08,
                    ),
                    predicted={
                        "reward": (0.08 + gap_strength * 0.16 + conflict_strength * 0.05) * reward_gain_multiplier,
                        "punishment": max(0.035, 0.08 + pressure * 0.025),
                        "expectation": max(expectation * 0.22, gap_strength * 0.20),
                        "pressure": max(0.0, pressure * 0.52 + gap_strength * 0.08),
                        "correctness": correctness * 0.22 + low_grasp * 0.20,
                        "confidence": 0.26 + gap_strength * 0.18 + conflict_strength * 0.08,
                    },
                    notes=[
                        "uncertainty_evidence_gap_probe",
                        "request_more_evidence_as_generic_llm_think",
                        "soft_competing_action_not_forced_rule",
                        f"gap_strength={_round4(gap_strength)}",
                        f"missing_visual={_round4(missing_visual)}",
                        f"missing_audio={_round4(missing_audio)}",
                        f"conflict_strength={_round4(conflict_strength)}",
                        f"low_grasp={_round4(low_grasp)}",
                    ],
                    consequence_estimates=consequence_estimates,
                    params={
                        "prompt_context": "need_more_evidence",
                        "reason": "uncertainty_evidence_gap",
                        "missing_modalities": list(evidence_gap.get("missing_modalities", []) or [])[:4],
                        "candidate_conflicts": list(evidence_gap.get("conflict_labels", []) or [])[:6],
                        "visible_text": str(draft_context.get("visible_text", "") or "")[:80],
                    },
                )
            )
            if missing_visual > 0.0:
                candidates.append(
                    self._candidate(
                        action_id="action::scan_visual_field",
                        actuator_id=action_actuator_id("action::scan_visual_field", "actuator::visual_gaze_center"),
                        base_drive=max(0.0, 0.18 + gap_strength * 0.28 + missing_visual * 0.36 + uncertainty * 0.08),
                        predicted={
                            "reward": (0.08 + missing_visual * 0.12 + gap_strength * 0.04) * reward_gain_multiplier,
                            "punishment": 0.035 + pressure * 0.012,
                            "expectation": max(expectation * 0.16, missing_visual * 0.20),
                            "pressure": max(0.0, pressure * 0.32 + gap_strength * 0.04),
                            "correctness": correctness * 0.14 + missing_visual * 0.18,
                            "confidence": 0.24 + missing_visual * 0.18,
                        },
                        notes=[
                            "uncertainty_visual_resample",
                            "missing_visual_evidence",
                            "scan_is_soft_sampling_action",
                            f"gap_strength={_round4(gap_strength)}",
                        ],
                        consequence_estimates=consequence_estimates,
                        params={"pattern": "uncertain_scene_resample", "reason": "missing_or_ambiguous_visual_evidence"},
                    )
                )
                candidates.append(
                    self._candidate(
                        action_id="action::widen_visual_focus",
                        actuator_id=action_actuator_id("action::widen_visual_focus", "actuator::visual_focus_scale"),
                        base_drive=max(0.0, 0.16 + missing_visual * 0.32 + ambiguity_pause * 0.12),
                        predicted={
                            "reward": (0.07 + missing_visual * 0.10) * reward_gain_multiplier,
                            "punishment": 0.03 + pressure * 0.01,
                            "expectation": max(expectation * 0.15, missing_visual * 0.18),
                            "pressure": max(0.0, pressure * 0.30 + ambiguity_pause * 0.03),
                            "correctness": correctness * 0.12 + missing_visual * 0.16,
                            "confidence": 0.22 + missing_visual * 0.14,
                        },
                        notes=[
                            "uncertainty_visual_widen",
                            "peripheral_or_missing_visual_evidence",
                        ],
                        consequence_estimates=consequence_estimates,
                        params={"scale": 1.18, "reason": "widen_for_uncertain_visual_evidence"},
                    )
                )
            if missing_audio > 0.0:
                candidates.append(
                    self._candidate(
                        action_id="action::widen_audio_band",
                        actuator_id=action_actuator_id("action::widen_audio_band", "actuator::auditory_band_width"),
                        base_drive=max(0.0, 0.17 + gap_strength * 0.26 + missing_audio * 0.38 + uncertainty * 0.07),
                        predicted={
                            "reward": (0.08 + missing_audio * 0.12 + gap_strength * 0.04) * reward_gain_multiplier,
                            "punishment": 0.035 + pressure * 0.012,
                            "expectation": max(expectation * 0.16, missing_audio * 0.20),
                            "pressure": max(0.0, pressure * 0.32 + gap_strength * 0.04),
                            "correctness": correctness * 0.14 + missing_audio * 0.18,
                            "confidence": 0.24 + missing_audio * 0.18,
                        },
                        notes=[
                            "uncertainty_audio_resample",
                            "missing_audio_evidence",
                            "audio_sampling_action_not_answer_hint",
                            f"gap_strength={_round4(gap_strength)}",
                        ],
                        consequence_estimates=consequence_estimates,
                        params={"width_hz": 3200, "reason": "missing_or_ambiguous_audio_evidence"},
                    )
                )
        if visual_target.get("available"):
            target_score = float(visual_target.get("score", 0.0) or 0.0)
            target_gain = float(visual_target.get("focus_gain", 0.0) or 0.0)
            target_precision = float(visual_target.get("focus_precision", 0.0) or 0.0)
            target_distance = float(visual_target.get("distance", 0.0) or 0.0)
            peripheral_need = float(visual_target.get("peripheral_need", 0.0) or 0.0)
            target_label = str(visual_target.get("sa_label", "") or "")
            target_fatigue = float(visual_target.get("target_fatigue", 0.0) or 0.0)
            gaze_params = {
                "x": _round4(float(visual_target.get("x", 0.5) or 0.5)),
                "y": _round4(float(visual_target.get("y", 0.5) or 0.5)),
                "target": target_label,
                "gaze_target_key": str(visual_target.get("gaze_target_key", "") or target_label),
                "bbox_norm": list(visual_target.get("bbox_norm", []) or []),
                "reason": str(visual_target.get("reason", "") or "visual_attention_target"),
                "score_components": dict(visual_target.get("score_components", {}) or {}),
                "target_distance": _round4(target_distance),
                "focus_precision": _round4(target_precision),
                "focus_gain": _round4(target_gain),
            }
            parameter_estimate = self._parameter_memory.estimate(
                action_id="action::move_gaze_to",
                proposed_params=gaze_params,
                current_gaze={
                    "center_x": _clamp(float(visual_target.get("current_gaze_x", 0.5) or 0.5), 0.0, 1.0),
                    "center_y": _clamp(float(visual_target.get("current_gaze_y", 0.5) or 0.5), 0.0, 1.0),
                },
            )
            parameter_drive_bias = float(parameter_estimate.get("drive_bias", 0.0) or 0.0)
            parameter_pressure_bias = float(parameter_estimate.get("pressure_bias", 0.0) or 0.0)
            if float(parameter_estimate.get("support", 0.0) or 0.0) > 0.0:
                gaze_params["learned_parameter_hint"] = dict(parameter_estimate)
            if target_distance > 0.045 or peripheral_need > 0.18:
                # This is the visual analogue of turning the eyes toward what the
                # cognitive field says matters. The target is derived from live SA
                # energy/attention, not from a fixed scan script.
                candidates.append(
                    self._candidate(
                        action_id="action::move_gaze_to",
                        actuator_id=action_actuator_id("action::move_gaze_to", "actuator::visual_gaze_center"),
                        base_drive=0.20 + target_score * 0.50 + peripheral_need * 0.22 + target_distance * 0.08 + parameter_drive_bias,
                        predicted={
                            "reward": (0.10 + target_score * 0.12 + peripheral_need * 0.06) * reward_gain_multiplier,
                            "punishment": max(0.018, 0.055 - target_score * 0.018),
                            "expectation": max(expectation * 0.22, target_score * 0.36),
                            "pressure": max(0.0, pressure * 0.32 + peripheral_need * 0.05 - target_score * 0.03 + parameter_pressure_bias),
                            "correctness": correctness * 0.18 + target_score * 0.22,
                            "confidence": 0.30 + target_score * 0.26 + peripheral_need * 0.10,
                        },
                        notes=[
                            "cognition_driven_visual_gaze",
                            "target_from_state_field_and_attention",
                            f"target={target_label}",
                            f"target_score={_round4(target_score)}",
                            f"target_fatigue={_round4(target_fatigue)}",
                            f"peripheral_need={_round4(peripheral_need)}",
                            f"distance={_round4(target_distance)}",
                        ]
                        + (
                            [
                                "parameter_memory_bias",
                                f"parameter_drive_bias={_round4(parameter_drive_bias)}",
                                f"parameter_similarity={_round4(float(parameter_estimate.get('similarity', 0.0) or 0.0))}",
                            ]
                            if float(parameter_estimate.get("support", 0.0) or 0.0) > 0.0
                            else []
                        ),
                        consequence_estimates=consequence_estimates,
                        params=gaze_params,
                    )
                )
            if target_gain >= 0.42 and target_score >= 0.18:
                candidates.append(
                    self._candidate(
                        action_id="action::hold_gaze",
                        actuator_id=action_actuator_id("action::hold_gaze", "actuator::visual_gaze_center"),
                        base_drive=max(0.0, 0.12 + target_score * 0.26 + target_gain * 0.14 + target_precision * 0.08 - target_fatigue * 0.22),
                        predicted={
                            "reward": (0.08 + target_score * 0.08 + target_precision * 0.05) * reward_gain_multiplier,
                            "punishment": 0.026,
                            "expectation": max(expectation * 0.18, target_score * 0.24),
                            "pressure": max(0.0, pressure * 0.22 - target_precision * 0.04),
                            "correctness": correctness * 0.16 + target_precision * 0.16,
                            "confidence": 0.28 + target_gain * 0.16 + target_score * 0.16,
                        },
                        notes=[
                            "visual_target_already_near_fovea",
                            "hold_to_continue_sampling",
                            f"target={target_label}",
                            f"focus_gain={_round4(target_gain)}",
                            f"target_fatigue={_round4(target_fatigue)}",
                        ],
                        consequence_estimates=consequence_estimates,
                        params={"target": target_label, "reason": "continue_visual_sampling", "bbox_norm": list(visual_target.get("bbox_norm", []) or [])},
                    )
                )
            if target_score >= 0.20 and target_precision < 0.86:
                candidates.append(
                    self._candidate(
                        action_id="action::zoom_visual_focus",
                        actuator_id=action_actuator_id("action::zoom_visual_focus", "actuator::visual_focus_scale"),
                        base_drive=max(0.0, 0.14 + target_score * 0.34 + max(0.0, 1.0 - target_precision) * 0.18 + uncertainty * 0.06 - target_fatigue * 0.10),
                        predicted={
                            "reward": (0.08 + target_score * 0.08 + max(0.0, 1.0 - target_precision) * 0.04) * reward_gain_multiplier,
                            "punishment": 0.032 + max(0.0, pressure - target_score) * 0.018,
                            "expectation": max(expectation * 0.16, target_score * 0.22),
                            "pressure": max(0.0, pressure * 0.26 + uncertainty * 0.03),
                            "correctness": correctness * 0.15 + target_score * 0.12,
                            "confidence": 0.26 + target_score * 0.16 + max(0.0, 1.0 - target_precision) * 0.08,
                        },
                        notes=[
                            "visual_detail_sampling_pressure",
                            "foveated_resolution_action",
                            f"target={target_label}",
                            f"focus_precision={_round4(target_precision)}",
                            f"target_fatigue={_round4(target_fatigue)}",
                        ],
                        consequence_estimates=consequence_estimates,
                        params={
                            "scale": 0.68,
                            "target": target_label,
                            "reason": "increase_visual_detail",
                            "bbox_norm": list(visual_target.get("bbox_norm", []) or []),
                            "target_distance": _round4(target_distance),
                            "focus_precision": _round4(target_precision),
                            "focus_gain": _round4(target_gain),
                        },
                    )
                )
            for alternative in list(visual_target.get("alternatives", []) or [])[:3]:
                if not isinstance(alternative, dict):
                    continue
                alt_score = float(alternative.get("score", 0.0) or 0.0)
                alt_need = float(alternative.get("peripheral_need", 0.0) or 0.0)
                alt_distance = float(alternative.get("distance", 0.0) or 0.0)
                alt_fatigue = float(alternative.get("target_fatigue", 0.0) or 0.0)
                # Alternatives are allowed into competition only when there is
                # real exploratory pressure: the current best is fatigued, or
                # the alternative is close enough in score and still unclear.
                if not (target_fatigue >= 0.12 or (alt_score >= max(0.14, target_score - 0.18) and alt_need > 0.16)):
                    continue
                alt_label = str(alternative.get("sa_label", "") or "")
                if not alt_label or alt_label == target_label:
                    continue
                alt_params = {
                    "x": _round4(float(alternative.get("x", 0.5) or 0.5)),
                    "y": _round4(float(alternative.get("y", 0.5) or 0.5)),
                    "target": alt_label,
                    "gaze_target_key": str(alternative.get("gaze_target_key", "") or alt_label),
                    "bbox_norm": list(alternative.get("bbox_norm", []) or []),
                    "reason": "visual_exploration_after_target_fatigue",
                    "score_components": dict(alternative.get("score_components", {}) or {}),
                    "target_distance": _round4(alt_distance),
                    "focus_precision": _round4(float(alternative.get("focus_precision", 0.0) or 0.0)),
                    "focus_gain": _round4(float(alternative.get("focus_gain", 0.0) or 0.0)),
                }
                alt_parameter_estimate = self._parameter_memory.estimate(
                    action_id="action::move_gaze_to",
                    proposed_params=alt_params,
                    current_gaze={
                        "center_x": _clamp(float(alternative.get("current_gaze_x", 0.5) or 0.5), 0.0, 1.0),
                        "center_y": _clamp(float(alternative.get("current_gaze_y", 0.5) or 0.5), 0.0, 1.0),
                    },
                )
                alt_parameter_drive_bias = float(alt_parameter_estimate.get("drive_bias", 0.0) or 0.0)
                if float(alt_parameter_estimate.get("support", 0.0) or 0.0) > 0.0:
                    alt_params["learned_parameter_hint"] = dict(alt_parameter_estimate)
                candidates.append(
                    self._candidate(
                        action_id="action::move_gaze_to",
                        actuator_id=action_actuator_id("action::move_gaze_to", "actuator::visual_gaze_center"),
                        base_drive=max(
                            0.0,
                            0.12
                            + alt_score * 0.34
                            + alt_need * 0.24
                            + alt_distance * 0.06
                            + target_fatigue * 0.18
                            - alt_fatigue * 0.20
                            + alt_parameter_drive_bias,
                        ),
                        predicted={
                            "reward": (0.08 + alt_score * 0.10 + alt_need * 0.05) * reward_gain_multiplier,
                            "punishment": max(0.022, 0.06 - alt_score * 0.012),
                            "expectation": max(expectation * 0.16, alt_score * 0.24),
                            "pressure": max(0.0, pressure * 0.24 + alt_need * 0.04),
                            "correctness": correctness * 0.12 + alt_score * 0.16,
                            "confidence": 0.22 + alt_score * 0.18 + alt_need * 0.08,
                        },
                        notes=[
                            "visual_exploration_alternative",
                            "alternative_target_from_fatigue_or_unclear_periphery",
                            f"target={alt_label}",
                            f"alt_score={_round4(alt_score)}",
                            f"best_target_fatigue={_round4(target_fatigue)}",
                            f"alt_target_fatigue={_round4(alt_fatigue)}",
                            f"peripheral_need={_round4(alt_need)}",
                        ]
                        + (
                            [
                                "parameter_memory_bias",
                                f"parameter_drive_bias={_round4(alt_parameter_drive_bias)}",
                                f"parameter_similarity={_round4(float(alt_parameter_estimate.get('similarity', 0.0) or 0.0))}",
                            ]
                            if float(alt_parameter_estimate.get("support", 0.0) or 0.0) > 0.0
                            else []
                        ),
                        consequence_estimates=consequence_estimates,
                        params=alt_params,
                    )
                )
        episode_risk = self._episode_replay_risk(
            action_consequence_trace=action_consequence_trace or {},
            pressure_anchor_level=pressure_anchor_level,
            pressure=pressure,
            expectation_gap=expectation_gap,
        )
        if pressure > 0.0 or pressure_anchor_level > 0.0 or episode_risk["risk"] > 0.0 or expectation_gap > 0.0:
            candidates.append(
                self._candidate(
                    action_id="action::replay_episode",
                    actuator_id=action_actuator_id("action::replay_episode", "actuator::memory_recall"),
                    base_drive=0.16
                    + pressure * 0.24
                    + pressure_anchor_level * 0.42
                    + expectation_gap * 0.16
                    + float(episode_risk.get("risk", 0.0) or 0.0) * 0.24,
                    predicted={
                        "reward": (0.10 + pressure_anchor_level * 0.08 + float(episode_risk.get("support", 0.0) or 0.0) * 0.06) * reward_gain_multiplier,
                        "punishment": max(0.04, pressure_anchor_level * 0.10 + float(episode_risk.get("punishment", 0.0) or 0.0) * 0.16),
                        "expectation": expectation * 0.28,
                        "pressure": max(pressure * 0.72, pressure_anchor_level * 0.62, float(episode_risk.get("risk", 0.0) or 0.0) * 0.58),
                        "correctness": correctness * 0.22 + pressure_anchor_level * 0.10,
                        "confidence": 0.24 + pressure_anchor_level * self.confidence_gain + float(episode_risk.get("support", 0.0) or 0.0) * 0.12,
                    },
                    notes=[
                        "episode_replay_before_risky_action",
                        "pressure_consequence_memory_query",
                        f"pressure_anchor_level={_round4(pressure_anchor_level)}",
                        f"episode_risk={_round4(float(episode_risk.get('risk', 0.0) or 0.0))}",
                    ],
                    consequence_estimates=consequence_estimates,
                    params={
                        "source_memory_id": str(episode_risk.get("source_memory_id", "") or ""),
                        "risk": _round4(float(episode_risk.get("risk", 0.0) or 0.0)),
                    },
                    supporting_anchors=[
                        anchor
                        for anchor in expectation_anchors[:4]
                        if str(anchor.get("anchor_type", "") or "") == "pressure"
                    ],
                )
            )
        candidates.append(
            self._candidate(
                action_id="action::stabilize_prediction",
                actuator_id=action_actuator_id("action::stabilize_prediction", "actuator::legacy_internal"),
                base_drive=0.10 + max(0.0, predicted_mass - 2) * 0.04 + expectation * 0.18 + pressure * 0.16 + expectation_gap * 0.1,
                predicted={
                    "reward": (0.12 + expectation * 0.18 + alignment_score * 0.08) * reward_gain_multiplier,
                    "punishment": max(0.0, 0.11 - correctness * 0.05),
                    "expectation": expectation * 0.74,
                    "pressure": pressure * 0.84,
                    "correctness": correctness * 0.64,
                    "confidence": 0.24 + correctness * self.confidence_gain,
                },
                notes=[
                    "prediction_binding",
                    "future_consequence_estimate",
                    f"expectation_gap={_round4(expectation_gap)}",
                ],
                consequence_estimates=consequence_estimates,
            )
        )
        candidates.append(
            self._candidate(
                action_id="action::wait",
                actuator_id=action_actuator_id("action::wait", "actuator::timing"),
                base_drive=self.wait_base_drive
                + max(0.0, 0.14 - focusable_count * 0.02)
                + uncertainty * 0.22
                + rhythm_expect * 0.18
                + ambiguity_pause * 0.22
                + cleanup_pressure * 0.10
                + max(0.0, pressure - correctness * 0.45) * 0.12
                + pressure_anchor_level * 0.16
                + max(0.0, fulfillment - 0.45) * 0.06
                - boredom * 0.04,
                predicted={
                    "reward": (0.06 + uncertainty * 0.06 + rhythm_expect * 0.05 + ambiguity_pause * 0.06 + pressure_anchor_level * 0.04) * reward_gain_multiplier,
                    "punishment": 0.025 + max(0.0, pressure - pressure_anchor_level) * 0.015,
                    "expectation": expectation * 0.18,
                    "pressure": max(0.0, pressure * 0.38 - 0.06 + cleanup_pressure * 0.03),
                    "correctness": correctness * 0.18,
                    "confidence": 0.18 + uncertainty * 0.08 + rhythm_expect * 0.08 + ambiguity_pause * 0.08,
                },
                notes=[
                    "timing_wait",
                    "legal_non_action",
                    "successor_ambiguity_pause" if ambiguity_pause >= 0.32 else "ordinary_timing_pause",
                    f"uncertainty={_round4(uncertainty)}",
                    f"rhythm_expectation={_round4(rhythm_expect)}",
                    f"ambiguity_pause={_round4(ambiguity_pause)}",
                    f"cleanup_pressure={_round4(cleanup_pressure)}",
                    f"boredom={_round4(boredom)}",
                    f"fulfillment={_round4(fulfillment)}",
                ],
                consequence_estimates=consequence_estimates,
                params={"duration_ticks": 1, "rhythm_expectation": _round4(rhythm_expect), "uncertainty": _round4(uncertainty)},
            )
        )
        candidates = self._merge_memory_predicted_action_energy(
            candidates,
            state_snapshot_items,
            drive_gain=memory_action_drive_gain,
            consequence_estimates=consequence_estimates,
            evidence_gap_context=evidence_gap,
        )
        candidates = self._merge_consequence_supported_action_energy(
            candidates,
            consequence_estimates,
            evidence_gap_context=evidence_gap,
        )
        candidates = self._apply_visual_orientation_arbitration(candidates)
        candidates = self._apply_pre_write_visual_closure_guard(candidates, draft_context)
        candidates = self._apply_text_cursor_readiness_guard(candidates, draft_context)
        candidates = self._suppress_unavailable_draft_actions(candidates, draft_context)
        candidates = self._apply_unclosed_insert_review_guard(candidates, draft_context)
        candidates = self._apply_draft_repetition_guard(candidates, draft_context)
        candidates = self._apply_draft_review_saturation(candidates, draft_context, draft_goal_alignment)
        candidates = self._apply_post_commit_empty_surface_guard(candidates, draft_context, draft_goal_alignment)
        candidates = self._merge_innate_action_nodes(
            candidates,
            innate_action_nodes or [],
            consequence_estimates=consequence_estimates,
            evidence_gap_context=evidence_gap,
        )
        candidates = self._apply_innate_action_biases(
            candidates,
            innate_action_biases or [],
            evidence_gap_context=evidence_gap,
        )
        candidates = self._apply_experience_supported_attention_arbitration(candidates)
        candidates = self._apply_attention_process_need_arbitration(candidates, attention_process_need)
        candidates = self._apply_visual_orientation_arbitration(candidates)
        candidates = self._apply_pre_write_visual_closure_guard(candidates, draft_context)
        candidates = self._apply_text_cursor_readiness_guard(candidates, draft_context)
        candidates = self._suppress_unavailable_draft_actions(candidates, draft_context)
        candidates = self._apply_unclosed_insert_review_guard(candidates, draft_context)
        candidates = self._apply_draft_repetition_guard(candidates, draft_context)
        candidates = self._apply_draft_review_saturation(candidates, draft_context, draft_goal_alignment)
        candidates = self._apply_post_commit_empty_surface_guard(candidates, draft_context, draft_goal_alignment)
        return candidates

    def _apply_experience_supported_attention_arbitration(self, candidates: list[dict]) -> list[dict]:
        """
        Let learned attention actions compete with generic residual inspection.

        A large pile of unexpected fresh text should not make AP inspect
        residuals forever when experience says a more specific attention action
        advances the task. This is still ordinary action competition: learned
        successor feedback only changes drive, and inspect_residual remains
        available when it has its own evidence or a real mismatch is strong.
        """

        attention_domain = "attention_focus_width_and_anchor"
        attention_rows = [
            row
            for row in candidates or []
            if isinstance(row, dict) and self._conflict_domain(row) == attention_domain
        ]
        if not attention_rows:
            return candidates
        for row in attention_rows:
            estimate = dict(row.get("consequence_estimate", {}) or {})
            support = _clamp(float(estimate.get("support", 0.0) or 0.0), 0.0, 1.0)
            reward = max(0.0, float(estimate.get("reward", 0.0) or 0.0))
            punishment = max(0.0, float(estimate.get("punishment", 0.0) or 0.0))
            correctness = max(0.0, float(estimate.get("correctness", 0.0) or 0.0))
            if support <= 0.0:
                continue
            utility = reward + correctness * 0.35 - punishment * 0.85
            if utility <= 0.0:
                continue
            bonus = min(0.52, support * utility * 0.50)
            if bonus <= 0.0:
                continue
            row["base_drive"] = _round4(float(row.get("base_drive", 0.0) or 0.0) + bonus)
            row["drive"] = _round4(_clamp(float(row.get("drive", 0.0) or 0.0) + bonus, 0.0, 1.8))
            row.setdefault("notes", [])
            row["notes"] = list(row.get("notes", []) or []) + [
                "experience_supported_attention_action_drive",
                f"experience_attention_bonus={_round4(bonus)}",
            ]

        non_inspect_best = max(
            [
                float(row.get("drive", 0.0) or 0.0)
                for row in attention_rows
                if str(row.get("action_id", "") or "") != "action::inspect_residual"
                and float((row.get("consequence_estimate", {}) or {}).get("support", 0.0) or 0.0) >= 0.25
            ]
            or [0.0]
        )
        if non_inspect_best <= 0.0:
            return candidates
        for row in attention_rows:
            if str(row.get("action_id", "") or "") != "action::inspect_residual":
                continue
            estimate = dict(row.get("consequence_estimate", {}) or {})
            if float(estimate.get("support", 0.0) or 0.0) > 0.0:
                continue
            notes = {str(note or "") for note in list(row.get("notes", []) or [])}
            mismatch_notes = [note for note in notes if note.startswith("mismatch_ratio=")]
            mismatch_ratio = 0.0
            if mismatch_notes:
                try:
                    mismatch_ratio = float(mismatch_notes[-1].split("=", 1)[-1])
                except ValueError:
                    mismatch_ratio = 0.0
            # If the only evidence is broad unexpected novelty, keep inspect as
            # a backup. If mismatch is extreme and no learned action is close,
            # this damping is small; with a learned attention action nearby, AP
            # can try that more specific move first.
            gap = max(0.0, float(row.get("drive", 0.0) or 0.0) - non_inspect_best)
            damping = min(0.68, 0.24 + gap * 0.18 + max(0.0, 0.72 - mismatch_ratio) * 0.18)
            row["base_drive"] = _round4(max(0.0, float(row.get("base_drive", 0.0) or 0.0) - damping))
            row["drive"] = _round4(max(0.0, float(row.get("drive", 0.0) or 0.0) - damping))
            row.setdefault("notes", [])
            row["notes"] = list(row.get("notes", []) or []) + [
                "generic_residual_inspect_yields_to_learned_attention_action",
                f"learned_attention_competitor_drive={_round4(non_inspect_best)}",
                f"residual_inspect_damping={_round4(damping)}",
            ]
        return candidates

    def _attention_action_need_scores(
        self,
        *,
        state_snapshot_items: list[dict],
        focus_targets: list[str],
        release_targets: list[str],
        surprise: float,
        pressure: float,
        expectation: float,
        grasp: float,
        uncertainty: float,
        evidence_gap: float,
        dissonance: float,
        coherence: float,
        correctness: float,
        fulfillment: float,
        task_available: float,
        rhythm_expect: float,
        learned_attention_process: bool,
    ) -> dict[str, float]:
        """
        Read the current process field as a soft need for attention actions.

        This is not a keyword route for a task. Labels such as process_step::*,
        state::low_grasp, and feeling::old_topic_interference are already AP
        state-field objects. Strong external emphasis can raise the same field
        through real_energy/cognitive_pressure/attention_gain, so the learned
        focus action remains teachable in ordinary open-world interaction.
        """

        rows = [dict(item) for item in state_snapshot_items or [] if isinstance(item, dict)]
        current_source_types = {"current_test_process", "current_target_sa", "external_text", "external_teacher"}
        current_rows = [
            row
            for row in rows
            if str(row.get("source_type", "") or "") in current_source_types
            and not self._is_attention_background_label(row)
        ]
        labels = {str(row.get("sa_label", "") or "") for row in rows if str(row.get("sa_label", "") or "")}
        current_labels = {str(row.get("sa_label", "") or "") for row in current_rows if str(row.get("sa_label", "") or "")}
        source_counts: dict[str, int] = defaultdict(int)
        prefix_counts: dict[str, int] = defaultdict(int)
        current_pressure = 0.0
        target_pressure = 0.0
        old_pressure = 0.0
        process_pressure = 0.0
        for row in rows:
            label = str(row.get("sa_label", "") or "")
            if not label:
                continue
            prefix = label.split("::", 1)[0]
            source_type = str(row.get("source_type", "") or "")
            pressure_value = max(
                float(row.get("cognitive_pressure", 0.0) or 0.0),
                float(row.get("real_energy", 0.0) or 0.0) * 0.72,
                float(row.get("attention_gain", 0.0) or 0.0),
            )
            source_counts[source_type] += 1
            prefix_counts[prefix] += 1
            is_current = source_type in current_source_types
            if is_current:
                current_pressure += pressure_value
            if label in set(focus_targets) and is_current:
                target_pressure += pressure_value
            if label in set(release_targets) or source_type == "old_episode_residue" or label.startswith(("old_episode::", "residue::")):
                old_pressure += pressure_value
            if is_current and prefix in {"state", "feeling", "goal", "process_step", "future_feedback"}:
                process_pressure += pressure_value

        current_norm = _clamp(current_pressure / max(1.0, len(current_rows) * 0.62), 0.0, 1.0)
        target_norm = _clamp(target_pressure / max(1.0, len(focus_targets) * 0.82), 0.0, 1.0)
        old_norm = _clamp(old_pressure / max(1.0, len(release_targets) * 0.72), 0.0, 1.0)
        process_norm = _clamp(process_pressure / max(1.0, len(current_rows) * 0.42), 0.0, 1.0)

        algorithm_continuation = 0.0
        if any(label.startswith("algorithm::") for label in current_labels):
            algorithm_continuation += 0.34
        if any(label.startswith("operation::") for label in current_labels):
            algorithm_continuation += 0.12
        if any(label.startswith("process_step::") and ("pending" in label or "sequence" in label or "type_char" in label) for label in current_labels):
            algorithm_continuation += 0.24
        if "state::learned_rule_available" in current_labels:
            algorithm_continuation += 0.18
        if "state::draft_not_complete" in current_labels or "token::next_char_available" in current_labels:
            algorithm_continuation += 0.20
        if "future_feedback::task_progress" in current_labels:
            algorithm_continuation += 0.16
        if "feeling::partial_confidence" in current_labels:
            algorithm_continuation += 0.10

        stuck_or_unclear = 0.0
        if "state::low_grasp" in current_labels:
            stuck_or_unclear += 0.26
        if "state::evidence_gap" in current_labels:
            stuck_or_unclear += 0.22
        if "state::low_coherence" in current_labels:
            stuck_or_unclear += 0.22
        if "feeling::dissonance" in current_labels:
            stuck_or_unclear += 0.14
        if "goal::reread_before_try" in current_labels or "goal::expand_observation" in current_labels or "goal::find_clean_signal" in current_labels:
            stuck_or_unclear += 0.20
        if any(label.startswith(("audio::", "noise::", "sound::")) for label in current_labels) and ("state::low_grasp" in current_labels or "state::low_coherence" in current_labels):
            stuck_or_unclear += 0.18

        release_old = 0.0
        if "state::topic_switch_detected" in current_labels:
            release_old += 0.30
        if "feeling::old_topic_interference" in current_labels:
            release_old += 0.30
        if "feeling::self_expression_repetition" in current_labels or "feeling::stale_focus" in current_labels:
            release_old += 0.22
        if "goal::release_old_focus" in current_labels:
            release_old += 0.18
        if old_norm > 0.0:
            release_old += min(0.26, old_norm * 0.28)

        new_anchor = 0.0
        if "state::current_input_new" in current_labels:
            new_anchor += 0.20
        if "feeling::surprise" in current_labels:
            new_anchor += 0.18
        if "goal::understand_current_task" in current_labels:
            new_anchor += 0.14
        if any(label.startswith(("desktop::", "vision::", "ocr::", "permission::", "intention::", "cue::", "opportunity::", "timefelt::")) for label in current_labels):
            new_anchor += 0.12
        if learned_attention_process:
            new_anchor += 0.08

        continue_score = _clamp(
            algorithm_continuation
            + expectation * 0.18
            + grasp * 0.16
            + correctness * 0.08
            + rhythm_expect * 0.10
            + fulfillment * 0.06
            + task_available * 0.06
            + target_norm * 0.10
            - max(0.0, evidence_gap - 0.28) * 0.10,
            0.0,
            1.2,
        )
        diverge_score = _clamp(
            stuck_or_unclear
            + uncertainty * 0.18
            + evidence_gap * 0.18
            + max(0.0, dissonance - coherence * 0.36) * 0.18
            + max(0.0, 0.55 - grasp) * 0.10
            - algorithm_continuation * 0.08,
            0.0,
            1.2,
        )
        release_score = _clamp(
            release_old
            + old_norm * 0.20
            + max(0.0, pressure - correctness) * 0.08
            + fulfillment * 0.08
            - target_norm * 0.04,
            0.0,
            1.2,
        )
        focus_score = _clamp(
            new_anchor
            + surprise * 0.18
            + pressure * 0.08
            + current_norm * 0.10
            + target_norm * 0.12
            + process_norm * 0.05
            - max(continue_score, release_score, diverge_score) * 0.08,
            0.0,
            1.2,
        )
        return {
            "schema_id": "attention_action_process_need/v1",
            "action::focus_anchor": _round4(focus_score),
            "action::continue_focus": _round4(continue_score),
            "action::release_focus": _round4(release_score),
            "action::diverge_attention": _round4(diverge_score),
            "current_pressure_norm": _round4(current_norm),
            "target_pressure_norm": _round4(target_norm),
            "old_residue_pressure_norm": _round4(old_norm),
            "process_pressure_norm": _round4(process_norm),
            "algorithm_continuation": _round4(algorithm_continuation),
            "stuck_or_unclear": _round4(stuck_or_unclear),
            "release_old": _round4(release_old),
            "new_anchor": _round4(new_anchor),
        }

    def _apply_attention_process_need_arbitration(self, candidates: list[dict], attention_process_need: dict | None) -> list[dict]:
        """
        Softly arbitrate learned attention actions by the current process field.

        This keeps the user's intended AP philosophy: focus/release/diverge are
        actions with predicted consequences. A candidate may still lose if its
        own learned support or utility is weak; this method only supplies a
        process-grounded nudge and a soft cost for the wrong attention mode.
        """

        need = dict(attention_process_need or {})
        attention_ids = {
            "action::focus_anchor",
            "action::continue_focus",
            "action::release_focus",
            "action::diverge_attention",
            "action::inspect_residual",
        }
        rows = [
            row
            for row in candidates or []
            if isinstance(row, dict) and str(row.get("action_id", "") or "") in attention_ids
        ]
        if not rows:
            return candidates
        best_action = ""
        best_need = 0.0
        for action_id in ("action::focus_anchor", "action::continue_focus", "action::release_focus", "action::diverge_attention"):
            value = float(need.get(action_id, 0.0) or 0.0)
            if value > best_need:
                best_need = value
                best_action = action_id
        if best_need <= 0.0:
            return candidates
        for row in rows:
            action_id = str(row.get("action_id", "") or "")
            own_need = float(need.get(action_id, 0.0) or 0.0)
            if action_id == "action::inspect_residual":
                own_need = max(own_need, float(need.get("action::diverge_attention", 0.0) or 0.0) * 0.42)
            support = _clamp(float((row.get("consequence_estimate", {}) or {}).get("support", 0.0) or 0.0), 0.0, 1.0)
            reward = max(0.0, float((row.get("consequence_estimate", {}) or {}).get("reward", 0.0) or 0.0))
            learned_support = min(0.12, support * max(0.0, reward) * 0.05)
            need_gap = max(0.0, best_need - own_need)
            second_need = max(
                [
                    float(need.get(other_action, 0.0) or 0.0)
                    for other_action in ("action::focus_anchor", "action::continue_focus", "action::release_focus", "action::diverge_attention")
                    if other_action != best_action
                ]
                or [0.0]
            )
            dominant_margin = max(0.0, best_need - second_need)
            boost = min(0.62, own_need * 0.30 + learned_support)
            penalty = min(0.62, need_gap * 0.34)
            if action_id == best_action and best_need >= 0.76:
                boost += min(0.38, 0.16 + best_need * 0.12 + dominant_margin * 0.22)
            if action_id != best_action and support >= 0.25 and need_gap > 0.08:
                # Learned action feedback is still only temporally/process
                # applicable. When the current process says "this is a stuck
                # or switched situation", an old successful attention habit is
                # remembered but its predicted usefulness is lower this tick.
                penalty += min(0.30, support * (0.06 + need_gap * 0.50))
            if action_id == "action::inspect_residual" and best_action in {"action::release_focus", "action::continue_focus"}:
                penalty += min(0.30, best_need * 0.20)
            if action_id == "action::inspect_residual" and best_action == "action::diverge_attention":
                penalty += min(0.22, best_need * 0.14)
            if best_action == "action::diverge_attention" and action_id in {"action::focus_anchor", "action::continue_focus"}:
                penalty += min(0.40, 0.10 + best_need * 0.18 + need_gap * 0.22)
            if best_action == "action::diverge_attention" and action_id == "action::continue_focus":
                penalty += min(0.36, 0.10 + support * 0.18 + need_gap * 0.25)
            if best_action == "action::continue_focus" and action_id == "action::focus_anchor":
                penalty += min(0.34, 0.08 + best_need * 0.14 + need_gap * 0.18)
            if best_action == "action::release_focus" and action_id in {"action::focus_anchor", "action::continue_focus", "action::inspect_residual"}:
                penalty += min(0.44, 0.12 + best_need * 0.18 + need_gap * 0.22)
            if best_action != "action::focus_anchor" and action_id == "action::focus_anchor" and best_need >= 0.84:
                penalty += min(0.28, 0.06 + best_need * 0.10)
            if best_action == "action::focus_anchor" and action_id == "action::continue_focus" and need_gap >= 0.08:
                penalty += min(0.22, 0.06 + need_gap * 0.22)
            delta = boost - penalty
            if abs(delta) <= 0.0001:
                continue
            row["base_drive"] = _round4(max(0.0, float(row.get("base_drive", 0.0) or 0.0) + delta))
            row["drive"] = _round4(_clamp(float(row.get("drive", 0.0) or 0.0) + delta, 0.0, 1.8))
            params = dict(row.get("params", {}) or {})
            params.setdefault("attention_process_need", {k: v for k, v in need.items() if k.startswith("action::")})
            row["params"] = params
            row.setdefault("notes", [])
            row["notes"] = list(row.get("notes", []) or []) + [
                "process_grounded_attention_need_arbitration",
                f"dominant_attention_need={best_action}:{_round4(best_need)}",
                f"own_attention_need={_round4(own_need)}",
                f"attention_need_delta={_round4(delta)}",
            ]
        return candidates

    def _attention_anchor_target_labels(self, state_snapshot_items: list[dict], *, limit: int) -> list[str]:
        rows = []
        for item in state_snapshot_items or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label or self._is_attention_background_label(item):
                continue
            prefix = label.split("::", 1)[0]
            if prefix == "text" and not self._is_semantic_text_anchor(label):
                continue
            pressure = float(item.get("cognitive_pressure", 0.0) or 0.0)
            real = float(item.get("real_energy", 0.0) or 0.0)
            attention_gain = float(item.get("attention_gain", 0.0) or 0.0)
            if pressure <= 0.0 and real <= 0.0 and attention_gain <= 0.0:
                continue
            meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
            priority = 0.28 if bool(meta.get("focus_target", False)) else 0.0
            if str(item.get("source_type", "") or "") in {"current_test_process", "current_target_sa", "external_text", "external_teacher"}:
                priority += 0.18
            if prefix in {"math", "operation", "algorithm", "number", "relation", "desktop", "vision", "ocr", "permission", "emotion", "tone", "audio", "noise", "sound", "intention", "cue", "timefelt", "opportunity", "draft", "token"}:
                priority += 0.58
            elif prefix in {"process_step", "goal"}:
                priority += 0.34
            elif prefix == "state":
                priority += 0.18
            elif prefix == "feeling":
                priority += 0.08
            if label in {"feeling::surprise", "feeling::coherence", "feeling::dissonance", "timefelt::elapsed", "expectation_pressure::pressure", "expectation_pressure::expectation"}:
                priority -= 0.22
            rows.append((pressure + real * 0.22 + attention_gain * 0.55 + priority, label))
        rows.sort(key=lambda pair: (-pair[0], pair[1]))
        labels = []
        for _score, label in rows:
            if label not in labels:
                labels.append(label)
            if len(labels) >= max(1, int(limit)):
                break
        return labels

    def _attention_release_target_labels(self, *, state_snapshot_items: list[dict], attention_trace: dict, limit: int) -> list[str]:
        labels = []
        for item in state_snapshot_items or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            source_type = str(item.get("source_type", "") or "")
            if not label:
                continue
            if source_type == "old_episode_residue" or label.startswith(("old_episode::", "residue::")) or "residue" in label:
                labels.append(label)
        for row in list((attention_trace or {}).get("selected_items", []) or []):
            if not isinstance(row, dict):
                continue
            label = str(row.get("sa_label", "") or "")
            fatigue = float(row.get("fatigue", 0.0) or 0.0)
            if label and fatigue >= 0.18:
                labels.append(label)
        unique = []
        for label in labels:
            if label not in unique:
                unique.append(label)
            if len(unique) >= max(1, int(limit)):
                break
        return unique

    def _dominant_label_family(self, labels: list[str]) -> str:
        counts: dict[str, int] = {}
        for label in labels or []:
            prefix = str(label or "").split("::", 1)[0]
            if not prefix:
                continue
            counts[prefix] = int(counts.get(prefix, 0) or 0) + 1
        if not counts:
            return ""
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    def _is_attention_background_label(self, item: dict) -> bool:
        label = str((item or {}).get("sa_label", "") or "")
        family = str((item or {}).get("family", "") or "")
        source_type = str((item or {}).get("source_type", "") or "")
        if source_type in {"old_episode_residue", "action_control", "action_feedback", "action_selection"}:
            return True
        if family in {"action", "action_control", "action_feedback", "signal"}:
            return True
        if label.startswith(("action::", "action_feedback::", "control::", "old_episode::")):
            return True
        return False

    def _is_semantic_text_anchor(self, label: str) -> bool:
        token = str(label or "").split("::", 1)[-1]
        if not token:
            return False
        if len(token) >= 3:
            return True
        if any(ch.isdigit() for ch in token):
            return True
        return False

    def _visual_gaze_target_context(self, *, state_snapshot_items: list[dict], attention_trace: dict, draft_context: dict | None = None) -> dict:
        """
        Pick a visual object worth looking at from the current cognitive field.

        This is intentionally a generic SA-energy readout, not a detector for a
        specific class or color. AP's gaze should be pulled by surprise, focus
        competition, motion, and low current clarity, the same signals a humanlike
        continuous system already uses elsewhere.
        """

        state_by_label: dict[str, dict] = {}
        for item in state_snapshot_items or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label or not self._is_visual_object_row(item):
                continue
            bbox = self._bbox_norm(item)
            if len(bbox) < 4:
                continue
            state_by_label[label] = item
        if not state_by_label:
            return {"available": False, "reason": "no_visual_object_bbox"}

        attention_rows: dict[str, dict] = {}
        max_focus_score = 0.0
        selected = {str(label or "") for label in list((attention_trace or {}).get("selected_labels", []) or []) if str(label or "")}
        for idx, row in enumerate(list((attention_trace or {}).get("selected_items", []) or []) + list((attention_trace or {}).get("ranked_items", []) or [])):
            if not isinstance(row, dict):
                continue
            label = str(row.get("sa_label", "") or "")
            if not label:
                continue
            current = attention_rows.get(label, {})
            focus_score = max(float(current.get("focus_score", 0.0) or 0.0), float(row.get("focus_score", 0.0) or 0.0))
            max_focus_score = max(max_focus_score, focus_score)
            attention_rows[label] = {
                **current,
                **row,
                "focus_score": focus_score,
                "attention_rank": min(int(current.get("attention_rank", idx) or idx), idx),
                "selected_by_attention": label in selected or bool(current.get("selected_by_attention", False)),
            }

        candidates: list[dict] = []
        for label, state_row in state_by_label.items():
            meta = dict(state_row.get("anchor_meta", {}) or {})
            sampling_focus = dict(meta.get("sampling_focus", {}) or {})
            if not sampling_focus:
                sampling_focus = self._sampling_focus_from_numeric(state_row)
            bbox = self._bbox_norm(state_row)
            attention_row = dict(attention_rows.get(label, {}) or {})
            focus_score = float(attention_row.get("focus_score", 0.0) or 0.0)
            attention_norm = _clamp(focus_score / max(0.18, max_focus_score), 0.0, 1.0) if max_focus_score > 0.0 else 0.0
            selected_bonus = 0.16 if bool(attention_row.get("selected_by_attention", False)) else 0.0
            cp = float(state_row.get("cognitive_pressure", 0.0) or 0.0)
            real = float(state_row.get("real_energy", 0.0) or 0.0)
            virtual = float(state_row.get("virtual_energy", 0.0) or 0.0)
            precision = _clamp(float(sampling_focus.get("precision", 0.24) or 0.24), 0.0, 1.0)
            gain = _clamp(float(sampling_focus.get("gain", 0.0) or 0.0), 0.0, 1.0)
            distance = _clamp(float(sampling_focus.get("distance", 0.0) or 0.0), 0.0, 1.5)
            peripheral_need = _clamp((1.0 - precision) * (0.58 + min(0.42, distance)), 0.0, 1.0)
            motion = self._visual_motion_strength(state_row)
            salience = _clamp(max(real, float(meta.get("salience", 0.0) or 0.0)), 0.0, 1.4) / 1.4
            familiar_expected = bool(meta.get("familiar", False) and meta.get("expected", False))
            if familiar_expected and abs(cp) < 0.05 and motion < 0.05:
                salience *= 0.34
                peripheral_need *= 0.36
                selected_bonus *= 0.20
                attention_norm *= 0.38
            target_key = self._visual_gaze_target_key(state_row)
            fatigue = _clamp(float(self._visual_target_fatigue.get(target_key, self._visual_target_fatigue.get(label, 0.0)) or 0.0), 0.0, 1.0)
            components = {
                "attention": _round4(attention_norm),
                "selected_bonus": _round4(selected_bonus),
                "positive_pressure": _round4(_clamp(max(0.0, cp), 0.0, 1.2) / 1.2),
                "abs_pressure": _round4(_clamp(abs(cp), 0.0, 1.4) / 1.4),
                "salience": _round4(salience),
                "peripheral_need": _round4(peripheral_need),
                "motion": _round4(motion),
                "virtual_hint": _round4(_clamp(virtual, 0.0, 1.2) / 1.2),
                "target_fatigue": _round4(fatigue),
                "familiar_expected_suppression": 1.0 if familiar_expected and abs(cp) < 0.05 and motion < 0.05 else 0.0,
            }
            raw_score = (
                components["attention"] * 0.30
                + selected_bonus
                + components["positive_pressure"] * 0.25
                + components["abs_pressure"] * 0.08
                + salience * 0.15
                + peripheral_need * 0.22
                + motion * 0.10
                + components["virtual_hint"] * 0.05
            )
            # Fatigue is a soft exploration pressure. It lowers the score of a
            # target that has already been sampled clearly, but strong fresh
            # pressure or motion can still make it win again.
            score = raw_score - fatigue * (0.20 + max(0.0, precision - 0.62) * 0.20)
            if familiar_expected and abs(cp) < 0.05 and motion < 0.05:
                score *= 0.46
            unread_pressure = self._next_unread_visual_region_pressure(
                label=label,
                bbox_norm=bbox,
                draft_context=draft_context,
                all_visual_rows=list(state_by_label.values()),
            )
            if unread_pressure:
                components.update(unread_pressure["components"])
                score = _clamp(score + float(unread_pressure.get("score_bonus", 0.0) or 0.0), 0.0, 1.0)
            candidates.append(
                {
                    "available": True,
                    "sa_label": label,
                    "x": _clamp(float(bbox[0]), 0.0, 1.0),
                    "y": _clamp(float(bbox[1]), 0.0, 1.0),
                    "bbox_norm": bbox,
                    "score": _round4(_clamp(score, 0.0, 1.0)),
                    "focus_score": _round4(focus_score),
                    "focus_gain": _round4(gain),
                    "focus_precision": _round4(precision),
                    "distance": _round4(distance),
                    "peripheral_need": _round4(peripheral_need),
                    "target_fatigue": _round4(fatigue),
                    "raw_score_before_fatigue": _round4(_clamp(raw_score, 0.0, 1.0)),
                    "current_gaze_x": _round4(float(meta.get("gaze_center_x", 0.5) or 0.5)),
                    "current_gaze_y": _round4(float(meta.get("gaze_center_y", 0.5) or 0.5)),
                    "gaze_target_key": target_key,
                    "score_components": components,
                    "reason": str(unread_pressure.get("reason", "") or "visual_attention_pressure_target") if unread_pressure else "visual_attention_pressure_target",
                }
            )
        if not candidates:
            return {"available": False, "reason": "no_scored_visual_target"}
        candidates = self._aggregate_visual_gaze_targets(candidates)
        candidates.sort(key=lambda row: (-float(row.get("score", 0.0) or 0.0), -float(row.get("focus_score", 0.0) or 0.0), str(row.get("gaze_target_key", "") or str(row.get("sa_label", "") or ""))))
        best = dict(candidates[0])
        best["alternatives"] = candidates[1:4]
        return best

    def _aggregate_visual_gaze_targets(self, candidates: list[dict]) -> list[dict]:
        """
        Collapse visual SA channels into executable gaze targets.

        The state pool keeps object, color, shape, spatial and motion rows as
        first-class SA because AP should recognize the whole field. The eye,
        however, can only move toward a place. This aggregation happens only at
        the actuator-parameter layer: feature rows still contribute pressure,
        motion and salience evidence, but they no longer multiply one object
        into several competing "places to look at".
        """

        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in candidates or []:
            if not isinstance(row, dict):
                continue
            key = str(row.get("gaze_target_key", "") or row.get("sa_label", "") or "")
            if not key:
                continue
            grouped[key].append(row)

        merged: list[dict] = []
        for key, rows in grouped.items():
            ordered = sorted(
                rows,
                key=lambda item: (
                    0 if str(item.get("sa_label", "") or "").startswith("vision_obj::") else 1,
                    -float(item.get("score", 0.0) or 0.0),
                    str(item.get("sa_label", "") or ""),
                ),
            )
            display_row = dict(ordered[0])
            strongest = max(rows, key=lambda item: float(item.get("score", 0.0) or 0.0))
            component_keys = {
                comp_key
                for row in rows
                for comp_key in dict(row.get("score_components", {}) or {}).keys()
            }
            components = {
                comp_key: _round4(
                    max(
                        float(dict(row.get("score_components", {}) or {}).get(comp_key, 0.0) or 0.0)
                        for row in rows
                    )
                )
                for comp_key in component_keys
            }
            merged_score = max(float(row.get("score", 0.0) or 0.0) for row in rows)
            merged_raw = max(float(row.get("raw_score_before_fatigue", 0.0) or 0.0) for row in rows)
            display_row.update(
                {
                    "available": True,
                    "sa_label": str(display_row.get("sa_label", "") or key),
                    "gaze_target_key": key,
                    "score": _round4(_clamp(merged_score, 0.0, 1.0)),
                    "raw_score_before_fatigue": _round4(_clamp(merged_raw, 0.0, 1.0)),
                    "focus_score": _round4(max(float(row.get("focus_score", 0.0) or 0.0) for row in rows)),
                    "focus_gain": _round4(max(float(row.get("focus_gain", 0.0) or 0.0) for row in rows)),
                    "focus_precision": _round4(max(float(row.get("focus_precision", 0.0) or 0.0) for row in rows)),
                    "peripheral_need": _round4(max(float(row.get("peripheral_need", 0.0) or 0.0) for row in rows)),
                    "target_fatigue": _round4(max(float(row.get("target_fatigue", 0.0) or 0.0) for row in rows)),
                    "score_components": components,
                    "source_labels": sorted({str(row.get("sa_label", "") or "") for row in rows if str(row.get("sa_label", "") or "")}),
                    "merged_source_count": len(rows),
                    "strongest_source_label": str(strongest.get("sa_label", "") or ""),
                    "reason": "visual_attention_pressure_target_group",
                }
            )
            merged.append(display_row)
        return merged

    def _next_unread_visual_region_pressure(
        self,
        *,
        label: str,
        bbox_norm: list[float],
        draft_context: dict | None,
        all_visual_rows: list[dict],
    ) -> dict:
        """
        Softly pull gaze toward the next unread visual region during charwise UI
        reading.

        This is a V2 process anchor, not a hidden OCR/order solver. It only uses
        AP-visible draft length plus spatial visual-object bboxes. Character
        labels, teacher references, and text answers are not inspected.
        """

        if not bbox_norm:
            return {}
        draft = dict(draft_context or {})
        try:
            visible_length = max(0, int(draft.get("visible_length", 0) or 0))
        except (TypeError, ValueError):
            visible_length = 0
        glyph_rows: list[tuple[float, str, dict]] = []
        for row in all_visual_rows or []:
            if not isinstance(row, dict):
                continue
            row_label = str(row.get("sa_label", "") or "")
            row_meta = dict(row.get("anchor_meta", {}) or {})
            row_bbox = self._bbox_norm(row)
            if len(row_bbox) < 4:
                continue
            is_glyph_like = (
                "glyph_slice" in row_label
                or "glyph_slice" in str(row_meta.get("object_anchor_id", "") or "")
                or str(row_meta.get("proposal_kind", "") or "") == "glyph_slice"
            )
            if not is_glyph_like:
                continue
            glyph_rows.append((float(row_bbox[0]), row_label, row))
        if len(glyph_rows) < 2:
            return {}
        glyph_rows.sort(key=lambda item: (item[0], item[1]))
        rank_by_label = {row_label: index for index, (_, row_label, _) in enumerate(glyph_rows)}
        if label not in rank_by_label:
            return {}
        rank = int(rank_by_label[label])
        target_rank = min(visible_length, len(glyph_rows) - 1)
        distance = abs(rank - target_rank)
        if distance == 0:
            base = 0.34 if visible_length <= 0 else 0.28
            reason = "empty_draft_leftmost_region_alignment" if visible_length <= 0 else "draft_length_aligned_unread_region"
            return {
                "score_bonus": _round4(base),
                "reason": "visual_attention_pressure_target_next_unread_region",
                "components": {
                    "next_unread_region_pressure": _round4(base),
                    "next_unread_target_rank": float(target_rank),
                    "next_unread_visual_rank": float(rank),
                    reason: 1.0,
                },
            }
        cost = _round4(max(0.0, 0.055 / (1.0 + distance * 1.8)))
        return {
            "score_bonus": cost,
            "reason": "visual_attention_pressure_target_non_target_unread_region",
            "components": {
                "next_unread_region_pressure": cost,
                "next_unread_target_rank": float(target_rank),
                "next_unread_visual_rank": float(rank),
                "next_unread_rank_distance": float(distance),
            },
        }

    def _visual_gaze_target_key(self, item: dict) -> str:
        """
        Resolve the spatial object that a gaze action would actually look at.

        The state field keeps visual objects and their shape/color/motion
        channels as first-class SA. For eye movement, however, those channels
        refer to the same place in the world. Fatigue and parameter learning
        therefore use this object/space key so one already-sampled object cannot
        bypass exploration pressure by reappearing as a different feature
        channel label.
        """

        label = str((item or {}).get("sa_label", "") or "")
        meta = dict((item or {}).get("anchor_meta", {}) or {})
        for key in ("object_anchor_id", "parent_object_label"):
            value = str(meta.get(key, "") or "")
            if value:
                return value
        if str((item or {}).get("family", "") or "") == "vision_object":
            return label
        bbox = self._bbox_norm(item)
        if len(bbox) >= 2:
            return self._visual_spatial_target_key(bbox)
        return label

    def _visual_spatial_target_key(self, bbox_norm: list[float]) -> str:
        x = float((bbox_norm or [0.5])[0] if bbox_norm else 0.5)
        y = float((bbox_norm or [0.5, 0.5])[1] if len(bbox_norm or []) > 1 else 0.5)
        if x < 0.34:
            x_bucket = "left"
        elif x > 0.66:
            x_bucket = "right"
        else:
            x_bucket = "center"
        if y < 0.34:
            y_bucket = "upper"
        elif y > 0.66:
            y_bucket = "lower"
        else:
            y_bucket = "mid"
        return f"vision_obj::{x_bucket}_{y_bucket}"

    def _is_visual_object_row(self, item: dict) -> bool:
        label = str((item or {}).get("sa_label", "") or "")
        family = str((item or {}).get("family", "") or "")
        source_type = str((item or {}).get("source_type", "") or "")
        return family == "vision_object" or label.startswith("vision_obj::") or (family.startswith("vision") and source_type == "vision_numeric")

    def _bbox_norm(self, item: dict) -> list[float]:
        meta = dict((item or {}).get("anchor_meta", {}) or {})
        bbox = list(meta.get("bbox_norm", []) or [])
        if len(bbox) >= 4:
            return [_round4(_clamp(float(value or 0.0), 0.0, 1.0)) for value in bbox[:4]]
        numeric = dict((item or {}).get("numeric_features", {}) or {})
        spatial = list(numeric.get("vision.spatial", []) or [])
        if len(spatial) >= 4:
            return [_round4(_clamp(float(value or 0.0), 0.0, 1.0)) for value in spatial[:4]]
        return []

    def _sampling_focus_from_numeric(self, item: dict) -> dict:
        numeric = dict((item or {}).get("numeric_features", {}) or {})
        values = list(numeric.get("vision.focus", []) or [])
        if len(values) < 2:
            return {}
        return {
            "precision": _clamp(float(values[0] or 0.0), 0.0, 1.0),
            "distance": _clamp(float(values[1] or 0.0), 0.0, 1.5),
            "gain": _clamp(max(0.0, float(values[0] or 0.0) - 0.24) / 0.76, 0.0, 1.0),
        }

    def _visual_motion_strength(self, item: dict) -> float:
        numeric = dict((item or {}).get("numeric_features", {}) or {})
        for key in ("vision.motion_vector", "vision.motion"):
            values = numeric.get(key, [])
            if isinstance(values, (list, tuple)) and values:
                try:
                    return _clamp(sum(abs(float(value or 0.0)) for value in values[:3]), 0.0, 1.0)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def _output_mismatch_context(self, state_snapshot_items: list[dict]) -> dict:
        mismatch_rows = []
        revision_rows = []
        reread_rows = []
        for item in state_snapshot_items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            source_type = str(item.get("source_type", "") or "")
            family = str(item.get("family", "") or "")
            anchor_meta = dict(item.get("anchor_meta", {}) or {})
            event_type = str(anchor_meta.get("event_type", "") or "")
            if source_type != "text_action" and family != "text_action" and not label.startswith("text_action::"):
                continue
            if label.startswith("text_action::revise::") or event_type in {"revise", "write_revision"}:
                revision_rows.append(item)
                continue
            if label.startswith("text_action::reread::") or event_type == "reread":
                reread_rows.append(item)
                continue
            if label.startswith("text_action::write::") and self._is_feedback_confirmed_text_mismatch(anchor_meta):
                mismatch_rows.append(item)
        unresolved_count = max(0, len(mismatch_rows) - len(revision_rows))
        latest_mismatch = self._latest_text_action_row(mismatch_rows)
        latest_reread = self._latest_text_action_row(reread_rows)
        mismatch_meta = dict((latest_mismatch or {}).get("anchor_meta", {}) or {})
        reread_meta = dict((latest_reread or {}).get("anchor_meta", {}) or {})
        latest_mismatch_tick = int(mismatch_meta.get("tick_index", -1) or -1) if latest_mismatch else -1
        latest_reread_tick = int(reread_meta.get("tick_index", -1) or -1) if latest_reread else -1
        latest_reference_token = self._feedback_reference_token(mismatch_meta)
        return {
            "mismatch_count": len(mismatch_rows),
            "revision_count": len(revision_rows),
            "reread_count": len(reread_rows),
            "unresolved_count": unresolved_count,
            "correction_pressure": _clamp(unresolved_count / 2.0, 0.0, 1.0),
            "latest_mismatch_token": str(mismatch_meta.get("token", "") or ""),
            "latest_expected_token": str(latest_reference_token or mismatch_meta.get("expected_token", "") or ""),
            "latest_mismatch_tick": latest_mismatch_tick,
            "latest_reread_tick": latest_reread_tick,
            "reread_after_mismatch": latest_mismatch_tick >= 0 and latest_reread_tick >= latest_mismatch_tick,
        }

    def _latest_text_action_row(self, rows: list[dict]) -> dict:
        if not rows:
            return {}
        def _tick(row: dict) -> int:
            meta = dict((row or {}).get("anchor_meta", {}) or {})
            try:
                return int(meta.get("tick_index", -1) or -1)
            except (TypeError, ValueError):
                return -1

        return dict(max([row for row in rows if isinstance(row, dict)], key=_tick, default={}))

    def _expected_text_context(
        self,
        *,
        fast_cn: list[dict],
        slow_cn: list[dict],
        fast_bn: list[dict] | None = None,
        slow_bn: list[dict] | None = None,
        draft_context: dict | None = None,
    ) -> dict:
        scores: dict[str, float] = defaultdict(float)
        sources: dict[str, set[str]] = defaultdict(set)
        position_notes_by_token: dict[str, set[str]] = defaultdict(set)
        draft = dict(draft_context or {})
        visible_length = int(draft.get("visible_length", 0) or 0)
        has_visible_text = bool(visible_length > 0)
        empty_draft_start_pressure = bool(
            not has_visible_text
            and (
                str(draft.get("process_anchor_role", "") or "") == "empty_draft_start_readout_context"
                or "empty_draft_state_planning_context" in {str(note or "") for note in list(draft.get("notes", []) or [])}
            )
        )
        visible_tokens = [
            str(token or "")
            for token in list(draft.get("visible_tokens", []) or [])
            if str(token or "")
        ]
        visible_text = str(draft.get("visible_text", "") or "")
        closed_tokens = set(visible_tokens)
        if not visible_tokens:
            visible_tokens = list(visible_text) if visible_text else []
            closed_tokens = set(visible_tokens)
        def _branch_text_labels(branch: dict | None) -> set[str]:
            labels: set[str] = set()
            if not isinstance(branch, dict):
                return labels
            preview = dict((branch or {}).get("snapshot_preview", {}) or {})
            for label_value in list(preview.get("focus_labels", []) or []) + list(preview.get("labels", []) or []):
                label = str(label_value or "")
                if label:
                    labels.add(label)
            for item in list((branch or {}).get("predicted_items", []) or []):
                label = str((item or {}).get("sa_label", "") or "")
                if label:
                    labels.add(label)
            return labels

        branch_text_labels: set[str] = set()
        for branch in list(fast_cn or []) + list(slow_cn or []):
            branch_text_labels.update(_branch_text_labels(branch))
        math_process_mode = bool(
            any(
                label.startswith(
                    (
                        "knowledge_atom::basic_math_science",
                        "dialogue_process::quantity_relation",
                        "operation::arithmetic_candidate",
                        "short_term_slot::calculation_task",
                        "goal::reread_before_try",
                        "math_paradigm::",
                        "math_trace::",
                        "math_word_problem::",
                    )
                )
                for label in branch_text_labels
            )
        )
        math_process_tokens = {
            "先列式": 1.18,
            "列式": 1.06,
            "列竖式": 1.10,
            "=": 1.22,
            "×": 1.18,
            "÷": 1.18,
            "乘回": 1.08,
            "试商": 1.06,
            "余数": 1.08,
            "算": 0.90,
        }

        def _position_meta(item: dict, branch: dict | None = None) -> dict:
            meta = dict((item or {}).get("anchor_meta", {}) or {})
            for key in ("source_type", "family", "sa_kind"):
                if key in (item or {}) and key not in meta:
                    meta[key] = (item or {}).get(key)
            if meta:
                return meta
            snap = dict((branch or {}).get("snapshot", {}) or {})
            for snap_item in list(snap.get("items", []) or []):
                if not isinstance(snap_item, dict):
                    continue
                if str(snap_item.get("sa_label", "") or "") == str((item or {}).get("sa_label", "") or ""):
                    meta = dict(snap_item.get("anchor_meta", {}) or {})
                    for key in ("source_type", "family", "sa_kind"):
                        if key in snap_item and key not in meta:
                            meta[key] = snap_item.get(key)
                    return meta
            return {}

        def _is_negative_text_prediction(meta: dict) -> bool:
            outcome = str(meta.get("feedback_outcome", "") or "")
            if outcome == "punished":
                return True
            try:
                punishment = float(meta.get("feedback_punishment", 0.0) or 0.0)
                reward = float(meta.get("feedback_reward", 0.0) or 0.0)
                correctness = float(meta.get("feedback_correctness", 0.0) or 0.0)
            except (TypeError, ValueError):
                return False
            return bool(punishment > max(reward, correctness) and punishment >= 0.18)

        def _has_output_process_evidence(meta: dict) -> bool:
            schema_id = str(meta.get("schema_id", "") or "")
            source = str(meta.get("source", "") or "")
            source_event_type = str(meta.get("source_event_type", "") or meta.get("event_type", "") or "")
            readout_role = str(meta.get("readout_semantic_role", "") or "")
            priority = str(meta.get("prediction_payload_priority", "") or "")
            return bool(
                bool(meta.get("self_generated", False))
                or schema_id in {
                    "gl_successful_skill_char_token/v1",
                    "text_visible_draft_token/v1",
                    "text_revision_opportunity/v1",
                    "text_slot_confirmation/v1",
                    "text_character_binding/v1",
                }
                or readout_role == "reply_char_slot"
                or source in {"action::text_insert", "action::text_reread", "text_actuator_direct_replace"}
                or source_event_type in {"draft_read_token", "insert", "replace", "write_revision", "visible_draft_token"}
                or priority.startswith(("current_glyph", "previous_prefix"))
            )

        def _is_external_input_text_meta(meta: dict) -> bool:
            source_type = str(meta.get("source_type", "") or "")
            source = str(meta.get("source", "") or "")
            notes = {str(note or "") for note in list(meta.get("notes", []) or [])}
            return bool(
                source_type in {"external_text", "external_text_readback", "external_teacher"}
                or source in {"external_text", "external_text_turn"}
                or "external_text_read_into_input_channel" in notes
                or "not_ap_visible_draft" in notes
            )

        def _position_weight(meta: dict, token: str) -> tuple[float, list[str]]:
            notes: list[str] = []
            try:
                raw_glyph_index = meta.get("current_glyph_index")
                if raw_glyph_index is None:
                    raw_glyph_index = meta.get("position")
                if raw_glyph_index is None:
                    raw_glyph_index = meta.get("cursor_before")
                if raw_glyph_index is None:
                    raw_glyph_index = meta.get("visible_length", -1)
                glyph_index = int(raw_glyph_index)
            except (TypeError, ValueError):
                glyph_index = -1
            role = str(meta.get("current_glyph_role", "") or "")
            source = str(meta.get("source", "") or "")
            source_event_type = str(meta.get("source_event_type", "") or meta.get("event_type", "") or "")
            self_generated = bool(meta.get("self_generated", False))
            priority = str(meta.get("prediction_payload_priority", "") or "")
            readout_role = str(meta.get("readout_semantic_role", "") or "")
            readout_pattern = str(meta.get("readout_pattern_id", "") or "")
            semantic_frame_role = str(meta.get("semantic_frame_role", "") or "")
            dynamic_slot_role = str(meta.get("dynamic_slot_role", "") or "")
            variant_text = str(meta.get("variant_text", "") or meta.get("expected_text", "") or "")
            previous_prefix = str(meta.get("previous_prefix", "") or "")
            has_readout_frame = bool(readout_role or readout_pattern or semantic_frame_role)
            output_process_evidence = _has_output_process_evidence(meta)
            internal_output_position = bool(
                self_generated
                or source in {"action::text_insert", "action::text_reread", "text_actuator_direct_replace"}
                or source_event_type in {"draft_read_token", "insert", "replace", "write_revision"}
            )
            if _is_external_input_text_meta(meta) and not output_process_evidence:
                notes.append("external_input_text_not_output_candidate")
                return 0.0, notes
            if glyph_index < 0 and not priority and not role and not has_readout_frame:
                # A text token recalled without position/process metadata is a
                # background familiarity signal for recognizing the current Bn,
                # not an output-side motor candidate. Direct draft writing needs
                # cursor/readout/process metadata; otherwise user-input glyphs
                # like "事" or "。" can become fragment replies.
                notes.append("unpositioned_text_memory_not_output_candidate")
                return 0.0, notes
            current_visible_text = str(visible_text or "")
            if previous_prefix and previous_prefix != current_visible_text:
                notes.append("previous_prefix_mismatch_suppressed")
                return 0.0, notes
            if variant_text:
                if current_visible_text:
                    if not variant_text.startswith(current_visible_text):
                        notes.append("variant_prefix_mismatch_suppressed")
                        return 0.0, notes
                    remaining = variant_text[len(current_visible_text) :]
                    if glyph_index == visible_length and remaining and not remaining.startswith(token):
                        notes.append("variant_next_token_mismatch_suppressed")
                        return 0.0, notes
                elif glyph_index == 0 and not variant_text.startswith(token):
                    notes.append("variant_start_token_mismatch_suppressed")
                    return 0.0, notes
            distance = abs(int(glyph_index) - int(visible_length)) if glyph_index >= 0 else 2
            if not has_visible_text:
                if glyph_index == 0:
                    notes.append("empty_draft_start_region_alignment")
                    if internal_output_position:
                        notes.append("internal_output_start_position")
                        return 2.35, notes
                    if empty_draft_start_pressure and has_readout_frame:
                        notes.append("empty_draft_readout_start_process_anchor")
                        return 2.35, notes
                    return 1.85, notes
                if glyph_index > 0:
                    notes.append("empty_draft_later_region_soft_cost")
                    cost = max(0.025, 0.105 / (1.0 + glyph_index * 1.35))
                    if empty_draft_start_pressure and has_readout_frame:
                        notes.append("empty_draft_readout_start_later_slot_suppression")
                        cost *= 0.42
                    return cost, notes
            if glyph_index == visible_length:
                notes.append("cursor_aligned_next_unread_region")
                if token in closed_tokens and visible_length >= 1:
                    notes.append("repeated_token_allowed_by_current_unread_region")
                weight = 1.68
                if has_readout_frame:
                    notes.append("whole_region_time_readout_frame_alignment")
                    weight += 0.22
                    if dynamic_slot_role == "dynamic_numeric_slot" and str(token).isdigit():
                        notes.append("dynamic_numeric_slot_confirmed_by_local_focus")
                        weight += 0.18
                    elif readout_role == "time_readout_region_slot":
                        notes.append("time_readout_slot_context")
                        weight += 0.08
                return min(2.15, weight), notes
            if token in closed_tokens and visible_length >= 1:
                notes.append("closed_visible_token_soft_cost")
                return 0.055 if visible_length >= 3 else 0.09, notes
            if glyph_index >= 0:
                notes.append("cursor_distance_soft_cost")
                return max(0.055, 0.24 / (1.0 + distance * 1.25)), notes
            return 1.0, notes

        # Slow focus predictions are the inner draft thread, so they get a tiny
        # preference, but not enough to override a clearly stronger fast-field
        # prediction. The result is a distribution summary, not a hard decision.
        for source, branches, source_weight in (("slow_cn", slow_cn, 1.08), ("fast_cn", fast_cn, 1.0)):
            for branch in branches or []:
                for item in list((branch or {}).get("predicted_items", []) or []):
                    label = str((item or {}).get("sa_label", "") or "")
                    if not label.startswith("text::"):
                        continue
                    token = label.split("::", 1)[-1]
                    if not token:
                        continue
                    strength = _clamp(float((item or {}).get("virtual_energy", 0.2) or 0.2) * source_weight, 0.0, 1.2)
                    if strength <= 0.0:
                        continue
                    meta = _position_meta(item, branch)
                    if math_process_mode:
                        math_bonus = 0.0
                        if token in math_process_tokens:
                            math_bonus = math_process_tokens[token]
                            if not has_visible_text and token in {"先列式", "列式", "列竖式"}:
                                math_bonus += 0.24
                            if token in {"=", "×", "÷"} and visible_length >= 1:
                                math_bonus += 0.18
                            if token in {"乘回", "试商", "余数"}:
                                math_bonus += 0.10
                        elif any(char in token for char in ("先", "列", "式", "算", "乘", "除", "余")):
                            math_bonus = 0.24
                        if math_bonus > 0.0:
                            strength += math_bonus
                            sources[token].add("math_visible_process_priority")
                            position_notes_by_token[token].add("math_visible_process_priority")
                    if _is_negative_text_prediction(meta):
                        sources[token].add("negative_feedback_text_prediction_suppressed")
                        position_notes_by_token[token].add("negative_feedback_text_prediction_suppressed")
                        continue
                    position_weight, position_notes = _position_weight(meta, token)
                    strength *= position_weight
                    if strength <= 0.0:
                        continue
                    scores[token] += strength
                    sources[token].add(source)
                    for note in position_notes[:2]:
                        sources[token].add(note)
                        position_notes_by_token[token].add(note)
        for source, branches, source_weight in (("slow_bn", slow_bn or [], 0.42), ("fast_bn", fast_bn or [], 0.36)):
            for branch in branches or []:
                preview = dict((branch or {}).get("snapshot_preview", {}) or {})
                labels = list(preview.get("focus_labels", []) or []) + list(preview.get("labels", []) or [])
                seen_labels: set[str] = set()
                for label_value in labels:
                    label = str(label_value or "")
                    if label in seen_labels or not label.startswith("text::"):
                        continue
                    seen_labels.add(label)
                    token = label.split("::", 1)[-1]
                    if not token:
                        continue
                    try:
                        score = float((branch or {}).get("normalized_weight", 0.0) or 0.0)
                        score += float((branch or {}).get("match_efficiency", 0.0) or 0.0) * 0.18
                    except (TypeError, ValueError):
                        score = 0.0
                    # Bn says "the current field resembles this memory"; Cn is
                    # the channel that proposes what should happen next.
                    if token not in scores:
                        continue
                    strength = _clamp(score * source_weight, 0.0, 0.55)
                    if strength <= 0.0:
                        continue
                    meta = _position_meta({"sa_label": label}, branch)
                    position_weight, position_notes = _position_weight(meta, token)
                    strength *= position_weight
                    if strength <= 0.0:
                        continue
                    scores[token] += strength
                    sources[token].add(source)
                    for note in position_notes[:2]:
                        sources[token].add(note)
                        position_notes_by_token[token].add(note)
        ranked = sorted(scores.items(), key=lambda item: (-float(item[1]), item[0]))
        if not ranked:
            return {
                "token": "",
                "strength": 0.0,
                "source": "",
                "alternatives": [],
                "candidate_count": 0,
                "top_share": 0.0,
                "dominance_gap": 0.0,
                "dominance_ratio": 0.0,
                "ambiguity": 0.0,
                "decisive": False,
            }
        total = max(1e-9, sum(score for _, score in ranked))
        top_token, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        top_share = _clamp(top_score / total, 0.0, 1.0)
        dominance_gap = _clamp(top_score - second_score, 0.0, 1.0)
        dominance_ratio = top_score / max(0.001, second_score)
        candidate_count = len(ranked)
        ambiguity = _clamp(
            (1.0 - top_share) * 0.55
            + max(0.0, 0.22 - dominance_gap) * 1.35
            + min(0.18, max(0, candidate_count - 2) * 0.035),
            0.0,
            1.0,
        )
        top_position_notes = position_notes_by_token.get(top_token, set())
        top_cursor_aligned = bool("cursor_aligned_next_unread_region" in top_position_notes)
        # Momentary state-pool refresh keeps process amplitudes bounded, so an
        # explicit cursor-aligned successor may have low absolute energy while
        # still being the clear next continuation. Treat that as decisive by
        # relative confidence; this preserves Cn semantics instead of forcing
        # process states to accumulate just to cross an absolute threshold.
        enough_strength = bool(
            top_score >= 0.18
            or (
                top_cursor_aligned
                and top_score > 0.0
                and (top_share >= 0.72 or dominance_ratio >= 1.75)
            )
        )
        decisive = bool(enough_strength and (candidate_count == 1 or top_share >= 0.55 or dominance_gap >= 0.22 or dominance_ratio >= 1.75))
        return {
            "token": top_token,
            "strength": _round4(_clamp(top_score, 0.0, 1.2)),
            "source": "+".join(sorted(sources[top_token])),
            "alternatives": [
                {
                    "token": token,
                    "score": _round4(score),
                    "share": _round4(score / total),
                    "sources": sorted(sources[token]),
                    "position_notes": sorted(position_notes_by_token.get(token, set())),
                }
                for token, score in ranked[:8]
            ],
            "candidate_count": int(candidate_count),
            "top_share": _round4(top_share),
            "dominance_gap": _round4(dominance_gap),
            "dominance_ratio": _round4(min(99.0, dominance_ratio)),
            "ambiguity": _round4(ambiguity),
            "decisive": decisive,
        }

    def _advance_expected_text_after_visible_closure(self, expected_text: dict, draft_context: dict) -> dict:
        """
        Prefer the next unresolved text candidate after AP has reread its own
        visible draft.

        This is a process-grounded continuation bias, not an answer table: it
        only uses AP's current predicted token distribution plus its own draft
        state. If the top prediction is the token AP just inserted/reread and
        there is another live candidate, the second candidate may become the
        write target so the text editor can continue instead of looping on
        reread forever.
        """

        row = dict(expected_text or {})
        alternatives = [dict(item) for item in list(row.get("alternatives", []) or []) if isinstance(item, dict)]
        if len(alternatives) < 2:
            return row
        visible_length = int(draft_context.get("visible_length", 0) or 0)
        if visible_length <= 0:
            return row
        last_visible_token = str(draft_context.get("last_visible_token", "") or "")
        top_token = str(row.get("token", "") or "")
        if not last_visible_token:
            return row
        last_insert_tick = int(draft_context.get("last_insert_tick", -1) or -1)
        last_reread_tick = int(draft_context.get("last_reread_tick", -1) or -1)
        last_commit_tick = int(draft_context.get("last_commit_tick", -1) or -1)
        if last_insert_tick < 0 or max(last_reread_tick, last_commit_tick) < last_insert_tick:
            return row
        visible_tokens = [
            str(token or "")
            for token in list(draft_context.get("visible_tokens", []) or [])
            if str(token or "")
        ]
        if not visible_tokens:
            visible_text = str(draft_context.get("visible_text", "") or "")
            visible_tokens = list(visible_text) if visible_text else [last_visible_token]
        closed_tokens = set(visible_tokens)
        def _is_cursor_aligned(item: dict) -> bool:
            return "cursor_aligned_next_unread_region" in {
                str(note or "") for note in list(item.get("position_notes", []) or [])
            }

        top_cursor_aligned = _is_cursor_aligned(alternatives[0])
        if top_cursor_aligned:
            # A repeated glyph can still be the correct next character in a
            # charwise trace (for example `12...12`). Cursor-aligned successor
            # evidence means "this instance is next", so do not treat the same
            # glyph elsewhere in the visible draft as already closed.
            return row

        top_already_closed = bool(top_token in closed_tokens)
        if top_token != last_visible_token and not top_already_closed:
            return row

        cursor_aligned_candidates = [
            item
            for item in alternatives[1:]
            if str(item.get("token", "") or "") and _is_cursor_aligned(item)
        ]
        if cursor_aligned_candidates:
            # Continuation is a body/process decision: after AP rereads its own
            # draft, the strongest next candidate should be the one whose
            # remembered process metadata says "this was the next unread
            # region at the current cursor". This is not a string answer table;
            # it only uses live candidate metadata plus the visible draft
            # length.
            next_row = cursor_aligned_candidates[0]
        elif top_already_closed and visible_length >= 2:
            next_row = next(
                (
                    item
                    for item in alternatives[1:]
                    if str(item.get("token", "") or "") and str(item.get("token", "") or "") not in closed_tokens
                ),
                {},
            )
        else:
            next_row = next((item for item in alternatives[1:] if str(item.get("token", "") or "") != last_visible_token), {})
        next_token = str(next_row.get("token", "") or "")
        if not next_token:
            return row
        next_score = float(next_row.get("score", 0.0) or 0.0)
        top_score = float(alternatives[0].get("score", row.get("strength", 0.0)) or 0.0)
        next_share = float(next_row.get("share", 0.0) or 0.0)
        candidate_count = int(row.get("candidate_count", len(alternatives)) or len(alternatives))
        # Do not jump to truly absent alternatives. Once AP has inserted and
        # reread the current token, a low-share but live successor is enough to
        # explore continuation; otherwise early glyphs with repeated feedback
        # can monopolize the distribution forever.
        floor = 0.012
        if candidate_count <= 1:
            floor = max(floor, top_score * 0.16)
        elif visible_length >= 1:
            floor = max(floor, min(0.08, top_score * 0.025))
        else:
            floor = max(floor, top_score * 0.10)
        if next_score < floor and next_share < 0.015:
            return row
        shifted = dict(row)
        shifted["token"] = next_token
        shifted["strength"] = _round4(_clamp(next_score, 0.0, 1.2))
        shifted["source"] = "+".join(list(next_row.get("sources", []) or [])) or str(row.get("source", "") or "")
        shifted["continuation_shift"] = {
            "schema_id": "visible_draft_closure_continuation_shift/v1",
            "from_token": top_token,
            "to_token": next_token,
            "visible_length": int(visible_length),
            "last_reread_tick": int(last_reread_tick),
            "last_insert_tick": int(last_insert_tick),
            "candidate_count": int(candidate_count),
            "next_share": _round4(next_share),
            "floor": _round4(floor),
            "closed_tokens": visible_tokens[:8],
            "top_already_closed": bool(top_already_closed),
            "cursor_aligned_shift": bool(_is_cursor_aligned(next_row)),
            "policy": "after_self_reread_closed_current_token_try_next_live_candidate_without_teacher_answer_lookup",
        }
        return shifted

    def expected_text_context(
        self,
        *,
        fast_cn: list[dict],
        slow_cn: list[dict],
        fast_bn: list[dict] | None = None,
        slow_bn: list[dict] | None = None,
        draft_context: dict | None = None,
    ) -> dict:
        """
        Read-only public bridge for runtime feeling channels.

        TaskFeeling reuses the same successor clarity that action planning uses,
        avoiding a second, conflicting definition of "can continue writing".
        """

        return self._expected_text_context(
            fast_bn=fast_bn,
            slow_bn=slow_bn,
            fast_cn=fast_cn,
            slow_cn=slow_cn,
            draft_context=draft_context,
        )

    def draft_writing_context(self, state_snapshot_items: list[dict], *, current_tick: int) -> dict:
        """
        Read-only bridge for teaching scaffolds and reports.

        Skill scaffolds must see the same draft surface that the planner sees.
        Exposing this compact context prevents a second, subtly different
        "draft state" parser from growing outside the action system.
        """

        return self._draft_writing_context(state_snapshot_items, current_tick=int(current_tick))

    def draft_commit_readiness_context(
        self,
        state_snapshot_items: list[dict],
        *,
        current_tick: int,
        fast_cn: list[dict] | None = None,
        slow_cn: list[dict] | None = None,
        correctness: float = 0.0,
        grasp: float = 0.0,
        pressure: float = 0.0,
        dissonance: float = 0.0,
        uncertainty: float = 0.0,
        pressure_anchor_level: float = 0.0,
        expectation_gap: float = 0.0,
    ) -> dict:
        """
        Read-only bridge for short-lived commit readiness state.

        This exposes the same draft appraisal used by the planner, but as a
        compact process-state object that runtime can write into the state pool
        and let the next tick learn from. It does not force submission.
        """

        draft_context = self._draft_writing_context(state_snapshot_items, current_tick=int(current_tick))
        expected_text = self._expected_text_context(
            fast_bn=[],
            slow_bn=[],
            fast_cn=list(fast_cn or []),
            slow_cn=list(slow_cn or []),
            draft_context=draft_context,
        )
        draft_eval = self._draft_self_evaluation(
            draft_context,
            expected_text,
            correctness=float(correctness),
            grasp=float(grasp),
            pressure=float(pressure),
            dissonance=float(dissonance),
            uncertainty=float(uncertainty),
        )
        draft_goal_alignment = self._draft_goal_alignment(
            state_snapshot_items=state_snapshot_items,
            draft_context=draft_context,
            fast_cn=list(fast_cn or []),
            slow_cn=list(slow_cn or []),
            consequence_estimates={},
            outcome_estimate=self._outcome_memory.estimate("action::text_commit"),
        )
        field = self._draft_satisfaction_field(
            draft_eval=draft_eval,
            draft_goal_alignment=draft_goal_alignment,
            correctness=float(correctness),
            grasp=float(grasp),
            pressure=float(pressure),
            dissonance=float(dissonance),
            uncertainty=float(uncertainty),
            pressure_anchor_level=float(pressure_anchor_level),
            expectation_gap=float(expectation_gap),
        )
        return {
            "schema_id": "text_commit_readiness_context/v1",
            "visible_text": str(draft_context.get("visible_text", "") or ""),
            "visible_length": int(draft_context.get("visible_length", 0) or 0),
            "last_reread_age": int(draft_context.get("last_reread_age", 9999) or 9999),
            "last_insert_age": int(draft_context.get("last_insert_age", 9999) or 9999),
            "last_delete_age": int(draft_context.get("last_delete_age", 9999) or 9999),
            "last_replace_age": int(draft_context.get("last_replace_age", 9999) or 9999),
            "last_commit_age": int(draft_context.get("last_commit_age", 9999) or 9999),
            "trailing_repeat_count": int(draft_context.get("trailing_repeat_count", 0) or 0),
            "duplicate_ratio": float(draft_context.get("duplicate_ratio", 0.0) or 0.0),
            "draft_eval": dict(draft_eval),
            "goal_alignment": dict(draft_goal_alignment),
            "satisfaction_field": dict(field),
            "commit_readiness": _clamp(
                float(field.get("satisfaction", 0.0) or 0.0)
                + float(field.get("closure_pressure", 0.0) or 0.0) * 0.18
                + float(field.get("goal_alignment", 0.0) or 0.0) * 0.12
                - float(field.get("revision_pressure", 0.0) or 0.0) * 0.18
                - float(field.get("risk_commit_pressure", 0.0) or 0.0) * 0.22,
                0.0,
                1.0,
            ),
            "commit_reread_need": _clamp(
                float(field.get("revision_pressure", 0.0) or 0.0) * 0.74
                + float(field.get("ambiguity_pause", 0.0) or 0.0) * 0.32
                + float(draft_eval.get("continuation_readiness", 0.0) or 0.0) * 0.18,
                0.0,
                1.0,
            ),
            "commit_delete_need": _clamp(
                float(field.get("cleanup_pressure", 0.0) or 0.0) if int(draft_context.get("trailing_repeat_count", 0) or 0) > 1 else 0.0,
                0.0,
                1.0,
            ),
            "commit_replace_need": _clamp(
                float(field.get("revision_pressure", 0.0) or 0.0)
                + float(draft_eval.get("continuation_readiness", 0.0) or 0.0) * 0.10,
                0.0,
                1.0,
            ),
            "notes": [
                "short_lived_commit_readiness_state",
                "derived_from_planner_draft_appraisal",
                "not_force_submit",
            ],
        }

    def _draft_writing_context(self, state_snapshot_items: list[dict], *, current_tick: int) -> dict:
        draft_state = {}
        inactive_draft_state = {}
        event_rows = []
        for item in state_snapshot_items or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            source_type = str(item.get("source_type", "") or "")
            family = str(item.get("family", "") or "")
            if source_type != "text_action" and family != "text_action" and not label.startswith("text_action::"):
                continue
            meta = dict(item.get("anchor_meta", {}) or {})
            if label == "text_action::draft_state" or str(meta.get("schema_id", "") or "") == "text_draft_state/v1":
                is_active = bool(meta.get("active_draft_surface", label == "text_action::draft_state"))
                if is_active:
                    draft_state = meta
                elif not inactive_draft_state:
                    inactive_draft_state = meta
                continue
            event_rows.append(meta)
        if not draft_state and inactive_draft_state:
            draft_state = inactive_draft_state
        if draft_state:
            ctx = dict(draft_state)
        else:
            # Fallback keeps planner robust when tests or older traces provide
            # text_action events without the compact draft_state item.
            visible_tokens = [
                str(row.get("token", "") or "")
                for row in event_rows
                if str(row.get("event_type", "") or "") in {"insert", "write_revision", "write", "write_mismatch"}
                and str(row.get("token", "") or "")
            ]
            ctx = {
                "visible_text": "".join(visible_tokens),
                "visible_tokens": visible_tokens,
                "visible_length": len(visible_tokens),
                "last_visible_token": visible_tokens[-1] if visible_tokens else "",
                "trailing_repeat_token": visible_tokens[-1] if visible_tokens else "",
                "trailing_repeat_count": self._trailing_repeat_count(visible_tokens),
                "duplicate_ratio": self._duplicate_ratio(visible_tokens),
                "insert_count": sum(1 for row in event_rows if str(row.get("event_type", "") or "") == "insert"),
                "external_write_count": sum(1 for row in event_rows if str(row.get("source", "") or "") == "external_text"),
                "mismatch_count": sum(1 for row in event_rows if self._is_text_mismatch_meta(row)),
                "revision_count": sum(1 for row in event_rows if str(row.get("event_type", "") or "") in {"revise", "replace", "write_revision"}),
                "reread_count": sum(1 for row in event_rows if str(row.get("event_type", "") or "") == "reread"),
                "delete_count": sum(1 for row in event_rows if str(row.get("event_type", "") or "") == "delete"),
                "replace_count": sum(1 for row in event_rows if str(row.get("event_type", "") or "") == "replace"),
                "commit_count": sum(1 for row in event_rows if str(row.get("event_type", "") or "") == "commit"),
                "last_event_type": str((event_rows[-1] if event_rows else {}).get("event_type", "") or ""),
                "last_event_tick": self._latest_meta_tick(event_rows),
                "last_insert_tick": self._latest_meta_tick([row for row in event_rows if str(row.get("event_type", "") or "") == "insert"]),
                "last_reread_tick": self._latest_meta_tick([row for row in event_rows if str(row.get("event_type", "") or "") == "reread"]),
                "last_delete_tick": self._latest_meta_tick([row for row in event_rows if str(row.get("event_type", "") or "") == "delete"]),
                "last_replace_tick": self._latest_meta_tick([row for row in event_rows if str(row.get("event_type", "") or "") == "replace"]),
                "last_revision_tick": self._latest_meta_tick([row for row in event_rows if str(row.get("event_type", "") or "") in {"revise", "replace", "write_revision"}]),
                "last_mutation_tick": self._latest_meta_tick([row for row in event_rows if str(row.get("event_type", "") or "") in {"insert", "delete", "replace", "revise", "write_revision"}]),
                "last_commit_tick": self._latest_meta_tick([row for row in event_rows if str(row.get("event_type", "") or "") == "commit"]),
            }
        raw_visible_tokens = [str(token or "") for token in list(ctx.get("visible_tokens", []) or []) if str(token or "")]
        if raw_visible_tokens:
            ctx["visible_tokens"] = raw_visible_tokens
        else:
            visible_text = str(ctx.get("visible_text", "") or "")
            ctx["visible_tokens"] = list(visible_text) if visible_text else []
        if "trailing_repeat_count" not in ctx:
            ctx["trailing_repeat_count"] = self._trailing_repeat_count([str(token or "") for token in list(ctx.get("visible_tokens", []) or [])])
        if "duplicate_ratio" not in ctx:
            ctx["duplicate_ratio"] = self._duplicate_ratio([str(token or "") for token in list(ctx.get("visible_tokens", []) or [])])
        if "trailing_repeat_token" not in ctx:
            tokens = [str(token or "") for token in list(ctx.get("visible_tokens", []) or []) if str(token or "")]
            ctx["trailing_repeat_token"] = tokens[-1] if tokens else str(ctx.get("last_visible_token", "") or "")
        for key in (
            "last_event_tick",
            "last_insert_tick",
            "last_reread_tick",
            "last_delete_tick",
            "last_replace_tick",
            "last_revision_tick",
            "last_mutation_tick",
            "last_commit_tick",
        ):
            try:
                tick = int(ctx.get(key, -1) or -1)
            except (TypeError, ValueError):
                tick = -1
            ctx[key] = tick
            age_key = key.replace("_tick", "_age")
            ctx[age_key] = 9999 if tick < 0 else max(0, int(current_tick) - tick)
        if "latest_mismatch_index" not in ctx or int(ctx.get("latest_mismatch_index", -1) or -1) < 0:
            latest_index, latest_row = self._latest_text_mismatch_meta(event_rows)
            ctx["latest_mismatch_index"] = latest_index
            ctx["latest_mismatch_tick"] = self._latest_meta_tick([latest_row]) if latest_row else -1
            ctx["latest_mismatch_token"] = str((latest_row or {}).get("token", "") or "")
            ctx["latest_mismatch_expected_token"] = str(
                self._feedback_reference_token(latest_row) or (latest_row or {}).get("expected_token", "") or ""
            )
        insert_count = int(ctx.get("insert_count", 0) or 0)
        revision_count = int(ctx.get("revision_count", 0) or 0)
        visible_length_now = int(ctx.get("visible_length", 0) or 0)
        ctx["has_internal_draft"] = bool(visible_length_now > 0 and (insert_count > 0 or revision_count > 0))
        ctx["has_any_visible_text"] = bool(int(ctx.get("visible_length", 0) or 0) > 0)
        ctx["can_reread"] = bool(str(ctx.get("visible_text", "") or "") and int(ctx.get("visible_length", 0) or 0) > 0)
        return ctx

    def _is_text_mismatch_meta(self, row: dict) -> bool:
        return self._is_feedback_confirmed_text_mismatch(row)

    def _is_feedback_confirmed_text_mismatch(self, row: dict) -> bool:
        """
        Return True only for hard, repair-worthy output mismatches.

        V2 teaching allows AP's internal expected_token to differ from a
        teacher-on or scaffolded action token. That difference can be useful
        uncertainty/review material, but it must not become direct revision
        pressure unless post-action feedback explicitly confirms the text was
        wrong. Legacy external write_mismatch events remain hard mismatch
        evidence because they model the old slip scenario rather than a teacher
        scaffold label.
        """

        event_type = str((row or {}).get("event_type", "") or "")
        if event_type == "write_mismatch":
            return True
        if bool((row or {}).get("feedback_token_mismatch", False)):
            return True
        outcome = str((row or {}).get("feedback_outcome", "") or "")
        reference = self._feedback_reference_token(row)
        return bool(outcome == "punished" and reference)

    def _feedback_reference_token(self, row: dict) -> str:
        return str(
            (row or {}).get("feedback_reference_token", "")
            or (row or {}).get("feedback_expected_token", "")
            or (row or {}).get("teacher_reference_token_post_action_only", "")
            or (row or {}).get("target_token", "")
            or ""
        )

    def _latest_text_mismatch_meta(self, rows: list[dict]) -> tuple[int, dict]:
        for index in range(len(rows) - 1, -1, -1):
            row = dict(rows[index] or {})
            if self._is_text_mismatch_meta(row):
                return index, row
        return -1, {}

    def _text_span(self, span) -> tuple[int, int]:
        if isinstance(span, dict):
            start = span.get("start", span.get("from", span.get("begin", 0)))
            end = span.get("end", span.get("to", span.get("stop", None)))
            try:
                left = int(start or 0)
                right = int(end if end is not None else left + 1)
            except (TypeError, ValueError):
                return (0, 1)
            return (max(0, left), max(max(0, left), right))
        if isinstance(span, (list, tuple)) and len(span) >= 2:
            try:
                left = int(span[0])
                right = int(span[1])
            except (TypeError, ValueError):
                return (0, 1)
            return (max(0, left), max(max(0, left), right))
        return (0, 1)

    def _text_revision_opportunities(self, state_snapshot_items: list[dict]) -> list[dict]:
        rows = []
        for item in state_snapshot_items or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            family = str(item.get("family", "") or "")
            source_type = str(item.get("source_type", "") or "")
            meta = dict(item.get("anchor_meta", {}) or {})
            if not (
                label.startswith("text_revision_opportunity::")
                or family == "text_revision_opportunity"
                or str(meta.get("schema_id", "") or "") == "text_revision_opportunity/v1"
            ):
                continue
            operation = str(meta.get("operation", "") or "")
            if operation not in {"insert", "delete", "replace"}:
                continue
            support = _clamp(
                max(
                    float(meta.get("support", 0.0) or 0.0),
                    float(item.get("virtual_energy", 0.0) or 0.0),
                    float(item.get("real_energy", 0.0) or 0.0),
                    float(item.get("cognitive_pressure", 0.0) or 0.0),
                ),
                0.0,
                1.2,
            )
            span = self._text_span(meta.get("span"))
            cursor = int(meta.get("cursor", span[0]) or 0)
            rows.append(
                {
                    "schema_id": "text_revision_opportunity/v1",
                    "operation": operation,
                    "conflict_kind": str(meta.get("conflict_kind", operation) or operation),
                    "span": list(span),
                    "cursor": max(0, cursor),
                    "candidate_text": str(meta.get("candidate_text", meta.get("to_text", meta.get("expected_text", ""))) or ""),
                    "from_text": str(meta.get("from_text", "") or ""),
                    "visible_text": str(meta.get("visible_text", "") or ""),
                    "support": _round4(support),
                    "source_type": source_type,
                    "sa_label": label,
                    "notes": list(meta.get("notes", []) or [])[:8],
                }
            )
        rows.sort(
            key=lambda row: (
                -float(row.get("support", 0.0) or 0.0),
                int(list(row.get("span", [0, 0]) or [0, 0])[0]),
                str(row.get("operation", "") or ""),
            )
        )
        return rows

    def _revision_opportunity_notes(self, opportunity: dict, parameter_estimate: dict, operation: str) -> list[str]:
        support = float((opportunity or {}).get("support", 0.0) or 0.0)
        notes = [
            "text_revision_opportunity_action",
            "not_spellchecker_state_field_opportunity",
            "requires_recent_reread",
            f"operation={operation}",
            f"conflict_kind={str((opportunity or {}).get('conflict_kind', '') or '')}",
            f"support={_round4(support)}",
        ]
        if float((parameter_estimate or {}).get("support", 0.0) or 0.0) > 0.0:
            notes.extend(
                [
                    "parameter_memory_bias",
                    f"parameter_drive_bias={_round4(float((parameter_estimate or {}).get('drive_bias', 0.0) or 0.0))}",
                    f"parameter_similarity={_round4(float((parameter_estimate or {}).get('similarity', 0.0) or 0.0))}",
                ]
            )
        return notes

    def _draft_self_evaluation(
        self,
        draft_context: dict,
        expected_text: dict,
        *,
        correctness: float,
        grasp: float,
        pressure: float,
        dissonance: float,
        uncertainty: float,
    ) -> dict:
        """
        Low-level draft appraisal for action competition.

        This is deliberately not a semantic quality judge. It only converts
        white-box state facts (successor distribution, recent reread/edit ages,
        repetition, pressure) into soft drive terms that write/wait/reread/edit
        actions can compete over.
        """

        visible_length = int((draft_context or {}).get("visible_length", 0) or 0)
        has_internal_draft = bool((draft_context or {}).get("has_internal_draft", False))
        last_insert_age = int((draft_context or {}).get("last_insert_age", 9999) or 9999)
        last_reread_age = int((draft_context or {}).get("last_reread_age", 9999) or 9999)
        last_delete_age = int((draft_context or {}).get("last_delete_age", 9999) or 9999)
        trailing_repeat_count = int((draft_context or {}).get("trailing_repeat_count", 0) or 0)
        duplicate_ratio = _clamp(float((draft_context or {}).get("duplicate_ratio", 0.0) or 0.0), 0.0, 1.0)
        top_share = _clamp(float((expected_text or {}).get("top_share", 0.0) or 0.0), 0.0, 1.0)
        dominance_gap = _clamp(float((expected_text or {}).get("dominance_gap", 0.0) or 0.0), 0.0, 1.0)
        expected_strength = _clamp(float((expected_text or {}).get("strength", 0.0) or 0.0), 0.0, 1.2)
        decisive = bool((expected_text or {}).get("decisive", False))
        candidate_count = int((expected_text or {}).get("candidate_count", 0) or 0)
        raw_ambiguity = _clamp(float((expected_text or {}).get("ambiguity", 0.0) or 0.0), 0.0, 1.0)
        no_clear_successor = bool(candidate_count <= 0 or not decisive)
        ambiguity_pause = _clamp(
            raw_ambiguity
            + (0.12 if no_clear_successor and has_internal_draft else 0.0)
            + (0.08 if last_insert_age <= 1 and not decisive else 0.0)
            + uncertainty * 0.16
            + pressure * 0.08,
            0.0,
            1.0,
        )
        cleanup_pressure = _clamp(
            max(0, trailing_repeat_count - 1) * 0.34
            + duplicate_ratio * 0.28
            + dissonance * 0.10
            - (0.14 if last_delete_age <= 2 else 0.0),
            0.0,
            1.0,
        )
        continuation_readiness = _clamp(
            expected_strength * 0.34
            + top_share * 0.34
            + dominance_gap * 0.44
            + (0.16 if decisive else 0.0)
            + (0.10 if has_internal_draft else 0.0)
            - ambiguity_pause * 0.35
            - cleanup_pressure * 0.28
            - pressure * 0.12,
            0.0,
            1.0,
        )
        recently_reviewed = last_reread_age <= 3
        satisfaction = _clamp(
            (0.16 if has_internal_draft and visible_length > 0 else 0.0)
            + (0.18 if recently_reviewed else 0.0)
            + correctness * 0.24
            + grasp * 0.20
            + max(0.0, 0.42 - ambiguity_pause) * 0.18
            - cleanup_pressure * 0.32
            - pressure * 0.16
            - dissonance * 0.12,
            0.0,
            1.0,
        )
        return {
            "schema_id": "draft_self_evaluation/v1",
            "continuation_readiness": _round4(continuation_readiness),
            "ambiguity_pause": _round4(ambiguity_pause),
            "cleanup_pressure": _round4(cleanup_pressure),
            "satisfaction": _round4(satisfaction),
            "successor_decisive": decisive,
            "candidate_count": int(candidate_count),
            "top_share": _round4(top_share),
            "dominance_gap": _round4(dominance_gap),
            "trailing_repeat_count": int(trailing_repeat_count),
            "duplicate_ratio": _round4(duplicate_ratio),
        }

    def _draft_goal_alignment(
        self,
        *,
        state_snapshot_items: list[dict],
        draft_context: dict,
        fast_cn: list[dict],
        slow_cn: list[dict],
        consequence_estimates: dict,
        outcome_estimate: dict,
    ) -> dict:
        """
        Build a soft goal / consequence view for draft closure.

        This is not a taredacted-test-key judge. It converts ordinary state-pool
        anchors, successor evidence, and action-outcome memory into short-lived
        pressure terms that text_insert / reread / revise / commit can compete
        over. The action-level habit path is deliberately capped because it is
        not yet a context-indexed habit memory.
        """

        draft = dict(draft_context or {})
        expected_text = self._expected_text_context(fast_cn=fast_cn, slow_cn=slow_cn, draft_context=draft)
        visible_tokens = [str(token or "") for token in list(draft.get("visible_tokens", []) or []) if str(token or "")]
        visible_text = str(draft.get("visible_text", "") or "")
        visible_length = int(draft.get("visible_length", len(visible_tokens)) or 0)
        last_reread_age = int(draft.get("last_reread_age", 9999) or 9999)
        trailing_repeat_count = int(draft.get("trailing_repeat_count", 0) or 0)
        duplicate_ratio = _clamp(float(draft.get("duplicate_ratio", 0.0) or 0.0), 0.0, 1.0)
        last_visible_token = str(draft.get("last_visible_token", "") or "")
        expected_token = str(expected_text.get("token", "") or "")
        expected_strength = _clamp(float(expected_text.get("strength", 0.0) or 0.0), 0.0, 1.2)
        top_share = _clamp(float(expected_text.get("top_share", 0.0) or 0.0), 0.0, 1.0)
        dominance_gap = _clamp(float(expected_text.get("dominance_gap", 0.0) or 0.0), 0.0, 1.0)
        ambiguity = _clamp(float(expected_text.get("ambiguity", 0.0) or 0.0), 0.0, 1.0)
        decisive = bool(expected_text.get("decisive", False))
        candidate_count = int(expected_text.get("candidate_count", 0) or 0)
        has_unexpressed_successor = bool(visible_length > 0 and expected_token and expected_token != last_visible_token)
        unexpressed_successor_pressure = _clamp(
            (0.18 if has_unexpressed_successor else 0.0)
            + expected_strength * 0.26
            + top_share * 0.10
            + dominance_gap * 0.18
            + (0.08 if decisive else 0.0)
            + (0.10 if has_unexpressed_successor and visible_length < 4 else 0.0)
            - ambiguity * 0.08,
            0.0,
            1.0,
        )
        visible_lookup = set(visible_tokens)
        visible_lookup.update(f"text::{token}" for token in visible_tokens)
        if visible_text:
            visible_lookup.add(visible_text)
            visible_lookup.add(f"text::{visible_text}")
        if visible_length > 0:
            visible_lookup.add("text_action::draft_state")
            visible_lookup.add("draft_state")
        if bool(draft.get("has_internal_draft", False)):
            visible_lookup.add("text_action::internal_draft")
            visible_lookup.add("internal_draft")

        task_anchors = []
        target_hits = []
        alignment_scores = []
        dialogue_closure_need = 0.0
        dialogue_anchor_count = 0
        current_turn_active = False
        for item in state_snapshot_items or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            family = str(item.get("family", "") or "")
            source_type = str(item.get("source_type", "") or "")
            if not (
                label.startswith("task::")
                or label.startswith("intention::")
                or family in {"task", "intention"}
                or source_type in {"task_anchor", "intention_anchor"}
            ):
                continue
            meta = dict(item.get("anchor_meta", {}) or {})
            if str(meta.get("schema_id", "") or "") == "dialogue_turn_closure_anchor/v1" or label in {
                "task::reply_to_current_user_turn",
                "intention::dialogue_turn_closure",
            }:
                dialogue_anchor_count += 1
                current_turn_active = True
                dialogue_closure_need = max(
                    dialogue_closure_need,
                    _clamp(
                        float(meta.get("reply_closure_need", 0.0) or 0.0),
                        0.0,
                        1.0,
                    ),
                )
            target_labels = self._draft_anchor_target_labels(item, meta)
            strictness = _clamp(float(meta.get("strictness", item.get("strictness", 0.35)) or 0.35), 0.0, 1.0)
            anchor_energy = _clamp(
                max(
                    float(item.get("real_energy", 0.0) or 0.0),
                    float(item.get("virtual_energy", 0.0) or 0.0),
                    abs(float(item.get("cognitive_pressure", 0.0) or 0.0)),
                    0.20,
                ),
                0.0,
                1.0,
            )
            hits = []
            for target in target_labels:
                raw = target.split("::", 1)[-1] if "::" in target else target
                process_target = target in {"action::text_commit", "commit_reply", "reply_closure"} or raw in {
                    "text_commit",
                    "commit_reply",
                    "reply_closure",
                }
                matched = bool(
                    target in visible_lookup
                    or raw in visible_lookup
                    or (raw and visible_text and raw in visible_text)
                    or (process_target and visible_length > 0 and unexpressed_successor_pressure <= 0.18)
                )
                if matched:
                    hits.append(target)
                    target_hits.append(target)
            hit_share = len(hits) / max(1, len(target_labels)) if target_labels else 0.0
            anchor_alignment = _clamp(hit_share * (0.25 + anchor_energy * 0.45 + strictness * 0.30), 0.0, 1.0)
            if target_labels:
                alignment_scores.append(anchor_alignment)
            task_anchors.append(
                {
                    "anchor_label": label,
                    "target_labels": target_labels[:8],
                    "hit_labels": hits[:8],
                    "strictness": _round4(strictness),
                    "anchor_energy": _round4(anchor_energy),
                    "alignment": _round4(anchor_alignment),
                }
            )

        max_alignment = max(alignment_scores or [0.0])
        avg_alignment = sum(alignment_scores) / max(1, len(alignment_scores)) if alignment_scores else 0.0
        goal_alignment = _clamp(max_alignment * 0.72 + avg_alignment * 0.28, 0.0, 1.0)
        if dialogue_closure_need > 0.0 and visible_length > 0:
            draft_stable_bonus = 0.18 if last_reread_age <= 3 else 0.06
            goal_alignment = _clamp(
                goal_alignment + dialogue_closure_need * (0.30 + draft_stable_bonus) * max(0.20, 1.0 - unexpressed_successor_pressure * 0.76),
                0.0,
                1.0,
            )

        continuation_pressure = _clamp(
            expected_strength * 0.34
            + top_share * 0.22
            + dominance_gap * 0.34
            + (0.12 if decisive else 0.0)
            + unexpressed_successor_pressure * 0.46
            - ambiguity * 0.18,
            0.0,
            1.0,
        )
        if dialogue_closure_need > 0.0 and visible_length > 0:
            continuation_pressure = _clamp(
                continuation_pressure - dialogue_closure_need * (0.18 if last_reread_age <= 3 else 0.10),
                0.0,
                1.0,
            )
        revision_pressure = _clamp(
            max(0, trailing_repeat_count - 1) * 0.32
            + duplicate_ratio * 0.22
            + float(draft.get("mismatch_count", 0) or 0) * 0.10,
            0.0,
            1.0,
        )

        commit_estimate = dict((consequence_estimates or {}).get("action::text_commit", {}) or {})
        consequence_support = _clamp(float(commit_estimate.get("support", 0.0) or 0.0), 0.0, 1.0)
        consequence_reward = max(0.0, float(commit_estimate.get("reward", 0.0) or 0.0))
        consequence_correctness = max(0.0, float(commit_estimate.get("correctness", 0.0) or 0.0))
        consequence_punishment = max(0.0, float(commit_estimate.get("punishment", 0.0) or 0.0))
        consequence_pressure = max(0.0, float(commit_estimate.get("pressure", 0.0) or 0.0))

        outcome = dict(outcome_estimate or {})
        outcome_support = _clamp(float(outcome.get("support", 0.0) or 0.0), 0.0, 1.0)
        outcome_reward = max(0.0, float(outcome.get("reward", 0.0) or 0.0))
        outcome_correctness = max(0.0, float(outcome.get("correctness", 0.0) or 0.0))
        outcome_punishment = max(0.0, float(outcome.get("punishment", 0.0) or 0.0))
        outcome_pressure = max(0.0, float(outcome.get("pressure", 0.0) or 0.0))
        approach_bias = max(0.0, float(outcome.get("approach_bias", 0.0) or 0.0))
        avoidance_bias = max(0.0, float(outcome.get("avoidance_bias", 0.0) or 0.0))
        drive_bias = float(outcome.get("drive_bias", 0.0) or 0.0)
        event_count = int(outcome.get("event_count", 0) or 0)
        success_count = int(outcome.get("success_count", 0) or 0)
        failure_count = int(outcome.get("failure_count", 0) or 0)
        failure_streak = int(outcome.get("failure_streak", 0) or 0)

        outcome_commit_pressure = _clamp(
            consequence_support * (consequence_reward * 0.38 + consequence_correctness * 0.32)
            + outcome_support * (outcome_reward * 0.30 + outcome_correctness * 0.26 + approach_bias * 0.20 + max(0.0, drive_bias) * 0.22),
            0.0,
            1.0,
        )
        habit_gate = min(1.0, event_count / 8.0)
        habitual_commit_pressure = _clamp(
            outcome_support
            * habit_gate
            * (0.16 + max(0.0, drive_bias) * 0.34 + min(0.18, success_count * 0.018))
            - outcome_support * min(0.12, failure_count * 0.018),
            0.0,
            0.28,
        )
        risk_commit_pressure = _clamp(
            consequence_support * (consequence_punishment * 0.52 + consequence_pressure * 0.42)
            + outcome_support * (outcome_punishment * 0.42 + outcome_pressure * 0.38 + avoidance_bias * 0.28 + max(0.0, -drive_bias) * 0.36)
            + min(0.18, failure_streak * 0.05),
            0.0,
            1.0,
        )
        closure_pressure = _clamp(
            (0.15 if visible_length > 0 else 0.0)
            + (0.16 if last_reread_age <= 3 else 0.0)
            + (0.10 if visible_length > 0 and candidate_count <= 0 else 0.0)
            + goal_alignment * 0.24
            + dialogue_closure_need * (0.28 if visible_length > 0 else 0.0)
            + outcome_commit_pressure * 0.16
            + habitual_commit_pressure * 0.14
            - continuation_pressure * 0.16
            - unexpressed_successor_pressure * 0.28
            - revision_pressure * 0.28
            - risk_commit_pressure * 0.18,
            0.0,
            1.0,
        )
        return {
            "schema_id": "draft_goal_alignment/v1",
            "goal_alignment": _round4(goal_alignment),
            "closure_pressure": _round4(closure_pressure),
            "continuation_pressure": _round4(continuation_pressure),
            "unexpressed_successor_pressure": _round4(unexpressed_successor_pressure),
            "has_unexpressed_successor": bool(has_unexpressed_successor),
            "revision_pressure": _round4(revision_pressure),
            "habitual_commit_pressure": _round4(habitual_commit_pressure),
            "outcome_commit_pressure": _round4(outcome_commit_pressure),
            "risk_commit_pressure": _round4(risk_commit_pressure),
            "task_anchor_count": len(task_anchors),
            "dialogue_anchor_count": int(dialogue_anchor_count),
            "current_turn_active": bool(current_turn_active),
            "dialogue_closure_need": _round4(dialogue_closure_need),
            "task_anchors": task_anchors[:8],
            "target_label_hits": sorted(set(target_hits))[:12],
            "expected_text": dict(expected_text),
            "habit_scope": "action_level_only" if outcome_support > 0.0 else "none",
            "outcome_support": _round4(outcome_support),
            "consequence_support": _round4(consequence_support),
        }

    def _evidence_gap_context(
        self,
        *,
        state_snapshot_items: list[dict],
        expected_text: dict,
        draft_context: dict,
        uncertainty: float,
        dissonance: float,
        pressure: float,
        ambiguity_pause: float,
        revision_opportunities: list[dict],
    ) -> dict:
        """
        Detect a generic need for more evidence from the state field.

        This does not decide a task answer. It only exposes the humanlike
        feeling of "I do not have enough evidence yet" as soft action material
        so wait, reread, gaze/audio resampling, recall, commit, and LLM/tool
        requests can compete in the ordinary action field.
        """

        counts: dict[str, int] = defaultdict(int)
        missing_modalities: set[str] = set()
        explicit_missing_modalities: set[str] = set()
        conflict_labels: list[str] = []
        explicit_gap = 0.0
        conflict_strength = 0.0
        low_grasp = 0.0
        labels: set[str] = set()
        for item in state_snapshot_items or []:
            if not isinstance(item, dict):
                continue
            family = str(item.get("family", "") or "")
            source_type = str(item.get("source_type", "") or "")
            label = str(item.get("sa_label", "") or "")
            if label:
                labels.add(label)
            meta = dict(item.get("anchor_meta", {}) or {})
            if family:
                counts[family] += 1
            if source_type:
                counts[source_type] += 1
            schema = str(meta.get("schema_id", "") or "")
            if (
                family in {"evidence_gap", "uncertainty_evidence_gap"}
                or label.startswith("evidence_gap::")
                or schema in {"evidence_gap/v1", "uncertainty_evidence_gap/v1"}
            ):
                explicit_gap = max(
                    explicit_gap,
                    float(item.get("real_energy", 0.0) or 0.0),
                    float(item.get("virtual_energy", 0.0) or 0.0),
                    abs(float(item.get("cognitive_pressure", 0.0) or 0.0)),
                    float(meta.get("strength", 0.0) or 0.0),
                )
                for value in list(meta.get("missing_modalities", []) or []):
                    if str(value or ""):
                        modality = str(value or "")
                        missing_modalities.add(modality)
                        explicit_missing_modalities.add(modality)
                for value in list(meta.get("conflict_labels", []) or []):
                    if str(value or ""):
                        conflict_labels.append(str(value or ""))
            if (
                family in {"evidence_conflict", "modality_conflict"}
                or label.startswith("evidence_conflict::")
                or schema in {"evidence_conflict/v1", "modality_conflict/v1"}
            ):
                conflict_strength = max(
                    conflict_strength,
                    float(item.get("real_energy", 0.0) or 0.0),
                    float(item.get("virtual_energy", 0.0) or 0.0),
                    abs(float(item.get("cognitive_pressure", 0.0) or 0.0)),
                    float(meta.get("strength", 0.0) or 0.0),
                )
                if label:
                    conflict_labels.append(label)
            if family in {"low_grasp", "cognitive_feeling"} or label.startswith("feeling::uncertainty"):
                low_grasp = max(
                    low_grasp,
                    float(item.get("real_energy", 0.0) or 0.0),
                    float(item.get("virtual_energy", 0.0) or 0.0),
                    abs(float(item.get("cognitive_pressure", 0.0) or 0.0)),
                )

        has_visual = bool(
            counts.get("vision_scene", 0) > 0
            or counts.get("vision_object", 0) > 0
            or counts.get("vision", 0) > 0
        )
        has_audio = bool(counts.get("audio_event", 0) > 0 or counts.get("audio_semantic", 0) > 0)

        expected_token = str((expected_text or {}).get("token", "") or "")
        expected_source = str((expected_text or {}).get("source", "") or "")
        expected_candidate_count = int((expected_text or {}).get("candidate_count", 0) or 0)
        draft_text_active = bool(
            (draft_context or {}).get("has_any_visible_text", False)
            or (draft_context or {}).get("has_internal_draft", False)
            or int((draft_context or {}).get("last_insert_age", 9999) or 9999) <= 16
            or int((draft_context or {}).get("last_reread_age", 9999) or 9999) <= 16
            or int((draft_context or {}).get("last_commit_age", 9999) or 9999) <= 16
        )
        text_successor_active = bool(expected_token or expected_candidate_count > 0 or expected_source)
        text_dialogue_active = bool(
            counts.get("external_text", 0) > 0
            or counts.get("task_anchor", 0) > 0
            or counts.get("intention_anchor", 0) > 0
            or counts.get("dialogue_turn_state", 0) > 0
            or "task::reply_to_current_user_turn" in labels
            or "intention::dialogue_turn_closure" in labels
            or draft_text_active
            or text_successor_active
        )
        explicit_visual_context = bool(
            "vision" in explicit_missing_modalities
            or has_visual
            or any(
                label.startswith(("vision::", "visual::", "ocr::", "screen::", "image::"))
                for label in labels
            )
        )
        explicit_audio_context = bool(
            "audio" in explicit_missing_modalities
            or has_audio
            or any(label.startswith(("audio::", "sound::", "noise::")) for label in labels)
        )
        auto_modality_probe_allowed = bool(
            not text_dialogue_active
            or (
                conflict_strength >= 0.42
                and any(
                    label.startswith(("vision::", "visual::", "ocr::", "screen::", "image::", "audio::", "sound::", "noise::"))
                    for label in labels
                )
            )
        )

        if not has_visual and (explicit_visual_context or auto_modality_probe_allowed):
            missing_modalities.add("vision")
        if not has_audio and (explicit_audio_context or auto_modality_probe_allowed):
            missing_modalities.add("audio")

        expected_strength = _clamp(float((expected_text or {}).get("strength", 0.0) or 0.0), 0.0, 1.2)
        top_share = _clamp(float((expected_text or {}).get("top_share", 0.0) or 0.0), 0.0, 1.0)
        dominance_gap = _clamp(float((expected_text or {}).get("dominance_gap", 0.0) or 0.0), 0.0, 1.0)
        candidate_count = int((expected_text or {}).get("candidate_count", 0) or 0)
        successor_unclear = _clamp(
            float(ambiguity_pause) * 0.46
            + max(0.0, 0.54 - top_share) * 0.30
            + max(0.0, 0.20 - dominance_gap) * 0.42
            + (0.10 if candidate_count > 2 else 0.0)
            + max(0.0, 0.50 - expected_strength) * 0.12,
            0.0,
            1.0,
        )
        has_draft = bool((draft_context or {}).get("has_any_visible_text", False))
        revision_pressure = max([float(row.get("support", 0.0) or 0.0) for row in revision_opportunities or []] or [0.0])
        missing_visual = 0.0
        missing_audio = 0.0
        if "vision" in missing_modalities:
            missing_visual = max(0.0, 0.52 + float(uncertainty) * 0.24 + explicit_gap * 0.18)
        if "audio" in missing_modalities:
            missing_audio = max(0.0, 0.50 + float(uncertainty) * 0.22 + explicit_gap * 0.18)
        modality_gap = _clamp(max(missing_visual, missing_audio) * 0.42, 0.0, 1.0)
        strength = _clamp(
            explicit_gap * 0.38
            + conflict_strength * 0.32
            + successor_unclear * 0.30
            + modality_gap
            + float(uncertainty) * 0.24
            + float(dissonance) * 0.16
            + max(0.0, float(pressure) - 0.36) * 0.10
            + min(0.16, revision_pressure * 0.10)
            - (0.10 if has_draft and revision_pressure <= 0.0 and explicit_gap <= 0.0 and conflict_strength <= 0.0 else 0.0),
            0.0,
            1.0,
        )
        available = bool(strength >= 0.22 or explicit_gap >= 0.24 or conflict_strength >= 0.24)
        return {
            "schema_id": "evidence_gap_context/v1",
            "available": available,
            "strength": _round4(strength),
            "explicit_gap": _round4(explicit_gap),
            "successor_unclear": _round4(successor_unclear),
            "conflict_strength": _round4(conflict_strength),
            "low_grasp": _round4(_clamp(max(low_grasp, float(uncertainty), max(0.0, 1.0 - expected_strength) * 0.34), 0.0, 1.0)),
            "missing_visual": _round4(missing_visual),
            "missing_audio": _round4(missing_audio),
            "missing_modalities": sorted(missing_modalities),
            "conflict_labels": sorted(set(conflict_labels))[:12],
            "text_dialogue_active": bool(text_dialogue_active),
            "draft_text_active": bool(draft_text_active),
            "text_successor_active": bool(text_successor_active),
            "explicit_visual_context": bool(explicit_visual_context),
            "explicit_audio_context": bool(explicit_audio_context),
            "auto_modality_probe_allowed": bool(auto_modality_probe_allowed),
            "policy": "soft_evidence_gap_action_material_not_answer_judge;missing_sensory_modalities_are_contextual_not_default_for_text_dialogue",
        }

    def _draft_satisfaction_field(
        self,
        *,
        draft_eval: dict,
        draft_goal_alignment: dict,
        correctness: float,
        grasp: float,
        pressure: float,
        dissonance: float,
        uncertainty: float,
        pressure_anchor_level: float,
        expectation_gap: float,
    ) -> dict:
        """
        Convert draft appraisal into a one-tick action field.

        The field is explanatory and short-lived. It should make commit drive
        easier to audit, but it must not lock the draft or force submission.
        """

        low_satisfaction = _clamp(float((draft_eval or {}).get("satisfaction", 0.0) or 0.0), 0.0, 1.0)
        ambiguity_pause = _clamp(float((draft_eval or {}).get("ambiguity_pause", 0.0) or 0.0), 0.0, 1.0)
        cleanup_pressure = _clamp(float((draft_eval or {}).get("cleanup_pressure", 0.0) or 0.0), 0.0, 1.0)
        continuation_pressure = _clamp(
            max(
                float((draft_eval or {}).get("continuation_readiness", 0.0) or 0.0),
                float((draft_goal_alignment or {}).get("continuation_pressure", 0.0) or 0.0),
            ),
            0.0,
            1.0,
        )
        revision_pressure = _clamp(
            max(
                cleanup_pressure,
                float((draft_goal_alignment or {}).get("revision_pressure", 0.0) or 0.0),
            ),
            0.0,
            1.0,
        )
        closure_pressure = _clamp(float((draft_goal_alignment or {}).get("closure_pressure", 0.0) or 0.0), 0.0, 1.0)
        goal_alignment = _clamp(float((draft_goal_alignment or {}).get("goal_alignment", 0.0) or 0.0), 0.0, 1.0)
        habitual_commit_pressure = _clamp(float((draft_goal_alignment or {}).get("habitual_commit_pressure", 0.0) or 0.0), 0.0, 1.0)
        outcome_commit_pressure = _clamp(float((draft_goal_alignment or {}).get("outcome_commit_pressure", 0.0) or 0.0), 0.0, 1.0)
        risk_commit_pressure = _clamp(float((draft_goal_alignment or {}).get("risk_commit_pressure", 0.0) or 0.0), 0.0, 1.0)
        pressure_cost = _clamp(float(pressure) * 0.16 + float(pressure_anchor_level) * 0.10 + float(expectation_gap) * 0.08, 0.0, 1.0)
        satisfaction = _clamp(
            low_satisfaction * 0.34
            + closure_pressure * 0.30
            + goal_alignment * 0.18
            + habitual_commit_pressure * 0.10
            + outcome_commit_pressure * 0.16
            + float(correctness) * 0.12
            + float(grasp) * 0.10
            - continuation_pressure * 0.22
            - revision_pressure * 0.30
            - risk_commit_pressure * 0.36
            - pressure_cost
            - float(dissonance) * 0.10
            - float(uncertainty) * 0.08
            - ambiguity_pause * 0.10,
            0.0,
            1.0,
        )
        return {
            "schema_id": "draft_satisfaction_field/v1",
            "satisfaction": _round4(satisfaction),
            "closure_pressure": _round4(closure_pressure),
            "continuation_pressure": _round4(continuation_pressure),
            "revision_pressure": _round4(revision_pressure),
            "habitual_commit_pressure": _round4(habitual_commit_pressure),
            "outcome_commit_pressure": _round4(outcome_commit_pressure),
            "risk_commit_pressure": _round4(risk_commit_pressure),
            "goal_alignment": _round4(goal_alignment),
            "ambiguity_pause": _round4(ambiguity_pause),
            "cleanup_pressure": _round4(cleanup_pressure),
            "pressure_cost": _round4(pressure_cost),
            "ttl_ticks": 1,
            "meaning": "short_term_action_field_not_locked_state",
        }

    def _draft_anchor_target_labels(self, item: dict, meta: dict) -> list[str]:
        labels = []
        for source in (
            (meta or {}).get("target_labels", []),
            (item or {}).get("target_labels", []),
            (meta or {}).get("targets", []),
            (item or {}).get("targets", []),
        ):
            if isinstance(source, str):
                labels.append(source)
            elif isinstance(source, (list, tuple, set)):
                labels.extend(str(value or "") for value in source)
        target_text = str((meta or {}).get("target_text", "") or (item or {}).get("target_text", "") or "")
        if target_text:
            labels.append(target_text if target_text.startswith("text::") else f"text::{target_text}")
        normalized = []
        seen = set()
        for label in labels:
            text = str(label or "").strip()
            if not text:
                continue
            if "::" not in text:
                text = f"text::{text}"
            if text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized[:16]

    def _draft_task_context_signature(self, draft_goal_alignment: dict) -> str:
        anchors = []
        for row in list((draft_goal_alignment or {}).get("task_anchors", []) or []):
            if not isinstance(row, dict):
                continue
            label = str(row.get("anchor_label", "") or "")
            if label:
                anchors.append(label)
            for target in list(row.get("target_labels", []) or [])[:4]:
                clean = str(target or "")
                if clean:
                    anchors.append(clean)
        if not anchors:
            return ""
        raw = json.dumps(sorted(set(anchors))[:12], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _trailing_repeat_count(self, tokens: list[str]) -> int:
        clean = [str(token or "") for token in tokens if str(token or "")]
        if not clean:
            return 0
        last = clean[-1]
        count = 0
        for token in reversed(clean):
            if token != last:
                break
            count += 1
        return count

    def _duplicate_ratio(self, tokens: list[str]) -> float:
        clean = [str(token or "") for token in tokens if str(token or "")]
        if not clean:
            return 0.0
        return _round4(1.0 - (len(set(clean)) / max(1, len(clean))))

    def _latest_meta_tick(self, rows: list[dict]) -> int:
        ticks = []
        for row in rows or []:
            try:
                ticks.append(int((row or {}).get("tick_index", -1) or -1))
            except (TypeError, ValueError):
                continue
        return max(ticks or [-1])

    def _apply_text_cursor_readiness_guard(self, candidates: list[dict], draft_context: dict) -> list[dict]:
        """
        Keep parameterized text actions aligned with AP's current draft body.

        A teacher-on charwise action can seed "write this token at cursor N",
        but the motor action should only compete strongly when AP's visible
        draft has actually reached cursor N. This is a body-state constraint,
        not a character/order answer table.
        """

        draft = dict(draft_context or {})
        try:
            visible_length = max(0, int(draft.get("visible_length", 0) or 0))
        except (TypeError, ValueError):
            visible_length = 0
        for row in candidates or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("action_id", "") or "") != "action::text_insert":
                continue
            params = dict(row.get("params", {}) or {})
            cursor = self._optional_int(params.get("cursor", params.get("cursor_hint", None)))
            if cursor is None or cursor <= visible_length:
                continue
            distance = max(1, cursor - visible_length)
            scale = max(0.04, 0.20 / (1.0 + distance * 1.75))
            row["base_drive"] = _round4(float(row.get("base_drive", 0.0) or 0.0) * scale)
            row["drive"] = _round4(float(row.get("drive", 0.0) or 0.0) * scale)
            row.setdefault("notes", [])
            row["notes"] = list(row.get("notes", []) or []) + [
                "text_cursor_not_reached_soft_guard",
                "parameterized_char_action_waits_for_visible_cursor",
                f"visible_length={visible_length}",
                f"candidate_cursor={cursor}",
                f"cursor_readiness_scale={_round4(scale)}",
            ]
            predicted = dict(row.get("predicted_outcome", {}) or {})
            predicted["pressure"] = _round4(float(predicted.get("pressure", 0.0) or 0.0) + min(0.18, distance * 0.04))
            predicted["confidence"] = _round4(max(0.0, float(predicted.get("confidence", 0.0) or 0.0) - min(0.22, distance * 0.05)))
            row["predicted_outcome"] = predicted
        return candidates

    def _suppress_unavailable_draft_actions(self, candidates: list[dict], draft_context: dict) -> list[dict]:
        draft = dict(draft_context or {})
        visible_length = int(draft.get("visible_length", 0) or 0)
        has_replace_target = bool(
            visible_length > 0
            or int(draft.get("latest_mismatch_index", -1) or -1) >= 0
            or int(draft.get("mismatch_count", 0) or 0) > 0
        )
        for row in candidates or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("action_id", "") or "") != "action::text_replace":
                continue
            params = dict(row.get("params", {}) or {})
            explicit_span = params.get("span")
            if explicit_span is not None or has_replace_target:
                continue
            row["base_drive"] = _round4(min(float(row.get("base_drive", 0.0) or 0.0), 0.02))
            row["drive"] = _round4(min(float(row.get("drive", 0.0) or 0.0), 0.03))
            row.setdefault("notes", [])
            row["notes"] = list(row.get("notes", []) or []) + [
                "text_replace_suppressed_no_visible_target",
                "replace_requires_existing_target_or_explicit_span",
                "empty_draft_replace_noop_not_competition_target",
            ]
            predicted = dict(row.get("predicted_outcome", {}) or {})
            predicted["pressure"] = _round4(float(predicted.get("pressure", 0.0) or 0.0) + 0.08)
            predicted["confidence"] = _round4(min(float(predicted.get("confidence", 0.0) or 0.0), 0.10))
            row["predicted_outcome"] = predicted
        if bool(draft.get("can_reread", False)):
            return candidates
        for row in candidates or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("action_id", "") or "") != "action::text_reread":
                continue
            row["base_drive"] = _round4(min(float(row.get("base_drive", 0.0) or 0.0), 0.04))
            row["drive"] = _round4(min(float(row.get("drive", 0.0) or 0.0), 0.06))
            row.setdefault("notes", [])
            row["notes"] = list(row.get("notes", []) or []) + ["draft_reread_suppressed_no_visible_draft"]
            predicted = dict(row.get("predicted_outcome", {}) or {})
            predicted["confidence"] = _round4(min(float(predicted.get("confidence", 0.0) or 0.0), 0.12))
            row["predicted_outcome"] = predicted
        return candidates

    def _apply_unclosed_insert_review_guard(self, candidates: list[dict], draft_context: dict) -> list[dict]:
        """
        Softly prefer reread/closure after any fresh draft mutation.

        This is a V2 process anchor: after AP has changed the visible draft, the
        next tick should feel an evidence-gap/closure pressure until the draft
        has been reread or committed. It only inspects AP's own action-feedback
        state, never teacher labels or reference answers.
        """

        draft = dict(draft_context or {})
        if not bool(draft.get("has_internal_draft", False)):
            return candidates
        if int(draft.get("visible_length", 0) or 0) <= 0:
            return candidates
        last_insert_tick = int(draft.get("last_insert_tick", -1) or -1)
        last_replace_tick = int(draft.get("last_replace_tick", -1) or -1)
        last_mutation_tick = max(last_insert_tick, last_replace_tick)
        if last_mutation_tick < 0:
            return candidates
        last_reread_tick = int(draft.get("last_reread_tick", -1) or -1)
        last_commit_tick = int(draft.get("last_commit_tick", -1) or -1)
        if max(last_reread_tick, last_commit_tick) >= last_mutation_tick:
            return candidates
        last_insert_age = int(draft.get("last_insert_age", 9999) or 9999)
        last_replace_age = int(draft.get("last_replace_age", 9999) or 9999)
        recent_mutation_age = min(last_insert_age, last_replace_age)
        if recent_mutation_age > 3:
            return candidates

        has_reread_candidate = any(
            isinstance(row, dict) and str(row.get("action_id", "") or "") == "action::text_reread"
            for row in candidates or []
        )
        scale = 0.28 if has_reread_candidate else 0.46
        if recent_mutation_age >= 2:
            scale = min(0.62, scale + 0.12 * (recent_mutation_age - 1))
        for row in candidates or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("action_id", "") or "") != "action::text_insert":
                continue
            row["base_drive"] = _round4(float(row.get("base_drive", 0.0) or 0.0) * scale)
            row["drive"] = _round4(float(row.get("drive", 0.0) or 0.0) * scale)
            row.setdefault("notes", [])
            row["notes"] = list(row.get("notes", []) or []) + [
                "unclosed_insert_review_guard",
                "process_anchor_write_then_reread_before_continue",
                f"last_mutation_tick={last_mutation_tick}",
                f"last_reread_tick={last_reread_tick}",
                f"recent_mutation_age={recent_mutation_age}",
                f"unclosed_insert_guard_scale={_round4(scale)}",
            ]
            predicted = dict(row.get("predicted_outcome", {}) or {})
            predicted["pressure"] = _round4(float(predicted.get("pressure", 0.0) or 0.0) + 0.12)
            predicted["confidence"] = _round4(max(0.0, float(predicted.get("confidence", 0.0) or 0.0) - 0.10))
            row["predicted_outcome"] = predicted
        return candidates

    def _apply_draft_repetition_guard(self, candidates: list[dict], draft_context: dict) -> list[dict]:
        last_token = str((draft_context or {}).get("last_visible_token", "") or "")
        if not last_token:
            return candidates
        visible_length = int((draft_context or {}).get("visible_length", 0) or 0)
        visible_tokens = [
            str(token or "")
            for token in list((draft_context or {}).get("visible_tokens", []) or [])
            if str(token or "")
        ]
        if not visible_tokens:
            visible_text = str((draft_context or {}).get("visible_text", "") or "")
            visible_tokens = list(visible_text) if visible_text else []
        closed_tokens = set(visible_tokens)
        last_reread_age = int((draft_context or {}).get("last_reread_age", 9999) or 9999)
        last_insert_age = int((draft_context or {}).get("last_insert_age", 9999) or 9999)
        should_guard = visible_length >= 3 or last_reread_age <= 2 or last_insert_age <= 1
        if not should_guard:
            return candidates
        for row in candidates or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("action_id", "") or "") != "action::text_insert":
                continue
            params = dict(row.get("params", {}) or {})
            token = str(params.get("token", params.get("text", "")) or "")
            if not token:
                continue
            reason = str(params.get("reason", "") or "")
            notes = [str(note or "") for note in list(row.get("notes", []) or [])]
            is_continuation = "continue_after_visible_prefix" in reason or any("continue_after_visible_prefix" in note for note in notes)
            scale = 1.0
            guard_notes = []
            if token == last_token:
                scale = min(scale, 0.42)
                guard_notes.extend(["draft_repetition_guard", f"last_visible_token={last_token}"])
            if (
                token in closed_tokens
                and visible_length >= 2
                and is_continuation
                and len(closed_tokens) >= 2
            ):
                scale = min(scale, 0.30)
                guard_notes.extend(["draft_closed_token_continuation_guard", f"closed_token={token}"])
            if scale >= 0.999:
                continue
            # This is a soft anti-loop pressure, not a ban on repeated text.
            # If a later memory/action pathway supplies stronger evidence, it
            # can still re-enter competition on a following tick.
            row["base_drive"] = _round4(float(row.get("base_drive", 0.0) or 0.0) * scale)
            row["drive"] = _round4(float(row.get("drive", 0.0) or 0.0) * scale)
            row.setdefault("notes", [])
            row["notes"] = list(row.get("notes", []) or []) + guard_notes + [
                f"visible_length={visible_length}",
                f"draft_repetition_guard_scale={_round4(scale)}",
            ]
            predicted = dict(row.get("predicted_outcome", {}) or {})
            predicted["pressure"] = _round4(float(predicted.get("pressure", 0.0) or 0.0) + 0.08)
            predicted["confidence"] = _round4(max(0.0, float(predicted.get("confidence", 0.0) or 0.0) - 0.08))
            row["predicted_outcome"] = predicted
        return candidates

    def _apply_draft_review_saturation(self, candidates: list[dict], draft_context: dict, draft_goal_alignment: dict | None) -> list[dict]:
        """
        Model the feeling that rereading the same unchanged draft is no longer
        yielding useful evidence.

        This is intentionally a process-level modulation rather than a ban on
        reread or a shortcut to an answer. It uses only AP-visible facts: how
        many rereads have happened, whether the draft changed recently, whether
        the current dialogue turn still wants closure, and whether the visible
        draft has repetition/low novelty. If later evidence makes reread useful
        again, it can still win normally.
        """

        draft = dict(draft_context or {})
        if not bool(draft.get("has_internal_draft", False)):
            return candidates
        visible_length = int(draft.get("visible_length", 0) or 0)
        if visible_length <= 0:
            return candidates
        reread_count = int(draft.get("reread_count", 0) or 0)
        if reread_count <= 0:
            return candidates
        last_event_type = str(draft.get("last_event_type", "") or "")
        last_reread_age = int(draft.get("last_reread_age", 9999) or 9999)
        last_mutation_age = int(draft.get("last_mutation_age", 9999) or 9999)
        last_insert_age = int(draft.get("last_insert_age", 9999) or 9999)
        last_delete_age = int(draft.get("last_delete_age", 9999) or 9999)
        last_replace_age = int(draft.get("last_replace_age", 9999) or 9999)
        trailing_repeat_count = int(draft.get("trailing_repeat_count", 0) or 0)
        duplicate_ratio = _clamp(float(draft.get("duplicate_ratio", 0.0) or 0.0), 0.0, 1.0)
        goal = dict(draft_goal_alignment or {})
        dialogue_closure_need = _clamp(
            float(goal.get("dialogue_closure_need", 0.0) or 0.0),
            0.0,
            1.0,
        )
        expected_text = dict(goal.get("expected_text", {}) or {})
        expected_strength = _clamp(float(expected_text.get("strength", 0.0) or 0.0), 0.0, 1.2)
        expected_top_share = _clamp(float(expected_text.get("top_share", 0.0) or 0.0), 0.0, 1.0)
        expected_dominance_gap = _clamp(float(expected_text.get("dominance_gap", 0.0) or 0.0), 0.0, 1.0)
        expected_decisive = bool(expected_text.get("decisive", False))
        expected_candidate_count = int(expected_text.get("candidate_count", 0) or 0)
        continuation_pressure = _clamp(float(goal.get("continuation_pressure", 0.0) or 0.0), 0.0, 1.0)
        continuation_shift = dict(expected_text.get("continuation_shift", {}) or {})
        expected_source = str(expected_text.get("source", "") or "")
        cursor_aligned_successor = bool(
            continuation_shift.get("cursor_aligned_shift", False)
            or expected_source == "cursor_aligned_next_unread_region"
            or "cursor_aligned_next_unread_region" in expected_source
        )
        clear_successor_continuation = bool(
            expected_candidate_count > 0
            and expected_decisive
            and (expected_strength >= 0.16 or expected_top_share >= 0.72 or cursor_aligned_successor)
            and (continuation_pressure >= 0.30 or expected_top_share >= 0.72 or expected_dominance_gap >= 0.16)
        )
        open_successor_pressure = _clamp(
            max(
                continuation_pressure if clear_successor_continuation else 0.0,
                min(1.0, expected_strength) if cursor_aligned_successor else 0.0,
                expected_top_share * 0.52 if clear_successor_continuation else 0.0,
            ),
            0.0,
            1.0,
        )
        closure_need_for_saturation = _clamp(
            dialogue_closure_need * (1.0 - open_successor_pressure * 0.72),
            0.0,
            1.0,
        )
        no_recent_mutation = bool(last_mutation_age >= 2 and last_insert_age >= 2 and last_delete_age > 1 and last_replace_age > 1)
        repeated_review = _clamp(
            min(1.0, reread_count / 6.0)
            + (0.22 if last_event_type == "reread" and last_reread_age <= 2 else 0.0)
            + (0.16 if no_recent_mutation else 0.0),
            0.0,
            1.0,
        )
        low_novelty = _clamp(
            duplicate_ratio * 0.62
            + max(0, trailing_repeat_count - 1) * 0.16
            + (0.10 if visible_length >= 3 and reread_count >= 3 else 0.0),
            0.0,
            1.0,
        )
        saturation = _clamp(
            repeated_review * 0.58
            + low_novelty * 0.28
            + closure_need_for_saturation * 0.22,
            0.0,
            1.0,
        )
        if saturation < 0.18:
            return candidates

        for row in candidates or []:
            if not isinstance(row, dict):
                continue
            action_id = str(row.get("action_id", "") or "")
            if action_id == "action::text_reread":
                scale = max(0.16, 1.0 - saturation * (0.62 if no_recent_mutation else 0.42))
                row["base_drive"] = _round4(float(row.get("base_drive", 0.0) or 0.0) * scale)
                row["drive"] = _round4(float(row.get("drive", 0.0) or 0.0) * scale)
                row.setdefault("notes", [])
                row["notes"] = list(row.get("notes", []) or []) + [
                    "draft_review_saturation",
                    "same_draft_reread_low_information_gain",
                    f"review_saturation={_round4(saturation)}",
                    f"reread_count={reread_count}",
                    f"last_mutation_age={last_mutation_age}",
                    f"dialogue_closure_need={_round4(dialogue_closure_need)}",
                    f"reread_saturation_scale={_round4(scale)}",
                ]
                predicted = dict(row.get("predicted_outcome", {}) or {})
                predicted["reward"] = _round4(max(0.0, float(predicted.get("reward", 0.0) or 0.0) - saturation * 0.10))
                predicted["pressure"] = _round4(float(predicted.get("pressure", 0.0) or 0.0) + saturation * 0.07)
                predicted["confidence"] = _round4(max(0.0, float(predicted.get("confidence", 0.0) or 0.0) - saturation * 0.12))
                row["predicted_outcome"] = predicted
            elif action_id == "action::text_insert":
                params = dict(row.get("params", {}) or {})
                reason = str(params.get("reason", "") or "")
                notes = {str(note or "") for note in list(row.get("notes", []) or [])}
                continuation_like = bool(
                    "continue_after_visible_prefix" in reason
                    or "text_revision_opportunity_action" in notes
                    or "draft_expected_token_write" in notes
                )
                if not continuation_like:
                    continue
                if clear_successor_continuation:
                    row.setdefault("notes", [])
                    row["notes"] = list(row.get("notes", []) or []) + [
                        "draft_review_saturation_preserves_clear_successor_continuation",
                        f"review_saturation={_round4(saturation)}",
                        f"continuation_pressure={_round4(continuation_pressure)}",
                        f"expected_strength={_round4(expected_strength)}",
                        f"expected_top_share={_round4(expected_top_share)}",
                        f"cursor_aligned_successor={cursor_aligned_successor}",
                    ]
                    continue
                scale = max(0.22, 1.0 - saturation * (0.38 + low_novelty * 0.28))
                row["base_drive"] = _round4(float(row.get("base_drive", 0.0) or 0.0) * scale)
                row["drive"] = _round4(float(row.get("drive", 0.0) or 0.0) * scale)
                row.setdefault("notes", [])
                row["notes"] = list(row.get("notes", []) or []) + [
                    "draft_review_saturation_slows_continuation",
                    "unchanged_draft_needs_closure_or_revision_before_more_tokens",
                    f"review_saturation={_round4(saturation)}",
                    f"continuation_saturation_scale={_round4(scale)}",
                ]
                predicted = dict(row.get("predicted_outcome", {}) or {})
                predicted["pressure"] = _round4(float(predicted.get("pressure", 0.0) or 0.0) + saturation * 0.05)
                predicted["confidence"] = _round4(max(0.0, float(predicted.get("confidence", 0.0) or 0.0) - saturation * 0.08))
                row["predicted_outcome"] = predicted
            elif action_id == "action::text_commit" and dialogue_closure_need > 0.0:
                if open_successor_pressure >= 0.34:
                    row.setdefault("notes", [])
                    row["notes"] = list(row.get("notes", []) or []) + [
                        "draft_review_saturation_waits_for_clear_successor_before_closure",
                        f"review_saturation={_round4(saturation)}",
                        f"open_successor_pressure={_round4(open_successor_pressure)}",
                        f"continuation_pressure={_round4(continuation_pressure)}",
                        f"expected_strength={_round4(expected_strength)}",
                        f"expected_top_share={_round4(expected_top_share)}",
                        f"cursor_aligned_successor={cursor_aligned_successor}",
                    ]
                    predicted = dict(row.get("predicted_outcome", {}) or {})
                    predicted["pressure"] = _round4(float(predicted.get("pressure", 0.0) or 0.0) + open_successor_pressure * 0.06)
                    predicted["confidence"] = _round4(max(0.0, float(predicted.get("confidence", 0.0) or 0.0) - open_successor_pressure * 0.08))
                    row["predicted_outcome"] = predicted
                    continue
                bonus = min(0.28, saturation * (0.18 + dialogue_closure_need * 0.18))
                row["base_drive"] = _round4(float(row.get("base_drive", 0.0) or 0.0) + bonus)
                row["drive"] = _round4(_clamp(float(row.get("drive", 0.0) or 0.0) + bonus, 0.0, 1.8))
                row.setdefault("notes", [])
                row["notes"] = list(row.get("notes", []) or []) + [
                    "draft_review_saturation_supports_closure",
                    "same_draft_has_been_reviewed_without_new_evidence",
                    f"review_saturation={_round4(saturation)}",
                    f"closure_bonus={_round4(bonus)}",
                    f"dialogue_closure_need={_round4(dialogue_closure_need)}",
                ]
                predicted = dict(row.get("predicted_outcome", {}) or {})
                predicted["reward"] = _round4(float(predicted.get("reward", 0.0) or 0.0) + bonus * 0.28)
                predicted["pressure"] = _round4(max(0.0, float(predicted.get("pressure", 0.0) or 0.0) - bonus * 0.10))
                predicted["confidence"] = _round4(float(predicted.get("confidence", 0.0) or 0.0) + bonus * 0.18)
                row["predicted_outcome"] = predicted
        return candidates

    def _apply_post_commit_empty_surface_guard(self, candidates: list[dict], draft_context: dict, draft_goal_alignment: dict | None) -> list[dict]:
        """
        Keep a cleared draft from being reopened by stale successor momentum.

        This is a process guard, not a repetition ban. A live user turn,
        unfinished text goal, rhythm/quote/repeat intention, or later learned
        context can still make text insertion compete normally.
        """

        draft = dict(draft_context or {})
        if int(draft.get("visible_length", 0) or 0) > 0:
            return candidates
        last_commit_tick = int(draft.get("last_commit_tick", -1) or -1)
        last_commit_age = int(draft.get("last_commit_age", 9999) or 9999)
        if last_commit_tick < 0 or last_commit_age > 64:
            return candidates
        last_event_type = str(draft.get("last_event_type", "") or "")
        stale_closed_surface = last_event_type in {"commit", "prepare_continue"}
        goal = dict(draft_goal_alignment or {})
        active_text_goal = bool(
            bool(goal.get("current_turn_active", False))
            or int(goal.get("task_anchor_count", 0) or 0) > 0
            or int(goal.get("dialogue_anchor_count", 0) or 0) > 0
            or float(goal.get("dialogue_closure_need", 0.0) or 0.0) > 0.0
        )
        if active_text_goal and not stale_closed_surface:
            return candidates
        for row in candidates or []:
            if not isinstance(row, dict) or str(row.get("action_id", "") or "") != "action::text_insert":
                continue
            params = dict(row.get("params", {}) or {})
            notes = [str(note or "") for note in list(row.get("notes", []) or [])]
            process_text = " ".join(notes + [str(params.get("reason", "") or "")]).lower()
            intentional_repetition = any(
                marker in process_text
                for marker in (
                    "intentional_repeat",
                    "requested_repeat",
                    "rhythm",
                    "quote",
                    "verbatim",
                )
            )
            if intentional_repetition:
                continue
            row["base_drive"] = _round4(min(float(row.get("base_drive", 0.0) or 0.0), 0.04))
            row["drive"] = _round4(min(float(row.get("drive", 0.0) or 0.0), 0.04))
            row.setdefault("notes", [])
            row["notes"] = list(row.get("notes", []) or []) + [
                "post_commit_empty_surface_guard",
                "closed_draft_should_not_reopen_from_stale_successor",
                f"last_commit_age={last_commit_age}",
                "no_active_text_goal",
            ]
            predicted = dict(row.get("predicted_outcome", {}) or {})
            predicted["reward"] = _round4(max(0.0, float(predicted.get("reward", 0.0) or 0.0) - 0.18))
            predicted["pressure"] = _round4(float(predicted.get("pressure", 0.0) or 0.0) + 0.10)
            predicted["confidence"] = _round4(max(0.0, float(predicted.get("confidence", 0.0) or 0.0) - 0.22))
            row["predicted_outcome"] = predicted
        return candidates

    def _episode_replay_risk(
        self,
        *,
        action_consequence_trace: dict,
        pressure_anchor_level: float,
        pressure: float,
        expectation_gap: float,
    ) -> dict:
        estimates = dict((action_consequence_trace or {}).get("action_estimates", {}) or {})
        best: dict = {}
        best_risk = 0.0
        for action_id, estimate in estimates.items():
            if not isinstance(estimate, dict):
                continue
            support = _clamp(float(estimate.get("support", 0.0) or 0.0), 0.0, 1.0)
            punishment = max(0.0, float(estimate.get("punishment", 0.0) or 0.0))
            est_pressure = max(0.0, float(estimate.get("pressure", 0.0) or 0.0))
            risk = support * (punishment * 0.62 + est_pressure * 0.38)
            if risk > best_risk:
                best_risk = risk
                best = dict(estimate)
                best["source_action_id"] = str(action_id or "")
        source_ids = list(best.get("source_memory_ids", []) or [])
        baseline = max(0.0, float(pressure_anchor_level) * 0.42 + float(pressure) * 0.22 + float(expectation_gap) * 0.18)
        return {
            "risk": _round4(max(best_risk, baseline)),
            "support": _round4(float(best.get("support", 0.0) or 0.0)),
            "punishment": _round4(float(best.get("punishment", 0.0) or 0.0)),
            "pressure": _round4(float(best.get("pressure", 0.0) or 0.0)),
            "source_action_id": str(best.get("source_action_id", "") or ""),
            "source_memory_id": str(source_ids[0] if source_ids else ""),
        }

    def _candidate(
        self,
        *,
        action_id: str,
        actuator_id: str,
        base_drive: float,
        predicted: dict,
        notes: list[str],
        consequence_estimates: dict | None = None,
        source: str = "analytic_planner",
        innate_nodes: list[dict] | None = None,
        params: dict | None = None,
        supporting_anchors: list[dict] | None = None,
    ) -> dict:
        estimate = dict((consequence_estimates or {}).get(str(action_id or ""), {}) or {})
        outcome_estimate = self._outcome_memory.estimate(action_id)
        predicted = self._merge_predicted_with_consequence(predicted, estimate)
        predicted = self._merge_predicted_with_outcome_memory(predicted, outcome_estimate)
        bias = float(self._drive_bias[action_id] or 0.0)
        fatigue = float(self._actuator_fatigue[actuator_id] or 0.0)
        parameter_fatigue = self._parameter_action_fatigue_estimate(action_id=action_id, actuator_id=actuator_id, params=dict(params or {}))
        feedback_modulation = self._current_feedback_modulation(action_id)
        outcome_drive_bias = float(outcome_estimate.get("drive_bias", 0.0) or 0.0) * float(outcome_estimate.get("support", 0.0) or 0.0)
        if action_id in PASSIVE_MAINTENANCE_ACTIONS and outcome_drive_bias > 0.0:
            # Outcome memory is a low-level consequence/familiarity modulator.
            # It must not become AP's main habit system; state-field Bn/C*
            # recall remains the source of content-sensitive habit. Passive
            # "keep doing this" actions therefore get weaker positive outcome
            # boost so current surprise, motion, and pressure can interrupt.
            outcome_drive_bias *= 0.38
            bias = min(bias, 0.42)
        utility = (
            float(predicted.get("reward", 0.0) or 0.0)
            + 0.45 * float(predicted.get("expectation", 0.0) or 0.0)
            + 0.35 * float(predicted.get("correctness", 0.0) or 0.0)
            - 0.95 * float(predicted.get("punishment", 0.0) or 0.0)
            - 0.55 * float(predicted.get("pressure", 0.0) or 0.0)
        )
        parameter_fatigue_scale = self._parameter_action_fatigue_drive_scale(action_id)
        drive = (base_drive + utility * 0.36 + bias * self.bias_gain + outcome_drive_bias - fatigue * 0.24 - float(parameter_fatigue.get("fatigue", 0.0) or 0.0) * parameter_fatigue_scale) * feedback_modulation
        return {
            "action_id": action_id,
            "actuator_id": actuator_id,
            "base_drive": _round4(base_drive),
            "predicted_outcome": {key: _round4(value) for key, value in predicted.items()},
            "utility": _round4(utility),
            "bias": _round4(bias),
            "outcome_drive_bias": _round4(outcome_drive_bias),
            "fatigue": _round4(fatigue),
            "parameter_action_fatigue": dict(parameter_fatigue),
            "feedback_modulation": _round4(feedback_modulation),
            "drive": _round4(_clamp(drive, 0.0, 1.8)),
            "notes": list(notes)
            + (
                [
                    "same_parameter_action_short_fatigue",
                    f"parameter_action_fatigue={_round4(float(parameter_fatigue.get('fatigue', 0.0) or 0.0))}",
                    f"parameter_action_fatigue_scale={_round4(parameter_fatigue_scale)}",
                    f"parameter_signature={str(parameter_fatigue.get('signature', '') or '')}",
                ]
                if float(parameter_fatigue.get("fatigue", 0.0) or 0.0) > 0.0
                else ["no_same_parameter_action_fatigue"]
            )
            + self._consequence_notes(estimate)
            + self._outcome_memory_notes(outcome_estimate),
            "consequence_estimate": estimate,
            "outcome_memory_estimate": outcome_estimate,
            "planner_selected": False,
            "source": source,
            "innate_nodes": list(innate_nodes or []),
            "params": dict(params or {}),
            "supporting_anchors": [dict(anchor) for anchor in list(supporting_anchors or [])],
        }

    def _merge_innate_action_nodes(
        self,
        candidates: list[dict],
        innate_nodes: list[dict],
        *,
        consequence_estimates: dict | None = None,
        evidence_gap_context: dict | None = None,
    ) -> list[dict]:
        if not innate_nodes:
            return candidates
        by_action = {str(row.get("action_id", "") or ""): row for row in candidates}
        for node in innate_nodes:
            if not isinstance(node, dict):
                continue
            action_id = str(node.get("action_id", "") or "")
            if not action_id:
                continue
            if not self._predicted_action_currently_applicable(action_id, evidence_gap_context=evidence_gap_context):
                continue
            drive_bonus = float(node.get("drive", 0.0) or 0.0)
            if abs(drive_bonus) <= 0.0:
                continue
            existing = by_action.get(action_id)
            if existing is not None:
                existing["base_drive"] = _round4(float(existing.get("base_drive", 0.0) or 0.0) + drive_bonus)
                existing["drive"] = _round4(_clamp(float(existing.get("drive", 0.0) or 0.0) + drive_bonus, 0.0, 1.8))
                existing.setdefault("notes", [])
                existing["notes"] = list(existing.get("notes", []) or []) + list(node.get("notes", []) or []) + ["innate_drive_merged"]
                existing.setdefault("innate_nodes", [])
                existing["innate_nodes"] = list(existing.get("innate_nodes", []) or []) + [dict(node)]
                existing["source"] = "analytic_planner+innate_rule"
                continue
            if action_id == "action::move_gaze_to" and not self._has_absolute_gaze_params(dict(node.get("params", {}) or {})):
                # A visual surprise rule may add urgency, but a gaze movement
                # must still be parameterized by an actual visual target. If no
                # state-field target was found, creating a naked move action
                # would turn the current text/thought label into a fake eye
                # coordinate and poison parameter learning.
                continue
            meta = action_meta(action_id)
            actuator_id = str(node.get("actuator_id", "") or meta.get("actuator_id", "") or "actuator::legacy_internal")
            predicted = self._innate_predicted_outcome(node)
            candidate = self._candidate(
                action_id=action_id,
                actuator_id=actuator_id,
                base_drive=max(0.0, drive_bonus),
                predicted=predicted,
                notes=list(node.get("notes", []) or []) + ["innate_rule_candidate"],
                consequence_estimates=consequence_estimates,
                source="innate_rule",
                innate_nodes=[dict(node)],
                params=dict(node.get("params", {}) or {}),
            )
            candidates.append(candidate)
            by_action[action_id] = candidate
        return candidates

    def _apply_innate_action_biases(
        self,
        candidates: list[dict],
        innate_biases: list[dict],
        *,
        evidence_gap_context: dict | None = None,
    ) -> list[dict]:
        if not innate_biases:
            return candidates
        for bias in innate_biases:
            if not isinstance(bias, dict):
                continue
            action_id = str(bias.get("action_id", "") or "")
            if not action_id:
                continue
            if not self._predicted_action_currently_applicable(action_id, evidence_gap_context=evidence_gap_context):
                continue
            drive_delta = float(bias.get("drive_delta", bias.get("drive", 0.0)) or 0.0)
            if abs(drive_delta) <= 0.00001:
                continue
            bias_params = dict(bias.get("params", {}) or {})
            matching_rows = [
                row
                for row in candidates
                if isinstance(row, dict) and self._action_bias_matches_candidate(bias, row)
            ]
            if drive_delta <= 0.0:
                # Negative biases are stage/process suppression. They may
                # damp several same-action candidates, but they must not merge
                # their params into a live token/cursor candidate; doing so can
                # pollute AP's low-grain action-feedback memory.
                for existing in matching_rows:
                    before_drive = float(existing.get("drive", 0.0) or 0.0)
                    existing["base_drive"] = _round4(max(0.0, float(existing.get("base_drive", 0.0) or 0.0) + drive_delta))
                    existing["drive"] = _round4(_clamp(float(existing.get("drive", 0.0) or 0.0) + drive_delta, 0.0, 1.8))
                    existing.setdefault("notes", [])
                    existing["notes"] = list(existing.get("notes", []) or []) + list(bias.get("notes", []) or []) + [f"innate_action_bias={_round4(drive_delta)}"]
                    if bias_params:
                        existing["notes"] = list(existing.get("notes", []) or []) + ["teaching_suppression_params_not_merged"]
                    existing.setdefault("innate_biases", [])
                    existing["innate_biases"] = list(existing.get("innate_biases", []) or []) + [dict(bias)]
                    existing["source"] = str(existing.get("source", "") or "analytic_planner") + "+innate_bias"
                    if before_drive >= self.selection_threshold and float(existing.get("drive", 0.0) or 0.0) < self.selection_threshold:
                        existing["notes"] = list(existing.get("notes", []) or []) + ["soft_negative_bias_below_threshold"]
                # Negative innate bias only suppresses actions that are already
                # present. It does not create a ghost candidate just to suppress it.
                continue
            existing = matching_rows[0] if matching_rows else None
            if existing is not None:
                before_drive = float(existing.get("drive", 0.0) or 0.0)
                bias_params = dict(bias.get("params", {}) or {})
                if bias_params:
                    existing_params = dict(existing.get("params", {}) or {})
                    # A teaching bias is still soft drive, but its actuator
                    # parameters are part of the low-granularity demonstration.
                    # Without this merge, a teacher can strengthen "look" while
                    # the existing candidate keeps looking at a different place.
                    existing_params.update(bias_params)
                    existing["params"] = existing_params
                    existing.setdefault("teaching_parameter_biases", [])
                    existing["teaching_parameter_biases"] = list(existing.get("teaching_parameter_biases", []) or []) + [
                        {
                            "schema_id": "teaching_parameter_bias_merge/v1",
                            "action_id": action_id,
                            "source": str(bias.get("source", "") or ""),
                            "teacher_kind": str(bias.get("teacher_kind", "") or ""),
                            "params": bias_params,
                        }
                    ]
                existing["base_drive"] = _round4(max(0.0, float(existing.get("base_drive", 0.0) or 0.0) + drive_delta))
                existing["drive"] = _round4(_clamp(float(existing.get("drive", 0.0) or 0.0) + drive_delta, 0.0, 1.8))
                existing.setdefault("notes", [])
                existing["notes"] = list(existing.get("notes", []) or []) + list(bias.get("notes", []) or []) + [f"innate_action_bias={_round4(drive_delta)}"]
                if bias_params:
                    existing["notes"] = list(existing.get("notes", []) or []) + ["teaching_parameter_bias_merged"]
                existing.setdefault("innate_biases", [])
                existing["innate_biases"] = list(existing.get("innate_biases", []) or []) + [dict(bias)]
                existing["source"] = str(existing.get("source", "") or "analytic_planner") + "+innate_bias"
                if before_drive >= self.selection_threshold and float(existing.get("drive", 0.0) or 0.0) < self.selection_threshold:
                    existing["notes"] = list(existing.get("notes", []) or []) + ["soft_negative_bias_below_threshold"]
                continue
            if action_id == "action::move_gaze_to" and not self._has_absolute_gaze_params(dict(bias.get("params", {}) or {})):
                # Positive gaze bias without a spatial target is only an
                # urgency hint. It should merge into an existing visual target,
                # not invent an unparameterized movement.
                continue
            meta = action_meta(action_id)
            candidate = self._candidate(
                action_id=action_id,
                actuator_id=str(bias.get("actuator_id", "") or meta.get("actuator_id", "") or "actuator::legacy_internal"),
                base_drive=drive_delta,
                predicted=self._innate_predicted_outcome(bias),
                notes=list(bias.get("notes", []) or []) + ["innate_bias_positive_candidate"],
                consequence_estimates={},
                source="innate_action_bias",
                innate_nodes=[],
                params=dict(bias.get("params", {}) or {}),
            )
            candidate["innate_biases"] = [dict(bias)]
            candidates.append(candidate)
        return candidates

    def _action_bias_matches_candidate(self, bias: dict, candidate: dict) -> bool:
        action_id = str((bias or {}).get("action_id", "") or "")
        if action_id != str((candidate or {}).get("action_id", "") or ""):
            return False
        params = dict((bias or {}).get("params", {}) or {})
        if action_id == "action::text_insert":
            bias_token = str(
                params.get(
                    "token",
                    params.get("text", params.get("candidate_token", params.get("expected_token", ""))),
                )
                or ""
            )
            if not bias_token:
                return True
            candidate_params = dict((candidate or {}).get("params", {}) or {})
            candidate_token = str(
                candidate_params.get(
                    "token",
                    candidate_params.get("text", candidate_params.get("candidate_token", candidate_params.get("expected_token", ""))),
                )
                or ""
            )
            if bias_token != candidate_token:
                return False
            bias_cursor = self._optional_int(params.get("cursor", params.get("cursor_hint", None)))
            candidate_cursor = self._optional_int(candidate_params.get("cursor", candidate_params.get("cursor_hint", None)))
            return bool(bias_cursor is None or candidate_cursor is None or bias_cursor == candidate_cursor)
        if action_id in {"action::text_replace", "action::text_delete"}:
            bias_span = params.get("span")
            if bias_span in (None, [], ""):
                return True
            candidate_span = dict((candidate or {}).get("params", {}) or {}).get("span")
            return self._text_span(bias_span) == self._text_span(candidate_span)
        return True

    def _has_absolute_gaze_params(self, params: dict) -> bool:
        if "x" in params or "y" in params:
            return True
        return len(list((params or {}).get("bbox_norm", []) or [])) >= 2

    def _is_teacher_on_charwise_text_insert_demo(self, row: dict) -> bool:
        if str(row.get("action_id", "") or "") != "action::text_insert":
            return False
        params = dict(row.get("params", {}) or {})
        reason = str(params.get("reason", "") or "")
        allowed_reasons = {
            "teacher_on_only_visual_glyph_to_character_cooccurrence",
            "teacher_on_charwise_free_dialogue_demo",
        }
        if reason not in allowed_reasons:
            return False
        token = str(params.get("token", params.get("candidate_token", "")) or "")
        if not token or len(token) > 1:
            return False
        notes = {str(note or "") for note in list(row.get("notes", []) or [])}
        if (
            "teacher_on_triggers_low_grain_text_actuator_feedback_loop" in notes
            or "teacher_on_low_grain_text_action" in notes
        ):
            return True
        for bias in list(row.get("innate_biases", []) or []):
            if not isinstance(bias, dict):
                continue
            bias_params = dict(bias.get("params", {}) or {})
            bias_notes = {str(note or "") for note in list(bias.get("notes", []) or [])}
            if (
                str(bias_params.get("reason", "") or "") == reason
                and (
                    "teacher_on_triggers_low_grain_text_actuator_feedback_loop" in bias_notes
                    or "teacher_on_low_grain_text_action" in bias_notes
                )
            ):
                return True
        return False

    def _is_self_predicted_charwise_text_insert_ready(self, row: dict) -> bool:
        if str(row.get("action_id", "") or "") != "action::text_insert":
            return False
        params = dict(row.get("params", {}) or {})
        if str(params.get("reason", "") or "") != "expected_token_draft_write":
            return False
        token = str(params.get("token", params.get("text", "")) or "")
        if not token or len(token) > 1:
            return False
        notes = [str(note or "") for note in list(row.get("notes", []) or [])]
        if "one_token_internal_draft_action" not in notes or "successor_decisive" not in notes:
            return False

        def note_value(prefix: str, default: float = 0.0) -> float:
            for note in notes:
                if not note.startswith(prefix):
                    continue
                try:
                    return float(note.split("=", 1)[1])
                except (IndexError, TypeError, ValueError):
                    return default
            return default

        expected_strength = note_value("expected_strength=")
        top_share = note_value("top_share=")
        dominance_gap = note_value("dominance_gap=")
        ambiguity_pause = note_value("ambiguity_pause=", 1.0)
        return bool(
            expected_strength >= 0.82
            and top_share >= 0.54
            and dominance_gap >= 0.46
            and ambiguity_pause <= 0.46
        )

    def _apply_visual_orientation_arbitration(self, candidates: list[dict]) -> list[dict]:
        """
        Let strong peripheral orientation pressure compete with gaze holding.

        This is a same-actuator arbitration step, not a global action cap. The
        AP field may still run memory, wait, text, and internal actions in the
        same tick. Only the single visual center lane is adjusted, because one
        pair of eyes cannot simultaneously keep staring at the old target and
        move to a new peripheral target.
        """

        rows = [dict(row) for row in list(candidates or []) if isinstance(row, dict)]
        move_rows = [
            row
            for row in rows
            if str(row.get("action_id", "") or "") == "action::move_gaze_to"
            and self._has_absolute_gaze_params(dict(row.get("params", {}) or {}))
        ]
        hold_rows = [row for row in rows if str(row.get("action_id", "") or "") == "action::hold_gaze"]
        if not move_rows or not hold_rows:
            return rows

        max_orientation_pressure = 0.0
        for row in move_rows:
            params = dict(row.get("params", {}) or {})
            components = dict(params.get("score_components", {}) or {})
            pressure = _clamp(
                float(components.get("peripheral_need", 0.0) or 0.0) * 0.38
                + float(components.get("motion", 0.0) or 0.0) * 0.24
                + float(components.get("abs_pressure", 0.0) or 0.0) * 0.20
                + float(components.get("salience", 0.0) or 0.0) * 0.18,
                0.0,
                1.0,
            )
            row["visual_orientation_pressure"] = _round4(pressure)
            if pressure >= 0.62:
                # A small generic nudge is enough to let strong anomaly beat
                # gaze-holding habit, while weak peripheral flicker remains free
                # to lose. This uses only the live candidate's score components.
                gain = min(0.16, (pressure - 0.62) * 0.42)
                row["base_drive"] = _round4(float(row.get("base_drive", 0.0) or 0.0) + gain)
                row["drive"] = _round4(_clamp(float(row.get("drive", 0.0) or 0.0) + gain, 0.0, 1.8))
                row.setdefault("notes", [])
                row["notes"] = list(row.get("notes", []) or []) + [
                    "visual_orientation_pressure_gain",
                    f"orientation_pressure={_round4(pressure)}",
                ]
            max_orientation_pressure = max(max_orientation_pressure, pressure)

        if max_orientation_pressure <= 0.0:
            return rows

        hold_discount = min(0.42, max(0.0, max_orientation_pressure - 0.48) * 0.74)
        if hold_discount <= 0.0:
            return rows

        for row in hold_rows:
            old_drive = float(row.get("drive", 0.0) or 0.0)
            old_base = float(row.get("base_drive", 0.0) or 0.0)
            row["drive"] = _round4(_clamp(old_drive - hold_discount, 0.0, 1.8))
            row["base_drive"] = _round4(max(0.0, old_base - hold_discount * 0.45))
            row["visual_orientation_hold_discount"] = _round4(hold_discount)
            row.setdefault("notes", [])
            row["notes"] = list(row.get("notes", []) or []) + [
                "visual_orientation_pressure_softens_hold_gaze",
                f"orientation_pressure={_round4(max_orientation_pressure)}",
            ]
        return rows

    def _apply_pre_write_visual_closure_guard(self, candidates: list[dict], draft_context: dict | None = None) -> list[dict]:
        """
        Softly delay text writing while AP's eyes are still moving or refocusing.

        UI reading should learn the process "turn gaze / sharpen evidence, then
        write", instead of inserting a character in the same tick that decides
        to move the fovea. The guard only reads live action candidates and their
        visual evidence parameters; it never uses target text or teacher labels.
        """

        rows = [dict(row) for row in list(candidates or []) if isinstance(row, dict)]
        if not rows:
            return rows
        draft = dict(draft_context or {})
        visible_length = int(draft.get("visible_length", 0) or 0)
        last_reread_age = int(draft.get("last_reread_age", 9999) or 9999)
        last_insert_age = int(draft.get("last_insert_age", 9999) or 9999)
        # Once AP has a visible draft and has just looked back at it, allow the
        # text lane to continue. The post-insert reread guard still handles
        # write -> review rhythm. This pre-write guard is mainly for the empty
        # or visually unsettled moment before the first/next evidence closure.
        if visible_length > 0 and last_reread_age <= 2 and last_insert_age > 0:
            return rows
        visual_rows = []
        for row in rows:
            action_id = str(row.get("action_id", "") or "")
            if action_id not in {"action::move_gaze_to", "action::zoom_visual_focus"}:
                continue
            params = dict(row.get("params", {}) or {})
            try:
                distance = float(params.get("target_distance", 0.0) or 0.0)
            except (TypeError, ValueError):
                distance = 0.0
            try:
                precision = float(params.get("focus_precision", 1.0) or 1.0)
            except (TypeError, ValueError):
                precision = 1.0
            components = dict(params.get("score_components", {}) or {})
            try:
                peripheral_need = float(components.get("peripheral_need", 0.0) or 0.0)
            except (TypeError, ValueError):
                peripheral_need = 0.0
            drive = float(row.get("drive", 0.0) or 0.0)
            visual_rows.append(
                {
                    "action_id": action_id,
                    "drive": drive,
                    "target": str(params.get("gaze_target_key", "") or params.get("target", "") or ""),
                    "distance": _clamp(distance, 0.0, 1.5),
                    "precision": _clamp(precision, 0.0, 1.0),
                    "peripheral_need": _clamp(peripheral_need, 0.0, 1.0),
                }
            )
        if not visual_rows:
            return rows
        visual_pressure = 0.0
        source = {}
        for row in visual_rows:
            needs_movement = row["action_id"] == "action::move_gaze_to" and (row["distance"] > 0.045 or row["peripheral_need"] > 0.16)
            needs_focus = row["action_id"] == "action::zoom_visual_focus" and row["precision"] < 0.86
            if not needs_movement and not needs_focus:
                continue
            pressure = _clamp(
                row["drive"] / 1.8 * 0.30
                + row["distance"] * 0.52
                + max(0.0, 0.86 - row["precision"]) * 0.46
                + row["peripheral_need"] * 0.32
                + (0.18 if needs_movement else 0.0)
                + (0.12 if needs_focus else 0.0),
                0.0,
                1.0,
            )
            if pressure > visual_pressure:
                visual_pressure = pressure
                source = row
        if visual_pressure < 0.22:
            return rows
        scale = 0.06
        for row in rows:
            if str(row.get("action_id", "") or "") != "action::text_insert":
                continue
            notes = {str(note or "") for note in list(row.get("notes", []) or [])}
            if {
                "pre_write_visual_closure_guard",
                "self_predicted_charwise_try_softens_prewrite_guard",
                "teacher_on_charwise_action_demo_bypasses_prewrite_guard",
            } & notes:
                continue
            if self._is_teacher_on_charwise_text_insert_demo(row):
                row.setdefault("notes", [])
                row["notes"] = list(row.get("notes", []) or []) + [
                    "teacher_on_charwise_action_demo_bypasses_prewrite_guard",
                    "v2_teacher_on_low_grain_action_feedback_seed",
                    f"visual_pressure={_round4(visual_pressure)}",
                    f"visual_source={str(source.get('action_id', '') or '')}",
                    f"visual_target={str(source.get('target', '') or '')}",
                ]
                continue
            if self._is_self_predicted_charwise_text_insert_ready(row):
                softened_scale = 0.32
                row["base_drive"] = _round4(float(row.get("base_drive", 0.0) or 0.0) * softened_scale)
                row["drive"] = _round4(float(row.get("drive", 0.0) or 0.0) * softened_scale)
                row.setdefault("notes", [])
                row["notes"] = list(row.get("notes", []) or []) + [
                    "self_predicted_charwise_try_softens_prewrite_guard",
                    "v2_low_grain_visible_attempt_after_high_grasp",
                    f"visual_pressure={_round4(visual_pressure)}",
                    f"visual_guard_scale={_round4(softened_scale)}",
                    f"visual_source={str(source.get('action_id', '') or '')}",
                    f"visual_target={str(source.get('target', '') or '')}",
                ]
                continue
            row["base_drive"] = _round4(float(row.get("base_drive", 0.0) or 0.0) * scale)
            row["drive"] = _round4(float(row.get("drive", 0.0) or 0.0) * scale)
            row.setdefault("notes", [])
            row["notes"] = list(row.get("notes", []) or []) + [
                "pre_write_visual_closure_guard",
                "process_anchor_refoveate_or_sharpen_before_text_insert",
                f"visual_pressure={_round4(visual_pressure)}",
                f"visual_guard_scale={_round4(scale)}",
                f"visual_source={str(source.get('action_id', '') or '')}",
                f"visual_target={str(source.get('target', '') or '')}",
            ]
            predicted = dict(row.get("predicted_outcome", {}) or {})
            predicted["pressure"] = _round4(float(predicted.get("pressure", 0.0) or 0.0) + visual_pressure * 0.10)
            predicted["confidence"] = _round4(max(0.0, float(predicted.get("confidence", 0.0) or 0.0) - visual_pressure * 0.12))
            row["predicted_outcome"] = predicted
        return rows

    def _merge_memory_predicted_action_energy(
        self,
        candidates: list[dict],
        state_snapshot_items: list[dict],
        *,
        drive_gain: float,
        consequence_estimates: dict | None = None,
        evidence_gap_context: dict | None = None,
    ) -> list[dict]:
        gain = max(0.0, float(drive_gain))
        if gain <= 0.0:
            return candidates
        by_action = {str(row.get("action_id", "") or ""): row for row in candidates}
        for item in state_snapshot_items or []:
            if not isinstance(item, dict):
                continue
            action_id = str(item.get("sa_label", "") or "")
            if not action_id.startswith("action::"):
                continue
            if not self._predicted_action_currently_applicable(action_id, evidence_gap_context=evidence_gap_context):
                continue
            virtual_energy = max(0.0, float(item.get("virtual_energy", 0.0) or 0.0))
            if virtual_energy <= 0.0:
                continue
            estimate = dict((consequence_estimates or {}).get(action_id, {}) or {})
            support = _clamp(float(estimate.get("support", 0.0) or 0.0), 0.0, 1.0)
            reward = _clamp(float(estimate.get("reward", 0.0) or 0.0), 0.0, 1.0)
            correctness = _clamp(float(estimate.get("correctness", 0.0) or 0.0), 0.0, 1.0)
            punishment = _clamp(float(estimate.get("punishment", 0.0) or 0.0), 0.0, 1.0)
            empirical_utility = max(0.0, reward + correctness * 0.35 - punishment * 0.85)
            drive_bonus = _clamp(virtual_energy * gain + support * empirical_utility * 0.55, 0.0, 0.78)
            if drive_bonus <= 0.0:
                continue
            anchor_meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
            predicted_params = dict(anchor_meta.get("params", {}) or {})
            existing = by_action.get(action_id)
            note = f"memory_predicted_action_virtual={_round4(virtual_energy)}"
            if existing is not None:
                existing["base_drive"] = _round4(float(existing.get("base_drive", 0.0) or 0.0) + drive_bonus)
                existing["drive"] = _round4(_clamp(float(existing.get("drive", 0.0) or 0.0) + drive_bonus, 0.0, 1.8))
                if predicted_params:
                    existing_params = dict(existing.get("params", {}) or {})
                    for key, value in predicted_params.items():
                        if key not in existing_params or existing_params[key] in (None, "", [], {}):
                            existing_params[key] = value
                    existing["params"] = existing_params
                    existing["memory_predicted_params"] = predicted_params
                existing.setdefault("notes", [])
                existing["notes"] = list(existing.get("notes", []) or []) + [
                    note,
                    "drive_source::memory_predicted_action",
                    f"memory_predicted_action_support={_round4(support)}",
                    "memory_predicted_action_params_merged" if predicted_params else "memory_predicted_action_no_params",
                ]
                existing["source"] = str(existing.get("source", "") or "analytic_planner") + "+memory_predicted_action"
                continue
            meta = action_meta(action_id)
            candidate = self._candidate(
                action_id=action_id,
                actuator_id=str(meta.get("actuator_id", "") or "actuator::legacy_internal"),
                base_drive=drive_bonus,
                predicted={
                    "reward": 0.08 + drive_bonus * 0.20,
                    "punishment": 0.08,
                    "expectation": 0.18 + drive_bonus * 0.35,
                    "pressure": 0.10,
                    "correctness": 0.08,
                    "confidence": 0.22 + min(0.28, drive_bonus),
                    "memory_action_virtual_energy": virtual_energy,
                },
                notes=[
                    note,
                    "drive_source::memory_predicted_action",
                    f"memory_predicted_action_support={_round4(support)}",
                    "memory_predicted_action_params_available" if predicted_params else "memory_predicted_action_no_params",
                ],
                consequence_estimates=consequence_estimates,
                source="memory_predicted_action",
                params=predicted_params,
            )
            candidates.append(candidate)
            by_action[action_id] = candidate
        return candidates

    def _merge_consequence_supported_action_energy(
        self,
        candidates: list[dict],
        consequence_estimates: dict | None,
        *,
        evidence_gap_context: dict | None = None,
    ) -> list[dict]:
        """
        Let B/Cn action-consequence evidence enter the action field directly.

        C* action nodes are still the strongest content path when they survive
        into the state pool, but low-grain actions such as avoid, timefelt recall
        or audio resampling should not disappear merely because the predicted
        action item's virtual energy was too small after budget normalization.
        The evidence is bounded and empirical: similar prior state -> successor
        action_feedback -> soft candidate/drive modulation.
        """

        estimates = dict(consequence_estimates or {})
        if not estimates:
            return candidates
        by_action = {str(row.get("action_id", "") or ""): row for row in candidates if isinstance(row, dict)}
        skip_prefixes = ("action::llm_",)
        skip_actions = {
            "action::tool_call",
            "action::keyboard_type",
            "action::keyboard_hotkey",
            "action::pointer_click",
            "action::pointer_drag",
            "action::pointer_scroll",
            "action::pointer_move",
            # These require concrete token/span/commit context. They may still
            # be strengthened when already present, but should not be created
            # from consequence support without parameters.
            "action::text_insert",
            "action::text_replace",
            "action::text_delete",
            "action::text_commit",
            "action::text_reread",
            "action::move_gaze_to",
        }
        for action_id, raw_estimate in estimates.items():
            action_id = str(action_id or "")
            estimate = dict(raw_estimate or {})
            if not action_id.startswith("action::"):
                continue
            if action_id.startswith(skip_prefixes) or action_id in skip_actions:
                continue
            meta = action_meta(action_id)
            if not meta:
                continue
            if not self._predicted_action_currently_applicable(action_id, evidence_gap_context=evidence_gap_context):
                continue
            support = _clamp(float(estimate.get("support", 0.0) or 0.0), 0.0, 1.0)
            if support <= 0.0:
                continue
            reward = _clamp(float(estimate.get("reward", 0.0) or 0.0), 0.0, 1.0)
            correctness = _clamp(float(estimate.get("correctness", 0.0) or 0.0), 0.0, 1.0)
            punishment = _clamp(float(estimate.get("punishment", 0.0) or 0.0), 0.0, 1.0)
            confidence = _clamp(float(estimate.get("confidence", 0.0) or 0.0), 0.0, 1.0)
            empirical_utility = reward + correctness * 0.35 - punishment * 0.85
            if empirical_utility <= 0.0 and support < 0.30:
                continue
            bonus = min(0.70, support * (0.16 + max(0.0, empirical_utility) * 0.46))
            if bonus <= 0.0:
                continue
            existing = by_action.get(action_id)
            note_pack = [
                "drive_source::action_consequence_supported_native_action",
                f"consequence_native_support={_round4(support)}",
                f"consequence_native_bonus={_round4(bonus)}",
            ]
            if existing is not None:
                existing["base_drive"] = _round4(float(existing.get("base_drive", 0.0) or 0.0) + bonus)
                existing["drive"] = _round4(_clamp(float(existing.get("drive", 0.0) or 0.0) + bonus, 0.0, 1.8))
                existing.setdefault("notes", [])
                existing["notes"] = list(existing.get("notes", []) or []) + note_pack
                existing["source"] = str(existing.get("source", "") or "analytic_planner") + "+action_consequence_supported"
                continue
            params: dict = {}
            if action_id == "action::widen_audio_band":
                params = {"width_hz": 3200, "reason": "experience_supported_audio_resample"}
            elif action_id == "action::widen_visual_focus":
                params = {"scale": 1.18, "reason": "experience_supported_visual_widen"}
            elif action_id == "action::scan_visual_field":
                params = {"pattern": "experience_supported_resample", "reason": "experience_supported_visual_scan"}
            elif action_id == "action::wait":
                params = {"duration_ticks": 1, "reason": "experience_supported_wait"}
            elif action_id == "action::avoid":
                params = {"target": "current_risk", "reason": "experience_supported_avoid"}
            candidate = self._candidate(
                action_id=action_id,
                actuator_id=str(meta.get("actuator_id", "") or "actuator::legacy_internal"),
                base_drive=bonus,
                predicted={
                    "reward": reward,
                    "punishment": punishment,
                    "expectation": min(0.72, 0.18 + support * 0.42),
                    "pressure": max(0.0, punishment * 0.72 - reward * 0.16),
                    "correctness": correctness,
                    "confidence": max(confidence, 0.26 + support * 0.24),
                },
                notes=note_pack,
                consequence_estimates=estimates,
                source="action_consequence_supported",
                params=params,
            )
            candidates.append(candidate)
            by_action[action_id] = candidate
        return candidates

    def _predicted_action_currently_applicable(self, action_id: str, *, evidence_gap_context: dict | None = None) -> bool:
        """
        Keep consequence support attached to the current body/task context.

        A remembered successor can say "scanning helped there", but it should
        not by itself create a visual/audio resampling impulse while AP is in a
        live text-draft episode with no visual or auditory gap. The action is
        still available when the current state field actually contains that
        modal gap, or when AP is not in a text dialogue process.
        """

        action = str(action_id or "")
        gap = dict(evidence_gap_context or {})
        if action in {"action::scan_visual_field", "action::widen_visual_focus"}:
            if bool(gap.get("text_dialogue_active", False)) and not bool(gap.get("explicit_visual_context", False)):
                return False
        if action == "action::widen_audio_band":
            if bool(gap.get("text_dialogue_active", False)) and not bool(gap.get("explicit_audio_context", False)):
                return False
        return True

    def _innate_predicted_outcome(self, node: dict) -> dict:
        action_id = str((node or {}).get("action_id", "") or "")
        strength = _clamp(float((node or {}).get("strength", 0.0) or 0.0), 0.0, 1.0)
        meta = action_meta(action_id)
        external = is_external_action(action_id, str(meta.get("actuator_id", "") or ""))
        confidence = 0.24 + strength * 0.26
        punishment = 0.08 + (0.10 if external else 0.0)
        pressure = 0.08 + (0.16 if external else 0.0)
        reward = 0.08 + strength * 0.14
        correctness = 0.06 + strength * 0.12
        return {
            "reward": _round4(reward),
            "punishment": _round4(punishment),
            "expectation": _round4(strength * 0.32),
            "pressure": _round4(pressure),
            "correctness": _round4(correctness),
            "confidence": _round4(confidence),
            "innate_strength": _round4(strength),
        }

    def _merge_predicted_with_consequence(self, predicted: dict, estimate: dict) -> dict:
        merged = dict(predicted or {})
        support = _clamp(float((estimate or {}).get("support", 0.0) or 0.0), 0.0, 1.0)
        if support <= 0.0:
            return merged
        # Keep the current analytic prediction, but let experience move it.
        # Support-gating prevents sparse old evidence from overruling live context.
        mix = min(0.42, 0.18 + support * 0.24)
        for key in ("reward", "punishment", "correctness", "pressure", "confidence"):
            if key not in estimate:
                continue
            live = float(merged.get(key, 0.0) or 0.0)
            empirical = float(estimate.get(key, 0.0) or 0.0)
            merged[key] = _round4(live * (1.0 - mix) + empirical * mix)
        merged["experience_support"] = _round4(support)
        merged["experience_mix"] = _round4(mix)
        return merged

    def _merge_predicted_with_outcome_memory(self, predicted: dict, estimate: dict) -> dict:
        merged = dict(predicted or {})
        support = _clamp(float((estimate or {}).get("support", 0.0) or 0.0), 0.0, 1.0)
        if support <= 0.0:
            return merged
        # Long-term reward / punishment memory is an action-shaping signal.
        # It remains more conservative than immediate successor evidence.
        mix = min(0.34, 0.10 + support * 0.24)
        for key in ("reward", "punishment", "correctness", "pressure", "confidence"):
            live = float(merged.get(key, 0.0) or 0.0)
            empirical = float((estimate or {}).get(key, 0.0) or 0.0)
            merged[key] = _round4(live * (1.0 - mix) + empirical * mix)
        merged["outcome_memory_support"] = _round4(support)
        merged["outcome_memory_mix"] = _round4(mix)
        merged["outcome_memory_drive_bias"] = _round4(float((estimate or {}).get("drive_bias", 0.0) or 0.0))
        return merged

    def _consequence_notes(self, estimate: dict) -> list[str]:
        support = float((estimate or {}).get("support", 0.0) or 0.0)
        if support <= 0.0:
            return ["experience_support=0.0"]
        return [
            f"experience_support={_round4(support)}",
            f"experience_reward={_round4(float(estimate.get('reward', 0.0) or 0.0))}",
            f"experience_punishment={_round4(float(estimate.get('punishment', 0.0) or 0.0))}",
        ]

    def _outcome_memory_notes(self, estimate: dict) -> list[str]:
        support = float((estimate or {}).get("support", 0.0) or 0.0)
        if support <= 0.0:
            return ["outcome_memory_support=0.0"]
        return [
            f"outcome_memory_support={_round4(support)}",
            f"outcome_drive_bias={_round4(float(estimate.get('drive_bias', 0.0) or 0.0))}",
            f"outcome_failures={int(estimate.get('failure_count', 0) or 0)}",
        ]

    def _build_action_items(self, selected_actions: list[dict], tick_index: int) -> list[dict]:
        items = []
        for row in selected_actions:
            action_name = str(row.get("action_id", "") or "").split("::")[-1]
            items.append(
                {
                    "sa_label": f"action::{action_name}",
                    "display_text": f"行动:{action_name}",
                    "source_type": "action_selection",
                    "family": "action",
                    "real_energy": _round4(float(row.get("drive", 0.0) or 0.0)),
                    "anchor_meta": {
                        "tick_index": int(tick_index),
                        "action_id": row.get("action_id", ""),
                        "actuator_id": row.get("actuator_id", ""),
                        "base_drive": row.get("base_drive", 0.0),
                        "drive": row.get("drive", 0.0),
                        "effective_decisiveness": row.get("effective_decisiveness", 0.0),
                        "predicted_outcome": dict(row.get("predicted_outcome", {}) or {}),
                        "utility": row.get("utility", 0.0),
                        "consequence_estimate": dict(row.get("consequence_estimate", {}) or {}),
                        "outcome_memory_estimate": dict(row.get("outcome_memory_estimate", {}) or {}),
                        "source": str(row.get("source", "") or ""),
                        "innate_nodes": list(row.get("innate_nodes", []) or []),
                        "params": dict(row.get("params", {}) or {}),
                        "supporting_anchors": list(row.get("supporting_anchors", []) or []),
                    },
                }
            )
        return items

    def _active_expectation_anchors(self, expectation_anchor_trace: dict) -> list[dict]:
        anchors = [dict(anchor) for anchor in list((expectation_anchor_trace or {}).get("anchors", []) or []) if isinstance(anchor, dict)]
        anchors = [
            anchor
            for anchor in anchors
            if str(anchor.get("source_memory_id", "") or "")
            and float(anchor.get("level", 0.0) or 0.0) > 0.0
        ]
        anchors.sort(
            key=lambda anchor: (
                -float(anchor.get("level", 0.0) or 0.0),
                0 if str(anchor.get("anchor_type", "") or "") == "pressure" else 1,
                str(anchor.get("anchor_id", "") or ""),
            )
        )
        return anchors

    def build_action_items(self, selected_actions: list[dict], *, tick_index: int) -> list[dict]:
        return self._build_action_items(selected_actions, tick_index=int(tick_index))

    def _advance_tick(self, tick_index: int) -> None:
        if self._last_tick < 0:
            self._last_tick = tick_index
            self._outcome_memory.advance_tick(tick_index)
            self._parameter_memory.advance_tick(tick_index)
            return
        delta = max(1, tick_index - self._last_tick)
        self._outcome_memory.advance_tick(tick_index)
        self._parameter_memory.advance_tick(tick_index)
        for key in list(self._actuator_fatigue.keys()):
            self._actuator_fatigue[key] = _clamp(float(self._actuator_fatigue[key]) * (self.fatigue_decay**delta), 0.0, 1.0)
            if self._actuator_fatigue[key] < 0.0001:
                self._actuator_fatigue.pop(key, None)
        for key in list(self._visual_target_fatigue.keys()):
            self._visual_target_fatigue[key] = _clamp(float(self._visual_target_fatigue[key]) * (self.fatigue_decay**delta), 0.0, 1.0)
            if self._visual_target_fatigue[key] < 0.0001:
                self._visual_target_fatigue.pop(key, None)
        for key in list(self._parameter_action_fatigue.keys()):
            entry = dict(self._parameter_action_fatigue.get(key, {}) or {})
            value = _clamp(float(entry.get("fatigue", 0.0) or 0.0) * (self.fatigue_decay**delta), 0.0, 1.0)
            if value < 0.0001:
                self._parameter_action_fatigue.pop(key, None)
                continue
            entry["fatigue"] = value
            self._parameter_action_fatigue[key] = entry
        for key in list(self._feedback_modulation.keys()):
            entry = dict(self._feedback_modulation.get(key, {}) or {})
            ttl = int(entry.get("ttl", 0) or 0)
            ttl = max(0, ttl - delta)
            if ttl <= 0:
                self._feedback_modulation.pop(key, None)
                continue
            entry["ttl"] = ttl
            self._feedback_modulation[key] = entry
        self._last_tick = tick_index

    def _record_parameter_action_fatigue(
        self,
        *,
        action_id: str,
        actuator_id: str,
        params: dict,
        confidence: float,
        utility: float,
    ) -> None:
        signature = self._parameter_action_signature(action_id=action_id, actuator_id=actuator_id, params=params)
        if not signature:
            return
        existing = dict(self._parameter_action_fatigue.get(signature, {}) or {})
        # Positive, successful repetitions get the strongest short fatigue: the
        # action probably already did its job. Bad outcomes still add a smaller
        # pause so AP does not thrash the exact same failed parameter.
        if str(action_id or "") == "action::text_commit":
            utility_gain = 26.0 if float(utility) >= 0.0 else 4.8
        else:
            utility_gain = 1.25 if float(utility) >= 0.0 else 0.58
        step = _clamp(self.fatigue_step * max(0.5, float(confidence)) * utility_gain, 0.0, 1.0)
        if step <= 0.0:
            return
        self._parameter_action_fatigue[signature] = {
            "schema_id": "parameter_action_short_fatigue/v1",
            "signature": signature,
            "action_id": str(action_id or ""),
            "actuator_id": str(actuator_id or ""),
            "fatigue": _clamp(float(existing.get("fatigue", 0.0) or 0.0) + step, 0.0, 1.0),
            "params_preview": self._parameter_signature_payload(action_id=action_id, params=params),
            "policy": "same_action_same_key_params_short_fatigue_only",
        }

    def _parameter_action_fatigue_estimate(self, *, action_id: str, actuator_id: str, params: dict) -> dict:
        signature = self._parameter_action_signature(action_id=action_id, actuator_id=actuator_id, params=params)
        if not signature:
            return {"available": False, "fatigue": 0.0}
        entry = dict(self._parameter_action_fatigue.get(signature, {}) or {})
        if not entry:
            return {"available": False, "signature": signature, "fatigue": 0.0}
        return {
            "available": True,
            "signature": signature,
            "fatigue": _round4(float(entry.get("fatigue", 0.0) or 0.0)),
            "action_id": str(entry.get("action_id", "") or action_id),
            "params_preview": dict(entry.get("params_preview", {}) or {}),
        }

    def _parameter_action_signature(self, *, action_id: str, actuator_id: str, params: dict) -> str:
        action = str(action_id or "")
        actuator = str(actuator_id or "")
        if not action:
            return ""
        payload = self._parameter_signature_payload(action_id=action, params=params or {})
        if not payload:
            return ""
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        return f"{actuator}|{action}|{digest}"

    def _parameter_signature_payload(self, *, action_id: str, params: dict) -> dict:
        action = str(action_id or "")
        data = dict(params or {})
        if action == "action::text_commit":
            text = str(data.get("draft_signature", data.get("visible_text", data.get("text", ""))) or "")
            target_channel = str(data.get("target_channel", "draft") or "draft")
            task_context = str(data.get("task_context_signature", data.get("task_id", data.get("episode_id", ""))) or "")
            # Existing experiments do not always pass a draft signature yet.
            # The channel-only fallback is intentionally weak and temporary; it
            # still keeps parameter fatigue distinct from actuator fatigue.
            return {"target_channel": target_channel, "draft_signature": text, "task_context": task_context}
        if action in {"action::text_insert", "action::text_replace", "action::text_delete", "action::text_reread"}:
            return {
                key: data.get(key)
                for key in ("token", "text", "span", "new_text", "target_channel", "cursor", "reason")
                if key in data
            }
        if action in {"action::move_gaze_to", "action::nudge_gaze", "action::scan_visual_field", "action::hold_gaze", "action::zoom_visual_focus"}:
            return {
                key: data.get(key)
                for key in ("target", "gaze_target_key", "x", "y", "dx", "dy", "bbox_norm", "scale", "pattern")
                if key in data
            }
        if action.startswith("action::recall"):
            return {
                key: data.get(key)
                for key in ("recall_mode", "horizon", "active_episode_id", "delta_t", "b_anchor")
                if key in data
            }
        return {key: data.get(key) for key in sorted(data) if key not in {"reason", "notes"}}

    def _parameter_action_fatigue_drive_scale(self, action_id: str) -> float:
        action = str(action_id or "")
        if action == "action::text_commit":
            return 5.4
        if action.startswith("action::text_"):
            return 1.15
        if action in {"action::move_gaze_to", "action::nudge_gaze", "action::hold_gaze", "action::zoom_visual_focus"}:
            return 0.62
        return 0.82

    def _parameter_events_by_action(self, parameter_events: list[dict]) -> dict[str, list[dict]]:
        rows: dict[str, list[dict]] = defaultdict(list)
        for event in parameter_events or []:
            if not isinstance(event, dict):
                continue
            action_id = str(event.get("action_id", "") or "")
            if not action_id:
                continue
            rows[action_id].append(dict(event))
        return rows

    def _update_visual_target_fatigue(self, *, selected_actions: list[dict], parameter_events: list[dict], observed_feedback: dict) -> None:
        """
        Build short-term exploration fatigue for already-clear gaze targets.

        This is not a scan script. It only reduces the chance that a recently
        clear target monopolizes the next few ticks; strong pressure, movement,
        or fresh surprise can still overcome it.
        """

        feedback_utility = (
            float((observed_feedback or {}).get("reward", 0.0) or 0.0)
            + float((observed_feedback or {}).get("correctness", 0.0) or 0.0) * 0.35
            - float((observed_feedback or {}).get("punishment", 0.0) or 0.0) * 0.65
        )
        events_by_action = self._parameter_events_by_action(parameter_events)
        for row in selected_actions or []:
            action_id = str((row or {}).get("action_id", "") or "")
            if action_id not in {"action::hold_gaze", "action::zoom_visual_focus", "action::move_gaze_to"}:
                continue
            params = dict((row or {}).get("params", {}) or {})
            target = str(params.get("gaze_target_key", "") or params.get("target", "") or "")
            if not target:
                for event in events_by_action.get(action_id, []):
                    target = str(event.get("gaze_target_key", "") or event.get("target", "") or "")
                    if target:
                        break
            if not target:
                continue
            focus_signal = 0.0
            for event in events_by_action.get(action_id, []):
                event_target = str(event.get("gaze_target_key", "") or event.get("target", "") or "")
                if event_target == target:
                    focus_signal = max(focus_signal, 1.0 - min(1.0, float(event.get("movement_distance", 0.0) or 0.0) * 2.0))
            if action_id == "hold_gaze":
                focus_signal = max(focus_signal, 0.78)
            elif action_id == "zoom_visual_focus":
                focus_signal = max(focus_signal, 0.62)
            else:
                focus_signal = max(focus_signal, 0.35)
            if feedback_utility < -0.05:
                focus_signal *= 0.45
            step = _clamp(0.08 + focus_signal * 0.18 + max(0.0, feedback_utility) * 0.04, 0.0, 0.32)
            self._visual_target_fatigue[target] = _clamp(float(self._visual_target_fatigue[target]) + step, 0.0, 1.0)

    def _current_feedback_modulation(self, action_id: str) -> float:
        entry = dict(self._feedback_modulation.get(str(action_id or ""), {}) or {})
        return _clamp(float(entry.get("modulation", 1.0) or 1.0), 0.35, 1.2)

    def _drive_snapshot(self) -> dict:
        return {
            "bias": {key: _round4(value) for key, value in self._drive_bias.items()},
            "fatigue": {key: _round4(value) for key, value in self._actuator_fatigue.items()},
            "visual_target_fatigue": {key: _round4(value) for key, value in self._visual_target_fatigue.items()},
            "parameter_action_fatigue": {
                key: {
                    **{k: v for k, v in dict(value).items() if k != "fatigue"},
                    "fatigue": _round4(float(dict(value).get("fatigue", 0.0) or 0.0)),
                }
                for key, value in self._parameter_action_fatigue.items()
            },
            "feedback_modulation": {key: dict(value) for key, value in self._feedback_modulation.items()},
            "outcome_memory": self._outcome_memory.snapshot(),
            "parameter_memory": self._parameter_memory.snapshot(),
        }
