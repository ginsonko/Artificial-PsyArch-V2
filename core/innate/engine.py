from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp

from core.action.registry import ACTION_NODE_REGISTRY, ACTUATOR_REGISTRY, action_meta, is_external_action
from core.innate.default_rules import FATIGUE_TYPES, RuleDef, default_rules


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


@dataclass(frozen=True)
class InnateRule:
    rule_id: str
    phase: str
    description: str
    condition: str
    outputs: tuple[dict, ...]
    fatigue_type: str
    threshold: float = 0.0
    anchor: str = "global"

    @classmethod
    def from_rule_def(cls, row: RuleDef) -> "InnateRule":
        return cls(**asdict(row))


class InnateCodingEngine:
    """
    Declarative innate coding layer.

    It reads APV2.1 white-box traces and emits auditable seeds: feeling SAs,
    emotion deltas, action-node drives, learning events and SafetyGate hints.
    It does not replace Bn/Cn recall, online embedding, or action outcome
    learning.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_fire_strength: float = 0.035,
        max_items_per_phase: int = 16,
        max_action_nodes_per_phase: int = 12,
        max_learning_events_per_phase: int = 24,
        apply_emit_sa: bool = True,
        apply_action_nodes: bool = True,
        apply_emotion_deltas: bool = True,
        rules: list[RuleDef | InnateRule] | None = None,
        fatigue_types: dict[str, dict] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.min_fire_strength = max(0.0, float(min_fire_strength))
        self.max_items_per_phase = max(1, int(max_items_per_phase))
        self.max_action_nodes_per_phase = max(1, int(max_action_nodes_per_phase))
        self.max_learning_events_per_phase = max(1, int(max_learning_events_per_phase))
        self.apply_emit_sa = bool(apply_emit_sa)
        self.apply_action_nodes = bool(apply_action_nodes)
        self.apply_emotion_deltas = bool(apply_emotion_deltas)
        self._rules = [self._coerce_rule(rule) for rule in (rules if rules is not None else default_rules())]
        self._rules_by_phase: dict[str, list[InnateRule]] = {}
        for rule in self._rules:
            self._rules_by_phase.setdefault(str(rule.phase), []).append(rule)
        self._fatigue_types = {key: dict(value) for key, value in (fatigue_types or FATIGUE_TYPES).items()}
        self._rule_fatigue: dict[str, float] = {}
        self._anchor_fatigue: dict[tuple[str, str], float] = {}
        self._last_tick = -1

    def validate(self) -> dict:
        errors: list[dict] = []
        seen = set()
        for rule in self._rules:
            if rule.rule_id in seen:
                errors.append({"rule_id": rule.rule_id, "error": "duplicate_rule_id"})
            seen.add(rule.rule_id)
            if rule.fatigue_type not in self._fatigue_types:
                errors.append({"rule_id": rule.rule_id, "error": "unknown_fatigue_type", "fatigue_type": rule.fatigue_type})
            for output in rule.outputs:
                output_type = str(output.get("type", "") or "")
                if output_type in {"action_node", "action_bias"}:
                    action_id = str(output.get("action_id", "") or "")
                    if action_id not in ACTION_NODE_REGISTRY:
                        errors.append({"rule_id": rule.rule_id, "error": "unknown_action_id", "action_id": action_id})
                if output_type == "emotion_delta":
                    channel = str(output.get("channel", "") or "")
                    if channel not in {"DA", "ADR", "OXY", "SER", "END", "COR", "NOV", "FOC"}:
                        errors.append({"rule_id": rule.rule_id, "error": "unknown_emotion_channel", "channel": channel})
        return {
            "schema_id": "innate_rule_validation/v1",
            "ok": not errors,
            "rule_count": len(self._rules),
            "phase_count": len(self._rules_by_phase),
            "actuator_count": len(ACTUATOR_REGISTRY),
            "action_node_count": len(ACTION_NODE_REGISTRY),
            "errors": errors,
        }

    def simulate(self, *, phase: str, context: dict) -> dict:
        return self.evaluate(phase=phase, context=context, tick_index=int(context.get("tick_index", 0) or 0), update_fatigue=False)

    def evaluate(self, *, phase: str, context: dict, tick_index: int, update_fatigue: bool = True) -> dict:
        phase_name = str(phase or "")
        self._advance_tick(int(tick_index))
        if not self.enabled:
            return self._empty_trace(phase_name, disabled=True)
        metrics = self._build_metrics(context or {})
        hits: list[dict] = []
        suppressed: list[dict] = []
        items: list[dict] = []
        action_nodes: list[dict] = []
        action_biases: list[dict] = []
        emotion_deltas: dict[str, float] = {}
        learning_events: list[dict] = []
        attention_biases: list[dict] = []
        safety_gate: list[dict] = []
        trace_logs: list[dict] = []

        for rule in self._rules_by_phase.get(phase_name, []):
            raw_strength = self._condition_strength(rule.condition, metrics)
            if raw_strength < float(rule.threshold):
                suppressed.append(
                    {
                        "rule_id": rule.rule_id,
                        "reason": "below_threshold",
                        "raw_strength": _round4(raw_strength),
                        "threshold": _round4(rule.threshold),
                    }
                )
                continue
            anchor_key = self._anchor_key(rule.anchor, metrics)
            fatigue = self._combined_fatigue(rule, anchor_key)
            params = self._fatigue_params(rule.fatigue_type)
            scale = _clamp(1.0 - fatigue * float(params.get("gain", 0.0) or 0.0), float(params.get("min_scale", 0.0) or 0.0), 1.0)
            strength = _clamp(raw_strength * scale, 0.0, 1.0)
            if strength < self.min_fire_strength:
                suppressed.append(
                    {
                        "rule_id": rule.rule_id,
                        "reason": "fatigued_below_min",
                        "raw_strength": _round4(raw_strength),
                        "effective_strength": _round4(strength),
                        "fatigue": _round4(fatigue),
                    }
                )
                continue
            if update_fatigue:
                self._record_fire(rule, anchor_key, strength)
            hit = {
                "rule_id": rule.rule_id,
                "phase": phase_name,
                "condition": rule.condition,
                "description": rule.description,
                "raw_strength": _round4(raw_strength),
                "effective_strength": _round4(strength),
                "fatigue_type": rule.fatigue_type,
                "fatigue_before": _round4(fatigue),
                "anchor_key": anchor_key,
            }
            hits.append(hit)
            output_trace = []
            for output in rule.outputs:
                output_type = str(output.get("type", "") or "")
                if output_type == "emit_sa" and self.apply_emit_sa:
                    item = self._build_item(rule, output, strength, metrics, anchor_key, tick_index=int(tick_index))
                    if len(items) < self.max_items_per_phase:
                        items.append(item)
                    output_trace.append({"type": output_type, "sa_label": item["sa_label"]})
                elif output_type == "action_node" and self.apply_action_nodes:
                    node = self._build_action_node(rule, output, strength, metrics, tick_index=int(tick_index))
                    if len(action_nodes) < self.max_action_nodes_per_phase:
                        action_nodes.append(node)
                    output_trace.append({"type": output_type, "action_id": node["action_id"], "drive": node["drive"]})
                elif output_type == "action_bias" and self.apply_action_nodes:
                    bias = self._build_action_bias(rule, output, strength, metrics, tick_index=int(tick_index))
                    if len(action_biases) < self.max_action_nodes_per_phase:
                        action_biases.append(bias)
                    output_trace.append({"type": output_type, "action_id": bias["action_id"], "drive_delta": bias["drive_delta"]})
                elif output_type == "emotion_delta" and self.apply_emotion_deltas:
                    channel = str(output.get("channel", "") or "")
                    if channel:
                        delta = float(output.get("delta", 0.0) or 0.0) * strength
                        emotion_deltas[channel] = _round4(float(emotion_deltas.get(channel, 0.0) or 0.0) + delta)
                    output_trace.append({"type": output_type, "channel": channel, "delta": _round4(delta)})
                elif output_type == "learning_event":
                    event = self._build_learning_event(rule, output, strength, metrics, tick_index=int(tick_index))
                    if len(learning_events) < self.max_learning_events_per_phase:
                        learning_events.append(event)
                    output_trace.append({"type": output_type, "event": event.get("event")})
                elif output_type == "attention_bias":
                    bias = self._build_attention_bias(rule, output, strength, metrics, anchor_key)
                    attention_biases.append(bias)
                    output_trace.append({"type": output_type, "bias": bias.get("bias")})
                elif output_type == "safety_gate":
                    row = dict(output)
                    row.update({"rule_id": rule.rule_id, "strength": _round4(strength), "anchor_key": anchor_key})
                    safety_gate.append(row)
                    output_trace.append({"type": output_type, "decision": row.get("decision")})
                elif output_type == "trace_log":
                    row = dict(output)
                    row.update({"rule_id": rule.rule_id, "strength": _round4(strength), "anchor_key": anchor_key})
                    trace_logs.append(row)
                    output_trace.append({"type": output_type, "topic": row.get("topic")})
            hit["outputs"] = output_trace

        return {
            "schema_id": "innate_phase_trace/v1",
            "enabled": True,
            "phase": phase_name,
            "tick_index": int(tick_index),
            "rule_count": len(self._rules_by_phase.get(phase_name, [])),
            "hit_count": len(hits),
            "hits": hits,
            "suppressed": suppressed[: max(4, self.max_items_per_phase)],
            "items": items,
            "action_nodes": self._merge_action_nodes(action_nodes),
            "action_biases": self._merge_action_biases(action_biases),
            "emotion_deltas": {key: _round4(value) for key, value in emotion_deltas.items() if abs(float(value)) > 0.0001},
            "learning_events": learning_events,
            "attention_biases": attention_biases,
            "safety_gate": safety_gate,
            "trace_logs": trace_logs,
            "metrics": self._compact_metrics(metrics),
            "fatigue": self.fatigue_snapshot(),
        }

    def actuator_registry(self) -> dict:
        return {key: dict(value) for key, value in ACTUATOR_REGISTRY.items()}

    def action_registry(self) -> dict:
        return {key: dict(value) for key, value in ACTION_NODE_REGISTRY.items()}

    def fatigue_snapshot(self) -> dict:
        top_rules = sorted(self._rule_fatigue.items(), key=lambda item: (-float(item[1]), item[0]))[:8]
        top_anchors = sorted(self._anchor_fatigue.items(), key=lambda item: (-float(item[1]), str(item[0])))[:8]
        return {
            "rule": {key: _round4(value) for key, value in top_rules},
            "anchor": {f"{key[0]}:{key[1]}": _round4(value) for key, value in top_anchors},
        }

    def _coerce_rule(self, row: RuleDef | InnateRule) -> InnateRule:
        if isinstance(row, InnateRule):
            return row
        return InnateRule.from_rule_def(row)

    def _empty_trace(self, phase: str, *, disabled: bool = False) -> dict:
        return {
            "schema_id": "innate_phase_trace/v1",
            "enabled": not disabled,
            "phase": str(phase or ""),
            "hit_count": 0,
            "hits": [],
            "suppressed": [],
            "items": [],
            "action_nodes": [],
            "action_biases": [],
            "emotion_deltas": {},
            "learning_events": [],
            "attention_biases": [],
            "safety_gate": [],
            "trace_logs": [],
            "metrics": {},
            "fatigue": self.fatigue_snapshot(),
        }

    def _advance_tick(self, tick_index: int) -> None:
        if self._last_tick < 0:
            self._last_tick = int(tick_index)
            return
        delta = max(0, int(tick_index) - int(self._last_tick))
        if delta <= 0:
            return
        decay_by_type = {key: float(value.get("decay", 0.8) or 0.8) for key, value in self._fatigue_types.items()}
        # We do not know a rule's fatigue type from the key alone, so use a
        # conservative average decay. Rule firing immediately records the exact
        # type-specific increase/gain.
        avg_decay = sum(decay_by_type.values()) / max(1, len(decay_by_type))
        for key in list(self._rule_fatigue.keys()):
            self._rule_fatigue[key] = _clamp(float(self._rule_fatigue[key]) * (avg_decay**delta), 0.0, 1.0)
            if self._rule_fatigue[key] < 0.0001:
                self._rule_fatigue.pop(key, None)
        for key in list(self._anchor_fatigue.keys()):
            self._anchor_fatigue[key] = _clamp(float(self._anchor_fatigue[key]) * (avg_decay**delta), 0.0, 1.0)
            if self._anchor_fatigue[key] < 0.0001:
                self._anchor_fatigue.pop(key, None)
        self._last_tick = int(tick_index)

    def _fatigue_params(self, fatigue_type: str) -> dict:
        return dict(self._fatigue_types.get(str(fatigue_type or ""), FATIGUE_TYPES["action_internal"]))

    def _combined_fatigue(self, rule: InnateRule, anchor_key: str) -> float:
        return _clamp(
            float(self._rule_fatigue.get(rule.rule_id, 0.0) or 0.0) * 0.62
            + float(self._anchor_fatigue.get((rule.rule_id, anchor_key), 0.0) or 0.0) * 0.38,
            0.0,
            1.0,
        )

    def _record_fire(self, rule: InnateRule, anchor_key: str, strength: float) -> None:
        params = self._fatigue_params(rule.fatigue_type)
        increase = float(params.get("increase", 0.2) or 0.2)
        self._rule_fatigue[rule.rule_id] = _clamp(float(self._rule_fatigue.get(rule.rule_id, 0.0) or 0.0) + strength * increase, 0.0, 1.0)
        key = (rule.rule_id, anchor_key)
        self._anchor_fatigue[key] = _clamp(float(self._anchor_fatigue.get(key, 0.0) or 0.0) + strength * increase * 1.15, 0.0, 1.0)

    def _build_metrics(self, context: dict) -> dict:
        state_items = [dict(item) for item in list(context.get("state_items", []) or []) if isinstance(item, dict)]
        feelings = dict(context.get("feelings", {}) or {})
        channels = dict(feelings.get("channels", {}) or feelings or {})
        prediction = dict(context.get("prediction_trace", {}) or {})
        residual = dict(context.get("residual_summary", {}) or {})
        attention = dict(context.get("attention", {}) or {})
        time_trace = dict(context.get("time_trace", {}) or {})
        rhythm_trace = dict(context.get("rhythm_trace", {}) or {})
        ui_trace = dict(context.get("ui_trace", {}) or {})
        pointer_trace = dict(context.get("pointer_trace", {}) or {})
        expectation_pressure = dict(context.get("expectation_pressure", {}) or {})
        ep_channels = dict(expectation_pressure.get("channels", {}) or {})
        emotion_state = dict(context.get("emotion_state", {}) or {})
        action_feedback = dict(context.get("action_feedback_trace", {}) or {})
        action_trace = dict(context.get("action_trace", {}) or {})
        consequence = dict(context.get("action_consequence_trace", {}) or {})
        runtime_load = dict(context.get("runtime_load_trace", {}) or {})
        runtime_channels = dict(runtime_load.get("channels", {}) or {})

        positive_rows = []
        negative_rows = []
        visual_rows = []
        audio_rows = []
        text_rows = []
        action_rows = []
        feedback_rows = []
        inhibition_rows = []
        for item in state_items:
            label = str(item.get("sa_label", "") or "")
            cp = float(item.get("cognitive_pressure", 0.0) or 0.0)
            if cp > 0:
                positive_rows.append(item)
            elif cp < 0:
                negative_rows.append(item)
            family = str(item.get("family", "") or "")
            source_type = str(item.get("source_type", "") or "")
            if family.startswith("vision") or source_type.startswith("vision") or label.startswith("vision::"):
                visual_rows.append(item)
            if family.startswith("audio") or source_type.startswith("audio") or label.startswith("audio::"):
                audio_rows.append(item)
            if family.startswith("text") or label.startswith(("text::", "phrase::")):
                text_rows.append(item)
            if family == "action" or label.startswith("action::"):
                action_rows.append(item)
            if family == "action_feedback" or label.startswith("action_feedback::"):
                feedback_rows.append(item)
            if family == "action_inhibition" or label.startswith("action_inhibition::"):
                inhibition_rows.append(item)

        positive_mass = sum(max(0.0, float(item.get("cognitive_pressure", 0.0) or 0.0)) for item in positive_rows)
        negative_mass = sum(max(0.0, -float(item.get("cognitive_pressure", 0.0) or 0.0)) for item in negative_rows)
        total_real = sum(max(0.0, float(item.get("real_energy", 0.0) or 0.0)) for item in state_items)
        total_virtual = sum(max(0.0, float(item.get("virtual_energy", 0.0) or 0.0)) for item in state_items)
        top_pos = max(positive_rows, key=lambda item: float(item.get("cognitive_pressure", 0.0) or 0.0), default={})
        top_neg = max(negative_rows, key=lambda item: -float(item.get("cognitive_pressure", 0.0) or 0.0), default={})
        fast_bn = list(context.get("fast_bn", []) or [])
        slow_bn = list(context.get("slow_bn", []) or [])
        bn_rows = fast_bn + slow_bn
        top_bn = max(
            [float(row.get("match_efficiency", row.get("grasp_confidence", row.get("score", 0.0))) or 0.0) for row in bn_rows if isinstance(row, dict)]
            or [0.0]
        )
        bn_entropy_proxy = self._weight_entropy_proxy(bn_rows)
        slow_cn = list(context.get("slow_cn", []) or [])
        fast_cn = list(context.get("fast_cn", []) or [])
        draft_context = dict(context.get("draft_context", {}) or {}) if isinstance(context.get("draft_context", {}), dict) else {}
        predicted_action_energy = self._predicted_action_energy(fast_cn + slow_cn)
        selected_actions = list(action_trace.get("selected_actions", []) or [])
        candidates = list(action_trace.get("candidates", []) or [])
        observed_feedback = dict(action_feedback.get("observed_feedback", {}) or {})
        reward = max(0.0, float(observed_feedback.get("reward", 0.0) or 0.0))
        punishment = max(0.0, float(observed_feedback.get("punishment", 0.0) or 0.0))
        feedback_correctness = max(0.0, float(observed_feedback.get("correctness", 0.0) or 0.0))
        feedback_confidence = _clamp(float(observed_feedback.get("confidence", 0.0) or 0.0), 0.0, 1.0)
        predicted_error = self._action_prediction_error(action_feedback)
        external_risk = self._external_risk(candidates + selected_actions, channels, emotion_state)
        expected_token_label = self._expected_token_label(fast_cn + slow_cn, draft_context=draft_context)
        ui_goal_trace = self._ui_goal_metrics(state_items=state_items, ui_trace=ui_trace, channels=channels)
        click_ready = self._click_ready_strength(
            ui_goal=float(ui_goal_trace.get("ui_goal", 0.0) or 0.0),
            ui_trace=ui_trace,
            pointer_trace=pointer_trace,
            channels=channels,
            observed_feedback=observed_feedback,
            emotion_state=emotion_state,
        )

        return {
            "state_items": state_items,
            "state_item_count": len(state_items),
            "positive_pressure_mass": _clamp(positive_mass / max(1.0, total_real + total_virtual + positive_mass), 0.0, 1.0),
            "negative_pressure_mass": _clamp(negative_mass / max(1.0, total_real + total_virtual + negative_mass), 0.0, 1.0),
            "top_positive_label": str(top_pos.get("sa_label", "") or ""),
            "top_negative_label": str(top_neg.get("sa_label", "") or ""),
            "top_positive_pressure": _clamp(float(top_pos.get("cognitive_pressure", 0.0) or 0.0), 0.0, 1.0),
            "top_negative_pressure": _clamp(-float(top_neg.get("cognitive_pressure", 0.0) or 0.0), 0.0, 1.0),
            "total_real": _round4(total_real),
            "total_virtual": _round4(total_virtual),
            "mismatch_ratio": _clamp(float(prediction.get("mismatch_ratio", 0.0) or 0.0), 0.0, 1.0),
            "alignment_score": _clamp(float(prediction.get("alignment_score", 0.0) or 0.0), 0.0, 1.0),
            "unexpected_count": int(prediction.get("unexpected_count", len(prediction.get("unexpected_labels", []) or [])) or 0),
            "missed_count": int(prediction.get("missed_count", len(prediction.get("missed_predicted_labels", []) or [])) or 0),
            "residual_mass": _clamp(float(residual.get("total_unresolved_mass", 0.0) or 0.0) / max(1.0, 1.0 + len(state_items)), 0.0, 1.0),
            "surprise": _clamp(float(channels.get("surprise", 0.0) or 0.0), 0.0, 1.0),
            "dissonance": _clamp(float(channels.get("dissonance", 0.0) or 0.0), 0.0, 1.0),
            "coherence": _clamp(float(channels.get("coherence", 0.0) or 0.0), 0.0, 1.0),
            "correctness": _clamp(float(channels.get("correctness", 0.0) or 0.0), 0.0, 1.0),
            "grasp": _clamp(max(float(channels.get("grasp", 0.0) or 0.0), top_bn), 0.0, 1.0),
            "expectation": _clamp(max(float(channels.get("expectation", 0.0) or 0.0), float(ep_channels.get("expectation_level", 0.0) or 0.0)), 0.0, 1.0),
            "pressure": _clamp(max(float(channels.get("pressure", 0.0) or 0.0), float(ep_channels.get("pressure_level", 0.0) or 0.0)), 0.0, 1.0),
            "satisfaction": _clamp(float(ep_channels.get("satisfaction_level", 0.0) or 0.0), 0.0, 1.0),
            "expectation_gap": _clamp(float(ep_channels.get("expectation_gap", 0.0) or 0.0), 0.0, 1.0),
            "uncertainty": _clamp(max(bn_entropy_proxy, 1.0 - max(top_bn, float(channels.get("grasp", 0.0) or 0.0))), 0.0, 1.0),
            "timefelt": _clamp(float((time_trace.get("channels", {}) or {}).get("confidence", 0.0) or 0.0), 0.0, 1.0),
            "rhythm_phase": _clamp(float((rhythm_trace.get("channels", {}) or {}).get("phase_expectation", 0.0) or 0.0), 0.0, 1.0),
            "complexity": _clamp(float(runtime_channels.get("complexity", channels.get("complexity", 0.0)) or 0.0), 0.0, 1.0),
            "simplicity": _clamp(float(runtime_channels.get("simplicity", channels.get("simplicity", 0.0)) or 0.0), 0.0, 1.0),
            "fatigue": _clamp(max(list(self._rule_fatigue.values()) or [0.0]), 0.0, 1.0),
            "novelty": _clamp(max(float(channels.get("surprise", 0.0) or 0.0), 1.0 - max(top_bn, float(channels.get("grasp", 0.0) or 0.0))), 0.0, 1.0),
            "transition": _clamp(len(text_rows) / 4.0 + len(audio_rows) / 8.0 + len(visual_rows) / 8.0, 0.0, 1.0),
            "multimodal_binding": _clamp((1.0 if len([x for x in (bool(visual_rows), bool(audio_rows), bool(text_rows)) if x]) >= 2 else 0.0), 0.0, 1.0),
            "visual_surprise": _clamp(max([float(item.get("cognitive_pressure", 0.0) or 0.0) for item in visual_rows] or [0.0]), 0.0, 1.0),
            "visual_motion": _clamp(max([self._motion_strength(item) for item in visual_rows] or [0.0]), 0.0, 1.0),
            "visual_uncertainty": _clamp((1.0 - max(top_bn, float(channels.get("grasp", 0.0) or 0.0))) * (1.0 if visual_rows else 0.0), 0.0, 1.0),
            "audio_surprise": _clamp(max([float(item.get("cognitive_pressure", 0.0) or 0.0) for item in audio_rows] or [0.0]), 0.0, 1.0),
            "voice_like": _clamp(max([self._voice_like_strength(item) for item in audio_rows] or [0.0]), 0.0, 1.0),
            "audio_low_grasp": _clamp((1.0 - max(top_bn, float(channels.get("grasp", 0.0) or 0.0))) * (1.0 if audio_rows else 0.0), 0.0, 1.0),
            "text_mismatch": _clamp(float(channels.get("dissonance", 0.0) or 0.0) * (1.0 if text_rows or candidates else 0.65), 0.0, 1.0),
            "expected_token": _clamp(self._expected_token_strength(fast_cn + slow_cn, draft_context=draft_context), 0.0, 1.0),
            "expected_token_label": expected_token_label,
            "text_revision": _clamp(float(channels.get("dissonance", 0.0) or 0.0) + feedback_correctness * 0.2, 0.0, 1.0),
            "text_commit_ready": _clamp(float(channels.get("correctness", 0.0) or 0.0) - float(channels.get("pressure", 0.0) or 0.0) * 0.6, 0.0, 1.0),
            "pressure_external_candidate": _clamp(float(channels.get("pressure", 0.0) or 0.0) * (1.0 if self._has_external_candidate(candidates) else 0.0), 0.0, 1.0),
            "ui_goal": _clamp(float(ui_goal_trace.get("ui_goal", 0.0) or 0.0), 0.0, 1.0),
            "ui_target_x": _clamp(float(ui_goal_trace.get("ui_target_x", 0.5) or 0.5), 0.0, 1.0),
            "ui_target_y": _clamp(float(ui_goal_trace.get("ui_target_y", 0.5) or 0.5), 0.0, 1.0),
            "ui_target_label": str(ui_goal_trace.get("ui_target_label", "") or ""),
            "click_ready": click_ready,
            "hard_task": _clamp(float(runtime_channels.get("complexity", 0.0) or 0.0) * (1.0 - max(top_bn, float(channels.get("grasp", 0.0) or 0.0))), 0.0, 1.0),
            "external_risk": external_risk,
            "risk": max(external_risk, float(channels.get("pressure", 0.0) or 0.0), float(emotion_state.get("COR", 0.0) or 0.0)),
            "reward": _clamp(reward + feedback_correctness * 0.35, 0.0, 1.0),
            "punishment": _clamp(punishment, 0.0, 1.0),
            "relief": _clamp(reward * 0.35 + feedback_correctness * 0.25 + float(ep_channels.get("satisfaction_level", 0.0) or 0.0), 0.0, 1.0),
            "sustained_pressure": _clamp(float(channels.get("pressure", 0.0) or 0.0) + punishment * 0.5 - reward * 0.4, 0.0, 1.0),
            "safe_validation": _clamp(feedback_correctness + reward * 0.3 - punishment * 0.5, 0.0, 1.0),
            "familiarity": _clamp(max(top_bn, float(channels.get("grasp", 0.0) or 0.0), float(channels.get("coherence", 0.0) or 0.0)), 0.0, 1.0),
            "action_feedback": 1.0 if action_feedback.get("applied") else 0.0,
            "positive_action_feedback": _clamp(reward + feedback_correctness * 0.35, 0.0, 1.0),
            "negative_action_feedback": _clamp(punishment, 0.0, 1.0),
            "action_prediction_error": predicted_error,
            "successor_action_feedback": _clamp(float(consequence.get("supported_action_count", 0) or 0) / 3.0, 0.0, 1.0),
            "memory_predicted_action": predicted_action_energy,
            "action_selected": 1.0 if selected_actions else 0.0,
            "action_inhibition": 1.0 if inhibition_rows else 0.0,
            "feedback_confidence": feedback_confidence,
            "top_action_id": str((selected_actions[0] if selected_actions else {}).get("action_id", "") or ""),
        }

    def _condition_strength(self, condition: str, metrics: dict) -> float:
        name = str(condition or "")
        if name == "positive_pressure":
            return max(float(metrics.get("positive_pressure_mass", 0.0) or 0.0), float(metrics.get("top_positive_pressure", 0.0) or 0.0))
        if name == "negative_pressure":
            return max(float(metrics.get("negative_pressure_mass", 0.0) or 0.0), float(metrics.get("top_negative_pressure", 0.0) or 0.0))
        if name == "alignment":
            return float(metrics.get("alignment_score", 0.0) or 0.0)
        if name == "coherence":
            return max(float(metrics.get("coherence", 0.0) or 0.0), float(metrics.get("alignment_score", 0.0) or 0.0) * 0.72)
        if name == "continue_focus":
            return _clamp((float(metrics.get("grasp", 0.0) or 0.0) + float(metrics.get("expectation", 0.0) or 0.0) + float(metrics.get("rhythm_phase", 0.0) or 0.0)) / 2.2, 0.0, 1.0)
        if name == "inspect_residual":
            return max(float(metrics.get("dissonance", 0.0) or 0.0), float(metrics.get("pressure", 0.0) or 0.0), float(metrics.get("residual_mass", 0.0) or 0.0))
        if name == "social_reward":
            return float(metrics.get("reward", 0.0) or 0.0) * 0.6
        if name == "social_punishment":
            return float(metrics.get("punishment", 0.0) or 0.0) * 0.6
        return _clamp(float(metrics.get(name, 0.0) or 0.0), 0.0, 1.0)

    def _build_item(self, rule: InnateRule, output: dict, strength: float, metrics: dict, anchor_key: str, tick_index: int) -> dict:
        label = str(output.get("label", "") or f"innate_rule::{rule.rule_id}")
        family = str(output.get("family", "") or "innate")
        return {
            "sa_label": label,
            "display_text": str(output.get("display_text", "") or label),
            "family": family,
            "source_type": "innate_rule",
            "real_energy": _round4(strength),
            "anchor_meta": {
                "schema_id": "innate_rule_emission/v1",
                "tick_index": int(tick_index),
                "rule_id": rule.rule_id,
                "condition": rule.condition,
                "fatigue_type": rule.fatigue_type,
                "anchor_key": anchor_key,
                "strength": _round4(strength),
                "top_positive_label": metrics.get("top_positive_label", ""),
                "top_negative_label": metrics.get("top_negative_label", ""),
            },
        }

    def _build_action_node(self, rule: InnateRule, output: dict, strength: float, metrics: dict, tick_index: int) -> dict:
        action_id = str(output.get("action_id", "") or "")
        meta = action_meta(action_id)
        base_drive = float(output.get("drive", 0.0) or 0.0)
        drive = _round4(base_drive * strength)
        return {
            "schema_id": "innate_action_node/v1",
            "action_id": action_id,
            "actuator_id": str(meta.get("actuator_id", "") or "actuator::legacy_internal"),
            "raw_drive": _round4(base_drive),
            "drive": drive,
            "strength": _round4(strength),
            "rule_id": rule.rule_id,
            "fatigue_type": rule.fatigue_type,
            "params": self._action_params(action_id=action_id, output=output, metrics=metrics),
            "external": is_external_action(action_id, str(meta.get("actuator_id", "") or "")),
            "base_threshold": float(meta.get("base_threshold", 0.0) or 0.0),
            "tick_index": int(tick_index),
            "notes": [f"innate_rule={rule.rule_id}", f"condition={rule.condition}"],
        }

    def _build_action_bias(self, rule: InnateRule, output: dict, strength: float, metrics: dict, tick_index: int) -> dict:
        action_id = str(output.get("action_id", "") or "")
        meta = action_meta(action_id)
        base_delta = float(output.get("drive", output.get("drive_delta", 0.0)) or 0.0)
        drive_delta = _round4(base_delta * strength)
        return {
            "schema_id": "innate_action_bias/v1",
            "action_id": action_id,
            "actuator_id": str(meta.get("actuator_id", "") or "actuator::legacy_internal"),
            "raw_drive_delta": _round4(base_delta),
            "drive_delta": drive_delta,
            "strength": _round4(strength),
            "rule_id": rule.rule_id,
            "fatigue_type": rule.fatigue_type,
            "external": is_external_action(action_id, str(meta.get("actuator_id", "") or "")),
            "base_threshold": float(meta.get("base_threshold", 0.0) or 0.0),
            "tick_index": int(tick_index),
            "notes": [f"innate_rule={rule.rule_id}", f"condition={rule.condition}", "innate_action_bias"],
        }

    def _build_attention_bias(self, rule: InnateRule, output: dict, strength: float, metrics: dict, anchor_key: str) -> dict:
        bias = dict(output)
        bias_type = str(bias.get("bias", "") or "")
        top_positive = str(metrics.get("top_positive_label", "") or "")
        top_negative = str(metrics.get("top_negative_label", "") or "")
        target_labels: list[str] = []
        if bias_type == "surprise_anchor" and top_positive:
            target_labels.append(top_positive)
        elif bias_type == "mismatch_pair":
            target_labels.extend([label for label in (top_positive, top_negative) if label])
        elif anchor_key and anchor_key != "global":
            target_labels.append(anchor_key)
        bias.update(
            {
                "schema_id": "innate_attention_bias/v1",
                "rule_id": rule.rule_id,
                "strength": _round4(strength),
                "anchor_key": anchor_key,
                "target_labels": list(dict.fromkeys(target_labels)),
                "top_positive_label": top_positive,
                "top_negative_label": top_negative,
                "policy": "attention_score_bonus_only;does_not_create_or_delete_SA",
            }
        )
        return bias

    def _build_learning_event(self, rule: InnateRule, output: dict, strength: float, metrics: dict, tick_index: int) -> dict:
        return {
            "schema_id": "innate_learning_event/v1",
            "event": str(output.get("event", "") or "trace"),
            "rule_id": rule.rule_id,
            "phase": rule.phase,
            "strength": _round4(strength),
            "tick_index": int(tick_index),
            "source": "innate_rule_trace",
            "top_positive_label": metrics.get("top_positive_label", ""),
            "top_negative_label": metrics.get("top_negative_label", ""),
            "top_action_id": metrics.get("top_action_id", ""),
            "policy": "audit_only_unless_consumed_by_dedicated_learning_layer",
        }

    def _merge_action_nodes(self, nodes: list[dict]) -> list[dict]:
        merged: dict[str, dict] = {}
        for node in nodes:
            action_id = str(node.get("action_id", "") or "")
            if not action_id:
                continue
            existing = merged.get(action_id)
            if existing is None:
                merged[action_id] = dict(node)
                continue
            existing["drive"] = _round4(float(existing.get("drive", 0.0) or 0.0) + float(node.get("drive", 0.0) or 0.0))
            existing["raw_drive"] = _round4(float(existing.get("raw_drive", 0.0) or 0.0) + float(node.get("raw_drive", 0.0) or 0.0))
            existing.setdefault("notes", [])
            existing["notes"] = list(existing.get("notes", []) or []) + list(node.get("notes", []) or [])
            existing.setdefault("rule_ids", [])
            ids = set(existing.get("rule_ids", []) or [existing.get("rule_id")])
            ids.add(str(node.get("rule_id", "") or ""))
            existing["rule_ids"] = sorted(x for x in ids if x)
        rows = list(merged.values())
        rows.sort(key=lambda item: (-float(item.get("drive", 0.0) or 0.0), str(item.get("action_id", "") or "")))
        return rows[: self.max_action_nodes_per_phase]

    def _merge_action_biases(self, biases: list[dict]) -> list[dict]:
        merged: dict[str, dict] = {}
        for bias in biases:
            action_id = str(bias.get("action_id", "") or "")
            if not action_id:
                continue
            existing = merged.get(action_id)
            if existing is None:
                merged[action_id] = dict(bias)
                continue
            existing["drive_delta"] = _round4(float(existing.get("drive_delta", 0.0) or 0.0) + float(bias.get("drive_delta", 0.0) or 0.0))
            existing["raw_drive_delta"] = _round4(float(existing.get("raw_drive_delta", 0.0) or 0.0) + float(bias.get("raw_drive_delta", 0.0) or 0.0))
            existing["notes"] = list(existing.get("notes", []) or []) + list(bias.get("notes", []) or [])
            ids = set(existing.get("rule_ids", []) or [existing.get("rule_id")])
            ids.add(str(bias.get("rule_id", "") or ""))
            existing["rule_ids"] = sorted(row for row in ids if row)
        rows = list(merged.values())
        rows.sort(key=lambda item: (-abs(float(item.get("drive_delta", 0.0) or 0.0)), str(item.get("action_id", "") or "")))
        return rows[: self.max_action_nodes_per_phase]

    def _anchor_key(self, anchor: str, metrics: dict) -> str:
        key = str(anchor or "global")
        if key == "top_positive_pressure":
            return str(metrics.get("top_positive_label", "") or "global")
        if key == "top_negative_pressure":
            return str(metrics.get("top_negative_label", "") or "global")
        return key

    def _compact_metrics(self, metrics: dict) -> dict:
        keys = (
            "positive_pressure_mass",
            "negative_pressure_mass",
            "mismatch_ratio",
            "alignment_score",
            "surprise",
            "dissonance",
            "coherence",
            "correctness",
            "grasp",
            "expectation",
            "pressure",
            "uncertainty",
            "timefelt",
            "rhythm_phase",
            "complexity",
            "simplicity",
            "novelty",
            "reward",
            "punishment",
            "external_risk",
            "pressure_external_candidate",
            "ui_goal",
            "click_ready",
            "action_feedback",
            "action_prediction_error",
            "successor_action_feedback",
            "memory_predicted_action",
            "action_selected",
        )
        return {key: _round4(metrics.get(key, 0.0) or 0.0) for key in keys}

    def _weight_entropy_proxy(self, rows: list[dict]) -> float:
        weights = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            weight = float(row.get("normalized_weight", row.get("score", 0.0)) or 0.0)
            if weight > 0:
                weights.append(weight)
        if len(weights) <= 1:
            return 0.0
        total = sum(weights)
        if total <= 0:
            return 0.0
        probs = [weight / total for weight in weights]
        peak = max(probs)
        return _clamp(1.0 - peak, 0.0, 1.0)

    def _predicted_action_energy(self, branches: list[dict]) -> float:
        total = 0.0
        for branch in branches or []:
            for item in list((branch or {}).get("predicted_items", []) or []):
                label = str((item or {}).get("sa_label", "") or "")
                if label.startswith("action::"):
                    total += max(0.0, float((item or {}).get("virtual_energy", 0.0) or 0.0))
        return _clamp(total, 0.0, 1.0)

    def _expected_token_strength(self, branches: list[dict], *, draft_context: dict | None = None) -> float:
        for branch in branches or []:
            for item in list((branch or {}).get("predicted_items", []) or []):
                label = str((item or {}).get("sa_label", "") or "")
                if label.startswith("text::") and self._is_output_expected_token_item(item, draft_context=draft_context):
                    return _clamp(float((item or {}).get("virtual_energy", 0.2) or 0.2), 0.0, 1.0)
        return 0.0

    def _expected_token_label(self, branches: list[dict], *, draft_context: dict | None = None) -> str:
        best_label = ""
        best_energy = 0.0
        for branch in branches or []:
            for item in list((branch or {}).get("predicted_items", []) or []):
                label = str((item or {}).get("sa_label", "") or "")
                if not label.startswith("text::"):
                    continue
                if not self._is_output_expected_token_item(item, draft_context=draft_context):
                    continue
                energy = float((item or {}).get("virtual_energy", 0.0) or 0.0)
                if not best_label or energy > best_energy:
                    best_label = label
                    best_energy = energy
        return best_label.split("::", 1)[-1] if best_label else ""

    def _prediction_item_meta(self, item: dict) -> dict:
        meta = dict((item or {}).get("anchor_meta", {}) or {}) if isinstance((item or {}).get("anchor_meta", {}), dict) else {}
        for key in ("source_type", "family", "sa_kind"):
            if key in (item or {}) and key not in meta:
                meta[key] = (item or {}).get(key)
        return meta

    def _has_output_text_process_evidence(self, meta: dict) -> bool:
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
                "predicted_text_payload_from_process_companion/v1",
            }
            or readout_role == "reply_char_slot"
            or source in {"action::text_insert", "action::text_reread", "text_actuator_direct_replace"}
            or source_event_type in {"draft_read_token", "insert", "replace", "write_revision", "visible_draft_token"}
            or priority.startswith(("current_glyph", "previous_prefix"))
        )

    def _is_external_input_text_meta(self, meta: dict) -> bool:
        source_type = str(meta.get("source_type", "") or "")
        source = str(meta.get("source", "") or "")
        notes = {str(note or "") for note in list(meta.get("notes", []) or [])}
        return bool(
            source_type in {"external_text", "external_text_readback", "external_teacher"}
            or source in {"external_text", "external_text_turn"}
            or "external_text_read_into_input_channel" in notes
            or "not_ap_visible_draft" in notes
        )

    def _is_output_expected_token_item(self, item: dict, *, draft_context: dict | None = None) -> bool:
        meta = self._prediction_item_meta(item)
        output_process = self._has_output_text_process_evidence(meta)
        if self._is_external_input_text_meta(meta) and not output_process:
            return False
        if not output_process:
            return False
        token = str((item or {}).get("sa_label", "") or "").split("::", 1)[-1]
        return self._text_token_position_aligned(token=token, meta=meta, draft_context=draft_context)

    def _text_token_position_aligned(self, *, token: str, meta: dict, draft_context: dict | None = None) -> bool:
        draft = dict(draft_context or {})
        visible_text = str(draft.get("visible_text", "") or "")
        try:
            visible_length = int(draft.get("visible_length", len(visible_text)) or len(visible_text))
        except (TypeError, ValueError):
            visible_length = len(visible_text)

        previous_prefix = str(meta.get("previous_prefix", "") or meta.get("visible_text_before", "") or "")
        if previous_prefix and previous_prefix != visible_text:
            return False

        variant_text = str(meta.get("variant_text", "") or meta.get("expected_text", "") or "")
        if variant_text:
            if not variant_text.startswith(visible_text):
                return False
            if not variant_text[len(visible_text) :].startswith(str(token or "")):
                return False

        position_keys = ("current_glyph_index", "position", "cursor_before", "cursor", "cursor_index")
        has_position = False
        for key in position_keys:
            if key not in meta:
                continue
            try:
                position = int(meta.get(key))
            except (TypeError, ValueError):
                return False
            has_position = True
            if position != visible_length:
                return False

        if "visible_length" in meta:
            try:
                meta_visible_length = int(meta.get("visible_length"))
            except (TypeError, ValueError):
                return False
            has_position = True
            if meta_visible_length != visible_length:
                return False

        return has_position

    def _motion_strength(self, item: dict) -> float:
        numeric = dict((item or {}).get("numeric_features", {}) or {})
        meta = dict((item or {}).get("anchor_meta", {}) or {})
        payload = dict((item or {}).get("reconstruction_payload", {}) or {})
        for source in (numeric, meta, payload):
            for key in ("motion", "motion_vector", "flow", "speed"):
                value = source.get(key)
                if isinstance(value, (int, float)):
                    return _clamp(float(value), 0.0, 1.0)
                if isinstance(value, (list, tuple)) and value:
                    return _clamp(sum(abs(float(x or 0.0)) for x in value[:2]), 0.0, 1.0)
        return 0.0

    def _voice_like_strength(self, item: dict) -> float:
        numeric = dict((item or {}).get("numeric_features", {}) or {})
        meta = dict((item or {}).get("anchor_meta", {}) or {})
        for source in (numeric, meta):
            for key in ("voice_like", "harmonicity", "pitch_confidence"):
                value = source.get(key)
                if isinstance(value, (int, float)):
                    return _clamp(float(value), 0.0, 1.0)
        return 0.0

    def _ui_goal_metrics(self, *, state_items: list[dict], ui_trace: dict, channels: dict) -> dict:
        explicit = dict(ui_trace.get("target", {}) or ui_trace.get("ui_target", {}) or {})
        best = {
            "ui_goal": _clamp(float(explicit.get("confidence", explicit.get("score", 0.0)) or 0.0), 0.0, 1.0),
            "ui_target_x": _clamp(float(explicit.get("x", 0.5) or 0.5), 0.0, 1.0),
            "ui_target_y": _clamp(float(explicit.get("y", 0.5) or 0.5), 0.0, 1.0),
            "ui_target_label": str(explicit.get("label", explicit.get("sa_label", "")) or ""),
        }
        for item in state_items or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            family = str(item.get("family", "") or "")
            source_type = str(item.get("source_type", "") or "")
            meta = dict(item.get("anchor_meta", {}) or {})
            numeric = dict(item.get("numeric_features", {}) or {})
            ui_role = str(meta.get("ui_role", meta.get("role", "")) or "")
            is_ui = (
                bool(meta.get("ui_target", False))
                or bool(numeric.get("ui.target", 0.0))
                or label.startswith(("ui::", "vision_ui::"))
                or "ui" in family
                or "ui" in source_type
                or ui_role in {"button", "link", "input", "target", "menu_item"}
            )
            if not is_ui:
                continue
            bbox = list(meta.get("bbox_norm", []) or [])
            if len(bbox) >= 4:
                x = _clamp(float(bbox[0]), 0.0, 1.0)
                y = _clamp(float(bbox[1]), 0.0, 1.0)
            else:
                x = _clamp(float(meta.get("x", 0.5) or 0.5), 0.0, 1.0)
                y = _clamp(float(meta.get("y", 0.5) or 0.5), 0.0, 1.0)
            numeric_target = numeric.get("ui.target", 0.0)
            confidence = max(
                float(meta.get("confidence", 0.0) or 0.0),
                float(numeric_target or 0.0) if isinstance(numeric_target, (int, float)) else 0.0,
                float(item.get("real_energy", 0.0) or 0.0) * 0.55,
            )
            score = _clamp(
                confidence
                + float(channels.get("expectation", 0.0) or 0.0) * 0.18
                + float(channels.get("correctness", 0.0) or 0.0) * 0.10,
                0.0,
                1.0,
            )
            if score > float(best.get("ui_goal", 0.0) or 0.0):
                best = {"ui_goal": score, "ui_target_x": x, "ui_target_y": y, "ui_target_label": label}
        pressure = float(channels.get("pressure", 0.0) or 0.0)
        if pressure > 0.0:
            best["ui_goal"] = _clamp(float(best.get("ui_goal", 0.0) or 0.0) - pressure * 0.22, 0.0, 1.0)
        return best

    def _click_ready_strength(self, *, ui_goal: float, ui_trace: dict, pointer_trace: dict, channels: dict, observed_feedback: dict, emotion_state: dict) -> float:
        pointer_on_target = _clamp(
            float(pointer_trace.get("on_target", pointer_trace.get("target_alignment", ui_trace.get("pointer_on_target", 0.0))) or 0.0),
            0.0,
            1.0,
        )
        safe = _clamp(
            float(ui_trace.get("safe_to_click", ui_trace.get("click_safe", 0.0)) or 0.0)
            + float(channels.get("correctness", 0.0) or 0.0) * 0.28
            + float(observed_feedback.get("correctness", 0.0) or 0.0) * 0.18
            + float(observed_feedback.get("reward", 0.0) or 0.0) * 0.12
            - float(channels.get("pressure", 0.0) or 0.0) * 0.52
            - float(emotion_state.get("COR", 0.0) or 0.0) * 0.34,
            0.0,
            1.0,
        )
        return _clamp(ui_goal * 0.38 + pointer_on_target * 0.34 + safe * 0.42, 0.0, 1.0)

    def _has_external_candidate(self, candidates: list[dict]) -> bool:
        for row in candidates or []:
            action_id = str((row or {}).get("action_id", "") or "")
            actuator_id = str((row or {}).get("actuator_id", "") or "")
            if is_external_action(action_id, actuator_id):
                return True
        return False

    def _external_risk(self, candidates: list[dict], channels: dict, emotion_state: dict) -> float:
        risk = 0.0
        for row in candidates or []:
            if not isinstance(row, dict):
                continue
            action_id = str(row.get("action_id", "") or "")
            actuator_id = str(row.get("actuator_id", "") or "")
            if not is_external_action(action_id, actuator_id):
                continue
            predicted = dict(row.get("predicted_outcome", {}) or {})
            risk = max(
                risk,
                float(predicted.get("punishment", 0.0) or 0.0),
                float(predicted.get("pressure", 0.0) or 0.0),
                1.0 - float(predicted.get("confidence", 0.0) or 0.0),
            )
        return _clamp(max(risk, float(channels.get("pressure", 0.0) or 0.0) * 0.6, float(emotion_state.get("COR", 0.0) or 0.0) * 0.72), 0.0, 1.0)

    def _action_params(self, *, action_id: str, output: dict, metrics: dict) -> dict:
        params = dict(output.get("params", {}) or {})
        if action_id == "action::text_insert" and not any(key in params for key in ("token", "text")):
            token = str(metrics.get("expected_token_label", "") or "")
            if token:
                params["token"] = token
        elif action_id == "action::text_replace" and "new_text" not in params:
            token = str(metrics.get("expected_token_label", "") or "")
            if token:
                params["new_text"] = token
        elif action_id == "action::pointer_move":
            params.setdefault("x", _round4(metrics.get("ui_target_x", 0.5) or 0.5))
            params.setdefault("y", _round4(metrics.get("ui_target_y", 0.5) or 0.5))
            target = str(metrics.get("ui_target_label", "") or "")
            if target:
                params.setdefault("target", target)
        elif action_id == "action::pointer_click":
            params.setdefault("button", "left")
            target = str(metrics.get("ui_target_label", "") or "")
            if target:
                params.setdefault("target", target)
        return params

    def _action_prediction_error(self, action_feedback: dict) -> float:
        if not action_feedback.get("applied"):
            return 0.0
        selected = list(action_feedback.get("selected_actions", []) or [])
        observed = dict(action_feedback.get("observed_feedback", {}) or {})
        if not selected or not observed:
            return 0.0
        predicted = dict((selected[0] if isinstance(selected[0], dict) else {}).get("predicted_outcome", {}) or {})
        if not predicted:
            return 0.0
        obs_utility = float(observed.get("reward", 0.0) or 0.0) + float(observed.get("correctness", 0.0) or 0.0) * 0.42 - float(observed.get("punishment", 0.0) or 0.0) * 1.08
        pred_utility = float(predicted.get("reward", 0.0) or 0.0) + float(predicted.get("correctness", 0.0) or 0.0) * 0.42 - float(predicted.get("punishment", 0.0) or 0.0) * 1.08
        return _clamp(abs(obs_utility - pred_utility), 0.0, 1.0)
