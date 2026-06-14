from __future__ import annotations

from collections import Counter


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


class ActionControlEffectRouter:
    """
    Translate selected AP action SA nodes into bounded cognitive control effects.

    This is deliberately not a generic "call the action function" layer. In AP
    terms, an action is a state-pool event that won drive competition. The effect
    router only turns that win into short-lived control SA / attention hints, so
    the next tick still has to pass through normal state-pool competition, B/C
    recall, attention selection, and feedback learning.
    """

    ATTENTION_ACTIONS = {
        "action::focus_anchor",
        "action::inspect_residual",
        "action::continue_focus",
        "action::release_focus",
        "action::diverge_attention",
    }
    TEXT_DRAFT_ACTIONS = {
        "action::text_insert",
        "action::text_reread",
        "action::text_replace",
    }

    def build(
        self,
        *,
        tick_index: int,
        selected_actions: list[dict],
        attention_trace: dict,
        state_snapshot_items: list[dict],
        prediction_trace: dict | None = None,
        residual_summary: dict | None = None,
        previous_focus_labels: list[str] | None = None,
    ) -> dict:
        trace = {
            "schema_id": "action_control_effect_router/v1",
            "tick_index": int(tick_index),
            "control_items": [],
            "attention_controls": [],
            "slow_query_hints": [],
            "family_budget_modulation": {},
            "effects": [],
        }
        if not selected_actions:
            return trace

        state_labels = {
            str(item.get("sa_label", "") or "")
            for item in state_snapshot_items or []
            if str(item.get("sa_label", "") or "")
        }
        label_to_row = {
            str(item.get("sa_label", "") or ""): dict(item)
            for item in state_snapshot_items or []
            if str(item.get("sa_label", "") or "")
        }
        attention_rank = {
            str(row.get("sa_label", "") or ""): idx
            for idx, row in enumerate(list((attention_trace or {}).get("ranked_items", []) or []))
            if str(row.get("sa_label", "") or "")
        }
        selected_labels = [
            str(label or "")
            for label in list((attention_trace or {}).get("selected_labels", []) or [])
            if str(label or "")
        ]
        residual_pairs = self._residual_pairs(prediction_trace or {}, residual_summary or {})

        for action in selected_actions:
            action_id = str((action or {}).get("action_id", "") or "")
            strength = self._action_strength(action)
            if action_id in self.TEXT_DRAFT_ACTIONS:
                effect = self._text_draft_surface_effect(
                    tick_index=tick_index,
                    action=action,
                    strength=strength,
                    state_labels=state_labels,
                    label_to_row=label_to_row,
                )
            elif action_id not in self.ATTENTION_ACTIONS:
                continue
            elif action_id == "action::focus_anchor":
                effect = self._focus_anchor_effect(
                    tick_index=tick_index,
                    action=action,
                    strength=strength,
                    state_labels=state_labels,
                    label_to_row=label_to_row,
                    residual_pairs=residual_pairs,
                    attention_rank=attention_rank,
                    selected_labels=selected_labels,
                    prediction_trace=prediction_trace or {},
                    residual_summary=residual_summary or {},
                )
            elif action_id == "action::inspect_residual":
                effect = self._inspect_residual_effect(
                    tick_index=tick_index,
                    action=action,
                    strength=strength,
                    residual_pairs=residual_pairs,
                    state_labels=state_labels,
                    prediction_trace=prediction_trace or {},
                    residual_summary=residual_summary or {},
                )
            elif action_id == "action::continue_focus":
                effect = self._continue_focus_effect(
                    tick_index=tick_index,
                    action=action,
                    strength=strength,
                    selected_labels=selected_labels,
                    previous_focus_labels=previous_focus_labels or [],
                )
            elif action_id == "action::release_focus":
                effect = self._release_focus_effect(
                    tick_index=tick_index,
                    action=action,
                    strength=strength,
                    selected_labels=selected_labels,
                    previous_focus_labels=previous_focus_labels or [],
                )
            elif action_id == "action::diverge_attention":
                effect = self._diverge_attention_effect(tick_index=tick_index, action=action, strength=strength)
            else:
                continue
            if not effect:
                continue
            trace["effects"].append(effect)
            trace["control_items"].extend(effect.get("control_items", []) or [])
            trace["attention_controls"].extend(effect.get("attention_controls", []) or [])
            trace["slow_query_hints"].extend(effect.get("slow_query_hints", []) or [])
            modulation = dict(effect.get("family_budget_modulation", {}) or {})
            if modulation:
                trace["family_budget_modulation"] = self._merge_family_modulation(
                    trace.get("family_budget_modulation", {}),
                    modulation,
                )
        return self._dedupe_trace(trace)

    def _action_strength(self, action: dict) -> float:
        decisiveness = float(action.get("effective_decisiveness", 0.0) or 0.0)
        drive = float(action.get("drive", 0.0) or 0.0)
        threshold = float(action.get("effective_threshold", 0.0) or 0.0)
        if decisiveness <= 0.0 and drive > threshold:
            decisiveness = drive - threshold
        innate_strength = max(
            [
                float(node.get("strength", 0.0) or 0.0)
                for node in list(action.get("innate_nodes", []) or [])
                if isinstance(node, dict)
            ]
            or [0.0]
        )
        return _clamp(0.18 + decisiveness * 0.55 + innate_strength * 0.18, 0.12, 0.85)

    def _focus_anchor_effect(
        self,
        *,
        tick_index: int,
        action: dict,
        strength: float,
        state_labels: set[str],
        label_to_row: dict[str, dict],
        residual_pairs: list[dict],
        attention_rank: dict[str, int],
        selected_labels: list[str],
        prediction_trace: dict,
        residual_summary: dict,
    ) -> dict:
        targets = self._focus_anchor_targets(
            action=action,
            state_labels=state_labels,
            label_to_row=label_to_row,
            residual_pairs=residual_pairs,
            attention_rank=attention_rank,
            selected_labels=selected_labels,
            prediction_trace=prediction_trace,
            residual_summary=residual_summary,
        )
        return self._single_control_effect(
            tick_index=tick_index,
            action_id="action::focus_anchor",
            control_label="control::attention_anchor",
            control_kind="focus_anchor",
            boost_labels=[label for label in targets if label in state_labels],
            slow_query_labels=[label for label in targets if label in state_labels],
            missing_labels=[label for label in targets if label not in state_labels],
            strength=_clamp(strength * 1.0, 0.12, 0.85),
            ttl=1,
            reason="surprise_or_novelty_anchor_focus",
            extra_meta={"target_labels": targets},
        )

    def _inspect_residual_effect(
        self,
        *,
        tick_index: int,
        action: dict,
        strength: float,
        residual_pairs: list[dict],
        state_labels: set[str],
        prediction_trace: dict,
        residual_summary: dict,
    ) -> dict:
        explicit = [
            str(label or "")
            for label in list(dict(action.get("params", {}) or {}).get("residual_labels", []) or [])
            if str(label or "")
        ]
        if explicit and not residual_pairs:
            residual_pairs = [{"predicted": "", "actual": label, "pair_quality": "explicit_single_sided"} for label in explicit[:4]]
        labels: list[str] = []
        for pair in residual_pairs:
            for key in ("predicted", "actual"):
                label = str(pair.get(key, "") or "")
                if label and label not in labels:
                    labels.append(label)
        if not labels:
            labels = explicit[:4]
        present_labels = [label for label in labels if label in state_labels]
        return self._single_control_effect(
            tick_index=tick_index,
            action_id="action::inspect_residual",
            control_label="control::residual_inspection",
            control_kind="inspect_residual",
            boost_labels=present_labels,
            slow_query_labels=present_labels,
            missing_labels=[label for label in labels if label not in state_labels],
            strength=_clamp(strength * 1.10, 0.14, 0.88),
            ttl=2 if len(residual_pairs) >= 2 else 1,
            reason="dissonance_residual_pairing",
            extra_meta={
                "paired_labels": residual_pairs[:4],
                "pair_quality": "paired" if any(pair.get("predicted") and pair.get("actual") for pair in residual_pairs) else "single_sided",
                "mismatch_ratio": _round4(float(prediction_trace.get("mismatch_ratio", 0.0) or 0.0)),
                "residual_mass": _round4(float(residual_summary.get("total_unresolved_mass", 0.0) or 0.0)),
            },
        )

    def _continue_focus_effect(
        self,
        *,
        tick_index: int,
        action: dict,
        strength: float,
        selected_labels: list[str],
        previous_focus_labels: list[str],
    ) -> dict:
        params = dict(action.get("params", {}) or {})
        explicit: list[str] = []
        for key in ("target_labels", "source_focus_labels", "boost_labels"):
            value = params.get(key)
            if isinstance(value, (list, tuple)):
                explicit.extend(str(label or "") for label in value if str(label or ""))
        labels = self._unique_labels(explicit + selected_labels + previous_focus_labels)[:6]
        return self._single_control_effect(
            tick_index=tick_index,
            action_id="action::continue_focus",
            control_label="control::focus_hold",
            control_kind="continue_focus",
            boost_labels=labels,
            slow_query_labels=labels,
            missing_labels=[],
            strength=_clamp(strength * 0.85, 0.10, 0.72),
            ttl=1,
            reason="slow_system_focus_continuation",
            extra_meta={
                "source_focus_labels": labels,
                "hold_window": 1,
                "successor_confidence": _round4(float((action.get("predicted_outcome", {}) or {}).get("confidence", 0.0) or 0.0)),
            },
        )

    def _release_focus_effect(
        self,
        *,
        tick_index: int,
        action: dict,
        strength: float,
        selected_labels: list[str],
        previous_focus_labels: list[str],
    ) -> dict:
        params = dict(action.get("params", {}) or {})
        explicit: list[str] = []
        for key in ("suppress_labels", "release_labels", "source_focus_labels"):
            value = params.get(key)
            if isinstance(value, (list, tuple)):
                explicit.extend(str(label or "") for label in value if str(label or ""))
        labels = self._unique_labels(explicit)[:6]
        if not labels:
            labels = self._repeated_labels(selected_labels=selected_labels, previous_focus_labels=previous_focus_labels)
        if not labels:
            labels = self._unique_labels(previous_focus_labels + selected_labels)[:4]
        effect = self._single_control_effect(
            tick_index=tick_index,
            action_id="action::release_focus",
            control_label="control::focus_release",
            control_kind="release_focus",
            boost_labels=[],
            slow_query_labels=[],
            suppress_labels=labels,
            missing_labels=[],
            strength=_clamp(strength * 1.0, 0.12, 0.82),
            ttl=2,
            reason="focus_fatigue_release",
            extra_meta={"suppressed_labels": labels, "release_reason": "focus_fatigue"},
        )
        effect["family_budget_modulation"] = {
            "schema_id": "focus_family_budget_modulation/v1",
            "control_kind": "release_focus",
            "source_action_id": "action::release_focus",
            "ttl": 2,
            "diversity_gain": _round4(min(0.28, strength * 0.18)),
            "release_labels": labels,
        }
        return effect

    def _diverge_attention_effect(self, *, tick_index: int, action: dict, strength: float) -> dict:
        top_n_scale = _clamp(1.0 + strength * 0.55, 1.08, 1.45)
        diversity_gain = _clamp(strength * 0.32, 0.08, 0.35)
        effect = self._single_control_effect(
            tick_index=tick_index,
            action_id="action::diverge_attention",
            control_label="control::attention_diverge",
            control_kind="diverge_attention",
            boost_labels=[],
            slow_query_labels=[],
            missing_labels=[],
            strength=_clamp(strength * 0.75, 0.10, 0.62),
            ttl=1,
            reason="low_grasp_or_novelty_exploration",
            extra_meta={"top_n_scale": _round4(top_n_scale), "family_diversity_gain": _round4(diversity_gain)},
        )
        effect["family_budget_modulation"] = {
            "schema_id": "focus_family_budget_modulation/v1",
            "control_kind": "diverge_attention",
            "source_action_id": "action::diverge_attention",
            "ttl": 1,
            "top_n_scale": _round4(top_n_scale),
            "diversity_gain": _round4(diversity_gain),
            "continuation_bias_multiplier": _round4(_clamp(1.0 - strength * 0.24, 0.72, 1.0)),
        }
        return effect

    def _text_draft_surface_effect(
        self,
        *,
        tick_index: int,
        action: dict,
        strength: float,
        state_labels: set[str],
        label_to_row: dict[str, dict],
    ) -> dict:
        action_id = str((action or {}).get("action_id", "") or "")
        params = dict((action or {}).get("params", {}) or {})
        draft_row = dict(label_to_row.get("text_action::draft_state", {}) or {})
        draft_meta = dict(draft_row.get("anchor_meta", {}) or {}) if isinstance(draft_row.get("anchor_meta", {}), dict) else {}
        visible_tokens = [
            str(token or "")
            for token in list(draft_meta.get("visible_tokens", []) or [])
            if str(token or "")
        ]
        if not visible_tokens:
            visible_text = str(draft_meta.get("visible_text", "") or "")
            visible_tokens = list(visible_text) if visible_text else []
        proposed_token = str(
            params.get(
                "token",
                params.get("text", params.get("candidate_token", params.get("expected_token", params.get("new_text", "")))),
            )
            or ""
        )
        if action_id in {"action::text_insert", "action::text_replace"} and proposed_token:
            visible_tokens = visible_tokens + [proposed_token]
        token_labels = []
        for token in visible_tokens[-6:]:
            label = f"text::{token}"
            if label not in token_labels:
                token_labels.append(label)
        if not token_labels:
            return {}
        slow_labels = self._unique_labels(
            token_labels
            + [
                "text_action::draft_state",
                "text_revision_opportunity::continue_after_visible_prefix",
                "action::text_insert",
            ]
        )
        boost_labels = [label for label in slow_labels if label in state_labels]
        return self._single_control_effect(
            tick_index=tick_index,
            action_id=action_id,
            control_label="control::draft_surface_continuation",
            control_kind="draft_surface_continuation",
            boost_labels=boost_labels,
            slow_query_labels=slow_labels,
            missing_labels=[label for label in slow_labels if label not in state_labels],
            strength=_clamp(strength * 0.92, 0.12, 0.82),
            ttl=2,
            reason="text_action_draft_surface_self_observation",
            extra_meta={
                "visible_tokens": visible_tokens[-8:],
                "proposed_token": proposed_token,
                "source_action_id": action_id,
                "learning_boundary": "text_action_controls_slow_query_without_deciding_reply_text",
            },
        )

    def _single_control_effect(
        self,
        *,
        tick_index: int,
        action_id: str,
        control_label: str,
        control_kind: str,
        boost_labels: list[str],
        slow_query_labels: list[str],
        missing_labels: list[str],
        strength: float,
        ttl: int,
        reason: str,
        extra_meta: dict | None = None,
        suppress_labels: list[str] | None = None,
    ) -> dict:
        boost_labels = self._unique_labels(boost_labels)
        suppress_labels = self._unique_labels(suppress_labels or [])
        slow_query_labels = self._unique_labels(slow_query_labels)
        missing_labels = self._unique_labels(missing_labels)
        virtual_energy = _round4(min(0.72, 0.14 + strength * 0.52))
        meta = {
            "schema_id": f"{control_kind}_control/v1",
            "action_id": action_id,
            "source_action_id": action_id,
            "source_tick_index": int(tick_index),
            "control_kind": control_kind,
            "boost_labels": boost_labels,
            "suppress_labels": suppress_labels,
            "slow_query_labels": slow_query_labels,
            "missing_labels": missing_labels,
            "strength": _round4(strength),
            "ttl": int(ttl),
            "reason": reason,
            "humanlike_testing": {
                "engineering_latency_ticks": "1-2",
                "behavior_window_ticks": "5-10",
                "meaning": "control_signal_is_fast;humanlike_behavior_is_trend_based",
            },
            "learning_boundary": "action_control_can_shape_action_outcome_but_is_not_concept_embedding_teaching",
        }
        meta.update(dict(extra_meta or {}))
        items = [
            {
                "sa_label": control_label,
                "display_text": control_label,
                "family": "action_control",
                "source_type": "action_control",
                "real_energy": 0.0,
                "virtual_energy": virtual_energy,
                "anchor_meta": meta,
            }
        ]
        for label in boost_labels[:8]:
            # The target SA is modulated, not fabricated. Use attention_gain
            # instead of prediction virtual_energy here: the action says "look
            # here next", not "this concept should be statistically closer".
            # That keeps action_control out of concept embedding teaching.
            items.append(
                {
                    "sa_label": label,
                    "display_text": label,
                    "family": "action_control",
                    "source_type": "action_control",
                    "real_energy": 0.0,
                    "virtual_energy": 0.0,
                    "attention_gain": _round4(min(0.64, strength * 0.46)),
                    "anchor_meta": {**meta, "target_label": label, "target_modulation": "boost"},
                }
            )
        attention_control = {
            "schema_id": "action_attention_control/v1",
            "source_action_id": action_id,
            "source_tick_index": int(tick_index),
            "control_kind": control_kind,
            "boost_labels": boost_labels,
            "suppress_labels": suppress_labels,
            "slow_query_labels": slow_query_labels,
            "strength": _round4(strength),
            "ttl": int(ttl),
            "reason": reason,
        }
        slow_hints = [
            {
                "schema_id": "action_slow_query_hint/v1",
                "source_action_id": action_id,
                "source_tick_index": int(tick_index),
                "control_kind": control_kind,
                "sa_label": label,
                "query_weight": _round4(min(1.25, 0.35 + strength * 0.72)),
                "virtual_energy": _round4(min(0.55, strength * 0.38)),
                "ttl": int(ttl),
                "reason": reason,
            }
            for label in slow_query_labels[:8]
        ]
        return {
            "schema_id": "action_control_effect/v1",
            "action_id": action_id,
            "control_kind": control_kind,
            "control_items": items,
            "attention_controls": [attention_control],
            "slow_query_hints": slow_hints,
            "strength": _round4(strength),
            "target_labels": boost_labels,
            "suppressed_labels": suppress_labels,
            "missing_labels": missing_labels,
            "reason": reason,
        }

    def _focus_anchor_targets(
        self,
        *,
        action: dict,
        state_labels: set[str],
        label_to_row: dict[str, dict],
        residual_pairs: list[dict],
        attention_rank: dict[str, int],
        selected_labels: list[str],
        prediction_trace: dict,
        residual_summary: dict,
    ) -> list[str]:
        params = dict(action.get("params", {}) or {})
        targets = []
        for key in ("target_labels", "boost_labels", "slow_query_labels"):
            value = params.get(key)
            if isinstance(value, (list, tuple)):
                targets.extend(str(label or "") for label in value if str(label or ""))
        for key in ("anchor_label", "anchor", "target"):
            value = params.get(key)
            if isinstance(value, str) and value:
                targets.append(value)
        target_family = str(params.get("target_family", "") or "")
        if target_family:
            for label, row in sorted(label_to_row.items(), key=lambda pair: (-float(pair[1].get("cognitive_pressure", 0.0) or 0.0), pair[0])):
                family = str(row.get("family", "") or "")
                source_type = str(row.get("source_type", "") or "")
                if family == target_family or source_type == target_family or label.startswith(f"{target_family}::"):
                    targets.append(label)
        for node in list(action.get("innate_nodes", []) or []):
            if not isinstance(node, dict):
                continue
            node_params = dict(node.get("params", {}) or {})
            for key in ("target_labels", "boost_labels", "slow_query_labels"):
                value = node_params.get(key)
                if isinstance(value, (list, tuple)):
                    targets.extend(str(label or "") for label in value if str(label or ""))
            for key in ("anchor_label", "anchor", "target"):
                value = node_params.get(key)
                if isinstance(value, str) and value:
                    targets.append(value)
            target_family = str(node_params.get("target_family", "") or "")
            if target_family:
                for label, row in sorted(label_to_row.items(), key=lambda pair: (-float(pair[1].get("cognitive_pressure", 0.0) or 0.0), pair[0])):
                    family = str(row.get("family", "") or "")
                    source_type = str(row.get("source_type", "") or "")
                    if family == target_family or source_type == target_family or label.startswith(f"{target_family}::"):
                        targets.append(label)
            anchor_key = str(node.get("anchor_key", "") or "")
            if anchor_key and anchor_key != "global":
                targets.append(anchor_key)
        for label in prediction_trace.get("unexpected_labels", []) or []:
            targets.append(str(label or ""))
        for pair in residual_pairs:
            actual = str(pair.get("actual", "") or "")
            if actual:
                targets.append(actual)
        top_rows = [
            dict(item)
            for item in label_to_row.values()
            if str(item.get("sa_label", "") or "") not in set(selected_labels)
        ]
        top_rows.sort(
            key=lambda item: (
                -float(item.get("cognitive_pressure", 0.0) or 0.0),
                int(attention_rank.get(str(item.get("sa_label", "") or ""), 10**9)),
                str(item.get("sa_label", "") or ""),
            )
        )
        for item in top_rows[:3]:
            if float(item.get("cognitive_pressure", 0.0) or 0.0) > 0.0:
                targets.append(str(item.get("sa_label", "") or ""))
        for row in list((residual_summary or {}).get("top", []) or []):
            targets.append(str(row.get("sa_label", "") or ""))
        return self._unique_labels([label for label in targets if label])[:6]

    def _residual_pairs(self, prediction_trace: dict, residual_summary: dict) -> list[dict]:
        missed = [str(label or "") for label in list((prediction_trace or {}).get("missed_predicted_labels", []) or []) if str(label or "")]
        unexpected = [str(label or "") for label in list((prediction_trace or {}).get("unexpected_labels", []) or []) if str(label or "")]
        pairs: list[dict] = []
        for index in range(max(len(missed), len(unexpected))):
            predicted = missed[index] if index < len(missed) else ""
            actual = unexpected[index] if index < len(unexpected) else ""
            if predicted or actual:
                pairs.append({"predicted": predicted, "actual": actual, "pair_quality": "paired" if predicted and actual else "single_sided"})
        if pairs:
            return pairs[:6]
        residual_rows = list((residual_summary or {}).get("top", []) or [])
        for row in residual_rows[:6]:
            label = str(row.get("sa_label", "") or "")
            if label:
                pairs.append({"predicted": label if str(row.get("last_reason", "") or "") == "prediction_miss" else "", "actual": label if str(row.get("last_reason", "") or "") != "prediction_miss" else "", "pair_quality": "residual_bucket_single_sided"})
        return pairs

    def _repeated_labels(self, *, selected_labels: list[str], previous_focus_labels: list[str]) -> list[str]:
        counts = Counter([str(label or "") for label in selected_labels + previous_focus_labels if str(label or "")])
        repeated = [label for label, count in counts.items() if count >= 2]
        if repeated:
            return repeated[:6]
        return self._unique_labels(previous_focus_labels + selected_labels)[:4]

    def _merge_family_modulation(self, existing: dict, incoming: dict) -> dict:
        if not existing:
            return dict(incoming)
        merged = dict(existing)
        sources = list(merged.get("sources", []) or [])
        for item in (merged.get("source_action_id"), incoming.get("source_action_id")):
            clean = str(item or "")
            if clean and clean not in sources:
                sources.append(clean)
        merged["sources"] = sources
        merged["top_n_scale"] = _round4(max(float(merged.get("top_n_scale", 1.0) or 1.0), float(incoming.get("top_n_scale", 1.0) or 1.0)))
        merged["diversity_gain"] = _round4(max(float(merged.get("diversity_gain", 0.0) or 0.0), float(incoming.get("diversity_gain", 0.0) or 0.0)))
        merged["ttl"] = max(int(merged.get("ttl", 1) or 1), int(incoming.get("ttl", 1) or 1))
        if incoming.get("release_labels"):
            merged["release_labels"] = self._unique_labels(list(merged.get("release_labels", []) or []) + list(incoming.get("release_labels", []) or []))
        return merged

    def _dedupe_trace(self, trace: dict) -> dict:
        trace["control_items"] = self._dedupe_items(trace.get("control_items", []))
        trace["attention_controls"] = self._dedupe_controls(trace.get("attention_controls", []))
        trace["slow_query_hints"] = self._dedupe_slow_hints(trace.get("slow_query_hints", []))
        return trace

    def _dedupe_items(self, items: list[dict]) -> list[dict]:
        rows: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for item in items or []:
            label = str(item.get("sa_label", "") or "")
            control_kind = str((item.get("anchor_meta", {}) or {}).get("control_kind", "") or "")
            key = (label, control_kind)
            if not label or key in seen:
                continue
            seen.add(key)
            rows.append(dict(item))
        return rows

    def _dedupe_controls(self, controls: list[dict]) -> list[dict]:
        rows = []
        seen = set()
        for control in controls or []:
            key = (str(control.get("source_action_id", "") or ""), str(control.get("control_kind", "") or ""))
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(control))
        return rows

    def _dedupe_slow_hints(self, hints: list[dict]) -> list[dict]:
        by_label: dict[str, dict] = {}
        for hint in hints or []:
            label = str(hint.get("sa_label", "") or "")
            if not label:
                continue
            existing = by_label.get(label)
            if existing is None or float(hint.get("query_weight", 0.0) or 0.0) > float(existing.get("query_weight", 0.0) or 0.0):
                by_label[label] = dict(hint)
        return sorted(by_label.values(), key=lambda item: (-float(item.get("query_weight", 0.0) or 0.0), str(item.get("sa_label", "") or "")))[:12]

    def _unique_labels(self, labels: list[str]) -> list[str]:
        seen = set()
        rows = []
        for label in labels or []:
            clean = str(label or "")
            if not clean or clean in seen:
                continue
            seen.add(clean)
            rows.append(clean)
        return rows
