from __future__ import annotations

from core.action.registry import is_external_action


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


class SafetyGate:
    """
    SafetyGate is not an actuator.

    It only reviews external action candidates after drive calculation and can
    emit inhibition SAs. Internal cognition actions are deliberately left alone.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        veto_pressure_threshold: float = 0.64,
        veto_cor_threshold: float = 0.68,
        review_pressure_threshold: float = 0.42,
        review_cor_threshold: float = 0.50,
        min_external_confidence: float = 0.58,
    ) -> None:
        self.enabled = bool(enabled)
        self.veto_pressure_threshold = _clamp(veto_pressure_threshold, 0.0, 1.0)
        self.veto_cor_threshold = _clamp(veto_cor_threshold, 0.0, 1.0)
        self.review_pressure_threshold = _clamp(review_pressure_threshold, 0.0, 1.0)
        self.review_cor_threshold = _clamp(review_cor_threshold, 0.0, 1.0)
        self.min_external_confidence = _clamp(min_external_confidence, 0.0, 1.0)

    def review(
        self,
        *,
        tick_index: int,
        candidates: list[dict],
        selected_actions: list[dict],
        cognitive_feelings: dict | None = None,
        emotion_state: dict | None = None,
        safety_trace: dict | None = None,
        expectation_pressure_trace: dict | None = None,
        action_control_items: list[dict] | None = None,
    ) -> dict:
        if not self.enabled:
            return {
                "schema_id": "safety_gate_trace/v1",
                "enabled": False,
                "applied": False,
                "reviewed": [],
                "vetoed_action_ids": [],
                "require_review_action_ids": [],
                "inhibition_items": [],
                "selected_actions": [dict(row) for row in list(selected_actions or []) if isinstance(row, dict)],
                "policy": {},
            }
        channels = dict((cognitive_feelings or {}).get("channels", {}) or {})
        pressure = max(
            float(channels.get("pressure", 0.0) or 0.0),
            float(channels.get("dissonance", 0.0) or 0.0) * 0.72,
        )
        cor = float((emotion_state or {}).get("COR", 0.0) or 0.0)
        policy_rows = list((safety_trace or {}).get("hits", []) or [])
        anchor_risk_trace = self._anchor_risk(expectation_pressure_trace or {})
        anchor_pressure = float(anchor_risk_trace.get("pressure", 0.0) or 0.0)
        control_risk_trace = self._control_risk(action_control_items or [])
        control_pressure = float(control_risk_trace.get("pressure", 0.0) or 0.0)
        reviewed = []
        vetoed_ids: set[str] = set()
        require_review_ids: set[str] = set()
        inhibition_items = []

        for row in list(selected_actions or []):
            action_id = str(row.get("action_id", "") or "")
            actuator_id = str(row.get("actuator_id", "") or "")
            if not action_id or not is_external_action(action_id, actuator_id):
                reviewed.append({"action_id": action_id, "decision": "not_external"})
                continue
            predicted = dict(row.get("predicted_outcome", {}) or {})
            params = dict(row.get("params", {}) or {})
            draft_commit_boundary = bool(
                action_id == "action::text_commit"
                and str(params.get("target_channel", "") or "") == "draft"
                and str(params.get("reason", "") or "").startswith("draft_")
            )
            confidence = _clamp(float(predicted.get("confidence", 0.0) or 0.0), 0.0, 1.0)
            punishment = max(0.0, float(predicted.get("punishment", 0.0) or 0.0))
            predicted_pressure = max(pressure, anchor_pressure, control_pressure, float(predicted.get("pressure", 0.0) or 0.0), punishment * 0.82)
            cor_risk = cor * (0.28 if draft_commit_boundary else 0.95)
            risk = _clamp(max(predicted_pressure, cor_risk, punishment), 0.0, 1.0)
            reasons = []
            if control_pressure >= self.review_pressure_threshold:
                reasons.append("action_control_review")
            if control_pressure >= self.veto_pressure_threshold:
                reasons.append("action_control_veto")
            if anchor_pressure >= self.review_pressure_threshold:
                reasons.append("pressure_anchor_review")
            if anchor_pressure >= self.veto_pressure_threshold:
                reasons.append("pressure_anchor_veto")
            if predicted_pressure >= self.veto_pressure_threshold:
                reasons.append("pressure_veto")
            if cor >= self.veto_cor_threshold and not draft_commit_boundary:
                reasons.append("cor_veto")
            elif draft_commit_boundary and cor >= self.veto_cor_threshold:
                reasons.append("cor_review_softened_for_internal_draft_commit")
            if confidence < self.min_external_confidence:
                reasons.append("low_external_confidence")
            policy_hit = next((hit for hit in policy_rows if str(hit.get("action_id", "") or "") == action_id), {})
            if policy_hit:
                reasons.append(str(policy_hit.get("rule_id", "safety_rule_hit") or "safety_rule_hit"))
                risk = max(risk, float(policy_hit.get("strength", 0.0) or 0.0))
            if predicted_pressure >= self.review_pressure_threshold or cor >= self.review_cor_threshold:
                require_review_ids.add(action_id)
            decision = "allow"
            if (
                "pressure_veto" in reasons
                or "cor_veto" in reasons
                or "action_control_veto" in reasons
                or ("low_external_confidence" in reasons and risk >= self.review_pressure_threshold)
            ):
                decision = "veto"
                vetoed_ids.add(action_id)
            elif draft_commit_boundary and not (
                "pressure_anchor_review" in reasons
                or "action_control_review" in reasons
                or "low_external_confidence" in reasons
                or policy_hit
            ):
                decision = "allow"
                require_review_ids.discard(action_id)
            elif action_id in require_review_ids or "low_external_confidence" in reasons or "action_control_review" in reasons:
                decision = "require_review"
            if decision in {"veto", "require_review"}:
                action_name = action_id.split("::")[-1]
                inhibition_items.append(
                    {
                        "sa_label": f"action_inhibition::{action_name}",
                        "display_text": f"行动抑制:{action_name}",
                        "family": "action_inhibition",
                        "source_type": "safety_gate",
                        "real_energy": _round4(0.12 + risk * 0.5),
                        "virtual_energy": _round4(predicted_pressure * 0.28),
                        "anchor_meta": {
                            "tick_index": int(tick_index),
                            "action_id": action_id,
                            "actuator_id": actuator_id,
                            "decision": decision,
                            "risk": _round4(risk),
                            "pressure": _round4(predicted_pressure),
                            "anchor_pressure": _round4(anchor_pressure),
                            "control_pressure": _round4(control_pressure),
                            "control_risk": dict(control_risk_trace),
                            "anchor_risk": dict(anchor_risk_trace),
                            "cor": _round4(cor),
                            "confidence": _round4(confidence),
                            "reasons": reasons,
                        },
                    }
                )
            reviewed.append(
                {
                    "action_id": action_id,
                    "actuator_id": actuator_id,
                    "decision": decision,
                    "risk": _round4(risk),
                    "pressure": _round4(predicted_pressure),
                    "anchor_pressure": _round4(anchor_pressure),
                    "control_pressure": _round4(control_pressure),
                    "cor": _round4(cor),
                    "confidence": _round4(confidence),
                    "reasons": reasons,
                }
            )

        blocked_ids = set(vetoed_ids) | set(require_review_ids)
        filtered_selected = [
            dict(row)
            for row in list(selected_actions or [])
            if str(row.get("action_id", "") or "") not in blocked_ids
        ]
        return {
            "schema_id": "safety_gate_trace/v1",
            "enabled": bool(self.enabled),
            "applied": bool(vetoed_ids or require_review_ids or inhibition_items),
            "reviewed": reviewed,
            "vetoed_action_ids": sorted(vetoed_ids),
            "require_review_action_ids": sorted(require_review_ids - vetoed_ids),
            "inhibition_items": inhibition_items,
            "selected_actions": filtered_selected,
            "policy": {
                "veto_pressure_threshold": _round4(self.veto_pressure_threshold),
                "veto_cor_threshold": _round4(self.veto_cor_threshold),
                "review_pressure_threshold": _round4(self.review_pressure_threshold),
                "review_cor_threshold": _round4(self.review_cor_threshold),
                "min_external_confidence": _round4(self.min_external_confidence),
                "pressure_anchor_coupling": "pressure_B_anchor_level_raises_external_action_risk",
                "action_control_coupling": "episode_replay_and_wait_controls_raise_external_review_pressure",
            },
            "anchor_risk": dict(anchor_risk_trace),
            "control_risk": dict(control_risk_trace),
        }

    def _anchor_risk(self, expectation_pressure_trace: dict) -> dict:
        trace = dict((expectation_pressure_trace or {}).get("anchor_verification", {}) or {})
        anchors = [dict(anchor) for anchor in list(trace.get("anchors", []) or []) if isinstance(anchor, dict)]
        pressure = 0.0
        support = 0.0
        top: dict = {}
        for anchor in anchors:
            if str(anchor.get("anchor_type", "") or "") != "pressure":
                continue
            level = _clamp(float(anchor.get("level", 0.0) or 0.0), 0.0, 1.0)
            expected_punishment = _clamp(float(anchor.get("expected_punishment", 0.0) or 0.0), 0.0, 1.0)
            expected_pressure = _clamp(float(anchor.get("expected_pressure", 0.0) or 0.0), 0.0, 1.0)
            row_pressure = _clamp(level * 0.68 + expected_punishment * 0.22 + expected_pressure * 0.10, 0.0, 1.0)
            support = max(support, level)
            if row_pressure > pressure:
                pressure = row_pressure
                top = dict(anchor)
        return {
            "schema_id": "safety_gate_pressure_anchor_risk/v1",
            "pressure": _round4(pressure),
            "support": _round4(support),
            "active_pressure_anchor_count": len([anchor for anchor in anchors if str(anchor.get("anchor_type", "") or "") == "pressure"]),
            "top_anchor_id": str(top.get("anchor_id", "") or ""),
            "top_source_memory_id": str(top.get("source_memory_id", "") or ""),
        }

    def _control_risk(self, action_control_items: list[dict]) -> dict:
        pressure = 0.0
        review_hint_count = 0
        wait_hint_count = 0
        top_source = ""
        for item in action_control_items or []:
            if not isinstance(item, dict):
                continue
            meta = dict(item.get("anchor_meta", {}) or {})
            control_kind = str(meta.get("control_kind", "") or "")
            if control_kind == "replay_episode":
                hint = dict(meta.get("safety_review_hint", {}) or {})
                risk = _clamp(float(hint.get("risk", 0.0) or 0.0), 0.0, 1.0)
                pressure = max(pressure, risk)
                review_hint_count += 1 if bool(hint.get("requires_external_review", False)) else 0
                if risk >= pressure:
                    top_source = str(meta.get("source_memory_id", "") or "")
            elif control_kind == "wait":
                hint = _clamp(float(meta.get("external_action_review_hint", 0.0) or 0.0), 0.0, 1.0)
                pressure = max(pressure, hint)
                wait_hint_count += 1
        return {
            "schema_id": "safety_gate_action_control_risk/v1",
            "pressure": _round4(pressure),
            "review_hint_count": int(review_hint_count),
            "wait_hint_count": int(wait_hint_count),
            "top_source_memory_id": top_source,
        }
