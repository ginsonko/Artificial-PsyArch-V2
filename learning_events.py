from __future__ import annotations

from collections import Counter


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clean(value: object) -> str:
    return str(value or "").strip()


class LearningEventBuilder:
    """
    Builds white-box APV2.1 learning events.

    The builder does not train anything. It gives BC-* rules, MemoryStore, and
    action outcome code one shared language for explaining which evidence is
    allowed to reach which learning layer. This neutral module deliberately
    avoids importing runtime/core packages, so memory can use it without a
    circular dependency.
    """

    SCHEMA_ID = "apv21_learning_event/v1"

    def build(
        self,
        *,
        event_type: str,
        learning_layer: str,
        writer: str,
        source: str = "",
        target: str = "",
        relation: str = "trace",
        weight: float = 0.0,
        tick_index: int | None = None,
        memory_id: str = "",
        memory_kind: str = "",
        bc_rule_id: str = "",
        write_mode: str = "trace_only",
        evidence: dict | None = None,
        guards: dict | None = None,
        source_event: dict | None = None,
        meaning: str = "",
        human_time_window_ticks: tuple[int, int] = (5, 10),
    ) -> dict:
        clean_event = _clean(event_type) or "trace"
        clean_source = _clean(source)
        clean_target = _clean(target)
        clean_rule = _clean(bc_rule_id)
        clean_memory = _clean(memory_id)
        event_id_parts = [
            "learn",
            str(tick_index) if tick_index is not None else "na",
            clean_rule or "BC-TRACE",
            clean_event,
            clean_source or "source",
            clean_target or "target",
        ]
        event_id = "|".join(part.replace("|", "/") for part in event_id_parts)
        return {
            "schema_id": self.SCHEMA_ID,
            "event_id": event_id,
            "tick_index": int(tick_index) if tick_index is not None else None,
            "memory_id": clean_memory,
            "memory_kind": _clean(memory_kind),
            "bc_rule_id": clean_rule,
            "event_type": clean_event,
            "learning_layer": _clean(learning_layer),
            "writer": _clean(writer),
            "write_mode": _clean(write_mode),
            "source": clean_source,
            "target": clean_target,
            "relation": _clean(relation) or "trace",
            "weight": _round4(float(weight or 0.0)),
            "evidence": dict(evidence or {}),
            "guards": dict(guards or {}),
            "source_event": dict(source_event or {}),
            "human_time_window_ticks": {
                "min": int(human_time_window_ticks[0]),
                "max": int(human_time_window_ticks[1]),
                "policy": "judge_learning_and_behavior_by_trend_not_single_tick_reflex",
            },
            "meaning": _clean(meaning),
        }

    def from_innate_event(self, *, event: dict, route: str, tick_index: int) -> dict:
        clean_event = _clean((event or {}).get("event", "")) or "trace"
        rule_id = _clean((event or {}).get("rule_id", ""))
        layer, writer, write_mode, relation, guards, meaning = self._innate_policy(clean_event, route)
        source = _clean((event or {}).get("top_positive_label", "")) or _clean((event or {}).get("top_action_id", ""))
        target = _clean((event or {}).get("top_negative_label", ""))
        return self.build(
            event_type=self._structured_event_type(clean_event),
            learning_layer=layer,
            writer=writer,
            source=source,
            target=target,
            relation=relation,
            weight=float((event or {}).get("strength", 0.0) or 0.0),
            tick_index=tick_index,
            bc_rule_id=rule_id,
            write_mode=write_mode,
            evidence={
                "innate_event": clean_event,
                "phase": _clean((event or {}).get("phase", "")),
                "top_positive_label": _clean((event or {}).get("top_positive_label", "")),
                "top_negative_label": _clean((event or {}).get("top_negative_label", "")),
                "top_action_id": _clean((event or {}).get("top_action_id", "")),
            },
            guards=guards,
            source_event=event,
            meaning=meaning,
        )

    def memory_event(
        self,
        *,
        raw_event: dict,
        tick_index: int,
        memory_id: str,
        memory_kind: str,
        bc_rule_id: str,
        event_type: str,
        learning_layer: str,
        writer: str,
        write_mode: str,
        meaning: str,
        guards: dict | None = None,
    ) -> dict:
        raw = dict(raw_event or {})
        evidence = dict(raw.get("evidence", {}) or {})
        evidence["raw_event_type"] = _clean(raw.get("event_type", ""))
        return self.build(
            event_type=event_type,
            learning_layer=learning_layer,
            writer=writer,
            source=_clean(raw.get("source", "")),
            target=_clean(raw.get("target", "")),
            relation=_clean(raw.get("relation", "")),
            weight=float(raw.get("weight", 0.0) or 0.0),
            tick_index=tick_index,
            memory_id=memory_id,
            memory_kind=memory_kind,
            bc_rule_id=bc_rule_id,
            write_mode=write_mode,
            evidence=evidence,
            guards=guards or self.concept_guards(),
            source_event=raw,
            meaning=meaning,
        )

    def summarize(self, events: list[dict]) -> dict:
        by_type = Counter()
        by_layer = Counter()
        by_writer = Counter()
        by_rule = Counter()
        for event in events or []:
            if not isinstance(event, dict):
                continue
            by_type[_clean(event.get("event_type", ""))] += 1
            by_layer[_clean(event.get("learning_layer", ""))] += 1
            by_writer[_clean(event.get("writer", ""))] += 1
            by_rule[_clean(event.get("bc_rule_id", ""))] += 1
        return {
            "schema_id": "apv21_learning_event_summary/v1",
            "event_count": len([event for event in events or [] if isinstance(event, dict)]),
            "by_type": dict(sorted((key, int(value)) for key, value in by_type.items() if key)),
            "by_layer": dict(sorted((key, int(value)) for key, value in by_layer.items() if key)),
            "by_writer": dict(sorted((key, int(value)) for key, value in by_writer.items() if key)),
            "by_rule": dict(sorted((key, int(value)) for key, value in by_rule.items() if key)),
        }

    def concept_guards(self) -> dict:
        return {
            "concept_guard": True,
            "exclude_action_feedback": True,
            "exclude_reward_punishment": True,
            "exclude_action_selection": False,
            "exclude_action_control": False,
            "policy": "p1k4_full_sa_state_field;reward_punishment_and_feedback_use_specialized_outcome_writers",
        }

    def _structured_event_type(self, event: str) -> str:
        mapping = {
            "positive_pair": "prediction_error_positive",
            "negative_pair": "prediction_error_negative",
            "transition": "order_transition",
            "multimodal_binding": "multimodal_binding",
            "action_outcome": "action_outcome",
            "verify_b_anchor": "b_anchor_verification",
        }
        return mapping.get(_clean(event), _clean(event) or "trace")

    def _innate_policy(self, event: str, route: str) -> tuple[str, str, str, str, dict, str]:
        clean_event = _clean(event)
        if clean_event == "positive_pair":
            return (
                "content_recognition_embedding",
                "MemoryStore._learn_from_snapshot",
                "audit_route",
                "positive",
                self.concept_guards(),
                "positive cognitive pressure marks underprediction; MemoryStore uses state pressure to pull the subject toward real context anchors",
            )
        if clean_event == "negative_pair":
            return (
                "content_recognition_embedding",
                "MemoryStore._learn_from_snapshot",
                "audit_route",
                "negative",
                self.concept_guards(),
                "negative cognitive pressure marks overprediction; MemoryStore uses state pressure to push the missed prediction away from actual context anchors",
            )
        if clean_event == "transition":
            return (
                "relation_order_embedding",
                "MemoryStore._learn_from_snapshot",
                "audit_route",
                "transition",
                {"concept_guard": True, "policy": "directed_order_learning_not_symmetric_similarity"},
                "ordered text audio and vision evidence teaches asymmetric successor and relation channels",
            )
        if clean_event == "multimodal_binding":
            return (
                "multimodal_binding_embedding",
                "MemoryStore._learn_from_snapshot",
                "audit_route",
                "positive",
                self.concept_guards(),
                "cross-modal co-presence binds text vision and audio handles while keeping numeric and white-box evidence visible",
            )
        if clean_event in {"action_outcome", "action_feedback", "reward_signal", "punishment_signal", "action_prediction_error"}:
            return (
                "action_outcome_memory",
                "ActionOutcomeMemory.record",
                "audit_route",
                "action_outcome",
                self.concept_guards(),
                "action reward and punishment update future drive tendency, not concept distance",
            )
        if clean_event == "verify_b_anchor":
            return (
                "expectation_pressure_anchor",
                "BAnchorExpectationVerifier.update",
                "consumed_by_verifier",
                "verify",
                {"concept_guard": True, "policy": "expectation_pressure_anchor_validation"},
                "expectation and pressure must be validated against later B-object recognition",
            )
        return (
            _clean(route) or "audit_trace",
            "trace",
            "trace_only",
            "trace",
            {"concept_guard": True},
            "unclassified learning trace kept for audit only",
        )
