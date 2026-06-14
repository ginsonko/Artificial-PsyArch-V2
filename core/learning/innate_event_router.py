from __future__ import annotations

from collections import Counter, defaultdict

from core.learning.event_builder import LearningEventBuilder


def _round4(value: float) -> float:
    return round(float(value), 4)


class InnateLearningEventRouter:
    """
    Consumes InnateCodingEngine learning_events without turning rules into a
    hidden learning model.

    It classifies events into existing APV2.1 learning layers. Direct online
    embedding writes remain inside MemoryStore snapshot learning, and direct
    action outcome writes remain inside ActionOutcomeMemory.
    """

    CONCEPT_EVENTS = {"positive_pair", "negative_pair", "multimodal_binding"}
    TRANSITION_EVENTS = {"transition"}
    ACTION_EVENTS = {
        "action_outcome",
        "action_selection",
        "action_feedback",
        "reward_signal",
        "punishment_signal",
        "action_outcome_success",
        "action_outcome_failure",
        "action_prediction_error",
        "action_consequence_estimate",
        "memory_predicted_action",
        "action_inhibition",
        "causal_window",
    }
    EXPECTATION_EVENTS = {"verify_b_anchor"}

    def __init__(self, *, recent_limit: int = 64) -> None:
        self.recent_limit = max(8, int(recent_limit))
        self._builder = LearningEventBuilder()
        self._total_by_event: Counter[str] = Counter()
        self._total_by_route: Counter[str] = Counter()
        self._total_by_layer: Counter[str] = Counter()
        self._recent: list[dict] = []

    def route(
        self,
        *,
        tick_index: int,
        innate_traces: dict | None,
        expectation_anchor_trace: dict | None = None,
        action_feedback_trace: dict | None = None,
    ) -> dict:
        events = self._collect_events(innate_traces or {})
        routed: list[dict] = []
        structured_events: list[dict] = []
        phase_counts: dict[str, int] = defaultdict(int)
        route_counts: Counter[str] = Counter()
        layer_counts: Counter[str] = Counter()
        for event in events:
            clean_event = str(event.get("event", "") or "trace")
            route = self._route_for_event(clean_event)
            structured = self._builder.from_innate_event(event=event, route=route, tick_index=int(tick_index))
            phase = str(event.get("phase", "") or "")
            phase_counts[phase] += 1
            route_counts[route] += 1
            layer = str(structured.get("learning_layer", "") or "")
            if layer:
                layer_counts[layer] += 1
                self._total_by_layer[layer] += 1
            self._total_by_event[clean_event] += 1
            self._total_by_route[route] += 1
            structured_events.append(structured)
            routed.append(
                {
                    "event": clean_event,
                    "route": route,
                    "phase": phase,
                    "rule_id": str(event.get("rule_id", "") or ""),
                    "strength": _round4(float(event.get("strength", 0.0) or 0.0)),
                    "policy": self._policy_for_route(route),
                    "structured_event_id": str(structured.get("event_id", "") or ""),
                    "learning_layer": layer,
                    "writer": str(structured.get("writer", "") or ""),
                    "write_mode": str(structured.get("write_mode", "") or ""),
                    "top_positive_label": str(event.get("top_positive_label", "") or ""),
                    "top_negative_label": str(event.get("top_negative_label", "") or ""),
                    "top_action_id": str(event.get("top_action_id", "") or ""),
                }
            )

        self._recent.extend(routed)
        if len(self._recent) > self.recent_limit:
            self._recent = self._recent[-self.recent_limit :]

        action_feedback = dict(action_feedback_trace or {})
        anchor_trace = dict(expectation_anchor_trace or {})
        return {
            "schema_id": "innate_learning_event_router/v1",
            "tick_index": int(tick_index),
            "input_event_count": len(events),
            "routed_event_count": len(routed),
            "route_counts": dict(sorted(route_counts.items())),
            "layer_counts": dict(sorted(layer_counts.items())),
            "phase_counts": dict(sorted((key, int(value)) for key, value in phase_counts.items())),
            "routes": routed[:24],
            "structured_events": structured_events[:24],
            "structured_summary": self._builder.summarize(structured_events),
            "totals": {
                "by_event": dict(sorted((key, int(value)) for key, value in self._total_by_event.items())),
                "by_route": dict(sorted((key, int(value)) for key, value in self._total_by_route.items())),
                "by_layer": dict(sorted((key, int(value)) for key, value in self._total_by_layer.items())),
            },
            "recent": list(self._recent[-16:]),
            "expectation_anchor_link": {
                "active_count": int(anchor_trace.get("active_count", 0) or 0),
                "created_count": len(anchor_trace.get("created", []) or []),
                "verified_count": len(anchor_trace.get("verified", []) or []),
                "missed_count": len(anchor_trace.get("missed", []) or []),
                "policy": "verify_b_anchor_events_are_consumed_by_expectation_pressure_b_anchor_verifier",
            },
            "action_feedback_link": {
                "applied": bool(action_feedback.get("applied", False)),
                "feedback_item_count": len(action_feedback.get("feedback_items", []) or []),
                "policy": "action_outcome_events_are_audit_routed;ActionOutcomeMemory_records_feedback_once_in_planner",
            },
            "safety": {
                "concept_learning_guard": "action_feedback_action_selection_and_reward_punishment_events_do_not_write_concept_similarity_here",
                "online_embedding_guard": "MemoryStore_snapshot_learning_remains_the_only_direct_online_embedding_writer",
            },
        }

    def _collect_events(self, innate_traces: dict) -> list[dict]:
        events: list[dict] = []
        for phase, trace in dict(innate_traces or {}).items():
            row = dict(trace or {})
            for event in list(row.get("learning_events", []) or []):
                if not isinstance(event, dict):
                    continue
                item = dict(event)
                item.setdefault("phase", str(phase))
                events.append(item)
        return events

    def _route_for_event(self, event: str) -> str:
        clean = str(event or "")
        if clean in self.CONCEPT_EVENTS:
            return "content_online_embedding_trace"
        if clean in self.TRANSITION_EVENTS:
            return "transition_online_embedding_trace"
        if clean in self.ACTION_EVENTS:
            return "action_outcome_learning_trace"
        if clean in self.EXPECTATION_EVENTS:
            return "expectation_pressure_b_anchor_verifier"
        return "audit_trace"

    def _policy_for_route(self, route: str) -> str:
        if route == "content_online_embedding_trace":
            return "audit_only;MemoryStore_energy_learning_consumes_state_snapshot_pressure_not_rule_event_directly"
        if route == "transition_online_embedding_trace":
            return "audit_only;MemoryStore_relation_and_successor_learning_are_the_direct_transition_writers"
        if route == "action_outcome_learning_trace":
            return "audit_only;planner_record_feedback_is_the_single_action_outcome_writer"
        if route == "expectation_pressure_b_anchor_verifier":
            return "consumed_by_B_anchor_verifier_for_cross_tick_expectation_pressure_validation"
        return "audit_only"
