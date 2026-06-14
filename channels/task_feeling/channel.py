from __future__ import annotations


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(float(low), min(float(high), float(value)))


class TaskFeelingChannel:
    """
    Cognitive feeling for "nothing to continue" vs "there is something to do".

    This channel is intentionally not a semantic taredacted-test-key judge. It reads
    white-box AP traces: successor clarity, unfinished short-term thoughts,
    recall availability, residual/pressure, and current external quietness.
    The result is a subjective state-field signal: boredom or fulfillment.
    """

    def __init__(
        self,
        *,
        min_activation: float = 0.12,
        boredom_gain: float = 1.0,
        fulfillment_gain: float = 1.0,
    ) -> None:
        self.min_activation = max(0.0, float(min_activation))
        self.boredom_gain = max(0.0, float(boredom_gain))
        self.fulfillment_gain = max(0.0, float(fulfillment_gain))

    def derive(
        self,
        *,
        tick_index: int,
        input_packet: dict | None,
        expected_text: dict | None,
        focus_continuation_trace: dict | None,
        short_term_memory_trace: dict | None,
        cognitive_feelings: dict | None,
        residual_summary: dict | None,
        action_trace: dict | None = None,
    ) -> dict:
        expected = dict(expected_text or {})
        focus = dict(focus_continuation_trace or {})
        stm = dict(short_term_memory_trace or {})
        feelings = dict((cognitive_feelings or {}).get("channels", {}) or {})
        residual = dict(residual_summary or {})
        input_units = len(list((input_packet or {}).get("units", []) or []))
        normalized_text = str((input_packet or {}).get("normalized_text", "") or "")
        external_quiet = 1.0 if input_units <= 0 and not normalized_text else 0.0

        expected_strength = _clamp(float(expected.get("strength", 0.0) or 0.0), 0.0, 1.2)
        top_share = _clamp(float(expected.get("top_share", 0.0) or 0.0), 0.0, 1.0)
        dominance_gap = _clamp(float(expected.get("dominance_gap", 0.0) or 0.0), 0.0, 1.0)
        decisive = bool(expected.get("decisive", False))
        successor_clarity = _clamp(expected_strength * 0.35 + top_share * 0.28 + dominance_gap * 0.32 + (0.14 if decisive else 0.0))

        readback = dict(focus.get("recent_thought_readback", {}) or {})
        branch_end = _clamp(float(readback.get("branch_end_score", 0.0) or 0.0))
        drift = _clamp(float(readback.get("drift_score", 0.0) or 0.0))
        continuation_strength = _clamp(float(focus.get("continuation_strength", 0.0) or 0.0))

        stm_recall = dict(stm.get("last_recall", {}) or {})
        recall_scores = [
            float(event.get("score", 0.0) or 0.0)
            for event in list(stm_recall.get("selected_events", []) or [])
            if isinstance(event, dict)
        ]
        recall_strength = _clamp((max(recall_scores) if recall_scores else 0.0) / 3.0)
        unfinished = dict(stm.get("unfinished", {}) or stm_recall.get("unfinished", {}) or {})
        unfinished_scores = [
            float(row.get("strength", 0.0) or 0.0)
            for row in list(unfinished.get("top", []) or [])
            if isinstance(row, dict)
        ]
        unfinished_strength = _clamp(max(unfinished_scores or [0.0]) / 1.5)

        residual_mass = _clamp(float(residual.get("total_unresolved_mass", 0.0) or 0.0) / 3.0)
        pressure = _clamp(float(feelings.get("pressure", 0.0) or 0.0))
        dissonance = _clamp(float(feelings.get("dissonance", 0.0) or 0.0))
        surprise = _clamp(float(feelings.get("surprise", 0.0) or 0.0))
        active_pressure = _clamp(residual_mass * 0.35 + pressure * 0.28 + dissonance * 0.22 + surprise * 0.15)

        selected_actions = list((action_trace or {}).get("selected_actions", []) or [])
        action_activity = _clamp(len(selected_actions) / 4.0)

        task_available = _clamp(
            successor_clarity * 0.42
            + unfinished_strength * 0.34
            + recall_strength * 0.22
            + active_pressure * 0.18
            + action_activity * 0.12
            + continuation_strength * 0.14
        )
        quiet_no_task = _clamp(
            external_quiet * 0.38
            + branch_end * 0.22
            + (1.0 - successor_clarity) * 0.22
            + (1.0 - recall_strength) * 0.12
            + (1.0 - unfinished_strength) * 0.18
            - active_pressure * 0.22
            - action_activity * 0.16
        )
        if int(stm.get("active_event_count", 0) or 0) <= 0 and int(stm.get("window_size", 0) or 0) <= 0:
            # A completely empty working-memory window is a stronger "nothing
            # to return to" signal than a window that merely failed this recall.
            # It stays generic: the channel does not care what the missing task
            # would have been, only that no continuable trace is available.
            quiet_no_task = _clamp(quiet_no_task + external_quiet * 0.16)
        boredom = _clamp(quiet_no_task * self.boredom_gain)
        fulfillment = _clamp(
            (
                task_available * 0.72
                + successor_clarity * 0.18
                + unfinished_strength * 0.16
                - drift * 0.10
            )
            * self.fulfillment_gain
        )
        if fulfillment > 0.34:
            boredom = _clamp(boredom - fulfillment * 0.45)

        channels = {
            "boredom": _round4(boredom),
            "fulfillment": _round4(fulfillment),
            "task_available": _round4(task_available),
            "unfinished_strength": _round4(unfinished_strength),
            "recall_strength": _round4(recall_strength),
            "successor_clarity": _round4(successor_clarity),
            "external_quiet": _round4(external_quiet),
        }
        items = []
        if boredom >= self.min_activation:
            items.append(self._item("boredom", "boredom", boredom, channels))
        if fulfillment >= self.min_activation:
            items.append(self._item("fulfillment", "fulfillment", fulfillment, channels))
        return {
            "schema_id": "task_feeling_trace/v1",
            "tick_index": int(tick_index),
            "channels": channels,
            "items": items,
            "components": {
                "branch_end": _round4(branch_end),
                "drift": _round4(drift),
                "continuation_strength": _round4(continuation_strength),
                "active_pressure": _round4(active_pressure),
                "action_activity": _round4(action_activity),
            },
            "policy": "boredom_when_quiet_no_successor_no_recall_fulfillment_when_continuable",
        }

    def _item(self, key: str, display: str, energy: float, channels: dict) -> dict:
        return {
            "sa_label": f"feeling::{key}",
            "display_text": display,
            "source_type": "task_feeling",
            "family": "cognitive_feeling",
            "real_energy": _round4(energy),
            "anchor_meta": {
                "schema_id": "task_feeling_item/v1",
                "feeling_key": key,
                "feeling_value": _round4(energy),
                "task_feeling": dict(channels),
                "meaning": "subjective_sense_of_no_task_or_continuable_task",
            },
        }
