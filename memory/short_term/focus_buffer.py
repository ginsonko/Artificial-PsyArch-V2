from __future__ import annotations

from collections import Counter, deque


def _round4(value: float) -> float:
    return round(float(value), 4)


def _int_value(value, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


class FocusBuffer:
    def __init__(
        self,
        *,
        focus_history_limit: int,
        recency_decay: float = 0.78,
        synthetic_query_weight: float = 1.1,
        replay_decay: float = 0.72,
        replay_query_weight: float = 0.82,
        max_replay_items: int = 8,
        episode_break_overlap: float = 0.22,
    ) -> None:
        self._history: deque[dict] = deque(maxlen=max(1, int(focus_history_limit)))
        self.recency_decay = max(0.0, min(1.0, float(recency_decay)))
        self.synthetic_query_weight = max(0.1, float(synthetic_query_weight))
        self.replay_decay = max(0.0, min(1.0, float(replay_decay)))
        self.replay_query_weight = max(0.0, float(replay_query_weight))
        self.max_replay_items = max(1, int(max_replay_items))
        self.episode_break_overlap = max(0.0, min(1.0, float(episode_break_overlap)))
        self._episode_id = 0
        self._episode_summaries: dict[int, dict] = {}
        self._interruption_events: deque[dict] = deque(maxlen=24)
        self._resumption_events: deque[dict] = deque(maxlen=24)
        self._episode_replay_count: Counter[int] = Counter()
        self._explicit_episode_boundaries: deque[dict] = deque(maxlen=24)

    def mark_external_turn_boundary(self, *, tick_index: int, reason: str = "new_external_text_turn") -> dict:
        """
        Start a new focus episode for a new external turn.

        Recent focus is still available for explicit recall/replay. The change
        only prevents previous-turn focus from entering the ordinary slow query
        at the same strength as the current turn.
        """

        if not self._history:
            return {
                "schema_id": "focus_external_turn_boundary/v1",
                "applied": False,
                "tick_index": int(tick_index),
                "reason": str(reason or ""),
                "note": "no_focus_history",
            }
        previous_episode_id = int(self._history[-1].get("episode_id", self._episode_id) or self._episode_id)
        self._episode_id += 1
        event = {
            "schema_id": "focus_external_turn_boundary/v1",
            "applied": True,
            "tick_index": int(tick_index),
            "from_episode_id": int(previous_episode_id),
            "to_episode_id": int(self._episode_id),
            "reason": str(reason or ""),
            "policy": "new_external_turn_reduces_default_focus_continuation_without_erasing_recall",
        }
        self._explicit_episode_boundaries.append(event)
        return dict(event)

    def push(self, focus_items: list[dict], *, tick_index: int) -> None:
        compact_items = []
        for item in focus_items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            compact_items.append(
                {
                    "sa_label": label,
                    "display_text": str(item.get("display_text", label) or label),
                    "family": str(item.get("family", "focus") or "focus"),
                    "query_weight": float(item.get("focus_score", item.get("attention_score", 0.0)) or 0.0),
                    "real_energy": float(item.get("real_energy", 0.0) or 0.0),
                    "virtual_energy": float(item.get("virtual_energy", 0.0) or 0.0),
                    "cognitive_pressure": float(item.get("cognitive_pressure", 0.0) or 0.0),
                }
            )
        if compact_items:
            labels = [str(item.get("sa_label", "") or "") for item in compact_items if str(item.get("sa_label", "") or "")]
            previous_labels = set(self.tail())
            previous_episode_id = int(self._history[-1].get("episode_id", self._episode_id) or self._episode_id) if self._history else int(self._episode_id)
            current_labels = set(labels)
            continuity_score = self._continuity_score(previous_labels, current_labels)
            resumed_episode_id = self._detect_resumption(current_labels, exclude_episode_id=previous_episode_id)
            broke_episode = False
            if self._history and continuity_score < self.episode_break_overlap:
                broke_episode = True
                self._interruption_events.append(
                    {
                        "schema_id": "focus_interruption_event/v1",
                        "tick_index": int(tick_index),
                        "from_episode_id": int(previous_episode_id),
                        "to_labels": labels,
                        "continuity_score": _round4(continuity_score),
                    }
                )
                self._episode_id += 1
            if resumed_episode_id is not None:
                self._resumption_events.append(
                    {
                        "schema_id": "focus_resumption_event/v1",
                        "tick_index": int(tick_index),
                        "resumed_episode_id": int(resumed_episode_id),
                        "new_episode_id": int(self._episode_id),
                        "labels": labels,
                        "interruption_gap_episode": int(previous_episode_id) if broke_episode else -1,
                    }
                )
            self._history.append(
                {
                    "tick_index": int(tick_index),
                    "episode_id": int(self._episode_id),
                    "continuity_score": _round4(continuity_score),
                    "resumed_episode_id": int(resumed_episode_id) if resumed_episode_id is not None else -1,
                    "items": compact_items,
                }
            )
            self._update_episode_summary(self._episode_id, compact_items, tick_index=int(tick_index), continuity_score=continuity_score)

    def tail(self) -> list[str]:
        if not self._history:
            return []
        return [str(item.get("sa_label", "") or "") for item in self._history[-1].get("items", []) if str(item.get("sa_label", "") or "")]

    def all_recent(self) -> list[list[str]]:
        rows = []
        for entry in self._history:
            rows.append([str(item.get("sa_label", "") or "") for item in entry.get("items", []) if str(item.get("sa_label", "") or "")])
        return rows

    def recent_labels(self) -> list[str]:
        labels = []
        seen = set()
        for entry in reversed(self._history):
            for item in entry.get("items", []) or []:
                label = str(item.get("sa_label", "") or "")
                if not label or label in seen:
                    continue
                seen.add(label)
                labels.append(label)
        return labels

    def build_query_items(self, state_snapshot_items: list[dict], *, tick_index: int) -> list[dict]:
        if not self._history:
            return []
        state_by_label = {str(item.get("sa_label", "") or ""): dict(item) for item in state_snapshot_items if str(item.get("sa_label", "") or "")}
        merged: dict[str, dict] = {}
        now_tick = int(tick_index)
        for history_entry in reversed(self._history):
            history_tick = _int_value(history_entry.get("tick_index", now_tick), now_tick)
            age = max(0, now_tick - history_tick)
            recency_scale = self.recency_decay ** age
            if int(history_entry.get("episode_id", self._episode_id) or self._episode_id) != int(self._episode_id):
                recency_scale *= self.episode_break_overlap
            for item in history_entry.get("items", []):
                label = str(item.get("sa_label", "") or "")
                if not label:
                    continue
                state_row = state_by_label.get(label, {})
                bucket = merged.setdefault(
                    label,
                    {
                        "sa_label": label,
                        "display_text": str(state_row.get("display_text", item.get("display_text", label)) or label),
                        "family": str(state_row.get("family", item.get("family", "focus")) or "focus"),
                        "query_weight": 0.0,
                        "real_energy": float(state_row.get("real_energy", item.get("real_energy", 0.0)) or 0.0),
                        "virtual_energy": float(state_row.get("virtual_energy", item.get("virtual_energy", 0.0)) or 0.0),
                        "source_type": "focus_continuation",
                    },
                )
                base_weight = float(item.get("query_weight", 0.0) or 0.0)
                if base_weight <= 0.0:
                    base_weight = float(state_row.get("query_weight", state_row.get("real_energy", 0.0)) or 0.0)
                bucket["query_weight"] = float(bucket["query_weight"]) + base_weight * recency_scale * self.synthetic_query_weight
        rows = list(merged.values())
        rows.sort(key=lambda item: (-float(item.get("query_weight", 0.0) or 0.0), str(item.get("sa_label", "") or "")))
        return rows

    def build_replay_query_items(self, state_snapshot_items: list[dict], *, tick_index: int) -> list[dict]:
        if not self._history or self.replay_query_weight <= 0.0:
            return []
        state_by_label = {str(item.get("sa_label", "") or ""): dict(item) for item in state_snapshot_items if str(item.get("sa_label", "") or "")}
        candidates = self.replay_candidates(tick_index=tick_index, limit=max(2, self.max_replay_items))
        merged: dict[str, dict] = {}
        for candidate in candidates:
            recency_weight = float(candidate.get("replay_weight", 0.0) or 0.0)
            if recency_weight <= 0.0:
                continue
            recency_weight *= float(candidate.get("governance", {}).get("weight_multiplier", 1.0) or 1.0)
            if recency_weight <= 0.0:
                continue
            for label in list(candidate.get("labels", []) or [])[: self.max_replay_items]:
                clean = str(label or "")
                if not clean:
                    continue
                state_row = state_by_label.get(clean, {})
                bucket = merged.setdefault(
                    clean,
                    {
                        "sa_label": clean,
                        "display_text": str(state_row.get("display_text", clean) or clean),
                        "family": str(state_row.get("family", "focus_replay") or "focus_replay"),
                        "source_type": "focus_replay",
                        "query_weight": 0.0,
                        "real_energy": float(state_row.get("real_energy", 0.0) or 0.0),
                        "virtual_energy": max(0.0, float(state_row.get("virtual_energy", 0.0) or 0.0)),
                    },
                )
                bucket["query_weight"] = float(bucket["query_weight"]) + recency_weight * self.replay_query_weight
                bucket["virtual_energy"] = max(float(bucket.get("virtual_energy", 0.0) or 0.0), recency_weight * 0.35)
        rows = list(merged.values())
        rows.sort(key=lambda item: (-float(item.get("query_weight", 0.0) or 0.0), str(item.get("sa_label", "") or "")))
        return rows[: self.max_replay_items]

    def readback_view(self, *, tick_index: int, horizon: int = 6, limit: int = 8) -> dict:
        """
        Return a white-box view of "what I was just thinking about".

        This is deliberately not a forced replay. The action planner can use it
        as evidence for a recent-thought readback candidate when continuation
        is weak or the focus stream drifted. The actual action still competes
        in the normal memory-recall conflict domain.
        """

        if not self._history:
            return {
                "schema_id": "recent_thought_readback_view/v1",
                "available": False,
                "tick_index": int(tick_index),
                "entries": [],
                "labels": [],
                "strength": 0.0,
                "drift_score": 0.0,
                "branch_end_score": 0.0,
            }
        now_tick = int(tick_index)
        max_horizon = max(1, int(horizon))
        selected_entries = [
            dict(entry)
            for entry in reversed(self._history)
            if now_tick - _int_value(entry.get("tick_index", now_tick), now_tick) <= max_horizon
        ][:max_horizon]
        if not selected_entries:
            selected_entries = [dict(self._history[-1])]
        counter: Counter[str] = Counter()
        ordered_entries = []
        continuity_values = []
        for entry in reversed(selected_entries):
            entry_tick = _int_value(entry.get("tick_index", now_tick), now_tick)
            age = max(0, now_tick - entry_tick)
            recency = self.recency_decay ** age
            labels = []
            for item in list(entry.get("items", []) or []):
                label = str(item.get("sa_label", "") or "")
                if not label:
                    continue
                labels.append(label)
                weight = float(item.get("query_weight", 0.0) or 0.0)
                if weight <= 0.0:
                    weight = float(item.get("real_energy", 0.0) or 0.0) + float(item.get("virtual_energy", 0.0) or 0.0) * 0.35
                counter[label] += max(0.01, weight) * recency
            continuity = float(entry.get("continuity_score", 0.0) or 0.0)
            continuity_values.append(continuity)
            ordered_entries.append(
                {
                    "tick_index": entry_tick,
                    "episode_id": _int_value(entry.get("episode_id", -1), -1),
                    "continuity_score": _round4(continuity),
                    "labels": labels[: max(1, int(limit))],
                }
            )
        labels = [label for label, _ in counter.most_common(max(1, int(limit)))]
        mean_continuity = sum(continuity_values) / max(1, len(continuity_values))
        recent_breaks = sum(1 for value in continuity_values if float(value) < self.episode_break_overlap)
        drift_score = min(1.0, recent_breaks / max(1.0, float(len(continuity_values)))) if continuity_values else 0.0
        latest_continuity = float(continuity_values[-1] if continuity_values else 0.0)
        branch_end_score = max(0.0, 1.0 - latest_continuity) if len(self._history) > 1 else 0.0
        strength = min(1.0, sum(counter.values()) / max(1.0, float(len(selected_entries)) * 1.6))
        return {
            "schema_id": "recent_thought_readback_view/v1",
            "available": bool(labels),
            "tick_index": int(tick_index),
            "horizon": int(max_horizon),
            "entries": ordered_entries[-max(1, int(limit)):],
            "labels": labels,
            "strength": _round4(strength),
            "drift_score": _round4(drift_score),
            "branch_end_score": _round4(branch_end_score),
            "mean_continuity": _round4(mean_continuity),
            "active_episode_id": int(self._history[-1].get("episode_id", -1) if self._history else -1),
            "policy": "short_term_self_observation_candidate_not_forced_replay",
        }

    def replay_candidates(self, *, tick_index: int, limit: int = 4) -> list[dict]:
        if len(self._history) <= 1:
            return []
        now_tick = int(tick_index)
        latest_episode = _int_value(self._history[-1].get("episode_id", 0), 0)
        latest_labels = set(self.tail())
        episode_rows: dict[int, dict] = {}
        for entry in reversed(self._history):
            episode_id = int(entry.get("episode_id", 0) or 0)
            labels = [str(item.get("sa_label", "") or "") for item in entry.get("items", []) or [] if str(item.get("sa_label", "") or "")]
            if not labels:
                continue
            if episode_id == latest_episode and set(labels) == latest_labels:
                continue
            bucket = episode_rows.setdefault(
                episode_id,
                {
                    "episode_id": episode_id,
                    "last_tick": _int_value(entry.get("tick_index", now_tick), now_tick),
                    "label_counter": Counter(),
                    "strength": 0.0,
                    "entry_count": 0,
                },
            )
            entry_tick = _int_value(entry.get("tick_index", now_tick), now_tick)
            bucket["last_tick"] = max(int(bucket["last_tick"]), entry_tick)
            bucket["entry_count"] += 1
            age = max(0, now_tick - entry_tick)
            recency = self.replay_decay ** age
            for item in entry.get("items", []) or []:
                label = str(item.get("sa_label", "") or "")
                if not label:
                    continue
                weight = float(item.get("query_weight", 0.0) or 0.0)
                if weight <= 0.0:
                    weight = float(item.get("real_energy", 0.0) or 0.0) + float(item.get("virtual_energy", 0.0) or 0.0) * 0.35
                bucket["label_counter"][label] += max(0.01, weight) * recency
                bucket["strength"] += max(0.01, weight) * recency
        rows = []
        for bucket in episode_rows.values():
            labels = [label for label, _ in bucket["label_counter"].most_common(self.max_replay_items)]
            if not labels:
                continue
            strength = float(bucket.get("strength", 0.0) or 0.0)
            governance = self._replay_governance(
                episode_id=int(bucket.get("episode_id", 0) or 0),
                labels=labels,
                latest_episode=latest_episode,
                latest_labels=latest_labels,
            )
            rows.append(
                {
                    "episode_id": int(bucket.get("episode_id", 0) or 0),
                    "last_tick": _int_value(bucket.get("last_tick", now_tick), now_tick),
                    "entry_count": int(bucket.get("entry_count", 0) or 0),
                    "labels": labels,
                    "replay_weight": _round4(strength * float(governance.get("weight_multiplier", 1.0) or 1.0)),
                    "raw_replay_weight": _round4(strength),
                    "governance": governance,
                }
            )
        rows.sort(key=lambda item: (-float(item.get("replay_weight", 0.0) or 0.0), -int(item.get("last_tick", 0) or 0), int(item.get("episode_id", 0) or 0)))
        return rows[: max(1, int(limit))]

    def trace(self, *, tick_index: int) -> dict:
        latest = dict(self._history[-1]) if self._history else {}
        episode_id = int(latest.get("episode_id", -1) if latest else -1)
        current_labels = [str(item.get("sa_label", "") or "") for item in latest.get("items", []) or [] if str(item.get("sa_label", "") or "")]
        recent_entries = []
        for entry in list(self._history)[-6:]:
            recent_entries.append(
                {
                    "tick_index": _int_value(entry.get("tick_index", -1), -1),
                    "episode_id": _int_value(entry.get("episode_id", -1), -1),
                    "continuity_score": float(entry.get("continuity_score", 0.0) or 0.0),
                    "labels": [str(item.get("sa_label", "") or "") for item in entry.get("items", []) or [] if str(item.get("sa_label", "") or "")],
                }
            )
        return {
            "schema_id": "focus_continuation_trace/v1",
            "tick_index": int(tick_index),
            "active_episode_id": episode_id,
            "current_labels": current_labels,
            "history_size": len(self._history),
            "continuation_strength": float(latest.get("continuity_score", 0.0) or 0.0) if latest else 0.0,
            "recent_entries": recent_entries,
            "replay_candidates": self.replay_candidates(tick_index=tick_index, limit=4),
            "recent_thought_readback": self.readback_view(tick_index=tick_index, horizon=6, limit=self.max_replay_items),
            "continuity_dynamics": self.continuity_dynamics(tick_index=tick_index),
            "external_turn_boundaries": [dict(row) for row in list(self._explicit_episode_boundaries)[-6:]],
        }

    def _continuity_score(self, previous_labels: set[str], current_labels: set[str]) -> float:
        if not previous_labels or not current_labels:
            return 0.0
        overlap = len(previous_labels & current_labels)
        union = len(previous_labels | current_labels)
        if union <= 0:
            return 0.0
        return overlap / float(union)

    def continuity_dynamics(self, *, tick_index: int) -> dict:
        """
        Explain slow-system continuity without forcing it.

        The buffer may notice "I was interrupted" or "this resembles an older
        episode again", but attention still decides normally. This preserves the
        humanlike freedom to resume, drift, or abandon a thought.
        """

        return {
            "schema_id": "focus_continuity_dynamics/v1",
            "tick_index": int(tick_index),
            "active_episode_id": int(self._history[-1].get("episode_id", -1) if self._history else -1),
            "episode_count": len(self._episode_summaries),
            "recent_interruptions": list(self._interruption_events)[-6:],
            "recent_resumptions": list(self._resumption_events)[-6:],
            "recent_external_turn_boundaries": list(self._explicit_episode_boundaries)[-6:],
            "episode_summaries": [
                self._public_episode_summary(summary)
                for summary in sorted(
                    self._episode_summaries.values(),
                    key=lambda row: (-int(row.get("last_tick", -1) or -1), int(row.get("episode_id", 0) or 0)),
                )[:6]
            ],
            "policy": {
                "humanlike_timing": "judge_resume_or_interruption_over_recent_ticks_not_single_tick_perfection",
                "governance": "replay_candidates_are_damped_not_forced",
            },
        }

    def mark_replay_selected(self, episode_id: int) -> None:
        clean_id = int(episode_id)
        if clean_id >= 0:
            self._episode_replay_count[clean_id] += 1

    def _update_episode_summary(self, episode_id: int, compact_items: list[dict], *, tick_index: int, continuity_score: float) -> None:
        summary = self._episode_summaries.setdefault(
            int(episode_id),
            {
                "episode_id": int(episode_id),
                "first_tick": int(tick_index),
                "last_tick": int(tick_index),
                "entry_count": 0,
                "label_counter": Counter(),
                "mean_continuity": 0.0,
            },
        )
        summary["last_tick"] = int(tick_index)
        summary["entry_count"] = int(summary.get("entry_count", 0) or 0) + 1
        count = max(1, int(summary["entry_count"]))
        old_mean = float(summary.get("mean_continuity", 0.0) or 0.0)
        summary["mean_continuity"] = old_mean + (float(continuity_score) - old_mean) / count
        counter = summary.setdefault("label_counter", Counter())
        if not isinstance(counter, Counter):
            counter = Counter(dict(counter or {}))
            summary["label_counter"] = counter
        for item in compact_items:
            label = str((item or {}).get("sa_label", "") or "")
            if label:
                counter[label] += max(0.01, float((item or {}).get("query_weight", 0.0) or 0.0))

    def _detect_resumption(self, current_labels: set[str], *, exclude_episode_id: int) -> int | None:
        if not current_labels:
            return None
        best_id = None
        best_score = 0.0
        for episode_id, summary in self._episode_summaries.items():
            if int(episode_id) == int(exclude_episode_id):
                continue
            counter = summary.get("label_counter", Counter())
            labels = set(dict(counter or {}).keys())
            score = self._continuity_score(labels, current_labels)
            if score > best_score:
                best_id = int(episode_id)
                best_score = score
        threshold = max(self.episode_break_overlap, 0.34)
        if best_id is not None and best_score >= threshold:
            return best_id
        return None

    def _replay_governance(self, *, episode_id: int, labels: list[str], latest_episode: int, latest_labels: set[str]) -> dict:
        repetition = int(self._episode_replay_count.get(int(episode_id), 0) or 0)
        same_as_current = int(episode_id) == int(latest_episode) or self._continuity_score(set(labels), set(latest_labels)) > 0.82
        multiplier = 1.0
        reasons = []
        if same_as_current:
            multiplier *= 0.15
            reasons.append("avoid_replaying_current_episode")
        if repetition > 0:
            multiplier *= max(0.25, 1.0 / (1.0 + repetition * 0.65))
            reasons.append("repetition_cooldown")
        if not reasons:
            reasons.append("eligible_past_episode")
        return {
            "schema_id": "focus_replay_governance/v1",
            "weight_multiplier": _round4(multiplier),
            "replay_count": repetition,
            "same_as_current": bool(same_as_current),
            "reasons": reasons,
        }

    def _public_episode_summary(self, summary: dict) -> dict:
        counter = summary.get("label_counter", Counter())
        if not isinstance(counter, Counter):
            counter = Counter(dict(counter or {}))
        return {
            "episode_id": int(summary.get("episode_id", -1) or -1),
            "first_tick": _int_value(summary.get("first_tick", -1), -1),
            "last_tick": _int_value(summary.get("last_tick", -1), -1),
            "entry_count": int(summary.get("entry_count", 0) or 0),
            "mean_continuity": _round4(float(summary.get("mean_continuity", 0.0) or 0.0)),
            "top_labels": [label for label, _ in counter.most_common(self.max_replay_items)],
        }
