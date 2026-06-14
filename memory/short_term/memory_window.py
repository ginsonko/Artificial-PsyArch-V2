from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Callable


def _round4(value: float) -> float:
    return round(float(value), 4)


def _int_value(value, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float_value(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _energy_of(item: dict) -> float:
    return max(
        0.0,
        _float_value((item or {}).get("real_energy"), 0.0)
        + _float_value((item or {}).get("virtual_energy"), 0.0) * 0.35
        + _float_value((item or {}).get("attention_gain", (item or {}).get("focus_score", (item or {}).get("attention_score"))), 0.0) * 0.55
        + abs(_float_value((item or {}).get("cognitive_pressure"), 0.0)) * 0.42,
    )


class ShortTermMemoryWindow:
    """
    Active, bounded working-memory window for recent AP experience.

    EchoBuffer answers "what residue is still ringing now"; this class answers
    "which recent part can AP deliberately recall". It stores all SA as
    first-class short-term events, but active recall only emits a small,
    fatigued, provenance-rich control view so it cannot become an automatic
    replay loop or a hidden long-term memory writer.
    """

    def __init__(
        self,
        *,
        history_limit: int = 48,
        max_age_ticks: int = 48,
        recency_decay: float = 0.86,
        fatigue_decay: float = 0.70,
        fatigue_step: float = 0.45,
        unfinished_decay: float = 0.90,
        max_items_per_event: int = 12,
        default_recall_limit: int = 8,
    ) -> None:
        self._events: deque[dict] = deque(maxlen=max(1, int(history_limit)))
        self.max_age_ticks = max(1, int(max_age_ticks))
        self.recency_decay = max(0.0, min(0.995, float(recency_decay)))
        self.fatigue_decay = max(0.0, min(0.995, float(fatigue_decay)))
        self.fatigue_step = max(0.0, float(fatigue_step))
        self.unfinished_decay = max(0.0, min(0.995, float(unfinished_decay)))
        self.max_items_per_event = max(1, int(max_items_per_event))
        self.default_recall_limit = max(1, int(default_recall_limit))
        self._event_counter = 0
        self._fatigue_by_event_id: dict[str, float] = {}
        self._last_fatigue_decay_tick = -1
        self._unfinished_by_event_id: dict[str, dict] = {}
        self._last_unfinished_decay_tick = -1
        self._last_recall: dict = {}

    def observe(
        self,
        items: list[dict],
        *,
        tick_index: int,
        source_kind: str,
        modality: str | None = None,
        role: str | None = None,
    ) -> dict:
        compact_items = self._compact_items(items)
        if not compact_items:
            return {"schema_id": "short_term_memory_observe_trace/v1", "stored": False, "reason": "no_compact_items"}
        inferred_modality = str(modality or self._dominant_modality(compact_items) or "thought")
        clean_source_kind = str(source_kind or "unknown")
        self._event_counter += 1
        event_id = f"stm::{int(tick_index):06d}::{self._event_counter:06d}"
        salience = self._event_salience(compact_items)
        event = {
            "schema_id": "short_term_memory_event/v1",
            "event_id": event_id,
            "tick_index": int(tick_index),
            "source_kind": clean_source_kind,
            "role": str(role or clean_source_kind),
            "modality": inferred_modality,
            "salience": _round4(salience),
            "items": compact_items,
            "tokens": self._event_tokens(compact_items, source_kind=clean_source_kind, modality=inferred_modality),
        }
        self._events.append(event)
        return {
            "schema_id": "short_term_memory_observe_trace/v1",
            "stored": True,
            "event_id": event_id,
            "tick_index": int(tick_index),
            "source_kind": clean_source_kind,
            "modality": inferred_modality,
            "item_count": len(compact_items),
            "salience": _round4(salience),
        }

    def recall(
        self,
        *,
        tick_index: int,
        cues: list[dict] | list[str] | None = None,
        limit: int | None = None,
        horizon_ticks: int | None = None,
        reason: str = "recall_recent_context",
        similarity_fn: Callable[[list[str], list[str]], dict] | None = None,
        update_fatigue: bool = True,
    ) -> dict:
        now_tick = int(tick_index)
        self._decay_fatigue(now_tick)
        self._decay_unfinished(now_tick)
        clean_limit = max(1, int(limit if limit is not None else self.default_recall_limit))
        horizon = max(1, int(horizon_ticks if horizon_ticks is not None else self.max_age_ticks))
        cue_tokens = self._cue_tokens(cues)
        candidates = []
        for event in list(self._events):
            age = max(0, now_tick - _int_value(event.get("tick_index"), now_tick))
            if age > min(horizon, self.max_age_ticks):
                continue
            score_trace = self._score_event(event, age=age, cue_tokens=cue_tokens, similarity_fn=similarity_fn)
            if float(score_trace.get("score", 0.0) or 0.0) <= 0.0:
                continue
            candidates.append({**self._public_event(event), "score_trace": score_trace, "score": _round4(float(score_trace["score"]))})
        candidates.sort(
            key=lambda row: (
                -float(row.get("score", 0.0) or 0.0),
                -int(row.get("tick_index", 0) or 0),
                str(row.get("event_id", "")),
            )
        )
        selected_events = self._select_diverse_events(candidates, limit=clean_limit)
        selected_items = self._items_from_selected_events(selected_events, limit=clean_limit)
        if update_fatigue and selected_events and self.fatigue_step > 0.0:
            for event in selected_events:
                event_id = str(event.get("event_id", "") or "")
                if event_id:
                    self._fatigue_by_event_id[event_id] = min(2.5, float(self._fatigue_by_event_id.get(event_id, 0.0) or 0.0) + self.fatigue_step)
        trace = {
            "schema_id": "short_term_memory_recall_trace/v1",
            "available": bool(selected_items),
            "tick_index": now_tick,
            "reason": str(reason or "recall_recent_context"),
            "cue_tokens": cue_tokens[:12],
            "cue_count": len(cue_tokens),
            "window_size": len(self._events),
            "candidate_count": len(candidates),
            "selected_events": selected_events,
            "selected_items": selected_items,
            "candidate_preview": candidates[: min(8, len(candidates))],
            "fatigue": self._fatigue_trace(),
            "unfinished": self._unfinished_trace(tick_index=now_tick),
            "fatigue_updated": bool(update_fatigue and selected_events),
            "policy": "partial_salient_or_cued_short_term_recall_with_soft_fatigue",
            "learning_boundary": "short_term_memory_readback_modulates_attention_and_slow_query_not_forced_answer",
        }
        self._last_recall = dict(trace)
        return trace

    def mark_unfinished(
        self,
        *,
        tick_index: int,
        labels: list[str],
        reason: str,
        strength: float,
        successor_labels: list[str] | None = None,
    ) -> dict:
        """
        Mark a recent thought event as "interrupted but still continuable".

        This is a soft short-term trace, not a task queue. It only biases later
        no-cue recall when the system has gone quiet again, and it decays/fatigues
        like any other working-memory evidence.
        """

        now_tick = int(tick_index)
        self._decay_unfinished(now_tick)
        clean_labels = [str(label or "") for label in list(labels or []) if str(label or "")]
        if not clean_labels:
            return {"schema_id": "short_term_unfinished_mark_trace/v1", "stored": False, "reason": "no_labels"}
        best_event = self._best_event_for_labels(clean_labels)
        if not best_event:
            return {"schema_id": "short_term_unfinished_mark_trace/v1", "stored": False, "reason": "no_matching_recent_event", "labels": clean_labels[:8]}
        event_id = str(best_event.get("event_id", "") or "")
        amount = max(0.0, min(1.0, float(strength or 0.0)))
        if amount <= 0.0:
            return {"schema_id": "short_term_unfinished_mark_trace/v1", "stored": False, "reason": "zero_strength", "event_id": event_id}
        previous = dict(self._unfinished_by_event_id.get(event_id, {}) or {})
        new_strength = min(2.2, float(previous.get("strength", 0.0) or 0.0) + amount)
        row = {
            "schema_id": "short_term_unfinished_thought/v1",
            "event_id": event_id,
            "tick_index": int(best_event.get("tick_index", now_tick) or now_tick),
            "last_mark_tick": now_tick,
            "labels": clean_labels[:8],
            "successor_labels": [str(label or "") for label in list(successor_labels or []) if str(label or "")][:8],
            "reason": str(reason or "unfinished_thought"),
            "strength": _round4(new_strength),
            "raw_added_strength": _round4(amount),
        }
        self._unfinished_by_event_id[event_id] = row
        return {
            "schema_id": "short_term_unfinished_mark_trace/v1",
            "stored": True,
            "event_id": event_id,
            "tick_index": now_tick,
            "labels": clean_labels[:8],
            "successor_labels": row["successor_labels"],
            "strength": _round4(new_strength),
            "reason": str(reason or "unfinished_thought"),
        }

    def resolve_unfinished(
        self,
        *,
        tick_index: int,
        labels: list[str],
        reason: str,
        amount: float = 1.0,
    ) -> dict:
        """
        Soften unfinished traces after an action consequence closes them.

        This is not a task-queue completion primitive. It is the short-term
        memory counterpart of mark_unfinished(...): when AP commits, sends, or
        otherwise closes a recent thought, the matching unfinished pressure
        should fade so another still-open trace can surface naturally.
        """

        now_tick = int(tick_index)
        self._decay_unfinished(now_tick)
        clean_labels = [str(label or "") for label in list(labels or []) if str(label or "")]
        if not clean_labels:
            return {"schema_id": "short_term_unfinished_resolve_trace/v1", "resolved": False, "reason": "no_labels"}
        amount = max(0.0, min(2.5, float(amount or 0.0)))
        if amount <= 0.0:
            return {"schema_id": "short_term_unfinished_resolve_trace/v1", "resolved": False, "reason": "zero_amount", "labels": clean_labels[:8]}
        wanted = set(clean_labels)
        wanted_specific = self._unfinished_specific_labels(wanted)
        resolved_rows = []
        next_values = {}
        for event_id, value in list(self._unfinished_by_event_id.items()):
            row = dict(value or {})
            row_labels = {str(label or "") for label in list(row.get("labels", []) or []) if str(label or "")}
            successor_labels = {str(label or "") for label in list(row.get("successor_labels", []) or []) if str(label or "")}
            pool = row_labels | successor_labels
            overlap = wanted & pool
            specific_overlap = wanted_specific & self._unfinished_specific_labels(pool)
            if not specific_overlap:
                next_values[event_id] = row
                continue
            before = max(0.0, float(row.get("strength", 0.0) or 0.0))
            after = max(0.0, before - amount)
            resolved_rows.append(
                {
                    "event_id": event_id,
                    "before_strength": _round4(before),
                    "after_strength": _round4(after),
                    "matched_labels": sorted(specific_overlap)[:8],
                    "generic_overlap_ignored": sorted(overlap - specific_overlap)[:8],
                }
            )
            if after > 0.02:
                row["strength"] = _round4(after)
                row["last_resolve_tick"] = now_tick
                row["resolve_reason"] = str(reason or "unfinished_resolved")
                next_values[event_id] = row
        self._unfinished_by_event_id = next_values
        return {
            "schema_id": "short_term_unfinished_resolve_trace/v1",
            "resolved": bool(resolved_rows),
            "tick_index": now_tick,
            "labels": clean_labels[:8],
            "reason": str(reason or "unfinished_resolved"),
            "amount": _round4(amount),
            "resolved_rows": resolved_rows,
            "remaining_active_count": len(next_values),
            "policy": "action_feedback_softens_matching_unfinished_traces_not_task_queue_completion",
        }

    def _unfinished_specific_labels(self, labels: set[str]) -> set[str]:
        """
        Return labels specific enough to close an unfinished trace.

        Generic skill/action-role labels are useful recall context, but they
        should not make one closed draft erase every other draft with the same
        broad skill. This keeps unfinished recovery as soft memory similarity,
        not a hidden task queue and not a global skill-level completion flag.
        """

        specific = set()
        for label in labels or set():
            clean = str(label or "")
            if not clean:
                continue
            if clean.startswith("skill::"):
                continue
            if clean in {"text_action::draft_state", "action::text_reread", "action::text_replace", "action::text_commit"}:
                continue
            specific.add(clean)
        return specific

    def trace(self, *, tick_index: int, last_recall: dict | None = None) -> dict:
        now_tick = int(tick_index)
        active_events = []
        modality_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        for event in list(self._events):
            age = max(0, now_tick - _int_value(event.get("tick_index"), now_tick))
            if age > self.max_age_ticks:
                continue
            modality_counts[str(event.get("modality", "unknown") or "unknown")] += 1
            source_counts[str(event.get("source_kind", "unknown") or "unknown")] += 1
            active_events.append(self._public_event(event))
        active_events.sort(key=lambda row: (-int(row.get("tick_index", 0) or 0), -float(row.get("salience", 0.0) or 0.0)))
        recall = dict(last_recall or self._last_recall or {})
        return {
            "schema_id": "short_term_memory_window_trace/v1",
            "tick_index": now_tick,
            "window_size": len(self._events),
            "active_event_count": len(active_events),
            "modality_counts": dict(sorted(modality_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "recent_events": active_events[:8],
            "last_recall": recall if recall else {"available": False},
            "fatigue": self._fatigue_trace(),
            "unfinished": self._unfinished_trace(tick_index=now_tick),
            "policy": "sliding_window_records_many_recalls_partial_segments",
        }

    def _compact_items(self, items: list[dict]) -> list[dict]:
        rows = []
        seen = set()
        for item in list(items or []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label or label in seen:
                continue
            if self._is_short_term_control_item(item):
                continue
            seen.add(label)
            meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
            compact = {
                "sa_label": label,
                "display_text": str(item.get("display_text", label) or label),
                "family": str(item.get("family", "") or ""),
                "source_type": str(item.get("source_type", "") or ""),
                "real_energy": _round4(max(0.0, _float_value(item.get("real_energy"), 0.0))),
                "virtual_energy": _round4(max(0.0, _float_value(item.get("virtual_energy"), 0.0))),
                "attention_gain": _round4(max(0.0, _float_value(item.get("attention_gain", item.get("focus_score", item.get("attention_score"))), 0.0))),
                "cognitive_pressure": _round4(_float_value(item.get("cognitive_pressure"), 0.0)),
                "position": item.get("position", 0),
                "modality": self._modality_for_item(item),
            }
            if meta:
                compact["anchor_meta"] = self._compact_meta(meta)
            if isinstance(item.get("numeric_features"), dict):
                compact["numeric_feature_channels"] = sorted(str(key) for key in dict(item.get("numeric_features", {}) or {}) if str(key or ""))[:8]
            rows.append(compact)
        rows.sort(key=lambda item: (-_energy_of(item), str(item.get("sa_label", "") or "")))
        return rows[: self.max_items_per_event]

    def _compact_meta(self, meta: dict) -> dict:
        keep_keys = {
            "schema_id",
            "event_type",
            "control_kind",
            "action_id",
            "source_action_id",
            "target_label",
            "origin_tick_index",
            "age_ticks",
            "echo_kind",
            "echo_modality",
            "original_modality",
            "not_new_external_input",
            "learnable_handle",
            "track_slot",
            "object_anchor_id",
            "bbox_norm",
        }
        return {str(key): value for key, value in dict(meta or {}).items() if str(key) in keep_keys}

    def _is_short_term_control_item(self, item: dict) -> bool:
        label = str((item or {}).get("sa_label", "") or "")
        if label.startswith("control::short_term_memory_recall"):
            return True
        meta = (item or {}).get("anchor_meta", {}) or {}
        if isinstance(meta, dict) and str(meta.get("schema_id", "") or "").startswith("short_term_memory_recall_control"):
            return True
        return False

    def _dominant_modality(self, items: list[dict]) -> str:
        counts: Counter[str] = Counter()
        for item in items:
            counts[str(item.get("modality", "other") or "other")] += 1
        if not counts:
            return "thought"
        return counts.most_common(1)[0][0]

    def _event_salience(self, items: list[dict]) -> float:
        if not items:
            return 0.0
        energies = sorted((_energy_of(item) for item in items), reverse=True)
        top = sum(energies[:3])
        tail = sum(energies[3:]) * 0.25
        return min(4.0, top + tail)

    def _event_tokens(self, items: list[dict], *, source_kind: str, modality: str) -> list[str]:
        tokens = []
        seen = set()
        for raw in [f"source::{source_kind}", f"modality::{modality}"]:
            if raw not in seen:
                seen.add(raw)
                tokens.append(raw)
        for item in items:
            for raw in [
                str(item.get("sa_label", "") or ""),
                f"family::{item.get('family', '')}",
                f"source_type::{item.get('source_type', '')}",
                f"modality::{item.get('modality', '')}",
            ]:
                clean = str(raw or "").strip()
                if not clean or clean.endswith("::") or clean in seen:
                    continue
                seen.add(clean)
                tokens.append(clean)
                if len(tokens) >= 48:
                    return tokens
        return tokens

    def _cue_tokens(self, cues: list[dict] | list[str] | None) -> list[str]:
        tokens = []
        seen = set()
        for cue in list(cues or []):
            if isinstance(cue, str):
                rows = [cue]
            elif isinstance(cue, dict):
                rows = [
                    str(cue.get("sa_label", "") or ""),
                    f"family::{cue.get('family', '')}",
                    f"source_type::{cue.get('source_type', '')}",
                    f"modality::{cue.get('modality', self._modality_for_item(cue))}",
                ]
            else:
                rows = []
            for raw in rows:
                clean = str(raw or "").strip()
                if not clean or clean.endswith("::") or clean in seen:
                    continue
                seen.add(clean)
                tokens.append(clean)
                if len(tokens) >= 32:
                    return tokens
        return tokens

    def _score_event(
        self,
        event: dict,
        *,
        age: int,
        cue_tokens: list[str],
        similarity_fn: Callable[[list[str], list[str]], dict] | None,
    ) -> dict:
        event_tokens = list(event.get("tokens", []) or [])
        salience = max(0.0, _float_value(event.get("salience"), 0.0))
        recency = self.recency_decay ** max(0, int(age))
        continuity = self._continuity_bonus(event)
        fatigue = max(0.0, float(self._fatigue_by_event_id.get(str(event.get("event_id", "") or ""), 0.0) or 0.0))
        unfinished = self._unfinished_boost(event)
        direct_overlap = 0.0
        learned_score = 0.0
        learned_trace: dict = {}
        if cue_tokens:
            direct_overlap = self._direct_similarity(cue_tokens, event_tokens)
            if similarity_fn is not None:
                try:
                    learned_trace = dict(similarity_fn(cue_tokens, event_tokens) or {})
                    learned_score = max(0.0, float(learned_trace.get("score", 0.0) or 0.0))
                except Exception as exc:  # pragma: no cover - defensive trace only
                    learned_trace = {"score": 0.0, "error": str(exc)}
                    learned_score = 0.0
        cue_bonus = direct_overlap * 1.35 + learned_score * 1.15
        base = salience * 0.72 + recency * 0.42 + continuity * 0.28
        score = max(0.0, base + cue_bonus + unfinished - fatigue)
        return {
            "schema_id": "short_term_memory_event_score/v1",
            "score": _round4(score),
            "salience": _round4(salience),
            "recency": _round4(recency),
            "continuity": _round4(continuity),
            "direct_cue_similarity": _round4(direct_overlap),
            "learned_similarity": _round4(learned_score),
            "unfinished_boost": _round4(unfinished),
            "fatigue_penalty": _round4(fatigue),
            "age_ticks": int(age),
            "learned_similarity_trace": learned_trace if learned_trace else {"score": 0.0},
        }

    def _direct_similarity(self, cue_tokens: list[str], event_tokens: list[str]) -> float:
        cues = set(str(token or "") for token in cue_tokens if str(token or ""))
        events = set(str(token or "") for token in event_tokens if str(token or ""))
        if not cues or not events:
            return 0.0
        exact = len(cues & events)
        if exact:
            return min(1.0, exact / max(1.0, len(cues)) + exact / max(3.0, len(events)) * 0.35)
        cue_roots = {token.split("::", 1)[-1] for token in cues if "::" in token}
        event_roots = {token.split("::", 1)[-1] for token in events if "::" in token}
        root_overlap = len(cue_roots & event_roots)
        return min(0.35, root_overlap / max(1.0, len(cue_roots)) * 0.35) if root_overlap else 0.0

    def _continuity_bonus(self, event: dict) -> float:
        labels = {str(item.get("sa_label", "") or "") for item in list(event.get("items", []) or []) if str(item.get("sa_label", "") or "")}
        if not labels:
            return 0.0
        idx = list(self._events).index(event) if event in self._events else -1
        if idx <= 0:
            return 0.0
        prev = list(self._events)[idx - 1]
        prev_labels = {str(item.get("sa_label", "") or "") for item in list(prev.get("items", []) or []) if str(item.get("sa_label", "") or "")}
        if not prev_labels:
            return 0.0
        return len(labels & prev_labels) / max(1.0, len(labels | prev_labels))

    def _select_diverse_events(self, candidates: list[dict], *, limit: int) -> list[dict]:
        if not candidates:
            return []
        selected = []
        modality_counts: defaultdict[str, int] = defaultdict(int)
        max_events = max(1, min(len(candidates), max(2, int(limit) // 2)))
        for candidate in candidates:
            modality = str(candidate.get("modality", "unknown") or "unknown")
            if len(selected) >= max_events:
                break
            if modality_counts[modality] >= 2 and len(selected) < max_events - 1:
                continue
            selected.append(candidate)
            modality_counts[modality] += 1
        if not selected:
            selected = candidates[:1]
        return selected

    def _items_from_selected_events(self, selected_events: list[dict], *, limit: int) -> list[dict]:
        rows = []
        seen = set()
        for event in selected_events:
            strength = max(0.0, min(1.0, _float_value(event.get("score"), 0.0) / 3.0))
            for item in list(event.get("items", []) or []):
                label = str(item.get("sa_label", "") or "")
                if not label or label in seen:
                    continue
                seen.add(label)
                meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
                rows.append(
                    {
                        "sa_label": label,
                        "display_text": str(item.get("display_text", label) or label),
                        "family": str(item.get("family", "") or ""),
                        "source_type": str(item.get("source_type", "") or ""),
                        "modality": str(item.get("modality", event.get("modality", "unknown")) or "unknown"),
                        "origin_tick_index": int(event.get("tick_index", -1) or -1),
                        "source_kind": str(event.get("source_kind", "") or ""),
                        "recall_strength": _round4(max(0.05, strength)),
                        "event_id": str(event.get("event_id", "") or ""),
                        "anchor_meta": meta,
                    }
                )
                if len(rows) >= max(1, int(limit)):
                    return rows
        return rows

    def _public_event(self, event: dict) -> dict:
        return {
            "event_id": str(event.get("event_id", "") or ""),
            "tick_index": int(event.get("tick_index", -1) or -1),
            "source_kind": str(event.get("source_kind", "") or ""),
            "role": str(event.get("role", "") or ""),
            "modality": str(event.get("modality", "") or ""),
            "salience": _round4(float(event.get("salience", 0.0) or 0.0)),
            "labels": [str(item.get("sa_label", "") or "") for item in list(event.get("items", []) or []) if str(item.get("sa_label", "") or "")],
            "items": [dict(item) for item in list(event.get("items", []) or [])[: self.max_items_per_event]],
        }

    def _fatigue_trace(self) -> dict:
        live = {
            key: _round4(value)
            for key, value in sorted(self._fatigue_by_event_id.items())
            if float(value) > 0.01
        }
        return {
            "schema_id": "short_term_memory_recall_fatigue/v1",
            "active_count": len(live),
            "by_event_id": live,
            "decay": _round4(self.fatigue_decay),
            "step": _round4(self.fatigue_step),
        }

    def _unfinished_trace(self, *, tick_index: int) -> dict:
        self._decay_unfinished(int(tick_index))
        active = {
            key: dict(value)
            for key, value in sorted(self._unfinished_by_event_id.items())
            if float((value or {}).get("strength", 0.0) or 0.0) > 0.01
        }
        rows = sorted(
            active.values(),
            key=lambda row: (-float(row.get("strength", 0.0) or 0.0), -int(row.get("last_mark_tick", 0) or 0), str(row.get("event_id", ""))),
        )
        return {
            "schema_id": "short_term_unfinished_trace/v1",
            "active_count": len(rows),
            "top": rows[:6],
            "by_event_id": active,
            "policy": "soft_decaying_unfinished_thought_boost_for_no_cue_recall",
        }

    def _unfinished_boost(self, event: dict) -> float:
        event_id = str((event or {}).get("event_id", "") or "")
        row = self._unfinished_by_event_id.get(event_id)
        if not row:
            return 0.0
        return min(1.2, max(0.0, float(row.get("strength", 0.0) or 0.0)) * 0.72)

    def _decay_fatigue(self, tick_index: int) -> None:
        if not self._fatigue_by_event_id:
            self._last_fatigue_decay_tick = int(tick_index)
            return
        if self._last_fatigue_decay_tick == int(tick_index):
            return
        delta = 1 if self._last_fatigue_decay_tick < 0 else max(1, int(tick_index) - int(self._last_fatigue_decay_tick))
        self._last_fatigue_decay_tick = int(tick_index)
        decay_factor = self.fatigue_decay ** delta
        next_values = {}
        for key, value in self._fatigue_by_event_id.items():
            decayed = float(value) * decay_factor
            if decayed > 0.01:
                next_values[key] = decayed
        self._fatigue_by_event_id = next_values

    def _decay_unfinished(self, tick_index: int) -> None:
        if not self._unfinished_by_event_id:
            self._last_unfinished_decay_tick = int(tick_index)
            return
        if self._last_unfinished_decay_tick == int(tick_index):
            return
        delta = 1 if self._last_unfinished_decay_tick < 0 else max(1, int(tick_index) - int(self._last_unfinished_decay_tick))
        self._last_unfinished_decay_tick = int(tick_index)
        decay_factor = self.unfinished_decay ** delta
        next_values = {}
        live_ids = {str(event.get("event_id", "") or "") for event in list(self._events)}
        for key, value in self._unfinished_by_event_id.items():
            row = dict(value or {})
            if key not in live_ids:
                continue
            decayed = float(row.get("strength", 0.0) or 0.0) * decay_factor
            if decayed > 0.02:
                row["strength"] = _round4(decayed)
                next_values[key] = row
        self._unfinished_by_event_id = next_values

    def _best_event_for_labels(self, labels: list[str]) -> dict:
        wanted = {str(label or "") for label in list(labels or []) if str(label or "")}
        if not wanted:
            return {}
        best = {}
        best_score = 0.0
        for event in list(self._events):
            event_labels = {str(item.get("sa_label", "") or "") for item in list(event.get("items", []) or []) if str(item.get("sa_label", "") or "")}
            overlap = len(wanted & event_labels)
            if overlap <= 0:
                continue
            score = overlap + float(event.get("salience", 0.0) or 0.0) * 0.08 + int(event.get("tick_index", 0) or 0) * 0.0001
            if score > best_score:
                best = dict(event)
                best_score = score
        return best

    def _modality_for_item(self, item: dict) -> str:
        source = str((item or {}).get("source_type", "") or "")
        family = str((item or {}).get("family", "") or "")
        label = str((item or {}).get("sa_label", "") or "")
        if source == "external_text" or family in {"text", "learned_text_phrase", "text_phrase"} or label.startswith(("text::", "phrase::")):
            return "text"
        if source.startswith("vision") or family.startswith("vision") or label.startswith(("vision::", "vision_obj::", "vision_mem::")):
            return "vision"
        if source.startswith("audio") or family.startswith("audio") or label.startswith(("audio::", "audio_event::")):
            return "audio"
        if label.startswith("action::") or family.startswith("action"):
            return "action"
        if label.startswith("feeling::") or family in {"cognitive_feeling", "expectation_pressure", "time_feeling", "rhythm_feeling"}:
            return "feeling"
        if source in {"focus_continuation", "focus_replay", "action_control"}:
            return "thought"
        return "thought"
