from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from math import sqrt


def _round4(value: float) -> float:
    return round(float(value), 4)


class RelativeRelationStore:
    """
    Bounded white-box relative-relation statistics for APV2.1 memory.

    This store is a specialised statistical view, not a replacement for Bn/Cn.
    It learns explicit relation tokens such as text order, and exposes a bounded
    score for recall reranking. The first implementation focuses on text/focus
    order while keeping the schema open for audio/vision relations.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_relation_tokens_per_snapshot: int = 256,
        max_events_per_snapshot: int = 128,
        context_limit: int = 8192,
        max_targets_per_context: int = 64,
        min_support: float = 0.08,
        decay: float = 0.996,
        rescale_threshold: float = 96.0,
        rescale_factor: float = 0.5,
        score_weight: float = 0.68,
        focus_score_weight: float = 0.92,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_relation_tokens_per_snapshot = max(8, int(max_relation_tokens_per_snapshot))
        self.max_events_per_snapshot = max(8, int(max_events_per_snapshot))
        self.context_limit = max(64, int(context_limit))
        self.max_targets_per_context = max(4, int(max_targets_per_context))
        self.min_support = max(0.0, float(min_support))
        self.decay = max(0.0, min(1.0, float(decay)))
        self.rescale_threshold = max(1.0, float(rescale_threshold))
        self.rescale_factor = max(0.05, min(1.0, float(rescale_factor)))
        self.score_weight = max(0.0, float(score_weight))
        self.focus_score_weight = max(0.0, float(focus_score_weight))
        self._relations_by_kind_id: dict[str, dict[str, dict]] = defaultdict(dict)
        self._contexts: OrderedDict[str, dict] = OrderedDict()
        self._observed_snapshot_ids: set[str] = set()
        self._last_events: list[dict] = []
        self._total_events = 0

    def build_features(self, *, memory_kind: str, items: list[dict], focus_labels: list[str] | None = None) -> dict:
        if not self.enabled:
            return self._empty_features(memory_kind=memory_kind)
        events = self.extract_events(memory_kind=memory_kind, items=items, focus_labels=focus_labels)
        token_weights: dict[str, float] = {}
        channel_weights: dict[str, float] = {}
        for event in events[: self.max_events_per_snapshot]:
            token = str(event.get("relation_token", "") or "")
            if not token:
                continue
            weight = float(event.get("weight", 0.0) or 0.0)
            if weight <= 0.0:
                continue
            token_weights[token] = max(float(token_weights.get(token, 0.0) or 0.0), weight)
            channel = str(event.get("relation_type", "") or "")
            if channel:
                channel_weights[channel] = float(channel_weights.get(channel, 0.0) or 0.0) + weight
        ordered = sorted(token_weights.items(), key=lambda row: (-float(row[1]), str(row[0])))[: self.max_relation_tokens_per_snapshot]
        return {
            "schema_id": "relative_relation_features/v1",
            "memory_kind": str(memory_kind or ""),
            "relation_tokens": [token for token, _ in ordered],
            "relation_token_weights": {token: _round4(weight) for token, weight in ordered},
            "relation_channels": {key: _round4(value) for key, value in sorted(channel_weights.items())},
            "relation_events": events[: min(len(events), 24)],
            "event_count": len(events),
        }

    def add_snapshot(self, *, memory_kind: str, memory_id: str, relation_features: dict, tick_index: int) -> None:
        kind = str(memory_kind or "")
        clean_id = str(memory_id or "")
        if not (self.enabled and kind and clean_id):
            return
        features = dict(relation_features or {})
        self._relations_by_kind_id[kind][clean_id] = {
            "memory_id": clean_id,
            "memory_kind": kind,
            "tick_index": int(tick_index),
            "relation_tokens": list(features.get("relation_tokens", []) or [])[: self.max_relation_tokens_per_snapshot],
            "relation_token_weights": dict(features.get("relation_token_weights", {}) or {}),
            "relation_channels": dict(features.get("relation_channels", {}) or {}),
        }
        if clean_id not in self._observed_snapshot_ids:
            self._observe_events(features.get("relation_events", []) or [], tick_index=int(tick_index))
            self._observed_snapshot_ids.add(clean_id)

    def remove_snapshot(self, *, memory_kind: str, memory_id: str) -> None:
        kind = str(memory_kind or "")
        clean_id = str(memory_id or "")
        if not kind or not clean_id:
            return
        self._relations_by_kind_id.get(kind, {}).pop(clean_id, None)
        self._observed_snapshot_ids.discard(clean_id)

    def score(self, *, memory_kind: str, query_features: dict, candidate_memory_id: str) -> dict:
        kind = str(memory_kind or "")
        clean_id = str(candidate_memory_id or "")
        if not (self.enabled and kind and clean_id):
            return self._empty_score()
        candidate = self._relations_by_kind_id.get(kind, {}).get(clean_id)
        if not candidate:
            return self._empty_score()
        query_weights = dict((query_features or {}).get("relation_token_weights", {}) or {})
        candidate_weights = dict(candidate.get("relation_token_weights", {}) or {})
        if not query_weights or not candidate_weights:
            return self._empty_score()
        shared = sorted(set(query_weights) & set(candidate_weights))
        if not shared:
            return self._empty_score()
        numerator = 0.0
        query_total = sum(float(value or 0.0) for value in query_weights.values())
        candidate_total = sum(float(value or 0.0) for value in candidate_weights.values())
        channel_scores: dict[str, float] = {}
        matches = []
        for token in shared:
            q = max(0.0, float(query_weights.get(token, 0.0) or 0.0))
            c = max(0.0, float(candidate_weights.get(token, 0.0) or 0.0))
            if q <= 0.0 or c <= 0.0:
                continue
            value = min(q, c)
            numerator += value
            channel = self._channel_from_token(token)
            channel_scores[channel] = float(channel_scores.get(channel, 0.0) or 0.0) + value
            if len(matches) < 8:
                matches.append({"relation_token": token, "channel": channel, "score": _round4(value)})
        denominator = max(1e-9, query_total + candidate_total - numerator)
        raw_score = max(0.0, min(1.0, numerator / denominator))
        weight = self.focus_score_weight if kind == "focus" else self.score_weight
        weighted_score = raw_score * weight
        return {
            "score": _round4(weighted_score),
            "raw_score": _round4(raw_score),
            "matched_count": len(shared),
            "relation_channels": {key: _round4(value) for key, value in sorted(channel_scores.items())},
            "relation_matches": matches,
        }

    def summary(self) -> dict:
        snapshot_count = sum(len(rows) for rows in self._relations_by_kind_id.values())
        context_count = len(self._contexts)
        target_edges = 0
        total_support = 0.0
        for context in self._contexts.values():
            targets = dict(context.get("targets", {}) or {})
            target_edges += len(targets)
            total_support += float(context.get("total", 0.0) or 0.0)
        return {
            "schema_id": "relative_relation_store/v1",
            "enabled": self.enabled,
            "snapshot_count": snapshot_count,
            "context_count": context_count,
            "target_edge_count": target_edges,
            "total_support": _round4(total_support),
            "total_events": int(self._total_events),
            "last_events": [dict(event) for event in self._last_events[:8]],
            "policy": {
                "max_relation_tokens_per_snapshot": int(self.max_relation_tokens_per_snapshot),
                "max_events_per_snapshot": int(self.max_events_per_snapshot),
                "context_limit": int(self.context_limit),
                "max_targets_per_context": int(self.max_targets_per_context),
                "update_semantics": "typed_relation_view;shared_evidence_not_shared_score",
            },
        }

    def extract_events(self, *, memory_kind: str, items: list[dict], focus_labels: list[str] | None = None) -> list[dict]:
        rows = self._ordered_rows(items)
        events: list[dict] = []
        events.extend(self._text_order_events(rows, scope="ordered"))
        events.extend(self._vision_relation_events(rows))
        events.extend(self._audio_relation_events(rows))
        focus_order = [str(label or "") for label in (focus_labels or []) if str(label or "")]
        if len(focus_order) >= 2:
            focus_rows = [self._row_from_label(label, idx) for idx, label in enumerate(focus_order)]
            events.extend(self._text_order_events(focus_rows, scope="focus"))
        events.sort(key=lambda event: (-float(event.get("weight", 0.0) or 0.0), str(event.get("relation_token", "") or "")))
        return events[: self.max_events_per_snapshot]

    def _text_order_events(self, rows: list[dict], *, scope: str) -> list[dict]:
        text_rows = [row for row in rows if self._is_text_label(str(row.get("sa_label", "") or ""))]
        if len(text_rows) < 2:
            return []
        events: list[dict] = []
        max_skip = 4
        for idx, left in enumerate(text_rows):
            for jdx in range(idx + 1, min(len(text_rows), idx + max_skip + 1)):
                right = text_rows[jdx]
                delta = jdx - idx
                distance_weight = 1.0 / sqrt(float(delta))
                if scope == "focus":
                    distance_weight *= 1.18
                weight = self._relation_weight(left, right, distance_weight=distance_weight)
                if weight <= 0.0:
                    continue
                left_label = str(left.get("sa_label", "") or "")
                right_label = str(right.get("sa_label", "") or "")
                relation_key = f"text_order::{scope}::forward::d{delta}"
                token = f"rel::{relation_key}::{left_label}>>{right_label}"
                events.append(
                    {
                        "relation_type": "text_order",
                        "relation_key": relation_key,
                        "relation_token": token,
                        "source_label": left_label,
                        "target_label": right_label,
                        "weight": _round4(weight),
                        "features": {
                            "delta_pos": delta,
                            "direction": 1,
                            "scope": scope,
                        },
                        "evidence": {
                            "real_energy_source": _round4(float(left.get("real_energy", 0.0) or 0.0)),
                            "real_energy_target": _round4(float(right.get("real_energy", 0.0) or 0.0)),
                            "focus_scope": scope == "focus",
                        },
                    }
                )
                if len(events) >= self.max_events_per_snapshot:
                    return events
        return events

    def _vision_relation_events(self, rows: list[dict]) -> list[dict]:
        objects = [row for row in rows if str(row.get("sa_label", "") or "").startswith("vision_obj::") or str(row.get("family", "") or "") == "vision_object"]
        events: list[dict] = []
        for idx, left in enumerate(objects):
            left_box = list(left.get("bbox_norm", []) or [])
            if len(left_box) < 4:
                continue
            for right in objects[idx + 1 : idx + 5]:
                right_box = list(right.get("bbox_norm", []) or [])
                if len(right_box) < 4:
                    continue
                events.extend(self._vision_spatial_pair_events(left, right, left_box, right_box))
                if len(events) >= self.max_events_per_snapshot:
                    return events
        for obj in objects[:12]:
            motion = list(obj.get("motion_vector", []) or [])
            if len(motion) < 3:
                continue
            events.extend(self._vision_motion_events(obj, motion))
            if len(events) >= self.max_events_per_snapshot:
                return events
        return events

    def _vision_spatial_pair_events(self, left: dict, right: dict, left_box: list[float], right_box: list[float]) -> list[dict]:
        lx, ly, lw, lh = [float(value or 0.0) for value in left_box[:4]]
        rx, ry, rw, rh = [float(value or 0.0) for value in right_box[:4]]
        dx = lx - rx
        dy = ly - ry
        distance = sqrt(dx * dx + dy * dy)
        relation_rows: list[tuple[str, str, str, str, float]] = []
        if abs(dx) >= max(0.03, abs(dy) * 0.75):
            relation_rows.append(("vision_spatial", "left_of" if dx < 0 else "right_of", str(left.get("sa_label", "")), str(right.get("sa_label", "")), abs(dx)))
            relation_rows.append(("vision_spatial", "right_of" if dx < 0 else "left_of", str(right.get("sa_label", "")), str(left.get("sa_label", "")), abs(dx)))
        if abs(dy) >= max(0.03, abs(dx) * 0.75):
            relation_rows.append(("vision_spatial", "above" if dy < 0 else "below", str(left.get("sa_label", "")), str(right.get("sa_label", "")), abs(dy)))
            relation_rows.append(("vision_spatial", "below" if dy < 0 else "above", str(right.get("sa_label", "")), str(left.get("sa_label", "")), abs(dy)))
        overlap = self._box_overlap(left_box, right_box)
        if overlap > 0.02:
            relation_rows.append(("vision_overlap", "overlap", str(left.get("sa_label", "")), str(right.get("sa_label", "")), overlap))
            relation_rows.append(("vision_overlap", "overlap", str(right.get("sa_label", "")), str(left.get("sa_label", "")), overlap))
        near_key = "near" if distance <= 0.38 else "far"
        near_strength = 1.0 - min(1.0, distance) if near_key == "near" else min(1.0, distance)
        relation_rows.append(("vision_spatial", near_key, str(left.get("sa_label", "")), str(right.get("sa_label", "")), near_strength))
        relation_rows.append(("vision_spatial", near_key, str(right.get("sa_label", "")), str(left.get("sa_label", "")), near_strength))
        events = []
        for relation_type, relation_name, source, target, strength in relation_rows:
            if not source or not target or source == target:
                continue
            weight = self._relation_weight(left, right, distance_weight=max(0.08, min(1.0, float(strength or 0.0))))
            if weight <= 0.0:
                continue
            relation_key = f"{relation_type}::ordered::{relation_name}"
            events.append(
                {
                    "relation_type": relation_type,
                    "relation_key": relation_key,
                    "relation_token": f"rel::{relation_key}::{source}>>{target}",
                    "source_label": source,
                    "target_label": target,
                    "weight": _round4(weight),
                    "features": {
                        "dx": _round4(dx),
                        "dy": _round4(dy),
                        "distance": _round4(distance),
                        "relation": relation_name,
                    },
                    "evidence": {
                        "real_energy_source": _round4(float(left.get("real_energy", 0.0) or 0.0)),
                        "real_energy_target": _round4(float(right.get("real_energy", 0.0) or 0.0)),
                    },
                }
            )
        return events

    def _vision_motion_events(self, row: dict, motion: list[float]) -> list[dict]:
        dx = float(motion[0] or 0.0)
        dy = float(motion[1] or 0.0)
        speed = float(motion[2] or 0.0)
        if speed <= 0.002:
            return []
        direction = "right" if abs(dx) >= abs(dy) and dx > 0 else "left" if abs(dx) >= abs(dy) else "down" if dy > 0 else "up"
        magnitude = "fast" if speed > 0.08 else "medium" if speed > 0.025 else "slow"
        source = str(row.get("sa_label", "") or "")
        relation_key = f"vision_motion::ordered::{direction}::{magnitude}"
        weight = self._relation_weight(row, row, distance_weight=max(0.1, min(1.0, speed * 8.0)))
        return [
            {
                "relation_type": "vision_motion",
                "relation_key": relation_key,
                "relation_token": f"rel::{relation_key}::{source}>>{source}",
                "source_label": source,
                "target_label": source,
                "weight": _round4(weight),
                "features": {"dx": _round4(dx), "dy": _round4(dy), "speed": _round4(speed), "direction": direction, "magnitude": magnitude},
                "evidence": {"real_energy_source": _round4(float(row.get("real_energy", 0.0) or 0.0))},
            }
        ]

    def _audio_relation_events(self, rows: list[dict]) -> list[dict]:
        audio_rows = [
            row
            for row in rows
            if str(row.get("sa_label", "") or "").startswith(("audio::", "audio_event::")) or str(row.get("family", "") or "").startswith("audio")
        ]
        if not audio_rows:
            return []
        events: list[dict] = []
        ordered = sorted(audio_rows, key=lambda row: (int(row.get("tick_index", -1)), int(row.get("position", 0)), str(row.get("sa_label", ""))))
        for left, right in zip(ordered, ordered[1:]):
            source = str(left.get("sa_label", "") or "")
            target = str(right.get("sa_label", "") or "")
            if not source or not target or source == target:
                continue
            relation_key = "audio_order::ordered::next"
            events.append(
                {
                    "relation_type": "audio_order",
                    "relation_key": relation_key,
                    "relation_token": f"rel::{relation_key}::{source}>>{target}",
                    "source_label": source,
                    "target_label": target,
                    "weight": _round4(self._relation_weight(left, right, distance_weight=0.82)),
                    "features": {"direction": 1, "scope": "ordered"},
                    "evidence": {
                        "real_energy_source": _round4(float(left.get("real_energy", 0.0) or 0.0)),
                        "real_energy_target": _round4(float(right.get("real_energy", 0.0) or 0.0)),
                    },
                }
            )
            if len(events) >= self.max_events_per_snapshot:
                return events
        for row in audio_rows[:16]:
            label = str(row.get("sa_label", "") or "")
            pitch = list(row.get("audio_pitch", []) or [])
            rhythm = list(row.get("audio_rhythm", []) or [])
            band = list(row.get("audio_band", []) or [])
            if len(pitch) >= 1 and float(pitch[0] or 0.0) > 0:
                band_name = "high" if float(pitch[0]) > 0.22 else "mid" if float(pitch[0]) > 0.07 else "low"
                events.append(self._self_relation_event(row, relation_type="audio_pitch", relation_name=band_name, source=label, strength=max(0.1, float(pitch[0]))))
            if len(rhythm) >= 5 and float(rhythm[4] or 0.0) > 0.08:
                events.append(self._self_relation_event(row, relation_type="audio_rhythm", relation_name="period_stable", source=label, strength=float(rhythm[4])))
            if len(band) >= 2:
                peak = max(range(len(band)), key=lambda idx: float(band[idx] or 0.0))
                band_name = "low_band" if peak < len(band) / 3 else "mid_band" if peak < len(band) * 2 / 3 else "high_band"
                events.append(self._self_relation_event(row, relation_type="audio_band", relation_name=band_name, source=label, strength=float(band[peak] or 0.0)))
            if len(events) >= self.max_events_per_snapshot:
                return events
        return [event for event in events if event]

    def _self_relation_event(self, row: dict, *, relation_type: str, relation_name: str, source: str, strength: float) -> dict:
        relation_key = f"{relation_type}::ordered::{relation_name}"
        return {
            "relation_type": relation_type,
            "relation_key": relation_key,
            "relation_token": f"rel::{relation_key}::{source}>>{source}",
            "source_label": source,
            "target_label": source,
            "weight": _round4(self._relation_weight(row, row, distance_weight=max(0.08, min(1.0, float(strength or 0.0))))),
            "features": {"relation": relation_name, "strength": _round4(strength)},
            "evidence": {"real_energy_source": _round4(float(row.get("real_energy", 0.0) or 0.0))},
        }

    def _observe_events(self, events: list[dict], *, tick_index: int) -> None:
        clean_events = []
        for event in events[: self.max_events_per_snapshot]:
            token = str((event or {}).get("relation_token", "") or "")
            source = str((event or {}).get("source_label", "") or "")
            target = str((event or {}).get("target_label", "") or "")
            if not token or not source or not target:
                continue
            weight = max(0.0, float((event or {}).get("weight", 0.0) or 0.0))
            if weight <= 0.0:
                continue
            context = self._touch_context(token, tick_index=tick_index)
            targets = context.setdefault("targets", OrderedDict())
            if target not in targets and len(targets) >= self.max_targets_per_context:
                self._evict_weakest_target(targets)
            row = targets.setdefault(target, {"weight": 0.0, "support": 0, "last_tick": int(tick_index)})
            row["weight"] = float(row.get("weight", 0.0) or 0.0) + weight
            row["support"] = int(row.get("support", 0) or 0) + 1
            row["last_tick"] = int(tick_index)
            context["total"] = float(context.get("total", 0.0) or 0.0) + weight
            self._rescale_if_needed(context)
            clean_events.append(
                {
                    "relation_type": str(event.get("relation_type", "") or ""),
                    "relation_token": token,
                    "source_label": source,
                    "target_label": target,
                    "weight": _round4(weight),
                }
            )
        if clean_events:
            self._last_events = clean_events[:8]
            self._total_events += len(clean_events)

    def _touch_context(self, key: str, *, tick_index: int) -> dict:
        clean = str(key or "")
        if not clean:
            clean = "empty"
        if clean not in self._contexts and len(self._contexts) >= self.context_limit:
            self._contexts.popitem(last=False)
        context = self._contexts.setdefault(
            clean,
            {
                "context_key": clean,
                "total": 0.0,
                "targets": OrderedDict(),
                "last_tick": int(tick_index),
            },
        )
        self._contexts.move_to_end(clean)
        last_tick = int(context.get("last_tick", tick_index) if context.get("last_tick", None) is not None else tick_index)
        age = max(0, int(tick_index) - last_tick)
        if age > 0 and self.decay < 1.0:
            scale = self.decay ** age
            targets = context.setdefault("targets", OrderedDict())
            for label in list(targets.keys()):
                row = targets[label]
                row["weight"] = float(row.get("weight", 0.0) or 0.0) * scale
                if float(row.get("weight", 0.0) or 0.0) < self.min_support * 0.05:
                    targets.pop(label, None)
            context["total"] = sum(float(row.get("weight", 0.0) or 0.0) for row in targets.values())
        context["last_tick"] = int(tick_index)
        return context

    def _rescale_if_needed(self, context: dict) -> None:
        total = float(context.get("total", 0.0) or 0.0)
        if total <= self.rescale_threshold:
            return
        targets = context.setdefault("targets", OrderedDict())
        for row in targets.values():
            row["weight"] = float(row.get("weight", 0.0) or 0.0) * self.rescale_factor
        context["total"] = sum(float(row.get("weight", 0.0) or 0.0) for row in targets.values())

    def _evict_weakest_target(self, targets: OrderedDict[str, dict]) -> None:
        weakest = None
        weakest_weight = None
        for label, row in targets.items():
            weight = float(row.get("weight", 0.0) or 0.0)
            if weakest is None or weight < float(weakest_weight or 0.0):
                weakest = label
                weakest_weight = weight
        if weakest is not None:
            targets.pop(weakest, None)

    def _relation_weight(self, left: dict, right: dict, *, distance_weight: float) -> float:
        left_real = max(0.0, float(left.get("real_energy", 0.0) or 0.0))
        right_real = max(0.0, float(right.get("real_energy", 0.0) or 0.0))
        if left_real <= 0.0 and right_real <= 0.0:
            return 0.0
        signal = sqrt(max(0.03, left_real) * max(0.03, right_real))
        return max(0.01, min(2.0, signal * max(0.05, float(distance_weight or 0.0))))

    def _box_overlap(self, left_box: list[float], right_box: list[float]) -> float:
        if len(left_box) < 4 or len(right_box) < 4:
            return 0.0
        lx, ly, lw, lh = [float(value or 0.0) for value in left_box[:4]]
        rx, ry, rw, rh = [float(value or 0.0) for value in right_box[:4]]
        l1, t1, r1, b1 = lx - lw / 2.0, ly - lh / 2.0, lx + lw / 2.0, ly + lh / 2.0
        l2, t2, r2, b2 = rx - rw / 2.0, ry - rh / 2.0, rx + rw / 2.0, ry + rh / 2.0
        inter_w = max(0.0, min(r1, r2) - max(l1, l2))
        inter_h = max(0.0, min(b1, b2) - max(t1, t2))
        inter = inter_w * inter_h
        union = max(1e-9, lw * lh + rw * rh - inter)
        return max(0.0, min(1.0, inter / union))

    def _ordered_rows(self, items: list[dict]) -> list[dict]:
        rows = []
        for index, item in enumerate(items or []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            anchor_meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
            numeric_features = dict(item.get("numeric_features", {}) or {}) if isinstance(item.get("numeric_features", {}), dict) else {}
            position = item.get("position", anchor_meta.get("position", index))
            tick = item.get("last_seen_tick", item.get("tick_index", anchor_meta.get("tick_index", -1)))
            try:
                position_key = int(position)
            except (TypeError, ValueError):
                position_key = index
            try:
                tick_key = int(tick)
            except (TypeError, ValueError):
                tick_key = -1
            rows.append(
                {
                    "sa_label": label,
                    "position": position_key,
                    "tick_index": tick_key,
                    "order_index": index,
                    "real_energy": float(item.get("real_energy", item.get("query_weight", 0.0)) or 0.0),
                    "family": str(item.get("family", "") or ""),
                    "source_type": str(item.get("source_type", "") or ""),
                    "bbox_norm": list(anchor_meta.get("bbox_norm", []) or []),
                    "motion_vector": list(numeric_features.get("vision.motion_vector", []) or anchor_meta.get("motion_vector", []) or []),
                    "audio_pitch": list(numeric_features.get("audio.pitch", []) or []),
                    "audio_rhythm": list(numeric_features.get("audio.rhythm", []) or []),
                    "audio_band": list(numeric_features.get("audio.band", []) or numeric_features.get("audio.spectrum", []) or []),
                }
            )
        rows.sort(key=lambda row: (int(row.get("tick_index", -1)), int(row.get("position", 0)), int(row.get("order_index", 0)), str(row.get("sa_label", ""))))
        return rows

    def _row_from_label(self, label: str, index: int) -> dict:
        return {
            "sa_label": str(label or ""),
            "position": int(index),
            "tick_index": -1,
            "order_index": int(index),
            "real_energy": 1.0,
            "family": "focus",
            "source_type": "focus_order",
        }

    def _is_text_label(self, label: str) -> bool:
        clean = str(label or "")
        return clean.startswith(("text::", "phrase::"))

    def _channel_from_token(self, token: str) -> str:
        clean = str(token or "")
        if clean.startswith("rel::"):
            parts = clean.split("::")
            if len(parts) >= 3:
                return str(parts[1] or "unknown")
        return "unknown"

    def _empty_features(self, *, memory_kind: str) -> dict:
        return {
            "schema_id": "relative_relation_features/v1",
            "memory_kind": str(memory_kind or ""),
            "relation_tokens": [],
            "relation_token_weights": {},
            "relation_channels": {},
            "relation_events": [],
            "event_count": 0,
        }

    def _empty_score(self) -> dict:
        return {
            "score": 0.0,
            "raw_score": 0.0,
            "matched_count": 0,
            "relation_channels": {},
            "relation_matches": [],
        }
