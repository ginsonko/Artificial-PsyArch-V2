from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _label_tokens(snapshot: dict) -> list[str]:
    labels: list[str] = []
    for item in list(snapshot.get("state_field_items", []) or snapshot.get("items", []) or []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("sa_label", "") or "")
        if label:
            labels.append(label)
    return labels


def _family_counts(snapshot: dict) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in list(snapshot.get("state_field_items", []) or snapshot.get("items", []) or []):
        if not isinstance(item, dict):
            continue
        family = str(item.get("family", "") or "unknown")
        counts[family] += 1
    return dict(counts)


def _source_text(snapshot: dict) -> str:
    return str(snapshot.get("source_text", "") or "")


def _checksum_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class LayeredMemoryConfig:
    hot_keep_ticks: int = 60
    hot_capacity: int = 32
    min_rehydrate_score: float = 0.38
    prototype_min_support: int = 3
    compression_level: int = 6


class LayeredFidelityMemory:
    """
    AP-compatible layered memory helper.

    This class does not replace MemoryStore. It is a storage/rehab layer that
    keeps B/C-recallable warm tickets while moving bulky traces into cold gzip
    archives. Cold payloads are only rehydrated when query pressure says the
    full trace is needed.
    """

    def __init__(self, archive_dir: str | Path, *, config: LayeredMemoryConfig | None = None) -> None:
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or LayeredMemoryConfig()
        self.hot_snapshots: dict[str, dict] = {}
        self.warm_tickets: dict[str, dict] = {}
        self.cold_refs: dict[str, dict] = {}
        self.successors_by_source: dict[str, list[str]] = defaultdict(list)
        self.prototypes: dict[str, dict] = {}
        self.slow_focus_tickets: dict[str, dict] = {}
        self._prototype_support: Counter[str] = Counter()
        self._prototype_sources: dict[str, list[str]] = defaultdict(list)
        self._rehydrated_cache: dict[str, dict] = {}
        self._last_current_tick = 0

    def add_snapshot(self, snapshot: dict, *, successor_ids: list[str] | None = None, prototype_key: str | None = None) -> dict:
        clean = dict(snapshot)
        memory_id = str(clean.get("memory_id", "") or "")
        if not memory_id:
            raise ValueError("snapshot requires memory_id")
        clean.setdefault("state_field_items", list(clean.get("items", []) or []))
        tick = int(clean.get("tick_index", 0) or 0)
        self._last_current_tick = max(self._last_current_tick, tick)
        labels = _label_tokens(clean)
        ticket = {
            "schema_id": "layered_memory_ticket/v1",
            "memory_id": memory_id,
            "tier": "hot",
            "tick_index": tick,
            "source_text": _source_text(clean),
            "labels": labels,
            "families": _family_counts(clean),
            "successor_ids": list(successor_ids or []),
            "prototype_key": str(prototype_key or self._infer_prototype_key(clean)),
            "cold_ref": None,
            "temporal_applicability_floor": 0.12,
            "all_sa_first_class": True,
        }
        self.hot_snapshots[memory_id] = clean
        self.warm_tickets[memory_id] = ticket
        for successor_id in list(successor_ids or []):
            if successor_id not in self.successors_by_source[memory_id]:
                self.successors_by_source[memory_id].append(successor_id)
        if ticket["prototype_key"]:
            self._prototype_support[ticket["prototype_key"]] += 1
            sources = self._prototype_sources[ticket["prototype_key"]]
            if memory_id not in sources:
                sources.append(memory_id)
            if self._prototype_support[ticket["prototype_key"]] >= self.config.prototype_min_support:
                self.prototypes[ticket["prototype_key"]] = {
                    "schema_id": "skill_prototype/v1",
                    "prototype_key": ticket["prototype_key"],
                    "support": int(self._prototype_support[ticket["prototype_key"]]),
                    "source_memory_ids": list(sources[-12:]),
                    "labels": sorted(set(labels)),
                    "policy": "prototype_is_reusable_process_not_answer_table",
                }
        return ticket

    def add_slow_focus(self, focus: dict) -> dict:
        clean = dict(focus)
        focus_id = str(clean.get("focus_id", "") or "")
        origin_memory_id = str(clean.get("origin_memory_id", "") or "")
        focus_label = str(clean.get("focus_label", "") or "")
        if not focus_id:
            raise ValueError("slow focus requires focus_id")
        if not origin_memory_id:
            raise ValueError("slow focus requires origin_memory_id")
        if origin_memory_id not in self.warm_tickets:
            raise ValueError(f"slow focus origin is not indexed: {origin_memory_id}")
        if not focus_label:
            raise ValueError("slow focus requires focus_label")
        origin_ticket = self.warm_tickets[origin_memory_id]
        extra_labels = [str(label) for label in clean.get("labels", []) or [] if str(label)]
        labels = list(dict.fromkeys([focus_label] + extra_labels))
        ticket = {
            "schema_id": "slow_focus_ticket/v1",
            "focus_id": focus_id,
            "focus_label": focus_label,
            "labels": labels,
            "origin_memory_id": origin_memory_id,
            "origin_tick_index": int(origin_ticket.get("tick_index", 0) or 0),
            "focus_strength": _clamp(float(clean.get("focus_strength", 1.0) or 1.0), 0.0, 1.5),
            "successor_ids": list(origin_ticket.get("successor_ids", []) or []),
            "policy": "slow_focus_is_wave_peak_index_for_fast_state_snapshot",
        }
        self.slow_focus_tickets[focus_id] = ticket
        return ticket

    def cool_down(self, *, current_tick: int) -> dict:
        self._last_current_tick = max(self._last_current_tick, int(current_tick))
        moved: list[str] = []
        for memory_id, snapshot in list(self.hot_snapshots.items()):
            age = max(0, int(current_tick) - int(snapshot.get("tick_index", 0) or 0))
            if age >= self.config.hot_keep_ticks or len(self.hot_snapshots) > self.config.hot_capacity:
                self._move_to_cold(memory_id, snapshot)
                moved.append(memory_id)
        return {
            "schema_id": "layered_memory_cooldown/v1",
            "current_tick": int(current_tick),
            "moved_to_cold": moved,
            "hot_count": len(self.hot_snapshots),
            "warm_ticket_count": len(self.warm_tickets),
            "cold_count": len(self.cold_refs),
            "prototype_count": len(self.prototypes),
        }

    def recall(self, query_items: list[dict], *, current_tick: int, top_k: int = 5, mode: str = "ordinary") -> list[dict]:
        query_labels = [str(item.get("sa_label", "") or "") for item in query_items if isinstance(item, dict) and item.get("sa_label")]
        content_query_labels = [label for label in query_labels if not self._is_rehab_control_label(label)]
        query_set = set(query_labels)
        content_query_set = set(content_query_labels) or query_set
        rehydration_pressure = self._rehydration_pressure(query_items, mode=mode)
        rehydrate = bool(rehydration_pressure["should_rehydrate"])
        rows: list[dict] = []
        for ticket in self.warm_tickets.values():
            label_set = set(ticket.get("labels", []) or [])
            overlap = len(content_query_set & label_set)
            if not overlap:
                continue
            union = max(1, len(content_query_set | label_set))
            label_score = overlap / union
            age = max(0, int(current_tick) - int(ticket.get("tick_index", 0) or 0))
            temporal = self._temporal_weight(age, floor=float(ticket.get("temporal_applicability_floor", 0.12) or 0.12))
            prototype_bonus = 0.08 if ticket.get("prototype_key") in self.prototypes else 0.0
            score = _round4(label_score * temporal + prototype_bonus)
            cold_ref = ticket.get("cold_ref")
            loaded_payload = None
            loaded = False
            rehydrate_threshold = self.config.min_rehydrate_score
            if str(mode) in {"rehab", "deep_replay", "audit"}:
                rehydrate_threshold = min(rehydrate_threshold, 0.12)
            if cold_ref and rehydrate and score >= rehydrate_threshold:
                loaded_payload = self.rehydrate(str(ticket["memory_id"]))
                loaded = loaded_payload is not None
            rows.append(
                {
                    "memory_id": ticket["memory_id"],
                    "tier": "cold_rehydrated" if loaded else str(ticket.get("tier", "warm_ticket")),
                    "score": score,
                    "label_score": _round4(label_score),
                    "temporal_applicability": _round4(temporal),
                    "source_text": ticket.get("source_text", ""),
                    "labels": list(ticket.get("labels", []) or []),
                    "successor_ids": list(ticket.get("successor_ids", []) or []),
                    "prototype": self.prototypes.get(str(ticket.get("prototype_key", "") or ""), {}),
                    "cold_ref": cold_ref,
                    "rehydrated": loaded,
                    "rehydrated_payload": loaded_payload if loaded else None,
                    "recall_policy": "warm_ticket_first_cold_on_rehab_pressure",
                    "rehydration_pressure": rehydration_pressure,
                }
            )
        rows.sort(key=lambda row: (-float(row.get("score", 0.0) or 0.0), str(row.get("memory_id", ""))))
        return rows[: max(1, int(top_k))]

    def recall_slow_focus_context(self, query_items: list[dict], *, current_tick: int, top_k: int = 5, mode: str = "ordinary") -> list[dict]:
        query_labels = [str(item.get("sa_label", "") or "") for item in query_items if isinstance(item, dict) and item.get("sa_label")]
        content_query_labels = [label for label in query_labels if not self._is_rehab_control_label(label)]
        if not content_query_labels:
            return []
        content_query_set = set(content_query_labels)
        rehydration_pressure = self._rehydration_pressure(query_items, mode=mode)
        has_pressure = bool(rehydration_pressure["should_rehydrate"])
        rows: list[dict] = []
        for focus_ticket in self.slow_focus_tickets.values():
            origin_memory_id = str(focus_ticket.get("origin_memory_id", "") or "")
            origin_ticket = self.warm_tickets.get(origin_memory_id)
            if not origin_ticket:
                continue
            label_set = set(focus_ticket.get("labels", []) or [])
            overlap = len(content_query_set & label_set)
            if not overlap:
                continue
            union = max(1, len(content_query_set | label_set))
            label_score = overlap / union
            age = max(0, int(current_tick) - int(focus_ticket.get("origin_tick_index", 0) or 0))
            temporal = self._temporal_weight(age, floor=float(origin_ticket.get("temporal_applicability_floor", 0.12) or 0.12))
            exact_focus_bonus = 0.22 if str(focus_ticket.get("focus_label", "") or "") in content_query_set else 0.0
            focus_strength_bonus = 0.06 * float(focus_ticket.get("focus_strength", 1.0) or 1.0)
            prototype_bonus = 0.06 if origin_ticket.get("prototype_key") in self.prototypes else 0.0
            score = _round4(_clamp(label_score * temporal + exact_focus_bonus + focus_strength_bonus + prototype_bonus, 0.0, 1.0))
            cold_ref = origin_ticket.get("cold_ref")
            prefetch_threshold = 0.16 if str(mode) in {"ordinary", "prefetch"} else 0.10
            should_prefetch = score >= prefetch_threshold
            payload = self.rehydrate(origin_memory_id) if should_prefetch else None
            prefetched = payload is not None
            promoted = bool(prefetched and has_pressure and score >= 0.10)
            rows.append(
                {
                    "schema_id": "slow_focus_context_recall/v1",
                    "focus_id": focus_ticket["focus_id"],
                    "focus_label": focus_ticket["focus_label"],
                    "origin_memory_id": origin_memory_id,
                    "score": score,
                    "label_score": _round4(label_score),
                    "temporal_applicability": _round4(temporal),
                    "source_snapshot_tier": str(origin_ticket.get("tier", "warm_ticket")),
                    "tier": "cold_prefetched" if prefetched and cold_ref else ("hot_prefetched" if prefetched else "warm_focus_ticket"),
                    "prefetched": prefetched,
                    "promoted": promoted,
                    "current_state_injection": False,
                    "virtual_context_items": self._virtual_context_items(payload, focus_ticket=focus_ticket, score=score) if promoted else [],
                    "successor_ids": list(origin_ticket.get("successor_ids", []) or []),
                    "rehydration_pressure": rehydration_pressure,
                    "policy": "slow_focus_prefetches_fast_snapshot_but_only_pressure_promotes_virtual_context",
                }
            )
        rows.sort(key=lambda row: (-float(row.get("score", 0.0) or 0.0), str(row.get("focus_id", ""))))
        return rows[: max(1, int(top_k))]

    def successors(self, memory_id: str, *, current_tick: int, rehydrate: bool = False) -> list[dict]:
        source_id = str(memory_id or "")
        rows: list[dict] = []
        for successor_id in self.successors_by_source.get(source_id, []):
            ticket = self.warm_tickets.get(successor_id)
            if not ticket:
                continue
            payload = self.rehydrate(successor_id) if rehydrate and ticket.get("cold_ref") else None
            rows.append(
                {
                    "source_memory_id": source_id,
                    "successor_memory_id": successor_id,
                    "tier": "cold_rehydrated" if payload else str(ticket.get("tier", "warm_ticket")),
                    "predicted_labels": list(ticket.get("labels", []) or []),
                    "source_text": ticket.get("source_text", ""),
                    "payload": payload,
                    "temporal_applicability": self._temporal_weight(max(0, int(current_tick) - int(ticket.get("tick_index", 0) or 0))),
                }
            )
        return rows

    def rehydrate(self, memory_id: str) -> dict | None:
        clean_id = str(memory_id or "")
        if clean_id in self.hot_snapshots:
            return dict(self.hot_snapshots[clean_id])
        if clean_id in self._rehydrated_cache:
            return dict(self._rehydrated_cache[clean_id])
        ref = self.cold_refs.get(clean_id)
        if not ref:
            return None
        path = Path(ref["path"])
        raw = path.read_bytes()
        if _checksum_bytes(raw) != ref.get("compressed_sha256"):
            raise ValueError(f"cold archive checksum mismatch: {clean_id}")
        data = gzip.decompress(raw)
        if _checksum_bytes(data) != ref.get("payload_sha256"):
            raise ValueError(f"cold payload checksum mismatch: {clean_id}")
        payload = json.loads(data.decode("utf-8"))
        self._rehydrated_cache[clean_id] = payload
        return dict(payload)

    def rewarm(self, memory_id: str, *, current_tick: int) -> dict:
        payload = self.rehydrate(memory_id)
        if not payload:
            return {"memory_id": memory_id, "rewarmed": False, "reason": "missing_cold_payload"}
        hot = dict(payload)
        hot["tick_index"] = int(current_tick)
        hot["source_text"] = f"rehab refresher::{_source_text(payload)}"
        self.hot_snapshots[str(memory_id)] = hot
        ticket = self.warm_tickets[str(memory_id)]
        ticket["tier"] = "hot"
        ticket["tick_index"] = int(current_tick)
        ticket["source_text"] = hot["source_text"]
        return {"memory_id": str(memory_id), "rewarmed": True, "current_tick": int(current_tick)}

    def storage_summary(self) -> dict:
        cold_bytes = sum(int(ref.get("compressed_bytes", 0) or 0) for ref in self.cold_refs.values())
        raw_bytes = sum(int(ref.get("raw_bytes", 0) or 0) for ref in self.cold_refs.values())
        return {
            "schema_id": "layered_memory_storage_summary/v1",
            "hot_count": len(self.hot_snapshots),
            "warm_ticket_count": len(self.warm_tickets),
            "cold_count": len(self.cold_refs),
            "prototype_count": len(self.prototypes),
            "slow_focus_count": len(self.slow_focus_tickets),
            "cold_raw_bytes": raw_bytes,
            "cold_compressed_bytes": cold_bytes,
            "compression_ratio": _round4(cold_bytes / raw_bytes) if raw_bytes else 1.0,
            "policy": "hot_full_warm_ticket_cold_lossless_archive_rehab_recall",
        }

    def _move_to_cold(self, memory_id: str, snapshot: dict) -> None:
        payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        compressed = gzip.compress(payload, compresslevel=self.config.compression_level)
        path = self.archive_dir / f"{memory_id}.json.gz"
        path.write_bytes(compressed)
        ref = {
            "schema_id": "cold_archive_ref/v1",
            "memory_id": memory_id,
            "path": str(path.resolve()),
            "raw_bytes": len(payload),
            "compressed_bytes": len(compressed),
            "payload_sha256": _checksum_bytes(payload),
            "compressed_sha256": _checksum_bytes(compressed),
            "lossless": True,
        }
        self.cold_refs[memory_id] = ref
        ticket = self.warm_tickets[memory_id]
        ticket["tier"] = "warm_ticket"
        ticket["cold_ref"] = ref
        self.hot_snapshots.pop(memory_id, None)
        self._rehydrated_cache.pop(memory_id, None)

    def _infer_prototype_key(self, snapshot: dict) -> str:
        labels = _label_tokens(snapshot)
        action_labels = [label for label in labels if label.startswith("action::")]
        feeling_labels = [label for label in labels if label.startswith("feeling::")]
        feedback_labels = [label for label in labels if label.startswith("action_feedback::") or "reward" in label or "punishment" in label]
        if action_labels:
            return "|".join(sorted(set(action_labels + feeling_labels[:2] + feedback_labels[:2])))
        return "|".join(sorted(set(labels[:4])))

    def _should_rehydrate(self, query_items: list[dict], *, mode: str) -> bool:
        return bool(self._rehydration_pressure(query_items, mode=mode)["should_rehydrate"])

    def _rehydration_pressure(self, query_items: list[dict], *, mode: str) -> dict:
        reasons: list[str] = []
        if str(mode) in {"rehab", "deep_replay", "audit"}:
            reasons.append(f"mode::{mode}")
        labels = {str(item.get("sa_label", "") or "") for item in query_items if isinstance(item, dict)}
        if labels & {"action::replay_episode", "action::recall_by_timefelt", "action::deep_reread_memory"}:
            reasons.append("replay_action")
        if labels & {"feeling::rehabilitation_need", "feeling::high_uncertainty", "feeling::dissonance", "feeling::after_failure"}:
            reasons.append("rehab_or_failure_feeling")
        if labels & {"feeling::context_needed", "feeling::successor_unclear"}:
            reasons.append("context_or_successor_pressure")
        return {
            "schema_id": "cold_rehydration_pressure/v1",
            "should_rehydrate": bool(reasons),
            "mode": str(mode),
            "reasons": reasons,
            "content_labels": sorted(label for label in labels if not self._is_rehab_control_label(label)),
            "control_labels": sorted(label for label in labels if self._is_rehab_control_label(label)),
            "policy": "control_labels_gate_rehydrate_but_do_not_replace_content_match",
        }

    def _is_rehab_control_label(self, label: str) -> bool:
        clean = str(label or "")
        return clean in {
            "action::replay_episode",
            "action::recall_by_timefelt",
            "action::deep_reread_memory",
            "feeling::rehabilitation_need",
            "feeling::high_uncertainty",
            "feeling::dissonance",
            "feeling::after_failure",
            "feeling::context_needed",
            "feeling::successor_unclear",
        }

    def _virtual_context_items(self, payload: dict | None, *, focus_ticket: dict, score: float, limit: int = 8) -> list[dict]:
        if not payload:
            return []
        items = list(payload.get("state_field_items", []) or payload.get("items", []) or [])
        virtual_items: list[dict] = []
        virtual_scale = _clamp(float(score) * 0.42, 0.05, 0.32)
        for item in items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label or self._is_rehab_control_label(label):
                continue
            promoted = dict(item)
            promoted["real_energy"] = 0.0
            promoted["virtual_energy"] = _round4(max(float(item.get("virtual_energy", 0.0) or 0.0), virtual_scale))
            promoted["cognitive_pressure"] = _round4(max(float(item.get("cognitive_pressure", 0.0) or 0.0) * 0.25, virtual_scale))
            promoted["source_type"] = "slow_focus_prefetched_virtual_context"
            meta = dict(promoted.get("anchor_meta", {}) or {})
            meta.update(
                {
                    "origin_memory_id": focus_ticket.get("origin_memory_id"),
                    "slow_focus_id": focus_ticket.get("focus_id"),
                    "current_state_injection": False,
                    "promotion_policy": "virtual_context_candidate_only",
                }
            )
            promoted["anchor_meta"] = meta
            virtual_items.append(promoted)
            if len(virtual_items) >= int(limit):
                break
        return virtual_items

    def _temporal_weight(self, age_ticks: int, *, floor: float = 0.12) -> float:
        if age_ticks <= 0:
            return 1.0
        if age_ticks < self.config.hot_keep_ticks:
            return 0.92
        # Long memories fade in applicability but keep a nonzero floor.
        half_life = max(1.0, self.config.hot_keep_ticks * 8.0)
        decayed = 0.18 + 0.82 * (0.5 ** (float(age_ticks) / half_life))
        return _clamp(decayed, floor, 1.0)


__all__ = ["LayeredFidelityMemory", "LayeredMemoryConfig"]
