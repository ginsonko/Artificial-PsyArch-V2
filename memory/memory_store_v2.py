# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict, defaultdict, deque
import json
from pathlib import Path
from typing import Any

import numpy as np

from .embedding_v2 import HashEmbeddingV2
from .spacetime_index_v2 import SpacetimeIndexV2
from .vector_index_v2 import VectorIndexV2


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _display_from_label(label: str) -> str:
    clean = str(label or "")
    for prefix in ("text::", "phrase::", "attr::", "vision::", "audio::", "action::"):
        if clean.startswith(prefix):
            return clean[len(prefix) :]
    return clean


class MemoryStoreV2:
    def __init__(
        self,
        *,
        max_recent: int = 2048,
        max_branch_neighbors: int = 8,
        recent_window_size: int = 64,
        vector_dim: int = 256,
        vector_backend: str = "auto",
        ann_enabled: bool = True,
        ann_top_k: int = 48,
        candidate_limit: int = 128,
        spacetime_backend: str = "bucket_grid",
        time_bucket_size: int = 8,
        space_bucket_size: float = 0.25,
        spacetime_time_radius: int = 24,
        spacetime_space_radius: float = 0.45,
        recall_fatigue_decay: float = 0.78,
        recall_fatigue_gain: float = 0.55,
        recall_fatigue_accumulate_scale: float = 0.4,
        recall_fatigue_max: float = 1.5,
        recall_fatigue_min_multiplier: float = 0.22,
        branch_credibility_decay: float = 0.86,
        branch_credibility_match_gain: float = 0.24,
        branch_credibility_miss_gain: float = 0.30,
        branch_credibility_min_multiplier: float = 0.28,
        branch_credibility_max_multiplier: float = 1.72,
    ) -> None:
        self.max_recent = max(8, int(max_recent))
        self.max_branch_neighbors = max(1, int(max_branch_neighbors))
        self.recent_window_size = max(8, int(recent_window_size))
        self.vector_dim = max(32, int(vector_dim))
        self.vector_backend = str(vector_backend or "auto")
        self.ann_enabled = bool(ann_enabled)
        self.ann_top_k = max(8, int(ann_top_k))
        self.candidate_limit = max(8, int(candidate_limit))
        self.spacetime_backend = str(spacetime_backend or "bucket_grid")
        self.recall_fatigue_decay = max(0.0, min(1.0, float(recall_fatigue_decay)))
        self.recall_fatigue_gain = max(0.0, float(recall_fatigue_gain))
        self.recall_fatigue_accumulate_scale = max(0.0, float(recall_fatigue_accumulate_scale))
        self.recall_fatigue_max = max(0.0, float(recall_fatigue_max))
        self.recall_fatigue_min_multiplier = max(0.0, min(1.0, float(recall_fatigue_min_multiplier)))
        self.branch_credibility_decay = max(0.0, min(1.0, float(branch_credibility_decay)))
        self.branch_credibility_match_gain = max(0.0, float(branch_credibility_match_gain))
        self.branch_credibility_miss_gain = max(0.0, float(branch_credibility_miss_gain))
        self.branch_credibility_min_multiplier = max(0.0, min(1.0, float(branch_credibility_min_multiplier)))
        self.branch_credibility_max_multiplier = max(
            self.branch_credibility_min_multiplier,
            float(branch_credibility_max_multiplier),
        )
        self._memories: list[dict[str, Any]] = []
        self._memories_by_id: dict[str, dict[str, Any]] = {}
        self._recent_ids: deque[str] = deque(maxlen=self.max_recent)
        self._recent_window_ids: deque[str] = deque(maxlen=self.recent_window_size)
        self._counter = 0
        self._posting_by_label: dict[str, set[str]] = defaultdict(set)
        self._posting_by_unit: dict[str, set[str]] = defaultdict(set)
        self._posting_by_bigram: dict[str, set[str]] = defaultdict(set)
        self._embedder = HashEmbeddingV2(dim=self.vector_dim)
        self._vector_index = VectorIndexV2(
            dim=self.vector_dim,
            backend=self.vector_backend,
            ann_enabled=self.ann_enabled,
            ann_top_k=self.ann_top_k,
        )
        self._spacetime_index = SpacetimeIndexV2(
            backend=self.spacetime_backend,
            time_bucket_size=time_bucket_size,
            space_bucket_size=space_bucket_size,
            default_time_radius=spacetime_time_radius,
            default_space_radius=spacetime_space_radius,
        )
        self._memory_revision = 0
        self._query_vector_cache_limit = 192
        self._candidate_cache_limit = 96
        self._neighbor_rows_cache_limit = 512
        self._pair_relation_cache_limit = 4096
        self._memory_profile_cache: dict[str, dict[str, Any]] = {}
        self._query_vector_cache: OrderedDict[str, tuple[np.ndarray, list[str]]] = OrderedDict()
        self._candidate_cache: OrderedDict[tuple[int, str], dict[str, Any]] = OrderedDict()
        self._neighbor_rows_cache: OrderedDict[tuple[int, str, int], list[dict[str, Any]]] = OrderedDict()
        self._pair_relation_cache: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._recall_fatigue: dict[str, dict[str, Any]] = {}
        self._branch_credibility: dict[str, dict[str, Any]] = {}
        self._cache_stats: dict[str, int] = {
            "query_vector_hit": 0,
            "query_vector_miss": 0,
            "candidate_hit": 0,
            "candidate_miss": 0,
            "neighbor_hit": 0,
            "neighbor_miss": 0,
            "pair_hit": 0,
            "pair_miss": 0,
        }

    def count(self) -> int:
        return len(self._memories)

    def latest_tick_index(self) -> int:
        if not self._memories:
            return -1
        return max(int(-1 if memory.get("tick_index", -1) is None else memory.get("tick_index", -1)) for memory in self._memories)

    def _recall_fatigue_value(self, memory_id: str, *, tick_index: int) -> float:
        clean_id = str(memory_id or "")
        if not clean_id:
            return 0.0
        entry = self._recall_fatigue.get(clean_id)
        if not isinstance(entry, dict):
            return 0.0
        last_tick = int(entry.get("tick_index", tick_index) or tick_index)
        value = max(0.0, float(entry.get("value", 0.0) or 0.0))
        steps = max(0, int(tick_index) - last_tick)
        if steps > 0:
            value *= self.recall_fatigue_decay ** steps
        value = _round4(value)
        if value <= 0.0001:
            self._recall_fatigue.pop(clean_id, None)
            return 0.0
        entry["value"] = value
        entry["tick_index"] = int(tick_index)
        return value

    def _recall_fatigue_multiplier(self, memory_id: str, *, tick_index: int) -> float:
        fatigue = self._recall_fatigue_value(memory_id, tick_index=tick_index)
        if fatigue <= 0.0 or self.recall_fatigue_gain <= 0.0:
            return 1.0
        return max(self.recall_fatigue_min_multiplier, 1.0 - fatigue * self.recall_fatigue_gain)

    def _commit_recall_fatigue(self, selected_rows: list[dict[str, Any]], *, tick_index: int) -> None:
        if self.recall_fatigue_accumulate_scale <= 0.0 or self.recall_fatigue_max <= 0.0:
            return
        updated_ids: set[str] = set()
        for row in selected_rows:
            if not isinstance(row, dict):
                continue
            memory_id = str(row.get("memory_id", "") or "")
            if not memory_id:
                continue
            score = max(0.0, float(row.get("raw_score", row.get("score", 0.0)) or 0.0))
            fatigue = self._recall_fatigue_value(memory_id, tick_index=tick_index)
            fatigue += score * self.recall_fatigue_accumulate_scale
            fatigue = min(self.recall_fatigue_max, fatigue)
            self._recall_fatigue[memory_id] = {
                "value": _round4(fatigue),
                "tick_index": int(tick_index),
            }
            updated_ids.add(memory_id)
        stale_ids = [memory_id for memory_id in list(self._recall_fatigue.keys()) if memory_id not in updated_ids]
        for memory_id in stale_ids[:64]:
            self._recall_fatigue_value(memory_id, tick_index=tick_index)

    def _branch_credibility_bias(self, memory_id: str, *, tick_index: int) -> float:
        clean_id = str(memory_id or "")
        if not clean_id:
            return 0.0
        entry = self._branch_credibility.get(clean_id)
        if not isinstance(entry, dict):
            return 0.0
        last_tick = int(entry.get("tick_index", tick_index) or tick_index)
        bias = float(entry.get("bias", 0.0) or 0.0)
        steps = max(0, int(tick_index) - last_tick)
        if steps > 0:
            bias *= self.branch_credibility_decay ** steps
        max_bias = max(0.0, self.branch_credibility_max_multiplier - 1.0)
        min_bias = min(0.0, self.branch_credibility_min_multiplier - 1.0)
        bias = _round4(_clamp(bias, min_bias, max_bias))
        if abs(bias) <= 0.0001:
            self._branch_credibility.pop(clean_id, None)
            return 0.0
        entry["bias"] = bias
        entry["tick_index"] = int(tick_index)
        return bias

    def _branch_credibility_multiplier(self, memory_id: str, *, tick_index: int) -> float:
        bias = self._branch_credibility_bias(memory_id, tick_index=tick_index)
        return _clamp(
            1.0 + bias,
            self.branch_credibility_min_multiplier,
            self.branch_credibility_max_multiplier,
        )

    def update_branch_credibility(
        self,
        *,
        c_i_list: list[dict[str, Any]],
        actual_items: list[dict[str, Any]] | None,
        tick_index: int,
    ) -> dict[str, Any]:
        actual_energy: dict[str, float] = {}
        for item in (actual_items or []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not self._is_commitment_comparable_label(label):
                continue
            actual_energy[label] = float(actual_energy.get(label, 0.0) or 0.0) + max(0.0, float(item.get("energy", 0.0) or 0.0))

        updated_rows: list[dict[str, Any]] = []
        for branch in (c_i_list or []):
            if not isinstance(branch, dict):
                continue
            branch_key = str(branch.get("credibility_key", "") or branch.get("source_bn_id", "") or branch.get("memory_id", "") or "")
            branch_items = [dict(item) for item in (branch.get("items", []) or []) if isinstance(item, dict)]
            if not branch_key or not branch_items:
                continue
            branch_total = 0.0
            matched_mass = 0.0
            missed_mass = 0.0
            for item in branch_items:
                label = str(item.get("sa_label", "") or "")
                if not self._is_commitment_comparable_label(label):
                    continue
                predicted_energy = max(0.0, float(item.get("energy", 0.0) or 0.0))
                if predicted_energy <= 0.0:
                    continue
                branch_total += predicted_energy
                observed = max(0.0, float(actual_energy.get(label, 0.0) or 0.0))
                matched_mass += min(predicted_energy, observed)
                missed_mass += max(0.0, predicted_energy - observed)
            if branch_total <= 0.0:
                continue
            match_ratio = matched_mass / max(0.001, branch_total)
            miss_ratio = missed_mass / max(0.001, branch_total)
            bias = self._branch_credibility_bias(branch_key, tick_index=tick_index)
            bias += self.branch_credibility_match_gain * match_ratio
            bias -= self.branch_credibility_miss_gain * miss_ratio
            max_bias = max(0.0, self.branch_credibility_max_multiplier - 1.0)
            min_bias = min(0.0, self.branch_credibility_min_multiplier - 1.0)
            bias = _round4(_clamp(bias, min_bias, max_bias))
            if abs(bias) <= 0.0001:
                self._branch_credibility.pop(branch_key, None)
            else:
                self._branch_credibility[branch_key] = {
                    "bias": bias,
                    "tick_index": int(tick_index),
                }
            updated_rows.append(
                {
                    "branch_key": branch_key,
                    "match_ratio": _round4(match_ratio),
                    "miss_ratio": _round4(miss_ratio),
                    "credibility_multiplier": _round4(_clamp(1.0 + bias, self.branch_credibility_min_multiplier, self.branch_credibility_max_multiplier)),
                }
            )
        updated_rows.sort(
            key=lambda item: (
                -float(item.get("match_ratio", 0.0) or 0.0),
                float(item.get("miss_ratio", 0.0) or 0.0),
                str(item.get("branch_key", "") or ""),
            )
        )
        return {
            "updated_count": len(updated_rows),
            "updated_preview": updated_rows[:12],
        }

    def write_memory(
        self,
        *,
        tick_index: int,
        memory_kind: str,
        units: list[str],
        items: list[dict[str, Any]],
        source_refs: list[str] | None = None,
        text: str = "",
        reality_weight: float = 1.0,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        memory, vector = self._build_memory_record(
            tick_index=tick_index,
            memory_kind=memory_kind,
            units=units,
            items=items,
            source_refs=source_refs,
            text=text,
            reality_weight=reality_weight,
            meta=meta,
        )
        self._commit_memory_records([(memory, vector)])
        return memory

    def write_memory_batch(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        built_rows: list[tuple[dict[str, Any], np.ndarray]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            memory, vector = self._build_memory_record(
                tick_index=int(row.get("tick_index", 0) or 0),
                memory_kind=str(row.get("memory_kind", "memory") or "memory"),
                units=list(row.get("units", []) or []),
                items=list(row.get("items", []) or []),
                source_refs=list(row.get("source_refs", []) or []),
                text=str(row.get("text", "") or ""),
                reality_weight=float(row.get("reality_weight", 1.0) or 1.0),
                meta=dict(row.get("meta", {}) or {}),
            )
            built_rows.append((memory, vector))
        self._commit_memory_records(built_rows)
        return [memory for memory, _ in built_rows]

    def _build_memory_record(
        self,
        *,
        tick_index: int,
        memory_kind: str,
        units: list[str],
        items: list[dict[str, Any]],
        source_refs: list[str] | None = None,
        text: str = "",
        reality_weight: float = 1.0,
        meta: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], np.ndarray]:
        self._counter += 1
        memory_id = f"mem_{self._counter:06d}"
        clean_units = [str(unit or "") for unit in units if str(unit or "")]
        normalized_items = self._normalize_items(items)
        labels = [str(item.get("sa_label", "") or "") for item in normalized_items if str(item.get("sa_label", "") or "")]
        label_weights: dict[str, float] = {}
        for item in normalized_items:
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            label_weights[label] = _round4(label_weights.get(label, 0.0) + float(item.get("energy", 0.0) or 0.0))
        retrieval_label_weights = self._build_retrieval_label_weights(normalized_items)
        retrieval_labels = sorted(retrieval_label_weights.keys())

        modalities = self._infer_modalities(normalized_items)
        spacetime = self._infer_spacetime(tick_index=tick_index, units=clean_units, items=normalized_items)
        vector, vector_tokens = self._embedder.build_memory_vector(
            units=clean_units,
            items=normalized_items,
            retrieval_label_weights=retrieval_label_weights,
            text=str(text or ""),
            modalities=modalities,
            spacetime=spacetime,
        )
        unit_bigrams = self._build_unit_bigrams(clean_units)

        memory = {
            "memory_id": memory_id,
            "tick_index": int(tick_index),
            "memory_kind": str(memory_kind or "memory"),
            "text": str(text or ""),
            "units": clean_units,
            "unit_bigrams": unit_bigrams,
            "sa_labels": labels,
            "label_weights": label_weights,
            "retrieval_labels": retrieval_labels,
            "retrieval_label_weights": retrieval_label_weights,
            "items": normalized_items,
            "source_refs": list(source_refs or []),
            "reality_weight": _round4(reality_weight),
            "meta": dict(meta or {}),
            "modalities": modalities,
            "spacetime": spacetime,
            "vector_tokens": vector_tokens,
            "total_item_energy": _round4(sum(float(item.get("energy", 0.0) or 0.0) for item in normalized_items)),
        }
        return memory, vector

    def _commit_memory_records(self, built_rows: list[tuple[dict[str, Any], np.ndarray]]) -> None:
        committed = False
        vector_rows: list[tuple[str, np.ndarray]] = []
        for memory, vector in built_rows:
            if not isinstance(memory, dict):
                continue
            memory_id = str(memory.get("memory_id", "") or "")
            if not memory_id:
                continue
            self._memories.append(memory)
            self._memories_by_id[memory_id] = memory
            self._recent_ids.append(memory_id)
            self._recent_window_ids.append(memory_id)
            self._index_memory(memory=memory, vector=vector, defer_vector_add=True)
            vector_rows.append((memory_id, vector))
            self._memory_profile_cache[memory_id] = self._build_memory_profile(memory)
            committed = True
        if vector_rows:
            self._vector_index.add_batch(vector_rows)
        if committed:
            self._touch_memory_revision(clear_pair_relation_cache=False)

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        return self._memories_by_id.get(str(memory_id or ""))

    def recall_bn(
        self,
        *,
        query_labels: list[str],
        query_weights: dict[str, float],
        top_k: int,
        tick_index: int,
        query_items: list[dict[str, Any]] | None = None,
        query_units: list[str] | None = None,
        recent_focus_units: list[str] | None = None,
        successor_bias_gain: float = 1.0,
        query_spacetime: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        expanded_query_weights = self._expand_query_label_weights(
            query_labels=query_labels,
            query_weights=query_weights,
            query_items=query_items,
        )
        query_set = {str(label or "") for label in expanded_query_weights.keys() if str(label or "")}
        query_units = [str(unit or "") for unit in (query_units or []) if str(unit or "")]
        query_bigrams = set(self._build_unit_bigrams(query_units))
        recent_focus_units = [str(unit or "") for unit in (recent_focus_units or []) if str(unit or "")]
        if not query_set and not query_units:
            return []

        query_signature = self._build_query_signature(
            query_labels=list(query_set),
            query_weights=expanded_query_weights,
            query_items=query_items,
            query_units=query_units,
            recent_focus_units=recent_focus_units,
            query_spacetime=query_spacetime,
        )
        query_vector, query_vector_tokens = self._get_or_build_query_vector(
            query_signature=query_signature,
            query_labels=list(query_set),
            query_weights=expanded_query_weights,
            query_items=query_items,
            query_units=query_units,
            recent_focus_units=recent_focus_units,
            query_spacetime=query_spacetime,
        )
        candidate_state = self._get_or_build_candidate_state(
            query_signature=query_signature,
            query_set=query_set,
            query_units=query_units,
            query_bigrams=query_bigrams,
            query_vector=query_vector,
        )
        ann_by_id = dict(candidate_state["ann_by_id"])
        candidate_ids = set(candidate_state["candidate_ids"])

        candidates: list[dict[str, Any]] = []
        query_channels = {channel for channel in (self._label_channel(label) for label in query_set) if channel}
        query_mass = sum(float(expanded_query_weights.get(label, 0.0) or 0.0) for label in query_set) or 1.0
        recent_focus_tail = recent_focus_units[-4:]
        query_contour_rows = self._extract_contour_rows(query_items)
        query_unit_set = set(query_units)
        query_units_top12 = tuple(query_units[:12])
        query_bigrams_top12 = tuple(sorted(query_bigrams)[:12])
        query_channel_base = len(query_channels)

        for memory_id in candidate_ids:
            memory = self.get_memory(memory_id)
            if not memory:
                continue
            profile = self._memory_profile(memory_id, memory=memory)
            mem_set = profile["label_set"]
            overlap = query_set & mem_set
            label_hit_ratio = len(overlap) / max(1, len(query_set)) if query_set else 0.0
            overlap_score = sum(
                min(float(expanded_query_weights.get(label, 0.0) or 0.0), float(profile["label_weights"].get(label, 0.0) or 0.0))
                for label in overlap
            )
            mem_mass = float(profile["label_mass"] or 1.0)
            weight_overlap = overlap_score / max(query_mass, mem_mass)
            jaccard = len(overlap) / max(1, len(query_set | mem_set)) if query_set else 0.0

            mem_unit_set = profile["unit_set"]
            unit_overlap = len(query_unit_set & mem_unit_set) / max(1, len(query_unit_set)) if query_unit_set else 0.0

            mem_bigrams = profile["bigram_set"]
            bigram_overlap = len(query_bigrams & mem_bigrams) / max(1, len(query_bigrams)) if query_bigrams else 0.0

            vector_row = ann_by_id.get(memory_id, {})
            vector_score = float(vector_row.get("vector_score", 0.0) or 0.0)
            if not vector_row:
                mem_vec = self._vector_index.get_vector(memory_id)
                if mem_vec is not None:
                    vector_score = self._embedder.cosine(query_vector, mem_vec)
            vector_similarity = max(0.0, (vector_score + 1.0) / 2.0)

            memory_channels = profile["modalities_set"]
            channel_overlap = (
                len(query_channels & memory_channels) / max(1, query_channel_base + len(memory_channels - query_channels))
                if query_channels
                else 0.0
            )
            contour_similarity = self._visual_contour_similarity(query_contour_rows, profile.get("contour_rows", []))

            recency_gap = max(0, tick_index - int(profile["tick_index"]))
            recency_bonus = 1.0 / (1.0 + float(recency_gap))
            recent_window_bonus = 1.0 if memory_id in self._recent_window_ids else 0.0
            successor_bias = self._compute_successor_bias(
                memory,
                recent_focus_tail,
                gain=successor_bias_gain,
                unit_positions=profile.get("unit_positions", {}),
                unit_count=int(profile.get("unit_count", 0) or 0),
            )
            reality_bonus = float(profile["reality_bonus"])
            time_intent_bonus = self._time_intent_match(memory, tick_index=tick_index, query_spacetime=query_spacetime)
            motion_intent_bonus = self._motion_intent_match(memory, query_spacetime=query_spacetime)
            rhythm_intent_bonus = self._rhythm_intent_match(memory, tick_index=tick_index, query_spacetime=query_spacetime)
            hearing_intent_bonus = self._hearing_intent_match(memory, query_spacetime=query_spacetime)
            feedback_intent_bonus = self._feedback_intent_match(memory, query_spacetime=query_spacetime)
            raw_score = (
                0.23 * vector_similarity
                + 0.18 * weight_overlap
                + 0.11 * unit_overlap
                + 0.10 * bigram_overlap
                + 0.07 * jaccard
                + 0.06 * label_hit_ratio
                + 0.07 * channel_overlap
                + 0.12 * contour_similarity
                + 0.08 * recency_bonus
                + 0.06 * recent_window_bonus
                + 0.05 * successor_bias
                + 0.11 * reality_bonus
                + 0.08 * time_intent_bonus
                + 0.05 * motion_intent_bonus
                + 0.05 * rhythm_intent_bonus
                + 0.05 * hearing_intent_bonus
                + 0.04 * feedback_intent_bonus
            )
            recall_fatigue_multiplier = self._recall_fatigue_multiplier(memory_id, tick_index=tick_index)
            score = raw_score * recall_fatigue_multiplier
            candidates.append(
                {
                    "memory_id": memory["memory_id"],
                    "raw_score": _round4(raw_score),
                    "score": _round4(score),
                    "memory_kind": memory["memory_kind"],
                    "tick_index": memory["tick_index"],
                    "text": memory.get("text", ""),
                    "_memory_modalities": memory.get("modalities", []),
                    "_overlap_labels": overlap,
                    "_vector_row": vector_row,
                    "_score_breakdown": {
                        "vector_similarity": _round4(vector_similarity),
                        "weight_overlap": _round4(weight_overlap),
                        "unit_overlap": _round4(unit_overlap),
                        "bigram_overlap": _round4(bigram_overlap),
                        "jaccard": _round4(jaccard),
                        "label_hit_ratio": _round4(label_hit_ratio),
                        "channel_overlap": _round4(channel_overlap),
                        "contour_similarity": _round4(contour_similarity),
                        "recency_bonus": _round4(recency_bonus),
                        "recent_window_bonus": _round4(recent_window_bonus),
                        "successor_bias": _round4(successor_bias),
                        "reality_bonus": _round4(reality_bonus),
                        "time_intent_bonus": _round4(time_intent_bonus),
                        "motion_intent_bonus": _round4(motion_intent_bonus),
                        "rhythm_intent_bonus": _round4(rhythm_intent_bonus),
                        "hearing_intent_bonus": _round4(hearing_intent_bonus),
                        "feedback_intent_bonus": _round4(feedback_intent_bonus),
                        "recall_fatigue_multiplier": _round4(recall_fatigue_multiplier),
                    },
                }
            )

        candidates.sort(
            key=lambda item: (
                -float(item.get("score", 0.0) or 0.0),
                -int(-1 if item.get("tick_index", -1) is None else item.get("tick_index", -1)),
                item["memory_id"],
            )
        )
        selected = candidates[: max(1, int(top_k))]
        finalized: list[dict[str, Any]] = []
        vector_engine_name = self._vector_index.engine_name()
        query_vector_tokens_preview = list(query_vector_tokens)[:8]
        for item in selected:
            memory_id = str(item.get("memory_id", "") or "")
            vector_row = dict(item.pop("_vector_row", {}) or {})
            overlap = sorted(str(label or "") for label in (item.pop("_overlap_labels", set()) or set()) if str(label or ""))
            score_breakdown = dict(item.pop("_score_breakdown", {}) or {})
            item["memory_modalities"] = list(item.pop("_memory_modalities", []) or [])[:8]
            item["overlap_labels"] = overlap
            item["candidate_sources"] = self._candidate_sources_for_memory(
                memory_id,
                query_set=query_set,
                query_units=query_units_top12,
                query_bigrams=query_bigrams_top12,
                ann_by_id=ann_by_id,
            )
            memory = self.get_memory(memory_id) or {}
            item["vector_tokens"] = list(memory.get("vector_tokens", []) or [])[:8]
            item["score_breakdown"] = score_breakdown
            item["query_vector_tokens"] = list(query_vector_tokens_preview)
            item["vector_engine"] = str(vector_row.get("engine", vector_engine_name) or vector_engine_name)
            finalized.append(item)
        self._commit_recall_fatigue(finalized, tick_index=tick_index)
        return finalized

    def build_prediction_branches(
        self,
        *,
        bn_list: list[dict[str, Any]],
        tick_index: int,
        recent_focus_units: list[str],
        max_neighbors: int,
        successor_bias_gain: float = 1.0,
        latent_candidates: list[dict[str, Any]] | None = None,
        latent_total_virtual_energy: float = 0.0,
        latent_item_ratio_prune_threshold: float = 0.002,
        latent_branch_item_limit: int = 96,
        c_star_item_ratio_prune_threshold: float = 0.005,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        c_i_list: list[dict[str, Any]] = []
        aggregate_weights: dict[str, float] = {}
        aggregate_sources: dict[str, list[str]] = defaultdict(list)
        aggregate_support: dict[str, dict[str, Any]] = {}
        recent_focus_units = [str(item or "") for item in recent_focus_units if str(item or "")]
        recent_focus_tail = recent_focus_units[-4:] if recent_focus_units else []
        bn_rank_by_id = {
            str(row.get("memory_id", "") or ""): index
            for index, row in enumerate(bn_list or [])
            if str(row.get("memory_id", "") or "")
        }
        bn_score_by_id = {
            str(row.get("memory_id", "") or ""): float(row.get("score", 0.0) or 0.0)
            for row in (bn_list or [])
            if str(row.get("memory_id", "") or "")
        }

        for bn in bn_list:
            memory = self.get_memory(str(bn.get("memory_id", "") or ""))
            if not memory:
                continue
            profile = self._memory_profile(str(memory.get("memory_id", "") or ""), memory=memory)
            primary_units = list(profile["units"])
            primary_label_set = profile["label_set"]
            candidate_rows = self._get_or_build_neighbor_rows(memory, limit=max(4, max_neighbors * 4))
            neighbors: list[dict[str, Any]] = []

            for row in candidate_rows:
                other = self.get_memory(str(row.get("memory_id", "") or ""))
                if not other:
                    continue
                other_id = str(other.get("memory_id", "") or "")
                other_profile = self._memory_profile(other_id, memory=other)
                overlap = primary_label_set & other_profile["label_set"]
                overlap_bonus = len(overlap) / max(1, len(other_profile["label_set"]))
                unit_overlap = len(set(primary_units) & other_profile["unit_set"]) / max(1, len(set(primary_units)))
                successor_bias = self._compute_successor_bias(
                    other,
                    recent_focus_tail,
                    gain=successor_bias_gain,
                    unit_positions=other_profile.get("unit_positions", {}),
                    unit_count=int(other_profile.get("unit_count", 0) or 0),
                )
                reality_bonus = float(other_profile["reality_bonus"])
                vector_related = self._pair_vector_related(memory["memory_id"], other_id)
                final_score = (
                    0.31 * float(row.get("spacetime_score", 0.0) or 0.0)
                    + 0.20 * overlap_bonus
                    + 0.18 * unit_overlap
                    + 0.15 * successor_bias
                    + 0.10 * vector_related
                    + 0.10 * reality_bonus
                )
                neighbors.append(
                    {
                        "memory_id": other["memory_id"],
                        "memory_kind": other.get("memory_kind", ""),
                        "text": other.get("text", ""),
                        "tick_index": other.get("tick_index", -1),
                        "neighbor_score": _round4(final_score),
                        "successor_bias": _round4(successor_bias),
                        "distance": int(row.get("distance_time", 0) or 0),
                        "items": other.get("items", []),
                        "score_breakdown": {
                            "spacetime_score": _round4(float(row.get("spacetime_score", 0.0) or 0.0)),
                            "overlap_bonus": _round4(overlap_bonus),
                            "unit_overlap": _round4(unit_overlap),
                            "successor_bias": _round4(successor_bias),
                            "vector_related": _round4(vector_related),
                            "reality_bonus": _round4(reality_bonus),
                            "temporal_bonus": _round4(float(row.get("temporal_bonus", 0.0) or 0.0)),
                            "space_bonus": _round4(float(row.get("space_bonus", 0.0) or 0.0)),
                        },
                    }
                )
            neighbors.sort(key=lambda item: (-float(item.get("neighbor_score", 0.0) or 0.0), int(item.get("distance", 0) or 0), item["memory_id"]))
            selected_neighbors = neighbors[: max(1, int(max_neighbors))]

            virtual_energy = float(bn.get("score", 0.0) or 0.0)
            branch_key = str(memory.get("memory_id", "") or "")
            branch_credibility = self._branch_credibility_multiplier(branch_key, tick_index=tick_index)
            credibility_bias = self._branch_credibility_bias(branch_key, tick_index=tick_index)
            bundle_id = f"c_local::{memory['memory_id']}"
            bundle_items: list[dict[str, Any]] = []
            source_bn_text = str(memory.get("text", "") or "").strip()
            source_bn_rank = int(bn_rank_by_id.get(str(memory.get("memory_id", "") or ""), 0))
            current_bn_score = float(bn_score_by_id.get(str(memory.get("memory_id", "") or ""), virtual_energy) or virtual_energy)

            branch_sources: list[dict[str, Any]] = [
                {
                    "memory_id": str(memory.get("memory_id", "") or ""),
                    "memory_kind": str(memory.get("memory_kind", "") or ""),
                    "text": str(memory.get("text", "") or ""),
                    "tick_index": int(memory.get("tick_index", -1) or -1),
                    "neighbor_score": _round4(1.0 + current_bn_score * 0.22),
                    "successor_bias": _round4(
                        self._compute_successor_bias(
                            memory,
                            recent_focus_tail,
                            gain=successor_bias_gain,
                            unit_positions=profile.get("unit_positions", {}),
                            unit_count=int(profile.get("unit_count", 0) or 0),
                        )
                    ),
                    "distance": 0,
                    "items": memory.get("items", []),
                    "score_breakdown": {
                        "self_source": 1.0,
                        "bn_score": _round4(current_bn_score),
                    },
                }
            ]
            branch_sources.extend(selected_neighbors)
            for source_index, nb in enumerate(branch_sources):
                source_factor = max(0.02, float(nb.get("neighbor_score", 0.0) or 0.0))
                source_memory_kind = str(nb.get("memory_kind", "") or "")
                distance = max(0, int(nb.get("distance", 0) or 0))
                is_self_source = source_index == 0 and str(nb.get("memory_id", "") or "") == str(memory.get("memory_id", "") or "")
                for item in nb.get("items", []):
                    label = str(item.get("sa_label", "") or "")
                    display_text = str(item.get("display_text", "") or _display_from_label(label))
                    if not label:
                        continue
                    item_energy = max(0.08, float(item.get("energy", 0.0) or 0.0))
                    base_weight = virtual_energy * source_factor * item_energy
                    weight = base_weight * branch_credibility
                    if is_self_source:
                        weight *= 1.34
                    else:
                        weight *= max(0.32, 1.0 - 0.08 * float(distance))
                    if source_memory_kind == "focus_chain":
                        weight *= 0.78 if not is_self_source else 1.0
                    elif source_memory_kind == "exact_external":
                        weight *= 1.08
                    elif source_memory_kind == "latent_state_snapshot":
                        weight *= 0.72
                    if label.startswith("text::"):
                        text_value = str(label.split("::", 1)[1] or "")
                        same_text = bool(source_bn_text and text_value == source_bn_text)
                        rank_bonus = max(0.35, 1.0 - 0.14 * float(source_bn_rank))
                        text_gain = 0.58 if same_text else -0.12
                        source_memory_text = str(nb.get("text", "") or "").strip()
                        source_text_match = bool(source_memory_text and text_value == source_memory_text)
                        source_text_gain = 0.22 if source_text_match else 0.0
                        weight = weight * max(0.18, 0.72 + rank_bonus * 0.28 + text_gain + source_text_gain)
                    bundle_items.append({"sa_label": label, "display_text": display_text, "energy": _round4(weight)})
                    aggregate_weights[label] = _round4(aggregate_weights.get(label, 0.0) + weight)
                    if bundle_id not in aggregate_sources[label]:
                        aggregate_sources[label].append(bundle_id)
                    support = aggregate_support.setdefault(
                        label,
                        {
                            "branch_count": 0,
                            "top_branch_support": 0.0,
                            "same_text_support": 0.0,
                            "weighted_support": 0.0,
                            "real_support": 0.0,
                            "top_source_support": 0.0,
                            "credibility_weighted_support": 0.0,
                            "max_branch_weight": 0.0,
                            "max_branch_key": "",
                            "branch_keys": [],
                            "source_bn_texts": [],
                        },
                    )
                    support["branch_count"] = int(support.get("branch_count", 0) or 0) + 1
                    support["weighted_support"] = float(support.get("weighted_support", 0.0) or 0.0) + float(weight)
                    support["credibility_weighted_support"] = float(support.get("credibility_weighted_support", 0.0) or 0.0) + float(weight * branch_credibility)
                    if source_bn_rank == 0:
                        support["top_branch_support"] = float(support.get("top_branch_support", 0.0) or 0.0) + float(weight)
                    if is_self_source:
                        support["top_source_support"] = float(support.get("top_source_support", 0.0) or 0.0) + float(weight)
                    if float(memory.get("reality_weight", 0.0) or 0.0) >= 0.75 or source_memory_kind == "exact_external":
                        support["real_support"] = float(support.get("real_support", 0.0) or 0.0) + float(weight)
                    if float(weight) > float(support.get("max_branch_weight", 0.0) or 0.0):
                        support["max_branch_weight"] = float(weight)
                        support["max_branch_key"] = branch_key
                    if branch_key and branch_key not in list(support.get("branch_keys", []) or []):
                        branch_keys = list(support.get("branch_keys", []) or [])
                        branch_keys.append(branch_key)
                        support["branch_keys"] = branch_keys[:6]
                    if label.startswith("text::"):
                        text_value = str(label.split("::", 1)[1] or "")
                        if source_bn_text and text_value == source_bn_text:
                            support["same_text_support"] = float(support.get("same_text_support", 0.0) or 0.0) + float(weight)
                    if source_bn_text and source_bn_text not in list(support.get("source_bn_texts", []) or []):
                        texts = list(support.get("source_bn_texts", []) or [])
                        texts.append(source_bn_text)
                        support["source_bn_texts"] = texts[:4]

            c_i_list.append(
                {
                    "bundle_id": bundle_id,
                    "source_bn_id": memory["memory_id"],
                    "virtual_energy": _round4(virtual_energy),
                    "credibility_key": branch_key,
                    "credibility_multiplier": _round4(branch_credibility),
                    "credibility_bias": _round4(credibility_bias),
                    "neighbors": selected_neighbors,
                    "items": bundle_items[:48],
                }
            )

        latent_rows = [dict(row) for row in (latent_candidates or []) if isinstance(row, dict)]
        latent_virtual_budget = max(0.0, float(latent_total_virtual_energy or 0.0))
        if latent_rows and latent_virtual_budget > 0.0:
            latent_mass = sum(max(0.0, float(row.get("score", 0.0) or 0.0)) for row in latent_rows)
            for latent_index, row in enumerate(latent_rows):
                memory = self.get_memory(str(row.get("memory_id", "") or ""))
                if not memory:
                    continue
                row_score = max(0.0, float(row.get("score", 0.0) or 0.0))
                if row_score <= 0.0:
                    continue
                branch_virtual_energy = latent_virtual_budget * (row_score / max(0.001, latent_mass))
                branch_items = [dict(item) for item in (memory.get("items", []) or []) if isinstance(item, dict)]
                branch_total_energy = sum(max(0.0, float(item.get("energy", 0.0) or 0.0)) for item in branch_items)
                if branch_total_energy <= 0.0:
                    continue
                branch_bundle_id = f"c_latent::{memory['memory_id']}"
                kept_count = 0
                for item in sorted(branch_items, key=lambda current: (-float(current.get("energy", 0.0) or 0.0), str(current.get("sa_label", "") or ""))):
                    label = str(item.get("sa_label", "") or "")
                    display_text = str(item.get("display_text", "") or _display_from_label(label))
                    if not label:
                        continue
                    item_energy = max(0.0, float(item.get("energy", 0.0) or 0.0))
                    share = item_energy / max(0.001, branch_total_energy)
                    if share < max(0.0, float(latent_item_ratio_prune_threshold)):
                        continue
                    weight = branch_virtual_energy * share
                    if weight <= 0.0:
                        continue
                    aggregate_weights[label] = _round4(aggregate_weights.get(label, 0.0) + weight)
                    if branch_bundle_id not in aggregate_sources[label]:
                        aggregate_sources[label].append(branch_bundle_id)
                    kept_count += 1
                    if kept_count >= max(8, int(latent_branch_item_limit)):
                        break
                c_i_list.append(
                    {
                        "bundle_id": branch_bundle_id,
                        "source_bn_id": "",
                        "memory_id": memory["memory_id"],
                        "memory_kind": str(memory.get("memory_kind", "") or ""),
                        "virtual_energy": _round4(branch_virtual_energy),
                        "neighbors": [],
                        "items": [],
                        "is_latent_projection": True,
                        "latent_rank": int(latent_index),
                    }
                )

        for label, weight in list(aggregate_weights.items()):
            if not str(label or "").startswith("text::"):
                continue
            support = dict(aggregate_support.get(label, {}) or {})
            weighted_support = float(support.get("weighted_support", weight) or weight)
            top_branch_support = float(support.get("top_branch_support", 0.0) or 0.0)
            same_text_support = float(support.get("same_text_support", 0.0) or 0.0)
            branch_count = int(support.get("branch_count", 0) or 0)
            support_ratio = same_text_support / max(0.001, weighted_support)
            top_ratio = top_branch_support / max(0.001, weighted_support)
            branch_bonus = min(0.18, 0.045 * max(0, branch_count - 1))
            multiplier = max(0.18, 0.58 + support_ratio * 0.82 + top_ratio * 0.34 + branch_bonus)
            aggregate_weights[label] = _round4(weight * multiplier)

        if aggregate_weights:
            total_weight = sum(max(0.0, float(value or 0.0)) for value in aggregate_weights.values())
            if total_weight > 0.0 and c_star_item_ratio_prune_threshold > 0.0:
                pruned = {
                    label: value
                    for label, value in aggregate_weights.items()
                    if (float(value or 0.0) / total_weight) >= float(c_star_item_ratio_prune_threshold)
                }
                if pruned:
                    aggregate_weights = pruned

        total_weight = sum(max(0.0, float(value or 0.0)) for value in aggregate_weights.values())
        peak_weight = max((max(0.0, float(value or 0.0)) for value in aggregate_weights.values()), default=0.0)

        c_star_items = [
            {
                "sa_label": label,
                "display_text": _display_from_label(label),
                "energy": _round4(energy),
                "sources": aggregate_sources.get(label, [])[:4],
                "support": (
                    {
                        "branch_count": int((aggregate_support.get(label, {}) or {}).get("branch_count", 0) or 0),
                        "top_branch_support": _round4(float((aggregate_support.get(label, {}) or {}).get("top_branch_support", 0.0) or 0.0)),
                        "same_text_support": _round4(float((aggregate_support.get(label, {}) or {}).get("same_text_support", 0.0) or 0.0)),
                        "weighted_support": _round4(float((aggregate_support.get(label, {}) or {}).get("weighted_support", 0.0) or 0.0)),
                        "real_support": _round4(float((aggregate_support.get(label, {}) or {}).get("real_support", 0.0) or 0.0)),
                        "top_source_support": _round4(float((aggregate_support.get(label, {}) or {}).get("top_source_support", 0.0) or 0.0)),
                        "credibility_weighted_support": _round4(float((aggregate_support.get(label, {}) or {}).get("credibility_weighted_support", 0.0) or 0.0)),
                        "max_branch_weight": _round4(float((aggregate_support.get(label, {}) or {}).get("max_branch_weight", 0.0) or 0.0)),
                        "max_branch_key": str((aggregate_support.get(label, {}) or {}).get("max_branch_key", "") or ""),
                        "branch_keys": list((aggregate_support.get(label, {}) or {}).get("branch_keys", []) or [])[:6],
                        "source_bn_texts": list((aggregate_support.get(label, {}) or {}).get("source_bn_texts", []) or [])[:4],
                    }
                    if (aggregate_support.get(label, {}) or {})
                    else {}
                ),
                "commitment": _round4(
                    self._compute_commitment(
                        label=label,
                        energy=float(energy or 0.0),
                        total_weight=float(total_weight or 0.0),
                        peak_weight=float(peak_weight or 0.0),
                        support=dict(aggregate_support.get(label, {}) or {}),
                        tick_index=tick_index,
                    )
                ),
            }
            for label, energy in sorted(aggregate_weights.items(), key=lambda item: (-item[1], item[0]))
        ]
        for item in c_star_items:
            commitment = float(item.get("commitment", 0.0) or 0.0)
            item["prediction_role"] = "core" if commitment >= 0.58 else ("supported" if commitment >= 0.32 else "halo")
        c_star = {
            "bundle_id": "c_star::current",
            "virtual_energy": _round4(sum(float(item.get("energy", 0.0) or 0.0) for item in c_star_items)),
            "items": c_star_items[:48],
            "summary": {
                "source_branch_count": len(c_i_list),
                "aggregated_sa_count": len(c_star_items),
                "core_prediction_count": sum(1 for item in c_star_items if str(item.get("prediction_role", "") or "") == "core"),
                "halo_prediction_count": sum(1 for item in c_star_items if str(item.get("prediction_role", "") or "") == "halo"),
                "text_support_preview": [
                    {
                        "label": str(item.get("sa_label", "") or ""),
                        "energy": _round4(float(item.get("energy", 0.0) or 0.0)),
                        "commitment": _round4(float(item.get("commitment", 0.0) or 0.0)),
                        "prediction_role": str(item.get("prediction_role", "") or ""),
                        "support": dict(item.get("support", {}) or {}),
                    }
                    for item in c_star_items
                ][:6],
            },
        }
        return c_i_list, c_star

    def export_payload(self) -> dict[str, Any]:
        return {
            "memory_count": len(self._memories),
            "counter": self._counter,
            "recent_ids": list(self._recent_ids),
            "recent_window_ids": list(self._recent_window_ids),
            "memories": list(self._memories),
            "recall_fatigue_config": {
                "decay": _round4(self.recall_fatigue_decay),
                "gain": _round4(self.recall_fatigue_gain),
                "accumulate_scale": _round4(self.recall_fatigue_accumulate_scale),
                "max": _round4(self.recall_fatigue_max),
                "min_multiplier": _round4(self.recall_fatigue_min_multiplier),
            },
            "recall_fatigue_state": {
                str(memory_id): {
                    "value": _round4(float((entry or {}).get("value", 0.0) or 0.0)),
                    "tick_index": int((entry or {}).get("tick_index", 0) or 0),
                }
                for memory_id, entry in sorted(self._recall_fatigue.items(), key=lambda row: str(row[0]))
                if str(memory_id or "")
            },
            "branch_credibility_config": {
                "decay": _round4(self.branch_credibility_decay),
                "match_gain": _round4(self.branch_credibility_match_gain),
                "miss_gain": _round4(self.branch_credibility_miss_gain),
                "min_multiplier": _round4(self.branch_credibility_min_multiplier),
                "max_multiplier": _round4(self.branch_credibility_max_multiplier),
            },
            "branch_credibility_state": {
                str(memory_id): {
                    "bias": _round4(float((entry or {}).get("bias", 0.0) or 0.0)),
                    "tick_index": int((entry or {}).get("tick_index", 0) or 0),
                }
                for memory_id, entry in sorted(self._branch_credibility.items(), key=lambda row: str(row[0]))
                if str(memory_id or "")
            },
            "index_summary": self.index_summary(),
            "vector_index": self._vector_index.export_payload(),
            "spacetime_index": self._spacetime_index.export_payload(),
        }

    def import_payload(self, payload: dict[str, Any]) -> None:
        self._memories = list(payload.get("memories", []) or [])
        self._counter = int(payload.get("counter", len(self._memories)) or len(self._memories))
        self._recent_ids = deque(list(payload.get("recent_ids", []) or []), maxlen=self.max_recent)
        self._recent_window_ids = deque(list(payload.get("recent_window_ids", []) or []), maxlen=self.recent_window_size)
        recall_fatigue_config = dict(payload.get("recall_fatigue_config", {}) or {})
        self.recall_fatigue_decay = max(0.0, min(1.0, float(recall_fatigue_config.get("decay", self.recall_fatigue_decay) or self.recall_fatigue_decay)))
        self.recall_fatigue_gain = max(0.0, float(recall_fatigue_config.get("gain", self.recall_fatigue_gain) or self.recall_fatigue_gain))
        self.recall_fatigue_accumulate_scale = max(
            0.0,
            float(recall_fatigue_config.get("accumulate_scale", self.recall_fatigue_accumulate_scale) or self.recall_fatigue_accumulate_scale),
        )
        self.recall_fatigue_max = max(0.0, float(recall_fatigue_config.get("max", self.recall_fatigue_max) or self.recall_fatigue_max))
        self.recall_fatigue_min_multiplier = max(
            0.0,
            min(1.0, float(recall_fatigue_config.get("min_multiplier", self.recall_fatigue_min_multiplier) or self.recall_fatigue_min_multiplier)),
        )
        recall_fatigue_state = dict(payload.get("recall_fatigue_state", {}) or {})
        self._recall_fatigue = {
            str(memory_id): {
                "value": _round4(max(0.0, float((entry or {}).get("value", 0.0) or 0.0))),
                "tick_index": int((entry or {}).get("tick_index", 0) or 0),
            }
            for memory_id, entry in recall_fatigue_state.items()
            if str(memory_id or "") and isinstance(entry, dict)
        }
        branch_credibility_config = dict(payload.get("branch_credibility_config", {}) or {})
        self.branch_credibility_decay = max(0.0, min(1.0, float(branch_credibility_config.get("decay", self.branch_credibility_decay) or self.branch_credibility_decay)))
        self.branch_credibility_match_gain = max(0.0, float(branch_credibility_config.get("match_gain", self.branch_credibility_match_gain) or self.branch_credibility_match_gain))
        self.branch_credibility_miss_gain = max(0.0, float(branch_credibility_config.get("miss_gain", self.branch_credibility_miss_gain) or self.branch_credibility_miss_gain))
        self.branch_credibility_min_multiplier = max(
            0.0,
            min(1.0, float(branch_credibility_config.get("min_multiplier", self.branch_credibility_min_multiplier) or self.branch_credibility_min_multiplier)),
        )
        self.branch_credibility_max_multiplier = max(
            self.branch_credibility_min_multiplier,
            float(branch_credibility_config.get("max_multiplier", self.branch_credibility_max_multiplier) or self.branch_credibility_max_multiplier),
        )
        branch_credibility_state = dict(payload.get("branch_credibility_state", {}) or {})
        self._branch_credibility = {
            str(memory_id): {
                "bias": _round4(float((entry or {}).get("bias", 0.0) or 0.0)),
                "tick_index": int((entry or {}).get("tick_index", 0) or 0),
            }
            for memory_id, entry in branch_credibility_state.items()
            if str(memory_id or "") and isinstance(entry, dict)
        }
        vector_payload = payload.get("vector_index")
        spacetime_payload = payload.get("spacetime_index")
        if isinstance(vector_payload, dict) and isinstance(spacetime_payload, dict):
            self._memories_by_id = {str(memory.get("memory_id", "") or ""): memory for memory in self._memories if str(memory.get("memory_id", "") or "")}
            self._posting_by_label = defaultdict(set)
            self._posting_by_unit = defaultdict(set)
            self._posting_by_bigram = defaultdict(set)
            for memory in self._memories:
                memory_id = str(memory.get("memory_id", "") or "")
                if not memory_id:
                    continue
                retrieval_labels = memory.get("retrieval_labels", memory.get("sa_labels", [])) or []
                for label in set(str(item or "") for item in retrieval_labels if str(item or "")):
                    self._posting_by_label[label].add(memory_id)
                for unit in set(str(item or "") for item in (memory.get("units", []) or []) if str(item or "")):
                    self._posting_by_unit[unit].add(memory_id)
                for bigram in set(str(item or "") for item in (memory.get("unit_bigrams", []) or []) if str(item or "")):
                    self._posting_by_bigram[bigram].add(memory_id)
            self._vector_index.import_payload(vector_payload)
            self._spacetime_index.import_payload(spacetime_payload)
            self.vector_backend = str(getattr(self._vector_index, "backend", self.vector_backend) or self.vector_backend)
            self.spacetime_backend = str(getattr(self._spacetime_index, "backend", self.spacetime_backend) or self.spacetime_backend)
            self._rebuild_memory_profiles()
            self._filter_recall_fatigue_to_live_memories()
            self._filter_branch_credibility_to_live_memories()
            self._touch_memory_revision(clear_pair_relation_cache=True)
        else:
            self._rebuild_indexes()

    def forget_cold_memories(
        self,
        *,
        keep_latest: int,
        min_reality_weight: float = 0.0,
        min_total_item_energy: float = 0.0,
        protect_memory_kinds: list[str] | None = None,
        max_memory_count: int | None = None,
        strategy: str = "latest_only",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        keep_latest = max(0, int(keep_latest))
        min_reality_weight = max(0.0, float(min_reality_weight or 0.0))
        min_total_item_energy = max(0.0, float(min_total_item_energy or 0.0))
        protected_kinds = {str(item or "").strip() for item in (protect_memory_kinds or []) if str(item or "").strip()}
        clean_strategy = str(strategy or "latest_only").strip().lower()
        if clean_strategy not in {"latest_only", "score_prune"}:
            clean_strategy = "latest_only"
        limit_override = None if max_memory_count is None else max(0, int(max_memory_count))

        before_count = len(self._memories)
        latest_group = list(self._memories[-keep_latest:]) if keep_latest > 0 else []
        older_group = list(self._memories[:-keep_latest]) if keep_latest > 0 else list(self._memories)
        protected_count = 0

        if clean_strategy == "latest_only":
            if before_count <= keep_latest:
                survivors = list(self._memories)
            elif min_reality_weight > 0.0 and keep_latest > 0:
                stable = [m for m in older_group if float(m.get("reality_weight", 0.0) or 0.0) >= min_reality_weight]
                survivors = stable + latest_group
            else:
                survivors = latest_group
            protected_count = max(0, len(survivors) - len(latest_group))
        else:
            latest_ids = {str(m.get("memory_id", "") or "") for m in latest_group if str(m.get("memory_id", "") or "")}
            scored_rows: list[tuple[bool, float, int, str, dict[str, Any]]] = []
            for index, memory in enumerate(older_group):
                score = self._forget_retention_score(
                    memory,
                    older_index=index,
                    older_total=len(older_group),
                    protected_kinds=protected_kinds,
                    min_reality_weight=min_reality_weight,
                    min_total_item_energy=min_total_item_energy,
                )
                pinned = self._forget_is_pinned(
                    memory,
                    protected_kinds=protected_kinds,
                    min_reality_weight=min_reality_weight,
                    min_total_item_energy=min_total_item_energy,
                )
                scored_rows.append(
                    (
                        pinned,
                        score,
                        int(memory.get("tick_index", -1) or -1),
                        str(memory.get("memory_id", "") or ""),
                        memory,
                    )
                )
            scored_rows.sort(key=lambda item: (not item[0], -item[1], -item[2], item[3]))
            if limit_override is None:
                selected_older = [memory for pinned, _, _, _, memory in scored_rows if pinned]
            else:
                selected_older = []
                older_budget = max(0, limit_override - len(latest_group))
                for _, _, _, _, memory in scored_rows[:older_budget]:
                    selected_older.append(memory)
            survivors = selected_older + latest_group
            protected_count = len([1 for memory in selected_older if str(memory.get("memory_id", "") or "") not in latest_ids])

        if limit_override is not None and len(survivors) > limit_override:
            mandatory_ids = {str(m.get("memory_id", "") or "") for m in latest_group if str(m.get("memory_id", "") or "")}
            protected_rows = []
            for memory in survivors:
                memory_id = str(memory.get("memory_id", "") or "")
                if memory_id in mandatory_ids:
                    continue
                score = self._forget_retention_score(
                    memory,
                    older_index=0,
                    older_total=max(1, len(survivors)),
                    protected_kinds=protected_kinds,
                    min_reality_weight=min_reality_weight,
                    min_total_item_energy=min_total_item_energy,
                )
                protected_rows.append((score, int(memory.get("tick_index", -1) or -1), memory_id, memory))
            protected_rows.sort(key=lambda item: (-item[0], -item[1], item[2]))
            capacity = max(0, limit_override - len(latest_group))
            survivors = [row[3] for row in protected_rows[:capacity]] + latest_group

        removed_ids = [
            str(memory.get("memory_id", "") or "")
            for memory in self._memories
            if str(memory.get("memory_id", "") or "") and str(memory.get("memory_id", "") or "") not in {str(item.get("memory_id", "") or "") for item in survivors}
        ]
        removed = len(removed_ids)
        after_count = len(survivors)
        result = {
            "removed": removed,
            "memory_count": after_count,
            "before_count": before_count,
            "strategy": clean_strategy,
            "keep_latest": keep_latest,
            "min_reality_weight": _round4(min_reality_weight),
            "min_total_item_energy": _round4(min_total_item_energy),
            "max_memory_count": limit_override if limit_override is not None else None,
            "protect_memory_kinds": sorted(protected_kinds),
            "protected_count": protected_count,
            "dry_run": bool(dry_run),
            "kind_histogram_before": self._memory_kind_histogram(self._memories),
            "kind_histogram_after": self._memory_kind_histogram(survivors),
            "retained_memory_ids_preview": [str(memory.get("memory_id", "") or "") for memory in survivors[-12:] if str(memory.get("memory_id", "") or "")],
            "removed_memory_ids_preview": removed_ids[:12],
        }
        if dry_run:
            return result
        self._memories = survivors
        self._recent_ids = deque([m["memory_id"] for m in self._memories[-self.max_recent :] if str(m.get("memory_id", "") or "")], maxlen=self.max_recent)
        self._recent_window_ids = deque([m["memory_id"] for m in self._memories[-self.recent_window_size :] if str(m.get("memory_id", "") or "")], maxlen=self.recent_window_size)
        self._rebuild_indexes()
        return result

    def index_summary(self) -> dict[str, Any]:
        return {
            "vector": self._vector_index.summary(),
            "spacetime": self._spacetime_index.summary(),
            "posting": {
                "label_terms": len(self._posting_by_label),
                "unit_terms": len(self._posting_by_unit),
                "bigram_terms": len(self._posting_by_bigram),
            },
            "cache": self.cache_summary(),
            "recall_fatigue": {
                "state_count": len(self._recall_fatigue),
                "decay": _round4(self.recall_fatigue_decay),
                "gain": _round4(self.recall_fatigue_gain),
                "accumulate_scale": _round4(self.recall_fatigue_accumulate_scale),
                "max": _round4(self.recall_fatigue_max),
                "min_multiplier": _round4(self.recall_fatigue_min_multiplier),
            },
            "branch_credibility": {
                "state_count": len(self._branch_credibility),
                "decay": _round4(self.branch_credibility_decay),
                "match_gain": _round4(self.branch_credibility_match_gain),
                "miss_gain": _round4(self.branch_credibility_miss_gain),
                "min_multiplier": _round4(self.branch_credibility_min_multiplier),
                "max_multiplier": _round4(self.branch_credibility_max_multiplier),
            },
            "bundle_format": "layered_v2",
        }

    def cache_summary(self) -> dict[str, Any]:
        return {
            "memory_revision": int(self._memory_revision),
            "query_vector_cache_size": len(self._query_vector_cache),
            "candidate_cache_size": len(self._candidate_cache),
            "neighbor_rows_cache_size": len(self._neighbor_rows_cache),
            "pair_relation_cache_size": len(self._pair_relation_cache),
            "stats": dict(self._cache_stats),
        }

    def save_deployment_bundle(self, directory: Path) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=True)
        payload = self.export_payload()
        bundle_meta = {
            "schema_id": "memory_store_bundle/v1",
            "schema_version": "1.0",
            "memory_count": len(self._memories),
            "counter": self._counter,
            "index_summary": self.index_summary(),
        }
        legacy_path = directory / "memory_store_v2.json"
        legacy_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        memories_path = directory / "memories.jsonl"
        memory_lines = [json.dumps(memory, ensure_ascii=False) for memory in self._memories]
        memories_path.write_text("\n".join(memory_lines), encoding="utf-8")

        posting_payload = {
            "label": {key: sorted(value) for key, value in self._posting_by_label.items()},
            "unit": {key: sorted(value) for key, value in self._posting_by_unit.items()},
            "bigram": {key: sorted(value) for key, value in self._posting_by_bigram.items()},
        }
        posting_path = directory / "posting_index.json"
        posting_path.write_text(json.dumps(posting_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        vector_result = self._vector_index.save_bundle(directory)
        spacetime_result = self._spacetime_index.save_bundle(directory)

        bundle_meta["schema_version"] = "2.0"
        bundle_meta["files"] = {
            "legacy_json": legacy_path.name,
            "memories_jsonl": memories_path.name,
            "posting_json": posting_path.name,
            "vector_meta": "vector_index_meta.json",
            "spacetime_meta": "spacetime_index_meta.json",
        }
        bundle_meta["vector_bundle"] = {
            "engine": self._vector_index.engine_name(),
            "result": vector_result,
        }
        bundle_meta["spacetime_bundle"] = spacetime_result
        (directory / "bundle_meta.json").write_text(json.dumps(bundle_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "ok": True,
            "directory": str(directory),
            "memory_count": len(self._memories),
            "index_summary": self.index_summary(),
            "bundle_format": "layered_v2",
        }

    def load_deployment_bundle(self, directory: Path) -> dict[str, Any]:
        meta_path = directory / "bundle_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            files = dict(meta.get("files", {}) or {})
            memories_path = directory / str(files.get("memories_jsonl", "memories.jsonl") or "memories.jsonl")
            posting_path = directory / str(files.get("posting_json", "posting_index.json") or "posting_index.json")
            if memories_path.exists():
                self._memories = []
                self._memories_by_id = {}
                for line in memories_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        memory = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(memory, dict):
                        self._memories.append(memory)
                        memory_id = str(memory.get("memory_id", "") or "")
                        if memory_id:
                            self._memories_by_id[memory_id] = memory
                self._counter = int(meta.get("counter", len(self._memories)) or len(self._memories))
                self._recent_ids = deque([m["memory_id"] for m in self._memories[-self.max_recent :] if str(m.get("memory_id", "") or "")], maxlen=self.max_recent)
                self._recent_window_ids = deque([m["memory_id"] for m in self._memories[-self.recent_window_size :] if str(m.get("memory_id", "") or "")], maxlen=self.recent_window_size)
                self._posting_by_label = defaultdict(set)
                self._posting_by_unit = defaultdict(set)
                self._posting_by_bigram = defaultdict(set)
                if posting_path.exists():
                    posting_payload = json.loads(posting_path.read_text(encoding="utf-8"))
                    for label, ids in dict(posting_payload.get("label", {}) or {}).items():
                        self._posting_by_label[str(label)] = set(str(item or "") for item in (ids or []) if str(item or ""))
                    for unit, ids in dict(posting_payload.get("unit", {}) or {}).items():
                        self._posting_by_unit[str(unit)] = set(str(item or "") for item in (ids or []) if str(item or ""))
                    for bigram, ids in dict(posting_payload.get("bigram", {}) or {}).items():
                        self._posting_by_bigram[str(bigram)] = set(str(item or "") for item in (ids or []) if str(item or ""))
                else:
                    for memory in self._memories:
                        memory_id = str(memory.get("memory_id", "") or "")
                        if not memory_id:
                            continue
                        retrieval_labels = memory.get("retrieval_labels", memory.get("sa_labels", [])) or []
                        for label in set(str(item or "") for item in retrieval_labels if str(item or "")):
                            self._posting_by_label[label].add(memory_id)
                        for unit in set(str(item or "") for item in (memory.get("units", []) or []) if str(item or "")):
                            self._posting_by_unit[unit].add(memory_id)
                        for bigram in set(str(item or "") for item in (memory.get("unit_bigrams", []) or []) if str(item or "")):
                            self._posting_by_bigram[bigram].add(memory_id)
                self._vector_index.load_bundle(directory)
                self._spacetime_index.load_bundle(directory)
                return {
                    "ok": True,
                    "directory": str(directory),
                    "memory_count": len(self._memories),
                    "index_summary": self.index_summary(),
                    "loaded_via": "layered_v2",
                }
        payload_path = directory / "memory_store_v2.json"
        if not payload_path.exists():
            return {"ok": False, "error": "memory_store bundle not found", "path": str(payload_path)}
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        self.import_payload(payload)
        return {
            "ok": True,
            "directory": str(directory),
            "memory_count": len(self._memories),
            "index_summary": self.index_summary(),
            "loaded_via": "legacy_json",
        }

    @staticmethod
    def inspect_deployment_bundle(directory: Path) -> dict[str, Any]:
        meta_path = directory / "bundle_meta.json"
        if not meta_path.exists():
            legacy_path = directory / "memory_store_v2.json"
            if not legacy_path.exists():
                return {"ok": False, "error": "memory_store bundle not found", "path": str(meta_path)}
            payload = json.loads(legacy_path.read_text(encoding="utf-8"))
            index_summary = dict(payload.get("index_summary", {}) or {})
            return {
                "ok": True,
                "directory": str(directory),
                "bundle_format": "legacy_json",
                "memory_count": int(payload.get("memory_count", len(payload.get("memories", []) or [])) or 0),
                "index_summary": index_summary,
                "files": {
                    "legacy_json": legacy_path.name,
                },
            }
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        files = dict(meta.get("files", {}) or {})
        file_status = {
            key: {
                "name": str(value or ""),
                "exists": bool((directory / str(value or "")).exists()) if str(value or "") else False,
            }
            for key, value in files.items()
        }
        return {
            "ok": True,
            "directory": str(directory),
            "bundle_format": "layered_v2" if str(meta.get("schema_version", "")) == "2.0" else "legacy_json",
            "schema_id": str(meta.get("schema_id", "") or ""),
            "schema_version": str(meta.get("schema_version", "") or ""),
            "memory_count": int(meta.get("memory_count", 0) or 0),
            "index_summary": dict(meta.get("index_summary", {}) or {}),
            "vector_bundle": dict(meta.get("vector_bundle", {}) or {}),
            "spacetime_bundle": dict(meta.get("spacetime_bundle", {}) or {}),
            "files": file_status,
        }

    def _collect_candidate_ids(self, *, query_set: set[str], query_units: list[str], query_bigrams: set[str]) -> set[str]:
        candidate_ids: set[str] = set()
        for label in query_set:
            candidate_ids.update(self._posting_by_label.get(label, set()))
        for unit in query_units[:24]:
            candidate_ids.update(self._posting_by_unit.get(unit, set()))
        for bigram in tuple(sorted(query_bigrams)[:24]):
            candidate_ids.update(self._posting_by_bigram.get(bigram, set()))
        for recent_id in self._recent_window_ids:
            candidate_ids.add(recent_id)
        return candidate_ids

    def _trim_candidates(self, candidate_ids: set[str], *, ann_by_id: dict[str, dict[str, Any]]) -> set[str]:
        if len(candidate_ids) <= self.candidate_limit:
            return candidate_ids
        rows = []
        for memory_id in candidate_ids:
            memory = self._memories_by_id.get(memory_id)
            tick_index = int(-1 if memory.get("tick_index", -1) is None else memory.get("tick_index", -1)) if memory else -1
            ann_score = float((ann_by_id.get(memory_id, {}) or {}).get("vector_score", -1.0) or -1.0)
            rows.append((ann_score, tick_index, memory_id))
        rows.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return {memory_id for _, _, memory_id in rows[: self.candidate_limit]}

    def _forget_is_pinned(
        self,
        memory: dict[str, Any],
        *,
        protected_kinds: set[str],
        min_reality_weight: float,
        min_total_item_energy: float,
    ) -> bool:
        kind = str(memory.get("memory_kind", "") or "")
        if kind and kind in protected_kinds:
            return True
        if min_reality_weight > 0.0 and float(memory.get("reality_weight", 0.0) or 0.0) >= min_reality_weight:
            return True
        if min_total_item_energy > 0.0 and float(memory.get("total_item_energy", 0.0) or 0.0) >= min_total_item_energy:
            return True
        return False

    def _forget_retention_score(
        self,
        memory: dict[str, Any],
        *,
        older_index: int,
        older_total: int,
        protected_kinds: set[str],
        min_reality_weight: float,
        min_total_item_energy: float,
    ) -> float:
        reality_weight = min(1.5, max(0.0, float(memory.get("reality_weight", 0.0) or 0.0))) / 1.5
        total_item_energy = min(12.0, max(0.0, float(memory.get("total_item_energy", 0.0) or 0.0))) / 12.0
        source_ref_score = min(1.0, len(list(memory.get("source_refs", []) or [])) / 4.0)
        modality_score = min(1.0, len(list(memory.get("modalities", []) or [])) / 3.0)
        recency_score = 0.0
        if older_total > 0:
            recency_score = 1.0 - (float(max(0, older_index)) / float(max(1, older_total)))
        pinned_bonus = 0.0
        if self._forget_is_pinned(
            memory,
            protected_kinds=protected_kinds,
            min_reality_weight=min_reality_weight,
            min_total_item_energy=min_total_item_energy,
        ):
            pinned_bonus = 0.35
        return _round4(
            0.34 * reality_weight
            + 0.24 * total_item_energy
            + 0.18 * recency_score
            + 0.12 * source_ref_score
            + 0.07 * modality_score
            + pinned_bonus
        )

    def _memory_kind_histogram(self, rows: list[dict[str, Any]]) -> dict[str, int]:
        hist: dict[str, int] = {}
        for memory in rows:
            kind = str(memory.get("memory_kind", "") or "memory")
            hist[kind] = int(hist.get(kind, 0) or 0) + 1
        return dict(sorted(hist.items(), key=lambda item: (item[0], item[1])))

    def _candidate_sources_for_memory(
        self,
        memory_id: str,
        *,
        query_set: set[str],
        query_units: list[str] | tuple[str, ...],
        query_bigrams: set[str] | tuple[str, ...],
        ann_by_id: dict[str, dict[str, Any]],
    ) -> list[str]:
        sources: list[str] = []
        if any(memory_id in self._posting_by_label.get(label, set()) for label in query_set):
            sources.append("label_posting")
        if any(memory_id in self._posting_by_unit.get(unit, set()) for unit in query_units[:12]):
            sources.append("unit_posting")
        bigram_iter = query_bigrams[:12] if isinstance(query_bigrams, tuple) else tuple(sorted(query_bigrams)[:12])
        if any(memory_id in self._posting_by_bigram.get(bigram, set()) for bigram in bigram_iter):
            sources.append("bigram_posting")
        if memory_id in self._recent_window_ids:
            sources.append("recent_window")
        if memory_id in ann_by_id:
            sources.append("vector_ann")
        return sources

    def _touch_memory_revision(self, *, clear_pair_relation_cache: bool = True) -> None:
        self._memory_revision += 1
        self._candidate_cache.clear()
        self._neighbor_rows_cache.clear()
        if clear_pair_relation_cache:
            self._pair_relation_cache.clear()

    def _filter_recall_fatigue_to_live_memories(self) -> None:
        live_ids = {str(memory.get("memory_id", "") or "") for memory in self._memories if str(memory.get("memory_id", "") or "")}
        self._recall_fatigue = {
            memory_id: {
                "value": _round4(max(0.0, float((entry or {}).get("value", 0.0) or 0.0))),
                "tick_index": int((entry or {}).get("tick_index", 0) or 0),
            }
            for memory_id, entry in self._recall_fatigue.items()
            if memory_id in live_ids and isinstance(entry, dict)
        }

    def _filter_branch_credibility_to_live_memories(self) -> None:
        live_ids = {str(memory.get("memory_id", "") or "") for memory in self._memories if str(memory.get("memory_id", "") or "")}
        self._branch_credibility = {
            memory_id: {
                "bias": _round4(float((entry or {}).get("bias", 0.0) or 0.0)),
                "tick_index": int((entry or {}).get("tick_index", 0) or 0),
            }
            for memory_id, entry in self._branch_credibility.items()
            if memory_id in live_ids and isinstance(entry, dict)
        }

    def _is_commitment_comparable_label(self, label: str) -> bool:
        clean = str(label or "")
        if not clean:
            return False
        if clean.startswith(
            (
                "text::",
                "phrase::",
                "audio::",
                "audio_core::",
                "audio_form::",
                "audio_global::",
                "audio_global_form::",
                "vision_mem::",
                "vision_core::",
                "vision_form::",
                "vision_global::",
                "vision_dyn::",
                "vision_dyn_core::",
                "vision_dyn_form::",
                "vision_contour_core::",
                "vision_contour_form::",
                "vision_global_contour::",
                "vision_global_contour_form::",
            )
        ):
            return True
        if clean.startswith("attr::"):
            return True
        return False

    def _compute_commitment(
        self,
        *,
        label: str,
        energy: float,
        total_weight: float,
        peak_weight: float,
        support: dict[str, Any],
        tick_index: int,
    ) -> float:
        weighted_support = max(0.0, float(support.get("weighted_support", energy) or energy))
        top_branch_support = max(0.0, float(support.get("top_branch_support", 0.0) or 0.0))
        top_source_support = max(0.0, float(support.get("top_source_support", 0.0) or 0.0))
        real_support = max(0.0, float(support.get("real_support", 0.0) or 0.0))
        max_branch_weight = max(0.0, float(support.get("max_branch_weight", 0.0) or 0.0))
        branch_count = max(0, int(support.get("branch_count", 0) or 0))
        branch_key = str(support.get("max_branch_key", "") or "")
        relative_energy = energy / max(1e-6, total_weight) if total_weight > 0.0 else 0.0
        peak_ratio = energy / max(1e-6, peak_weight) if peak_weight > 0.0 else 0.0
        concentration = max_branch_weight / max(0.001, weighted_support)
        top_ratio = top_branch_support / max(0.001, weighted_support)
        source_ratio = top_source_support / max(0.001, weighted_support)
        real_ratio = real_support / max(0.001, weighted_support)
        branch_focus = 1.0 / max(1.0, float(branch_count))
        credibility = self._branch_credibility_multiplier(branch_key, tick_index=tick_index) if branch_key else 1.0
        fatigue_multiplier = self._recall_fatigue_multiplier(branch_key, tick_index=tick_index) if branch_key else 1.0
        credibility_term = _clamp((credibility - self.branch_credibility_min_multiplier) / max(1e-6, self.branch_credibility_max_multiplier - self.branch_credibility_min_multiplier), 0.0, 1.0)
        fatigue_term = _clamp(fatigue_multiplier, 0.0, 1.0)
        modality_bias = (
            0.04
            if str(label or "").startswith(
                (
                    "vision_mem::",
                    "vision_core::",
                    "vision_form::",
                    "vision_global::",
                    "vision_dyn::",
                    "vision_dyn_core::",
                    "vision_dyn_form::",
                    "vision_contour_core::",
                    "vision_contour_form::",
                    "vision_global_contour::",
                    "vision_global_contour_form::",
                    "audio::",
                    "audio_core::",
                    "audio_form::",
                    "audio_global::",
                    "audio_global_form::",
                )
            )
            else 0.0
        )
        commitment = (
            0.26 * relative_energy
            + 0.18 * peak_ratio
            + 0.17 * concentration
            + 0.12 * top_ratio
            + 0.09 * source_ratio
            + 0.08 * real_ratio
            + 0.05 * branch_focus
            + 0.03 * credibility_term
            + 0.02 * fatigue_term
            + modality_bias
        )
        return _clamp(commitment, 0.0, 1.0)

    def _rebuild_memory_profiles(self) -> None:
        self._memory_profile_cache = {}
        for memory in self._memories:
            memory_id = str(memory.get("memory_id", "") or "")
            if memory_id:
                self._memory_profile_cache[memory_id] = self._build_memory_profile(memory)

    def _build_query_signature(
        self,
        *,
        query_labels: list[str],
        query_weights: dict[str, float],
        query_items: list[dict[str, Any]] | None,
        query_units: list[str],
        recent_focus_units: list[str],
        query_spacetime: dict[str, Any] | None,
    ) -> str:
        hasher = hashlib.blake2b(digest_size=16)
        for label, weight in sorted((str(label or ""), _round4(float(query_weights.get(str(label or ""), 0.0) or 0.0))) for label in query_labels if str(label or "")):
            hasher.update(f"L|{label}|{weight:.4f}\n".encode("utf-8"))
        item_rows: list[tuple[float, str, str, str, str, str]] = []
        for item in (query_items or []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            raw_attrs = item.get("attributes", {}) or {}
            attrs = raw_attrs if isinstance(raw_attrs, dict) else {}
            raw_coords = item.get("coords", {}) or {}
            coords = raw_coords if isinstance(raw_coords, dict) else {}
            attr_sig = "|".join(
                f"{str(k)}={_round4(float(v)) if isinstance(v, (int, float)) else str(v)}"
                for k, v in sorted(attrs.items(), key=lambda row: str(row[0]))[:12]
            )
            coord_sig = "|".join(
                f"{str(k)}={_round4(float(v))}"
                for k, v in sorted(coords.items(), key=lambda row: str(row[0]))
                if isinstance(v, (int, float))
            )
            item_rows.append(
                (
                    -_round4(float(item.get("energy", 0.0) or 0.0)),
                    label,
                    str(item.get("channel", "") or ""),
                    str(item.get("sa_kind", "") or ""),
                    attr_sig,
                    coord_sig,
                )
            )
        item_rows.sort()
        for neg_energy, label, channel, sa_kind, attr_sig, coord_sig in item_rows[:48]:
            hasher.update(f"I|{label}|{-neg_energy:.4f}|{channel}|{sa_kind}|{attr_sig}|{coord_sig}\n".encode("utf-8"))
        focus_tail = [str(item or "") for item in recent_focus_units[-4:] if str(item or "")]
        spacetime = dict(query_spacetime or {})
        for unit in (str(item or "") for item in query_units if str(item or "")):
            hasher.update(f"U|{unit}\n".encode("utf-8"))
        for unit in focus_tail:
            hasher.update(f"F|{unit}\n".encode("utf-8"))
        for key in (
            "has_space",
            "t",
            "target_t",
            "target_delta_t",
            "time_sigma",
            "time_confidence",
            "motion_center_speed",
            "motion_sigma",
            "motion_confidence",
            "rhythm_period_ticks",
            "rhythm_period_sigma",
            "rhythm_confidence",
            "rhythm_phase_error",
            "rhythm_time_to_next",
            "rhythm_family_key",
            "feedback_valence",
            "feedback_sigma",
            "feedback_confidence",
            "x",
            "y",
            "z",
            "has_relative_space",
            "rel_x",
            "rel_y",
            "rel_r",
            "screen_w",
            "screen_h",
            "local_order_span",
        ):
            value = spacetime.get(key)
            if isinstance(value, bool):
                norm = "1" if value else "0"
            elif isinstance(value, (int, float)):
                norm = f"{_round4(float(value)):.4f}"
            else:
                norm = str(value or "")
            hasher.update(f"S|{key}|{norm}\n".encode("utf-8"))
        return hasher.hexdigest()

    def _get_or_build_query_vector(
        self,
        *,
        query_signature: str,
        query_labels: list[str],
        query_weights: dict[str, float],
        query_items: list[dict[str, Any]] | None,
        query_units: list[str],
        recent_focus_units: list[str],
        query_spacetime: dict[str, Any] | None,
    ) -> tuple[np.ndarray, list[str]]:
        cached = self._query_vector_cache.get(query_signature)
        if cached is not None:
            self._cache_stats["query_vector_hit"] += 1
            self._query_vector_cache.move_to_end(query_signature)
            return cached[0].copy(), list(cached[1])
        self._cache_stats["query_vector_miss"] += 1
        vector, tokens = self._embedder.build_query_vector(
            query_labels=query_labels,
            query_weights=query_weights,
            query_items=query_items,
            query_units=query_units,
            recent_focus_units=recent_focus_units,
            query_spacetime=query_spacetime,
        )
        self._query_vector_cache[query_signature] = (vector.copy(), list(tokens))
        self._bounded_ordered_dict(self._query_vector_cache, self._query_vector_cache_limit)
        return vector, tokens

    def _get_or_build_candidate_state(
        self,
        *,
        query_signature: str,
        query_set: set[str],
        query_units: list[str],
        query_bigrams: set[str],
        query_vector: np.ndarray,
    ) -> dict[str, Any]:
        cache_key = (int(self._memory_revision), query_signature)
        cached = self._candidate_cache.get(cache_key)
        if cached is not None:
            self._cache_stats["candidate_hit"] += 1
            self._candidate_cache.move_to_end(cache_key)
            return {
                "candidate_ids": set(cached["candidate_ids"]),
                "ann_by_id": dict(cached["ann_by_id"]),
            }
        self._cache_stats["candidate_miss"] += 1
        posting_candidates = self._collect_candidate_ids(query_set=query_set, query_units=query_units, query_bigrams=query_bigrams)
        ann_rows = self._vector_index.search(query_vector, top_k=self.ann_top_k)
        ann_by_id = {str(row.get("memory_id", "") or ""): row for row in ann_rows if str(row.get("memory_id", "") or "")}
        candidate_ids = set(posting_candidates) | set(ann_by_id.keys())
        candidate_ids = self._trim_candidates(candidate_ids, ann_by_id=ann_by_id)
        payload = {
            "candidate_ids": tuple(sorted(candidate_ids)),
            "ann_by_id": dict(ann_by_id),
        }
        self._candidate_cache[cache_key] = payload
        self._bounded_ordered_dict(self._candidate_cache, self._candidate_cache_limit)
        return {
            "candidate_ids": set(payload["candidate_ids"]),
            "ann_by_id": dict(payload["ann_by_id"]),
        }

    def _memory_profile(self, memory_id: str, *, memory: dict[str, Any] | None = None) -> dict[str, Any]:
        clean_id = str(memory_id or "")
        cached = self._memory_profile_cache.get(clean_id)
        if cached is not None:
            return cached
        row = memory or self.get_memory(clean_id) or {}
        profile = self._build_memory_profile(row)
        if clean_id:
            self._memory_profile_cache[clean_id] = profile
        return profile

    def _build_memory_profile(self, memory: dict[str, Any]) -> dict[str, Any]:
        stored_retrieval_weights = dict(memory.get("retrieval_label_weights", {}) or {})
        label_weights_source = stored_retrieval_weights if stored_retrieval_weights else dict(memory.get("label_weights", {}) or {})
        label_weights = {str(key): float(value or 0.0) for key, value in label_weights_source.items() if str(key)}
        labels = set(label_weights)
        units = [str(item or "") for item in (memory.get("units", []) or []) if str(item or "")]
        unit_positions: dict[str, int] = {}
        for index, unit in enumerate(units):
            if unit and unit not in unit_positions:
                unit_positions[unit] = index
        modalities = {str(item or "") for item in (memory.get("modalities", []) or []) if str(item or "")}
        memory_kind = str(memory.get("memory_kind", "") or "")
        items = [dict(item) for item in (memory.get("items", []) or []) if isinstance(item, dict)]
        return {
            "memory_id": str(memory.get("memory_id", "") or ""),
            "memory_kind": memory_kind,
            "label_weights": label_weights,
            "label_set": labels,
            "label_mass": sum(float(label_weights.get(label, 0.0) or 0.0) for label in labels) or 1.0,
            "units": units,
            "unit_set": set(units),
            "unit_positions": unit_positions,
            "unit_count": len(units),
            "bigram_set": {str(item or "") for item in (memory.get("unit_bigrams", []) or []) if str(item or "")},
            "modalities_set": modalities,
            "tick_index": int(memory.get("tick_index", -1) or -1),
            "reality_bonus": min(1.25, max(0.05, float(memory.get("reality_weight", 1.0) or 1.0))) / 1.25,
            "items": items,
            "contour_rows": self._extract_contour_rows(items),
        }

    def _extract_contour_rows(self, items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in (items or []):
            if not isinstance(item, dict):
                continue
            raw_attrs = item.get("attributes", {}) or {}
            if not isinstance(raw_attrs, dict):
                continue
            string_values = tuple(
                str(raw_attrs.get(key, "") or "")
                for key in (
                    "hu_signature",
                    "radial_signature",
                    "proj_h_bin",
                    "proj_v_bin",
                    "radial_bin",
                    "quadrant_bin",
                    "edge_contact_bin",
                    "bbox_signature",
                    "rgb_signature",
                    "foreground_polarity",
                )
            )
            if not (string_values[0] or string_values[1]):
                continue
            numeric_values = tuple(
                float(raw_attrs.get(key, 0.0) or 0.0) if key in raw_attrs else None
                for key in (
                    "area_ratio",
                    "bbox_fill",
                    "solidity",
                    "roundness",
                    "aspect_ratio",
                    "hole_like",
                    "center_void",
                    "horizontal_symmetry",
                    "vertical_symmetry",
                    "avg_r",
                    "avg_g",
                    "avg_b",
                    "brightness",
                    "motion_strength",
                    "motion_peak",
                    "motion",
                )
            )
            rows.append({"strings": string_values, "numbers": numeric_values})
        return rows

    def _visual_contour_similarity(self, query_rows: list[dict[str, Any]] | None, memory_rows: list[dict[str, Any]] | None) -> float:
        if not query_rows or not memory_rows:
            return 0.0
        best = 0.0
        for left in query_rows[:6]:
            left_strings = tuple(left.get("strings", ()) or ())
            left_numbers = tuple(left.get("numbers", ()) or ())
            for right in memory_rows[:12]:
                right_strings = tuple(right.get("strings", ()) or ())
                right_numbers = tuple(right.get("numbers", ()) or ())
                score = 0.0
                pieces = 0
                for str_index, (lv, rv) in enumerate(zip(left_strings, right_strings)):
                    if not lv or not rv:
                        continue
                    length = min(len(lv), len(rv))
                    if length <= 0:
                        continue
                    match = sum(1 for index in range(length) if lv[index] == rv[index]) / float(length)
                    weight = 1.0
                    if str_index in {0, 1}:
                        weight = 1.55
                    elif str_index in {2, 3, 4, 5}:
                        weight = 1.15
                    elif str_index in {6, 7, 8}:
                        weight = 1.05
                    score += match * weight
                    pieces += 1
                for index, (lv, rv) in enumerate(zip(left_numbers, right_numbers)):
                    if lv is None or rv is None:
                        continue
                    diff = abs(lv - rv)
                    scale = 1.0
                    if index == 4:
                        scale = 2.0
                    elif index >= 9:
                        scale = 0.35
                    numeric_match = max(0.0, 1.0 - diff / scale)
                    weight = 1.0
                    if index in {0, 1, 2, 3, 4}:
                        weight = 1.1
                    elif index in {9, 10, 11, 12}:
                        weight = 0.95
                    score += numeric_match * weight
                    pieces += 1
                if pieces <= 0:
                    continue
                best = max(best, score / float(pieces))
        return _clamp(best, 0.0, 1.0)

    def _get_or_build_neighbor_rows(self, memory: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
        memory_id = str(memory.get("memory_id", "") or "")
        cache_key = (int(self._memory_revision), memory_id, int(limit))
        cached = self._neighbor_rows_cache.get(cache_key)
        if cached is not None:
            self._cache_stats["neighbor_hit"] += 1
            self._neighbor_rows_cache.move_to_end(cache_key)
            return [dict(row) for row in cached]
        self._cache_stats["neighbor_miss"] += 1
        rows = self._spacetime_index.neighbors_for_memory(memory, limit=limit)
        stored = [dict(row) for row in rows]
        self._neighbor_rows_cache[cache_key] = stored
        self._bounded_ordered_dict(self._neighbor_rows_cache, self._neighbor_rows_cache_limit)
        return [dict(row) for row in stored]

    def _pair_vector_related(self, left_memory_id: str, right_memory_id: str) -> float:
        clean_left = str(left_memory_id or "")
        clean_right = str(right_memory_id or "")
        if not clean_left or not clean_right:
            return 0.0
        if clean_left == clean_right:
            return 1.0
        if clean_left < clean_right:
            cache_key = (clean_left, clean_right)
        else:
            cache_key = (clean_right, clean_left)
        cached = self._pair_relation_cache.get(cache_key)
        if cached is not None:
            self._cache_stats["pair_hit"] += 1
            self._pair_relation_cache.move_to_end(cache_key)
            return float(cached)
        self._cache_stats["pair_miss"] += 1
        left_vec = self._vector_index.get_vector(clean_left)
        right_vec = self._vector_index.get_vector(clean_right)
        vector_related = 0.0
        if left_vec is not None and right_vec is not None:
            vector_related = max(0.0, (self._embedder.cosine(left_vec, right_vec) + 1.0) / 2.0)
        self._pair_relation_cache[cache_key] = float(vector_related)
        self._bounded_ordered_dict(self._pair_relation_cache, self._pair_relation_cache_limit)
        return float(vector_related)

    def _bounded_ordered_dict(self, mapping: OrderedDict[Any, Any], limit: int) -> None:
        while len(mapping) > max(1, int(limit)):
            mapping.popitem(last=False)

    def _index_memory(self, *, memory: dict[str, Any], vector: np.ndarray, defer_vector_add: bool = False) -> None:
        memory_id = str(memory.get("memory_id", "") or "")
        if not memory_id:
            return
        retrieval_labels = memory.get("retrieval_labels", memory.get("sa_labels", [])) or []
        for label in set(str(item or "") for item in retrieval_labels if str(item or "")):
            self._posting_by_label[label].add(memory_id)
        for unit in set(str(item or "") for item in (memory.get("units", []) or []) if str(item or "")):
            self._posting_by_unit[unit].add(memory_id)
        for bigram in set(str(item or "") for item in (memory.get("unit_bigrams", []) or []) if str(item or "")):
            self._posting_by_bigram[bigram].add(memory_id)
        if not defer_vector_add:
            self._vector_index.add(memory_id, vector)
        self._spacetime_index.add(memory)

    def _rebuild_indexes(self) -> None:
        self._memories_by_id = {}
        self._posting_by_label = defaultdict(set)
        self._posting_by_unit = defaultdict(set)
        self._posting_by_bigram = defaultdict(set)
        self._memory_profile_cache = {}
        self._vector_index = VectorIndexV2(
            dim=self.vector_dim,
            backend=self.vector_backend,
            ann_enabled=self.ann_enabled,
            ann_top_k=self.ann_top_k,
        )
        spacetime_summary = self._spacetime_index.summary()
        self._spacetime_index = SpacetimeIndexV2(
            backend=self.spacetime_backend,
            time_bucket_size=int(spacetime_summary.get("time_bucket_size", 8) or 8),
            space_bucket_size=float(spacetime_summary.get("space_bucket_size", 0.25) or 0.25),
            default_time_radius=int(spacetime_summary.get("default_time_radius", 24) or 24),
            default_space_radius=float(spacetime_summary.get("default_space_radius", 0.45) or 0.45),
        )
        for memory in self._memories:
            memory_id = str(memory.get("memory_id", "") or "")
            if not memory_id:
                continue
            self._memories_by_id[memory_id] = memory
            vector, _ = self._embedder.build_memory_vector(
                units=list(memory.get("units", []) or []),
                items=list(memory.get("items", []) or []),
                retrieval_label_weights=dict(memory.get("retrieval_label_weights", {}) or {}),
                text=str(memory.get("text", "") or ""),
                modalities=list(memory.get("modalities", []) or []),
                spacetime=dict(memory.get("spacetime", {}) or {}),
            )
            self._index_memory(memory=memory, vector=vector)
            self._memory_profile_cache[memory_id] = self._build_memory_profile(memory)
        self._filter_recall_fatigue_to_live_memories()
        self._filter_branch_credibility_to_live_memories()
        self._touch_memory_revision(clear_pair_relation_cache=True)

    def _build_unit_bigrams(self, units: list[str]) -> list[str]:
        if len(units) < 2:
            return []
        return [f"{units[index]}__{units[index + 1]}" for index in range(0, len(units) - 1)]

    def _expand_query_label_weights(
        self,
        *,
        query_labels: list[str],
        query_weights: dict[str, float],
        query_items: list[dict[str, Any]] | None,
    ) -> dict[str, float]:
        expanded: dict[str, float] = {}
        for raw_label in query_labels:
            clean_label = str(raw_label or "")
            if not clean_label:
                continue
            expanded[clean_label] = _round4(
                float(expanded.get(clean_label, 0.0) or 0.0)
                + max(0.05, float(query_weights.get(clean_label, 0.0) or 0.0))
            )
        for item in (query_items or []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            base_weight = max(
                0.04,
                float(query_weights.get(label, 0.0) or 0.0),
                float(item.get("query_weight", item.get("energy", 0.0)) or 0.0),
            )
            for alias_label, alias_scale in self._item_retrieval_label_rows(item):
                clean_alias = str(alias_label or "")
                if not clean_alias:
                    continue
                expanded[clean_alias] = _round4(
                    float(expanded.get(clean_alias, 0.0) or 0.0)
                    + base_weight * max(0.05, float(alias_scale or 0.0))
                )
        return {key: _round4(value) for key, value in expanded.items() if float(value or 0.0) > 0.0}

    def _build_retrieval_label_weights(self, items: list[dict[str, Any]]) -> dict[str, float]:
        weights: dict[str, float] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            energy = max(0.0, float(item.get("energy", 0.0) or 0.0))
            if energy <= 0.0:
                continue
            for label, factor in self._item_retrieval_label_rows(item):
                clean_label = str(label or "")
                if not clean_label:
                    continue
                weights[clean_label] = _round4(
                    float(weights.get(clean_label, 0.0) or 0.0)
                    + energy * max(0.05, float(factor or 0.0))
                )
        return {key: _round4(value) for key, value in weights.items() if float(value or 0.0) > 0.0}

    def _item_retrieval_label_rows(self, item: dict[str, Any]) -> list[tuple[str, float]]:
        label = str(item.get("sa_label", "") or "")
        if not label:
            return []
        rows: dict[str, float] = {label: 1.0}
        attrs = dict(item.get("attributes", {}) or {})
        sample_role = str(attrs.get("sample_role", "") or "")
        sa_kind = str(item.get("sa_kind", "") or "")

        if label.startswith("vision_mem::global_"):
            rest = label.split("vision_mem::", 1)[1]
            rows[f"vision_global::{rest}"] = max(float(rows.get(f"vision_global::{rest}", 0.0) or 0.0), 0.62)
            if rest.startswith("global_edge::"):
                parts = rest.split("::")
                if len(parts) >= 3:
                    invariant_edge = f"vision_global::{parts[0]}::{parts[1]}::{parts[2]}"
                    rows[invariant_edge] = max(float(rows.get(invariant_edge, 0.0) or 0.0), 0.38)

        if label.startswith("vision_mem::") and (
            sample_role == "memory_feature"
            or sa_kind == "visual_focus_feature_unit"
            or sa_kind == "visual_contour_feature_unit"
            or bool(attrs.get("memory_feature_code"))
        ):
            core_label, form_label = self._vision_memory_alias_labels(attrs)
            if core_label:
                rows[core_label] = max(float(rows.get(core_label, 0.0) or 0.0), 0.72)
            if form_label:
                rows[form_label] = max(float(rows.get(form_label, 0.0) or 0.0), 0.42)
        if label.startswith("vision_mem::global_contour::") or str(attrs.get("global_feature_group", "") or "") == "contour_component":
            core_label, form_label = self._vision_memory_alias_labels(attrs)
            if core_label:
                global_core = core_label.replace("vision_contour_core::", "vision_global_contour::", 1)
                rows[global_core] = max(float(rows.get(global_core, 0.0) or 0.0), 0.82)
                rows[core_label] = max(float(rows.get(core_label, 0.0) or 0.0), 0.58)
            if form_label:
                global_form = form_label.replace("vision_contour_form::", "vision_global_contour_form::", 1)
                rows[global_form] = max(float(rows.get(global_form, 0.0) or 0.0), 0.56)
                rows[form_label] = max(float(rows.get(form_label, 0.0) or 0.0), 0.40)
        if label.startswith("vision_dyn::") and (
            sample_role == "dynamic_motion_summary"
            or sa_kind == "visual_dynamic_track_unit"
            or bool(attrs.get("track_id"))
        ):
            core_label, form_label = self._vision_memory_alias_labels(attrs)
            if core_label:
                dyn_core = core_label.replace("vision_core::", "vision_dyn_core::", 1)
                rows[dyn_core] = max(float(rows.get(dyn_core, 0.0) or 0.0), 0.78)
                rows[core_label] = max(float(rows.get(core_label, 0.0) or 0.0), 0.58)
            if form_label:
                dyn_form = form_label.replace("vision_form::", "vision_dyn_form::", 1)
                rows[dyn_form] = max(float(rows.get(dyn_form, 0.0) or 0.0), 0.54)
                rows[form_label] = max(float(rows.get(form_label, 0.0) or 0.0), 0.36)
        if label.startswith("audio::mem::") and (
            sample_role == "memory_feature"
            or sa_kind == "audio_memory_feature_unit"
            or bool(attrs.get("memory_feature_code"))
        ):
            core_label, form_label = self._audio_memory_alias_labels(attrs)
            if core_label:
                rows[core_label] = max(float(rows.get(core_label, 0.0) or 0.0), 0.74)
            if form_label:
                rows[form_label] = max(float(rows.get(form_label, 0.0) or 0.0), 0.44)
        if label.startswith("audio::global::") and (
            sample_role == "global_structure"
            or sa_kind == "audio_global_feature_unit"
            or bool(attrs.get("global_feature_code"))
        ):
            core_label, form_label = self._audio_memory_alias_labels(attrs)
            if core_label:
                global_core = core_label.replace("audio_core::", "audio_global::", 1)
                rows[global_core] = max(float(rows.get(global_core, 0.0) or 0.0), 0.78)
                rows[core_label] = max(float(rows.get(core_label, 0.0) or 0.0), 0.56)
            if form_label:
                global_form = form_label.replace("audio_form::", "audio_global_form::", 1)
                rows[global_form] = max(float(rows.get(global_form, 0.0) or 0.0), 0.52)
                rows[form_label] = max(float(rows.get(form_label, 0.0) or 0.0), 0.34)
        return [(key, _round4(value)) for key, value in rows.items() if key and float(value or 0.0) > 0.0]

    def _vision_memory_alias_labels(self, attrs: dict[str, Any]) -> tuple[str, str]:
        if not isinstance(attrs, dict):
            return "", ""
        hu_signature = str(attrs.get("hu_signature", "") or "")
        radial_signature = str(attrs.get("radial_signature", "") or "")
        if hu_signature or radial_signature:
            proj_h = str(attrs.get("proj_h_bin", "") or "0000")[:4]
            proj_v = str(attrs.get("proj_v_bin", "") or "0000")[:4]
            radial_bin = str(attrs.get("radial_bin", "") or "0000")[:4]
            quadrant_bin = str(attrs.get("quadrant_bin", "") or "0000")[:4]
            polarity = str(attrs.get("foreground_polarity", "bright") or "bright")[:6]
            edge_contact = str(attrs.get("edge_contact_bin", "") or "0000")[:4]
            bbox_signature = str(attrs.get("bbox_signature", "") or "x0_y0_w0_h0")[:20]
            rgb_signature = str(attrs.get("rgb_signature", "") or "000")[:3]
            hole_count = int(max(0, min(3, int(attrs.get("hole_count", 0) or 0))))
            area_bin = int(max(0, min(9, math.floor(float(attrs.get("area_ratio", 0.0) or 0.0) * 20.0))))
            fill_bin = int(max(0, min(9, math.floor(float(attrs.get("bbox_fill", 0.0) or 0.0) * 10.0))))
            solidity_bin = int(max(0, min(9, math.floor(float(attrs.get("solidity", 0.0) or 0.0) * 10.0))))
            round_bin = int(max(0, min(9, math.floor(float(attrs.get("roundness", 0.0) or 0.0) * 10.0))))
            aspect_ratio = float(attrs.get("aspect_ratio", 1.0) or 1.0)
            aspect_bin = int(max(0, min(9, math.floor(min(1.9, aspect_ratio) / 1.9 * 10.0))))
            core_label = (
                f"vision_contour_core::hu{hu_signature[:7]}_rs{radial_signature[:8]}"
                f"_ph{proj_h}_pv{proj_v}_rb{radial_bin}_qb{quadrant_bin}"
                f"_ec{edge_contact}_bb{bbox_signature}_rgb{rgb_signature}"
                f"_a{area_bin}_f{fill_bin}_s{solidity_bin}_r{round_bin}_ar{aspect_bin}_h{hole_count}_p{polarity}"
            )
            form_label = (
                f"vision_contour_form::hu{hu_signature[:7]}_rs{radial_signature[:8]}"
                f"_ph{proj_h}_pv{proj_v}_rb{radial_bin}_qb{quadrant_bin}"
                f"_ec{edge_contact}_rgb{rgb_signature}_h{hole_count}"
            )
            return core_label, form_label

        def bin10(value: float) -> int:
            return int(max(0, min(9, math.floor(float(value) * 10.0))))

        def bin4(value: float) -> int:
            return int(max(0, min(3, math.floor(float(value) * 4.0))))

        stroke_bin = bin10(float(attrs.get("stroke_likeness", 0.0) or 0.0))
        endpoint_bin = bin10(float(attrs.get("endpoint_likeness", 0.0) or 0.0))
        corner_bin = bin10(float(attrs.get("corner_likeness", 0.0) or 0.0))
        opening_bin = bin10(float(attrs.get("opening_likeness", 0.0) or 0.0))
        closure_bin = bin10(float(attrs.get("closure_likeness", 0.0) or 0.0))
        arc_bin = bin10(float(attrs.get("arc_balance", 0.0) or 0.0))
        symmetry_bin = int(max(0, min(2, math.floor(float(attrs.get("local_symmetry", 0.0) or 0.0) * 3.0))))
        discriminability_bin = bin4(float(attrs.get("structure_discriminability", 0.0) or 0.0))
        hole_bin = bin4(float(attrs.get("hole_like", 0.0) or 0.0))
        center_void_bin = bin4(float(attrs.get("center_void", 0.0) or 0.0))
        hsym_bin = bin4(float(attrs.get("horizontal_symmetry", 0.0) or 0.0))
        vsym_bin = bin4(float(attrs.get("vertical_symmetry", 0.0) or 0.0))
        signature = str(attrs.get("local_patch_signature", "") or "")[:9]
        proj_h = str(attrs.get("proj_h_bin", "") or "0000")[:4]
        proj_v = str(attrs.get("proj_v_bin", "") or "0000")[:4]
        orient_bin = str(attrs.get("orient_hist_bin", "") or "0000")[:4]
        radial_bin = str(attrs.get("radial_hist_bin", "") or "0000")[:4]

        shape_scores = {
            "l": float(attrs.get("straight_likeness", 0.0) or 0.0),
            "c": float(attrs.get("curvilinear_likeness", 0.0) or 0.0),
            "a": float(attrs.get("angularity", 0.0) or 0.0),
            "r": float(attrs.get("roundness", 0.0) or 0.0),
        }
        shape_family, shape_strength = max(shape_scores.items(), key=lambda item: (item[1], item[0]))
        shape_strength_bin = bin4(shape_strength)
        opening_dir_x = float(attrs.get("opening_dir_x", 0.0) or 0.0)
        opening_dir_y = float(attrs.get("opening_dir_y", 0.0) or 0.0)
        opening_direction_strength = float(attrs.get("opening_direction_strength", 0.0) or 0.0)
        if float(attrs.get("closure_likeness", 0.0) or 0.0) >= 0.62 and float(attrs.get("roundness", 0.0) or 0.0) >= 0.42:
            opening_tag = "cl"
        elif opening_direction_strength >= 0.14 and float(attrs.get("opening_likeness", 0.0) or 0.0) >= 0.08:
            if abs(opening_dir_x) >= abs(opening_dir_y):
                opening_tag = "or" if opening_dir_x > 0 else "ol"
            else:
                opening_tag = "od" if opening_dir_y > 0 else "ou"
        else:
            opening_tag = "ox"

        core_label = (
            f"vision_core::s{signature}_k{stroke_bin}_n{endpoint_bin}_c{corner_bin}"
            f"_o{opening_bin}_q{closure_bin}_u{arc_bin}_f{shape_family}{shape_strength_bin}"
            f"_g{opening_tag}_y{symmetry_bin}_d{discriminability_bin}"
            f"_ph{proj_h}_pv{proj_v}_oh{orient_bin}_rh{radial_bin}"
            f"_hl{hole_bin}_cv{center_void_bin}_hs{hsym_bin}_vs{vsym_bin}"
        )
        form_label = (
            f"vision_form::f{shape_family}{shape_strength_bin}_g{opening_tag}"
            f"_q{closure_bin}_o{opening_bin}_ph{proj_h}_pv{proj_v}"
            f"_oh{orient_bin}_rh{radial_bin}_hl{hole_bin}"
            f"_cv{center_void_bin}_hs{hsym_bin}_vs{vsym_bin}"
        )
        return core_label, form_label

    def _audio_memory_alias_labels(self, attrs: dict[str, Any]) -> tuple[str, str]:
        if not isinstance(attrs, dict):
            return "", ""

        def bin6(value: float) -> int:
            return int(max(0, min(5, math.floor(_clamp(float(value), 0.0, 1.0) * 6.0))))

        tonal = bin6(float(attrs.get("tonal_clarity", 0.0) or 0.0))
        noise = bin6(float(attrs.get("noisiness", 0.0) or 0.0))
        pitch = bin6(float(attrs.get("pitch_stability", 0.0) or 0.0))
        harmonic = bin6(float(attrs.get("harmonic_ratio", 0.0) or 0.0))
        percussive = bin6(float(attrs.get("percussive_ratio", 0.0) or 0.0))
        voiced = bin6(float(attrs.get("voiced_probability", 0.0) or 0.0))
        contrast = bin6(float(attrs.get("spectral_contrast", 0.0) or 0.0))
        flatness = bin6(float(attrs.get("spectral_flatness", 0.0) or 0.0))
        bandwidth = bin6(float(attrs.get("spectral_bandwidth_ratio", 0.0) or 0.0))
        rolloff = bin6(float(attrs.get("spectral_rolloff_ratio", 0.0) or 0.0))
        centroid = bin6(float(attrs.get("spectral_centroid_ratio", 0.0) or 0.0))
        dominant_band_index = max(0, int(attrs.get("dominant_band_index", 0) or 0))
        profile = str(
            attrs.get("structure_profile")
            or attrs.get("dominant_profile")
            or attrs.get("audio_profile")
            or ""
        ).strip()[:8]
        if not profile:
            profile_scores = {
                "tonal": float(attrs.get("tonal_clarity", 0.0) or 0.0),
                "noisy": float(attrs.get("noisiness", 0.0) or 0.0),
                "percussive": float(attrs.get("percussive_ratio", 0.0) or 0.0),
                "harmonic": float(attrs.get("harmonic_ratio", 0.0) or 0.0),
            }
            profile = max(profile_scores.items(), key=lambda row: (row[1], row[0]))[0]
        hz_bin = int(max(0, min(15, math.floor(max(0.0, float(attrs.get("dominant_hz", 0.0) or 0.0)) / 500.0))))
        core_label = (
            f"audio_core::pf{profile[:2]}_tc{tonal}_nz{noise}_ps{pitch}_hr{harmonic}"
            f"_pr{percussive}_vp{voiced}_ct{contrast}_fl{flatness}"
            f"_bw{bandwidth}_ro{rolloff}_ce{centroid}_db{dominant_band_index}_hz{hz_bin}"
        )
        form_label = (
            f"audio_form::pf{profile[:2]}_tc{tonal}_nz{noise}_ps{pitch}"
            f"_pr{percussive}_ct{contrast}_db{dominant_band_index}"
        )
        return core_label, form_label

    def _compute_successor_bias(
        self,
        memory: dict[str, Any],
        recent_focus_tail: list[str],
        *,
        gain: float = 1.0,
        unit_positions: dict[str, int] | None = None,
        unit_count: int | None = None,
    ) -> float:
        positions = dict(unit_positions or {})
        count = int(unit_count or 0)
        if (not positions or count <= 0) and isinstance(memory, dict):
            units = [str(item or "") for item in memory.get("units", []) if str(item or "")]
            positions = {}
            for index, unit in enumerate(units):
                if unit and unit not in positions:
                    positions[unit] = index
            count = len(units)
        if not positions or count <= 0 or not recent_focus_tail:
            return 0.0
        score = 0.0
        for offset, unit in enumerate(reversed(recent_focus_tail), start=1):
            pos = positions.get(unit)
            if pos is not None:
                relative_pos = pos / max(1.0, float(count - 1))
                forward_bonus = 1.0 if pos < count - 1 else 0.18
                symmetry_penalty = 0.18 * relative_pos
                score += max(0.0, forward_bonus - symmetry_penalty) / float(offset)
        base = score / max(1.0, math.sqrt(len(recent_focus_tail)))
        return min(1.5, max(0.0, base * max(0.0, float(gain))))

    def _normalize_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            display_text = str(item.get("display_text", "") or _display_from_label(label))
            normalized.append(
                {
                    "sa_label": label,
                    "display_text": display_text,
                    "energy": _round4(float(item.get("energy", 0.0) or 0.0)),
                    "position": int(item.get("position", len(normalized)) or len(normalized)),
                    "source_type": str(item.get("source_type", "") or ""),
                    "sa_kind": str(item.get("sa_kind", "") or ""),
                    "channel": str(item.get("channel", "") or self._label_channel(label)),
                    "coords": dict(item.get("coords", {}) or {}),
                    "attributes": dict(item.get("attributes", {}) or {}),
                }
            )
        return normalized

    def _infer_modalities(self, items: list[dict[str, Any]]) -> list[str]:
        seen: list[str] = []
        for item in items:
            channel = str(item.get("channel", "") or self._label_channel(str(item.get("sa_label", "") or "")))
            if channel and channel not in seen:
                seen.append(channel)
        return seen or ["generic"]

    def _infer_spacetime(self, *, tick_index: int, units: list[str], items: list[dict[str, Any]]) -> dict[str, Any]:
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        rel_xs: list[float] = []
        rel_ys: list[float] = []
        rel_rs: list[float] = []
        screen_ws: list[float] = []
        screen_hs: list[float] = []
        positions: list[int] = []
        for item in items:
            coords = dict(item.get("coords", {}) or {})
            if "screen_x" in coords and "screen_y" in coords:
                screen_x = float(coords.get("screen_x", 0.0) or 0.0)
                screen_y = float(coords.get("screen_y", 0.0) or 0.0)
                screen_w = float(coords.get("screen_w", 0.0) or 0.0)
                screen_h = float(coords.get("screen_h", 0.0) or 0.0)
                screen_ws.append(screen_w)
                screen_hs.append(screen_h)
                xs.append(screen_x + screen_w * 0.5)
                ys.append(screen_y + screen_h * 0.5)
                zs.append(float(coords.get("z", 0.0) or 0.0))
            if "cx" in coords and "cy" in coords:
                xs.append(float(coords.get("cx", 0.0) or 0.0))
                ys.append(float(coords.get("cy", 0.0) or 0.0))
                zs.append(float(coords.get("z", 0.0) or 0.0))
            if "x" in coords and "y" in coords:
                xs.append(float(coords.get("x", 0.0) or 0.0))
                ys.append(float(coords.get("y", 0.0) or 0.0))
                zs.append(float(coords.get("z", 0.0) or 0.0))
            if "dx_from_gaze" in coords:
                rel_xs.append(float(coords.get("dx_from_gaze", 0.0) or 0.0))
                rel_ys.append(float(coords.get("dy_from_gaze", 0.0) or 0.0))
                rel_rs.append(float(coords.get("dr_from_gaze", 0.0) or 0.0))
            positions.append(int(item.get("position", len(positions)) or len(positions)))
        has_space = bool(xs and ys)
        has_relative_space = bool(rel_xs and rel_ys)
        local_order_span = (max(positions) - min(positions)) if positions else max(0, len(units) - 1)
        screen_w_mean = (sum(screen_ws) / len(screen_ws)) if screen_ws else 0.0
        screen_h_mean = (sum(screen_hs) / len(screen_hs)) if screen_hs else 0.0
        return {
            "t": int(tick_index),
            "has_space": has_space,
            "x": _round4(sum(xs) / len(xs)) if xs else 0.0,
            "y": _round4(sum(ys) / len(ys)) if ys else 0.0,
            "z": _round4(sum(zs) / len(zs)) if zs else 0.0,
            "has_relative_space": has_relative_space,
            "rel_x": _round4(sum(rel_xs) / len(rel_xs)) if rel_xs else 0.0,
            "rel_y": _round4(sum(rel_ys) / len(rel_ys)) if rel_ys else 0.0,
            "rel_r": _round4(sum(rel_rs) / len(rel_rs)) if rel_rs else 0.0,
            "screen_w": _round4(screen_w_mean),
            "screen_h": _round4(screen_h_mean),
            "space_source_count": len(xs),
            "relative_space_source_count": len(rel_xs),
            "local_order_span": int(local_order_span),
        }

    def _time_intent_match(
        self,
        memory: dict[str, Any],
        *,
        tick_index: int,
        query_spacetime: dict[str, Any] | None,
    ) -> float:
        spacetime = dict(query_spacetime or {})
        confidence = _clamp(float(spacetime.get("time_confidence", 0.0) or 0.0), 0.0, 1.0)
        sigma = max(0.1, float(spacetime.get("time_sigma", 0.0) or 0.0))
        if confidence <= 0.0:
            return 0.0
        target_delta = float(spacetime.get("target_delta_t", 0.0) or 0.0)
        memory_tick = int(memory.get("tick_index", -1) or -1)
        if memory_tick < 0:
            return 0.0
        memory_delta = max(0.0, float(int(tick_index) - memory_tick))
        diff = memory_delta - target_delta
        match = math.exp(-(diff * diff) / max(1e-6, 2.0 * sigma * sigma))
        gain = max(0.0, float(spacetime.get("time_recall_gain", 0.0) or 0.0))
        return _clamp(match * confidence * max(1.0, gain), 0.0, 1.0)

    def _motion_intent_match(self, memory: dict[str, Any], *, query_spacetime: dict[str, Any] | None) -> float:
        spacetime = dict(query_spacetime or {})
        confidence = _clamp(float(spacetime.get("motion_confidence", 0.0) or 0.0), 0.0, 1.0)
        sigma = max(0.05, float(spacetime.get("motion_sigma", 0.0) or 0.0))
        if confidence <= 0.0:
            return 0.0
        items = [dict(item) for item in (memory.get("items", []) or []) if isinstance(item, dict)]
        motion_values: list[float] = []
        for item in items:
            attrs = dict(item.get("attributes", {}) or {})
            motion_speed = float(attrs.get("motion_speed", attrs.get("motion", 0.0)) or 0.0)
            if motion_speed > 0.0:
                motion_values.append(motion_speed)
        if not motion_values:
            return 0.0
        center = float(spacetime.get("motion_center_speed", 0.0) or 0.0)
        memory_center = sum(motion_values) / len(motion_values)
        diff = memory_center - center
        match = math.exp(-(diff * diff) / max(1e-6, 2.0 * sigma * sigma))
        gain = max(0.0, float(spacetime.get("motion_recall_gain", 0.0) or 0.0))
        return _clamp(match * confidence * max(1.0, gain), 0.0, 1.0)

    def _rhythm_intent_match(
        self,
        memory: dict[str, Any],
        *,
        tick_index: int,
        query_spacetime: dict[str, Any] | None,
    ) -> float:
        spacetime = dict(query_spacetime or {})
        confidence = _clamp(float(spacetime.get("rhythm_confidence", 0.0) or 0.0), 0.0, 1.0)
        if confidence <= 0.0:
            return 0.0
        target_period = max(0.0, float(spacetime.get("rhythm_period_ticks", 0.0) or 0.0))
        sigma = max(0.05, float(spacetime.get("rhythm_period_sigma", 0.0) or 0.0))
        family_key = str(spacetime.get("rhythm_family_key", "") or "")
        gain = max(0.0, float(spacetime.get("rhythm_recall_gain", 0.0) or 0.0))
        if target_period <= 0.0:
            return 0.0
        memory_tick = int(memory.get("tick_index", -1) or -1)
        if memory_tick < 0:
            return 0.0
        memory_delta = max(0.0, float(int(tick_index) - memory_tick))
        period_error = abs(memory_delta - target_period)
        interval_match = math.exp(-(period_error * period_error) / max(1e-6, 2.0 * sigma * sigma))

        label_match = 0.0
        if family_key:
            retrieval_weights = dict(memory.get("retrieval_label_weights", {}) or {})
            if family_key in retrieval_weights:
                label_match = 1.0
            else:
                for item in (memory.get("items", []) or []):
                    if not isinstance(item, dict):
                        continue
                    aliases = self._item_retrieval_label_rows(item)
                    if any(str(alias or "") == family_key for alias, _ in aliases):
                        label_match = 1.0
                        break
        phase_match = 1.0
        if "rhythm_time_to_next" in spacetime:
            next_dt = max(0.0, float(spacetime.get("rhythm_time_to_next", 0.0) or 0.0))
            phase_error = abs(memory_delta - next_dt)
            phase_sigma = max(0.05, float(spacetime.get("rhythm_period_sigma", sigma) or sigma))
            phase_match = math.exp(-(phase_error * phase_error) / max(1e-6, 2.0 * phase_sigma * phase_sigma))
        blended = interval_match * (0.7 + 0.3 * phase_match)
        if family_key:
            blended *= 0.55 + 0.45 * label_match
        return _clamp(blended * confidence * max(1.0, gain), 0.0, 1.0)

    def _hearing_intent_match(self, memory: dict[str, Any], *, query_spacetime: dict[str, Any] | None) -> float:
        spacetime = dict(query_spacetime or {})
        confidence = _clamp(float(spacetime.get("hearing_confidence", 0.0) or 0.0), 0.0, 1.0)
        if confidence <= 0.0:
            return 0.0
        items = [dict(item) for item in (memory.get("items", []) or []) if isinstance(item, dict)]
        if not items:
            return 0.0
        feature_keys = (
            ("hearing_timbre_center", "hearing_timbre_sigma", "hearing_timbre_recall_gain", ("tonal_clarity",)),
            ("hearing_noise_center", "hearing_noise_sigma", "hearing_noise_recall_gain", ("noisiness",)),
            ("hearing_pitch_stability_center", "hearing_pitch_stability_sigma", "hearing_pitch_recall_gain", ("pitch_stability",)),
            ("hearing_percussive_center", "hearing_percussive_sigma", "hearing_percussive_recall_gain", ("percussive_ratio",)),
        )
        score_sum = 0.0
        weight_sum = 0.0
        for center_key, sigma_key, gain_key, attr_keys in feature_keys:
            target = float(spacetime.get(center_key, 0.0) or 0.0)
            sigma = max(0.05, float(spacetime.get(sigma_key, 0.0) or 0.0))
            gain = max(0.0, float(spacetime.get(gain_key, 0.0) or 0.0))
            if sigma <= 0.0 or gain <= 0.0:
                continue
            values: list[float] = []
            for item in items:
                attrs = dict(item.get("attributes", {}) or {})
                for attr_key in attr_keys:
                    value = float(attrs.get(attr_key, 0.0) or 0.0)
                    if value > 0.0:
                        values.append(value)
                        break
            if not values:
                continue
            memory_center = sum(values) / len(values)
            diff = memory_center - target
            match = math.exp(-(diff * diff) / max(1e-6, 2.0 * sigma * sigma))
            score_sum += match * max(1.0, gain)
            weight_sum += max(1.0, gain)

        dominant_target_hz = float(spacetime.get("hearing_dominant_hz", 0.0) or 0.0)
        if dominant_target_hz > 0.0:
            hz_values = []
            for item in items:
                attrs = dict(item.get("attributes", {}) or {})
                hz = float(attrs.get("dominant_hz", 0.0) or 0.0)
                if hz > 0.0:
                    hz_values.append(hz)
            if hz_values:
                memory_hz = sum(hz_values) / len(hz_values)
                diff = abs(math.log(max(40.0, memory_hz), 2.0) - math.log(max(40.0, dominant_target_hz), 2.0))
                hz_sigma = 0.55
                hz_match = math.exp(-(diff * diff) / max(1e-6, 2.0 * hz_sigma * hz_sigma))
                pitch_gain = max(1.0, float(spacetime.get("hearing_pitch_recall_gain", 0.0) or 0.0))
                score_sum += hz_match * pitch_gain
                weight_sum += pitch_gain

        if weight_sum <= 0.0:
            return 0.0
        return _clamp((score_sum / weight_sum) * confidence, 0.0, 1.0)

    def _feedback_intent_match(self, memory: dict[str, Any], *, query_spacetime: dict[str, Any] | None) -> float:
        spacetime = dict(query_spacetime or {})
        confidence = _clamp(float(spacetime.get("feedback_confidence", 0.0) or 0.0), 0.0, 1.0)
        sigma = max(0.05, float(spacetime.get("feedback_sigma", 0.0) or 0.0))
        if confidence <= 0.0:
            return 0.0
        target = float(spacetime.get("feedback_valence", 0.0) or 0.0)
        value = 0.0
        for item in (memory.get("items", []) or []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            energy = max(0.0, float(item.get("energy", 0.0) or 0.0))
            if label == "attr::reward_signal":
                value += energy
            elif label == "attr::punishment_signal":
                value -= energy
        if abs(value) <= 1e-6:
            return 0.0
        diff = value - target
        match = math.exp(-(diff * diff) / max(1e-6, 2.0 * sigma * sigma))
        gain = max(0.0, float(spacetime.get("feedback_recall_gain", 0.0) or 0.0))
        return _clamp(match * confidence * max(1.0, gain), 0.0, 1.0)

    def _label_channel(self, label: str) -> str:
        clean = str(label or "")
        if not clean:
            return "generic"
        if clean.startswith(("text::", "phrase::")):
            return "text"
        if clean.startswith(
            (
                "vision::",
                "vision_mem::",
                "vision_core::",
                "vision_form::",
                "vision_global::",
                "vision_dyn::",
                "vision_dyn_core::",
                "vision_dyn_form::",
                "vision_contour_core::",
                "vision_contour_form::",
                "vision_global_contour::",
                "vision_global_contour_form::",
            )
        ):
            return "vision"
        if clean.startswith(
            (
                "audio::",
                "hearing::",
                "audio_core::",
                "audio_form::",
                "audio_global::",
                "audio_global_form::",
            )
        ):
            return "audio"
        if clean.startswith("action::"):
            return "action"
        if clean.startswith("attr::"):
            return "attr"
        if "::" not in clean:
            return "generic"
        return clean.split("::", 1)[0]
