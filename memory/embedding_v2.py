# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import math
from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np


@lru_cache(maxsize=65536)
def _stable_index(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % max(1, int(dim))


@lru_cache(maxsize=65536)
def _stable_sign(token: str) -> float:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=1).digest()
    return 1.0 if (digest[0] % 2 == 0) else -1.0


def _round4(value: float) -> float:
    return round(float(value), 4)


class HashEmbeddingV2:
    def __init__(self, *, dim: int = 256) -> None:
        self.dim = max(32, int(dim))
        self._quantized_attr_cache: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
        self._quantized_attr_cache_order: list[tuple[tuple[str, str], ...]] = []
        self._quantized_attr_cache_limit = 4096

    def build_memory_vector(
        self,
        *,
        units: list[str],
        items: list[dict[str, Any]],
        retrieval_label_weights: dict[str, float] | None = None,
        text: str = "",
        modalities: list[str] | None = None,
        spacetime: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        features = Counter()
        clean_units = [str(unit or "") for unit in units if str(unit or "")]
        for unit in clean_units:
            features[f"unit::{unit}"] += 1.0
        for index in range(0, max(0, len(clean_units) - 1)):
            features[f"bigram::{clean_units[index]}__{clean_units[index + 1]}"] += 0.75
        for item in items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            energy = max(0.0, float(item.get("energy", 0.0) or 0.0))
            channel = str(item.get("channel", "") or self._channel_from_label(label))
            sa_kind = str(item.get("sa_kind", "") or "")
            features[f"label::{label}"] += max(0.12, energy)
            if channel:
                features[f"channel::{channel}"] += max(0.08, energy * 0.35)
            if sa_kind:
                features[f"kind::{sa_kind}"] += max(0.05, energy * 0.25)
            for attr_key, attr_value in self._quantized_attrs(item).items():
                features[f"attr::{attr_key}::{attr_value}"] += max(0.05, energy * 0.22)
        alias_pairs = sorted(
            (
                (str(label or ""), max(0.0, float(weight or 0.0)))
                for label, weight in dict(retrieval_label_weights or {}).items()
                if str(label or "") and float(weight or 0.0) > 0.0
            ),
            key=lambda row: (-row[1], row[0]),
        )
        for label, weight in alias_pairs[:32]:
            scaled = max(0.04, min(2.4, weight * 0.78))
            features[f"label::{label}"] += scaled
            channel = self._channel_from_label(label)
            if channel:
                features[f"channel::{channel}"] += max(0.03, scaled * 0.24)
        if text:
            features[f"textlen::{min(8, max(1, len(text) // 8 + 1))}"] += 0.2
        for modality in (modalities or []):
            features[f"modality::{modality}"] += 0.12
        if isinstance(spacetime, dict):
            for key, bucket in self._spacetime_tokens(spacetime).items():
                features[f"sp::{key}::{bucket}"] += 0.08
        return self._vectorize(features)

    def build_query_vector(
        self,
        *,
        query_labels: list[str],
        query_weights: dict[str, float],
        query_items: list[dict[str, Any]] | None = None,
        query_units: list[str] | None = None,
        recent_focus_units: list[str] | None = None,
        query_spacetime: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        features = Counter()
        query_units = [str(unit or "") for unit in (query_units or []) if str(unit or "")]
        recent_focus_units = [str(unit or "") for unit in (recent_focus_units or []) if str(unit or "")]
        for label in query_labels:
            clean = str(label or "")
            if not clean:
                continue
            weight = max(0.05, float(query_weights.get(clean, 0.0) or 0.0))
            features[f"label::{clean}"] += weight
            channel = self._channel_from_label(clean)
            if channel:
                features[f"channel::{channel}"] += weight * 0.28
        for item in (query_items or []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            weight = max(
                0.04,
                float(query_weights.get(label, 0.0) or 0.0),
                float(item.get("energy", 0.0) or 0.0) * 0.7,
            )
            channel = str(item.get("channel", "") or self._channel_from_label(label))
            sa_kind = str(item.get("sa_kind", "") or "")
            if channel:
                features[f"channel::{channel}"] += max(0.04, weight * 0.24)
            if sa_kind:
                features[f"kind::{sa_kind}"] += max(0.03, weight * 0.18)
            for attr_key, attr_value in self._quantized_attrs(item).items():
                features[f"attr::{attr_key}::{attr_value}"] += max(0.03, weight * 0.16)
        for unit in query_units:
            features[f"unit::{unit}"] += 1.0
        for index in range(0, max(0, len(query_units) - 1)):
            features[f"bigram::{query_units[index]}__{query_units[index + 1]}"] += 0.85
        focus_tail = recent_focus_units[-4:]
        for offset, unit in enumerate(reversed(focus_tail), start=1):
            features[f"focus::{unit}"] += 0.4 / float(offset)
            if offset == 1:
                features[f"unit::{unit}"] += 0.3
        if isinstance(query_spacetime, dict):
            for key, bucket in self._spacetime_tokens(query_spacetime).items():
                features[f"sp::{key}::{bucket}"] += 0.12
        return self._vectorize(features)

    def cosine(self, left: np.ndarray, right: np.ndarray) -> float:
        if left.size == 0 or right.size == 0:
            return 0.0
        score = float(np.dot(left, right))
        return max(-1.0, min(1.0, score))

    def preview(self, tokens: list[str], *, limit: int = 8) -> list[str]:
        return [str(token or "") for token in tokens[: max(1, int(limit))] if str(token or "")]

    def _vectorize(self, features: Counter[str]) -> tuple[np.ndarray, list[str]]:
        vector = np.zeros((self.dim,), dtype=np.float32)
        ordered = sorted(features.items(), key=lambda item: (-float(item[1]), item[0]))
        for token, weight in ordered:
            index = _stable_index(token, self.dim)
            vector[index] += np.float32(weight * _stable_sign(token))
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector.astype(np.float32), [token for token, _ in ordered[:16]]

    def _channel_from_label(self, label: str) -> str:
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

    def _quantized_attrs(self, item: dict[str, Any]) -> dict[str, str]:
        attrs = item.get("attributes", {}) or {}
        if not isinstance(attrs, dict):
            return {}
        cache_key_parts: list[tuple[str, str]] = []
        for key, value in attrs.items():
            if isinstance(value, (int, float)):
                cache_key_parts.append((str(key), f"n:{_round4(float(value))}"))
            elif isinstance(value, str) and value:
                cache_key_parts.append((str(key), f"s:{value[:40]}"))
        cache_key = tuple(sorted(cache_key_parts))
        cached = self._quantized_attr_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        result: dict[str, str] = {}
        for key, value in attrs.items():
            if isinstance(value, (int, float)):
                bucket = int(max(0, min(9, math.floor(float(value) * 10.0))))
                result[str(key)] = str(bucket)
            elif isinstance(value, str) and value:
                keep_key = str(key)
                if keep_key in {
                    "proj_h_bin",
                    "proj_v_bin",
                    "radial_bin",
                    "quadrant_bin",
                    "orient_hist_bin",
                    "radial_hist_bin",
                    "hu_signature",
                    "radial_signature",
                    "foreground_polarity",
                    "memory_feature_code",
                    "global_feature_code",
                }:
                    result[keep_key] = value[:40]
                else:
                    result[keep_key] = value[:24]
        if cache_key:
            self._quantized_attr_cache[cache_key] = dict(result)
            self._quantized_attr_cache_order.append(cache_key)
            if len(self._quantized_attr_cache_order) > self._quantized_attr_cache_limit:
                stale = self._quantized_attr_cache_order.pop(0)
                self._quantized_attr_cache.pop(stale, None)
        return result

    def _spacetime_tokens(self, spacetime: dict[str, Any]) -> dict[str, str]:
        tokens: dict[str, str] = {}
        if bool(spacetime.get("has_space", False)):
            for axis in ("x", "y", "z"):
                value = float(spacetime.get(axis, 0.0) or 0.0)
                tokens[axis] = str(int(max(0, min(9, math.floor(value * 10.0)))))
            for axis in ("screen_w", "screen_h"):
                value = float(spacetime.get(axis, 0.0) or 0.0)
                tokens[axis] = str(int(max(0, min(9, math.floor(value * 10.0)))))
        if bool(spacetime.get("has_relative_space", False)):
            rel_x = float(spacetime.get("rel_x", 0.0) or 0.0)
            rel_y = float(spacetime.get("rel_y", 0.0) or 0.0)
            rel_r = float(spacetime.get("rel_r", 0.0) or 0.0)
            tokens["rel_x"] = str(int(max(0, min(9, math.floor((rel_x + 1.0) * 5.0)))))
            tokens["rel_y"] = str(int(max(0, min(9, math.floor((rel_y + 1.0) * 5.0)))))
            tokens["rel_r"] = str(int(max(0, min(9, math.floor(rel_r * 10.0)))))
        if "target_delta_t" in spacetime:
            delta_t = max(0.0, float(spacetime.get("target_delta_t", 0.0) or 0.0))
            tokens["target_dt"] = str(int(max(0, min(24, math.floor(delta_t)))))
        if "motion_center_speed" in spacetime:
            motion_speed = max(0.0, float(spacetime.get("motion_center_speed", 0.0) or 0.0))
            tokens["motion_speed"] = str(int(max(0, min(9, math.floor(motion_speed * 10.0)))))
        if "feedback_valence" in spacetime:
            feedback_valence = float(spacetime.get("feedback_valence", 0.0) or 0.0)
            tokens["feedback_valence"] = str(int(max(0, min(18, math.floor((feedback_valence + 1.0) * 9.0)))))
        if "rhythm_period_ticks" in spacetime:
            period = max(0.0, float(spacetime.get("rhythm_period_ticks", 0.0) or 0.0))
            tokens["rhythm_period"] = str(int(max(0, min(24, math.floor(period)))))
        if "rhythm_time_to_next" in spacetime:
            next_dt = max(0.0, float(spacetime.get("rhythm_time_to_next", 0.0) or 0.0))
            tokens["rhythm_next"] = str(int(max(0, min(24, math.floor(next_dt)))))
        tokens["order_span"] = str(int(max(0, min(12, int(spacetime.get("local_order_span", 0) or 0)))))
        return tokens


def vector_preview(values: np.ndarray, *, limit: int = 8) -> list[float]:
    if values.size == 0:
        return []
    return [_round4(item) for item in values[: max(1, int(limit))].tolist()]
